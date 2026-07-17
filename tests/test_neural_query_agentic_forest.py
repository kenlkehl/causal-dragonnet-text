import inspect
import json
import re

import numpy as np

import oci.inference.neural_query_agentic_forest as neural_query_module
from oci.inference.neural_query_agentic_forest import (
    NeuralQueryAgenticForestConfig,
    apply_review_candidates_to_registry,
    build_query_evidence,
    build_query_feature_context,
    build_query_rag_documents,
    build_query_registry_context,
    extraction_request_groups,
    query_candidates_from_response,
    query_feature_response_issues,
    query_registry_response_issues,
    query_review_response_issues,
    registry_from_response,
    render_query_feature_prompt,
)


def test_query_bank_counts_are_independent_and_default_to_five():
    default = NeuralQueryAgenticForestConfig()
    assert [default.query_count(bank) for bank in ("treatment", "outcome", "effect")] == [
        5,
        5,
        5,
    ]
    custom = NeuralQueryAgenticForestConfig(
        treatment_query_count=2,
        outcome_query_count=3,
        effect_query_count=4,
        max_features_per_query=2,
        max_raw_feature_candidates=18,
        max_canonical_features=8,
    )
    custom.validate()
    assert custom.query_count("treatment") == 2
    assert custom.query_count("outcome") == 3
    assert custom.query_count("effect") == 4


def test_one_query_prompt_allows_multiple_traceable_features_without_gate():
    config = NeuralQueryAgenticForestConfig()
    evidence = {
        "query_id": "effect_query_001",
        "bank": "effect",
        "statistical_gate_applied": False,
        "member_count": 5,
        "member_subfolds": [1, 2, 3, 4, 5],
        "fit_standardized_score": 2.1,
        "top_contrastive_ngrams": [
            {"term": "marker alpha"},
            {"term": "marker beta"},
        ],
        "top_chunks": [
            {
                "evidence_id": "effect_query_001__row_00001__chunk_000",
                "text": "Baseline panel: marker alpha 6.0, marker beta 11.2 units.",
            }
        ],
    }
    context = build_query_feature_context(evidence, config=config)
    prompt = render_query_feature_prompt(context)
    assert "multiple variables" in prompt
    assert "statistical gate" in prompt
    assert "Prior treatments" in prompt
    assert "responses, toxicities, and outcomes are valid baseline history" in prompt
    assert "evidence_field_cues" in prompt
    assert {row["cue"].lower() for row in context["evidence_field_cues"]} >= {
        "marker alpha",
        "marker beta",
    }
    response = {
        "general_topic": "baseline panel",
        "query_quality": "coherent",
        "proposals": [
            {
                "action": "add",
                "name": "baseline_marker_alpha",
                "type": "continuous",
                "categories": None,
                "description": "Baseline marker alpha in the documented units.",
                "clinical_domain": "measurement panel",
                "parent_object": "baseline panel",
                "supporting_evidence_ids": ["effect_query_001__row_00001__chunk_000"],
                "supporting_phrases": ["marker alpha 6.0"],
                "rationale": "Explicitly present.",
            },
            {
                "action": "add",
                "name": "baseline_marker_beta",
                "type": "continuous",
                "categories": None,
                "description": "Baseline marker beta in the documented units.",
                "clinical_domain": "measurement panel",
                "parent_object": "baseline panel",
                "supporting_evidence_ids": ["effect_query_001__row_00001__chunk_000"],
                "supporting_phrases": ["marker beta 11.2 units"],
                "rationale": "Explicitly present.",
            },
        ],
    }
    assert query_feature_response_issues(response, context) == []
    candidates = query_candidates_from_response(response, context)
    assert [candidate["name"] for candidate in candidates] == [
        "baseline_marker_alpha",
        "baseline_marker_beta",
    ]
    assert all(candidate["roles"] == ["effect_modifier"] for candidate in candidates)


def test_feature_prompt_has_no_fixed_hidden_variable_seed_vocabulary():
    source = inspect.getsource(neural_query_module)
    prompt = render_query_feature_prompt(
        build_query_feature_context(
            {
                "query_id": "effect_query_001",
                "bank": "effect",
                "top_contrastive_ngrams": [{"term": "marker zeta"}],
                "top_chunks": [
                    {
                        "evidence_id": "training_evidence_001",
                        "text": "Marker zeta: 4.2 units.",
                    }
                ],
            },
            config=NeuralQueryAgenticForestConfig(),
        )
    )
    for fixed_term in ("QX7", "RZ8", "VT9", "widgetonium", "Wdg", "QAZ"):
        pattern = rf"\b{re.escape(fixed_term)}\b"
        assert re.search(pattern, source, flags=re.IGNORECASE) is None
        assert re.search(pattern, prompt, flags=re.IGNORECASE) is None
    assert "marker zeta" in prompt.lower()
    assert 'uses_fixed_clinical_vocabulary": false' in prompt


def test_query_evidence_and_prompt_context_isolate_heldout_and_oracle_payloads():
    class GuardedChunkTexts:
        def __len__(self):
            return 3

        def __getitem__(self, row_id):
            if int(row_id) == 1:
                raise AssertionError("outer-held-out text was accessed during discovery")
            return [
                ["training marker alpha 2.0"],
                [],
                ["training marker beta 3.0"],
            ][int(row_id)]

    config = NeuralQueryAgenticForestConfig(
        evidence_top_patients=1,
        evidence_background_patients=1,
    )
    evidence = build_query_evidence(
        bank="effect",
        queries=np.array([[1.0, 0.0]], dtype=np.float32),
        query_records=[
            {
                "query_id": "effect_query_001",
                "member_count": 2,
                "true_ite_prob": "ORACLE_RECORD_SENTINEL",
            }
        ],
        row_ids=[0, 2],
        chunk_matrices=[
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[0.8, 0.2]], dtype=np.float32),
        ],
        all_chunk_texts=GuardedChunkTexts(),
        config=config,
        device="cpu",
        seed=7,
    )[0]
    assert "ORACLE_RECORD_SENTINEL" not in json.dumps(evidence)

    evidence["outer_heldout_payload"] = "HELDOUT_METADATA_SENTINEL"
    evidence["top_chunks"][0]["true_ite_prob"] = "ORACLE_CHUNK_SENTINEL"
    evidence["top_contrastive_ngrams"][0]["oracle_score"] = "ORACLE_NGRAM_SENTINEL"
    context = build_query_feature_context(evidence, config=config)
    serialized = json.dumps(context)
    for sentinel in (
        "HELDOUT_METADATA_SENTINEL",
        "ORACLE_CHUNK_SENTINEL",
        "ORACLE_NGRAM_SENTINEL",
    ):
        assert sentinel not in serialized
    assert context["field_cue_policy"]["forwards_unlisted_evidence_metadata"] is False


def test_registry_role_union_and_extraction_group_cap():
    config = NeuralQueryAgenticForestConfig(
        max_canonical_features=4,
        max_raw_feature_candidates=6,
    )
    candidates = [
        {
            "candidate_id": "treatment_query_001__candidate_01",
            "name": "histology",
            "type": "categorical",
            "categories": ["adenocarcinoma", "squamous", "other"],
            "roles": ["confounder"],
            "description": "Baseline NSCLC histology.",
            "clinical_domain": "pathology",
            "parent_object": "tumor pathology",
            "supporting_phrases": ["squamous carcinoma"],
            "provenance": [{"query_id": "treatment_query_001"}],
        },
        {
            "candidate_id": "effect_query_002__candidate_01",
            "name": "tumor_histology",
            "type": "categorical",
            "categories": ["adenocarcinoma", "squamous", "other"],
            "roles": ["effect_modifier"],
            "description": "Pretreatment tumor histology.",
            "clinical_domain": "pathology",
            "parent_object": "tumor pathology",
            "supporting_phrases": ["adenocarcinoma"],
            "provenance": [{"query_id": "effect_query_002"}],
        },
    ]
    context = build_query_registry_context(candidates, config=config)
    response = {
        "features": [
            {
                "name": "baseline_histology",
                "type": "categorical",
                "categories": ["adenocarcinoma", "squamous", "other"],
                "description": "Pretreatment NSCLC histology.",
                "clinical_domain": "pathology",
                "parent_object": "tumor pathology",
                "source_candidate_ids": [
                    "treatment_query_001__candidate_01",
                    "effect_query_002__candidate_01",
                ],
                "reason": "Exact aliases.",
            }
        ],
        "dropped_candidates": [],
    }
    assert query_registry_response_issues(response, context) == []
    registry, dropped = registry_from_response(response, context)
    assert not dropped
    assert registry[0]["roles"] == ["confounder", "effect_modifier"]
    assert extraction_request_groups(registry * 10, maximum=10) == [
        ["baseline_histology"] * 10
    ]


def test_query_rag_document_reads_all_selected_chunks_and_deduplicates():
    config = NeuralQueryAgenticForestConfig(
        rag_chunks_per_query=1,
        rag_max_chunks_per_patient=2,
    )
    chunks = [
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    ]
    document = build_query_rag_documents(
        row_ids=[0],
        chunk_matrices=chunks,
        all_chunk_texts=[["pathology evidence", "laboratory evidence"]],
        queries=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        query_ids=["treatment_query_001", "effect_query_001"],
        query_banks=["treatment", "effect"],
        config=config,
        device="cpu",
    )[0]
    assert document.count("<retrieved_excerpt") == 2
    assert "pathology evidence" in document
    assert "laboratory evidence" in document


def test_review_is_additive_bounded_and_preserves_role_union():
    registry = [
        {
            "name": "baseline_histology",
            "type": "categorical",
            "categories": ["adenocarcinoma", "squamous", "other"],
            "roles": ["confounder"],
            "description": "original contract",
            "provenance": [{"query_id": "treatment_query_001"}],
        }
    ]
    reviewed, decisions = apply_review_candidates_to_registry(
        registry,
        [
            {
                "candidate_id": "review_01__candidate_01",
                "action": "refine",
                "name": "baseline_histology",
                "type": "categorical",
                "categories": ["adenocarcinoma", "squamous", "other"],
                "roles": ["effect_modifier"],
                "description": "Histology before the current decision.",
                "clinical_domain": "pathology",
                "parent_object": "tumor pathology",
                "provenance": [{"query_id": "effect_query_001"}],
            },
            {
                "candidate_id": "review_01__candidate_02",
                "action": "add",
                "name": "baseline_hemoglobin",
                "type": "continuous",
                "categories": None,
                "roles": ["effect_modifier"],
                "description": "Latest hemoglobin in g/dL before the current decision.",
                "clinical_domain": "hematology",
                "parent_object": "baseline CBC",
                "provenance": [{"query_id": "effect_query_002"}],
            },
        ],
        maximum=2,
    )
    assert len(reviewed) == 2
    assert reviewed[0]["roles"] == ["confounder", "effect_modifier"]
    assert registry[0]["description"] == "original contract"
    assert all(row["accepted"] for row in decisions)


def test_review_roles_are_validated_from_query_ids_not_agent_claims():
    context = {
        "max_additions": 1,
        "current_registry": [],
        "query_evidence": [{"query_id": "effect_query_001", "bank": "effect"}],
    }
    response = {
        "proposals": [
            {
                "action": "add",
                "name": "brain_metastases",
                "type": "categorical",
                "categories": ["present", "absent", "not_documented"],
                "roles": ["confounder"],
                "description": "Status before the current treatment decision.",
                "source_query_ids": ["effect_query_001"],
                "rationale": "Retrieved evidence is explicit.",
            }
        ]
    }
    assert query_review_response_issues(response, context) == []
