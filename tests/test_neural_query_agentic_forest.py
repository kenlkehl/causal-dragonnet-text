import numpy as np

from oci.inference.neural_query_agentic_forest import (
    NeuralQueryAgenticForestConfig,
    apply_review_candidates_to_registry,
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
        "top_contrastive_ngrams": [{"term": "baseline cbc"}],
        "top_chunks": [
            {
                "evidence_id": "effect_query_001__row_00001__chunk_000",
                "text": "Baseline ANC 6.0, ALC 1.0, NLR 6.0, Hgb 11.2 g/dL.",
            }
        ],
    }
    context = build_query_feature_context(evidence, config=config)
    prompt = render_query_feature_prompt(context)
    assert "multiple variables" in prompt
    assert "statistical gate" in prompt
    assert "Prior treatments" in prompt
    assert "responses, toxicities, and outcomes are valid baseline history" in prompt
    response = {
        "general_topic": "baseline CBC",
        "query_quality": "coherent",
        "proposals": [
            {
                "action": "add",
                "name": "baseline_nlr",
                "type": "continuous",
                "categories": None,
                "description": "Baseline neutrophil-to-lymphocyte ratio.",
                "clinical_domain": "hematology",
                "parent_object": "baseline CBC",
                "supporting_evidence_ids": [
                    "effect_query_001__row_00001__chunk_000"
                ],
                "supporting_phrases": ["NLR 6.0"],
                "rationale": "Explicitly present.",
            },
            {
                "action": "add",
                "name": "baseline_hemoglobin",
                "type": "continuous",
                "categories": None,
                "description": "Baseline hemoglobin in g/dL.",
                "clinical_domain": "hematology",
                "parent_object": "baseline CBC",
                "supporting_evidence_ids": [
                    "effect_query_001__row_00001__chunk_000"
                ],
                "supporting_phrases": ["Hgb 11.2 g/dL"],
                "rationale": "Explicitly present.",
            },
        ],
    }
    assert query_feature_response_issues(response, context) == []
    candidates = query_candidates_from_response(response, context)
    assert [candidate["name"] for candidate in candidates] == [
        "baseline_nlr",
        "baseline_hemoglobin",
    ]
    assert all(candidate["roles"] == ["effect_modifier"] for candidate in candidates)


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
