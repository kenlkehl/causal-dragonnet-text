from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from oci.config import AgenticFeatureSearchConfig
from oci.inference.agentic_explicit_feature_forest import (
    OpenAICompatibleFeatureSearchAgent,
)
from oci.inference.all_evidence_fusion import (
    ALL_SOURCE_FAMILIES,
    BOW_NUISANCE,
    BOW_R_LOSS,
    CandidateContract,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    EVIDENCE_CONTRACT_GROUNDING_VERSION,
    FUSION_PROMPT_VERSION,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
    HTR_NEURAL,
    LEGACY_ALL_SOURCE,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    NEURAL_QUERY_SOURCE,
    QUERY_MOMENTS,
    SPARSE_QUERY_MOMENTS,
    SPARSE_QUERY_SOURCE,
    SOURCE_TEXT_TEMPORAL_POLICY,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPICS,
    TFIDF_TOPIC_SOURCE,
    _normalize_agent_response_citation_families,
    all_evidence_fusion_response_issues,
    build_all_evidence_fusion_repair_prompt,
    ground_evidence_to_extraction_contract,
    prepare_all_evidence_fusion,
    render_all_evidence_fusion_context_prompt,
    validate_all_evidence_fusion_response,
)


def _provenance(*, outer_fold: int = 1) -> FoldEvidenceProvenance:
    return FoldEvidenceProvenance(
        outer_fold=outer_fold,
        train_row_ids=(10, 11, 12, 13),
        heldout_row_ids=(20, 21),
        artifact_id=f"artifact-fold-{outer_fold}",
    )


def _legacy_payload() -> dict:
    return {
        "outer_fold": 1,
        "context": {
            "evidence_digest": {
                "confounders": {
                    "bow_blurbs": [
                        {
                            "source": "sparse_nuisance.confounder_overlap",
                            "meaning": "terms shared by assignment and baseline-load models",
                            "rows": [{"feature": "inlet valve position score", "score": 2.1}],
                        }
                    ],
                    "embedding_chunks": [
                        {
                            "name": "whole_cohort_treatment_residual",
                            "contrast_family": "whole_cohort",
                            "positive_aligned_chunks": [
                                {"text": "inlet valve position documented"}
                            ],
                        }
                    ],
                    "htr_blurbs": [
                        {
                            "stage": "nuisance",
                            "rows": [
                                {
                                    "_oci_row_id": 10,
                                    "attended_token_summary": "prerun vibration amplitude",
                                }
                            ],
                        }
                    ],
                },
                "effect_modifiers": {
                    "bow_blurbs": [
                        {
                            "source": "ensemble_r.pseudo_target_positive",
                            "rows": [{"feature": "alloy phase status", "score": 3.2}],
                        },
                        {
                            "source": "matched_pair_uplift.uplift_pair_features",
                            "rows": [{"feature": "prior load magnitude", "score": 1.7}],
                        },
                    ],
                    "embedding_chunks": [
                        {
                            "name": "effect_residual_cluster_component_1",
                            "contrast_family": "cluster_component",
                            "cluster_component_index": 1,
                            "negative_aligned_chunks": [
                                {"text": "prerun sensor pattern"}
                            ],
                        }
                    ],
                    "htr_blurbs": [
                        {
                            "stage": "effect",
                            "rows": [
                                {
                                    "row_id": 11,
                                    "top_token_spans": [{"text": "alloy composition result"}],
                                }
                            ],
                        },
                        {
                            "stage": "pair_uplift",
                            "rows": [
                                {
                                    "candidate_row_id": 12,
                                    "control_row_id": 13,
                                    "evidence_snippet": "prerun material pattern",
                                }
                            ],
                        },
                    ],
                },
            }
        },
    }


def _tfidf_payload() -> dict:
    topic = lambda topic_id, phrase: {  # noqa: E731
        "topic_id": topic_id,
        "terms": [{"term": phrase, "loading": 0.8, "signed_score": 2.0}],
    }
    return {
        "outer_fold": 1,
        "discovery": {
            "topic_banks": {
                "treatment": {"topics": [topic("treatment_1", "prerun calibration pattern")]},
                "outcome": {"topics": [topic("outcome_1", "prerun load pattern")]},
                "effect": {"topics": [topic("effect_1", "prerun alloy phase phrase")]},
            },
            "topic_score_tests": {
                "effect_orphan_ngram_branch": {
                    "selected_cluster_ids": ["orphan_1"],
                    "selected_clusters": [
                        {
                            "cluster_id": "orphan_1",
                            "terms": [{"term": "unmodeled baseline phrase", "fit_rank": 23}],
                        }
                    ],
                }
            },
        },
    }


def _query_payload(*, query_id: str = "effect_query_01") -> dict:
    return {
        "outer_fold": 1,
        "query_evidence": [
            {
                "query_id": query_id,
                "bank": "effect",
                "fit_standardized_score": 2.4,
                "member_count": 3,
                "top_chunks": [
                    {
                        "evidence_id": "effect_query_01__row_00012__chunk_001",
                        "_oci_row_id": 12,
                        "chunk_index": 1,
                        "text": "baseline assessment category present",
                    }
                ],
                "top_contrastive_ngrams": [{"term": "assessment category", "tfidf_contrast": 0.4}],
            }
        ],
    }


def _all_inputs() -> list[FoldEvidenceInput]:
    provenance = _provenance()
    return [
        FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(), provenance),
        FoldEvidenceInput(TFIDF_TOPIC_SOURCE, _tfidf_payload(), provenance),
        FoldEvidenceInput(NEURAL_QUERY_SOURCE, _query_payload(), provenance),
        FoldEvidenceInput(
            SPARSE_QUERY_SOURCE,
            _query_payload(query_id="sparse_effect_query_01"),
            provenance,
        ),
    ]


def _candidate_spec(name: str = "inlet_valve_position") -> dict:
    return {
        "name": name,
        "type": "categorical",
        "categories": ["absent", "present", "not_documented"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Inlet valve position explicitly documented before treatment selection.",
    }


def _concept_evidence(text: str | list[str]) -> dict:
    content = {"concept": text} if isinstance(text, str) else {"terms": text}
    return {"evidence_id": "evidence_0001", "content": content}


def _continuous_contract(name: str, description: str) -> dict:
    return {
        "name": name,
        "type": "continuous",
        "roles": ["confounder"],
        "description": description,
    }


def test_compacts_every_evidence_family_and_removes_row_identifiers():
    candidate = CandidateContract(
        _candidate_spec(),
        source_families=(BOW_NUISANCE, QUERY_MOMENTS),
    )
    request = prepare_all_evidence_fusion(
        _all_inputs(),
        candidates=[candidate],
        max_candidates=1,
    )

    coverage = request.source_family_coverage
    assert set(coverage["present_source_families"]) == set(ALL_SOURCE_FAMILIES)
    assert coverage["evidence_block_count_by_source_family"] == {
        BOW_NUISANCE: 1,
        BOW_R_LOSS: 1,
        MATCHED_PAIR_UPLIFT: 2,
        HTR_NEURAL: 3,
        EMBEDDING_WHOLE_COHORT: 1,
        EMBEDDING_CLUSTERED: 1,
        TFIDF_TOPICS: 3,
        TFIDF_ORPHAN_NGRAMS: 1,
        NEURAL_QUERY_MOMENTS: 1,
        SPARSE_QUERY_MOMENTS: 1,
    }
    prompt = request.render_prompt()
    assert request.mode == "select"
    assert "candidate_0001" in prompt
    assert "row_00012" not in prompt
    assert "_oci_row_id" not in prompt
    assert "candidate_row_id" not in prompt
    assert "control_row_id" not in prompt
    assert "oracle" not in prompt.lower()
    assert "ground truth" not in prompt.lower()


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda payload: payload["context"]["evidence_digest"]["confounders"].update(
                {"true_ite": [0.1, 0.2]}
            ),
            "forbidden oracle/true field",
        ),
        (
            lambda payload: payload["context"]["evidence_digest"]["confounders"]["bow_blurbs"][0][
                "rows"
            ][0].update({"feature": "oracle-selected feature"}),
            "forbidden oracle/true string",
        ),
        (
            lambda payload: payload["context"]["evidence_digest"]["confounders"]["htr_blurbs"][0][
                "rows"
            ][0].update({"_oci_row_id": 20}),
            "heldout or unknown row",
        ),
    ],
)
def test_rejects_recursive_leakage_before_allowlisting(mutate, match):
    payload = _legacy_payload()
    mutate(payload)
    source = FoldEvidenceInput(LEGACY_ALL_SOURCE, payload, _provenance())
    with pytest.raises(ValueError, match=match):
        prepare_all_evidence_fusion([source])


def test_requires_exactly_aligned_fold_provenance_across_sources():
    first = FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(), _provenance())
    other_provenance = FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=(10, 11, 12),
        heldout_row_ids=(13, 20, 21),
        artifact_id="different-split",
    )
    second = FoldEvidenceInput(TFIDF_TOPIC_SOURCE, _tfidf_payload(), other_provenance)
    with pytest.raises(ValueError, match="identical fold train/heldout provenance"):
        prepare_all_evidence_fusion([first, second])

    wrong_fold = _legacy_payload()
    wrong_fold["outer_fold"] = 2
    with pytest.raises(ValueError, match="does not match provenance outer_fold"):
        prepare_all_evidence_fusion(
            [FoldEvidenceInput(LEGACY_ALL_SOURCE, wrong_fold, _provenance())]
        )


def test_selection_cap_unknown_ids_and_contract_immutability():
    mutable_spec = _candidate_spec()
    contract = CandidateContract(
        mutable_spec,
        source_families=(BOW_NUISANCE,),
    )
    request = prepare_all_evidence_fusion(
        [FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(), _provenance())],
        candidates=[contract, CandidateContract(_candidate_spec("baseline_measure"))],
        max_candidates=1,
    )
    mutable_spec["name"] = "mutated_after_freeze"
    mutable_spec["categories"].append("injected")

    evidence = next(
        block for block in request.evidence_blocks if BOW_NUISANCE in block.source_families
    )
    response = {
        "selected_candidate_ids": ["candidate_0001"],
        "selection_notes": [
            {
                "candidate_id": "candidate_0001",
                "supporting_evidence_ids": [evidence.evidence_id],
                "supporting_source_families": [BOW_NUISANCE],
                "reason": "supported by a fold-local sparse-text pattern",
            }
        ],
    }
    result = validate_all_evidence_fusion_response(request, response)
    assert result.selected_specs == [_candidate_spec()]
    result.selected_specs[0]["name"] = "also_mutated"
    assert result.selected_specs == [_candidate_spec()]

    with pytest.raises(ValueError, match="exceeds max_candidates"):
        validate_all_evidence_fusion_response(
            request,
            {"selected_candidate_ids": ["candidate_0001", "candidate_0002"]},
        )
    with pytest.raises(ValueError, match="unknown candidate IDs"):
        validate_all_evidence_fusion_response(
            request,
            {"selected_candidate_ids": ["baseline_status"]},
        )
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_all_evidence_fusion_response(
            request,
            {
                "selected_candidate_ids": ["candidate_0001"],
                "rewritten_spec": _candidate_spec("agent_rewrite"),
            },
        )


def test_selection_rejects_real_but_conceptually_unrelated_evidence():
    request = _selection_request()
    unrelated = next(
        block for block in request.evidence_blocks if BOW_R_LOSS in block.source_families
    )
    response = {
        "selected_candidate_ids": ["candidate_0001"],
        "selection_notes": [
            {
                "candidate_id": "candidate_0001",
                "supporting_evidence_ids": [unrelated.evidence_id],
                "supporting_source_families": [BOW_R_LOSS],
                "reason": "This intentionally cites an unrelated real evidence block.",
            }
        ],
    }

    with pytest.raises(ValueError, match="evidence unrelated to the selected contract"):
        validate_all_evidence_fusion_response(request, response)


@pytest.mark.parametrize(
    "evidence_text, contract",
    [
        (
            "baseline copper density",
            _continuous_contract(
                "quartz_width",
                "Quartz width; copper density may also be recorded.",
            ),
        ),
        (
            "assembly hue",
            _continuous_contract(
                "assembly_temperature",
                "Assembly temperature; hue may also be recorded.",
            ),
        ),
        (
            "alloy pressure",
            _continuous_contract(
                "ribbon_width",
                "Ribbon width measured on the alloy specimen under pressure.",
            ),
        ),
        (
            "gear 4 documented",
            {
                "name": "gear_ratio",
                "type": "categorical",
                "categories": ["4", "5"],
                "roles": ["effect_modifier"],
                "description": "Documented gear ratio.",
            },
        ),
        (
            "normal valve",
            {
                "name": "valve_pressure_category",
                "type": "categorical",
                "categories": ["normal", "impaired"],
                "roles": ["confounder"],
                "description": "Valve pressure category.",
            },
        ),
        (
            "former coating finish",
            {
                "name": "coating_method_status",
                "type": "categorical",
                "categories": ["sprayed", "dipped", "rolled"],
                "value_aliases": {
                    "sprayed": ["mist finish"],
                    "dipped": ["bath finish"],
                    "rolled": ["former coating finish"],
                },
                "roles": ["confounder"],
                "description": "Coating method status.",
            },
        ),
        (
            "assembly panel texture",
            {
                "name": "assembly_ribbon",
                "type": "categorical",
                "categories": ["narrow", "wide"],
                "roles": ["confounder"],
                "description": "Assembly ribbon width class.",
            },
        ),
        (
            "ZX grade",
            {
                "name": "qy_grade",
                "type": "categorical",
                "categories": ["low", "high"],
                "roles": ["effect_modifier"],
                "description": "QY grade.",
            },
        ),
    ],
    ids=(
        "description-stuffing",
        "incidental-shared-name-token",
        "description-context-word",
        "numeric-category",
        "state-category",
        "value-alias",
        "shared-broad-assembly-token",
        "other-short-acronym",
    ),
)
def test_contract_grounding_rejects_unrelated_single_token_or_agent_controlled_matches(
    evidence_text,
    contract,
):
    grounding = ground_evidence_to_extraction_contract(
        _concept_evidence(evidence_text),
        contract,
    )

    assert grounding.supported is False
    assert grounding.schema_version == EVIDENCE_CONTRACT_GROUNDING_VERSION


@pytest.mark.parametrize(
    "name, evidence_text",
    [
        ("baseline_status", "baseline status"),
        ("value_4", "value 4"),
        ("normal_state", "normal state"),
        ("baseline_measurement", "baseline measurement"),
    ],
)
def test_contract_grounding_rejects_generic_state_and_numeric_only_names(
    name,
    evidence_text,
):
    grounding = ground_evidence_to_extraction_contract(
        _concept_evidence(evidence_text),
        _continuous_contract(name, "A prerun synthetic measurement."),
    )

    assert grounding.supported is False
    assert grounding.match_rule == "invalid_generic_or_numeric_contract_name"
    assert grounding.required_name_anchors == ()


def test_contract_grounding_requires_all_anchors_in_one_concept_entry():
    grounding = ground_evidence_to_extraction_contract(
        _concept_evidence(["quartz panel", "copper ribbon"]),
        {
            "name": "quartz_ribbon",
            "type": "categorical",
            "categories": ["narrow", "wide"],
            "roles": ["confounder"],
            "description": "Quartz ribbon width class.",
        },
    )

    assert grounding.supported is False
    assert grounding.match_rule == "missing_required_name_anchors"
    assert grounding.matched_evidence_anchors == ("quartz", "ribbon")


@pytest.mark.parametrize(
    "name, evidence_text, expected_anchors",
    [
        ("quartz_width", "quartz width 24", ("quartz", "width")),
        ("qa_status", "QA present", ("qa",)),
        ("zrq_code", "ZRQ result", ("zrq",)),
        ("sensor_l1_result", "sensor L1", ("l1", "sensor")),
        ("ribbon_at_baseline", "ribbon", ("ribbon",)),
    ],
)
def test_contract_grounding_accepts_only_exact_anchors_and_structural_qualifiers(
    name,
    evidence_text,
    expected_anchors,
):
    grounding = ground_evidence_to_extraction_contract(
        _concept_evidence(evidence_text),
        _continuous_contract(name, f"Measurement for {name}."),
    )

    assert grounding.supported is True
    assert grounding.match_rule == "all_required_name_anchors"
    assert grounding.matched_evidence_anchors == expected_anchors
    assert grounding.matched_evidence_paths == ("content.concept",)
    assert grounding.as_dict()["schema_version"] == EVIDENCE_CONTRACT_GROUNDING_VERSION


@pytest.mark.parametrize(
    "name, evidence_text, partial_anchors",
    [
        ("surface_color", "surface hue", ("surface",)),
        ("quartz_width", "QW", ()),
        ("qx_status", "QY status", ()),
        ("xy_status", "XYZ status", ()),
        ("ribbons", "ribbon", ()),
    ],
)
def test_contract_grounding_does_not_infer_synonyms_acronyms_substrings_or_morphology(
    name,
    evidence_text,
    partial_anchors,
):
    grounding = ground_evidence_to_extraction_contract(
        _concept_evidence(evidence_text),
        _continuous_contract(name, f"Measurement for {name}."),
    )

    assert grounding.supported is False
    assert grounding.match_rule == "missing_required_name_anchors"
    assert grounding.matched_evidence_anchors == partial_anchors


@pytest.mark.parametrize(
    "name, evidence_text, missing_anchor",
    [
        ("quartz_signature", "signature", "quartz"),
        ("alloy_viscosity", "viscosity", "alloy"),
        ("inspection_profile", "profile", "inspection"),
        ("torque_gradient", "gradient", "torque"),
        ("annealing_duration", "duration", "annealing"),
        ("coating_duration", "duration", "coating"),
        ("spectrum_profile", "profile", "spectrum"),
    ],
)
def test_contract_grounding_requires_content_words_not_just_other_name_tokens(
    name,
    evidence_text,
    missing_anchor,
):
    grounding = ground_evidence_to_extraction_contract(
        _concept_evidence(evidence_text),
        _continuous_contract(name, f"Measurement for {name}."),
    )

    assert grounding.supported is False
    assert missing_anchor in grounding.required_name_anchors


def test_prompt_is_deterministic_across_mapping_and_source_order():
    inputs = _all_inputs()
    candidate = CandidateContract(_candidate_spec(), source_families=(TFIDF_TOPICS,))
    first = prepare_all_evidence_fusion(
        inputs,
        candidates=[candidate],
        max_candidates=1,
    )

    reversed_inputs = []
    for item in reversed(inputs):
        reversed_payload = {key: copy.deepcopy(item.payload[key]) for key in reversed(item.payload)}
        reversed_inputs.append(
            FoldEvidenceInput(item.source_kind, reversed_payload, item.provenance)
        )
    second = prepare_all_evidence_fusion(
        reversed_inputs,
        candidates=[candidate],
        max_candidates=1,
    )
    assert first.split_fingerprint == second.split_fingerprint
    assert first.render_prompt() == second.render_prompt()


def test_no_pool_uses_concept_bound_grounded_proposal_schema():
    request = prepare_all_evidence_fusion(
        [FoldEvidenceInput(TFIDF_TOPIC_SOURCE, _tfidf_payload(), _provenance())],
        max_candidates=2,
    )
    assert request.mode == "propose"
    evidence = next(
        block
        for block in request.evidence_blocks
        if any("alloy" in str(term.get("term")) for term in block.content.get("terms", []))
    )
    family = evidence.source_families[0]
    result = validate_all_evidence_fusion_response(
        request,
        {
            "proposals": [
                {
                    "name": "alloy_phase_measurement",
                    "type": "continuous",
                    "roles": ["confounder"],
                    "description": (
                        "Numeric alloy-phase value explicitly recorded before assignment; "
                        "missing when not documented."
                    ),
                    "supporting_evidence_ids": [evidence.evidence_id],
                    "supporting_source_families": [family],
                    "rationale": "the cited topic contains a repeated alloy phase",
                }
            ]
        },
    )
    assert result.mode == "propose"
    assert result.proposed_specs == (
        {
            "description": (
                "Numeric alloy-phase value explicitly recorded before assignment; "
                "missing when not documented."
            ),
            "name": "alloy_phase_measurement",
            "roles": ["confounder"],
            "type": "continuous",
        },
    )

    with pytest.raises(ValueError, match="unknown evidence IDs"):
        validate_all_evidence_fusion_response(
            request,
            {
                "proposals": [
                    {
                        "name": "another_measurement",
                        "type": "continuous",
                        "roles": ["confounder"],
                        "description": "A pre-treatment numeric measurement.",
                        "supporting_evidence_ids": ["evidence_semantic_name"],
                        "supporting_source_families": [family],
                        "rationale": "unsupported citation",
                    }
                ]
            },
        )

    with pytest.raises(ValueError, match="evidence unrelated to the proposed contract"):
        validate_all_evidence_fusion_response(
            request,
            {
                "proposals": [
                    {
                        "name": "quartz_width",
                        "type": "continuous",
                        "roles": ["confounder"],
                        "description": "Quartz width measurement.",
                        "supporting_evidence_ids": [evidence.evidence_id],
                        "supporting_source_families": [family],
                        "rationale": "This intentionally cites an unrelated real block.",
                    }
                ]
            },
        )


def test_initial_proposal_normalization_and_validation_preserve_value_aliases():
    request = prepare_all_evidence_fusion(
        [FoldEvidenceInput(TFIDF_TOPIC_SOURCE, _tfidf_payload(), _provenance())],
        max_candidates=1,
    )
    evidence = next(
        block
        for block in request.evidence_blocks
        if any("alloy" in str(term.get("term")) for term in block.content.get("terms", []))
    )
    response = {
        "proposals": [
            {
                "name": "alloy_phase_status",
                "type": "categorical",
                "categories": ["negative", "positive", "indeterminate"],
                "value_aliases": {
                    "negative": ["not detected"],
                    "positive": ["detected"],
                    "indeterminate": ["equivocal"],
                },
                "roles": ["effect_modifier"],
                "description": "Prerun alloy phase result using calibration-specific categories.",
                "supporting_evidence_ids": [evidence.evidence_id],
                "supporting_source_families": [evidence.source_families[0]],
                "rationale": "The cited effect topic explicitly names an alloy phase.",
            }
        ]
    }

    result = validate_all_evidence_fusion_response(request, response)
    assert result.proposed_specs[0]["value_aliases"] == response["proposals"][0]["value_aliases"]

    client = _FakeClient([json.dumps(response)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client
    assert agent.propose(request.context()) == response


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0)
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            model="mock-remote-model",
            id=f"mock-response-{len(self.calls)}",
            created=0,
            usage=None,
        )


class _FakeClient:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)
        self.models = SimpleNamespace()


def _selection_request(*, pool_size: int = 1, maximum: int = 1):
    contracts = [
        CandidateContract(
            _candidate_spec(f"inlet_valve_position_{index}"),
            source_families=(BOW_NUISANCE,),
        )
        for index in range(1, pool_size + 1)
    ]
    return prepare_all_evidence_fusion(
        [FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(), _provenance())],
        candidates=contracts,
        max_candidates=maximum,
    )


def test_candidate_contract_rejects_identifiers_but_accepts_temporal_wording():
    with pytest.raises(ValueError, match="identifier rather than a patient variable"):
        CandidateContract(_candidate_spec("account_number"))

    temporal = {
        **_candidate_spec("process_response_status"),
        "description": "Response to the current process setting at follow-up.",
    }
    assert CandidateContract(temporal).extraction_spec == temporal


@pytest.mark.parametrize(
    "categories",
    [
        ["required for categorical variables", "pre-treatment"],
        ["canonical categories for categorical variables", "not_documented"],
        ["category_a", "category_b"],
    ],
)
def test_candidate_contract_rejects_instructional_or_placeholder_categories(categories):
    spec = _candidate_spec("documented_material_phase")
    spec["categories"] = categories

    with pytest.raises(ValueError, match="instructional or placeholder"):
        CandidateContract(spec)


@pytest.mark.parametrize(
    "categories",
    [
        ["crystalline", "amorphous", "other", "not_documented"],
        ["phase_one", "phase_two", "phase_three", "phase_four", "not_documented"],
        ["negative", "positive", "not_documented"],
        ["baseline", "follow_up", "not_documented"],
        ["not_required", "required", "not_documented"],
        ["0", "1", "2", "3", "4", "not_documented"],
    ],
)
def test_candidate_contract_preserves_variable_specific_categories(categories):
    spec = _candidate_spec("documented_material_phase")
    spec["categories"] = categories

    assert CandidateContract(spec).extraction_spec["categories"] == categories


def test_candidate_contract_rejects_more_than_eight_categories():
    spec = _candidate_spec("documented_material_phase")
    spec["categories"] = [f"state_{index}" for index in range(9)]

    with pytest.raises(ValueError, match="at most eight"):
        CandidateContract(spec)


def test_candidate_contract_rejects_normalized_duplicate_categories():
    spec = _candidate_spec("documented_material_phase")
    spec["categories"] = ["phase_three", "Phase Three", "not_documented"]

    with pytest.raises(ValueError, match="distinct after case/spacing normalization"):
        CandidateContract(spec)


@pytest.mark.parametrize(
    ("aliases", "message"),
    [
        ({"unknown": ["other"]}, "keys must exactly match"),
        ({"absent": "no"}, "non-empty string list"),
        ({"absent": ["present"]}, "normalized collision"),
        ({"absent": ["no"], "present": [" NO "]}, "normalized collision"),
        ({"absent": ["absent"]}, "normalized collision"),
    ],
)
def test_candidate_contract_rejects_ambiguous_value_alias_maps(aliases, message):
    spec = _candidate_spec("documented_material_phase")
    spec["value_aliases"] = aliases

    with pytest.raises(ValueError, match=message):
        CandidateContract(spec)


def test_candidate_contract_preserves_collision_free_value_alias_map():
    spec = _candidate_spec("documented_material_phase")
    spec["value_aliases"] = {
        "absent": ["negative", "not present"],
        "present": ["positive", "detected"],
    }

    assert CandidateContract(spec).extraction_spec["value_aliases"] == spec["value_aliases"]


def test_placeholder_categories_cannot_enter_selection_candidate_pool():
    invalid = _candidate_spec("assembly_phase")
    invalid["categories"] = ["required for categorical variables", "pre-treatment"]

    with pytest.raises(ValueError, match="instructional or placeholder"):
        prepare_all_evidence_fusion(
            [FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(), _provenance())],
            candidates=[invalid],
            max_candidates=1,
        )


def _valid_selection_response(request, candidate_id="candidate_0001"):
    evidence = next(
        block for block in request.evidence_blocks if BOW_NUISANCE in block.source_families
    )
    return {
        "selected_candidate_ids": [candidate_id],
        "selection_notes": [
            {
                "candidate_id": candidate_id,
                "supporting_evidence_ids": [evidence.evidence_id],
                "supporting_source_families": [BOW_NUISANCE],
                "reason": "supported by the cited fold-training sparse-text evidence",
            }
        ],
    }


def test_context_renderer_and_openai_agent_return_validated_fusion_object():
    request = _selection_request()
    context = request.context()
    response = _valid_selection_response(request)
    assert render_all_evidence_fusion_context_prompt(context) == request.render_prompt()
    assert all_evidence_fusion_response_issues(response, context) == []

    client = _FakeClient([json.dumps(response)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client

    validated = agent.propose(context)

    assert validated == response
    assert len(client.completions.calls) == 1
    assert client.completions.calls[0]["messages"][0]["content"] == request.render_prompt()


def test_proposal_prompt_uses_variable_specific_category_contract_version():
    request = prepare_all_evidence_fusion(
        [FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(), _provenance())],
        max_candidates=2,
    )
    context = request.context()

    assert FUSION_PROMPT_VERSION != "all_evidence_candidate_fusion_v1"
    assert context["prompt_version"] == FUSION_PROMPT_VERSION
    assert context["source_text_temporal_policy"]["policy"] == SOURCE_TEXT_TEMPORAL_POLICY
    assert context["source_text_temporal_policy"]["temporal_boundary_enforced"] is False
    assert context["response_contract"]["proposals"][0]["categories"] == [
        "absent",
        "present",
    ]
    assert "values specific to that variable" in request.render_prompt()


def test_openai_agent_repairs_placeholder_categories_before_selection():
    request = prepare_all_evidence_fusion(
        [FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(), _provenance())],
        max_candidates=4,
    )
    evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_NUISANCE,)
    )
    continuous = {
        "name": "inlet_valve_position_score",
        "type": "continuous",
        "roles": ["confounder"],
        "description": "Numeric valve position score documented before treatment selection.",
        "supporting_evidence_ids": [evidence.evidence_id],
        "supporting_source_families": [BOW_NUISANCE],
        "rationale": "The cited fold-training evidence supports this measure.",
    }
    invalid_stage = {
        "name": "inlet_valve_position",
        "type": "categorical",
        "categories": ["required for categorical variables", "pre-treatment"],
        "roles": ["effect_modifier"],
        "description": "Inlet valve position documented before treatment selection.",
        "supporting_evidence_ids": [evidence.evidence_id],
        "supporting_source_families": [BOW_NUISANCE],
        "rationale": "The cited fold-training evidence supports stage.",
    }
    repaired_stage = {
        **invalid_stage,
        "categories": [
            "poor",
            "intermediate",
            "good",
            "not_documented",
        ],
    }
    invalid = {"proposals": [continuous, invalid_stage]}
    repaired = {"proposals": [continuous, repaired_stage]}
    assert (
        "instructional or placeholder"
        in all_evidence_fusion_response_issues(
            invalid,
            request.context(),
        )[0]
    )

    client = _FakeClient([json.dumps(invalid), json.dumps(repaired)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=1,
        )
    )
    agent._client = client

    assert agent.propose(request.context()) == repaired
    assert len(client.completions.calls) == 2
    repair_prompt = client.completions.calls[1]["messages"][-1]["content"]
    assert "instructional or placeholder" in repair_prompt
    assert "specific to that variable" in repair_prompt


def test_openai_agent_repairs_over_cap_fusion_response():
    request = _selection_request(pool_size=2, maximum=1)
    invalid = {
        "selected_candidate_ids": ["candidate_0001", "candidate_0002"],
        "selection_notes": [],
    }
    repaired = _valid_selection_response(request)
    client = _FakeClient([json.dumps(invalid), json.dumps(repaired)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=1,
        )
    )
    agent._client = client

    assert agent.propose(request.context()) == repaired
    assert len(client.completions.calls) == 2
    repair = client.completions.calls[1]["messages"][-1]["content"]
    assert "all-evidence fusion response" in repair
    assert "selection exceeds max_candidates" in repair
    assert "Do not return or alter extraction specs" in repair


def test_fusion_repair_prompt_contains_exact_evidence_family_allowlist():
    request = _selection_request()
    prompt = build_all_evidence_fusion_repair_prompt(
        ["selection_notes[0] cites an unsupported source family"],
        request.context(),
    )
    mapping_header = "Authoritative evidence_id -> allowed source_families mapping:\n"
    rules_header = "\n\nCitation correction rules"
    serialized_mapping = prompt.split(mapping_header, 1)[1].split(rules_header, 1)[0]
    expected_mapping = {
        block.evidence_id: list(block.source_families) for block in request.evidence_blocks
    }

    assert json.loads(serialized_mapping) == expected_mapping
    assert "the mapping above is exact and exhaustive" in prompt
    assert "subset of the UNION" in prompt
    assert "Never invent an ID or family" in prompt


def test_openai_agent_normalizes_wrong_family_from_authoritative_citation():
    request = _selection_request()
    nuisance_evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_NUISANCE,)
    )
    invalid = _valid_selection_response(request)
    invalid["selection_notes"][0]["supporting_evidence_ids"] = [nuisance_evidence.evidence_id]
    invalid["selection_notes"][0]["supporting_source_families"] = [BOW_R_LOSS]
    with pytest.raises(ValueError, match="not present in cited evidence"):
        validate_all_evidence_fusion_response(request, invalid)

    expected = _valid_selection_response(request)
    client = _FakeClient([json.dumps(invalid)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=1,
        )
    )
    agent._client = client

    assert agent.propose(request.context()) == expected
    assert len(client.completions.calls) == 1
    audit = agent.last_response_trace["citation_family_normalization"]
    assert audit["eligible_item_count"] == 1
    assert audit["changed_item_count"] == 1
    assert audit["items"] == [
        {
            "path": "selection_notes[0]",
            "supporting_evidence_ids": [nuisance_evidence.evidence_id],
            "original_field_present": True,
            "original_supporting_source_families": [BOW_R_LOSS],
            "derived_supporting_source_families": [BOW_NUISANCE],
            "changed": True,
        }
    ]


def test_openai_agent_derives_proposal_families_in_canonical_order():
    payload = _legacy_payload()
    payload["context"]["evidence_digest"]["confounders"]["bow_blurbs"][0]["rows"][0][
        "feature"
    ] = "alloy phase score"
    request = prepare_all_evidence_fusion(
        [FoldEvidenceInput(LEGACY_ALL_SOURCE, payload, _provenance())],
        max_candidates=2,
    )
    nuisance_evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_NUISANCE,)
    )
    r_loss_evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_R_LOSS,)
    )
    response = {
        "proposals": [
            {
                "name": "alloy_phase_measure",
                "type": "continuous",
                "roles": ["confounder"],
                "description": "Numeric alloy-phase measure documented before assignment.",
                "supporting_evidence_ids": [
                    r_loss_evidence.evidence_id,
                    nuisance_evidence.evidence_id,
                ],
                "supporting_source_families": [BOW_R_LOSS],
                "rationale": "the cited sparse evidence supports this alloy phase",
            }
        ]
    }
    client = _FakeClient([json.dumps(response)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client

    normalized = agent.propose(request.context())

    assert normalized["proposals"][0]["supporting_source_families"] == [
        BOW_NUISANCE,
        BOW_R_LOSS,
    ]
    audit = agent.last_response_trace["citation_family_normalization"]
    assert audit["canonical_family_order"] == list(ALL_SOURCE_FAMILIES)
    assert audit["changed_item_count"] == 1


def test_fresh_proposal_prunes_only_unrelated_known_citation_deterministically():
    request = prepare_all_evidence_fusion(
        [FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(), _provenance())],
        max_candidates=4,
    )
    nuisance_evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_NUISANCE,)
    )
    r_loss_evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_R_LOSS,)
    )
    mixed = {
        "name": "inlet_valve_position_score",
        "type": "continuous",
        "roles": ["confounder"],
        "description": "Numeric valve position score documented before treatment selection.",
        "supporting_evidence_ids": [
            nuisance_evidence.evidence_id,
            r_loss_evidence.evidence_id,
        ],
        # This is already the correct family after pruning, so the family audit
        # must still count the item as changed because its evidence IDs changed.
        "supporting_source_families": [BOW_NUISANCE],
        "rationale": "The cited evidence supports this baseline measure.",
    }
    all_unrelated = {
        "name": "quartz_width",
        "type": "continuous",
        "roles": ["confounder"],
        "description": "Numeric quartz width documented in the source.",
        "supporting_evidence_ids": [r_loss_evidence.evidence_id],
        "supporting_source_families": [BOW_R_LOSS],
        "rationale": "The cited evidence supports this baseline measure.",
    }
    raw_response = {"proposals": [mixed, all_unrelated]}

    # The standalone boundary used by cached or injected responses stays strict.
    with pytest.raises(ValueError, match="cites evidence unrelated"):
        validate_all_evidence_fusion_response(request, raw_response)

    original = copy.deepcopy(raw_response)
    normalized_one, audit_one = _normalize_agent_response_citation_families(
        raw_response,
        request.context(),
    )
    normalized_two, audit_two = _normalize_agent_response_citation_families(
        copy.deepcopy(raw_response),
        request.context(),
    )

    assert raw_response == original
    assert (
        normalized_one
        == normalized_two
        == {
            "proposals": [
                {
                    **mixed,
                    "supporting_evidence_ids": [nuisance_evidence.evidence_id],
                }
            ]
        }
    )
    assert audit_one == audit_two
    validate_all_evidence_fusion_response(request, normalized_one)

    grounding = audit_one["citation_grounding_normalization"]
    rows_by_path = {row["path"]: row for row in grounding["items"]}
    assert grounding["evaluated_item_count"] == 2
    assert grounding["evaluated_citation_count"] == 3
    assert grounding["retained_citation_count"] == 1
    assert grounding["dropped_unrelated_citation_count"] == 2
    assert grounding["changed_item_count"] == 2
    assert grounding["zero_grounding_item_count"] == 1
    assert grounding["removed_zero_grounding_item_count"] == 1
    assert grounding["left_for_remote_repair_item_count"] == 0
    assert rows_by_path["proposals[0]"]["original_supporting_evidence_ids"] == [
        nuisance_evidence.evidence_id,
        r_loss_evidence.evidence_id,
    ]
    assert rows_by_path["proposals[0]"]["retained_supporting_evidence_ids"] == [
        nuisance_evidence.evidence_id
    ]
    assert rows_by_path["proposals[0]"]["dropped_unrelated_evidence_ids"] == [
        r_loss_evidence.evidence_id
    ]
    assert rows_by_path["proposals[1]"]["item_retained"] is False
    assert audit_one["citation_family_normalization"]["changed_item_count"] == 1
    assert audit_one["rejected_items"][0]["reason_code"] == ("no_lexically_grounded_citations")

    # Audit rows identify opaque IDs and matcher outcomes, never evidence text.
    serialized_audit = json.dumps(audit_one, sort_keys=True)
    assert "inlet valve position score" not in serialized_audit
    assert "alloy phase status" not in serialized_audit

    client = _FakeClient([json.dumps(raw_response)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client

    assert agent.propose(request.context()) == normalized_one
    assert len(client.completions.calls) == 1


def test_only_unrelated_proposal_citation_triggers_semantic_repair():
    request = prepare_all_evidence_fusion(
        [FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(), _provenance())],
        max_candidates=4,
    )
    nuisance_evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_NUISANCE,)
    )
    r_loss_evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_R_LOSS,)
    )
    unrelated = {
        "proposals": [
            {
                "name": "quartz_width",
                "type": "continuous",
                "roles": ["confounder"],
                "description": "Numeric quartz width documented in the source.",
                "supporting_evidence_ids": [r_loss_evidence.evidence_id],
                "supporting_source_families": [BOW_R_LOSS],
                "rationale": "The cited evidence supports this baseline measure.",
            }
        ]
    }
    corrected = {
        "proposals": [
            {
                "name": "inlet_valve_position_score",
                "type": "continuous",
                "roles": ["confounder"],
                "description": "Numeric valve position score documented before treatment selection.",
                "supporting_evidence_ids": [nuisance_evidence.evidence_id],
                "supporting_source_families": [BOW_NUISANCE],
                "rationale": "The cited evidence supports this baseline measure.",
            }
        ]
    }
    client = _FakeClient([json.dumps(unrelated), json.dumps(corrected)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=1,
        )
    )
    agent._client = client

    assert agent.propose(request.context()) == corrected
    assert len(client.completions.calls) == 2
    repair_prompt = client.completions.calls[1]["messages"][-1]["content"]
    assert "semantic validation:" in repair_prompt
    assert "cites evidence unrelated" in repair_prompt
    assert "normalized exact lexical identity anchors" in repair_prompt

    attempts = agent.last_response_trace["repair_attempts"]
    first_audit = attempts[0]["fresh_response_normalization"]
    assert first_audit["salvage_applied"] is False
    assert first_audit["retained_item_count"] == 0
    first_grounding = first_audit["citation_grounding_normalization"]
    assert first_grounding["zero_grounding_item_count"] == 1
    assert first_grounding["removed_zero_grounding_item_count"] == 0
    assert first_grounding["left_for_remote_repair_item_count"] == 1
    assert first_audit["rejections_applied_to_response"] is False


def test_fresh_openai_response_drops_one_category_row_but_standalone_stays_strict():
    request = prepare_all_evidence_fusion(
        [FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(), _provenance())],
        max_candidates=4,
    )
    evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_NUISANCE,)
    )
    valid = {
        "name": "inlet_valve_position_score",
        "type": "continuous",
        "roles": ["confounder"],
        "description": "Numeric valve position score documented before treatment selection.",
        "supporting_evidence_ids": [evidence.evidence_id],
        "supporting_source_families": [BOW_NUISANCE],
        "rationale": "The cited fold-training evidence supports this baseline measure.",
    }
    one_category = {
        "name": "baseline_documented_group",
        "type": "categorical",
        "categories": ["present"],
        "roles": ["effect_modifier"],
        "description": "Baseline group documented before treatment selection.",
        "supporting_evidence_ids": [evidence.evidence_id],
        "supporting_source_families": [BOW_NUISANCE],
        "rationale": "The cited fold-training evidence supports this baseline group.",
    }
    raw_response = {"proposals": [valid, one_category]}

    # Cached and injected responses never pass through the fresh-response
    # normalizer and therefore retain the exact strict contract.
    with pytest.raises(ValueError, match=r"proposals\[1\]\.categories requires at least two"):
        validate_all_evidence_fusion_response(request, raw_response)

    client = _FakeClient([json.dumps(raw_response)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client

    normalized = agent.propose(request.context())

    assert len(client.completions.calls) == 1
    assert normalized == {"proposals": [valid]}
    validate_all_evidence_fusion_response(request, normalized)
    audit = agent.last_response_trace["fresh_response_normalization"]
    assert audit["salvage_applied"] is True
    assert audit["input_item_count"] == 2
    assert audit["retained_item_count"] == 1
    assert audit["rejected_items"] == [
        {
            "path": "proposals[1]",
            "reason_code": "malformed_spec",
            "reason": ("proposals[1].categories requires at least two and at most eight values"),
        }
    ]


def test_fresh_selection_normalization_keeps_ranked_known_grounded_ids_under_cap():
    request = _selection_request(pool_size=3, maximum=2)
    evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_NUISANCE,)
    )

    def note(candidate_id):
        return {
            "candidate_id": candidate_id,
            "supporting_evidence_ids": [evidence.evidence_id],
            "supporting_source_families": [BOW_R_LOSS],
            "reason": "Supported by the cited fold-training evidence.",
        }

    raw_response = {
        "selected_candidate_ids": [
            "candidate_0002",
            "unknown_candidate",
            "candidate_0002",
            "candidate_0001",
            "candidate_0003",
        ],
        # Note order is deliberately different from selected rank order.
        "selection_notes": [
            note("candidate_0001"),
            note("candidate_0003"),
            note("candidate_0002"),
        ],
    }
    client = _FakeClient([json.dumps(raw_response)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client

    normalized = agent.propose(request.context())

    assert normalized["selected_candidate_ids"] == ["candidate_0002", "candidate_0001"]
    assert [item["candidate_id"] for item in normalized["selection_notes"]] == [
        "candidate_0002",
        "candidate_0001",
    ]
    assert all(
        item["supporting_source_families"] == [BOW_NUISANCE]
        for item in normalized["selection_notes"]
    )
    validate_all_evidence_fusion_response(request, normalized)
    audit = agent.last_response_trace["fresh_response_normalization"]
    assert audit["retained_item_count"] == 2
    assert [item["reason_code"] for item in audit["rejected_items"]] == [
        "unknown_selected_candidate_id",
        "duplicate_selected_candidate_id",
        "max_candidates_truncation",
        "unretained_selection_note",
    ]


def test_fresh_selection_note_prunes_only_unrelated_known_citation():
    request = _selection_request()
    nuisance_evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_NUISANCE,)
    )
    r_loss_evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_R_LOSS,)
    )
    raw_response = _valid_selection_response(request)
    raw_response["selection_notes"][0]["supporting_evidence_ids"] = [
        nuisance_evidence.evidence_id,
        r_loss_evidence.evidence_id,
    ]
    # The already-correct retained family ensures the change is attributable to
    # semantic ID pruning rather than ordinary redundant-family correction.
    raw_response["selection_notes"][0]["supporting_source_families"] = [BOW_NUISANCE]

    with pytest.raises(ValueError, match="cites evidence unrelated"):
        validate_all_evidence_fusion_response(request, raw_response)

    client = _FakeClient([json.dumps(raw_response)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client

    normalized = agent.propose(request.context())

    assert normalized["selected_candidate_ids"] == ["candidate_0001"]
    assert normalized["selection_notes"] == [
        {
            **raw_response["selection_notes"][0],
            "supporting_evidence_ids": [nuisance_evidence.evidence_id],
        }
    ]
    validate_all_evidence_fusion_response(request, normalized)
    audit = agent.last_response_trace["fresh_response_normalization"]
    grounding = audit["citation_grounding_normalization"]
    assert grounding["evaluated_item_count"] == 1
    assert grounding["retained_citation_count"] == 1
    assert grounding["dropped_unrelated_citation_count"] == 1
    assert grounding["changed_item_count"] == 1
    assert grounding["zero_grounding_item_count"] == 0
    assert grounding["removed_zero_grounding_item_count"] == 0
    assert grounding["items"][0]["dropped_unrelated_evidence_ids"] == [r_loss_evidence.evidence_id]
    assert audit["citation_family_normalization"]["changed_item_count"] == 1


def test_fresh_selection_drops_only_candidate_with_zero_grounded_note():
    request = _selection_request(pool_size=2, maximum=2)
    nuisance_evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_NUISANCE,)
    )
    r_loss_evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_R_LOSS,)
    )

    def note(candidate_id, evidence, family):
        return {
            "candidate_id": candidate_id,
            "supporting_evidence_ids": [evidence.evidence_id],
            "supporting_source_families": [family],
            "reason": "Supported by the cited fold-training evidence.",
        }

    raw_response = {
        "selected_candidate_ids": ["candidate_0001", "candidate_0002"],
        "selection_notes": [
            note("candidate_0001", r_loss_evidence, BOW_R_LOSS),
            note("candidate_0002", nuisance_evidence, BOW_NUISANCE),
        ],
    }
    with pytest.raises(ValueError, match="cites evidence unrelated"):
        validate_all_evidence_fusion_response(request, raw_response)

    client = _FakeClient([json.dumps(raw_response)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client

    normalized = agent.propose(request.context())

    assert normalized == {
        "selected_candidate_ids": ["candidate_0002"],
        "selection_notes": [note("candidate_0002", nuisance_evidence, BOW_NUISANCE)],
    }
    validate_all_evidence_fusion_response(request, normalized)
    audit = agent.last_response_trace["fresh_response_normalization"]
    assert audit["rejections_applied_to_response"] is True
    assert [row["reason_code"] for row in audit["rejected_items"]] == [
        "no_lexically_grounded_citations",
        "missing_valid_selection_note",
    ]
    grounding_rows = audit["citation_grounding_normalization"]["items"]
    assert [row["path"] for row in grounding_rows] == [
        "selection_notes[0]",
        "selection_notes[1]",
    ]
    assert [row["normalization_disposition"] for row in grounding_rows] == [
        "removed_zero_grounding",
        "retained",
    ]


def test_only_unrelated_selection_note_triggers_semantic_repair():
    request = _selection_request()
    nuisance_evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_NUISANCE,)
    )
    r_loss_evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_R_LOSS,)
    )
    unrelated = _valid_selection_response(request)
    unrelated["selection_notes"][0]["supporting_evidence_ids"] = [r_loss_evidence.evidence_id]
    unrelated["selection_notes"][0]["supporting_source_families"] = [BOW_R_LOSS]
    corrected = _valid_selection_response(request)
    corrected["selection_notes"][0]["supporting_evidence_ids"] = [nuisance_evidence.evidence_id]

    client = _FakeClient([json.dumps(unrelated), json.dumps(corrected)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=1,
        )
    )
    agent._client = client

    assert agent.propose(request.context()) == corrected
    assert len(client.completions.calls) == 2
    repair_prompt = client.completions.calls[1]["messages"][-1]["content"]
    assert "semantic validation:" in repair_prompt
    assert "cites evidence unrelated" in repair_prompt
    first_audit = agent.last_response_trace["repair_attempts"][0]["fresh_response_normalization"]
    assert first_audit["salvage_applied"] is False
    assert first_audit["rejections_applied_to_response"] is False
    assert first_audit["citation_grounding_normalization"]["left_for_remote_repair_item_count"] == 1


def test_fresh_response_forbidden_content_is_rejected_before_row_salvage():
    request = prepare_all_evidence_fusion(
        [FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(), _provenance())],
        max_candidates=2,
    )
    evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_NUISANCE,)
    )
    valid = {
        "name": "baseline_documented_measure",
        "type": "continuous",
        "roles": ["confounder"],
        "description": "Numeric measure documented before treatment selection.",
        "supporting_evidence_ids": [evidence.evidence_id],
        "supporting_source_families": [BOW_NUISANCE],
        "rationale": "The cited fold-training evidence supports this baseline measure.",
    }
    forbidden = {
        **valid,
        "name": "other_baseline_measure",
        "oracle_score": 1.0,
    }
    client = _FakeClient([json.dumps({"proposals": [valid, forbidden]})])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client

    with pytest.raises(ValueError, match="forbidden oracle/true field"):
        agent.propose(request.context())


def test_openai_agent_rejects_mutated_fusion_candidate_id_without_repair():
    request = _selection_request()
    mutated = _valid_selection_response(request, candidate_id="baseline_status_1")
    client = _FakeClient([json.dumps(mutated)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client

    with pytest.raises(ValueError, match="unknown candidate IDs"):
        agent.propose(request.context())


@pytest.mark.parametrize(
    "artifact_scope",
    ["candidate_selection_inner_fit", "candidate_consistency_inner_train"],
)
def test_openai_agent_validates_proposal_mode_from_exact_inner_artifact(
    artifact_scope,
):
    provenance = FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=(10, 11, 12, 13),
        heldout_row_ids=(20, 21),
        scope="inner_train",
        inner_fold=2,
        artifact_id=f"exact-inner-{artifact_scope}",
    )
    payload = _legacy_payload()
    payload["scope"] = artifact_scope
    payload["inner_fold"] = 2
    request = prepare_all_evidence_fusion(
        [FoldEvidenceInput(LEGACY_ALL_SOURCE, payload, provenance)],
        max_candidates=2,
    )
    evidence = next(
        block for block in request.evidence_blocks if block.source_families == (BOW_NUISANCE,)
    )
    response = {
        "proposals": [
            {
                "name": "inlet_valve_position_score",
                "type": "continuous",
                "roles": ["confounder"],
                "description": (
                    "Numeric valve position score documented before treatment selection; "
                    "missing when not recorded."
                ),
                "supporting_evidence_ids": [evidence.evidence_id],
                "supporting_source_families": list(evidence.source_families),
                "rationale": "the cited inner-training evidence supports this measure",
            }
        ]
    }
    client = _FakeClient([json.dumps(response)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client

    assert request.mode == "propose"
    assert agent.propose(request.context()) == response
