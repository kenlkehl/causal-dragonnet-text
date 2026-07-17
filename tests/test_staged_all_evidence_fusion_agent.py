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
    LEGACY_ALL_SOURCE,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPICS,
    TFIDF_TOPIC_SOURCE,
    CandidateContract,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
    evidence_supports_extraction_contract,
    prepare_all_evidence_fusion,
    validate_all_evidence_fusion_response,
)
from oci.inference.staged_all_evidence_fusion_agent import (
    STAGED_FUSION_AUDIT_SCHEMA_VERSION,
    STAGED_SAME_NAME_MERGE_VERSION,
    STAGED_SELECTION_BACKFILL_VERSION,
    STAGED_SELECTION_UNION_POSTPROCESSING_VERSION,
    StagedAllEvidenceFusionAgent,
)

_CONTINUOUS_CONTRACT_NAME = "coolant_pressure_reading"
_CATEGORICAL_CONTRACT_NAME = "surface_finish_class"
_FIXTURE_CONCEPTS_BY_FAMILY = {
    BOW_NUISANCE: (
        _CONTINUOUS_CONTRACT_NAME,
        "remote_selected_measure",
        "remote_recurrence_anchor",
        "remote_stable_anchor",
        "first_union_contract",
    ),
    BOW_R_LOSS: (
        _CONTINUOUS_CONTRACT_NAME,
        "stable_confounder_measure",
        "complementary_modifier_measure",
        "multi_family_confounder_measure",
        "first_stable_modifier",
        "second_stable_modifier",
        "unselected_union_contract",
    ),
    TFIDF_TOPICS: (
        _CONTINUOUS_CONTRACT_NAME,
        "multi_family_confounder_measure",
        "stable_singleton_modifier",
        "recurrently_validated_modifier",
        "third_union_contract",
    ),
    TFIDF_ORPHAN_NGRAMS: (
        _CONTINUOUS_CONTRACT_NAME,
        _CATEGORICAL_CONTRACT_NAME,
        "multi_family_confounder_measure",
    ),
}


def _fixture_concept_phrases(family: str) -> list[str]:
    return [name.replace("_", " ") for name in _FIXTURE_CONCEPTS_BY_FAMILY[family]]


def _provenance() -> FoldEvidenceProvenance:
    return FoldEvidenceProvenance(
        outer_fold=2,
        train_row_ids=(1, 2, 3, 4),
        heldout_row_ids=(5, 6),
        artifact_id="generic-fold-training-evidence",
    )


def _legacy_payload() -> dict:
    return {
        "outer_fold": 2,
        "context": {
            "evidence_digest": {
                "confounders": {
                    "bow_blurbs": [
                        {
                            "source": "sparse_nuisance.confounder_overlap",
                            "meaning": "shared preassignment sparse-text signal",
                            "rows": [
                                {
                                    "feature": "private_fold_training_sensor_phrase",
                                    "score": 2.0,
                                },
                                *[
                                    {"feature": phrase, "score": 1.0}
                                    for phrase in _fixture_concept_phrases(BOW_NUISANCE)
                                ],
                            ],
                        }
                    ]
                },
                "effect_modifiers": {
                    "bow_blurbs": [
                        {
                            "source": "ensemble_r.pseudo_target_positive",
                            "meaning": "residualized sparse-text signal",
                            "rows": [
                                {"feature": "baseline sensor state phrase", "score": 1.5},
                                *[
                                    {"feature": phrase, "score": 1.0}
                                    for phrase in _fixture_concept_phrases(BOW_R_LOSS)
                                ],
                            ],
                        }
                    ]
                },
            }
        },
    }


def _tfidf_payload() -> dict:
    def topic(topic_id: str, term: str) -> dict:
        return {
            "topic_id": topic_id,
            "terms": [
                {"term": term, "loading": 0.8},
                *[
                    {"term": phrase, "loading": 0.7}
                    for phrase in _fixture_concept_phrases(TFIDF_TOPICS)
                ],
            ],
        }

    return {
        "outer_fold": 2,
        "discovery": {
            "topic_banks": {
                "treatment": {
                    "topics": [topic("treatment_topic_1", "baseline routing selection phrase")]
                },
                "effect": {
                    "topics": [topic("effect_topic_1", "baseline component heterogeneity phrase")]
                },
            },
            "effect_orphan_ngram_branch": {
                "selected_cluster_ids": ["orphan_cluster_1"],
                "selected_clusters": [
                    {
                        "cluster_id": "orphan_cluster_1",
                        "terms": [
                            {"term": "additional baseline sensor phrase", "fit_rank": 1},
                            *[
                                {"term": phrase, "fit_rank": index + 2}
                                for index, phrase in enumerate(
                                    _fixture_concept_phrases(TFIDF_ORPHAN_NGRAMS)
                                )
                            ],
                        ],
                    }
                ],
            },
        },
    }


def _request(*, maximum: int = 4):
    provenance = _provenance()
    return prepare_all_evidence_fusion(
        [
            FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(), provenance),
            FoldEvidenceInput(TFIDF_TOPIC_SOURCE, _tfidf_payload(), provenance),
        ],
        max_candidates=maximum,
    )


def _continuous_spec() -> dict:
    return {
        "name": _CONTINUOUS_CONTRACT_NAME,
        "type": "continuous",
        "roles": ["confounder"],
        "description": "Coolant pressure reading documented before system assignment.",
    }


def _categorical_spec() -> dict:
    return {
        "name": _CATEGORICAL_CONTRACT_NAME,
        "type": "categorical",
        "categories": ["matte", "gloss", "other_or_not_documented"],
        "roles": ["effect_modifier"],
        "description": "Surface finish class documented in the source text.",
    }


def _supporting_evidence(context: dict, spec: dict, family: str | None = None) -> dict:
    return next(
        block
        for block in context["evidence"]
        if (family is None or family in block["source_families"])
        and evidence_supports_extraction_contract(block, spec)
    )


def _proposal_for(context: dict, spec: dict, family: str) -> dict:
    block = _supporting_evidence(context, spec, family)
    return {
        "proposals": [
            {
                **copy.deepcopy(spec),
                "supporting_evidence_ids": [block["evidence_id"]],
                "supporting_source_families": [family],
                "rationale": "Supported by the cited fold-training evidence.",
            }
        ]
    }


def _named_continuous_spec(name: str, *, roles: list[str]) -> dict:
    spec = _continuous_spec()
    spec.update(
        {
            "name": name,
            "roles": list(roles),
            "description": f"Baseline sensor measure for the generic {name} contract.",
        }
    )
    return spec


def _proposal_rows(context: dict, entries: list[tuple[dict, list[str]]]) -> dict:
    proposals = []
    for spec, families in entries:
        evidence_ids = []
        for family in families:
            block = _supporting_evidence(context, spec, family)
            if block["evidence_id"] not in evidence_ids:
                evidence_ids.append(block["evidence_id"])
        proposals.append(
            {
                **copy.deepcopy(spec),
                "supporting_evidence_ids": evidence_ids,
                "supporting_source_families": list(families),
                "rationale": "Supported by the cited fold-training evidence.",
            }
        )
    return {"proposals": proposals}


class _SuccessfulFakeAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def propose(self, context):
        detached = copy.deepcopy(context)
        self.calls.append(detached)
        if context["mode"] == "select":
            candidates = context["candidates"][: context["max_candidates"]]
            notes = []
            for candidate in candidates:
                family = candidate["source_families"][0]
                evidence = _supporting_evidence(context, candidate["extraction_spec"], family)
                notes.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "supporting_evidence_ids": [evidence["evidence_id"]],
                        "supporting_source_families": [family],
                        "reason": "Selected using the cited original evidence block.",
                    }
                )
            return {
                "selected_candidate_ids": [item["candidate_id"] for item in candidates],
                "selection_notes": notes,
            }

        proposal_index = sum(call["mode"] == "propose" for call in self.calls)
        if proposal_index == 1:
            return _proposal_for(context, _continuous_spec(), BOW_NUISANCE)
        if proposal_index == 2:
            # Exact duplicate of the full-context contract, but cited through
            # the filtered request's newly assigned evidence namespace.
            return _proposal_for(context, _continuous_spec(), BOW_NUISANCE)
        if proposal_index == 3:
            return _proposal_for(context, _categorical_spec(), TFIDF_ORPHAN_NGRAMS)
        raise AssertionError("unexpected extra proposal call")


def test_reasoning_enabled_reaches_every_staged_request_and_final_selector():
    calls = []
    private_reasoning = "private reasoning must never enter the staged audit"

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            prompt = kwargs["messages"][0]["content"]
            context = json.loads(prompt.rsplit("Fusion context:\n", 1)[1])
            if context["mode"] == "propose":
                evidence = _supporting_evidence(context, _continuous_spec())
                payload = {
                    "proposals": [
                        {
                            **_continuous_spec(),
                            "roles": ["confounder", "effect_modifier"],
                            "supporting_evidence_ids": [evidence["evidence_id"]],
                            "supporting_source_families": [evidence["source_families"][0]],
                            "rationale": "Supported by cited fold-training evidence.",
                        }
                    ]
                }
            else:
                candidate = context["candidates"][0]
                family = candidate["source_families"][0]
                grounding = _supporting_evidence(context, candidate["extraction_spec"], family)
                payload = {
                    "selected_candidate_ids": [candidate["candidate_id"]],
                    "selection_notes": [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "supporting_evidence_ids": [grounding["evidence_id"]],
                            "supporting_source_families": [family],
                            "reason": "Selected from cited fold-training evidence.",
                        }
                    ],
                }
            message = SimpleNamespace(
                content=json.dumps(payload),
                reasoning_content=private_reasoning,
            )
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(
                choices=[choice],
                model="served-agent-model",
                id=f"response-{len(calls)}",
                created=0,
                usage=None,
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    base = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="served-agent-model",
            agent_enable_thinking=True,
            agent_thinking_token_budget=4096,
            agent_schema_repair_attempts=0,
        )
    )
    base._client = client
    staged = StagedAllEvidenceFusionAgent(base, final_max_candidates=1)

    response = staged.propose(_request(maximum=1).context())

    assert len(response["proposals"]) == 1
    assert len(calls) == 4
    assert all(
        call["extra_body"]
        == {
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_token_budget": 4096,
        }
        for call in calls
    )
    assert calls[-1]["response_format"] == {"type": "json_object"}
    audit = staged.last_stage_audit
    assert [stage["reasoning_trace_presence"] for stage in audit["stages"]] == [
        {
            "response_trace_available": True,
            "completion_attempt_count": 1,
            "reasoning_content_present_count": 1,
            "reasoning_present_count": 0,
            "any_reasoning_present": True,
        }
    ] * 4
    assert private_reasoning not in json.dumps(audit, sort_keys=True)


def test_staged_agent_deduplicates_remaps_and_returns_original_valid_response():
    request = _request()
    original_nuisance_id = next(
        block.evidence_id
        for block in request.evidence_blocks
        if BOW_NUISANCE in block.source_families
    )
    assert original_nuisance_id != "evidence_0001"

    fake = _SuccessfulFakeAgent()
    agent = StagedAllEvidenceFusionAgent(fake, final_max_candidates=2)
    response = agent.propose(request.context())

    assert [call["mode"] for call in fake.calls] == [
        "propose",
        "propose",
        "propose",
        "select",
    ]
    assert all(
        call["source_text_temporal_policy"] == request.context()["source_text_temporal_policy"]
        for call in fake.calls
    )
    confounder_context = fake.calls[1]
    assert all(block["role_hint"] == "confounder" for block in confounder_context["evidence"])
    assert all(
        set(block["source_families"]) <= set(ALL_SOURCE_FAMILIES)
        for block in confounder_context["evidence"]
    )
    modifier_context = fake.calls[2]
    assert all(block["role_hint"] == "effect_modifier" for block in modifier_context["evidence"])
    assert all(
        set(block["source_families"]) <= set(ALL_SOURCE_FAMILIES)
        for block in modifier_context["evidence"]
    )

    selection_context = fake.calls[3]
    assert selection_context["max_candidates"] == 2
    assert len(selection_context["candidates"]) == 2
    assert selection_context["evidence"] == request.context()["evidence"]

    validated = validate_all_evidence_fusion_response(request, response)
    assert [spec["name"] for spec in validated.proposed_specs] == [
        _CONTINUOUS_CONTRACT_NAME,
        _CATEGORICAL_CONTRACT_NAME,
    ]

    audit = agent.last_stage_audit
    assert audit is not None
    assert audit["schema_version"] == STAGED_FUSION_AUDIT_SCHEMA_VERSION
    assert audit["role_specific_proposal_policy"] == {
        "version": "role_specific_all_evidence_families_v1",
        "eligible_source_families": list(ALL_SOURCE_FAMILIES),
        "neural_query_moments_eligible": True,
        "matched_pair_htr_embedding_and_tfidf_evidence_eligible": True,
    }
    proposal_union = audit["proposal_union"]
    assert proposal_union["validated_proposal_count"] == 3
    assert proposal_union["unique_contract_count"] == 2
    assert proposal_union["exact_duplicate_count"] == 1
    assert proposal_union["same_name_merge"] == {
        "version": STAGED_SAME_NAME_MERGE_VERSION,
        "merged_contract_count": 0,
        "final_candidate_pool_count": 2,
    }
    safe_union = proposal_union["safe_union"]
    assert safe_union["exact_duplicate_candidate_ids"] == ["candidate_0002"]
    assert safe_union["compatible_role_merge_candidate_ids"] == []
    assert safe_union["omitted_conflict_candidate_ids"] == []
    assert safe_union["incompatible_variant_support_or_roles_propagated"] is False
    assert safe_union["semantic_fields_used_for_conflict_ranking"] is False
    assert audit["stages"][1]["evidence_id_map_to_original"]["evidence_0001"] == (
        original_nuisance_id
    )
    assert audit["stages"][1]["mapped_grounding_evidence_ids"] == [original_nuisance_id]
    assert audit["stages"][-1]["selected_candidate_ids"] == [
        "candidate_0001",
        "candidate_0002",
    ]
    assert audit["selection_backfill_version"] == STAGED_SELECTION_BACKFILL_VERSION
    assert (
        audit["selection_union_postprocessing_version"]
        == STAGED_SELECTION_UNION_POSTPROCESSING_VERSION
    )
    assert audit["stages"][-1]["remote_selected_count"] == 2
    assert audit["stages"][-1]["final_selected_count"] == 2
    assert audit["stages"][-1]["backfilled_candidate_ids"] == []
    serialized_audit = json.dumps(audit, sort_keys=True)
    assert "private_fold_training_phrase" not in serialized_audit
    assert _CONTINUOUS_CONTRACT_NAME not in serialized_audit

    audit["stages"][0]["stage"] = "mutated"
    assert agent.last_stage_audit["stages"][0]["stage"] == "full_evidence_proposal"


class _SelectionSupportAgent:
    def __init__(
        self,
        *,
        spec: dict,
        selection_evidence_id: str | None = None,
        selection_family: str = TFIDF_TOPICS,
    ) -> None:
        self.spec = copy.deepcopy(spec)
        self.selection_evidence_id = selection_evidence_id
        self.selection_family = selection_family
        self.calls: list[dict] = []

    def propose(self, context):
        self.calls.append(copy.deepcopy(context))
        if context["mode"] == "select":
            candidate = context["candidates"][0]
            evidence_id = self.selection_evidence_id
            if evidence_id is None:
                evidence_id = _supporting_evidence(
                    context,
                    candidate["extraction_spec"],
                    self.selection_family,
                )["evidence_id"]
            return {
                "selected_candidate_ids": [candidate["candidate_id"]],
                "selection_notes": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "supporting_evidence_ids": [evidence_id],
                        "supporting_source_families": [self.selection_family],
                        "reason": "Selected using additional grounded original evidence.",
                    }
                ],
            }

        proposal_index = sum(call["mode"] == "propose" for call in self.calls)
        if proposal_index == 1:
            return _proposal_for(context, self.spec, BOW_NUISANCE)
        return {"proposals": []}


def test_staged_agent_accepts_validated_selection_support_beyond_proposal_support():
    request = _request(maximum=2)
    fake = _SelectionSupportAgent(spec=_continuous_spec())
    agent = StagedAllEvidenceFusionAgent(fake, final_max_candidates=1)

    response = agent.propose(request.context())

    proposal = response["proposals"][0]
    assert proposal["supporting_source_families"] == [BOW_NUISANCE, TFIDF_TOPICS]
    assert len(proposal["supporting_evidence_ids"]) == 2
    assert set(proposal["supporting_evidence_ids"]) <= {
        block.evidence_id for block in request.evidence_blocks
    }
    validate_all_evidence_fusion_response(request, response)


def test_staged_agent_rejects_selection_support_outside_original_evidence_namespace():
    fake = _SelectionSupportAgent(
        spec=_continuous_spec(),
        selection_evidence_id="evidence_9999",
    )
    agent = StagedAllEvidenceFusionAgent(fake, final_max_candidates=1)

    with pytest.raises(ValueError, match="cites unknown evidence IDs"):
        agent.propose(_request(maximum=2).context())

    assert agent.stage_audits == []


def test_staged_agent_rejects_selection_support_without_lexical_grounding():
    spec = _named_continuous_spec("remote_selected_measure", roles=["confounder"])
    request = _request(maximum=2)
    unrelated = next(
        block
        for block in request.context()["evidence"]
        if TFIDF_TOPICS in block["source_families"]
        and not evidence_supports_extraction_contract(block, spec)
    )
    fake = _SelectionSupportAgent(
        spec=spec,
        selection_evidence_id=unrelated["evidence_id"],
    )
    agent = StagedAllEvidenceFusionAgent(fake, final_max_candidates=1)

    with pytest.raises(ValueError, match="cites evidence unrelated to the selected contract"):
        agent.propose(request.context())

    assert agent.stage_audits == []


class _SameNameSelectionAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def propose(self, context):
        self.calls.append(copy.deepcopy(context))
        if context["mode"] == "select":
            assert len(context["candidates"]) == 1
            candidate = context["candidates"][0]
            families = list(candidate["source_families"])
            evidence_ids = []
            for family in families:
                evidence = _supporting_evidence(context, candidate["extraction_spec"], family)
                if evidence["evidence_id"] not in evidence_ids:
                    evidence_ids.append(evidence["evidence_id"])
            return {
                "selected_candidate_ids": [candidate["candidate_id"]],
                "selection_notes": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "supporting_evidence_ids": evidence_ids,
                        "supporting_source_families": families,
                        "reason": "Selected the consolidated grounded contract.",
                    }
                ],
            }

        proposal_index = sum(call["mode"] == "propose" for call in self.calls)
        first = _continuous_spec()
        if proposal_index in {1, 2}:
            return _proposal_for(context, first, BOW_NUISANCE)
        if proposal_index == 3:
            distinct_same_name = _categorical_spec()
            distinct_same_name.update(
                {
                    "name": first["name"],
                    "description": "A different categorical sensor interpretation of the "
                    "same baseline name.",
                }
            )
            return _proposal_for(
                context,
                distinct_same_name,
                TFIDF_ORPHAN_NGRAMS,
            )
        raise AssertionError("unexpected extra proposal call")


def test_staged_agent_keeps_incompatible_same_name_support_out_of_winner():
    request = _request(maximum=3)
    fake = _SameNameSelectionAgent()
    agent = StagedAllEvidenceFusionAgent(fake, final_max_candidates=3)

    response = agent.propose(request.context())

    selection_context = fake.calls[-1]
    assert selection_context["mode"] == "select"
    assert len(selection_context["candidates"]) == 1
    retained = selection_context["candidates"][0]
    assert retained["extraction_spec"] == _continuous_spec()
    assert retained["source_families"] == [BOW_NUISANCE]

    assert [proposal["name"] for proposal in response["proposals"]] == [_continuous_spec()["name"]]
    assert response["proposals"][0]["type"] == "continuous"
    assert "categories" not in response["proposals"][0]
    assert response["proposals"][0]["roles"] == ["confounder"]
    assert response["proposals"][0]["supporting_source_families"] == [BOW_NUISANCE]
    validate_all_evidence_fusion_response(request, response)

    audit = agent.last_stage_audit
    assert audit is not None
    proposal_union = audit["proposal_union"]
    assert proposal_union["validated_proposal_count"] == 3
    assert proposal_union["unique_contract_count"] == 2
    assert proposal_union["exact_duplicate_count"] == 1
    assert proposal_union["same_name_merge"] == {
        "version": STAGED_SAME_NAME_MERGE_VERSION,
        "merged_contract_count": 0,
        "final_candidate_pool_count": 1,
    }
    safe_union = proposal_union["safe_union"]
    assert safe_union["compatible_role_merge_candidate_ids"] == []
    assert safe_union["omitted_conflict_candidate_ids"] == ["candidate_0003"]
    assert len(safe_union["conflicts"]) == 1
    assert safe_union["conflicts"][0]["differing_non_role_fields"] == [
        "type",
        "categories",
        "description",
    ]
    assert audit["stages"][-1]["candidate_pool_count"] == 1
    assert audit["stages"][-1]["remote_selected_candidate_ids"] == ["candidate_0001"]
    assert (
        audit["stages"][-1]["selection_union_postprocessing_version"]
        == STAGED_SELECTION_UNION_POSTPROCESSING_VERSION
    )


def test_staged_agent_clamps_final_cap_to_original_request_cap():
    request = _request(maximum=1)
    fake = _SuccessfulFakeAgent()
    agent = StagedAllEvidenceFusionAgent(fake, final_max_candidates=5)

    response = agent.propose(request.context())

    assert fake.calls[-1]["max_candidates"] == 1
    assert len(response["proposals"]) == 1
    assert agent.last_stage_audit["configured_final_cap"] == 5
    assert agent.last_stage_audit["effective_final_cap"] == 1
    validate_all_evidence_fusion_response(request, response)


class _CoverageBackfillFakeAgent:
    def __init__(
        self,
        stage_entries: list[list[tuple[dict, list[str]]]],
        *,
        selected_candidate_indexes: tuple[int, ...] = (0,),
    ) -> None:
        self.stage_entries = stage_entries
        self.selected_candidate_indexes = selected_candidate_indexes
        self.calls: list[dict] = []

    def propose(self, context):
        self.calls.append(copy.deepcopy(context))
        if context["mode"] == "select":
            selected = [context["candidates"][index] for index in self.selected_candidate_indexes]
            notes = []
            for candidate in selected:
                family = candidate["source_families"][0]
                evidence = _supporting_evidence(context, candidate["extraction_spec"], family)
                notes.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "supporting_evidence_ids": [evidence["evidence_id"]],
                        "supporting_source_families": [family],
                        "reason": "Remote selector retained the cited candidate.",
                    }
                )
            return {
                "selected_candidate_ids": [candidate["candidate_id"] for candidate in selected],
                "selection_notes": notes,
            }
        proposal_call = sum(call["mode"] == "propose" for call in self.calls) - 1
        return _proposal_rows(context, self.stage_entries[proposal_call])


def test_staged_agent_backfills_greedy_family_coverage_then_role_complementarity():
    remote = _named_continuous_spec("remote_selected_measure", roles=["confounder"])
    stable_conf = _named_continuous_spec("stable_confounder_measure", roles=["confounder"])
    complementary = _named_continuous_spec(
        "complementary_modifier_measure", roles=["effect_modifier"]
    )
    multi_coverage = _named_continuous_spec("multi_family_confounder_measure", roles=["confounder"])
    fake = _CoverageBackfillFakeAgent(
        [
            [
                (remote, [BOW_NUISANCE]),
                (stable_conf, [BOW_R_LOSS]),
                (complementary, [BOW_R_LOSS]),
                (multi_coverage, [TFIDF_ORPHAN_NGRAMS]),
            ],
            [(multi_coverage, [TFIDF_TOPICS])],
            [(multi_coverage, [BOW_R_LOSS])],
        ]
    )
    request = _request(maximum=5)
    agent = StagedAllEvidenceFusionAgent(fake, final_max_candidates=3)

    response = agent.propose(request.context())

    # The remote contract stays first.  Candidate 4 then wins by adding three
    # source families; candidate 3 beats the lower stable ID by adding the
    # effect-modifier role not yet covered by the selected contracts.
    assert [proposal["name"] for proposal in response["proposals"]] == [
        "remote_selected_measure",
        "multi_family_confounder_measure",
        "complementary_modifier_measure",
    ]
    assert response["proposals"][1]["supporting_source_families"] == [
        BOW_R_LOSS,
        TFIDF_TOPICS,
        TFIDF_ORPHAN_NGRAMS,
    ]
    original_ids = {block.evidence_id for block in request.evidence_blocks}
    assert set(response["proposals"][1]["supporting_evidence_ids"]) <= original_ids
    validate_all_evidence_fusion_response(request, response)

    selection_audit = agent.last_stage_audit["stages"][-1]
    assert selection_audit["remote_selected_candidate_ids"] == ["candidate_0001"]
    assert selection_audit["remote_selected_count"] == 1
    assert selection_audit["final_selected_count"] == 3
    assert selection_audit["backfilled_candidate_ids"] == [
        "candidate_0004",
        "candidate_0003",
    ]
    assert selection_audit["selection_backfill_version"] == STAGED_SELECTION_BACKFILL_VERSION
    assert agent.last_stage_audit["remote_selected_count"] == 1
    assert agent.last_stage_audit["final_selected_count"] == 3
    assert agent.last_stage_audit["backfilled_candidate_ids"] == [
        "candidate_0004",
        "candidate_0003",
    ]


def test_staged_agent_backfill_prefers_repeated_validated_contract_over_stable_id():
    remote = _named_continuous_spec("remote_recurrence_anchor", roles=["confounder"])
    stable_singleton = _named_continuous_spec(
        "stable_singleton_modifier", roles=["effect_modifier"]
    )
    recurrent = _named_continuous_spec("recurrently_validated_modifier", roles=["effect_modifier"])
    fake = _CoverageBackfillFakeAgent(
        [
            [
                (remote, [BOW_NUISANCE]),
                (stable_singleton, [TFIDF_TOPICS]),
                (recurrent, [TFIDF_TOPICS]),
            ],
            [(recurrent, [TFIDF_TOPICS])],
            [],
        ]
    )
    agent = StagedAllEvidenceFusionAgent(fake, final_max_candidates=2)

    response = agent.propose(_request(maximum=4).context())

    assert [proposal["name"] for proposal in response["proposals"]] == [
        "remote_recurrence_anchor",
        "recurrently_validated_modifier",
    ]
    assert agent.last_stage_audit["stages"][-1]["backfilled_candidate_ids"] == ["candidate_0003"]


def test_staged_agent_backfill_uses_stable_candidate_id_as_final_tie_breaker():
    remote = _named_continuous_spec("remote_stable_anchor", roles=["confounder"])
    first = _named_continuous_spec("first_stable_modifier", roles=["effect_modifier"])
    second = _named_continuous_spec("second_stable_modifier", roles=["effect_modifier"])
    fake = _CoverageBackfillFakeAgent(
        [
            [
                (remote, [BOW_NUISANCE]),
                (first, [BOW_R_LOSS]),
                (second, [BOW_R_LOSS]),
            ],
            [],
            [],
        ]
    )
    agent = StagedAllEvidenceFusionAgent(fake, final_max_candidates=2)

    response = agent.propose(_request(maximum=4).context())

    assert [proposal["name"] for proposal in response["proposals"]] == [
        "remote_stable_anchor",
        "first_stable_modifier",
    ]
    assert agent.last_stage_audit["stages"][-1]["backfilled_candidate_ids"] == ["candidate_0002"]


def test_staged_agent_backfill_preserves_remote_contract_order_before_append():
    first = _named_continuous_spec("first_union_contract", roles=["confounder"])
    append = _named_continuous_spec("unselected_union_contract", roles=["effect_modifier"])
    third = _named_continuous_spec("third_union_contract", roles=["effect_modifier"])
    fake = _CoverageBackfillFakeAgent(
        [
            [
                (first, [BOW_NUISANCE]),
                (append, [BOW_R_LOSS]),
                (third, [TFIDF_TOPICS]),
            ],
            [],
            [],
        ],
        selected_candidate_indexes=(2, 0),
    )
    agent = StagedAllEvidenceFusionAgent(fake, final_max_candidates=3)

    response = agent.propose(_request(maximum=4).context())

    assert [proposal["name"] for proposal in response["proposals"]] == [
        "third_union_contract",
        "first_union_contract",
        "unselected_union_contract",
    ]
    selection_audit = agent.last_stage_audit["stages"][-1]
    assert selection_audit["remote_selected_candidate_ids"] == [
        "candidate_0003",
        "candidate_0001",
    ]
    assert selection_audit["backfilled_candidate_ids"] == ["candidate_0002"]


class _FirstResponseAgent:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []

    def propose(self, context):
        self.calls.append(copy.deepcopy(context))
        return copy.deepcopy(self.response)


@pytest.mark.parametrize(
    "response, message",
    [
        (
            {
                "proposals": [
                    {
                        **_continuous_spec(),
                        "name": "patient_id",
                        "supporting_evidence_ids": ["evidence_0001"],
                        "supporting_source_families": [BOW_R_LOSS],
                        "rationale": "Cited evidence.",
                    }
                ]
            },
            "identifier rather than a patient variable",
        ),
        (
            {
                "proposals": [
                    {
                        **_continuous_spec(),
                        "oracle_score": 1.0,
                        "supporting_evidence_ids": ["evidence_0001"],
                        "supporting_source_families": [BOW_R_LOSS],
                        "rationale": "Cited evidence.",
                    }
                ]
            },
            "forbidden oracle/true field",
        ),
    ],
)
def test_staged_agent_fails_closed_on_unsafe_stage_response(response, message):
    fake = _FirstResponseAgent(response)
    agent = StagedAllEvidenceFusionAgent(fake)

    with pytest.raises(ValueError, match=message):
        agent.propose(_request().context())

    assert len(fake.calls) == 1
    assert agent.stage_audits == []


def test_staged_agent_rejects_unsafe_original_context_before_dispatch():
    context = copy.deepcopy(_request().context())
    context["evidence"][0]["content"]["true_score"] = 1.0
    fake = _FirstResponseAgent({"proposals": []})
    agent = StagedAllEvidenceFusionAgent(fake)

    with pytest.raises(ValueError, match="forbidden oracle/true field"):
        agent.propose(context)

    assert fake.calls == []


class _ContractMutatingSelector(_SuccessfulFakeAgent):
    def propose(self, context):
        if context["mode"] != "select":
            return super().propose(context)
        self.calls.append(copy.deepcopy(context))
        candidate = context["candidates"][0]
        family = candidate["source_families"][0]
        evidence = _supporting_evidence(context, candidate["extraction_spec"], family)
        return {
            "selected_candidate_ids": [candidate["candidate_id"]],
            "selection_notes": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "supporting_evidence_ids": [evidence["evidence_id"]],
                    "supporting_source_families": [family],
                    "reason": "Selected using the cited original evidence block.",
                }
            ],
            "extraction_spec": _categorical_spec(),
        }


def test_staged_agent_fails_closed_if_final_selector_tries_to_rewrite_contract():
    fake = _ContractMutatingSelector()
    agent = StagedAllEvidenceFusionAgent(fake)

    with pytest.raises(ValueError, match="selection response contains unsupported fields"):
        agent.propose(_request().context())

    assert len(fake.calls) == 4
    assert agent.stage_audits == []


def test_staged_agent_requires_original_propose_mode_context():
    request = prepare_all_evidence_fusion(
        [
            FoldEvidenceInput(
                LEGACY_ALL_SOURCE,
                _legacy_payload(),
                _provenance(),
            )
        ],
        candidates=[CandidateContract(_continuous_spec(), source_families=(BOW_NUISANCE,))],
        max_candidates=1,
    )
    fake = _FirstResponseAgent({"selected_candidate_ids": [], "selection_notes": []})
    agent = StagedAllEvidenceFusionAgent(fake)

    with pytest.raises(ValueError, match="original propose-mode context"):
        agent.propose(request.context())

    assert fake.calls == []
