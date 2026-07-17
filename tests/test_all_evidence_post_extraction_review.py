from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.config import AgenticFeatureSearchConfig
from oci.inference.agentic_explicit_feature_forest import (
    CodexCLIFeatureSearchAgent,
    CodexCLIResponse,
    OpenAICompatibleFeatureSearchAgent,
    build_agent_prompt,
)
from oci.inference.all_evidence_fusion import (
    EVIDENCE_CONTRACT_GROUNDING_VERSION,
    source_text_temporal_policy_audit,
)
from oci.inference.all_evidence_post_extraction_review import (
    POST_EXTRACTION_REVIEW_PROMPT_VERSION,
    POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
    _normalize_fresh_post_extraction_review_response,
    apply_post_extraction_review_operations,
    build_post_extraction_review_repair_prompt,
    build_extraction_quality_diagnostics,
    build_redundancy_diagnostics,
    collect_post_extraction_diagnostic_targets,
    collect_post_extraction_diagnostic_ids,
    extraction_semantics_sha256,
    post_extraction_review_response_issues,
    validate_post_extraction_review_response,
)


def _continuous(name: str, *, roles: list[str] | None = None, description: str | None = None):
    return {
        "name": name,
        "type": "continuous",
        "roles": roles or ["confounder"],
        "description": description or f"Baseline numeric value for {name} before treatment.",
    }


def _categorical(name: str, *, roles: list[str] | None = None):
    return {
        "name": name,
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": roles or ["effect_modifier"],
        "description": f"Pretreatment {name} status.",
    }


def _response(operation: dict):
    return {
        "schema_version": POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
        "operations": [operation],
    }


def _operation(action: str, targets: list[str], contract=None):
    return {
        "action": action,
        "target_names": targets,
        "contract": contract,
        "supporting_diagnostic_ids": ["diagnostic_0001"],
        "supporting_evidence_ids": ["evidence_0001"],
        "reason": "The cited observable diagnostic supports this bounded revision.",
    }


def _evidence_catalog(concept: str):
    return [
        {
            "evidence_id": "evidence_0001",
            "source_families": ["bow_r_loss"],
            "role_hint": "effect_modifier",
            "content": {"concept": concept},
        }
    ]


def _review_context(specs=None, *, evidence=None):
    contracts = list(specs or [_continuous("baseline_measure")])
    return {
        "prompt_version": POST_EXTRACTION_REVIEW_PROMPT_VERSION,
        "source_text_temporal_policy": source_text_temporal_policy_audit(),
        "outer_fold": 1,
        "review_round": 1,
        "max_operations": 4,
        "current_contracts": contracts,
        "diagnostics": [
            {
                "diagnostic_id": "diagnostic_0001",
                "kind": "feature_quality",
                "feature_name": contracts[0]["name"],
                "coverage": 0.5,
            }
        ],
        "sanitized_evidence_catalog": list(evidence or []),
        "acceptance_gate_disclosed_to_agent": False,
    }


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.responses.pop(0))
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


def test_fresh_review_normalizes_compact_drop_without_weakening_standalone_validation():
    before = _continuous("baseline_measure")
    context = _review_context([before])
    compact = {
        "schema_version": POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
        "operations": [
            {
                "action": "drop",
                "target_names": ["baseline_measure"],
                "supporting_diagnostic_ids": ["diagnostic_0001"],
                "reason": "Observable extraction quality is too weak to retain this contract.",
            }
        ],
    }
    original = copy.deepcopy(compact)

    assert "missing fields=['contract', 'supporting_evidence_ids']" in (
        post_extraction_review_response_issues(compact, context)[0]
    )
    normalized_one, audit_one = _normalize_fresh_post_extraction_review_response(
        compact,
        context,
    )
    normalized_two, audit_two = _normalize_fresh_post_extraction_review_response(
        copy.deepcopy(compact),
        context,
    )

    assert compact == original
    assert (
        normalized_one
        == normalized_two
        == {
            "schema_version": POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
            "operations": [
                {
                    **compact["operations"][0],
                    "contract": None,
                    "supporting_evidence_ids": [],
                }
            ],
        }
    )
    assert audit_one == audit_two
    assert audit_one["inserted_field_count"] == 2
    assert audit_one["operations"][0]["inserted_fields"] == [
        "contract",
        "supporting_evidence_ids",
    ]
    assert "baseline_measure" not in json.dumps(audit_one, sort_keys=True)
    assert "diagnostic_0001" not in json.dumps(audit_one, sort_keys=True)
    assert post_extraction_review_response_issues(normalized_one, context) == []


def test_fresh_review_normalizes_only_remote_response_schema_metadata():
    context = _review_context()
    response = _response(_operation("drop", ["baseline_measure"]))
    response["schema_version"] = POST_EXTRACTION_REVIEW_PROMPT_VERSION
    response["operations"][0]["contract"] = None
    response["operations"][0]["supporting_evidence_ids"] = []

    assert post_extraction_review_response_issues(response, context) == [
        "unsupported post-extraction review response schema"
    ]
    normalized, audit = _normalize_fresh_post_extraction_review_response(response, context)

    assert response["schema_version"] == POST_EXTRACTION_REVIEW_PROMPT_VERSION
    assert normalized["schema_version"] == POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION
    assert audit["original_response_schema_present"] is True
    assert audit["original_response_schema_matched"] is False
    assert audit["response_schema_normalized"] is True
    assert post_extraction_review_response_issues(normalized, context) == []


def test_fresh_review_normalizes_compact_stop_and_optional_re_role_evidence():
    before = _continuous("baseline_measure", roles=["confounder"])
    context = _review_context([before])
    compact_stop = {
        "schema_version": POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
        "operations": [
            {
                "action": "stop",
                "reason": "No defensible observable improvement remains.",
            }
        ],
    }
    normalized_stop, stop_audit = _normalize_fresh_post_extraction_review_response(
        compact_stop,
        context,
    )
    assert normalized_stop["operations"][0] == {
        "action": "stop",
        "target_names": [],
        "contract": None,
        "supporting_diagnostic_ids": [],
        "supporting_evidence_ids": [],
        "reason": "No defensible observable improvement remains.",
    }
    assert stop_audit["inserted_field_count"] == 4
    assert post_extraction_review_response_issues(normalized_stop, context) == []

    after = _continuous("baseline_measure", roles=["effect_modifier"])
    compact_re_role = {
        "schema_version": POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
        "operations": [
            {
                "action": "re_role",
                "target_names": ["baseline_measure"],
                "contract": after,
                "supporting_diagnostic_ids": ["diagnostic_0001"],
                "reason": "The observable diagnostic supports effect-modifier use only.",
            }
        ],
    }
    normalized_re_role, re_role_audit = _normalize_fresh_post_extraction_review_response(
        compact_re_role,
        context,
    )
    assert normalized_re_role["operations"][0]["supporting_evidence_ids"] == []
    assert re_role_audit["operations"][0]["inserted_fields"] == ["supporting_evidence_ids"]
    assert post_extraction_review_response_issues(normalized_re_role, context) == []


def test_fresh_review_removes_only_self_aliases_and_same_owner_duplicates():
    before = _categorical("inlet_valve")
    context = _review_context(
        [before],
        evidence=_evidence_catalog("inlet valve"),
    )
    revised = {
        **before,
        "categories": ["negative", "positive"],
        "value_aliases": {
            "negative": ["negative", "not detected", "Not_Detected"],
            "positive": ["positive", "detected", "DETECTED"],
        },
    }
    response = _response(_operation("revise", ["inlet_valve"], revised))

    with pytest.raises(ValueError, match="normalized collision"):
        validate_post_extraction_review_response(
            response,
            current_specs=[before],
            available_diagnostic_ids=["diagnostic_0001"],
            available_diagnostic_targets={"diagnostic_0001": ["inlet_valve"]},
            available_evidence_ids=["evidence_0001"],
            available_evidence_catalog=_evidence_catalog("inlet valve"),
        )

    normalized, audit = _normalize_fresh_post_extraction_review_response(response, context)
    assert normalized["operations"][0]["contract"]["value_aliases"] == {
        "negative": ["not detected"],
        "positive": ["detected"],
    }
    assert audit["self_alias_removed_count"] == 2
    assert audit["same_owner_duplicate_removed_count"] == 2
    assert post_extraction_review_response_issues(normalized, context) == []

    cross_owner = copy.deepcopy(response)
    cross_owner["operations"][0]["contract"]["value_aliases"] = {
        "negative": ["detected"],
        "positive": ["DETECTED"],
    }
    normalized_cross_owner, cross_owner_audit = _normalize_fresh_post_extraction_review_response(
        cross_owner, context
    )
    assert normalized_cross_owner == cross_owner
    assert cross_owner_audit["changed_operation_count"] == 0
    assert (
        "normalized collision"
        in post_extraction_review_response_issues(
            normalized_cross_owner,
            context,
        )[0]
    )

    unknown_owner = copy.deepcopy(response)
    unknown_owner["operations"][0]["contract"]["value_aliases"] = {
        "unsupported_category": ["unsupported_category"]
    }
    normalized_unknown_owner, unknown_owner_audit = (
        _normalize_fresh_post_extraction_review_response(unknown_owner, context)
    )
    assert normalized_unknown_owner == unknown_owner
    assert unknown_owner_audit["changed_operation_count"] == 0
    assert (
        "keys must exactly match declared categories"
        in post_extraction_review_response_issues(
            normalized_unknown_owner,
            context,
        )[0]
    )


def test_fresh_review_does_not_invent_semantic_fields_or_remove_unknown_fields():
    context = _review_context()
    missing_reason = {
        "schema_version": POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
        "operations": [
            {
                "action": "drop",
                "target_names": ["baseline_measure"],
                "supporting_diagnostic_ids": ["diagnostic_0001"],
            }
        ],
    }
    normalized_missing, audit_missing = _normalize_fresh_post_extraction_review_response(
        missing_reason,
        context,
    )
    assert "reason" not in normalized_missing["operations"][0]
    assert audit_missing["operations"][0]["missing_allowed_fields"] == [
        "contract",
        "reason",
        "supporting_evidence_ids",
    ]
    assert (
        "missing fields=['reason']"
        in post_extraction_review_response_issues(
            normalized_missing,
            context,
        )[0]
    )

    unexpected = copy.deepcopy(missing_reason)
    unexpected["operations"][0]["reason"] = "The cited diagnostic supports removal."
    unexpected["operations"][0]["confidence"] = 0.9
    normalized_unexpected, audit_unexpected = _normalize_fresh_post_extraction_review_response(
        unexpected, context
    )
    assert normalized_unexpected["operations"][0]["confidence"] == 0.9
    assert audit_unexpected["operations"][0]["unexpected_field_count"] == 1
    assert (
        "unexpected fields=['confidence']"
        in post_extraction_review_response_issues(
            normalized_unexpected,
            context,
        )[0]
    )


def test_fresh_openai_review_response_normalizes_compact_drop_in_one_call():
    context = _review_context()
    compact = {
        "schema_version": POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
        "operations": [
            {
                "action": "drop",
                "target_names": ["baseline_measure"],
                "supporting_diagnostic_ids": ["diagnostic_0001"],
                "reason": "Observable extraction quality is too weak to retain this contract.",
            }
        ],
    }
    client = _FakeClient([json.dumps(compact)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client

    result = agent.propose(context)

    assert len(client.completions.calls) == 1
    assert result["operations"][0]["contract"] is None
    assert result["operations"][0]["supporting_evidence_ids"] == []
    audit = agent.last_response_trace["fresh_response_normalization"]
    assert audit["inserted_field_count"] == 2


def test_unsafe_review_alias_reaches_actionable_remote_repair_prompt():
    before = _categorical("inlet_valve")
    context = _review_context(
        [before],
        evidence=_evidence_catalog("inlet valve"),
    )
    bad_contract = {
        **before,
        "categories": ["negative", "positive"],
        "value_aliases": {
            "negative": ["detected"],
            "positive": ["DETECTED"],
        },
    }
    invalid = _response(_operation("revise", ["inlet_valve"], bad_contract))
    corrected = _response(_operation("drop", ["inlet_valve"]))
    corrected["operations"][0].pop("contract")
    corrected["operations"][0].pop("supporting_evidence_ids")
    client = _FakeClient([json.dumps(invalid), json.dumps(corrected)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=1,
        )
    )
    agent._client = client

    result = agent.propose(context)

    assert result["operations"][0]["action"] == "drop"
    assert result["operations"][0]["contract"] is None
    assert result["operations"][0]["supporting_evidence_ids"] == []
    assert len(client.completions.calls) == 2
    repair_prompt = client.completions.calls[1]["messages"][-1]["content"]
    assert "exactly these six keys" in repair_prompt
    assert "contract:null" in repair_prompt
    assert "globally unique" in repair_prompt
    assert "normalized collision" in repair_prompt


def test_unrelated_review_citation_repair_lists_exact_grounded_alternatives():
    before = _categorical("qx_grade_status")
    context = _review_context(
        [before],
        evidence=[
            {
                "evidence_id": "evidence_0001",
                "source_families": ["bow_r_loss"],
                "content": {"concept": "QX grade"},
            },
            {
                "evidence_id": "evidence_0002",
                "source_families": ["tfidf_topics"],
                "content": {"concept": "copper grade"},
            },
        ],
    )
    replacement = _categorical("copper_grade_status")
    invalid = _response(_operation("replace", ["qx_grade_status"], replacement))
    invalid["operations"][0]["supporting_evidence_ids"] = [
        "evidence_0001",
        "evidence_0002",
    ]
    corrected = _response(_operation("drop", ["qx_grade_status"]))
    corrected["operations"][0].pop("contract")
    corrected["operations"][0].pop("supporting_evidence_ids")
    client = _FakeClient([json.dumps(invalid), json.dumps(corrected)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=1,
        )
    )
    agent._client = client

    result = agent.propose(context)

    assert result["operations"][0]["action"] == "drop"
    assert len(client.completions.calls) == 2
    repair_prompt = client.completions.calls[1]["messages"][-1]["content"]
    assert "Deterministic citation-grounding repair hints" in repair_prompt
    assert '"proposed_contract_name":"copper_grade_status"' in repair_prompt
    assert '"evidence_id":"evidence_0001"' in repair_prompt
    assert '"eligible_evidence_ids_for_exact_proposed_name":["evidence_0002"]' in (repair_prompt)
    assert "Do not repeat any failed contract/evidence pairing" in repair_prompt
    assert "diagnostic-grounded drop per unsafe target" in repair_prompt
    assert "do not stop" in repair_prompt


def test_grounding_repair_hints_cover_all_bad_operations_and_empty_eligible_set():
    context = _review_context(
        [_categorical("qx_grade_status"), _categorical("ribbon_texture_status")],
        evidence=[
            {
                "evidence_id": "evidence_0001",
                "source_families": ["bow_r_loss"],
                "content": {"concept": "QX grade"},
            },
            {
                "evidence_id": "evidence_0002",
                "source_families": ["tfidf_topics"],
                "content": {"concept": "copper grade"},
            },
            {
                "evidence_id": "evidence_0003",
                "source_families": ["matched_pair_uplift"],
                "content": {"concept": "ribbon texture"},
            },
        ],
    )
    first = _operation(
        "replace",
        ["qx_grade_status"],
        _categorical("copper_grade_status"),
    )
    first["supporting_evidence_ids"] = ["evidence_0001", "evidence_0002"]
    second = _operation(
        "replace",
        ["ribbon_texture_status"],
        _categorical("novel_quartz_status"),
    )
    second["supporting_evidence_ids"] = ["evidence_0003"]
    failed = {
        "schema_version": POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
        "operations": [first, second],
    }

    repair = build_post_extraction_review_repair_prompt(
        ["operations[0].replace cites evidence unrelated to the proposed contract"],
        context=context,
        failed_response=failed,
    )

    assert repair.count('"operation_index":') == 2
    assert '"operation_index":0' in repair
    assert '"failed_citations":[{"evidence_id":"evidence_0001"' in repair
    assert '"eligible_evidence_ids_for_exact_proposed_name":["evidence_0002"]' in repair
    assert '"operation_index":1' in repair
    assert '"proposed_contract_name":"novel_quartz_status"' in repair
    assert '"eligible_evidence_ids_for_exact_proposed_name":[]' in repair


def test_malformed_openai_review_response_repairs_without_stale_grounding_hints():
    context = _review_context()
    corrected = _response(_operation("drop", ["baseline_measure"]))
    corrected["operations"][0].pop("contract")
    corrected["operations"][0].pop("supporting_evidence_ids")
    client = _FakeClient(["not-json", json.dumps(corrected)])
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="mock-remote-model",
            agent_schema_repair_attempts=1,
        )
    )
    agent._client = client

    result = agent.propose(context)

    assert result["operations"][0]["action"] == "drop"
    repair_prompt = client.completions.calls[1]["messages"][-1]["content"]
    assert "malformed JSON" in repair_prompt
    assert "Deterministic citation-grounding repair hints" not in repair_prompt


def test_malformed_codex_review_response_repairs_without_stale_grounding_hints():
    context = _review_context()
    corrected = _response(_operation("drop", ["baseline_measure"]))
    corrected["operations"][0].pop("contract")
    corrected["operations"][0].pop("supporting_evidence_ids")
    responses = iter(
        [
            CodexCLIResponse(
                content="not-json",
                command=["codex", "exec"],
                stdout="",
                stderr="",
                returncode=0,
            ),
            CodexCLIResponse(
                content=json.dumps(corrected),
                command=["codex", "exec"],
                stdout="",
                stderr="",
                returncode=0,
            ),
        ]
    )
    prompts = []
    agent = CodexCLIFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_provider="codex_cli",
            agent_schema_repair_attempts=1,
        )
    )

    def run(prompt):
        prompts.append(prompt)
        return next(responses)

    agent._run = run

    result = agent.propose(context)

    assert result["operations"][0]["action"] == "drop"
    assert "malformed JSON" in prompts[1]
    assert "Deterministic citation-grounding repair hints" not in prompts[1]


def test_role_only_revision_reuses_extraction_values():
    before = _continuous("baseline_measure", roles=["confounder"])
    after = _continuous("baseline_measure", roles=["effect_modifier"])
    validated = validate_post_extraction_review_response(
        _response(_operation("re_role", ["baseline_measure"], after)),
        current_specs=[before],
        available_diagnostic_ids=["diagnostic_0001"],
        available_evidence_ids=["evidence_0001"],
    )
    applied = apply_post_extraction_review_operations([before], validated)
    assert not applied.reextract_specs
    assert applied.role_only_changed_names == ("baseline_measure",)
    assert extraction_semantics_sha256(before) == extraction_semantics_sha256(after)


def test_category_revision_reextracts_only_changed_contract():
    first = _categorical("inlet_valve")
    second = _continuous("sensor_reading")
    revised = {
        **first,
        "categories": ["negative", "positive", "indeterminate"],
        "description": "Pretreatment inlet valve result using the revised calibration categories.",
    }
    validated = validate_post_extraction_review_response(
        _response(_operation("revise", ["inlet_valve"], revised)),
        current_specs=[first, second],
        available_diagnostic_ids=["diagnostic_0001"],
        available_evidence_ids=["evidence_0001"],
        available_evidence_catalog=_evidence_catalog("inlet valve"),
    )
    applied = apply_post_extraction_review_operations([first, second], validated)
    assert applied.extraction_changed_names == ("inlet_valve",)
    assert [spec["name"] for spec in applied.reextract_specs] == ["inlet_valve"]
    assert [spec["name"] for spec in applied.specs] == ["inlet_valve", "sensor_reading"]


def test_merge_is_atomic_and_cannot_collide_with_untouched_feature():
    specs = [_continuous("gauge_a"), _continuous("gauge_b"), _continuous("gauge_c")]
    factor = _continuous("combined_gauge", roles=["confounder", "effect_modifier"])
    validated = validate_post_extraction_review_response(
        _response(_operation("merge", ["gauge_a", "gauge_b"], factor)),
        current_specs=specs,
        available_diagnostic_ids=["diagnostic_0001"],
        available_evidence_ids=["evidence_0001"],
        available_evidence_catalog=_evidence_catalog("combined gauge"),
    )
    applied = apply_post_extraction_review_operations(specs, validated)
    assert [spec["name"] for spec in applied.specs] == ["combined_gauge", "gauge_c"]
    assert set(applied.removed_names) == {"gauge_a", "gauge_b"}
    assert applied.added_names == ("combined_gauge",)

    with pytest.raises(ValueError, match="collides"):
        validate_post_extraction_review_response(
            _response(_operation("merge", ["gauge_a", "gauge_b"], specs[2])),
            current_specs=specs,
            available_diagnostic_ids=["diagnostic_0001"],
            available_evidence_ids=["evidence_0001"],
        )


def test_response_accepts_temporal_wording_and_fails_closed_on_unknown_diagnostic():
    before = _continuous("baseline_measure")
    temporal = _continuous(
        "response_measure",
        description="Process response measured after assignment.",
    )
    validated = validate_post_extraction_review_response(
        _response(_operation("replace", ["baseline_measure"], temporal)),
        current_specs=[before],
        available_diagnostic_ids=["diagnostic_0001"],
        available_evidence_ids=["evidence_0001"],
        available_evidence_catalog=_evidence_catalog("response measure"),
    )
    assert validated.operations[0].contract == temporal

    payload = _response(_operation("drop", ["baseline_measure"]))
    payload["operations"][0]["supporting_diagnostic_ids"] = ["diagnostic_9999"]
    with pytest.raises(ValueError, match="unknown diagnostic"):
        validate_post_extraction_review_response(
            payload,
            current_specs=[before],
            available_diagnostic_ids=["diagnostic_0001"],
            available_evidence_ids=["evidence_0001"],
        )


def test_diagnostic_target_mapping_keeps_nested_ablations_specific():
    diagnostics = [
        {
            "diagnostic_id": "diagnostic_0001",
            "kind": "feature_quality",
            "feature_name": "inlet_valve",
        },
        {
            "diagnostic_id": "diagnostic_0002",
            "kind": "redundancy",
            "feature_names": ["gauge_a", "gauge_b"],
        },
        {
            "diagnostic_id": "diagnostic_0003",
            "kind": "nested_observable_causal_quality",
            "contract_ablations": [
                {
                    "diagnostic_id": "diagnostic_0004",
                    "kind": "contract_ablation",
                    "contract_name": "sensor_reading",
                }
            ],
        },
    ]

    assert collect_post_extraction_diagnostic_targets(diagnostics) == {
        "diagnostic_0001": ("inlet_valve",),
        "diagnostic_0002": ("gauge_a", "gauge_b"),
        "diagnostic_0003": (),
        "diagnostic_0004": ("sensor_reading",),
    }


def test_review_operation_must_cite_a_diagnostic_for_every_target():
    specs = [_continuous("gauge_a"), _continuous("gauge_b")]
    response = _response(_operation("drop", ["gauge_a"]))

    with pytest.raises(ValueError, match="directly names every target"):
        validate_post_extraction_review_response(
            response,
            current_specs=specs,
            available_diagnostic_ids=["diagnostic_0001"],
            available_diagnostic_targets={"diagnostic_0001": ["gauge_b"]},
            available_evidence_ids=["evidence_0001"],
        )


def test_contract_changing_operation_requires_source_evidence():
    before = _categorical("inlet_valve")
    revised = {
        **before,
        "categories": ["negative", "positive", "indeterminate"],
    }
    operation = _operation("revise", ["inlet_valve"], revised)
    operation["supporting_evidence_ids"] = []

    with pytest.raises(ValueError, match="must cite source evidence"):
        validate_post_extraction_review_response(
            _response(operation),
            current_specs=[before],
            available_diagnostic_ids=["diagnostic_0001"],
            available_diagnostic_targets={"diagnostic_0001": ["inlet_valve"]},
            available_evidence_ids=["evidence_0001"],
        )


def test_contract_changing_operation_requires_semantic_evidence_catalog():
    before = _continuous("ambiguous_baseline_measure")
    replacement = _continuous(
        "quartz_width",
        description="Quartz width measurement.",
    )

    with pytest.raises(ValueError, match="requires available_evidence_catalog"):
        validate_post_extraction_review_response(
            _response(_operation("replace", ["ambiguous_baseline_measure"], replacement)),
            current_specs=[before],
            available_diagnostic_ids=["diagnostic_0001"],
            available_diagnostic_targets={"diagnostic_0001": ["ambiguous_baseline_measure"]},
            available_evidence_ids=["evidence_0001"],
        )


def test_target_grounded_merge_can_cite_one_pairwise_redundancy_diagnostic():
    specs = [_continuous("gauge_a"), _continuous("gauge_b")]
    replacement = _continuous("combined_gauge")

    validated = validate_post_extraction_review_response(
        _response(_operation("merge", ["gauge_a", "gauge_b"], replacement)),
        current_specs=specs,
        available_diagnostic_ids=["diagnostic_0001"],
        available_diagnostic_targets={"diagnostic_0001": ["gauge_a", "gauge_b"]},
        available_evidence_ids=["evidence_0001"],
        available_evidence_catalog=_evidence_catalog("combined gauge"),
    )

    assert validated.operations[0].target_names == ("gauge_a", "gauge_b")


def test_target_and_evidence_grounded_replace_is_applied():
    before = _continuous("ambiguous_baseline_measure")
    replacement = _continuous(
        "assembly_ribbon_width",
        description="Assembly ribbon width in millimeters.",
    )

    validated = validate_post_extraction_review_response(
        _response(_operation("replace", ["ambiguous_baseline_measure"], replacement)),
        current_specs=[before],
        available_diagnostic_ids=["diagnostic_0001"],
        available_diagnostic_targets={"diagnostic_0001": ["ambiguous_baseline_measure"]},
        available_evidence_ids=["evidence_0001"],
        available_evidence_catalog=_evidence_catalog("assembly ribbon width"),
    )
    applied = apply_post_extraction_review_operations([before], validated)

    assert applied.removed_names == ("ambiguous_baseline_measure",)
    assert applied.added_names == ("assembly_ribbon_width",)
    assert applied.extraction_changed_names == ("assembly_ribbon_width",)
    assert list(applied.specs) == [replacement]
    grounding = applied.operation_audit[0]["evidence_contract_grounding"][0]
    assert grounding["schema_version"] == EVIDENCE_CONTRACT_GROUNDING_VERSION
    assert grounding["supported"] is True
    assert grounding["matched_evidence_paths"] == ["content.concept"]


def test_replace_rejects_real_but_conceptually_unrelated_evidence():
    before = _continuous("ambiguous_baseline_measure")
    replacement = _continuous(
        "quartz_width",
        description="Quartz width measurement.",
    )

    with pytest.raises(ValueError, match="evidence unrelated to the proposed contract"):
        validate_post_extraction_review_response(
            _response(_operation("replace", ["ambiguous_baseline_measure"], replacement)),
            current_specs=[before],
            available_diagnostic_ids=["diagnostic_0001"],
            available_diagnostic_targets={"diagnostic_0001": ["ambiguous_baseline_measure"]},
            available_evidence_ids=["evidence_0001"],
            available_evidence_catalog=_evidence_catalog("copper density"),
        )


def test_replace_accepts_exact_lexical_anchors_in_evidence():
    before = _continuous("ambiguous_baseline_measure")
    replacement = _continuous(
        "quartz_width",
        description="Quartz width measurement.",
    )

    validated = validate_post_extraction_review_response(
        _response(_operation("replace", ["ambiguous_baseline_measure"], replacement)),
        current_specs=[before],
        available_diagnostic_ids=["diagnostic_0001"],
        available_diagnostic_targets={"diagnostic_0001": ["ambiguous_baseline_measure"]},
        available_evidence_ids=["evidence_0001"],
        available_evidence_catalog=_evidence_catalog("quartz width"),
    )

    assert validated.operations[0].contract == replacement


def test_quality_diagnostics_cover_missingness_plausibility_and_stability_without_timing():
    specs = [_continuous("sensor_reading"), _categorical("inlet_valve")]
    frame = pd.DataFrame(
        {
            "explicit_feat_sensor_reading": [60.0, 61.0, np.nan, 63.0, 64.0, 65.0],
            "explicit_feat_sensor_reading_missing": [False, False, True, False, False, False],
            "explicit_feat_inlet_valve": [
                "absent",
                "present",
                "other",
                "absent",
                "present",
                "absent",
            ],
            "explicit_feat_inlet_valve_missing": [False] * 6,
        }
    )
    result = build_extraction_quality_diagnostics(
        frame,
        specs,
        fold_ids=[1, 1, 1, 2, 2, 2],
        maximum_unknown_category_rate=0.10,
    )
    assert result["summary"]["feature_count"] == 2
    reading, phase = result["features"]
    assert reading["coverage"] == pytest.approx(5 / 6)
    assert "temporal_correctness" not in reading
    assert reading["source_text_temporal_policy"]["temporal_boundary_enforced"] is False
    assert result["source_text_temporal_policy"] == source_text_temporal_policy_audit()
    assert len(reading["inner_fold_stability"]["coverage_by_fold"]) == 2
    assert "out_of_contract_category_values" in phase["hard_failures"]
    assert phase["value_plausibility"]["unknown_category_rate"] == pytest.approx(1 / 6)


def test_redundancy_diagnostics_find_value_and_missingness_duplicates():
    specs = [_continuous("measure_a"), _continuous("measure_b"), _categorical("material_phase")]
    frame = pd.DataFrame(
        {
            "explicit_feat_measure_a": [1.0, 2.0, 3.0, 4.0],
            "explicit_feat_measure_a_missing": [False, False, False, True],
            "explicit_feat_measure_b": [2.0, 4.0, 6.0, 8.0],
            "explicit_feat_measure_b_missing": [False, False, False, True],
            "explicit_feat_material_phase": ["absent", "present", "absent", "present"],
            "explicit_feat_material_phase_missing": [False, False, False, False],
        }
    )
    rows = build_redundancy_diagnostics(frame, specs)
    pair = next(row for row in rows if row["feature_names"] == ["measure_a", "measure_b"])
    assert pair["association"] == pytest.approx(1.0)
    assert pair["missingness_agreement"] == pytest.approx(1.0)
    assert pair["missingness_jaccard"] == pytest.approx(1.0)


def test_redundancy_does_not_treat_shared_complete_observation_as_signal():
    specs = [_continuous("measure_a"), _continuous("measure_b"), _categorical("material_phase")]
    frame = pd.DataFrame(
        {
            "explicit_feat_measure_a": np.arange(1.0, 9.0),
            "explicit_feat_measure_a_missing": [False] * 8,
            "explicit_feat_measure_b": [1.0, -1.0, 2.0, -2.0, 1.5, -1.5, 0.5, -0.5],
            "explicit_feat_measure_b_missing": [False] * 8,
            "explicit_feat_material_phase": ["absent", "present"] * 4,
            "explicit_feat_material_phase_missing": [False] * 8,
        }
    )

    assert build_redundancy_diagnostics(frame, specs) == []


def test_nested_contract_ablation_diagnostic_id_is_citable():
    context = {
        "current_contracts": [_continuous("baseline_measure")],
        "max_operations": 4,
        "diagnostics": [
            {
                "diagnostic_id": "diagnostic_0001",
                "kind": "nested_observable_causal_quality",
                "contract_ablations": [
                    {
                        "diagnostic_id": "diagnostic_0002",
                        "kind": "contract_ablation",
                        "contract_name": "baseline_measure",
                    }
                ],
            }
        ],
        "sanitized_evidence_catalog": [],
    }
    response = _response(_operation("drop", ["baseline_measure"]))
    response["operations"][0]["supporting_diagnostic_ids"] = ["diagnostic_0002"]
    response["operations"][0]["supporting_evidence_ids"] = []

    assert collect_post_extraction_diagnostic_ids(context["diagnostics"]) == (
        "diagnostic_0001",
        "diagnostic_0002",
    )
    assert post_extraction_review_response_issues(response, context) == []


def test_reasoning_agent_prompt_exposes_operations_but_not_acceptance_gate():
    context = {
        "prompt_version": POST_EXTRACTION_REVIEW_PROMPT_VERSION,
        "source_text_temporal_policy": source_text_temporal_policy_audit(),
        "outer_fold": 1,
        "review_round": 1,
        "max_operations": 4,
        "current_contracts": [_continuous("baseline_measure")],
        "diagnostics": [
            {
                "diagnostic_id": "diagnostic_0001",
                "kind": "feature_quality",
                "feature_name": "baseline_measure",
                "coverage": 0.5,
            }
        ],
        "sanitized_evidence_catalog": [
            {"evidence_id": "evidence_0001", "source_families": ["bow_r_loss"]}
        ],
        "acceptance_gate_disclosed_to_agent": False,
    }
    prompt = build_agent_prompt(context, AgenticFeatureSearchConfig())
    assert "drop|merge|re_role|replace|revise|stop" in prompt
    assert "acceptance gate is intentionally absent" in prompt
    assert "true treatment effects" in prompt
    assert "diagnostic_0001" in prompt
    assert "Every operation object always contains exactly these six keys" in prompt
    assert 'drop uses "contract": null and "supporting_evidence_ids": []' in prompt
    assert 'stop is the sole operation and uses "target_names": [], "contract": null' in prompt
    assert "repeat a canonical category as its own alias" in prompt
    assert "required_safety_remediation" in prompt
    assert "sealed candidate workspace" in prompt

    repair = build_post_extraction_review_repair_prompt(["operations[0] is missing neutral fields"])
    assert "exactly these six keys" in repair
    assert "contract:null" in repair
    assert "supporting_evidence_ids:[]" in repair
    assert "globally unique" in repair
