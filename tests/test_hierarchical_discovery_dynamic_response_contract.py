from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

from oci.inference.adaptive_hierarchical_stage1_reconsideration import (
    adaptive_hierarchical_implementation_bundle,
    adaptive_hierarchical_stage1_prompt_contract,
)
from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    OUTCOME_AXIS,
    DiscoveryCandidate,
    DiscoveryEvidenceItem,
    canonical_json,
    content_sha256,
    validate_consolidation_response,
    validate_interpret_evidence_chunk_response,
    validate_rejection_critic_response,
)
from oci.inference.hierarchical_all_architecture_discovery import (
    AUTHENTICATED_MESSAGE_ENVELOPE_BINDING,
    AUTHENTICATED_RESPONSE_CONTRACT_BINDING,
    DISCOVERY_JSON_JOB_VERSION,
    INTERPRET_CHUNK_JOB,
    DiscoveryJobSettings,
    DiscoveryJsonJob,
)
from oci.inference.hierarchical_discovery_job_cache import (
    AuthenticatedHierarchicalDiscoveryJobCache,
)
from tests.hierarchy_resource_test_support import (
    HIERARCHY_JOB_CACHE_CONFIG,
)
from oci.inference.hierarchical_discovery_response_contract import (
    HIERARCHICAL_DISCOVERY_INTERPRET_TOKEN_BUDGET,
    HIERARCHICAL_DISCOVERY_MAX_ATOMS_PER_INTERPRET_JOB,
    HIERARCHICAL_DISCOVERY_MAX_FINDINGS_PER_ATOMIC_REVIEW,
    HIERARCHICAL_DISCOVERY_MAX_INTERPRET_AMBIGUITY_LENGTH,
    HIERARCHICAL_DISCOVERY_MAX_INTERPRET_DESCRIPTION_LENGTH,
    HIERARCHICAL_DISCOVERY_MAX_INTERPRET_NAME_LENGTH,
    HIERARCHICAL_DISCOVERY_MAX_INTERPRET_REASON_LENGTH,
    HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB,
    HIERARCHICAL_DISCOVERY_MAX_PAIR_RELATION_PEERS,
    HIERARCHICAL_DISCOVERY_MAX_TEXT_LENGTH,
    HIERARCHICAL_DISCOVERY_WIRE_RESPONSE_BUDGET_VERSION,
    attach_hierarchical_discovery_response_contract,
    build_hierarchical_discovery_response_contract,
)

FAMILY_A = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
FAMILY_B = ACTIVE_STAGE1_CONCEPT_FAMILIES[1]

_PLACEHOLDER_IDENTIFIER_VALUES = {
    "evidence_id",
    "candidate_id",
    "supplied_member_id",
    "active_source_family",
    "dossier_evidence_id",
    "diagnostic_id",
    "lower_snake_case",
}


def _candidate(
    candidate_id: str,
    evidence_id: str,
    *,
    family: str = FAMILY_A,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "feature_name": f"feature_{candidate_id.replace('.', '_')}",
        "description": f"Description for {candidate_id}.",
        "supporting_evidence_ids": [evidence_id],
        "source_families": [family],
        "value_shape_hypothesis": "continuous",
        "unresolved_ambiguity": "",
    }


def _request_cases() -> tuple[tuple[str, dict[str, Any]], ...]:
    evidence = [
        {"evidence_id": "ev.alpha", "member_ids": ["member.alpha"]},
        {"evidence_id": "ev.beta", "member_ids": ["member.beta"]},
    ]
    candidates = [
        _candidate("candidate.alpha", "ev.alpha"),
        _candidate("candidate.beta", "ev.beta"),
    ]
    consolidation = {
        "canonical_concepts": [
            {
                "canonical_name": "marker_alpha",
                "member_candidate_ids": ["candidate.alpha"],
            },
            {
                "canonical_name": "marker_beta",
                "member_candidate_ids": ["candidate.beta"],
            },
        ],
        "candidate_dispositions": [],
    }
    base_dossiers = [
        {
            "source_family": FAMILY_A,
            "architecture_candidates": candidates,
        }
    ]
    adaptive_dossiers = [
        {
            "source_family": FAMILY_A,
            "coverage": {"lookback_evidence_ids": ["ev.alpha"]},
            "architecture_candidates": [candidates[0]],
        },
        {
            "source_family": FAMILY_B,
            "coverage": {"lookback_evidence_ids": ["ev.beta"]},
            "architecture_candidates": [_candidate("candidate.beta", "ev.beta", family=FAMILY_B)],
        },
    ]
    registry = [{"feature_name": "age"}]
    diagnostics = [{"diagnostic_id": "diag.one"}]
    return (
        (
            "interpret_architecture_chunk",
            {"job": "interpret_evidence_chunk", "evidence": evidence},
        ),
        (
            "consolidate_architecture_candidates",
            {
                "job": "consolidate_candidate_ledger",
                "source_family": FAMILY_A,
                "candidates": candidates,
            },
        ),
        (
            "consolidate_architecture_candidates",
            {
                "job": "consolidate_adaptive_architecture_candidates",
                "source_family": FAMILY_A,
                "candidates": candidates,
            },
        ),
        (
            "audit_architecture_coverage",
            {
                "job": "audit_architecture_chunk_coverage",
                "evidence": evidence,
                "consolidation": consolidation,
            },
        ),
        (
            "audit_architecture_coverage",
            {
                "job": "audit_adaptive_architecture_coverage",
                "evidence": evidence,
                "family_consolidation": consolidation,
            },
        ),
        (
            "plan_cross_architecture_integration",
            {
                "job": "plan_cross_architecture_integration",
                "architecture_dossiers": base_dossiers,
                "maximum_raw_evidence_lookback_ids": 4,
            },
        ),
        (
            "integrate_cross_architecture_candidates",
            {
                "job": "integrate_cross_architecture_candidates",
                "architecture_context": {"architecture_dossiers": base_dossiers},
                "maximum_integrated_features": 16,
            },
        ),
        (
            "audit_rejected_candidates",
            {
                "job": "audit_every_rejected_candidate",
                "rejected_candidates": candidates,
            },
        ),
        (
            "define_one_extraction_feature",
            {
                "job": "define_one_extraction_feature",
                "canonical_name": "age",
                "value_shape_hypothesis": "ambiguous",
                "supporting_evidence_ids": ["ev.alpha"],
            },
        ),
        (
            "plan_cross_architecture_integration",
            {
                "job": "plan_adaptive_stage1_reconsideration",
                "architecture_dossiers": adaptive_dossiers,
                "current_registry": registry,
                "diagnostics": diagnostics,
                "lookback_bounds": {
                    "max_ids_per_target": 8,
                    "max_total_ids": 24,
                    "max_total_bytes": 96_000,
                },
            },
        ),
        (
            "integrate_cross_architecture_candidates",
            {
                "job": "propose_adaptive_registry_revision",
                "architecture_dossiers": adaptive_dossiers,
                "current_registry": registry,
                "diagnostics": diagnostics,
                "review_plan": {
                    "review_targets": [
                        {
                            "target": "age",
                            "relevant_architectures": [FAMILY_A],
                            "requested_evidence_ids": ["ev.alpha"],
                        }
                    ]
                },
                "requested_evidence": [{"evidence_id": "ev.alpha", "source_family": FAMILY_A}],
                "maximum_operations": 4,
            },
        ),
    )


def test_bounded_relation_page_has_an_exact_common_wire_budget_and_peer_cap() -> None:
    peers = [f"candidate.peer.{index}" for index in range(7)]
    attached = attach_hierarchical_discovery_response_contract(
        job_kind="consolidate_architecture_candidates",
        request={
            "job": "compare_consolidation_candidate_relations",
            "anchor_candidate_id": "candidate.anchor",
            "peer_candidate_ids": peers,
        },
    )
    budget = attached["identifier_ownership"]["ownership"]["wire_response_budget"]
    assert budget["budget_contract_version"] == (
        HIERARCHICAL_DISCOVERY_WIRE_RESPONSE_BUDGET_VERSION
    )
    assert budget["maximum_canonical_json_bytes"] < 20_000
    assert budget["maximum_estimated_tokens"] == budget["maximum_canonical_json_bytes"]
    assert attached["output_schema"]["properties"]["comparisons"]["required"] == peers


def test_lossless_raw_evidence_page_and_fold_contracts_stay_under_wire_budget() -> None:
    review_ids = [f"review.{index}" for index in range(8)]
    cases = (
        (
            "integrate_cross_architecture_candidates",
            {
                "job": "review_integration_group_evidence",
                "group_id": "group.alpha",
                "evidence_id": "ev.alpha",
            },
        ),
        (
            "integrate_cross_architecture_candidates",
            {
                "job": "fold_integration_group_evidence_reviews",
                "group_id": "group.alpha",
                "review_input_ids": review_ids,
            },
        ),
        (
            "audit_rejected_candidates",
            {
                "job": "review_rejection_candidate_evidence",
                "candidate_id": "candidate.alpha",
                "evidence_id": "ev.alpha",
            },
        ),
        (
            "audit_rejected_candidates",
            {
                "job": "fold_rejection_candidate_evidence_reviews",
                "candidate_id": "candidate.alpha",
                "review_input_ids": review_ids,
            },
        ),
        (
            "define_one_extraction_feature",
            {
                "job": "review_extraction_feature_evidence",
                "canonical_name": "marker_alpha",
                "evidence_id": "ev.alpha",
            },
        ),
        (
            "define_one_extraction_feature",
            {
                "job": "fold_extraction_evidence_definitions",
                "canonical_name": "marker_alpha",
                "value_shape_hypothesis": "ambiguous",
                "review_input_ids": review_ids,
            },
        ),
    )
    for job_kind, request in cases:
        attached = attach_hierarchical_discovery_response_contract(
            job_kind=job_kind,
            request=request,
        )
        Draft202012Validator.check_schema(attached["output_schema"])
        ownership = attached["identifier_ownership"]["ownership"]
        budget = ownership["wire_response_budget"]
        assert budget["maximum_canonical_json_bytes"] <= 20_000
        assert budget["maximum_transport_bytes"] == 20_000


def test_lossless_fold_requires_one_disposition_for_every_authenticated_input() -> None:
    review_ids = [f"review.{index}" for index in range(8)]
    attached = attach_hierarchical_discovery_response_contract(
        job_kind="define_one_extraction_feature",
        request={
            "job": "fold_extraction_evidence_definitions",
            "canonical_name": "marker_alpha",
            "value_shape_hypothesis": "ambiguous",
            "review_input_ids": review_ids,
        },
    )
    response = {
        "feature_name": "marker_alpha",
        "measurement": "The documented marker measurement.",
        "representation": {"kind": "unresolved", "unit": "", "categories": []},
        "aliases": [],
        "distinguish_from": [],
        "missing_or_ambiguous": "The evidence does not establish a stable scale.",
        "input_dispositions": {
            review_id: {
                "action": "integrated",
                "reason": "The observation was incorporated into the accumulator.",
            }
            for review_id in review_ids
        },
        "supporting_evidence_reviewed": True,
    }
    validator = Draft202012Validator(attached["output_schema"])
    validator.validate(response)
    incomplete = json.loads(canonical_json(response))
    incomplete["input_dispositions"].pop(review_ids[-1])
    with pytest.raises(ValidationError):
        validator.validate(incomplete)

    with pytest.raises(ValueError, match="peer bound"):
        attach_hierarchical_discovery_response_contract(
            job_kind="consolidate_architecture_candidates",
            request={
                "job": "compare_consolidation_candidate_relations",
                "anchor_candidate_id": "candidate.anchor",
                "peer_candidate_ids": [
                    f"candidate.peer.{index}"
                    for index in range(HIERARCHICAL_DISCOVERY_MAX_PAIR_RELATION_PEERS + 1)
                ],
            },
        )


def _scalar_values(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _scalar_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _scalar_values(child)
    else:
        yield value


def _schema_literal_values(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"enum", "const"}:
                yield from _scalar_values(child)
            else:
                yield from _schema_literal_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _schema_literal_values(child)


def _assert_no_unique_items(value: Any) -> None:
    if isinstance(value, dict):
        assert "uniqueItems" not in value
        for child in value.values():
            _assert_no_unique_items(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_unique_items(child)


def _schema_nodes(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _schema_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _schema_nodes(child)


@pytest.mark.parametrize(("job_kind", "request_payload"), _request_cases())
def test_every_base_and_adaptive_job_has_a_strict_dynamic_placeholder_free_contract(
    job_kind: str,
    request_payload: dict[str, Any],
) -> None:
    attached = attach_hierarchical_discovery_response_contract(
        job_kind=job_kind,
        request=request_payload,
    )
    schema, ownership = build_hierarchical_discovery_response_contract(
        job_kind=job_kind,
        request=attached,
    )

    assert attached["output_schema"] == schema
    assert attached["identifier_ownership"] == ownership
    assert schema.get("additionalProperties") is False
    Draft202012Validator.check_schema(schema)
    _assert_no_unique_items(schema)
    schema_literals = set(_schema_literal_values(schema))
    domain_values = {
        value for values in ownership["identifier_domains"].values() for value in values
    }
    assert schema_literals.isdisjoint(_PLACEHOLDER_IDENTIFIER_VALUES)
    assert domain_values.isdisjoint(_PLACEHOLDER_IDENTIFIER_VALUES)


@pytest.mark.parametrize(("job_kind", "request_payload"), _request_cases())
def test_every_dynamic_contract_bounds_generated_strings_and_arrays(
    job_kind: str,
    request_payload: dict[str, Any],
) -> None:
    schema = attach_hierarchical_discovery_response_contract(
        job_kind=job_kind,
        request=request_payload,
    )["output_schema"]
    Draft202012Validator.check_schema(schema)
    _assert_no_unique_items(schema)

    for node in _schema_nodes(schema):
        if node.get("type") == "string" and not ({"enum", "const"} & set(node)):
            assert 0 < node["maxLength"] <= HIERARCHICAL_DISCOVERY_MAX_TEXT_LENGTH
        if node.get("type") == "array" and "const" not in node:
            assert node["maxItems"] >= node.get("minItems", 0)


def test_every_free_string_pattern_uses_absolute_end_not_final_lf_semantics() -> None:
    patterns: set[str] = set()
    for job_kind, request_payload in _request_cases():
        schema = attach_hierarchical_discovery_response_contract(
            job_kind=job_kind,
            request=request_payload,
        )["output_schema"]
        patterns.update(
            str(node["pattern"])
            for node in _schema_nodes(schema)
            if node.get("type") == "string" and "pattern" in node
        )
    assert patterns
    assert all(re.search(pattern, "a\n") is None for pattern in patterns)


def test_exact_domains_and_ownership_maps_are_derived_only_from_designated_fields() -> None:
    interpret = attach_hierarchical_discovery_response_contract(
        job_kind="interpret_architecture_chunk",
        request=_request_cases()[0][1],
    )["identifier_ownership"]
    assert interpret["identifier_domains"] == {
        "evidence_ids": ["ev.alpha", "ev.beta"],
        "member_ids": ["member.alpha", "member.beta"],
    }
    assert interpret["ownership"]["member_ids_by_evidence_id"] == {
        "ev.alpha": ["member.alpha"],
        "ev.beta": ["member.beta"],
    }
    budget = interpret["ownership"]["wire_response_budget"]
    assert interpret["ownership"]["response_domain_bounds"]["atomic_review_count"] == 4
    assert budget["maximum_estimated_tokens"] < budget["generation_token_budget"]

    integration = attach_hierarchical_discovery_response_contract(
        job_kind="integrate_cross_architecture_candidates",
        request=_request_cases()[6][1],
    )["identifier_ownership"]
    integration_relations = {
        key: value
        for key, value in integration["ownership"].items()
        if key != "wire_response_budget"
    }
    assert integration_relations == {
        "evidence_ids_by_candidate_id": {
            "candidate.alpha": ["ev.alpha"],
            "candidate.beta": ["ev.beta"],
        },
        "source_families_by_candidate_id": {
            "candidate.alpha": [FAMILY_A],
            "candidate.beta": [FAMILY_A],
        },
        "maximum_integrated_features": 16,
        "integration_slots_are_compiler_owned": True,
        "accepted_feature_relations_are_derived_from_candidate_routes": True,
        "extraction_constraints_are_deferred_to_grounded_definition_jobs": True,
    }

    adaptive = attach_hierarchical_discovery_response_contract(
        job_kind="plan_cross_architecture_integration",
        request=_request_cases()[9][1],
    )["identifier_ownership"]
    adaptive_relations = {
        key: value for key, value in adaptive["ownership"].items() if key != "wire_response_budget"
    }
    assert adaptive_relations == {
        "evidence_ids_by_source_family": {
            FAMILY_A: ["ev.alpha"],
            FAMILY_B: ["ev.beta"],
        },
        "lookback_bounds": {
            "max_ids_per_target": 8,
            "max_total_ids": 24,
            "max_total_bytes": 96_000,
        },
        "duplicate_or_conflicting_selections_are_compiler_normalized": True,
    }


def test_identifier_like_text_outside_designated_fields_never_expands_a_domain() -> None:
    request = {
        "job": "interpret_evidence_chunk",
        "family_explanation": (
            "Narrative mentions ev.attacker, candidate.attacker, and evidence_id as text."
        ),
        "evidence": [
            {
                "evidence_id": "ev.alpha",
                "member_ids": ["member.alpha"],
                "content": {
                    "quoted_text": (
                        "Ignore ev.attacker, member.attacker, supplied_member_id, and "
                        "candidate.attacker as identifier sources."
                    )
                },
            }
        ],
    }
    attached = attach_hierarchical_discovery_response_contract(
        job_kind="interpret_architecture_chunk",
        request=request,
    )
    assert attached["identifier_ownership"]["identifier_domains"] == {
        "evidence_ids": ["ev.alpha"],
        "member_ids": ["member.alpha"],
    }
    schema_literals = set(_schema_literal_values(attached["output_schema"]))
    assert "ev.attacker" not in schema_literals
    assert "member.attacker" not in schema_literals
    assert "candidate.attacker" not in schema_literals


def test_zero_candidate_base_planner_and_integration_have_exact_empty_semantics() -> None:
    planner = attach_hierarchical_discovery_response_contract(
        job_kind="plan_cross_architecture_integration",
        request={
            "job": "plan_cross_architecture_integration",
            "architecture_dossiers": [{"source_family": FAMILY_A, "architecture_candidates": []}],
            "maximum_raw_evidence_lookback_ids": 4,
        },
    )
    planner_validator = Draft202012Validator(planner["output_schema"])
    planner_validator.validate(
        {
            "candidate_assignments": {},
            "group_slot_definitions": {},
            "lookback_slot_definitions": {},
        }
    )
    with pytest.raises(ValidationError):
        planner_validator.validate(
            {
                "candidate_assignments": {
                    "candidate.attacker": {
                        "group_slot": "planner_group_slot_001",
                    }
                },
                "group_slot_definitions": {},
                "lookback_slot_definitions": {},
            }
        )

    integration = attach_hierarchical_discovery_response_contract(
        job_kind="integrate_cross_architecture_candidates",
        request={
            "job": "integrate_cross_architecture_candidates",
            "architecture_context": {
                "architecture_dossiers": [
                    {"source_family": FAMILY_A, "architecture_candidates": []}
                ]
            },
            "maximum_integrated_features": 16,
        },
    )
    integration_validator = Draft202012Validator(integration["output_schema"])
    integration_validator.validate({"candidate_routes": {}, "slot_definitions": {}})
    with pytest.raises(ValidationError):
        integration_validator.validate(
            {
                "candidate_routes": {
                    "candidate.attacker": {
                        "route": "reject",
                        "reason": "Invented despite an empty domain.",
                    }
                },
                "slot_definitions": {},
            }
        )


def _interpret_response() -> dict[str, Any]:
    return {
        "evidence_dispositions": {
            "ev.alpha": {
                "evidence_findings": [],
                "member_dispositions": {"member.alpha": {"findings": []}},
                "reason": "Reviewed alpha.",
            },
            "ev.beta": {
                "evidence_findings": [],
                "member_dispositions": {"member.beta": {"findings": []}},
                "reason": "Reviewed beta.",
            },
        },
    }


def test_strict_schema_rejects_copied_placeholders_and_unsupplied_identifiers() -> None:
    attached = attach_hierarchical_discovery_response_contract(
        job_kind="interpret_architecture_chunk",
        request=_request_cases()[0][1],
    )
    validator = Draft202012Validator(attached["output_schema"])
    valid = _interpret_response()
    validator.validate(valid)

    copied_placeholder = json.loads(canonical_json(valid))
    copied_placeholder["evidence_dispositions"]["evidence_id"] = copied_placeholder[
        "evidence_dispositions"
    ].pop("ev.alpha")
    with pytest.raises(ValidationError):
        validator.validate(copied_placeholder)

    unsupplied_member = json.loads(canonical_json(valid))
    members = unsupplied_member["evidence_dispositions"]["ev.alpha"]["member_dispositions"]
    members["member.unsupplied"] = members.pop("member.alpha")
    with pytest.raises(ValidationError):
        validator.validate(unsupplied_member)

    extraction = attach_hierarchical_discovery_response_contract(
        job_kind="define_one_extraction_feature",
        request=_request_cases()[8][1],
    )
    extraction_response = {
        "feature_name": "age",
        "measurement": "Extract age.",
        "representation": {"kind": "unresolved", "unit": "", "categories": []},
        "aliases": [],
        "distinguish_from": [],
        "missing_or_ambiguous": "Return null when unresolved.",
        "supporting_evidence_reviewed": True,
    }
    extraction_validator = Draft202012Validator(extraction["output_schema"])
    extraction_validator.validate(extraction_response)
    extraction_response["feature_name"] = "lower_snake_case"
    with pytest.raises(ValidationError):
        extraction_validator.validate(extraction_response)


def test_adaptive_schema_rejects_unsupplied_target_evidence_family_and_diagnostic() -> None:
    request = _request_cases()[10][1]
    attached = attach_hierarchical_discovery_response_contract(
        job_kind="integrate_cross_architecture_candidates",
        request=request,
    )
    validator = Draft202012Validator(attached["output_schema"])
    valid = {
        "operations": [
            {
                "operation": "revise_definition",
                "targets": ["age"],
                "proposed_feature": {
                    "feature_name": "age",
                    "description": "Revised age definition.",
                    "value_shape_hypothesis": "continuous",
                    "definition_summary": "Use the supported age statement.",
                    "source_families": [FAMILY_A],
                },
                "supporting_evidence_ids": ["ev.alpha"],
                "diagnostic_ids": ["diag.one"],
                "reason": "The current definition is incomplete.",
            }
        ],
        "converged": False,
    }
    validator.validate(valid)
    mutations = (
        ("targets", "age_unsupplied"),
        ("supporting_evidence_ids", "ev.unsupplied"),
        ("diagnostic_ids", "diag.unsupplied"),
    )
    for field, replacement in mutations:
        mutated = json.loads(canonical_json(valid))
        mutated["operations"][0][field] = [replacement]
        with pytest.raises(ValidationError):
            validator.validate(mutated)
    mutated = json.loads(canonical_json(valid))
    mutated["operations"][0]["proposed_feature"]["source_families"] = ["family.unsupplied"]
    with pytest.raises(ValidationError):
        validator.validate(mutated)


def test_action_specific_coverage_and_rejection_wires_are_structurally_consistent() -> None:
    coverage = attach_hierarchical_discovery_response_contract(
        job_kind="audit_architecture_coverage",
        request=_request_cases()[3][1],
    )
    coverage_validator = Draft202012Validator(coverage["output_schema"])
    invalid_addition = {
        "findings": [
            {
                "action": "add_concept",
                "affected_canonical_names": [],
                "proposed_name": "",
                "description": "",
                "supporting_evidence_ids": ["ev.alpha"],
                "reason": "The atom may contain an omitted concept.",
            }
        ],
        "reviewed_evidence_ids": {"ev.alpha": True, "ev.beta": True},
    }
    with pytest.raises(ValidationError):
        coverage_validator.validate(invalid_addition)

    rejection = attach_hierarchical_discovery_response_contract(
        job_kind="audit_rejected_candidates",
        request=_request_cases()[7][1],
    )
    rejection_validator = Draft202012Validator(rejection["output_schema"])
    invalid_uphold = {
        "reconsiderations": {
            "candidate.alpha": {
                "decision": "uphold",
                "proposed_name": "invented_name",
                "supporting_evidence_ids": [],
                "reason": "The rejection remains supported.",
            },
            "candidate.beta": {
                "decision": "uphold",
                "proposed_name": "",
                "supporting_evidence_ids": [],
                "reason": "The rejection remains supported.",
            },
        }
    }
    with pytest.raises(ValidationError):
        rejection_validator.validate(invalid_uphold)


@pytest.mark.parametrize(
    ("value_shape", "representation"),
    [
        (
            "continuous",
            {"kind": "categorical", "unit": "", "categories": ["no", "yes"]},
        ),
        (
            "categorical",
            {"kind": "continuous", "unit": "as_documented", "categories": []},
        ),
        (
            "categorical",
            {"kind": "categorical", "unit": "", "categories": ["yes"]},
        ),
        (
            "ambiguous",
            {"kind": "unresolved", "unit": "as_documented", "categories": []},
        ),
    ],
)
def test_extraction_representation_shape_is_exact_by_construction(
    value_shape: str,
    representation: dict[str, Any],
) -> None:
    attached = attach_hierarchical_discovery_response_contract(
        job_kind="define_one_extraction_feature",
        request={
            "job": "define_one_extraction_feature",
            "canonical_name": "age",
            "value_shape_hypothesis": value_shape,
            "supporting_evidence_ids": ["ev.alpha"],
        },
    )
    response = {
        "feature_name": "age",
        "measurement": "Extract the supported age measurement.",
        "representation": representation,
        "aliases": [],
        "distinguish_from": [],
        "missing_or_ambiguous": "Return null when absent or ambiguous.",
        "supporting_evidence_reviewed": True,
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(attached["output_schema"]).validate(response)


def _semantic_evidence() -> tuple[DiscoveryEvidenceItem, ...]:
    return (
        DiscoveryEvidenceItem(
            evidence_id="ev.alpha",
            source_family=FAMILY_A,
            observable_axes=(OUTCOME_AXIS,),
            member_ids=("member.alpha",),
            content={"phrase": "alpha marker"},
        ),
        DiscoveryEvidenceItem(
            evidence_id="ev.beta",
            source_family=FAMILY_A,
            observable_axes=(OUTCOME_AXIS,),
            member_ids=("member.beta",),
            content={"phrase": "beta marker"},
        ),
    )


def test_cross_owner_member_ids_fail_exact_owner_schema_and_relational_validator() -> None:
    evidence = _semantic_evidence()
    request = {
        "job": "interpret_evidence_chunk",
        "evidence": [item.as_prompt_item() for item in evidence],
    }
    schema = attach_hierarchical_discovery_response_contract(
        job_kind="interpret_architecture_chunk",
        request=request,
    )["output_schema"]
    response = _interpret_response()
    alpha_members = response["evidence_dispositions"]["ev.alpha"]["member_dispositions"]
    beta_members = response["evidence_dispositions"]["ev.beta"]["member_dispositions"]
    alpha_members["member.beta"] = alpha_members.pop("member.alpha")
    beta_members["member.alpha"] = beta_members.pop("member.beta")
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(response)

    with pytest.raises(ValueError, match="keys differ"):
        validate_interpret_evidence_chunk_response(response, evidence=evidence)


def test_interpret_schema_uses_exact_empty_member_dispositions_for_memberless_evidence() -> None:
    schema = attach_hierarchical_discovery_response_contract(
        job_kind="interpret_architecture_chunk",
        request={
            "job": "interpret_evidence_chunk",
            "evidence": [{"evidence_id": "ev.empty", "member_ids": []}],
        },
    )["output_schema"]
    validator = Draft202012Validator(schema)
    valid = {
        "evidence_dispositions": {
            "ev.empty": {
                "evidence_findings": [],
                "member_dispositions": {},
                "reason": "Reviewed.",
            }
        },
    }
    validator.validate(valid)
    invalid = json.loads(canonical_json(valid))
    invalid["evidence_dispositions"]["ev.empty"]["member_dispositions"] = {
        "member.invented": {"findings": []}
    }
    with pytest.raises(ValidationError):
        validator.validate(invalid)

    supporting = {
        "evidence_dispositions": {
            "ev.empty": {
                "evidence_findings": [
                    {
                        "feature_name": "evidence_level_marker",
                        "description": "An evidence-level marker.",
                        "value_shape_hypothesis": "continuous",
                        "unresolved_ambiguity": "",
                    }
                ],
                "member_dispositions": {},
                "reason": "The evidence itself supports the concept.",
            }
        },
    }
    validator.validate(supporting)
    semantic_evidence = (
        DiscoveryEvidenceItem(
            evidence_id="ev.empty",
            source_family=FAMILY_A,
            observable_axes=(OUTCOME_AXIS,),
            member_ids=(),
            content={"phrase": "evidence-level marker"},
        ),
    )
    normalized = validate_interpret_evidence_chunk_response(
        supporting,
        evidence=semantic_evidence,
    )
    assert normalized["evidence_dispositions"] == [
        {
            "evidence_id": "ev.empty",
            "status": "supports_concept",
            "feature_names": ["evidence_level_marker"],
            "member_dispositions": [],
            "reason": "The evidence itself supports the concept.",
        }
    ]


def test_worst_case_interpret_contract_has_linear_member_rows_and_capped_name_fanout() -> None:
    owned_members = [
        f"member.{index:02d}"
        for index in range(HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB)
    ]
    evidence = [
        {"evidence_id": "ev.owned.00", "member_ids": owned_members[:2]},
        {"evidence_id": "ev.owned.01", "member_ids": owned_members[2:]},
    ]
    schema = attach_hierarchical_discovery_response_contract(
        job_kind="interpret_architecture_chunk",
        request={"job": "interpret_evidence_chunk", "evidence": evidence},
    )["output_schema"]
    Draft202012Validator.check_schema(schema)
    _assert_no_unique_items(schema)

    dispositions = schema["properties"]["evidence_dispositions"]
    assert dispositions["required"] == [row["evidence_id"] for row in evidence]
    by_evidence_id = dispositions["properties"]
    assert len(by_evidence_id) == HIERARCHICAL_DISCOVERY_MAX_ATOMS_PER_INTERPRET_JOB
    bounded_member_rows = sum(
        len(row["properties"]["member_dispositions"]["properties"])
        for row in by_evidence_id.values()
    )
    bounded_member_finding_slots = sum(
        member_schema["properties"]["findings"]["maxItems"]
        for row in by_evidence_id.values()
        for member_schema in row["properties"]["member_dispositions"]["properties"].values()
    )
    assert bounded_member_rows == len(owned_members)
    assert bounded_member_finding_slots == (
        len(owned_members) * HIERARCHICAL_DISCOVERY_MAX_FINDINGS_PER_ATOMIC_REVIEW
    )
    for properties in (row["properties"] for row in by_evidence_id.values()):
        assert (
            properties["evidence_findings"]["maxItems"]
            == HIERARCHICAL_DISCOVERY_MAX_FINDINGS_PER_ATOMIC_REVIEW
        )
        assert (
            properties["evidence_findings"]["items"]["properties"]["feature_name"]["maxLength"]
            == HIERARCHICAL_DISCOVERY_MAX_INTERPRET_NAME_LENGTH
        )
    ownership = attach_hierarchical_discovery_response_contract(
        job_kind="interpret_architecture_chunk",
        request={"job": "interpret_evidence_chunk", "evidence": evidence},
    )["identifier_ownership"]
    budget = ownership["ownership"]["wire_response_budget"]
    assert ownership["ownership"]["response_domain_bounds"]["maximum_findings"] == (
        (
            HIERARCHICAL_DISCOVERY_MAX_ATOMS_PER_INTERPRET_JOB
            + HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB
        )
        * HIERARCHICAL_DISCOVERY_MAX_FINDINGS_PER_ATOMIC_REVIEW
    )
    assert budget["maximum_estimated_tokens"] < HIERARCHICAL_DISCOVERY_INTERPRET_TOKEN_BUDGET


def _maximum_interpret_wire_fixture(text_character: str) -> tuple[dict[str, Any], dict[str, Any]]:
    members = [
        f"member.{index:02d}"
        for index in range(HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB)
    ]
    evidence = [
        {"evidence_id": "ev.max.00", "member_ids": members[:2]},
        {"evidence_id": "ev.max.01", "member_ids": members[2:]},
    ]
    attached = attach_hierarchical_discovery_response_contract(
        job_kind="interpret_architecture_chunk",
        request={"job": "interpret_evidence_chunk", "evidence": evidence},
    )
    finding = {
        "feature_name": "f" * HIERARCHICAL_DISCOVERY_MAX_INTERPRET_NAME_LENGTH,
        "description": (text_character * HIERARCHICAL_DISCOVERY_MAX_INTERPRET_DESCRIPTION_LENGTH),
        "value_shape_hypothesis": "categorical",
        "unresolved_ambiguity": (
            text_character * HIERARCHICAL_DISCOVERY_MAX_INTERPRET_AMBIGUITY_LENGTH
        ),
    }
    findings = [dict(finding) for _ in range(HIERARCHICAL_DISCOVERY_MAX_FINDINGS_PER_ATOMIC_REVIEW)]
    response = {
        "evidence_dispositions": {
            row["evidence_id"]: {
                "evidence_findings": list(findings),
                "member_dispositions": {
                    member_id: {"findings": list(findings)} for member_id in row["member_ids"]
                },
                "reason": (text_character * HIERARCHICAL_DISCOVERY_MAX_INTERPRET_REASON_LENGTH),
            }
            for row in evidence
        }
    }
    return attached, response


def test_interpret_utf8_budget_is_a_true_maximum_for_astral_and_json_escape_text() -> None:
    attached, astral_response = _maximum_interpret_wire_fixture("\U0010ffff")
    validator = Draft202012Validator(attached["output_schema"])
    validator.validate(astral_response)
    budget = attached["identifier_ownership"]["ownership"]["wire_response_budget"]
    actual_astral_bytes = len(canonical_json(astral_response).encode("utf-8"))
    assert actual_astral_bytes == budget["maximum_canonical_json_bytes"]
    assert budget["maximum_estimated_tokens"] == budget["maximum_canonical_json_bytes"]
    assert budget["maximum_transport_bytes"] == HIERARCHICAL_DISCOVERY_INTERPRET_TOKEN_BUDGET
    assert budget["maximum_estimated_tokens"] < budget["generation_token_budget"]

    for character in ('"', "\\"):
        _, escaped_response = _maximum_interpret_wire_fixture(character)
        validator.validate(escaped_response)
        assert (
            len(canonical_json(escaped_response).encode("utf-8"))
            <= budget["maximum_canonical_json_bytes"]
        )


@pytest.mark.parametrize(
    "unsafe_character",
    ["\x00", "\n", "\x1f", "\x7f", "\x9f", "\ud800", "\udfff"],
)
def test_interpret_schema_and_semantic_validator_reject_controls_and_surrogates(
    unsafe_character: str,
) -> None:
    attached, response = _maximum_interpret_wire_fixture("x")
    first = next(iter(response["evidence_dispositions"].values()))
    first["evidence_findings"][0]["description"] = unsafe_character
    with pytest.raises(ValidationError):
        Draft202012Validator(attached["output_schema"]).validate(response)

    evidence = tuple(
        DiscoveryEvidenceItem(
            evidence_id=row["evidence_id"],
            source_family=FAMILY_A,
            observable_axes=(OUTCOME_AXIS,),
            member_ids=tuple(row["member_ids"]),
            content={"phrase": "safe fixture"},
        )
        for row in [
            {"evidence_id": "ev.max.00", "member_ids": ["member.00", "member.01"]},
            {"evidence_id": "ev.max.01", "member_ids": ["member.02"]},
        ]
    )
    with pytest.raises(ValueError, match="forbidden control or surrogate"):
        validate_interpret_evidence_chunk_response(response, evidence=evidence)


def test_exact_consolidation_evidence_is_derived_and_rejection_support_is_owner_bounded() -> None:
    candidates = (
        DiscoveryCandidate(
            candidate_id="candidate.alpha",
            feature_name="alpha_marker",
            description="Alpha marker.",
            supporting_evidence_ids=("ev.alpha",),
            source_families=(FAMILY_A,),
            value_shape_hypothesis="continuous",
        ),
        DiscoveryCandidate(
            candidate_id="candidate.beta",
            feature_name="beta_marker",
            description="Beta marker.",
            supporting_evidence_ids=("ev.beta",),
            source_families=(FAMILY_A,),
            value_shape_hypothesis="continuous",
        ),
    )
    request = {
        "job": "consolidate_candidate_ledger",
        "source_family": FAMILY_A,
        "candidates": [item.as_prompt_item() for item in candidates],
    }
    schema = attach_hierarchical_discovery_response_contract(
        job_kind="consolidate_architecture_candidates",
        request=request,
    )["output_schema"]
    response = {
        "candidate_assignments": {
            "candidate.alpha": {
                "cluster_slot": "consolidation_slot_001",
                "reason": "Kept distinct.",
            },
            "candidate.beta": {
                "cluster_slot": "consolidation_slot_002",
                "reason": "Kept distinct.",
            },
        },
        "slot_definitions": {
            "consolidation_slot_001": {
                "canonical_name": "alpha_marker",
                "description": "Alpha marker.",
                "unresolved_ambiguity": "",
            },
            "consolidation_slot_002": {
                "canonical_name": "beta_marker",
                "description": "Beta marker.",
                "unresolved_ambiguity": "",
            },
        },
    }
    Draft202012Validator(schema).validate(response)
    normalized = validate_consolidation_response(
        response,
        source_family=FAMILY_A,
        candidates=candidates,
    )
    assert [row["supporting_evidence_ids"] for row in normalized["canonical_concepts"]] == [
        ["ev.alpha"],
        ["ev.beta"],
    ]

    rejection_request = {
        "job": "audit_every_rejected_candidate",
        "rejected_candidates": [item.as_prompt_item() for item in candidates],
    }
    rejection_schema = attach_hierarchical_discovery_response_contract(
        job_kind="audit_rejected_candidates",
        request=rejection_request,
    )["output_schema"]
    rejection_response = {
        "reconsiderations": {
            "candidate.alpha": {
                "decision": "restore",
                "proposed_name": "alpha_marker",
                "supporting_evidence_ids": ["ev.beta"],
                "reason": "Restore.",
            },
            "candidate.beta": {
                "decision": "restore",
                "proposed_name": "beta_marker",
                "supporting_evidence_ids": ["ev.alpha"],
                "reason": "Restore.",
            },
        }
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(rejection_schema).validate(rejection_response)
    with pytest.raises(ValueError, match="another candidate"):
        validate_rejection_critic_response(
            rejection_response,
            rejected_candidate_evidence={
                "candidate.alpha": ("ev.alpha",),
                "candidate.beta": ("ev.beta",),
            },
        )


@pytest.mark.parametrize(
    "job_kind,request_payload,match",
    [
        (
            "interpret_architecture_chunk",
            {
                "job": "interpret_evidence_chunk",
                "evidence": [
                    {"evidence_id": "ev.same", "member_ids": []},
                    {"evidence_id": "ev.same", "member_ids": []},
                ],
            },
            "duplicate designated identifiers",
        ),
        (
            "consolidate_architecture_candidates",
            {
                "job": "consolidate_candidate_ledger",
                "source_family": FAMILY_A,
                "candidates": [
                    _candidate("candidate.same", "ev.alpha"),
                    _candidate("candidate.same", "ev.beta"),
                ],
            },
            "duplicate designated identifiers",
        ),
        (
            "plan_cross_architecture_integration",
            {
                "job": "plan_adaptive_stage1_reconsideration",
                "architecture_dossiers": [
                    {
                        "source_family": FAMILY_A,
                        "coverage": {"lookback_evidence_ids": ["ev.alpha"]},
                    },
                    {
                        "source_family": FAMILY_A,
                        "coverage": {"lookback_evidence_ids": ["ev.beta"]},
                    },
                ],
                "current_registry": [],
                "diagnostics": [],
                "lookback_bounds": {
                    "max_ids_per_target": 8,
                    "max_total_ids": 24,
                    "max_total_bytes": 96_000,
                },
            },
            "duplicate designated identifiers",
        ),
    ],
)
def test_duplicate_primary_identifier_domains_fail_closed(
    job_kind: str,
    request_payload: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        attach_hierarchical_discovery_response_contract(
            job_kind=job_kind,
            request=request_payload,
        )


def _interpret_job() -> DiscoveryJsonJob:
    request = attach_hierarchical_discovery_response_contract(
        job_kind=INTERPRET_CHUNK_JOB,
        request={
            "job": "interpret_evidence_chunk",
            "evidence": [{"evidence_id": "ev.alpha", "member_ids": []}],
        },
    )
    return DiscoveryJsonJob.create(
        job_kind=INTERPRET_CHUNK_JOB,
        scope="contract.test",
        dependencies=(),
        settings=DiscoveryJobSettings.selector(),
        messages=(
            {"role": "system", "content": "Return one strict JSON object."},
            {"role": "user", "content": canonical_json(request)},
        ),
        input_bindings={"fixture": "dynamic-contract"},
    )


def test_job_binding_authenticates_full_schema_ownership_hashes_and_bytes() -> None:
    job = _interpret_job()
    binding = job.input_bindings[AUTHENTICATED_RESPONSE_CONTRACT_BINDING]
    assert binding["response_schema"] == job.response_schema
    assert binding["identifier_ownership"] == job.identifier_ownership
    schema_json = canonical_json(job.response_schema)
    ownership_json = canonical_json(job.identifier_ownership)
    assert binding["response_schema_canonical_json_utf8"] == schema_json
    assert binding["response_schema_sha256"] == content_sha256(job.response_schema)
    assert binding["response_schema_byte_count"] == len(schema_json.encode("utf-8"))
    assert binding["identifier_ownership_canonical_json_utf8"] == ownership_json
    assert binding["identifier_ownership_sha256"] == content_sha256(job.identifier_ownership)
    assert binding["identifier_ownership_byte_count"] == len(ownership_json.encode("utf-8"))
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    assert binding["binding_sha256"] == content_sha256(body)


def _unchecked_rehashed_schema_tamper(job: DiscoveryJsonJob) -> DiscoveryJsonJob:
    identity = job.as_dict()
    identity.pop("job_id")
    messages = identity["messages"]
    request = json.loads(messages[1]["content"])
    disposition = request["output_schema"]["properties"]["evidence_dispositions"]
    disposition["properties"]["ev.attacker"] = disposition["properties"].pop("ev.alpha")
    disposition["required"] = ["ev.attacker"]
    messages[1]["content"] = canonical_json(request)

    bindings = identity["input_bindings"]
    envelope = bindings[AUTHENTICATED_MESSAGE_ENVELOPE_BINDING]
    envelope["sha256"] = content_sha256(messages)
    envelope["byte_count"] = len(canonical_json(messages).encode("utf-8"))
    response_binding = bindings[AUTHENTICATED_RESPONSE_CONTRACT_BINDING]
    response_binding["response_schema"] = request["output_schema"]
    schema_json = canonical_json(request["output_schema"])
    response_binding["response_schema_canonical_json_utf8"] = schema_json
    response_binding["response_schema_sha256"] = content_sha256(request["output_schema"])
    response_binding["response_schema_byte_count"] = len(schema_json.encode("utf-8"))
    response_body = {
        key: value for key, value in response_binding.items() if key != "binding_sha256"
    }
    response_binding["binding_sha256"] = content_sha256(response_body)
    identity["messages"] = messages
    identity["input_bindings"] = bindings
    assert identity["schema_version"] == DISCOVERY_JSON_JOB_VERSION
    tampered_job_id = f"job_{content_sha256(identity)}"

    tampered = object.__new__(DiscoveryJsonJob)
    object.__setattr__(tampered, "job_id", tampered_job_id)
    object.__setattr__(tampered, "job_kind", job.job_kind)
    object.__setattr__(tampered, "scope", job.scope)
    object.__setattr__(tampered, "dependencies", job.dependencies)
    object.__setattr__(tampered, "settings", job.settings)
    object.__setattr__(tampered, "_messages_json", canonical_json(messages))
    object.__setattr__(tampered, "_input_bindings_json", canonical_json(bindings))
    return tampered


def test_schema_tamper_fails_even_after_rehashing_job_binding_and_cache_key(tmp_path: Path) -> None:
    job = _interpret_job()
    request = json.loads(job.messages[1]["content"])
    disposition = request["output_schema"]["properties"]["evidence_dispositions"]
    disposition["properties"]["ev.attacker"] = disposition["properties"].pop("ev.alpha")
    disposition["required"] = ["ev.attacker"]
    with pytest.raises(ValueError, match="differs from its designated-field derivation"):
        DiscoveryJsonJob.create(
            job_kind=job.job_kind,
            scope=job.scope,
            dependencies=job.dependencies,
            settings=job.settings,
            messages=(job.messages[0], {"role": "user", "content": canonical_json(request)}),
            input_bindings={"fixture": "dynamic-contract"},
        )

    tampered = _unchecked_rehashed_schema_tamper(job)
    with pytest.raises(ValueError, match="differs from its designated-field derivation"):
        tampered.__post_init__()

    runner_body = {"schema_version": "contract_test_runner_v1", "name": "offline"}
    runner_identity = {**runner_body, "identity_sha256": content_sha256(runner_body)}
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=tmp_path / "cache",
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    cache.begin_execution(
        hierarchy_inner_precommit_sha256="a" * 64,
        runner_identity=runner_identity,
    )
    with pytest.raises(ValueError, match="differs from its designated-field derivation"):
        cache.replay_validated(
            job=tampered,
            hierarchy_inner_precommit_sha256="a" * 64,
            runner_identity=runner_identity,
            validator_code_sha256="b" * 64,
            validator=lambda value: value,
        )


def test_adaptive_static_contract_and_source_bundle_have_no_identifier_examples() -> None:
    contract = adaptive_hierarchical_stage1_prompt_contract()
    serialized_contract = canonical_json(contract)
    assert "<active_source_family>" not in serialized_contract
    assert "<canonical_name>" not in serialized_contract
    for stage in contract["stages"]:
        assert stage["output_schema"]["example_identifier_values_present"] is False

    bundle = adaptive_hierarchical_implementation_bundle()
    filename = "hierarchical_discovery_response_contract.py"
    assert filename in bundle["files"]
    source = Path(__file__).parents[1] / "oci" / "inference" / filename
    assert bundle["files"][filename] == hashlib.sha256(source.read_bytes()).hexdigest()
    body = {key: value for key, value in bundle.items() if key != "implementation_bundle_sha256"}
    assert bundle["implementation_bundle_sha256"] == content_sha256(body)
