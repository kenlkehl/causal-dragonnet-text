from __future__ import annotations

from dataclasses import replace

import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    BOW_NUISANCE,
    DiscoveryCandidate,
    bounded_candidate_relation_pages,
    candidate_definition_fold_batches,
    content_sha256,
)
from oci.inference.hierarchical_all_architecture_discovery import (
    HierarchicalDiscoveryConfig,
    _compile_bounded_consolidation,
)
from oci.inference.hierarchical_discovery_response_contract import (
    HIERARCHY_WIRE_BUDGET_SCHEMA_VERSION,
    HierarchyWireBudget,
    LEGACY_HIERARCHY_WIRE_BUDGET,
    attach_hierarchical_discovery_response_contract,
)


def _candidate(index: int) -> DiscoveryCandidate:
    return DiscoveryCandidate(
        candidate_id=f"candidate_{index:03d}",
        feature_name=f"feature_{index:03d}",
        description=f"candidate {index}",
        supporting_evidence_ids=(f"evidence_{index:03d}",),
        source_families=(BOW_NUISANCE,),
        value_shape_hypothesis="continuous",
    )


def _profiles() -> tuple[HierarchyWireBudget, HierarchyWireBudget]:
    compact = replace(
        LEGACY_HIERARCHY_WIRE_BUDGET,
        max_pair_relation_peers_per_page=2,
        max_definition_fold_inputs=3,
    )
    roomy = replace(
        LEGACY_HIERARCHY_WIRE_BUDGET,
        max_pair_relation_peers_per_page=5,
        max_definition_fold_inputs=6,
    )
    return compact, roomy


def test_wire_budget_profiles_page_losslessly_and_change_scientific_identity() -> None:
    compact, roomy = _profiles()
    candidates = tuple(_candidate(index) for index in range(9))
    expected_pairs = len(candidates) * (len(candidates) - 1) // 2

    compact_pages = bounded_candidate_relation_pages(
        candidates,
        wire_budget=compact,
    )
    roomy_pages = bounded_candidate_relation_pages(
        candidates,
        wire_budget=roomy,
    )
    assert sum(page["pair_count"] for page in compact_pages) == expected_pairs
    assert sum(page["pair_count"] for page in roomy_pages) == expected_pairs
    assert max(page["pair_count"] for page in compact_pages) == 2
    assert max(page["pair_count"] for page in roomy_pages) == 5
    assert len(compact_pages) > len(roomy_pages)

    members = tuple(candidate.candidate_id for candidate in candidates)
    compact_folds = candidate_definition_fold_batches(
        group_id="candidate_group_test",
        member_candidate_ids=members,
        wire_budget=compact,
    )
    roomy_folds = candidate_definition_fold_batches(
        group_id="candidate_group_test",
        member_candidate_ids=members,
        wire_budget=roomy,
    )
    for folds in (compact_folds, roomy_folds):
        flattened = tuple(
            member
            for fold in folds
            for member in fold["member_candidate_ids"]
        )
        assert flattened == members
        assert len(flattened) == len(set(flattened))
    assert len(compact_folds) > len(roomy_folds)

    compact_identity = content_sha256(
        HierarchicalDiscoveryConfig(wire_budget=compact).as_dict()
    )
    roomy_identity = content_sha256(
        HierarchicalDiscoveryConfig(wire_budget=roomy).as_dict()
    )
    assert compact_identity != roomy_identity


def test_wire_budget_is_closed_versioned_and_changes_attached_contract() -> None:
    compact, roomy = _profiles()
    request = {
        "job": "compare_consolidation_candidate_relations",
        "anchor_candidate_id": "candidate_000",
        "peer_candidate_ids": ["candidate_001", "candidate_002"],
    }
    compact_request = attach_hierarchical_discovery_response_contract(
        job_kind="consolidate_architecture_candidates",
        request=request,
        wire_budget=compact,
    )
    roomy_request = attach_hierarchical_discovery_response_contract(
        job_kind="consolidate_architecture_candidates",
        request=request,
        wire_budget=roomy,
    )
    assert compact_request["hierarchy_wire_budget"] == compact.as_dict()
    assert roomy_request["hierarchy_wire_budget"] == roomy.as_dict()
    assert content_sha256(compact_request) != content_sha256(roomy_request)
    assert (
        compact_request["identifier_ownership"]["hierarchy_wire_budget"]
        == compact.as_dict()
    )

    missing = compact.as_dict()
    missing.pop("max_generated_list_items")
    with pytest.raises(ValueError, match="keys differ"):
        HierarchyWireBudget.from_mapping(missing)

    extra = compact.as_dict()
    extra["unregistered_limit"] = 1
    with pytest.raises(ValueError, match="keys differ"):
        HierarchyWireBudget.from_mapping(extra)

    wrong_version = compact.as_dict()
    wrong_version["budget_version"] = HIERARCHY_WIRE_BUDGET_SCHEMA_VERSION + "_unknown"
    with pytest.raises(ValueError, match="budget_version"):
        HierarchyWireBudget.from_mapping(wrong_version)


def test_duplicate_name_compiler_uses_explicit_nonlegacy_wire_capacity() -> None:
    narrow = replace(
        LEGACY_HIERARCHY_WIRE_BUDGET,
        max_generated_name_chars=40,
        max_interpret_name_chars=40,
    )
    roomy = replace(
        LEGACY_HIERARCHY_WIRE_BUDGET,
        max_generated_name_chars=56,
        max_interpret_name_chars=56,
    )
    candidates = (_candidate(0), _candidate(1))
    group_ids = ("candidate_group_000", "candidate_group_001")
    grouped = {
        "groups": [
            {
                "group_id": group_id,
                "member_candidate_ids": [candidate.candidate_id],
            }
            for group_id, candidate in zip(group_ids, candidates, strict=True)
        ],
        "pair_relation_audit": {"expected_pair_count": 1},
    }
    proposed = "patient_treatment_response_measure"
    definitions = {
        group_id: {
            "canonical_name": proposed,
            "description": "One configured duplicate-name capacity regression.",
            "unresolved_ambiguity": "",
            "reason": "Exercise deterministic duplicate-name disambiguation.",
        }
        for group_id in group_ids
    }

    narrow_result = _compile_bounded_consolidation(
        source_family=BOW_NUISANCE,
        candidates=candidates,
        grouped=grouped,
        definitions_by_group_id=definitions,
        wire_budget=narrow,
    )
    roomy_result = _compile_bounded_consolidation(
        source_family=BOW_NUISANCE,
        candidates=candidates,
        grouped=grouped,
        definitions_by_group_id=definitions,
        wire_budget=roomy,
    )
    narrow_names = [
        row["canonical_name"] for row in narrow_result["canonical_concepts"]
    ]
    roomy_names = [
        row["canonical_name"] for row in roomy_result["canonical_concepts"]
    ]

    assert narrow_names[0] == roomy_names[0] == proposed
    assert len(narrow_names[1]) == narrow.max_generated_name_chars
    assert len(roomy_names[1]) <= roomy.max_generated_name_chars
    assert narrow_names[1] != roomy_names[1]
    assert narrow.content_sha256 != roomy.content_sha256
    assert content_sha256(narrow_result) != content_sha256(roomy_result)
