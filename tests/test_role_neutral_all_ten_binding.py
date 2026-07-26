from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)
from oci.inference.production_stage1_legacy_scope_fragments import (
    build_role_neutral_fit_only_family_seal,
)
from oci.inference.production_stage1_scope_scheduler import (
    build_canonical_stage1_scope_plan,
)
from tests.stage1_test_support import PHYSICAL_FIT_IDENTITY
from oci.inference.role_neutral_all_ten_binding import (
    AuthenticatedRoleNeutralComponentReceipt,
    EXPECTED_COMPONENT_FAMILIES,
    PORTABLE_TO_NATIVE_FAMILY,
    merge_all_ten_components_for_owner,
    persist_complete_role_neutral_stage1_bindings,
    validate_complete_role_neutral_stage1_bindings,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _registry() -> dict:
    all_rows = tuple(range(25))
    folds = []
    for outer_fold in range(1, 6):
        heldout = tuple(range((outer_fold - 1) * 5, outer_fold * 5))
        fit = tuple(row for row in all_rows if row not in set(heldout))
        partitions = tuple(fit[index::5] for index in range(5))
        folds.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": list(fit),
                "heldout_row_ids": list(heldout),
                "inner_folds": [
                    {
                        "inner_fold": index,
                        "fit_row_ids": [
                            row for row in fit if row not in set(partition)
                        ],
                        "heldout_row_ids": list(partition),
                    }
                    for index, partition in enumerate(partitions, start=1)
                ],
            }
        )
    return {"dataset_row_count": len(all_rows), "outer_folds": folds}


def _plan(*, gpu_ids: tuple[int, ...] = ()):
    return build_canonical_stage1_scope_plan(
        registry=_registry(),
        registry_content_sha256="a" * 64,
        global_seed=42,
        physical_fit_identity=PHYSICAL_FIT_IDENTITY,
        gpu_ids=gpu_ids,
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=5,
        expected_inner_fold_count=5,
    )


_COMPONENT_FAMILIES = {
    "bow": (BOW_NUISANCE, BOW_R_LOSS),
    "htr": (HTR_NEURAL,),
    "matched_pair": (MATCHED_PAIR_UPLIFT,),
    "embeddings": (
        EMBEDDING_WHOLE_COHORT,
        EMBEDDING_CLUSTERED,
        TFIDF_SEMANTIC_RETRIEVAL,
    ),
    "tfidf": (TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS),
    "neural_query": (NEURAL_QUERY_MOMENTS,),
}


def _receipt(plan, owner_scope_id: str, component: str, *, tree_tag: str = "a"):
    owner = plan.scope(owner_scope_id)
    members = next(
        members
        for candidate, members in plan.physical_scope_groups
        if candidate.scope_id == owner_scope_id
    )
    seals = {}
    views = {}
    for family in _COMPONENT_FAMILIES[component]:
        payload = {
            "schema_version": "native_stage1_family_concept_evidence_v1",
            "family": family,
            "architecture_evidence": [
                {
                    "kind": "complete_test_atom",
                    "family": family,
                    "owner": owner_scope_id,
                    "sentinel_after_configured_page_boundary": True,
                }
            ],
        }
        seals[family] = build_role_neutral_fit_only_family_seal(
            plan=plan,
            physical_owner_scope_id=owner_scope_id,
            family=family,
            evidence_payload=payload,
            producer_identity_sha256=_sha(
                {"component": component, "family": family, "kind": "producer"}
            ),
            configuration_identity_sha256=_sha(
                {"component": component, "family": family, "kind": "config"}
            ),
            fit_state_artifact_sha256=_sha(
                {"owner": owner_scope_id, "family": family, "kind": "state"}
            ),
        )
        views[family] = {
            member.scope_id: _sha(
                {
                    "owner": owner_scope_id,
                    "scope": member.scope_id,
                    "family": family,
                    "purpose": member.scope_kind,
                }
            )
            for member in members
        }
    return AuthenticatedRoleNeutralComponentReceipt.create(
        plan=plan,
        physical_owner_scope_id=owner_scope_id,
        component=component,
        family_fit_seals=seals,
        family_logical_view_content_sha256=views,
        source_terminal_content_sha256=_sha(
            {"owner": owner_scope_id, "component": component, "kind": "terminal"}
        ),
        source_tree_sha256=_sha(
            {
                "owner": owner_scope_id,
                "component": component,
                "kind": "tree",
                "execution": tree_tag,
            }
        ),
    )


def _receipts(plan, owner_scope_id: str):
    return tuple(
        _receipt(plan, owner_scope_id, component)
        for component in _COMPONENT_FAMILIES
    )


def test_portable_and_native_family_mapping_is_exact_and_independent():
    assert tuple(PORTABLE_TO_NATIVE_FAMILY) == (
        "word_treatment_outcome",
        "word_residual_effect",
        "hierarchical_transformer",
        "matched_patient_uplift",
        "whole_cohort_embeddings",
        "cluster_local_embeddings",
        "lexical_semantic_retrieval",
        "tfidf_topics",
        "residual_tfidf_ngrams",
        "learned_neural_queries",
    )
    assert set(PORTABLE_TO_NATIVE_FAMILY.values()) == set(
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    assert len(set(PORTABLE_TO_NATIVE_FAMILY.values())) == 10
    assert dict(EXPECTED_COMPONENT_FAMILIES) == _COMPONENT_FAMILIES


def test_owner_merge_requires_all_ten_and_keeps_distinct_logical_views():
    plan = _plan()
    owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if len(members) == 2
    )
    merged = merge_all_ten_components_for_owner(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
        components=_receipts(plan, owner.scope_id),
    )
    assert set(merged["family_fit_seals"]) == set(
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    assert merged["all_ten_nonempty_fit_families_present"] is True
    assert merged["text_truncation_applied"] is False
    source_ids = {
        scope_id: value["content_sha256"]
        for scope_id, value in merged["logical_source_identities"].items()
    }
    assert set(source_ids) == {member.scope_id for member in members}
    assert len(set(source_ids.values())) == len(members)

    with pytest.raises(ValueError, match="incomplete family coverage"):
        merge_all_ten_components_for_owner(
            plan=plan,
            physical_owner_scope_id=owner.scope_id,
            components=_receipts(plan, owner.scope_id)[:-1],
        )


def test_receipt_rejects_any_truncation_or_lossy_selection():
    plan = _plan()
    owner = plan.physical_scopes[0]
    receipt = _receipt(plan, owner.scope_id, "htr")
    kwargs = {
        "plan": plan,
        "physical_owner_scope_id": owner.scope_id,
        "component": receipt.component,
        "family_fit_seals": receipt.family_fit_seals,
        "family_logical_view_content_sha256": (
            receipt.family_logical_view_content_sha256
        ),
        "source_terminal_content_sha256": (
            receipt.source_terminal_content_sha256
        ),
        "source_tree_sha256": receipt.source_tree_sha256,
    }
    with pytest.raises(ValueError, match="text truncation"):
        AuthenticatedRoleNeutralComponentReceipt.create(
            **kwargs,
            text_truncation_applied=True,
        )
    with pytest.raises(ValueError, match="lossy evidence"):
        AuthenticatedRoleNeutralComponentReceipt.create(
            **kwargs,
            lossy_evidence_selection_applied=True,
        )


def test_receipt_rejects_wrong_component_partition_and_duplicate_views():
    plan = _plan()
    owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if len(members) == 2
    )
    receipt = _receipt(plan, owner.scope_id, "htr")
    with pytest.raises(ValueError, match="canonical native family partition"):
        AuthenticatedRoleNeutralComponentReceipt.create(
            plan=plan,
            physical_owner_scope_id=owner.scope_id,
            component="bow",
            family_fit_seals=receipt.family_fit_seals,
            family_logical_view_content_sha256=(
                receipt.family_logical_view_content_sha256
            ),
            source_terminal_content_sha256=(
                receipt.source_terminal_content_sha256
            ),
            source_tree_sha256=receipt.source_tree_sha256,
        )
    duplicated = {
        HTR_NEURAL: {
            member.scope_id: "f" * 64
            for member in members
        }
    }
    with pytest.raises(ValueError, match="purpose-specific scopes"):
        AuthenticatedRoleNeutralComponentReceipt.create(
            plan=plan,
            physical_owner_scope_id=owner.scope_id,
            component="htr",
            family_fit_seals=receipt.family_fit_seals,
            family_logical_view_content_sha256=duplicated,
            source_terminal_content_sha256=(
                receipt.source_terminal_content_sha256
            ),
            source_tree_sha256=receipt.source_tree_sha256,
        )


def test_merge_rejects_mutated_or_resigned_receipt_handle():
    plan = _plan()
    owner = plan.physical_scopes[0]
    receipts = list(_receipts(plan, owner.scope_id))
    bow = receipts[0]
    bow.family_logical_view_content_sha256[BOW_NUISANCE][
        owner.scope_id
    ] = "f" * 64
    with pytest.raises(RuntimeError, match="receipt was mutated"):
        merge_all_ten_components_for_owner(
            plan=plan,
            physical_owner_scope_id=owner.scope_id,
            components=receipts,
        )
    receipts = list(_receipts(plan, owner.scope_id))
    receipts[0] = replace(
        receipts[0],
        authentication_content_sha256="f" * 64,
    )
    with pytest.raises(RuntimeError, match="receipt was mutated"):
        merge_all_ten_components_for_owner(
            plan=plan,
            physical_owner_scope_id=owner.scope_id,
            components=receipts,
        )


def test_scientific_receipts_ignore_devices_and_execution_tree_identity():
    cpu = _plan(gpu_ids=())
    heterogeneous = _plan(gpu_ids=(2, 7, 11))
    owner_id = cpu.physical_scopes[0].scope_id
    cpu_receipt = _receipt(cpu, owner_id, "bow", tree_tag="cpu")
    gpu_receipt = _receipt(
        heterogeneous,
        owner_id,
        "bow",
        tree_tag="heterogeneous-gpu",
    )
    assert cpu.content_sha256 != heterogeneous.content_sha256
    assert cpu_receipt.source_tree_sha256 != gpu_receipt.source_tree_sha256
    assert cpu_receipt.scientific_dict() == gpu_receipt.scientific_dict()
    assert (
        cpu_receipt.execution_attestation()
        != gpu_receipt.execution_attestation()
    )


def test_persisted_full_plan_contains_35_physical_and_40_logical_records(
    tmp_path: Path,
):
    plan = _plan(gpu_ids=(4, 9))
    components = {
        owner.scope_id: _receipts(plan, owner.scope_id)
        for owner in plan.physical_scopes
    }
    root = (tmp_path / "all_ten_bindings").resolve()
    terminal = persist_complete_role_neutral_stage1_bindings(
        root=root,
        plan=plan,
        components_by_physical_owner=components,
    )
    assert terminal["physical_fit_count"] == 35
    assert terminal["logical_scope_count"] == 40
    assert terminal["deduplicated_fit_count"] == 5
    assert (
        validate_complete_role_neutral_stage1_bindings(
            root=root,
            plan=plan,
        )
        == terminal
    )
    assert len(tuple((root / "physical_fit_payloads").glob("*.json"))) == 35
    assert len(tuple((root / "logical_views").glob("*.json"))) == 40
