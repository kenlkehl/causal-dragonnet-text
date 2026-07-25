from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oci.inference.production_stage1_bundle import (
    publish_authenticated_role_neutral_stage1_bindings,
    validate_authenticated_role_neutral_stage1_bindings,
)
from oci.inference.production_stage1_legacy_scope_adapter import (
    LegacyStage1RoleSpecificDeduplicationError,
)
from oci.inference.production_stage1_legacy_scope_fragments import (
    build_role_neutral_fit_only_family_seal,
)
from oci.inference.production_stage1_role_neutral_coordinator import (
    ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION,
    ROLE_NEUTRAL_SCIENTIFIC_BINDING_DIRECTORY,
    RoleNeutralComponentArtifactSource,
    _component_tree_sha256,
)
from oci.inference.production_stage1_scope_scheduler import (
    build_canonical_stage1_scope_plan,
)
from oci.inference.role_neutral_all_ten_binding import (
    AuthenticatedRoleNeutralComponentReceipt,
    EXPECTED_COMPONENT_FAMILIES,
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
                        "fit_row_ids": [row for row in fit if row not in set(partition)],
                        "heldout_row_ids": list(partition),
                    }
                    for index, partition in enumerate(
                        partitions,
                        start=1,
                    )
                ],
            }
        )
    return {"dataset_row_count": len(all_rows), "outer_folds": folds}


def _plan():
    return build_canonical_stage1_scope_plan(
        registry=_registry(),
        registry_content_sha256="a" * 64,
        global_seed=42,
        gpu_ids=(3, 8),
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=5,
        expected_inner_fold_count=5,
    )


def _sources(plan, root: Path):
    sources = {}
    for owner, members in plan.physical_scope_groups:
        owner_sources = []
        for component, families in EXPECTED_COMPONENT_FAMILIES.items():
            component_root = (root / owner.scope_id / component).resolve()
            component_root.mkdir(parents=True, exist_ok=False)
            terminal_body = {
                "schema_version": "test_role_neutral_terminal_v1",
                "physical_owner_scope_id": owner.scope_id,
                "component": component,
            }
            terminal = {
                **terminal_body,
                "content_sha256": _sha(terminal_body),
            }
            (component_root / "execution_manifest.json").write_text(
                json.dumps(
                    terminal,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            seals = {}
            views = {}
            for family in families:
                payload = {
                    "schema_version": ("native_stage1_family_concept_evidence_v1"),
                    "family": family,
                    "architecture_evidence": [
                        {
                            "kind": "complete_test_atom",
                            "family": family,
                            "physical_owner_scope_id": owner.scope_id,
                            "sentinel_after_configured_page_boundary": True,
                        }
                    ],
                }
                seals[family] = build_role_neutral_fit_only_family_seal(
                    plan=plan,
                    physical_owner_scope_id=owner.scope_id,
                    family=family,
                    evidence_payload=payload,
                    producer_identity_sha256=_sha(
                        {
                            "component": component,
                            "family": family,
                            "producer": "test",
                        }
                    ),
                    configuration_identity_sha256=_sha(
                        {
                            "component": component,
                            "family": family,
                            "configuration": "explicit",
                        }
                    ),
                    fit_state_artifact_sha256=_sha(
                        {
                            "owner": owner.scope_id,
                            "family": family,
                            "fit_state": "complete",
                        }
                    ),
                )
                views[family] = {
                    member.scope_id: _sha(
                        {
                            "owner": owner.scope_id,
                            "scope": member.scope_id,
                            "purpose": member.scope_kind,
                            "component": component,
                            "family": family,
                        }
                    )
                    for member in members
                }
            receipt = AuthenticatedRoleNeutralComponentReceipt.create(
                plan=plan,
                physical_owner_scope_id=owner.scope_id,
                component=component,
                family_fit_seals=seals,
                family_logical_view_content_sha256=views,
                source_terminal_content_sha256=terminal["content_sha256"],
                source_tree_sha256=_component_tree_sha256(component_root),
            )
            owner_sources.append(
                RoleNeutralComponentArtifactSource(
                    root=component_root,
                    receipt=receipt,
                )
            )
        sources[owner.scope_id] = tuple(owner_sources)
    return sources


def test_explicit_production_gate_publishes_35_physical_and_40_logical(
    tmp_path: Path,
):
    plan = _plan()
    sources = _sources(plan, (tmp_path / "component_roots").resolve())
    gate_root = (tmp_path / "role_neutral_gate").resolve()
    manifest = publish_authenticated_role_neutral_stage1_bindings(
        root=gate_root,
        plan=plan,
        sources_by_physical_owner=sources,
    )
    assert manifest["physical_fit_count"] == 35
    assert manifest["logical_scope_count"] == 40
    assert manifest["deduplicated_fit_count"] == 5
    assert manifest["producer_component_count_per_physical_owner"] == 6
    assert manifest["legacy_role_specific_fragments_adopted"] is False
    assert manifest["component_root_locators_in_scientific_identity"] is False
    assert (
        validate_authenticated_role_neutral_stage1_bindings(
            root=gate_root,
            plan=plan,
        )
        == manifest
    )
    binding_root = gate_root / ROLE_NEUTRAL_SCIENTIFIC_BINDING_DIRECTORY
    assert len(tuple((binding_root / "physical_fit_payloads").glob("*.json"))) == 35
    assert len(tuple((binding_root / "logical_views").glob("*.json"))) == 40
    attestation = json.loads(
        (gate_root / ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION).read_text(encoding="utf-8")
    )
    assert attestation["registration_count"] == 35 * 6

    first_source = sources[plan.physical_scopes[0].scope_id][0]
    (first_source.root / "execution_manifest.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="component tree changed"):
        validate_authenticated_role_neutral_stage1_bindings(
            root=gate_root,
            plan=plan,
        )


def test_gate_rejects_missing_or_substituted_components_before_output(
    tmp_path: Path,
):
    plan = _plan()
    sources = _sources(plan, (tmp_path / "component_roots").resolve())
    first_owner = plan.physical_scopes[0].scope_id

    incomplete = dict(sources)
    incomplete[first_owner] = incomplete[first_owner][:-1]
    missing_root = (tmp_path / "missing_component_gate").resolve()
    with pytest.raises(ValueError, match="canonical six"):
        publish_authenticated_role_neutral_stage1_bindings(
            root=missing_root,
            plan=plan,
            sources_by_physical_owner=incomplete,
        )
    assert not missing_root.exists()

    substituted = dict(sources)
    rows = list(substituted[first_owner])
    rows[1] = RoleNeutralComponentArtifactSource(
        root=rows[0].root,
        receipt=rows[1].receipt,
    )
    substituted[first_owner] = tuple(rows)
    substituted_root = (tmp_path / "substituted_component_gate").resolve()
    with pytest.raises(
        ValueError,
        match="source tree changed|distinct, nonnested",
    ):
        publish_authenticated_role_neutral_stage1_bindings(
            root=substituted_root,
            plan=plan,
            sources_by_physical_owner=substituted,
        )
    assert not substituted_root.exists()


def test_legacy_role_specific_failure_type_remains_public():
    assert issubclass(
        LegacyStage1RoleSpecificDeduplicationError,
        RuntimeError,
    )
