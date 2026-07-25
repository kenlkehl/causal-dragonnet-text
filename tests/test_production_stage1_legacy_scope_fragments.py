from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import copy

import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
)
from oci.inference.production_stage1_legacy_scope_fragments import (
    LEGACY_STAGE1_FRAGMENT_MERGE_ACCUMULATORS_SCHEMA,
    build_role_neutral_fit_only_family_seal,
    build_role_neutral_logical_evidence_bindings,
    build_role_neutral_physical_fit_artifact,
    merge_legacy_stage1_scope_fragments,
    persist_role_neutral_logical_evidence_bindings,
    seal_legacy_stage1_scope_fragment,
    validate_legacy_stage1_fragment_merge,
    validate_legacy_stage1_fragment_merge_from_path,
    validate_persisted_role_neutral_logical_evidence_bindings,
    validate_role_neutral_logical_evidence_bindings,
    validate_legacy_stage1_scope_fragment,
)
from oci.inference.production_stage1_scope_scheduler import (
    build_canonical_stage1_scope_plan,
)

_REGISTRY_SHA = "a" * 64
_REQUEST_SHA = "b" * 64


def _registry() -> dict:
    row_count = 25
    all_rows = tuple(range(row_count))
    outer_rows = []
    for outer_fold in range(1, 6):
        heldout = tuple(range((outer_fold - 1) * 5, outer_fold * 5))
        fit = tuple(row for row in all_rows if row not in set(heldout))
        partitions = tuple(fit[index::5] for index in range(5))
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": list(fit),
                "heldout_row_ids": list(heldout),
                "inner_folds": [
                    {
                        "inner_fold": inner_fold,
                        "fit_row_ids": [row for row in fit if row not in set(inner_heldout)],
                        "heldout_row_ids": list(inner_heldout),
                    }
                    for inner_fold, inner_heldout in enumerate(partitions, start=1)
                ],
            }
        )
    return {"dataset_row_count": row_count, "outer_folds": outer_rows}


def _plan(*, gpu_ids: tuple[int, ...] = (0, 1)):
    return build_canonical_stage1_scope_plan(
        registry=_registry(),
        registry_content_sha256=_REGISTRY_SHA,
        global_seed=42,
        gpu_ids=gpu_ids,
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=5,
        expected_inner_fold_count=5,
    )


def _attempt_sha(scope_id: str) -> str:
    return hashlib.sha256(f"attempt:{scope_id}".encode()).hexdigest()


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _role_neutral_bindings():
    plan = _plan()
    physical = {}
    for owner in plan.physical_scopes:
        physical[owner.scope_id] = build_role_neutral_physical_fit_artifact(
            plan=plan,
            physical_owner_scope_id=owner.scope_id,
            fit_artifact_sha256=_json_sha(
                {"physical_owner": owner.scope_id, "kind": "complete_fit"}
            ),
            family_fit_artifact_sha256={
                family: _json_sha(
                    {
                        "physical_owner": owner.scope_id,
                        "family": family,
                        "kind": "fit_side_artifact",
                    }
                )
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            },
        )
    views = {
        scope.scope_id: _json_sha(
            {
                "logical_scope": scope.scope_id,
                "purpose": scope.scope_kind,
                "kind": "logical_evidence_view",
            }
        )
        for scope in plan.scopes
    }
    bindings = build_role_neutral_logical_evidence_bindings(
        plan=plan,
        physical_fit_artifacts_by_owner=physical,
        logical_view_artifact_sha256_by_scope=views,
    )
    return plan, physical, views, bindings


def test_role_neutral_binding_separates_35_fits_from_40_logical_views():
    plan, _physical, _views, bindings = _role_neutral_bindings()

    validated = validate_role_neutral_logical_evidence_bindings(
        bindings,
        plan=plan,
    )
    assert validated["logical_scope_count"] == 40
    assert validated["physical_fit_count"] == 35
    assert validated["deduplicated_fit_count"] == 5
    assert (
        validated["all_ten_family_fit_artifact_ids_equal_within_group"]
        is True
    )
    assert validated["cross_purpose_logical_view_equality_claimed"] is False
    rows = {
        row["logical_scope_id"]: row for row in validated["logical_views"]
    }
    for owner, members in plan.physical_scope_groups:
        if len(members) == 1:
            continue
        assert len(members) == 2
        alias = members[1]
        assert owner.scope_kind == "exact_inner"
        assert alias.scope_kind == "cumulative_spent"
        assert (
            rows[alias.scope_id]["family_fit_artifact_sha256"]
            == rows[owner.scope_id]["family_fit_artifact_sha256"]
        )
        assert (
            rows[alias.scope_id]["logical_view_artifact_sha256"]
            != rows[owner.scope_id]["logical_view_artifact_sha256"]
        )
        assert (
            rows[alias.scope_id]["view_input_policy"]
            == "sealed_row_ids_only_no_sealed_text_or_labels_v1"
        )


def test_role_neutral_scientific_bindings_ignore_device_inventory():
    cpu_plan = _plan(gpu_ids=())
    heterogeneous_plan = _plan(gpu_ids=(2, 7, 11))
    assert cpu_plan.content_sha256 != heterogeneous_plan.content_sha256
    assert (
        cpu_plan.scientific_content_sha256
        == heterogeneous_plan.scientific_content_sha256
    )

    def bindings(plan):
        physical = {
            owner.scope_id: build_role_neutral_physical_fit_artifact(
                plan=plan,
                physical_owner_scope_id=owner.scope_id,
                fit_artifact_sha256=_json_sha(
                    {"owner": owner.scope_id, "kind": "fit"}
                ),
                family_fit_artifact_sha256={
                    family: _json_sha(
                        {"owner": owner.scope_id, "family": family}
                    )
                    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
                },
            )
            for owner in plan.physical_scopes
        }
        views = {
            scope.scope_id: _json_sha(
                {"scope": scope.scope_id, "kind": "logical_view"}
            )
            for scope in plan.scopes
        }
        return build_role_neutral_logical_evidence_bindings(
            plan=plan,
            physical_fit_artifacts_by_owner=physical,
            logical_view_artifact_sha256_by_scope=views,
        )

    assert bindings(cpu_plan) == bindings(heterogeneous_plan)


def test_role_neutral_fit_requires_all_ten_families():
    plan = _plan()
    owner = plan.physical_scopes[0]
    incomplete = {
        family: _json_sha({"family": family})
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES[:-1]
    }
    with pytest.raises(ValueError, match="exactly all ten"):
        build_role_neutral_physical_fit_artifact(
            plan=plan,
            physical_owner_scope_id=owner.scope_id,
            fit_artifact_sha256=_json_sha({"owner": owner.scope_id}),
            family_fit_artifact_sha256=incomplete,
        )


def test_cross_purpose_views_cannot_fabricate_byte_equality():
    plan, physical, views, _bindings = _role_neutral_bindings()
    owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if len(members) > 1
    )
    alias = members[1]
    views[alias.scope_id] = views[owner.scope_id]
    with pytest.raises(ValueError, match="cross-purpose"):
        build_role_neutral_logical_evidence_bindings(
            plan=plan,
            physical_fit_artifacts_by_owner=physical,
            logical_view_artifact_sha256_by_scope=views,
        )


def test_self_consistent_logical_family_substitution_fails_closed():
    plan, _physical, _views, bindings = _role_neutral_bindings()
    tampered = copy.deepcopy(bindings)
    owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if len(members) > 1
    )
    alias = members[1]
    row = next(
        value
        for value in tampered["logical_views"]
        if value["logical_scope_id"] == alias.scope_id
    )
    row["family_fit_artifact_sha256"][ACTIVE_STAGE1_CONCEPT_FAMILIES[0]] = (
        _json_sha({"substitution": alias.scope_id})
    )
    row_body = {key: value for key, value in row.items() if key != "content_sha256"}
    row["content_sha256"] = _json_sha(row_body)
    top_body = {
        key: value for key, value in tampered.items() if key != "content_sha256"
    }
    tampered["content_sha256"] = _json_sha(top_body)

    with pytest.raises(ValueError, match="invalid binding"):
        validate_role_neutral_logical_evidence_bindings(
            tampered,
            plan=plan,
        )


def _fit_only_payload_inputs(plan):
    seals = {}
    sources = {}
    for owner in plan.physical_scopes:
        seals[owner.scope_id] = {}
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
            payload = {
                "schema_version": "native_stage1_family_concept_evidence_v1",
                "family": family,
                "architecture_evidence": [
                    {
                        "physical_owner": owner.scope_id,
                        "term": f"{family}-fit-side-term",
                    }
                ],
            }
            seals[owner.scope_id][family] = (
                build_role_neutral_fit_only_family_seal(
                    plan=plan,
                    physical_owner_scope_id=owner.scope_id,
                    family=family,
                    evidence_payload=payload,
                    producer_identity_sha256=_json_sha(
                        {"family": family, "kind": "producer"}
                    ),
                    configuration_identity_sha256=_json_sha(
                        {"family": family, "kind": "configuration"}
                    ),
                    fit_state_artifact_sha256=_json_sha(
                        {
                            "physical_owner": owner.scope_id,
                            "family": family,
                            "kind": "fit_state",
                        }
                    ),
                )
            )
    for scope in plan.scopes:
        sources[scope.scope_id] = _json_sha(
            {
                "logical_scope": scope.scope_id,
                "source_artifact": "authenticated",
            }
        )
    return seals, sources


def test_persisted_role_neutral_artifact_writes_payload_once_per_fit(
    tmp_path: Path,
):
    plan = _plan()
    seals, sources = _fit_only_payload_inputs(plan)
    root = (tmp_path / "role_neutral").resolve()

    manifest = persist_role_neutral_logical_evidence_bindings(
        root=root,
        plan=plan,
        family_fit_seal_by_physical_owner=seals,
        logical_source_artifact_sha256_by_scope=sources,
    )
    reopened = validate_persisted_role_neutral_logical_evidence_bindings(
        root=root,
        plan=plan,
    )

    assert reopened == manifest
    assert manifest["physical_fit_count"] == 35
    assert manifest["logical_scope_count"] == 40
    assert len(tuple((root / "physical_fit_payloads").glob("*.json"))) == 35
    assert len(tuple((root / "logical_views").glob("*.json"))) == 40
    assert manifest["payload_bytes_persisted_once_per_physical_fit"] is True
    assert manifest["logical_views_are_reference_only"] is True


def test_persisted_role_neutral_artifact_rejects_logical_alias_as_physical_input(
    tmp_path: Path,
):
    plan = _plan()
    seals, sources = _fit_only_payload_inputs(plan)
    _owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if len(members) > 1
    )
    alias = members[1]
    seals[alias.scope_id] = copy.deepcopy(seals[_owner.scope_id])
    root = (tmp_path / "must_not_publish").resolve()

    with pytest.raises(ValueError, match="physical plan"):
        persist_role_neutral_logical_evidence_bindings(
            root=root,
            plan=plan,
            family_fit_seal_by_physical_owner=seals,
            logical_source_artifact_sha256_by_scope=sources,
        )
    assert not root.exists()


def test_persisted_role_neutral_artifact_rejects_tampered_physical_payload(
    tmp_path: Path,
):
    plan = _plan()
    seals, sources = _fit_only_payload_inputs(plan)
    root = (tmp_path / "role_neutral").resolve()
    persist_role_neutral_logical_evidence_bindings(
        root=root,
        plan=plan,
        family_fit_seal_by_physical_owner=seals,
        logical_source_artifact_sha256_by_scope=sources,
    )
    target = next((root / "physical_fit_payloads").glob("*.json"))
    target.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="changed|invalid"):
        validate_persisted_role_neutral_logical_evidence_bindings(
            root=root,
            plan=plan,
        )


def test_fit_only_family_seal_rejects_reordered_view_event(
    tmp_path: Path,
):
    plan = _plan()
    seals, sources = _fit_only_payload_inputs(plan)
    owner = plan.physical_scopes[0]
    family = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    tampered = copy.deepcopy(seals)
    seal = tampered[owner.scope_id][family]
    seal["event_order"][0]["registered_heldout_text_accessed"] = True
    body = {key: value for key, value in seal.items() if key != "content_sha256"}
    seal["content_sha256"] = _json_sha(body)

    with pytest.raises(ValueError, match="event binding"):
        persist_role_neutral_logical_evidence_bindings(
            root=(tmp_path / "must_not_publish_event_tamper").resolve(),
            plan=plan,
            family_fit_seal_by_physical_owner=tampered,
            logical_source_artifact_sha256_by_scope=sources,
        )


def test_cumulative_logical_reference_reuses_fit_without_sealed_text(
    tmp_path: Path,
):
    plan = _plan()
    seals, sources = _fit_only_payload_inputs(plan)
    root = (tmp_path / "role_neutral").resolve()
    persist_role_neutral_logical_evidence_bindings(
        root=root,
        plan=plan,
        family_fit_seal_by_physical_owner=seals,
        logical_source_artifact_sha256_by_scope=sources,
    )
    owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if len(members) > 1
    )
    alias = members[1]
    owner_view = json.loads(
        (root / "logical_views" / f"{owner.scope_id}.json").read_text(
            encoding="utf-8"
        )
    )
    alias_view = json.loads(
        (root / "logical_views" / f"{alias.scope_id}.json").read_text(
            encoding="utf-8"
        )
    )

    assert alias.scope_kind == "cumulative_spent"
    assert alias_view["view_input_policy"] == (
        "sealed_row_ids_only_no_sealed_text_or_labels_v1"
    )
    assert alias_view["registered_heldout_text_accessed"] is False
    assert alias_view["heldout_labels_supplied"] is False
    assert alias_view["logical_view_transform_performed"] is False
    assert alias_view["published_after_all_family_fit_seals"] is True
    assert (
        alias_view["family_fit_artifact_sha256"]
        == owner_view["family_fit_artifact_sha256"]
    )
    assert alias_view["physical_payload"] == owner_view["physical_payload"]
    assert "logical_heldout_texts" not in alias_view
    assert "logical_heldout_labels" not in alias_view


def _emit_fragments(
    tmp_path: Path,
    *,
    collision_scopes: frozenset[str] = frozenset(),
):
    plan = _plan()
    roots = {}
    attempts = {}
    for scope in plan.scopes:
        root = (tmp_path / "fragments" / scope.scope_id).resolve()
        artifacts = root / "artifacts"
        destination = (
            Path("collision/shared.json")
            if scope.scope_id in collision_scopes
            else Path("scopes") / scope.scope_id / "evidence.json"
        )
        path = artifacts / destination
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "scope_id": scope.scope_id,
                    "scope_kind": scope.scope_kind,
                    "canonical_index": scope.canonical_index,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        attempt_sha = _attempt_sha(scope.scope_id)
        seal_legacy_stage1_scope_fragment(
            fragment_root=root,
            plan=plan,
            scope_id=scope.scope_id,
            stage1_request_sha256=_REQUEST_SHA,
            scope_attempt_request_sha256=attempt_sha,
            accumulator={
                "handoff_rows": [{"scope_id": scope.scope_id}],
                "scope_index_rows": [{"scope_id": scope.scope_id}],
            },
        )
        roots[scope.scope_id] = root
        attempts[scope.scope_id] = attempt_sha
    return plan, roots, attempts


def test_all_40_fragments_merge_in_canonical_order_independent_of_mapping_order(
    tmp_path: Path,
):
    plan, roots, attempts = _emit_fragments(tmp_path)
    reverse_roots = dict(reversed(tuple(roots.items())))
    reverse_attempts = dict(reversed(tuple(attempts.items())))

    first = merge_legacy_stage1_scope_fragments(
        plan=plan,
        stage1_request_sha256=_REQUEST_SHA,
        fragment_roots_by_scope=reverse_roots,
        scope_attempt_request_sha256_by_scope=reverse_attempts,
        destination_root=(tmp_path / "merged_a").resolve(),
    )
    second = merge_legacy_stage1_scope_fragments(
        plan=plan,
        stage1_request_sha256=_REQUEST_SHA,
        fragment_roots_by_scope=roots,
        scope_attempt_request_sha256_by_scope=attempts,
        destination_root=(tmp_path / "merged_b").resolve(),
    )

    assert first == second
    assert first["scope_count"] == 40
    assert first["canonical_scope_order"] == [scope.scope_id for scope in plan.scopes]
    assert (tmp_path / "merged_a" / "merge_manifest.json").read_bytes() == (
        tmp_path / "merged_b" / "merge_manifest.json"
    ).read_bytes()
    accumulator = json.loads((tmp_path / "merged_a" / "scope_accumulators.json").read_text())
    assert accumulator["schema_version"] == LEGACY_STAGE1_FRAGMENT_MERGE_ACCUMULATORS_SCHEMA
    assert [row["scope"]["scope_id"] for row in accumulator["scopes"]] == [
        scope.scope_id for scope in plan.scopes
    ]


def test_worker_can_seal_with_one_scope_authority_and_parent_reopens_with_plan(
    tmp_path: Path,
):
    plan = _plan()
    scope = plan.scopes[7]
    root = (tmp_path / "one_scope_fragment").resolve()
    artifact = root / "artifacts" / "selected" / "evidence.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"scope_id": scope.scope_id}), encoding="utf-8")
    attempt_sha = _attempt_sha(scope.scope_id)

    sealed = seal_legacy_stage1_scope_fragment(
        fragment_root=root,
        scope_authority=scope,
        plan_content_sha256=plan.content_sha256,
        scope_id=scope.scope_id,
        stage1_request_sha256=_REQUEST_SHA,
        scope_attempt_request_sha256=attempt_sha,
        accumulator={"scope_id": scope.scope_id},
    )
    reopened = validate_legacy_stage1_scope_fragment(
        fragment_root=root,
        plan=plan,
        scope_id=scope.scope_id,
        stage1_request_sha256=_REQUEST_SHA,
        scope_attempt_request_sha256=attempt_sha,
    )

    assert sealed.identity() == reopened.identity()


def test_merge_validates_complete_coverage_before_creating_destination(
    tmp_path: Path,
):
    plan, roots, attempts = _emit_fragments(tmp_path)
    roots.pop(plan.scopes[-1].scope_id)
    destination = (tmp_path / "must_not_exist").resolve()

    with pytest.raises(ValueError, match="coverage differs"):
        merge_legacy_stage1_scope_fragments(
            plan=plan,
            stage1_request_sha256=_REQUEST_SHA,
            fragment_roots_by_scope=roots,
            scope_attempt_request_sha256_by_scope=attempts,
            destination_root=destination,
        )
    assert not destination.exists()


def test_fragment_tampering_aborts_before_merge_publication(tmp_path: Path):
    plan, roots, attempts = _emit_fragments(tmp_path)
    scope_id = plan.scopes[7].scope_id
    artifact = next((roots[scope_id] / "artifacts").rglob("*.json"))
    artifact.write_text("tampered", encoding="utf-8")
    destination = (tmp_path / "must_not_exist").resolve()

    with pytest.raises(ValueError, match="inventory changed"):
        merge_legacy_stage1_scope_fragments(
            plan=plan,
            stage1_request_sha256=_REQUEST_SHA,
            fragment_roots_by_scope=roots,
            scope_attempt_request_sha256_by_scope=attempts,
            destination_root=destination,
        )
    assert not destination.exists()


def test_fragment_symlink_is_rejected(tmp_path: Path):
    plan, roots, attempts = _emit_fragments(tmp_path)
    scope_id = plan.scopes[0].scope_id
    artifact = next((roots[scope_id] / "artifacts").rglob("*.json"))
    external = tmp_path / "external.json"
    external.write_text("external", encoding="utf-8")
    artifact.unlink()
    artifact.symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        validate_legacy_stage1_scope_fragment(
            fragment_root=roots[scope_id],
            plan=plan,
            scope_id=scope_id,
            stage1_request_sha256=_REQUEST_SHA,
            scope_attempt_request_sha256=attempts[scope_id],
        )


def test_duplicate_merge_destination_collision_is_rejected(tmp_path: Path):
    plan = _plan()
    colliding = frozenset((plan.scopes[0].scope_id, plan.scopes[1].scope_id))
    plan, roots, attempts = _emit_fragments(tmp_path, collision_scopes=colliding)
    destination = (tmp_path / "must_not_exist").resolve()

    with pytest.raises(ValueError, match="path collision"):
        merge_legacy_stage1_scope_fragments(
            plan=plan,
            stage1_request_sha256=_REQUEST_SHA,
            fragment_roots_by_scope=roots,
            scope_attempt_request_sha256_by_scope=attempts,
            destination_root=destination,
        )
    assert not destination.exists()


def test_terminal_merge_validator_rejects_post_merge_tampering(tmp_path: Path):
    plan, roots, attempts = _emit_fragments(tmp_path)
    destination = (tmp_path / "merged").resolve()
    merge_legacy_stage1_scope_fragments(
        plan=plan,
        stage1_request_sha256=_REQUEST_SHA,
        fragment_roots_by_scope=roots,
        scope_attempt_request_sha256_by_scope=attempts,
        destination_root=destination,
    )
    artifact = next((destination / "scopes").rglob("*.json"))
    artifact.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="changed"):
        validate_legacy_stage1_fragment_merge(
            plan=plan,
            stage1_request_sha256=_REQUEST_SHA,
            fragment_roots_by_scope=roots,
            scope_attempt_request_sha256_by_scope=attempts,
            destination_root=destination,
        )


def test_fresh_path_only_merge_validator_reopens_all_source_fragments(tmp_path: Path):
    plan, roots, attempts = _emit_fragments(tmp_path)
    destination = (tmp_path / "merged").resolve()
    expected = merge_legacy_stage1_scope_fragments(
        plan=plan,
        stage1_request_sha256=_REQUEST_SHA,
        fragment_roots_by_scope=roots,
        scope_attempt_request_sha256_by_scope=attempts,
        destination_root=destination,
    )

    observed = validate_legacy_stage1_fragment_merge_from_path(
        plan=plan,
        stage1_request_sha256=_REQUEST_SHA,
        destination_root=destination,
    )

    assert observed == expected


def test_path_only_merge_validator_rejects_rehashed_manifest_tampering(tmp_path: Path):
    plan, roots, attempts = _emit_fragments(tmp_path)
    destination = (tmp_path / "merged").resolve()
    merge_legacy_stage1_scope_fragments(
        plan=plan,
        stage1_request_sha256=_REQUEST_SHA,
        fragment_roots_by_scope=roots,
        scope_attempt_request_sha256_by_scope=attempts,
        destination_root=destination,
    )
    manifest_path = destination / "merge_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["copied_files"][0]["sha256"] = "0" * 64
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="order or identity changed"):
        validate_legacy_stage1_fragment_merge_from_path(
            plan=plan,
            stage1_request_sha256=_REQUEST_SHA,
            destination_root=destination,
        )


@pytest.mark.parametrize("entry_kind", ["empty_directory", "hard_link", "fifo"])
def test_closed_fragment_tree_rejects_unregistered_entry_types(
    tmp_path: Path,
    entry_kind: str,
):
    plan, roots, attempts = _emit_fragments(tmp_path)
    scope_id = plan.scopes[0].scope_id
    root = roots[scope_id]
    if entry_kind == "empty_directory":
        (root / "artifacts" / "unregistered_empty").mkdir()
    elif entry_kind == "hard_link":
        source = next((root / "artifacts").rglob("*.json"))
        os.link(source, root / "artifacts" / "unregistered_hardlink.json")
    else:
        os.mkfifo(root / "artifacts" / "unregistered_fifo")

    with pytest.raises(ValueError, match="unregistered entries|hard link|special"):
        validate_legacy_stage1_scope_fragment(
            fragment_root=root,
            plan=plan,
            scope_id=scope_id,
            stage1_request_sha256=_REQUEST_SHA,
            scope_attempt_request_sha256=attempts[scope_id],
        )
