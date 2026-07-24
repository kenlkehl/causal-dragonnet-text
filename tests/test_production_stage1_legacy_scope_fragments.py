from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from oci.inference.production_stage1_legacy_scope_fragments import (
    LEGACY_STAGE1_FRAGMENT_MERGE_ACCUMULATORS_SCHEMA,
    merge_legacy_stage1_scope_fragments,
    seal_legacy_stage1_scope_fragment,
    validate_legacy_stage1_fragment_merge,
    validate_legacy_stage1_fragment_merge_from_path,
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


def _plan():
    return build_canonical_stage1_scope_plan(
        registry=_registry(),
        registry_content_sha256=_REGISTRY_SHA,
        global_seed=42,
        gpu_ids=(0, 1),
        review_rounds=2,
        expected_outer_fold_count=5,
        expected_inner_fold_count=5,
    )


def _attempt_sha(scope_id: str) -> str:
    return hashlib.sha256(f"attempt:{scope_id}".encode()).hexdigest()


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
