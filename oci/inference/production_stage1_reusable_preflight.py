"""Reusable, owner-granular scientific precomputation for Stage 1 preflight.

This module is deliberately independent of workflow durable roots and runtime
resource assignment.  It stores three kinds of immutable scientific artifact:

* one exact full-cohort HTR non-truncation audit;
* one clustered-embedding feasibility/state artifact per physical owner; and
* one small assembled context that binds the global audit, every owner, and
  the canonical logical-to-physical plan.

The first successful load hashes every registered byte.  A private operational
proof records that authentication and the exact stat identity of every file and
directory.  Later loads may use metadata-only authentication when, and only
when, the proof and every stat field remain exactly continuous.  Any ambiguity
falls back to ordinary full-byte authentication.

Scientific identities never contain paths, hostnames, devices, worker counts,
or Stage 2 implementation details.  Absolute paths occur only in locator
payloads and operational proofs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import resource
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .production_stage1_cluster_preflight_artifact_v2 import (
    PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_SCHEMA,
    PORTABLE_CLUSTER_PREFLIGHT_RESULT_SCHEMA,
    PortableProductionStage1ClusterPreflightArtifact,
    _build_compact_index,
    _read_owner_parquet,
    _reference_from_index,
    _validate_compact_index,
    _validate_stage1_request_with_reference,
    _write_owner_parquet,
)


REUSABLE_PREFLIGHT_STORE_SCHEMA = (
    "production_stage1_reusable_preflight_store_v2"
)
REUSABLE_GLOBAL_AUDIT_SCHEMA = (
    "production_stage1_reusable_global_nontruncation_artifact_v1"
)
REUSABLE_OWNER_ARTIFACT_SCHEMA = (
    "production_stage1_reusable_cluster_owner_artifact_v1"
)
REUSABLE_ASSEMBLED_ARTIFACT_SCHEMA = (
    "production_stage1_reusable_assembled_preflight_artifact_v3"
)
REUSABLE_PREFLIGHT_REFERENCE_SCHEMA = (
    "production_stage1_reusable_preflight_reference_v1"
)
REUSABLE_STATE_BUNDLE_REFERENCE_SCHEMA = (
    "production_stage1_reusable_cluster_state_bundle_reference_v1"
)
REUSABLE_ACCEPTED_CONTEXT_SCHEMA = (
    "production_stage1_reusable_accepted_context_v2"
)
REUSABLE_AUTH_PROOF_SCHEMA = (
    "production_stage1_reusable_preflight_authentication_proof_v3"
)
REUSABLE_AUTH_RECEIPT_TERMINAL_SCHEMA = (
    "production_stage1_reusable_preflight_authentication_receipt_terminal_v2"
)
REUSABLE_PREFLIGHT_ARTIFACT_VERSION = (
    "production_stage1_reusable_preflight_v1"
)
GLOBAL_AUDIT_COMPATIBILITY_SCHEMA = (
    "production_stage1_global_nontruncation_compatibility_v1"
)
OWNER_COMPATIBILITY_SCHEMA = (
    "production_stage1_cluster_owner_compatibility_v3"
)
ASSEMBLED_COMPATIBILITY_SCHEMA = (
    "production_stage1_assembled_preflight_compatibility_v3"
)
PREFLIGHT_SCOPE_PLAN_PROJECTION_SCHEMA = (
    "production_stage1_preflight_scope_plan_projection_v1"
)

GLOBAL_TERMINAL = "global_nontruncation_terminal.json"
OWNER_TERMINAL = "owner_preflight_terminal.json"
ASSEMBLED_TERMINAL = "assembled_preflight_terminal.json"
REFERENCE_MANIFEST = "cluster_preflight_manifest.json"
STATE_BUNDLE_MANIFEST = "cluster_state_bundle_manifest.json"
ACCEPTED_CONTEXT_TERMINAL = "accepted_context_terminal.json"

_READ_ONLY_FILE_MODE = 0o444
_READ_ONLY_DIRECTORY_MODE = 0o555
_PRIVATE_FILE_MODE = 0o600
_HEX = frozenset("0123456789abcdef")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"reusable preflight value is not JSON serializable: "
        f"{type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TypeError(f"{label} must be closed finite JSON") from exc


def stage1_scope_plan_from_mapping(value: Mapping[str, Any]) -> Any:
    """Reopen the sealed canonical plan without cohort or cache I/O."""

    from .production_stage1_scope_scheduler import (
        Stage1PhysicalFitIdentity,
        Stage1ScopeAssignment,
        Stage1ScopePlan,
        Stage1ScopeSpec,
    )

    closed = _json_copy(value, label="sealed Stage 1 scope plan")
    raw_scopes = closed.get("scopes")
    raw_assignments = closed.get("assignments")
    if not isinstance(raw_scopes, list) or not isinstance(
        raw_assignments,
        list,
    ):
        raise ValueError("sealed Stage 1 scope plan has no closed rows")
    scopes = tuple(
        Stage1ScopeSpec(
            canonical_index=int(row["canonical_index"]),
            scope_id=str(row["scope_id"]),
            scope_kind=str(row["scope_kind"]),
            outer_fold=int(row["outer_fold"]),
            inner_fold=(
                None
                if row["inner_fold"] is None
                else int(row["inner_fold"])
            ),
            context_epoch=(
                None
                if row["context_epoch"] is None
                else int(row["context_epoch"])
            ),
            provider_inner_fold=(
                None
                if row["provider_inner_fold"] is None
                else int(row["provider_inner_fold"])
            ),
            fit_row_ids=tuple(map(int, row["fit_row_ids"])),
            heldout_row_ids=tuple(
                map(int, row["heldout_row_ids"])
            ),
            global_seed=int(row["global_seed"]),
            scope_seed=int(row["scope_seed"]),
        )
        for row in raw_scopes
    )
    assignments = tuple(
        Stage1ScopeAssignment(
            scope_id=str(row["scope_id"]),
            gpu_id=(
                None
                if row["gpu_id"] is None
                else int(row["gpu_id"])
            ),
            execution_rank=int(row["execution_rank"]),
            fit_row_count=int(row["fit_row_count"]),
            assigned_gpu_load_after=int(
                row["assigned_gpu_load_after"]
            ),
        )
        for row in raw_assignments
    )
    plan = Stage1ScopePlan(
        registry_content_sha256=str(
            closed["registry_content_sha256"]
        ),
        global_seed=int(closed["global_seed"]),
        review_rounds=int(closed["review_rounds"]),
        initial_training_partitions=int(
            closed["initial_training_partitions"]
        ),
        physical_fit_identity=(
            Stage1PhysicalFitIdentity.from_mapping(
                closed["physical_fit_identity"]
            )
        ),
        gpu_ids=tuple(map(int, closed["gpu_ids"])),
        scope_workers_per_gpu=int(
            closed["scope_workers_per_gpu"]
        ),
        scopes=scopes,
        assignments=assignments,
        content_sha256=str(closed["content_sha256"]),
    )
    if plan.as_dict() != closed:
        raise ValueError("sealed Stage 1 scope plan changed")
    return plan


def preflight_scope_plan_projection(
    value: Any,
) -> dict[str, Any]:
    """Project only the row/split/seed topology that preflight can observe.

    ``Stage1ScopePlan.scientific_content_sha256`` deliberately binds the
    all-ten-family physical-fit identity.  That identity is correct for model
    execution, but is too broad for reusable HTR input auditing and
    cluster-local KMeans/SVD state: it can change when Stage 2 or an unrelated
    Stage 1 producer changes.  This projection retains every row, fold,
    canonical seed, and logical-to-physical deduplication decision while
    excluding that unrelated producer/configuration envelope and all resource
    assignments.
    """

    from .production_stage1_scope_scheduler import Stage1ScopePlan

    plan = (
        stage1_scope_plan_from_mapping(value)
        if isinstance(value, Mapping)
        else value
    )
    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError(
            "preflight scope-plan projection requires a Stage1ScopePlan"
        )
    scopes = tuple(plan.scopes)
    physical_groups = tuple(plan.physical_scope_groups)
    owner_by_scope = {
        member.scope_id: owner.scope_id
        for owner, members in physical_groups
        for member in members
    }
    if set(owner_by_scope) != {scope.scope_id for scope in scopes}:
        raise RuntimeError(
            "preflight scope-plan projection lost a logical scope"
        )
    body = {
        "schema_version": PREFLIGHT_SCOPE_PLAN_PROJECTION_SCHEMA,
        "registry_content_sha256": _require_sha256(
            plan.registry_content_sha256,
            label="preflight split registry",
        ),
        "global_seed": int(plan.global_seed),
        "scope_seed_derivation": (
            "sha256(global_seed,canonical_ordered_fit_rows)_31bit_v2"
        ),
        "review_rounds": int(plan.review_rounds),
        "initial_training_partitions": int(
            plan.initial_training_partitions
        ),
        "canonical_scope_order": [
            scope.scope_id for scope in scopes
        ],
        "physical_scope_order": [
            owner.scope_id for owner, _members in physical_groups
        ],
        "logical_to_physical_owner": [
            {
                "logical_scope_id": scope.scope_id,
                "physical_owner_scope_id": owner_by_scope[
                    scope.scope_id
                ],
            }
            for scope in scopes
        ],
        "scopes": [scope.as_dict() for scope in scopes],
        "logical_scope_count": len(scopes),
        "physical_scope_count": len(physical_groups),
        "heldout_labels_present": False,
        "physical_fit_identity_included": False,
        "resource_assignment_included": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _normalize_stage1_scope_plan(value: Any) -> Any:
    if isinstance(value, Mapping):
        return stage1_scope_plan_from_mapping(value)
    if not callable(getattr(value, "as_dict", None)):
        raise TypeError("reusable preflight requires a canonical scope plan")
    return stage1_scope_plan_from_mapping(value.as_dict())


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(
                f"reusable preflight JSON repeats key {key!r}"
            )
        output[key] = value
    return output


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    state = os.lstat(path)
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISREG(state.st_mode)
        or int(state.st_nlink) != 1
    ):
        raise ValueError(f"{label} must be one private regular file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{label} contains {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not closed UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one object")
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> int:
    payload = (_canonical_json(dict(value)) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        _PRIVATE_FILE_MODE,
    )
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return len(payload)


def _atomic_write_private_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        _write_new_json(temporary, value)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stable_file_sha256(path: Path) -> tuple[str, int]:
    before = os.lstat(path)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise ValueError("reusable preflight payload must be regular data")
    digest = hashlib.sha256()
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    if _stat_tuple(before) != _stat_tuple(opened) or _stat_tuple(
        before
    ) != _stat_tuple(after):
        raise RuntimeError(
            "reusable preflight payload changed during authentication"
        )
    return digest.hexdigest(), int(before.st_size)


def _stat_tuple(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _tree_stat_inventory(root: Path) -> list[dict[str, Any]]:
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
    ):
        raise ValueError(
            "reusable preflight inventory requires one canonical root"
        )
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    paths = (
        root,
        *sorted(
            root.rglob("*"),
            key=lambda value: value.relative_to(root).as_posix(),
        ),
    )
    for path in paths:
        state = os.lstat(path)
        if stat.S_ISDIR(state.st_mode):
            kind = "directory"
        elif stat.S_ISREG(state.st_mode) and int(state.st_nlink) == 1:
            kind = "file"
        else:
            raise ValueError(
                "reusable preflight tree contains a link or special entry"
            )
        inode = (int(state.st_dev), int(state.st_ino))
        if inode in seen:
            raise ValueError(
                "reusable preflight tree contains an inode alias"
            )
        seen.add(inode)
        output.append(
            {
                "relative_path": (
                    "."
                    if path == root
                    else path.relative_to(root).as_posix()
                ),
                "kind": kind,
                "device": int(state.st_dev),
                "inode": int(state.st_ino),
                "mode": int(state.st_mode),
                "link_count": int(state.st_nlink),
                "uid": int(state.st_uid),
                "gid": int(state.st_gid),
                "size_bytes": int(state.st_size),
                "mtime_ns": int(state.st_mtime_ns),
                "ctime_ns": int(state.st_ctime_ns),
            }
        )
    return output


def _registered_path(root: Path, relative: Any) -> Path:
    pure = PurePosixPath(str(relative))
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(
            "reusable preflight registration path is unsafe"
        )
    path = root / Path(*pure.parts)
    if path.is_symlink():
        raise ValueError(
            "reusable preflight registration is symlinked"
        )
    return path


def _manifest_body(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    if value.get("content_sha256") != _sha256_json(body):
        raise ValueError(
            "reusable preflight manifest content identity changed"
        )
    return body


def _file_registration(
    path: Path,
    *,
    root: Path,
    kind: str,
) -> dict[str, Any]:
    digest, size = _stable_file_sha256(path)
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "kind": str(kind),
        "sha256": digest,
        "size_bytes": size,
    }


def _proof_root(
    *,
    store_root: Path,
    artifact_kind: str,
    scientific_key: str,
) -> Path:
    return (
        store_root
        / "authentication_proofs"
        / artifact_kind
        / scientific_key
    )


def _protected_proof_parent(
    *,
    store_root: Path,
    artifact_kind: str,
    create: bool,
) -> Path | None:
    """Return a private, owner-controlled proof parent.

    A mode-0700 receipt directory is insufficient if an untrusted user can
    replace one of its writable ancestors.  The reusable store itself may be
    world-readable, but it and both proof-specific ancestors must be owned by
    the current uid and must not be writable by group or other users.  The
    two proof ancestors are additionally fixed to mode 0700.
    """

    kind = str(artifact_kind).strip()
    if (
        not kind
        or kind in {".", ".."}
        or "/" in kind
        or os.sep in kind
    ):
        raise ValueError("authentication-proof artifact kind is unsafe")
    proof_base = store_root / "authentication_proofs"
    kind_root = proof_base / kind
    if create:
        proof_base.mkdir(mode=0o700, exist_ok=True)
        kind_root.mkdir(mode=0o700, exist_ok=True)
    elif not proof_base.exists() or not kind_root.exists():
        return None
    for index, path in enumerate(
        (store_root, proof_base, kind_root)
    ):
        state = os.lstat(path)
        mode = stat.S_IMODE(state.st_mode)
        if (
            stat.S_ISLNK(state.st_mode)
            or not stat.S_ISDIR(state.st_mode)
            or int(state.st_uid) != os.getuid()
        ):
            raise ValueError(
                "reusable preflight authentication-proof ancestry "
                "is not protected"
            )
        if index == 0 and mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(
                "reusable preflight store is writable by another "
                "principal"
            )
        if index > 0 and mode != 0o700:
            if not create:
                raise ValueError(
                    "reusable preflight authentication-proof ancestry "
                    "has nonprivate permissions"
                )
            os.chmod(path, 0o700)
            repaired = os.lstat(path)
            if (
                stat.S_ISLNK(repaired.st_mode)
                or not stat.S_ISDIR(repaired.st_mode)
                or int(repaired.st_uid) != os.getuid()
                or stat.S_IMODE(repaired.st_mode) != 0o700
            ):
                raise ValueError(
                    "reusable preflight authentication-proof ancestry "
                    "could not be protected"
                )
    return kind_root


def _proof_path(
    *,
    store_root: Path,
    artifact_kind: str,
    scientific_key: str,
) -> Path:
    return (
        _proof_root(
            store_root=store_root,
            artifact_kind=artifact_kind,
            scientific_key=scientific_key,
        )
        / "authentication_proof.json"
    )


def _proof_terminal_path(
    *,
    store_root: Path,
    artifact_kind: str,
    scientific_key: str,
) -> Path:
    return (
        _proof_root(
            store_root=store_root,
            artifact_kind=artifact_kind,
            scientific_key=scientific_key,
        )
        / "authentication_receipt_terminal.json"
    )


def _authentication_probe_ctime_ns(store_root: Path) -> int:
    """Create one protected same-filesystem timestamp probe."""

    parent = store_root / "authentication_timestamp_probes"
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    state = os.lstat(parent)
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISDIR(state.st_mode)
        or int(state.st_uid) != os.getuid()
    ):
        raise ValueError(
            "reusable preflight timestamp-probe root is not protected"
        )
    if stat.S_IMODE(state.st_mode) != 0o700:
        os.chmod(parent, 0o700)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".probe-",
        dir=parent,
    )
    path = Path(raw_path)
    try:
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        os.write(descriptor, b"reusable-preflight-stat-barrier\n")
        os.fsync(descriptor)
        probe = os.fstat(descriptor)
        if (
            not stat.S_ISREG(probe.st_mode)
            or int(probe.st_nlink) != 1
            or int(probe.st_uid) != os.getuid()
        ):
            raise ValueError(
                "reusable preflight timestamp probe is invalid"
            )
        return int(probe.st_ctime_ns)
    finally:
        os.close(descriptor)
        try:
            path.unlink()
        finally:
            _fsync_directory(parent)


def _wait_for_timestamp_barrier(
    *,
    store_root: Path,
    after_ctime_ns: int,
    timeout_seconds: float = 10.0,
) -> int:
    """Wait only at first trust establishment for a distinguishable tick."""

    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        observed = _authentication_probe_ctime_ns(store_root)
        if observed > int(after_ctime_ns):
            return observed
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "filesystem timestamps did not advance enough to establish "
                "a reusable-preflight stat-continuity proof"
            )
        time.sleep(0.01)


def _proof_stat(path: Path) -> dict[str, int]:
    state = os.lstat(path)
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISREG(state.st_mode)
        or int(state.st_nlink) != 1
        or int(state.st_uid) != os.getuid()
        or stat.S_IMODE(state.st_mode) != _PRIVATE_FILE_MODE
    ):
        raise ValueError(
            "reusable preflight authentication proof is not protected"
        )
    return {
        "device": int(state.st_dev),
        "inode": int(state.st_ino),
        "mode": int(state.st_mode),
        "link_count": int(state.st_nlink),
        "uid": int(state.st_uid),
        "gid": int(state.st_gid),
        "size_bytes": int(state.st_size),
        "mtime_ns": int(state.st_mtime_ns),
        "ctime_ns": int(state.st_ctime_ns),
    }


def _validate_proof_receipt_tree(root: Path) -> None:
    state = os.lstat(root)
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISDIR(state.st_mode)
        or int(state.st_uid) != os.getuid()
        or stat.S_IMODE(state.st_mode) != 0o700
    ):
        raise ValueError(
            "reusable preflight authentication receipt is not protected"
        )
    observed: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        child = os.lstat(path)
        if (
            stat.S_ISLNK(child.st_mode)
            or not stat.S_ISREG(child.st_mode)
            or int(child.st_nlink) != 1
            or int(child.st_uid) != os.getuid()
            or stat.S_IMODE(child.st_mode) != _PRIVATE_FILE_MODE
        ):
            raise ValueError(
                "reusable preflight authentication receipt contains "
                "an unprotected entry"
            )
        observed.add(relative)
    if observed != {
        "authentication_proof.json",
        "authentication_receipt_terminal.json",
    }:
        raise ValueError(
            "reusable preflight authentication receipt tree is not closed"
        )


def _load_fast_proof(
    *,
    store_root: Path,
    artifact_kind: str,
    scientific_key: str,
    artifact_root: Path,
    terminal_content_sha256: str,
    producer_identity: str,
    schema_identity: str,
) -> tuple[dict[str, Any], float] | None:
    started = time.perf_counter()
    try:
        receipt_parent = _protected_proof_parent(
            store_root=store_root,
            artifact_kind=artifact_kind,
            create=False,
        )
    except (OSError, ValueError):
        return None
    if receipt_parent is None:
        return None
    receipt_root = _proof_root(
        store_root=store_root,
        artifact_kind=artifact_kind,
        scientific_key=scientific_key,
    )
    path = receipt_root / "authentication_proof.json"
    terminal_path = receipt_root / "authentication_receipt_terminal.json"
    if (
        not receipt_root.is_dir()
        or receipt_root.is_symlink()
        or not path.is_file()
        or path.is_symlink()
        or not terminal_path.is_file()
        or terminal_path.is_symlink()
    ):
        return None
    try:
        _validate_proof_receipt_tree(receipt_root)
        receipt_terminal_sha, receipt_terminal_size = (
            _stable_file_sha256(terminal_path)
        )
        receipt_terminal = _read_json(
            terminal_path,
            label="reusable preflight authentication receipt terminal",
        )
        _manifest_body(receipt_terminal)
        proof_registration = receipt_terminal.get(
            "authentication_proof_registration"
        )
        if (
            set(receipt_terminal)
            != {
                "schema_version",
                "status",
                "artifact_kind",
                "scientific_key",
                "artifact_terminal_content_sha256",
                "artifact_scientific_content_sha256",
                "producer_identity",
                "schema_identity",
                "authentication_proof_registration",
                "ordinary_full_byte_authentication_completed",
                "terminal_published_after_proof",
                "timestamp_stabilization_barrier_completed",
                "proof_barrier_probe_ctime_ns",
                "content_sha256",
            }
            or receipt_terminal.get("schema_version")
            != REUSABLE_AUTH_RECEIPT_TERMINAL_SCHEMA
            or receipt_terminal.get("status") != "complete"
            or receipt_terminal.get("artifact_kind") != artifact_kind
            or receipt_terminal.get("scientific_key") != scientific_key
            or receipt_terminal.get(
                "artifact_terminal_content_sha256"
            )
            != terminal_content_sha256
            or receipt_terminal.get("producer_identity")
            != str(producer_identity)
            or receipt_terminal.get("schema_identity")
            != str(schema_identity)
            or receipt_terminal.get(
                "ordinary_full_byte_authentication_completed"
            )
            is not True
            or receipt_terminal.get("terminal_published_after_proof")
            is not True
            or receipt_terminal.get(
                "timestamp_stabilization_barrier_completed"
            )
            is not True
            or not isinstance(
                receipt_terminal.get(
                    "proof_barrier_probe_ctime_ns"
                ),
                int,
            )
            or not isinstance(proof_registration, Mapping)
            or set(proof_registration)
            != {
                "relative_path",
                "sha256",
                "size_bytes",
                "stat_identity",
            }
            or proof_registration.get("relative_path")
            != "authentication_proof.json"
        ):
            return None
        proof_state = _proof_stat(path)
        if (
            proof_state != proof_registration.get("stat_identity")
            or proof_state["mtime_ns"] != proof_state["ctime_ns"]
        ):
            return None
        proof_sha, proof_size = _stable_file_sha256(path)
        if (
            proof_sha != proof_registration.get("sha256")
            or proof_size != proof_registration.get("size_bytes")
        ):
            return None
        proof = _read_json(
            path,
            label="reusable preflight authentication proof",
        )
        body = _manifest_body(proof)
        inventory = proof.get("tree_stat_inventory")
        if (
            set(proof)
            != {
                "schema_version",
                "artifact_kind",
                "scientific_key",
                "terminal_content_sha256",
                "artifact_scientific_content_sha256",
                "producer_identity",
                "schema_identity",
                "tree_stat_inventory",
                "tree_stat_inventory_content_sha256",
                "ordinary_full_byte_authentication_completed",
                "proof_is_operational_not_scientific",
                "path_in_scientific_identity",
                "mtime_only_trust_used",
                "timestamp_stabilization_protocol",
                "full_authentication_start_probe_ctime_ns",
                "artifact_barrier_probe_ctime_ns",
                "content_sha256",
            }
            or proof.get("schema_version")
            != REUSABLE_AUTH_PROOF_SCHEMA
            or proof.get("artifact_kind") != artifact_kind
            or proof.get("scientific_key") != scientific_key
            or proof.get("terminal_content_sha256")
            != terminal_content_sha256
            or proof.get("producer_identity") != str(producer_identity)
            or proof.get("schema_identity") != str(schema_identity)
            or proof.get("artifact_scientific_content_sha256")
            != receipt_terminal.get(
                "artifact_scientific_content_sha256"
            )
            or not isinstance(inventory, list)
            or proof.get("tree_stat_inventory_content_sha256")
            != _sha256_json(inventory)
            or proof.get(
                "ordinary_full_byte_authentication_completed"
            )
            is not True
            or proof.get("proof_is_operational_not_scientific")
            is not True
            or proof.get("path_in_scientific_identity") is not False
            or proof.get("mtime_only_trust_used") is not False
            or proof.get("timestamp_stabilization_protocol")
            != "same_filesystem_probe_then_vulnerable_byte_recheck_v1"
            or not isinstance(
                proof.get(
                    "full_authentication_start_probe_ctime_ns"
                ),
                int,
            )
            or not isinstance(
                proof.get("artifact_barrier_probe_ctime_ns"),
                int,
            )
            or int(proof["artifact_barrier_probe_ctime_ns"])
            <= int(
                proof[
                    "full_authentication_start_probe_ctime_ns"
                ]
            )
            or _tree_stat_inventory(artifact_root) != inventory
        ):
            return None
        _require_sha256(
            proof.get("artifact_scientific_content_sha256"),
            label="reusable preflight proof scientific content",
        )
        # The protected proof itself must remain one exact inode throughout
        # this check.  It is intentionally not part of the scientific tree.
        if _proof_stat(path) != proof_state:
            raise RuntimeError(
                "reusable preflight proof changed during authentication"
            )
        if _stable_file_sha256(terminal_path) != (
            receipt_terminal_sha,
            receipt_terminal_size,
        ):
            raise RuntimeError(
                "reusable preflight authentication receipt terminal "
                "changed during authentication"
            )
        all_ctimes = [
            int(row["ctime_ns"]) for row in inventory
        ]
        all_ctimes.extend(
            (
                int(proof_state["ctime_ns"]),
                int(os.lstat(terminal_path).st_ctime_ns),
                int(
                    receipt_terminal[
                        "proof_barrier_probe_ctime_ns"
                    ]
                ),
            )
        )
        if _authentication_probe_ctime_ns(store_root) <= max(
            all_ctimes
        ):
            return None
        if (
            _tree_stat_inventory(artifact_root) != inventory
            or _proof_stat(path) != proof_state
        ):
            return None
        return proof, time.perf_counter() - started
    except (OSError, TypeError, ValueError, RuntimeError):
        return None


def _publish_full_auth_proof(
    *,
    store_root: Path,
    artifact_kind: str,
    scientific_key: str,
    artifact_root: Path,
    terminal_content_sha256: str,
    artifact_scientific_content_sha256: str,
    producer_identity: str,
    schema_identity: str,
    full_authentication_start_probe_ctime_ns: int,
    authenticated_byte_inventory: Mapping[
        str, tuple[str, int]
    ],
) -> dict[str, Any]:
    inventory = _tree_stat_inventory(artifact_root)
    expected_files = {
        str(row["relative_path"])
        for row in inventory
        if row["kind"] == "file"
    }
    if set(authenticated_byte_inventory) != expected_files:
        raise ValueError(
            "full authentication omitted a reusable-preflight payload"
        )
    vulnerable_files = [
        row
        for row in inventory
        if row["kind"] == "file"
        and int(row["ctime_ns"])
        >= int(full_authentication_start_probe_ctime_ns)
    ]
    artifact_barrier_probe_ctime_ns = _wait_for_timestamp_barrier(
        store_root=store_root,
        after_ctime_ns=max(
            int(full_authentication_start_probe_ctime_ns),
            *(int(row["ctime_ns"]) for row in inventory),
        ),
    )
    for row in vulnerable_files:
        relative = str(row["relative_path"])
        observed = _stable_file_sha256(
            artifact_root / relative
        )
        if observed != tuple(
            authenticated_byte_inventory[relative]
        ):
            raise RuntimeError(
                "reusable preflight payload changed within the "
                "filesystem timestamp-granularity window"
            )
    if _tree_stat_inventory(artifact_root) != inventory:
        raise RuntimeError(
            "reusable preflight tree changed across its timestamp barrier"
        )
    body = {
        "schema_version": REUSABLE_AUTH_PROOF_SCHEMA,
        "artifact_kind": artifact_kind,
        "scientific_key": scientific_key,
        "terminal_content_sha256": _require_sha256(
            terminal_content_sha256,
            label="terminal content",
        ),
        "artifact_scientific_content_sha256": _require_sha256(
            artifact_scientific_content_sha256,
            label="artifact scientific content",
        ),
        "producer_identity": str(producer_identity),
        "schema_identity": str(schema_identity),
        "tree_stat_inventory": inventory,
        "tree_stat_inventory_content_sha256": _sha256_json(inventory),
        "ordinary_full_byte_authentication_completed": True,
        "proof_is_operational_not_scientific": True,
        "path_in_scientific_identity": False,
        "mtime_only_trust_used": False,
        "timestamp_stabilization_protocol": (
            "same_filesystem_probe_then_vulnerable_byte_recheck_v1"
        ),
        "full_authentication_start_probe_ctime_ns": int(
            full_authentication_start_probe_ctime_ns
        ),
        "artifact_barrier_probe_ctime_ns": int(
            artifact_barrier_probe_ctime_ns
        ),
    }
    proof = {**body, "content_sha256": _sha256_json(body)}
    receipt_parent = _protected_proof_parent(
        store_root=store_root,
        artifact_kind=artifact_kind,
        create=True,
    )
    assert receipt_parent is not None
    target = receipt_parent / scientific_key
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{scientific_key}.receipt-attempt-",
            dir=receipt_parent,
        )
    )
    published = False
    try:
        proof_path = temporary / "authentication_proof.json"
        _write_new_json(proof_path, proof)
        proof_sha, proof_size = _stable_file_sha256(proof_path)
        proof_state = _proof_stat(proof_path)
        if proof_state["mtime_ns"] != proof_state["ctime_ns"]:
            raise RuntimeError(
                "new reusable preflight proof lacks exact stat continuity"
            )
        terminal_body = {
            "schema_version": REUSABLE_AUTH_RECEIPT_TERMINAL_SCHEMA,
            "status": "complete",
            "artifact_kind": artifact_kind,
            "scientific_key": scientific_key,
            "artifact_terminal_content_sha256": _require_sha256(
                terminal_content_sha256,
                label="authentication receipt artifact terminal",
            ),
            "artifact_scientific_content_sha256": _require_sha256(
                artifact_scientific_content_sha256,
                label="authentication receipt artifact scientific content",
            ),
            "producer_identity": str(producer_identity),
            "schema_identity": str(schema_identity),
            "authentication_proof_registration": {
                "relative_path": "authentication_proof.json",
                "sha256": proof_sha,
                "size_bytes": proof_size,
                "stat_identity": proof_state,
            },
            "ordinary_full_byte_authentication_completed": True,
            "terminal_published_after_proof": True,
            "timestamp_stabilization_barrier_completed": True,
            # Filled with the lower artifact barrier here; the load path also
            # requires a fresh probe beyond the actual proof-tree ctimes.
            "proof_barrier_probe_ctime_ns": int(
                artifact_barrier_probe_ctime_ns
            ),
        }
        _write_new_json(
            temporary / "authentication_receipt_terminal.json",
            {
                **terminal_body,
                "content_sha256": _sha256_json(terminal_body),
            },
        )
        _fsync_directory(temporary)
        if target.exists() or target.is_symlink():
            recovery = (
                store_root
                / "recovery"
                / "authentication_proofs"
                / artifact_kind
            )
            recovery.mkdir(parents=True, exist_ok=True)
            os.rename(
                target,
                recovery
                / (
                    f"{scientific_key}.superseded-"
                    f"{time.time_ns()}-{os.getpid()}"
                ),
            )
            _fsync_directory(recovery)
        os.rename(temporary, target)
        _fsync_directory(receipt_parent)
        published = True
        target_inventory = _tree_stat_inventory(target.resolve(strict=True))
        proof_barrier_probe_ctime_ns = _wait_for_timestamp_barrier(
            store_root=store_root,
            after_ctime_ns=max(
                *(int(row["ctime_ns"]) for row in inventory),
                *(
                    int(row["ctime_ns"])
                    for row in target_inventory
                ),
            ),
        )
        if (
            _tree_stat_inventory(artifact_root) != inventory
            or _tree_stat_inventory(target.resolve(strict=True))
            != target_inventory
            or _stable_file_sha256(
                target / "authentication_proof.json"
            )
            != (proof_sha, proof_size)
        ):
            raise RuntimeError(
                "reusable preflight proof changed across its timestamp "
                "stabilization barrier"
            )
        receipt = _read_json(
            target / "authentication_receipt_terminal.json",
            label="new authentication receipt terminal",
        )
        if (
            receipt.get("content_sha256")
            != _sha256_json(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "content_sha256"
                }
            )
            or proof_barrier_probe_ctime_ns
            <= max(
                int(row["ctime_ns"])
                for row in target_inventory
            )
        ):
            raise RuntimeError(
                "reusable preflight proof barrier did not stabilize"
            )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        if published and target.exists() and not target.is_symlink():
            recovery = (
                store_root
                / "recovery"
                / "authentication_proofs"
                / artifact_kind
            )
            recovery.mkdir(parents=True, exist_ok=True)
            os.rename(
                target,
                recovery
                / (
                    f"{scientific_key}.ineligible-"
                    f"{time.time_ns()}-{os.getpid()}"
                ),
            )
            _fsync_directory(recovery)
        raise
    return proof


def _publish_optional_full_auth_proof(
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Best-effort fast proof after authoritative full-byte validation.

    Failure to establish an inode/timestamp proof is not scientific
    invalidity.  Reauthenticate the complete closed byte inventory once more;
    accept the deep result without a shortcut only when every byte still
    matches.  Actual mutation, replacement, linking, or tree changes therefore
    continue to propagate to the caller and quarantine path.
    """

    try:
        return _publish_full_auth_proof(**kwargs)
    except (OSError, RuntimeError, TypeError, ValueError) as proof_error:
        artifact_root = Path(kwargs["artifact_root"])
        authenticated = dict(
            kwargs["authenticated_byte_inventory"]
        )
        inventory = _tree_stat_inventory(artifact_root)
        observed_files = {
            str(row["relative_path"])
            for row in inventory
            if row["kind"] == "file"
        }
        if observed_files != set(authenticated):
            raise ValueError(
                "reusable preflight tree changed while its optional "
                "fast-authentication proof was being established"
            ) from proof_error
        for relative, expected in authenticated.items():
            if _stable_file_sha256(
                artifact_root / relative
            ) != tuple(expected):
                raise ValueError(
                    "reusable preflight payload changed while its optional "
                    "fast-authentication proof was being established"
                ) from proof_error
        # Deep bytes remain authoritative. A future reopen may retry proof
        # establishment or simply deep-authenticate again.
        return None


def _close_tree_read_only(root: Path) -> None:
    for path in sorted(
        root.rglob("*"),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        if path.is_file():
            path.chmod(_READ_ONLY_FILE_MODE)
        elif path.is_dir():
            path.chmod(_READ_ONLY_DIRECTORY_MODE)
    root.chmod(_READ_ONLY_DIRECTORY_MODE)


def _validate_registered_tree(
    *,
    root: Path,
    terminal_name: str,
    files: Any,
    read_payload_bytes: bool,
) -> tuple[
    dict[str, Mapping[str, Any]],
    int,
    dict[str, tuple[str, int]],
]:
    if not isinstance(files, list):
        raise ValueError(
            "reusable preflight manifest file inventory is malformed"
        )
    registrations: dict[str, Mapping[str, Any]] = {}
    bytes_read = 0
    authenticated_bytes: dict[str, tuple[str, int]] = {}
    for row in files:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"relative_path", "kind", "sha256", "size_bytes"}
        ):
            raise ValueError(
                "reusable preflight file registration is malformed"
            )
        relative = str(row["relative_path"])
        if relative in registrations:
            raise ValueError(
                "reusable preflight file inventory repeats a path"
            )
        _require_sha256(
            row.get("sha256"),
            label=f"{relative} registered SHA",
        )
        if (
            isinstance(row.get("size_bytes"), bool)
            or not isinstance(row.get("size_bytes"), int)
            or int(row["size_bytes"]) < 1
        ):
            raise ValueError(
                "reusable preflight file registration size is invalid"
            )
        path = _registered_path(root, relative)
        state = os.lstat(path)
        if (
            not stat.S_ISREG(state.st_mode)
            or int(state.st_nlink) != 1
            or int(state.st_size) != int(row["size_bytes"])
            or stat.S_IMODE(state.st_mode) != _READ_ONLY_FILE_MODE
        ):
            raise ValueError(
                "reusable preflight registered file metadata changed"
            )
        if read_payload_bytes:
            digest, size = _stable_file_sha256(path)
            bytes_read += size
            if digest != row["sha256"] or size != row["size_bytes"]:
                raise ValueError(
                    "reusable preflight registered bytes changed"
                )
            authenticated_bytes[relative] = (digest, size)
        registrations[relative] = row
    expected_files = {terminal_name, *registrations}
    observed_files: set[str] = set()
    for path in root.rglob("*"):
        state = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(state.st_mode):
            if stat.S_IMODE(state.st_mode) != _READ_ONLY_DIRECTORY_MODE:
                raise ValueError(
                    "reusable preflight directory permissions changed"
                )
        elif stat.S_ISREG(state.st_mode) and int(state.st_nlink) == 1:
            if stat.S_IMODE(state.st_mode) != _READ_ONLY_FILE_MODE:
                raise ValueError(
                    "reusable preflight file permissions changed"
                )
            observed_files.add(relative)
        else:
            raise ValueError(
                "reusable preflight tree contains a link or special entry"
            )
    if observed_files != expected_files:
        raise ValueError(
            "reusable preflight artifact tree is not closed"
        )
    return registrations, bytes_read, authenticated_bytes


def _store_root(root: Path | str) -> Path:
    supplied = Path(root)
    if not supplied.is_absolute():
        raise ValueError(
            "reusable preflight store root must be absolute"
        )
    supplied.mkdir(parents=True, exist_ok=True)
    resolved = supplied.resolve(strict=True)
    if supplied != resolved or supplied.is_symlink():
        raise ValueError(
            "reusable preflight store root must be canonical"
        )
    manifest_path = resolved / "store_manifest.json"
    body = {
        "schema_version": REUSABLE_PREFLIGHT_STORE_SCHEMA,
        "artifact_directories": [
            "global_audits",
            "owner_artifacts",
            "assembled_contexts",
            "accepted_contexts",
            "recovery",
        ],
        "authentication_proof_directory": "authentication_proofs",
        "operational_paths_in_scientific_keys": False,
        "resource_assignments_in_scientific_keys": False,
        "stage2_identity_in_scientific_keys": False,
    }
    manifest = {**body, "content_sha256": _sha256_json(body)}
    if manifest_path.is_file() and not manifest_path.is_symlink():
        if _read_json(
            manifest_path,
            label="reusable preflight store manifest",
        ) != manifest:
            raise ValueError(
                "reusable preflight store manifest changed"
            )
    elif manifest_path.exists() or manifest_path.is_symlink():
        raise ValueError(
            "reusable preflight store manifest path is invalid"
        )
    else:
        _write_new_json(manifest_path, manifest)
    for name in (
        "global_audits",
        "owner_artifacts",
        "assembled_contexts",
        "accepted_contexts",
        "authentication_proofs",
        "recovery",
    ):
        (resolved / name).mkdir(exist_ok=True)
    proof_root = resolved / "authentication_proofs"
    proof_state = os.lstat(proof_root)
    if (
        stat.S_ISLNK(proof_state.st_mode)
        or not stat.S_ISDIR(proof_state.st_mode)
        or int(proof_state.st_uid) != os.getuid()
    ):
        raise ValueError(
            "reusable preflight authentication-proof root is invalid"
        )
    if stat.S_IMODE(proof_state.st_mode) != 0o700:
        os.chmod(proof_root, 0o700)
    return resolved


def _quarantine_invalid_artifact(
    *,
    store_root: Path | str,
    artifact_directory: str,
    scientific_key_value: str,
    failure: BaseException,
) -> None:
    """Preserve an invalid canonical artifact and make recomputation possible.

    The move is atomic within the reusable store.  No payload is deleted, and
    an incomplete attempt can never become canonical merely by being placed in
    recovery.  A concurrent process that already moved the same target is
    treated as having completed the recovery action.
    """

    store = _store_root(store_root)
    if artifact_directory not in {
        "global_audits",
        "owner_artifacts",
        "assembled_contexts",
        "accepted_contexts",
    }:
        raise ValueError("reusable preflight recovery kind is invalid")
    key = _require_sha256(
        scientific_key_value,
        label="reusable preflight recovery scientific key",
    )
    target = store / artifact_directory / key
    if not target.exists() and not target.is_symlink():
        return
    recovery_root = store / "recovery" / artifact_directory
    recovery_root.mkdir(parents=True, exist_ok=True)
    destination = recovery_root / (
        f"{key}.invalid-{time.time_ns()}-{os.getpid()}"
    )
    original_mode: int | None = None
    try:
        target_stat = os.lstat(target)
        if stat.S_ISLNK(target_stat.st_mode) or not stat.S_ISDIR(
            target_stat.st_mode
        ):
            raise ValueError(
                "invalid reusable-preflight target is not a real directory"
            )
        original_mode = stat.S_IMODE(target_stat.st_mode)
        os.chmod(target, original_mode | stat.S_IWUSR)
        os.rename(target, destination)
        os.chmod(destination, original_mode)
    except FileNotFoundError:
        return
    except BaseException:
        if (
            original_mode is not None
            and target.exists()
            and not target.is_symlink()
        ):
            try:
                os.chmod(target, original_mode)
            except OSError:
                pass
        raise
    _fsync_directory(target.parent)
    _fsync_directory(recovery_root)
    body = {
        "schema_version": (
            "production_stage1_reusable_preflight_recovery_record_v1"
        ),
        "artifact_directory": artifact_directory,
        "scientific_key": key,
        "recovered_entry_name": destination.name,
        "failure_type": type(failure).__name__,
        "failure_message": str(failure),
        "payload_deleted": False,
        "canonical_completion_status_granted": False,
    }
    _atomic_write_private_json(
        recovery_root / f"{destination.name}.recovery.json",
        {**body, "content_sha256": _sha256_json(body)},
    )


def scientific_key(
    compatibility: Mapping[str, Any],
    *,
    expected_schema: str,
) -> str:
    closed = _json_copy(
        compatibility,
        label="reusable preflight compatibility",
    )
    if (
        not isinstance(closed, dict)
        or closed.get("schema_version") != expected_schema
        or any(
            name in _canonical_json(closed)
            for name in (
                '"hostname"',
                '"device"',
                '"gpu_ids"',
                '"worker_count"',
                '"durable_root"',
                '"scratch_root"',
                '"stage2"',
                '"endpoint"',
            )
        )
    ):
        raise ValueError(
            "reusable preflight compatibility is not narrowly scientific"
        )
    return _sha256_json(closed)


@dataclass(frozen=True)
class ReusableGlobalAuditArtifact:
    root: Path
    terminal: Mapping[str, Any]
    audit: Mapping[str, Any]
    authentication_mode: str
    authentication_seconds: float
    payload_bytes_read: int

    @property
    def scientific_key(self) -> str:
        return str(self.terminal["scientific_key"])

    @property
    def scientific_content_sha256(self) -> str:
        return str(
            self.terminal["artifact_scientific_content_sha256"]
        )


def seal_reusable_global_audit(
    *,
    store_root: Path | str,
    compatibility: Mapping[str, Any],
    audit: Mapping[str, Any],
    row_text_sha256: Sequence[str],
    row_chunk_counts: Sequence[int],
    token_lengths: Sequence[int],
    producer_identity: str,
) -> ReusableGlobalAuditArtifact:
    store = _store_root(store_root)
    key = scientific_key(
        compatibility,
        expected_schema=(
            "production_stage1_global_nontruncation_compatibility_v1"
        ),
    )
    target = store / "global_audits" / key
    if target.is_dir() and not target.is_symlink():
        return load_reusable_global_audit(
            store_root=store,
            compatibility=compatibility,
            producer_identity=producer_identity,
        )
    if target.exists() or target.is_symlink():
        raise ValueError(
            "reusable global audit target is not a directory"
        )
    row_hashes = tuple(map(str, row_text_sha256))
    chunk_counts = np.asarray(row_chunk_counts, dtype=np.int64)
    token_counts = np.asarray(token_lengths, dtype=np.int64)
    if (
        len(row_hashes) < 1
        or len(row_hashes) != len(chunk_counts)
        or any(
            len(value) != 64
            or any(character not in _HEX for character in value)
            for value in row_hashes
        )
        or np.any(chunk_counts < 1)
        or int(chunk_counts.sum()) != len(token_counts)
        or np.any(token_counts < 1)
        or int(audit.get("row_count", -1)) != len(row_hashes)
        or int(audit.get("total_chunks", -1)) != len(token_counts)
        or audit.get("ordered_chunk_counts_sha256")
        != _sha256_json(chunk_counts.tolist())
        or audit.get("ordered_token_counts_sha256")
        != _sha256_json(token_counts.tolist())
    ):
        raise ValueError(
            "global non-truncation inventory differs from its exact audit"
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{key}.attempt-",
            dir=target.parent,
        )
    )
    try:
        audit_path = temporary / "audit.json"
        row_hash_path = temporary / "ordered_row_text_sha256.json"
        chunks_path = temporary / "row_chunk_counts.npy"
        tokens_path = temporary / "chunk_token_lengths.npy"
        _write_new_json(audit_path, dict(audit))
        _write_new_json(
            row_hash_path,
            {
                "schema_version": (
                    "production_stage1_ordered_row_text_sha256_v1"
                ),
                "row_text_sha256": list(row_hashes),
                "content_sha256": _sha256_json(
                    {
                        "schema_version": (
                            "production_stage1_ordered_row_text_sha256_v1"
                        ),
                        "row_text_sha256": list(row_hashes),
                    }
                ),
            },
        )
        with chunks_path.open("xb") as handle:
            np.save(handle, chunk_counts, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        with tokens_path.open("xb") as handle:
            np.save(handle, token_counts, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        files = [
            _file_registration(
                audit_path,
                root=temporary,
                kind="validated_nontruncation_audit",
            ),
            _file_registration(
                row_hash_path,
                root=temporary,
                kind="ordered_row_text_identity",
            ),
            _file_registration(
                chunks_path,
                root=temporary,
                kind="exact_row_chunk_counts",
            ),
            _file_registration(
                tokens_path,
                root=temporary,
                kind="exact_chunk_token_lengths",
            ),
        ]
        scientific_body = {
            "compatibility": _json_copy(
                compatibility,
                label="global audit compatibility",
            ),
            "audit_content_sha256": _require_sha256(
                audit.get("content_sha256"),
                label="global audit",
            ),
            "normalized_text_projection_sha256": _require_sha256(
                audit.get("normalized_text_projection_sha256"),
                label="global audit normalized text projection",
            ),
            "ordered_chunk_counts_sha256": _require_sha256(
                audit.get("ordered_chunk_counts_sha256"),
                label="global audit ordered chunk counts",
            ),
            "ordered_token_lengths_sha256": _require_sha256(
                audit.get("ordered_token_counts_sha256"),
                label="global audit ordered token lengths",
            ),
            "row_count": len(row_hashes),
            "chunk_count": len(token_counts),
        }
        if (
            scientific_body["ordered_chunk_counts_sha256"]
            != _sha256_json(chunk_counts.tolist())
            or scientific_body["ordered_token_lengths_sha256"]
            != _sha256_json(token_counts.tolist())
        ):
            raise ValueError(
                "global audit exact arrays differ from their scientific hashes"
            )
        artifact_scientific = _sha256_json(scientific_body)
        body = {
            "schema_version": REUSABLE_GLOBAL_AUDIT_SCHEMA,
            "status": "complete",
            "scientific_key": key,
            "compatibility": _json_copy(
                compatibility,
                label="global audit compatibility",
            ),
            "producer_identity": str(producer_identity),
            "artifact_scientific_content_sha256": artifact_scientific,
            "audit_content_sha256": audit["content_sha256"],
            "row_count": len(row_hashes),
            "chunk_count": len(token_counts),
            "files": files,
            "every_row_and_chunk_accounted_once": True,
            "chunk_or_token_sampling_used": False,
            "operational_paths_in_scientific_identity": False,
        }
        _write_new_json(
            temporary / GLOBAL_TERMINAL,
            {**body, "content_sha256": _sha256_json(body)},
        )
        _close_tree_read_only(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.chmod(0o700)
            for path in temporary.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_reusable_global_audit(
        store_root=store,
        compatibility=compatibility,
        producer_identity=producer_identity,
    )


def seal_reusable_global_audit_from_authenticated_legacy(
    *,
    store_root: Path | str,
    compatibility: Mapping[str, Any],
    audit: Mapping[str, Any],
    authenticated_source: Mapping[str, Any],
    producer_identity: str,
) -> ReusableGlobalAuditArtifact:
    """Reject hash-only legacy coverage as a complete reusable inventory.

    Older portable preflight checkpoints retained the exact ordered coverage
    hashes and nonbinding proof but not the underlying per-row integer arrays.
    The current schema requires those arrays.  Callers must recompute the
    deterministic tokenizer inventory once, compare its hashes with the
    authenticated legacy audit, and publish through
    :func:`seal_reusable_global_audit`.
    """

    raise ValueError(
        "hash-only legacy HTR coverage cannot be sealed as a complete "
        "reusable global inventory; materialize exact row/chunk arrays first"
    )

    store = _store_root(store_root)
    key = scientific_key(
        compatibility,
        expected_schema=(
            "production_stage1_global_nontruncation_compatibility_v1"
        ),
    )
    target = store / "global_audits" / key
    if target.is_dir() and not target.is_symlink():
        return load_reusable_global_audit(
            store_root=store,
            compatibility=compatibility,
            producer_identity=producer_identity,
        )
    if target.exists() or target.is_symlink():
        raise ValueError(
            "reusable global audit target is not a directory"
        )
    accepted_audit = _json_copy(
        audit,
        label="authenticated legacy HTR non-truncation audit",
    )
    if "content_sha256" not in accepted_audit:
        accepted_audit["content_sha256"] = _sha256_json(
            accepted_audit
        )
    audit_body = {
        name: copy.deepcopy(value)
        for name, value in accepted_audit.items()
        if name != "content_sha256"
    }
    if (
        accepted_audit.get("content_sha256")
        != _sha256_json(audit_body)
        or int(accepted_audit.get("row_count", -1)) < 1
        or int(accepted_audit.get("total_chunks", -1)) < 1
        or accepted_audit.get("chunk_cap_nonbinding") is not True
        or accepted_audit.get(
            "all_chunks_within_effective_max_length"
        )
        is not True
        or accepted_audit.get("semantic_truncation_allowed")
        is not False
        or accepted_audit.get("tokenizer_truncation_allowed")
        is not False
    ):
        raise ValueError(
            "legacy global audit is not an exact non-truncation proof"
        )
    for name in (
        "normalized_text_projection_sha256",
        "ordered_chunk_counts_sha256",
        "ordered_token_counts_sha256",
        "htr_model_tree_sha256",
    ):
        _require_sha256(
            accepted_audit.get(name),
            label=f"legacy global audit {name}",
        )
    source = _json_copy(
        authenticated_source,
        label="authenticated legacy preflight source",
    )
    required_source = {
        "source_kind",
        "manifest_path",
        "manifest_sha256",
        "manifest_content_sha256",
        "artifact_scientific_content_sha256",
        "payload_bytes_deeply_authenticated",
        "kmeans_or_svd_refit_performed",
        "htr_retokenization_performed",
    }
    if (
        not isinstance(source, dict)
        or set(source) != required_source
        or source.get("source_kind")
        != "portable_cluster_preflight_artifact_v2"
        or not Path(str(source.get("manifest_path"))).is_absolute()
        or source.get("payload_bytes_deeply_authenticated") is not True
        or source.get("kmeans_or_svd_refit_performed") is not False
        or source.get("htr_retokenization_performed") is not False
    ):
        raise ValueError(
            "legacy global audit lacks authenticated source provenance"
        )
    for name in (
        "manifest_sha256",
        "manifest_content_sha256",
        "artifact_scientific_content_sha256",
    ):
        _require_sha256(
            source.get(name),
            label=f"legacy preflight source {name}",
        )
    source_scientific = {
        name: copy.deepcopy(source[name])
        for name in (
            "source_kind",
            "manifest_sha256",
            "manifest_content_sha256",
            "artifact_scientific_content_sha256",
            "payload_bytes_deeply_authenticated",
            "kmeans_or_svd_refit_performed",
            "htr_retokenization_performed",
        )
    }
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{key}.attempt-",
            dir=target.parent,
        )
    )
    try:
        audit_path = temporary / "audit.json"
        provenance_path = temporary / "legacy_exact_inventory_proof.json"
        _write_new_json(audit_path, accepted_audit)
        provenance_body = {
            "schema_version": (
                "production_stage1_legacy_exact_nontruncation_adoption_v2"
            ),
            "authenticated_source_scientific_identity": (
                source_scientific
            ),
            "source_path_persisted": False,
            "ordered_per_note_arrays_persisted_by_legacy_source": False,
            "ordered_per_note_coverage_hashes_persisted": True,
            "source_exact_validator_accepted_all_rows_and_chunks": True,
            "scientific_refit_or_retokenization_required": False,
        }
        _write_new_json(
            provenance_path,
            {
                **provenance_body,
                "content_sha256": _sha256_json(provenance_body),
            },
        )
        files = [
            _file_registration(
                audit_path,
                root=temporary,
                kind="validated_nontruncation_audit",
            ),
            _file_registration(
                provenance_path,
                root=temporary,
                kind="authenticated_legacy_exact_inventory_proof",
            ),
        ]
        scientific_body = {
            "compatibility": _json_copy(
                compatibility,
                label="global audit compatibility",
            ),
            "audit_content_sha256": accepted_audit["content_sha256"],
            "normalized_text_projection_sha256": (
                accepted_audit[
                    "normalized_text_projection_sha256"
                ]
            ),
            "ordered_chunk_counts_sha256": (
                accepted_audit["ordered_chunk_counts_sha256"]
            ),
            "ordered_token_lengths_sha256": (
                accepted_audit["ordered_token_counts_sha256"]
            ),
            "row_count": int(accepted_audit["row_count"]),
            "chunk_count": int(accepted_audit["total_chunks"]),
        }
        body = {
            "schema_version": REUSABLE_GLOBAL_AUDIT_SCHEMA,
            "status": "complete",
            "scientific_key": key,
            "compatibility": _json_copy(
                compatibility,
                label="global audit compatibility",
            ),
            "producer_identity": str(producer_identity),
            "artifact_scientific_content_sha256": _sha256_json(
                scientific_body
            ),
            "audit_content_sha256": accepted_audit["content_sha256"],
            "row_count": int(accepted_audit["row_count"]),
            "chunk_count": int(accepted_audit["total_chunks"]),
            "files": files,
            "every_row_and_chunk_accounted_once": True,
            "chunk_or_token_sampling_used": False,
            "exact_inventory_persistence_mode": (
                "authenticated_legacy_ordered_hashes_v1"
            ),
            "htr_retokenization_performed_for_adoption": False,
            "operational_paths_in_scientific_identity": False,
        }
        _write_new_json(
            temporary / GLOBAL_TERMINAL,
            {**body, "content_sha256": _sha256_json(body)},
        )
        _close_tree_read_only(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.chmod(0o700)
            for path in temporary.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_reusable_global_audit(
        store_root=store,
        compatibility=compatibility,
        producer_identity=producer_identity,
    )


def load_reusable_global_audit(
    *,
    store_root: Path | str,
    compatibility: Mapping[str, Any],
    producer_identity: str,
) -> ReusableGlobalAuditArtifact:
    store = _store_root(store_root)
    key = scientific_key(
        compatibility,
        expected_schema=(
            "production_stage1_global_nontruncation_compatibility_v1"
        ),
    )
    root = (store / "global_audits" / key).resolve(strict=True)
    terminal_path = root / GLOBAL_TERMINAL
    full_authentication_start_probe_ctime_ns = (
        _authentication_probe_ctime_ns(store)
    )
    terminal_digest, terminal_size = _stable_file_sha256(terminal_path)
    terminal = _read_json(
        terminal_path,
        label="reusable global audit terminal",
    )
    _manifest_body(terminal)
    if (
        terminal.get("schema_version") != REUSABLE_GLOBAL_AUDIT_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("scientific_key") != key
        or terminal.get("compatibility")
        != _json_copy(
            compatibility,
            label="global audit compatibility",
        )
        or terminal.get("producer_identity") != str(producer_identity)
        or terminal.get("every_row_and_chunk_accounted_once") is not True
        or terminal.get("chunk_or_token_sampling_used") is not False
        or terminal.get(
            "operational_paths_in_scientific_identity"
        )
        is not False
    ):
        raise ValueError(
            "reusable global audit terminal is incompatible"
        )
    _require_sha256(
        terminal.get("artifact_scientific_content_sha256"),
        label="global audit scientific content",
    )
    proof = _load_fast_proof(
        store_root=store,
        artifact_kind="global_audit",
        scientific_key=key,
        artifact_root=root,
        terminal_content_sha256=terminal["content_sha256"],
        producer_identity=str(producer_identity),
        schema_identity=REUSABLE_GLOBAL_AUDIT_SCHEMA,
    )
    if proof is not None:
        registrations, _, _authenticated = _validate_registered_tree(
            root=root,
            terminal_name=GLOBAL_TERMINAL,
            files=terminal["files"],
            read_payload_bytes=False,
        )
        authentication_mode = "prior_proof_stat_continuity"
        authentication_seconds = proof[1]
        bytes_read = terminal_size
    else:
        started = time.perf_counter()
        (
            registrations,
            payload_read,
            authenticated_bytes,
        ) = _validate_registered_tree(
            root=root,
            terminal_name=GLOBAL_TERMINAL,
            files=terminal["files"],
            read_payload_bytes=True,
        )
        authenticated_bytes[GLOBAL_TERMINAL] = (
            terminal_digest,
            terminal_size,
        )
        authentication_seconds = time.perf_counter() - started
        bytes_read = terminal_size + payload_read
        authentication_mode = "full_byte_reauthentication"
    audit_rows = [
        row
        for row in registrations.values()
        if row["kind"] == "validated_nontruncation_audit"
    ]
    if len(audit_rows) != 1:
        raise ValueError(
            "reusable global audit lacks its audit payload"
        )
    audit = _read_json(
        _registered_path(root, audit_rows[0]["relative_path"]),
        label="reusable global non-truncation audit",
    )
    if audit.get("content_sha256") != terminal["audit_content_sha256"]:
        raise ValueError(
            "reusable global audit scientific payload changed"
        )
    if proof is None:
        _publish_optional_full_auth_proof(
            store_root=store,
            artifact_kind="global_audit",
            scientific_key=key,
            artifact_root=root,
            terminal_content_sha256=terminal["content_sha256"],
            artifact_scientific_content_sha256=(
                terminal["artifact_scientific_content_sha256"]
            ),
            producer_identity=str(producer_identity),
            schema_identity=REUSABLE_GLOBAL_AUDIT_SCHEMA,
            full_authentication_start_probe_ctime_ns=(
                full_authentication_start_probe_ctime_ns
            ),
            authenticated_byte_inventory=authenticated_bytes,
        )
    if _stable_file_sha256(terminal_path) != (
        terminal_digest,
        terminal_size,
    ):
        raise RuntimeError(
            "reusable global terminal changed while reopening"
        )
    return ReusableGlobalAuditArtifact(
        root=root,
        terminal=MappingProxyType(copy.deepcopy(terminal)),
        audit=MappingProxyType(copy.deepcopy(audit)),
        authentication_mode=authentication_mode,
        authentication_seconds=authentication_seconds,
        payload_bytes_read=bytes_read,
    )


def try_load_reusable_global_audit(
    *,
    store_root: Path | str,
    compatibility: Mapping[str, Any],
    producer_identity: str,
) -> ReusableGlobalAuditArtifact | None:
    try:
        return load_reusable_global_audit(
            store_root=store_root,
            compatibility=compatibility,
            producer_identity=producer_identity,
        )
    except FileNotFoundError:
        return None
    except (OSError, RuntimeError, ValueError) as exc:
        _quarantine_invalid_artifact(
            store_root=store_root,
            artifact_directory="global_audits",
            scientific_key_value=scientific_key(
                compatibility,
                expected_schema=(
                    "production_stage1_global_nontruncation_compatibility_v1"
                ),
            ),
            failure=exc,
        )
        return None


def owner_compatibility(
    *,
    cluster_compatibility: Mapping[str, Any],
    physical_scope: Mapping[str, Any],
    fit_input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    scope = _json_copy(
        physical_scope,
        label="physical preflight scope",
    )
    fit_rows = scope.get("fit_row_ids")
    if (
        not isinstance(scope, dict)
        or not isinstance(fit_rows, list)
        or not fit_rows
        or len(fit_rows) != len(set(fit_rows))
        or isinstance(scope.get("scope_seed"), bool)
        or not isinstance(scope.get("scope_seed"), int)
    ):
        raise ValueError(
            "owner preflight compatibility scope is invalid"
        )
    binding = _json_copy(
        fit_input_binding,
        label="physical owner fit-input binding",
    )
    required_binding = {
        "schema_version",
        "ordered_fit_modeling_rows_sha256",
        "ordered_fit_embedding_rows_sha256",
        "ordered_fit_row_count",
        "embedding_row_digest_schema_version",
    }
    if (
        not isinstance(binding, dict)
        or set(binding) != required_binding
        or binding.get("schema_version")
        != "production_stage1_owner_fit_input_binding_v2"
        or binding.get("ordered_fit_row_count") != len(fit_rows)
    ):
        raise ValueError(
            "owner preflight fit-input binding is invalid"
        )
    for name in (
        "ordered_fit_modeling_rows_sha256",
        "ordered_fit_embedding_rows_sha256",
    ):
        _require_sha256(
            binding.get(name),
            label=f"owner fit-input {name}",
        )
    heldout_rows = scope.get("heldout_row_ids")
    if (
        not isinstance(heldout_rows, list)
        or len(heldout_rows) != len(set(heldout_rows))
        or set(fit_rows).intersection(heldout_rows)
    ):
        raise ValueError(
            "owner preflight split binding is invalid"
        )
    scope_projection = {
        name: copy.deepcopy(scope[name])
        for name in sorted(scope)
        if name
        not in {
            "device",
            "device_ids",
            "gpu_ids",
            "resource_lease",
            "worker_count",
            "worker_lane",
        }
    }
    return {
        "schema_version": OWNER_COMPATIBILITY_SCHEMA,
        "cluster_compatibility": _json_copy(
            cluster_compatibility,
            label="cluster preflight compatibility",
        ),
        "physical_owner_scope_id": str(scope["scope_id"]),
        "ordered_fit_row_ids": list(map(int, fit_rows)),
        "ordered_fit_row_ids_sha256": _sha256_json(
            list(map(int, fit_rows))
        ),
        "ordered_heldout_row_ids": list(map(int, heldout_rows)),
        "ordered_heldout_row_ids_sha256": _sha256_json(
            list(map(int, heldout_rows))
        ),
        "scope_scientific_projection": scope_projection,
        "scope_scientific_projection_sha256": _sha256_json(
            scope_projection
        ),
        "fit_input_binding": binding,
        "canonical_owner_seed": int(scope["scope_seed"]),
        "outer_fold": int(scope["outer_fold"]),
        "scope_kind": str(scope["scope_kind"]),
    }


def captured_state_from_authenticated_canonical_state(
    *,
    state: Any,
    owner_scope_id: str,
    expected_fit_identity_content_sha256: str,
) -> dict[str, Any]:
    """Transcode one authenticated legacy state without fitting it again.

    The portable-v2 state bundle authenticates metadata eagerly and individual
    arrays on owner access.  This function accepts only that fully materialized
    owner state, reconstructs the original in-memory KMeans/SVD capture, and
    round-trips it through the same normalization validator used by fresh
    preflight fitting.
    """

    owner = str(owner_scope_id)
    expected_fit = _require_sha256(
        expected_fit_identity_content_sha256,
        label="legacy owner fit identity",
    )
    manifest = getattr(state, "manifest", None)
    arrays = getattr(state, "arrays", None)
    if not isinstance(manifest, Mapping) or not isinstance(
        arrays,
        Mapping,
    ):
        raise TypeError(
            "legacy owner adoption requires one authenticated canonical state"
        )
    state_metadata = manifest.get("state_metadata")
    binding = manifest.get("preflight_binding")
    if (
        manifest.get("status") != "complete"
        or manifest.get("physical_owner_scope_id") != owner
        or not isinstance(state_metadata, Mapping)
        or not isinstance(binding, Mapping)
        or binding.get("cluster_fit_identity_sha256") != expected_fit
    ):
        raise ValueError(
            "legacy canonical state belongs to another physical owner"
        )
    try:
        kmeans_state = {
            "fit_row_ids": list(manifest["fit_row_ids"]),
            "parameters": copy.deepcopy(
                state_metadata["kmeans_parameters"]
            ),
            "scientific_configuration": copy.deepcopy(
                state_metadata["cluster_scientific_configuration"]
            ),
            "canonical_group_seed": int(
                state_metadata["canonical_group_seed"]
            ),
            "ordered_fit_row_seed_policy": str(
                state_metadata["ordered_fit_row_seed_policy"]
            ),
            "usable_mask": np.ascontiguousarray(
                arrays["cluster_kmeans_usable_mask"]
            ),
            "cluster_labels": np.ascontiguousarray(
                arrays["cluster_kmeans_labels"]
            ),
            "cluster_centers": np.ascontiguousarray(
                arrays["cluster_kmeans_centers"]
            ),
            "cluster_counts": np.ascontiguousarray(
                arrays["cluster_kmeans_counts"]
            ),
            "n_iter": int(state_metadata["kmeans_n_iter"]),
            "inertia": float.fromhex(
                str(state_metadata["kmeans_inertia_hex"])
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "legacy canonical KMeans state is incomplete"
        ) from exc
    svd_states: list[dict[str, Any]] = []
    rows = state_metadata.get("svd_states")
    if not isinstance(rows, list):
        raise ValueError(
            "legacy canonical SVD state inventory is incomplete"
        )
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(
                "legacy canonical SVD state inventory is malformed"
            )
        try:
            svd_states.append(
                {
                    "family_key": str(row["family_key"]),
                    "item_cluster_ids": copy.deepcopy(
                        row["item_cluster_ids"]
                    ),
                    "weighted_matrix": np.ascontiguousarray(
                        arrays[str(row["weighted_matrix"])]
                    ),
                    "singular_values": np.ascontiguousarray(
                        arrays[str(row["singular_values"])]
                    ),
                    "components": np.ascontiguousarray(
                        arrays[str(row["components"])]
                    ),
                    "parameters": copy.deepcopy(row["parameters"]),
                    "sign_canonicalization_policy": str(
                        row["sign_canonicalization_policy"]
                    ),
                    "rank_tolerance_policy": str(
                        row["rank_tolerance_policy"]
                    ),
                    "rank_tolerance_dtype": str(
                        row["rank_tolerance_dtype"]
                    ),
                    "rank_tolerance_multiplier": float.fromhex(
                        str(row["rank_tolerance_multiplier_hex"])
                    ),
                    "rank_tolerance": float.fromhex(
                        str(row["rank_tolerance_hex"])
                    ),
                    "numerical_rank": int(row["numerical_rank"]),
                    "replay_comparison_policy": str(
                        row["replay_comparison_policy"]
                    ),
                    "replay_relative_tolerance": float.fromhex(
                        str(row["replay_relative_tolerance_hex"])
                    ),
                    "replay_absolute_tolerance": float.fromhex(
                        str(row["replay_absolute_tolerance_hex"])
                    ),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "legacy canonical SVD state is incomplete"
            ) from exc
    scope_record = getattr(state, "scope_record", None)
    if not isinstance(scope_record, Mapping):
        raise ValueError(
            "legacy canonical state lacks its authenticated owner scope"
        )
    from .role_neutral_embedding_group_execution import (
        _normalize_cluster_state_arrays,
    )

    normalized, normalized_metadata = _normalize_cluster_state_arrays(
        scope=scope_record,
        kmeans_state=kmeans_state,
        svd_states=tuple(svd_states),
    )
    if (
        normalized_metadata != state_metadata
        or set(normalized) != set(arrays)
        or any(
            not np.array_equal(
                np.asarray(normalized[name]),
                np.asarray(arrays[name]),
                equal_nan=True,
            )
            for name in normalized
        )
    ):
        raise ValueError(
            "legacy canonical state changed during no-refit transcode"
        )
    return {
        "schema_version": (
            "production_stage1_cluster_preflight_scope_state_capture_v2"
        ),
        "scope_id": owner,
        "cluster_fit_identity_content_sha256": expected_fit,
        "kmeans_state": kmeans_state,
        "svd_states": svd_states,
        "captured_from_canonical_preflight_fit": True,
        "refit_performed_for_state_capture": False,
    }


def _encode_state_tree(
    value: Any,
    *,
    arrays_root: Path,
    names: list[str],
    stem: str,
) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise ValueError(
                "reusable preflight state cannot contain object arrays"
            )
        name = f"{len(names):04d}_{stem}.npy"
        path = arrays_root / name
        with path.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        digest, size = _stable_file_sha256(path)
        names.append(name)
        return {
            "__reusable_npy__": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": digest,
            "size_bytes": size,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _encode_state_tree(
                child,
                arrays_root=arrays_root,
                names=names,
                stem=f"{stem}_{key}",
            )
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return {
            "__reusable_tuple__": [
                _encode_state_tree(
                    child,
                    arrays_root=arrays_root,
                    names=names,
                    stem=f"{stem}_{index}",
                )
                for index, child in enumerate(value)
            ]
        }
    if isinstance(value, list):
        return [
            _encode_state_tree(
                child,
                arrays_root=arrays_root,
                names=names,
                stem=f"{stem}_{index}",
            )
            for index, child in enumerate(value)
        ]
    if value is None or isinstance(value, (str, int, float, bool)):
        _canonical_json(value)
        return value
    raise TypeError(
        f"unsupported reusable preflight state value: "
        f"{type(value).__name__}"
    )


def _decode_state_tree(value: Any, *, root: Path) -> Any:
    if isinstance(value, Mapping) and set(value) == {
        "__reusable_npy__",
        "dtype",
        "shape",
        "sha256",
        "size_bytes",
    }:
        path = root / "state_arrays" / str(value["__reusable_npy__"])
        digest, size = _stable_file_sha256(path)
        if digest != value["sha256"] or size != value["size_bytes"]:
            raise ValueError(
                "reusable preflight state array bytes changed"
            )
        loaded = np.load(path, allow_pickle=False)
        if (
            loaded.dtype.str != value["dtype"]
            or list(loaded.shape) != value["shape"]
        ):
            raise ValueError(
                "reusable preflight state array metadata changed"
            )
        return np.ascontiguousarray(loaded)
    if isinstance(value, Mapping) and set(value) == {
        "__reusable_tuple__"
    }:
        children = value["__reusable_tuple__"]
        if not isinstance(children, list):
            raise ValueError(
                "reusable preflight tuple state is malformed"
            )
        return tuple(
            _decode_state_tree(child, root=root)
            for child in children
        )
    if isinstance(value, Mapping):
        return {
            str(key): _decode_state_tree(child, root=root)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _decode_state_tree(child, root=root)
            for child in value
        ]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class ReusableOwnerArtifact:
    root: Path
    terminal: Mapping[str, Any]
    authentication_mode: str
    authentication_seconds: float
    payload_bytes_read: int
    _loaded: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    @property
    def manifest_path(self) -> Path:
        return self.root / OWNER_TERMINAL

    @property
    def scientific_key(self) -> str:
        return str(self.terminal["scientific_key"])

    @property
    def owner_scope_id(self) -> str:
        return str(self.terminal["physical_owner_scope_id"])

    def load_scope_audit(self) -> dict[str, Any]:
        with self._lock:
            cached = self._loaded.get("scope_audit")
            if cached is not None:
                return copy.deepcopy(cached)
            terminal = self.terminal
            compact = _read_json(
                self.root / "compact_fit_identity.json",
                label="reusable owner compact fit identity",
            )
            fit_identity = _read_owner_parquet(
                self.root / "concepts.parquet",
                expected_compact_fit=compact,
            )
            scope_without = _read_json(
                self.root / "scope_without_fit_identity.json",
                label="reusable owner scope audit",
            )
            scope = {
                **scope_without,
                "cluster_fit_identity": fit_identity,
            }
            if _sha256_json(scope) != terminal[
                "source_scope_record_sha256"
            ]:
                # Source scope records use the same canonical finite JSON
                # encoding as this module for all supported values.
                from .production_stage1_cluster_preflight_artifact_v2 import (
                    _sha256_json_streaming,
                )

                if _sha256_json_streaming(scope) != terminal[
                    "source_scope_record_sha256"
                ]:
                    raise ValueError(
                        "reusable owner scope audit changed"
                    )
            self._loaded["scope_audit"] = copy.deepcopy(scope)
            return copy.deepcopy(scope)

    def _captured_state_envelope(self) -> dict[str, Any]:
        with self._lock:
            cached = self._loaded.get("captured_state_envelope")
            if cached is not None:
                return copy.deepcopy(cached)
            encoded_state = _read_json(
                self.root / "captured_state.json",
                label="reusable owner captured state",
            )
            if (
                encoded_state.get("content_sha256")
                != _sha256_json(
                    {
                        "schema_version": encoded_state.get(
                            "schema_version"
                        ),
                        "state": encoded_state.get("state"),
                    }
                )
                or not isinstance(encoded_state.get("state"), Mapping)
                or encoded_state["state"].get("scope_id")
                != self.owner_scope_id
            ):
                raise ValueError(
                    "reusable owner captured state changed"
                )
            self._loaded["captured_state_envelope"] = copy.deepcopy(
                encoded_state
            )
            return copy.deepcopy(encoded_state)

    def load_captured_state(self) -> dict[str, Any]:
        """Deserialize one owner's safe arrays only when that owner is used."""

        with self._lock:
            cached = self._loaded.get("captured_state")
            if cached is not None:
                return copy.deepcopy(cached)
            encoded_state = self._captured_state_envelope()
            state = _decode_state_tree(
                encoded_state["state"],
                root=self.root,
            )
            if (
                not isinstance(state, dict)
                or state.get("scope_id") != self.owner_scope_id
            ):
                raise ValueError(
                    "reusable owner captured state changed"
                )
            self._loaded["captured_state"] = copy.deepcopy(state)
            return copy.deepcopy(state)

    def load_scope_audit_and_state(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.load_scope_audit(), self.load_captured_state()


def seal_reusable_owner_artifact(
    *,
    store_root: Path | str,
    compatibility: Mapping[str, Any],
    scope_audit: Mapping[str, Any],
    captured_state: Mapping[str, Any],
    producer_identity: str,
    parquet_compression: str,
) -> ReusableOwnerArtifact:
    store = _store_root(store_root)
    key = scientific_key(
        compatibility,
        expected_schema=OWNER_COMPATIBILITY_SCHEMA,
    )
    target = store / "owner_artifacts" / key
    if target.is_dir() and not target.is_symlink():
        return load_reusable_owner_artifact(
            store_root=store,
            compatibility=compatibility,
            producer_identity=producer_identity,
        )
    if target.exists() or target.is_symlink():
        raise ValueError(
            "reusable owner artifact target is invalid"
        )
    owner = str(compatibility["physical_owner_scope_id"])
    fit_identity = scope_audit.get("cluster_fit_identity")
    if (
        scope_audit.get("scope_id") != owner
        or not isinstance(fit_identity, Mapping)
        or captured_state.get("scope_id") != owner
        or captured_state.get(
            "cluster_fit_identity_content_sha256"
        )
        != fit_identity.get("content_sha256")
    ):
        raise ValueError(
            "reusable owner payload belongs to another physical owner"
        )
    from .production_stage1_cluster_preflight_artifact_v2 import (
        _compact_fit_record,
        _sha256_json_streaming,
    )

    compact, concept_rows = _compact_fit_record(
        fit_identity,
        owner_scope_id=owner,
        verify_source_content=True,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{key}.attempt-",
            dir=target.parent,
        )
    )
    started = time.perf_counter()
    try:
        scope_without = {
            name: copy.deepcopy(child)
            for name, child in scope_audit.items()
            if name != "cluster_fit_identity"
        }
        _write_new_json(
            temporary / "scope_without_fit_identity.json",
            scope_without,
        )
        _write_new_json(
            temporary / "compact_fit_identity.json",
            compact,
        )
        _write_owner_parquet(
            temporary / "concepts.parquet",
            concept_rows,
            parquet_compression=parquet_compression,
        )
        arrays_root = temporary / "state_arrays"
        arrays_root.mkdir()
        array_names: list[str] = []
        encoded = _encode_state_tree(
            captured_state,
            arrays_root=arrays_root,
            names=array_names,
            stem="state",
        )
        state_body = {
            "schema_version": (
                "production_stage1_reusable_captured_cluster_state_v1"
            ),
            "state": encoded,
        }
        _write_new_json(
            temporary / "captured_state.json",
            {
                **state_body,
                "content_sha256": _sha256_json(state_body),
            },
        )
        files = [
            _file_registration(
                temporary / "scope_without_fit_identity.json",
                root=temporary,
                kind="owner_scope_without_fit_identity",
            ),
            _file_registration(
                temporary / "compact_fit_identity.json",
                root=temporary,
                kind="owner_compact_fit_identity",
            ),
            _file_registration(
                temporary / "concepts.parquet",
                root=temporary,
                kind="complete_owner_concepts",
            ),
            _file_registration(
                temporary / "captured_state.json",
                root=temporary,
                kind="captured_kmeans_svd_state",
            ),
            *[
                _file_registration(
                    arrays_root / name,
                    root=temporary,
                    kind="captured_kmeans_svd_array",
                )
                for name in array_names
            ],
        ]
        source_scope_sha = _sha256_json_streaming(scope_audit)
        scientific_body = {
            "compatibility": _json_copy(
                compatibility,
                label="owner compatibility",
            ),
            "source_scope_record_sha256": source_scope_sha,
            "source_fit_identity_content_sha256": (
                fit_identity["content_sha256"]
            ),
            "concept_payload_content_sha256": (
                compact["concept_payload_content_sha256"]
            ),
            "captured_state_content_sha256": _sha256_json(state_body),
        }
        artifact_scientific = _sha256_json(scientific_body)
        body = {
            "schema_version": REUSABLE_OWNER_ARTIFACT_SCHEMA,
            "status": "complete",
            "scientific_key": key,
            "compatibility": _json_copy(
                compatibility,
                label="owner compatibility",
            ),
            "producer_identity": str(producer_identity),
            "physical_owner_scope_id": owner,
            "source_scope_record_sha256": source_scope_sha,
            "source_fit_identity_content_sha256": (
                fit_identity["content_sha256"]
            ),
            "compact_fit_identity_content_sha256": compact[
                "content_sha256"
            ],
            "concept_payload_content_sha256": compact[
                "concept_payload_content_sha256"
            ],
            "artifact_scientific_content_sha256": artifact_scientific,
            "files": files,
            "fit_rows_only": True,
            "heldout_labels_persisted": False,
            "cluster_refit_required_for_reuse": False,
            "temporary_attempt_then_atomic_terminal": True,
            "operational_paths_in_scientific_identity": False,
        }
        _write_new_json(
            temporary / OWNER_TERMINAL,
            {**body, "content_sha256": _sha256_json(body)},
        )
        _close_tree_read_only(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.chmod(0o700)
            for path in temporary.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError:
            pass
        # Preserve the attempt for recovery/forensics.  It never has the
        # canonical terminal location and therefore can never be adopted.
        raise
    artifact = load_reusable_owner_artifact(
        store_root=store,
        compatibility=compatibility,
        producer_identity=producer_identity,
    )
    artifact._loaded["scope_audit"] = copy.deepcopy(dict(scope_audit))
    artifact._loaded["captured_state"] = copy.deepcopy(
        dict(captured_state)
    )
    object.__setattr__(
        artifact,
        "authentication_seconds",
        time.perf_counter() - started,
    )
    return artifact


def load_reusable_owner_artifact(
    *,
    store_root: Path | str,
    compatibility: Mapping[str, Any],
    producer_identity: str,
) -> ReusableOwnerArtifact:
    store = _store_root(store_root)
    key = scientific_key(
        compatibility,
        expected_schema=OWNER_COMPATIBILITY_SCHEMA,
    )
    root = (store / "owner_artifacts" / key).resolve(strict=True)
    terminal_path = root / OWNER_TERMINAL
    full_authentication_start_probe_ctime_ns = (
        _authentication_probe_ctime_ns(store)
    )
    terminal_sha, terminal_size = _stable_file_sha256(terminal_path)
    terminal = _read_json(
        terminal_path,
        label="reusable owner terminal",
    )
    _manifest_body(terminal)
    if (
        terminal.get("schema_version") != REUSABLE_OWNER_ARTIFACT_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("scientific_key") != key
        or terminal.get("compatibility")
        != _json_copy(
            compatibility,
            label="owner compatibility",
        )
        or terminal.get("producer_identity") != str(producer_identity)
        or terminal.get("physical_owner_scope_id")
        != compatibility["physical_owner_scope_id"]
        or terminal.get("fit_rows_only") is not True
        or terminal.get("heldout_labels_persisted") is not False
        or terminal.get("cluster_refit_required_for_reuse") is not False
        or terminal.get(
            "temporary_attempt_then_atomic_terminal"
        )
        is not True
        or terminal.get(
            "operational_paths_in_scientific_identity"
        )
        is not False
    ):
        raise ValueError(
            "reusable owner artifact terminal is incompatible"
        )
    proof = _load_fast_proof(
        store_root=store,
        artifact_kind="owner_artifact",
        scientific_key=key,
        artifact_root=root,
        terminal_content_sha256=terminal["content_sha256"],
        producer_identity=str(producer_identity),
        schema_identity=REUSABLE_OWNER_ARTIFACT_SCHEMA,
    )
    if proof is not None:
        _validate_registered_tree(
            root=root,
            terminal_name=OWNER_TERMINAL,
            files=terminal["files"],
            read_payload_bytes=False,
        )
        mode = "prior_proof_stat_continuity"
        elapsed = proof[1]
        bytes_read = terminal_size
        artifact = ReusableOwnerArtifact(
            root=root,
            terminal=MappingProxyType(copy.deepcopy(terminal)),
            authentication_mode=mode,
            authentication_seconds=elapsed,
            payload_bytes_read=bytes_read,
        )
    else:
        started = time.perf_counter()
        (
            _registrations,
            _payload_read,
            authenticated_bytes,
        ) = _validate_registered_tree(
            root=root,
            terminal_name=OWNER_TERMINAL,
            files=terminal["files"],
            read_payload_bytes=True,
        )
        authenticated_bytes[OWNER_TERMINAL] = (
            terminal_sha,
            terminal_size,
        )
        artifact = ReusableOwnerArtifact(
            root=root,
            terminal=MappingProxyType(copy.deepcopy(terminal)),
            authentication_mode="full_byte_reauthentication",
            authentication_seconds=0.0,
            payload_bytes_read=sum(
                int(row["size_bytes"])
                for row in terminal["files"]
            )
            + terminal_size,
        )
        scope = artifact.load_scope_audit()
        encoded = artifact._captured_state_envelope()
        if (
            scope.get("scope_id")
            != terminal["physical_owner_scope_id"]
            or encoded["state"].get("scope_id")
            != terminal["physical_owner_scope_id"]
        ):
            raise ValueError(
                "reusable owner deep authentication changed its owner"
            )
        elapsed = time.perf_counter() - started
        object.__setattr__(
            artifact,
            "authentication_seconds",
            elapsed,
        )
        _publish_optional_full_auth_proof(
            store_root=store,
            artifact_kind="owner_artifact",
            scientific_key=key,
            artifact_root=root,
            terminal_content_sha256=terminal["content_sha256"],
            artifact_scientific_content_sha256=(
                terminal["artifact_scientific_content_sha256"]
            ),
            producer_identity=str(producer_identity),
            schema_identity=REUSABLE_OWNER_ARTIFACT_SCHEMA,
            full_authentication_start_probe_ctime_ns=(
                full_authentication_start_probe_ctime_ns
            ),
            authenticated_byte_inventory=authenticated_bytes,
        )
    return artifact


def try_load_reusable_owner_artifact(
    *,
    store_root: Path | str,
    compatibility: Mapping[str, Any],
    producer_identity: str,
) -> ReusableOwnerArtifact | None:
    try:
        return load_reusable_owner_artifact(
            store_root=store_root,
            compatibility=compatibility,
            producer_identity=producer_identity,
        )
    except FileNotFoundError:
        return None
    except (OSError, RuntimeError, ValueError) as exc:
        _quarantine_invalid_artifact(
            store_root=store_root,
            artifact_directory="owner_artifacts",
            scientific_key_value=scientific_key(
                compatibility,
                expected_schema=OWNER_COMPATIBILITY_SCHEMA,
            ),
            failure=exc,
        )
        return None


class ReusableProductionStage1ClusterPreflightArtifact(
    PortableProductionStage1ClusterPreflightArtifact
):
    """Small assembled preflight with owner concepts loaded on demand."""

    def __init__(
        self,
        *,
        root: Path,
        manifest_path: Path,
        audit_index: Mapping[str, Any],
        scientific_request: Mapping[str, Any],
        owner_handles: Mapping[str, ReusableOwnerArtifact],
        identity: Mapping[str, Any],
        authentication: Mapping[str, Any],
    ) -> None:
        object.__setattr__(self, "root", Path(root))
        object.__setattr__(self, "manifest_path", Path(manifest_path))
        object.__setattr__(
            self,
            "audit_path",
            self.root / "audit_index.json",
        )
        object.__setattr__(
            self,
            "stage1_request_path",
            self.root / "scientific_request.json",
        )
        object.__setattr__(
            self,
            "audit",
            MappingProxyType(_validate_compact_index(audit_index)),
        )
        object.__setattr__(
            self,
            "stage1_request",
            MappingProxyType(
                copy.deepcopy(dict(scientific_request))
            ),
        )
        object.__setattr__(
            self,
            "reference",
            MappingProxyType(_reference_from_index(self.audit)),
        )
        object.__setattr__(self, "_owners", dict(owner_handles))
        object.__setattr__(
            self,
            "_identity",
            MappingProxyType(copy.deepcopy(dict(identity))),
        )
        object.__setattr__(
            self,
            "authentication",
            MappingProxyType(copy.deepcopy(dict(authentication))),
        )
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "_lock", threading.RLock())
        # Base portable fields are intentionally unused by this lazy
        # path-neutral implementation, but keeping them closed prevents base
        # dataclass methods from observing partially initialized state.
        object.__setattr__(self, "_payload_snapshots", MappingProxyType({}))
        object.__setattr__(self, "_owner_fit_cache", {})
        object.__setattr__(self, "_cache_lock", threading.Lock())

    @property
    def is_portable_v2(self) -> bool:
        # This is API compatibility, not a schema claim.  It preserves the
        # existing lazy portable consumer route.
        return True

    @property
    def is_reusable_preflight_v1(self) -> bool:
        return True

    def identity(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self._identity))

    def require_stage1_request(
        self,
        expected_stage1_request: Mapping[str, Any],
    ) -> None:
        validated = _validate_stage1_request_with_reference(
            expected_stage1_request,
            expected_reference=self.reference,
        )
        if _reusable_preflight_request_projection(
            validated
        ) != dict(self.stage1_request):
            raise ValueError(
                "consumer Stage 1 request differs from reusable preflight"
            )

    def owner_fit_identity(self, owner_scope_id: str) -> dict[str, Any]:
        owner = str(owner_scope_id)
        with self._lock:
            cached = self._cache.get(owner)
            if cached is not None:
                return copy.deepcopy(dict(cached))
            handle = self._owners.get(owner)
            if handle is None:
                raise ValueError(
                    "reusable preflight has no unique physical owner"
                )
            scope = handle.load_scope_audit()
            identity = scope.get("cluster_fit_identity")
            if not isinstance(identity, Mapping):
                raise ValueError(
                    "reusable preflight owner lacks fit identity"
                )
            self._cache.clear()
            self._cache[owner] = copy.deepcopy(dict(identity))
            return copy.deepcopy(dict(identity))

    def logical_scope_record(
        self,
        scope_id: str,
        *,
        include_concepts: bool = True,
    ) -> dict[str, Any]:
        rows = [
            row
            for row in self.audit["logical_scopes"]
            if row["scope_id"] == str(scope_id)
        ]
        if len(rows) != 1:
            raise ValueError(
                "reusable preflight has no unique logical scope"
            )
        logical = rows[0]
        output = copy.deepcopy(
            dict(logical["scope_without_fit_identity"])
        )
        if not include_concepts:
            output["cluster_fit_reference"] = copy.deepcopy(
                dict(logical["physical_fit_reference"])
            )
            return output
        owner = logical["physical_fit_reference"][
            "physical_owner_scope_id"
        ]
        output["cluster_fit_identity"] = self.owner_fit_identity(owner)
        from .production_stage1_cluster_preflight_artifact_v2 import (
            _sha256_json_streaming,
        )

        if _sha256_json_streaming(output) != logical[
            "source_scope_record_sha256"
        ]:
            raise ValueError(
                "reusable logical scope differs from source preflight"
            )
        return output

    def source_audit_header(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.audit["audit_header"]))

    def owner_manifest_path(self, owner_scope_id: str) -> Path:
        handle = self._owners.get(str(owner_scope_id))
        if handle is None:
            raise ValueError(
                "reusable preflight lacks an owner state"
            )
        return handle.manifest_path

    def owner_fit_input_binding(
        self,
        owner_scope_id: str,
    ) -> dict[str, Any]:
        handle = self._owners.get(str(owner_scope_id))
        compatibility = (
            handle.terminal.get("compatibility")
            if handle is not None
            else None
        )
        binding = (
            compatibility.get("fit_input_binding")
            if isinstance(compatibility, Mapping)
            else None
        )
        if not isinstance(binding, Mapping):
            raise ValueError(
                "reusable preflight owner lacks its fit-input binding"
            )
        return copy.deepcopy(dict(binding))


def assembled_compatibility(
    *,
    cluster_compatibility: Mapping[str, Any],
    preflight_plan_content_sha256: str,
    physical_owner_keys: Mapping[str, str],
    global_audit_scientific_key: str,
) -> dict[str, Any]:
    return {
        "schema_version": ASSEMBLED_COMPATIBILITY_SCHEMA,
        "cluster_compatibility": _json_copy(
            cluster_compatibility,
            label="cluster compatibility",
        ),
        "preflight_plan_content_sha256": _require_sha256(
            preflight_plan_content_sha256,
            label="preflight plan scientific content",
        ),
        "global_audit_scientific_key": _require_sha256(
            global_audit_scientific_key,
            label="global audit scientific key",
        ),
        "physical_owner_scientific_keys": {
            str(owner): _require_sha256(
                key,
                label=f"{owner} owner scientific key",
            )
            for owner, key in physical_owner_keys.items()
        },
        "physical_owner_order": [
            str(owner) for owner in physical_owner_keys
        ],
    }


def _reusable_preflight_request_projection(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Project only inputs that can alter clustered preflight science.

    The complete Stage 1 request remains authenticated by the prepared context.
    The reusable clustered-preflight assembly deliberately excludes the broad
    behavior/source-tree identity, Stage 2 catalog code, endpoint configuration,
    and operational runtime.  Those values cannot alter the already sealed
    global audit or owner KMeans/SVD states and therefore must not prevent their
    reuse after an unrelated source change.
    """

    audit = request.get("embedding_cluster_feasibility_audit")
    plan = request.get("stage1_scope_plan")
    htr_audit = request.get("htr_input_nontruncation_audit")
    semantic = request.get("semantic_witness_scientific_config")
    if not all(
        isinstance(value, Mapping)
        for value in (audit, plan, htr_audit, semantic)
    ):
        raise ValueError(
            "reusable preflight request lacks a scientific binding"
        )
    body = {
        "schema_version": (
            "production_stage1_reusable_preflight_request_projection_v2"
        ),
        "request_schema_version": request.get("schema_version"),
        "split_registry_content_sha256": _require_sha256(
            request.get("split_registry_content_sha256"),
            label="reusable preflight split registry",
        ),
        "preflight_scope_plan": preflight_scope_plan_projection(plan),
        "cluster_preflight_reference": _json_copy(
            audit,
            label="reusable cluster preflight reference",
        ),
        "global_htr_nontruncation_audit_content_sha256": _require_sha256(
            htr_audit.get("content_sha256"),
            label="reusable global HTR audit",
        ),
        "semantic_witness_scientific_config": _json_copy(
            semantic,
            label="reusable semantic witness configuration",
        ),
        "broad_behavior_identity_included": False,
        "stage2_identity_included": False,
        "operational_runtime_included": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _reusable_path_neutral_scientific_content_sha256(
    *,
    index: Mapping[str, Any],
    scientific_request: Mapping[str, Any],
) -> str:
    projection = copy.deepcopy(dict(scientific_request))
    projection.pop("content_sha256", None)
    projection["cluster_preflight_reference"] = {
        "source_audit_content_sha256": _require_sha256(
            index.get("source_audit_content_sha256"),
            label="reusable preflight source audit scientific content",
        )
    }
    body = {
        "schema_version": (
            "production_stage1_reusable_preflight_scientific_content_v1"
        ),
        "source_audit_content_sha256": index[
            "source_audit_content_sha256"
        ],
        "normalized_preflight_request_content_sha256": _sha256_json(
            projection
        ),
    }
    return _sha256_json(body)


def seal_reusable_assembled_preflight(
    *,
    store_root: Path | str,
    compatibility: Mapping[str, Any],
    audit: Mapping[str, Any],
    stage1_request: Mapping[str, Any],
    owner_handles: Mapping[str, ReusableOwnerArtifact],
    global_audit: ReusableGlobalAuditArtifact,
    plan: Any,
    producer_identity: str,
    owner_producer_identity: str,
    global_audit_producer_identity: str,
) -> ReusableProductionStage1ClusterPreflightArtifact:
    store = _store_root(store_root)
    key = scientific_key(
        compatibility,
        expected_schema=ASSEMBLED_COMPATIBILITY_SCHEMA,
    )
    target = store / "assembled_contexts" / key
    if target.is_dir() and not target.is_symlink():
        return load_reusable_assembled_preflight(
            store_root=store,
            compatibility=compatibility,
            expected_stage1_request=stage1_request,
            global_audit=global_audit,
            plan=plan,
            producer_identity=producer_identity,
            owner_producer_identity=owner_producer_identity,
            global_audit_producer_identity=(
                global_audit_producer_identity
            ),
        )[0]
    if target.exists() or target.is_symlink():
        raise ValueError(
            "reusable assembled preflight target is invalid"
        )
    if audit.get("schema_version") == (
        PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_SCHEMA
    ):
        index = _validate_compact_index(audit)
    else:
        index, _payload = _build_compact_index(
            audit,
            verify_source_audit_content=True,
            verify_source_fit_content=True,
        )
    reference = _reference_from_index(index)
    validated_request = _validate_stage1_request_with_reference(
        stage1_request,
        expected_reference=reference,
    )
    scientific_request = _reusable_preflight_request_projection(
        validated_request
    )
    if (
        compatibility.get("global_audit_scientific_key")
        != global_audit.scientific_key
    ):
        raise ValueError(
            "reusable assembled compatibility changed its global audit"
        )
    expected_owners = list(index["physical_scope_order"])
    if (
        list(owner_handles) != expected_owners
        or compatibility["physical_owner_order"] != expected_owners
        or {
            owner: handle.scientific_key
            for owner, handle in owner_handles.items()
        }
        != compatibility["physical_owner_scientific_keys"]
    ):
        raise ValueError(
            "reusable assembled preflight owner inventory changed"
        )
    indexed_fit_sha = {
        str(row["physical_owner_scope_id"]): str(
            row["compact_fit_identity"][
                "source_fit_identity_content_sha256"
            ]
        )
        for row in index["physical_fits"]
    }
    if any(
        owner_handles[owner].terminal[
            "source_fit_identity_content_sha256"
        ]
        != indexed_fit_sha[owner]
        for owner in expected_owners
    ):
        raise ValueError(
            "reusable assembled preflight owner fit identity changed"
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{key}.attempt-",
            dir=target.parent,
        )
    )
    try:
        _write_new_json(temporary / "audit_index.json", index)
        _write_new_json(
            temporary / "scientific_request.json",
            scientific_request,
        )
        locator_body = {
            "schema_version": (
                "production_stage1_reusable_owner_locator_index_v1"
            ),
            "owner_order": expected_owners,
            "owners": [
                {
                    "physical_owner_scope_id": owner,
                    "owner_scientific_key": (
                        owner_handles[owner].scientific_key
                    ),
                    "owner_terminal_content_sha256": (
                        owner_handles[owner].terminal[
                            "content_sha256"
                        ]
                    ),
                    "owner_artifact_scientific_content_sha256": (
                        owner_handles[owner].terminal[
                            "artifact_scientific_content_sha256"
                        ]
                    ),
                }
                for owner in expected_owners
            ],
            "paths_are_operational_capabilities": True,
            "paths_in_scientific_identity": False,
            "owner_paths_derived_from_current_store": True,
        }
        _write_new_json(
            temporary / "owner_locators.json",
            {
                **locator_body,
                "content_sha256": _sha256_json(locator_body),
            },
        )
        global_locator_body = {
            "schema_version": (
                "production_stage1_reusable_global_audit_locator_v1"
            ),
            "global_scientific_key": global_audit.scientific_key,
            "global_terminal_content_sha256": (
                global_audit.terminal["content_sha256"]
            ),
            "global_artifact_scientific_content_sha256": (
                global_audit.scientific_content_sha256
            ),
            "terminal_path_derived_from_current_store": True,
            "path_in_scientific_identity": False,
        }
        _write_new_json(
            temporary / "global_audit_locator.json",
            {
                **global_locator_body,
                "content_sha256": _sha256_json(global_locator_body),
            },
        )
        files = [
            _file_registration(
                temporary / "audit_index.json",
                root=temporary,
                kind="compact_scientific_audit_index",
            ),
            _file_registration(
                temporary / "scientific_request.json",
                root=temporary,
                kind="scientific_request_projection",
            ),
            _file_registration(
                temporary / "owner_locators.json",
                root=temporary,
                kind="operational_owner_locators",
            ),
            _file_registration(
                temporary / "global_audit_locator.json",
                root=temporary,
                kind="operational_global_audit_locator",
            ),
        ]
        scientific_content = (
            _reusable_path_neutral_scientific_content_sha256(
                index=index,
                scientific_request=scientific_request,
            )
        )
        body = {
            "schema_version": REUSABLE_ASSEMBLED_ARTIFACT_SCHEMA,
            "status": "complete",
            "scientific_key": key,
            "compatibility": _json_copy(
                compatibility,
                label="assembled compatibility",
            ),
            "producer_identity": str(producer_identity),
            "owner_producer_identity": str(
                owner_producer_identity
            ),
            "global_audit_producer_identity": str(
                global_audit_producer_identity
            ),
            "portable_cluster_preflight_reference": reference,
            "audit_index_content_sha256": index["content_sha256"],
            "scientific_request_content_sha256": (
                scientific_request["content_sha256"]
            ),
            "artifact_scientific_content_sha256": scientific_content,
            "global_audit_scientific_content_sha256": (
                global_audit.scientific_content_sha256
            ),
            "physical_owner_count": len(expected_owners),
            "logical_scope_count": len(index["scope_order"]),
            "files": files,
            "owner_payloads_embedded": False,
            "owner_states_loaded_during_reopen": False,
            "operational_paths_in_scientific_identity": False,
            "canonical_merge_order": list(index["scope_order"]),
        }
        _write_new_json(
            temporary / ASSEMBLED_TERMINAL,
            {**body, "content_sha256": _sha256_json(body)},
        )
        _close_tree_read_only(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.chmod(0o700)
            for path in temporary.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_reusable_assembled_preflight(
        store_root=store,
        compatibility=compatibility,
        expected_stage1_request=stage1_request,
        global_audit=global_audit,
        plan=plan,
        producer_identity=producer_identity,
        owner_producer_identity=owner_producer_identity,
        global_audit_producer_identity=(
            global_audit_producer_identity
        ),
    )[0]


def _assembled_identity(
    *,
    root: Path,
    terminal: Mapping[str, Any],
    index: Mapping[str, Any],
    scientific_request: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = root / ASSEMBLED_TERMINAL
    manifest_sha, _ = _stable_file_sha256(manifest_path)
    audit_sha, _ = _stable_file_sha256(root / "audit_index.json")
    request_sha, _ = _stable_file_sha256(
        root / "scientific_request.json"
    )
    body = {
        "schema_version": PORTABLE_CLUSTER_PREFLIGHT_RESULT_SCHEMA,
        "artifact_version": REUSABLE_PREFLIGHT_ARTIFACT_VERSION,
        "artifact_code_sha256": _sha256_json(
            {
                "producer_identity": terminal["producer_identity"],
                "schema": REUSABLE_ASSEMBLED_ARTIFACT_SCHEMA,
            }
        ),
        "root": str(root),
        "manifest_path": str(manifest_path),
        "audit_path": str(root / "audit_index.json"),
        "stage1_request_path": str(
            root / "scientific_request.json"
        ),
        "manifest_sha256": manifest_sha,
        "audit_sha256": audit_sha,
        "stage1_request_file_sha256": request_sha,
        "stage1_request_sha256": scientific_request["content_sha256"],
        "cluster_audit_content_sha256": index[
            "source_audit_content_sha256"
        ],
        "portable_audit_reference_content_sha256": (
            terminal["portable_cluster_preflight_reference"][
                "content_sha256"
            ]
        ),
        "compact_audit_index_content_sha256": index["content_sha256"],
        "payload_inventory_content_sha256": _sha256_json(
            terminal["files"]
        ),
        "physical_storage": {
            "schema_version": (
                "production_stage1_reusable_preflight_storage_v1"
            ),
            "owner_payloads_embedded": False,
            "owner_payloads_referenced_by_scientific_key": True,
        },
        "scope_count": index["logical_scope_count"],
        "physical_fit_count": index["physical_fit_count"],
        "scope_order": list(index["scope_order"]),
        "physical_scope_order": list(index["physical_scope_order"]),
        "scope_fit_identity_sha256": _sha256_json(
            [
                row["physical_fit_reference"][
                    "source_fit_identity_content_sha256"
                ]
                for row in index["logical_scopes"]
            ]
        ),
        "path_neutral_scientific_content_sha256": terminal[
            "artifact_scientific_content_sha256"
        ],
    }
    return {**body, "content_sha256": _sha256_json(body)}


def load_reusable_assembled_preflight(
    *,
    store_root: Path | str,
    compatibility: Mapping[str, Any],
    expected_stage1_request: Mapping[str, Any] | None,
    global_audit: ReusableGlobalAuditArtifact,
    plan: Any,
    producer_identity: str,
    owner_producer_identity: str,
    global_audit_producer_identity: str,
) -> tuple[
    ReusableProductionStage1ClusterPreflightArtifact,
    "ReusableClusterPreflightStateBundle",
]:
    store = _store_root(store_root)
    key = scientific_key(
        compatibility,
        expected_schema=ASSEMBLED_COMPATIBILITY_SCHEMA,
    )
    root = (store / "assembled_contexts" / key).resolve(strict=True)
    terminal_path = root / ASSEMBLED_TERMINAL
    full_authentication_start_probe_ctime_ns = (
        _authentication_probe_ctime_ns(store)
    )
    terminal_sha, terminal_size = _stable_file_sha256(terminal_path)
    terminal = _read_json(
        terminal_path,
        label="reusable assembled preflight terminal",
    )
    _manifest_body(terminal)
    if (
        terminal.get("schema_version")
        != REUSABLE_ASSEMBLED_ARTIFACT_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("scientific_key") != key
        or terminal.get("compatibility")
        != _json_copy(
            compatibility,
            label="assembled compatibility",
        )
        or terminal.get("producer_identity") != str(producer_identity)
        or terminal.get("owner_producer_identity")
        != str(owner_producer_identity)
        or terminal.get("global_audit_producer_identity")
        != str(global_audit_producer_identity)
        or terminal.get("owner_payloads_embedded") is not False
        or terminal.get("owner_states_loaded_during_reopen") is not False
        or terminal.get(
            "operational_paths_in_scientific_identity"
        )
        is not False
        or terminal.get(
            "global_audit_scientific_content_sha256"
        )
        != global_audit.scientific_content_sha256
        or compatibility.get("global_audit_scientific_key")
        != global_audit.scientific_key
    ):
        raise ValueError(
            "reusable assembled preflight is incompatible"
        )
    proof = _load_fast_proof(
        store_root=store,
        artifact_kind="assembled_context",
        scientific_key=key,
        artifact_root=root,
        terminal_content_sha256=terminal["content_sha256"],
        producer_identity=str(producer_identity),
        schema_identity=REUSABLE_ASSEMBLED_ARTIFACT_SCHEMA,
    )
    started = time.perf_counter()
    (
        registrations,
        payload_read,
        authenticated_bytes,
    ) = _validate_registered_tree(
        root=root,
        terminal_name=ASSEMBLED_TERMINAL,
        files=terminal["files"],
        read_payload_bytes=proof is None,
    )
    if proof is None:
        authenticated_bytes[ASSEMBLED_TERMINAL] = (
            terminal_sha,
            terminal_size,
        )
    index = _validate_compact_index(
        _read_json(
            root / "audit_index.json",
            label="reusable assembled audit index",
        )
    )
    scientific_request = _read_json(
        root / "scientific_request.json",
        label="reusable assembled scientific request",
    )
    owner_locators = _read_json(
        root / "owner_locators.json",
        label="reusable assembled owner locators",
    )
    global_locator = _read_json(
        root / "global_audit_locator.json",
        label="reusable assembled global audit locator",
    )
    if (
        index["content_sha256"]
        != terminal["audit_index_content_sha256"]
        or scientific_request.get("content_sha256")
        != terminal["scientific_request_content_sha256"]
        or _reference_from_index(index)
        != terminal["portable_cluster_preflight_reference"]
        or terminal["canonical_merge_order"] != index["scope_order"]
        or terminal["physical_owner_count"]
        != len(index["physical_scope_order"])
        or terminal["logical_scope_count"] != len(index["scope_order"])
        or global_locator.get("global_scientific_key")
        != global_audit.scientific_key
        or global_locator.get(
            "global_terminal_content_sha256"
        )
        != global_audit.terminal["content_sha256"]
    ):
        raise ValueError(
            "reusable assembled preflight bindings changed"
        )
    locator_rows = owner_locators.get("owners")
    if (
        owner_locators.get("owner_order")
        != index["physical_scope_order"]
        or not isinstance(locator_rows, list)
        or len(locator_rows) != len(index["physical_scope_order"])
    ):
        raise ValueError(
            "reusable assembled owner locator inventory changed"
        )
    owner_handles: dict[str, ReusableOwnerArtifact] = {}
    modes = {
        "prior_proof_stat_continuity": 0,
        "full_byte_reauthentication": 0,
    }
    auth_seconds = time.perf_counter() - started
    bytes_read = terminal_size + payload_read
    for owner, row in zip(
        index["physical_scope_order"],
        locator_rows,
        strict=True,
    ):
        expected_key = compatibility[
            "physical_owner_scientific_keys"
        ][owner]
        manifest_path = (
            store
            / "owner_artifacts"
            / str(expected_key)
            / OWNER_TERMINAL
        ).resolve(strict=True)
        if (
            row.get("physical_owner_scope_id") != owner
            or row.get("owner_scientific_key") != expected_key
            or "owner_manifest_path" in row
        ):
            raise ValueError(
                "reusable assembled owner locator changed"
            )
        owner_compat = _read_json(
            manifest_path,
            label=f"{owner} reusable owner terminal",
        ).get("compatibility")
        handle = load_reusable_owner_artifact(
            store_root=store,
            compatibility=owner_compat,
            producer_identity=owner_producer_identity,
        )
        if (
            handle.scientific_key != expected_key
            or handle.manifest_path != manifest_path
            or handle.terminal["content_sha256"]
            != row["owner_terminal_content_sha256"]
            or handle.terminal[
                "artifact_scientific_content_sha256"
            ]
            != row[
                "owner_artifact_scientific_content_sha256"
            ]
        ):
            raise ValueError(
                "reusable assembled owner artifact changed"
            )
        owner_handles[owner] = handle
        modes[handle.authentication_mode] += 1
        auth_seconds += handle.authentication_seconds
        bytes_read += handle.payload_bytes_read
    identity = _assembled_identity(
        root=root,
        terminal=terminal,
        index=index,
        scientific_request=scientific_request,
    )
    authentication = {
        "schema_version": (
            "production_stage1_reusable_preflight_reopen_telemetry_v1"
        ),
        "assembled_authentication_mode": (
            "prior_proof_stat_continuity"
            if proof is not None
            else "full_byte_reauthentication"
        ),
        "owner_fast_stat_count": modes[
            "prior_proof_stat_continuity"
        ],
        "owner_deep_auth_count": modes[
            "full_byte_reauthentication"
        ],
        "owner_recomputed_count": 0,
        "authentication_seconds": auth_seconds,
        "payload_bytes_read": bytes_read,
        "owner_state_payloads_deserialized": 0,
        "bulk_owner_payload_read_during_unchanged_fast_path": False,
        "peak_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
    }
    artifact = (
        ReusableProductionStage1ClusterPreflightArtifact(
            root=root,
            manifest_path=terminal_path,
            audit_index=index,
            scientific_request=scientific_request,
            owner_handles=owner_handles,
            identity=identity,
            authentication=authentication,
        )
    )
    if expected_stage1_request is not None:
        artifact.require_stage1_request(expected_stage1_request)
    if proof is None:
        _publish_optional_full_auth_proof(
            store_root=store,
            artifact_kind="assembled_context",
            scientific_key=key,
            artifact_root=root,
            terminal_content_sha256=terminal["content_sha256"],
            artifact_scientific_content_sha256=(
                terminal["artifact_scientific_content_sha256"]
            ),
            producer_identity=str(producer_identity),
            schema_identity=REUSABLE_ASSEMBLED_ARTIFACT_SCHEMA,
            full_authentication_start_probe_ctime_ns=int(
                full_authentication_start_probe_ctime_ns
            ),
            authenticated_byte_inventory=authenticated_bytes,
        )
    bundle = ReusableClusterPreflightStateBundle(
        preflight=artifact,
        plan=plan,
    )
    return artifact, bundle


def try_load_reusable_assembled_preflight(
    *,
    store_root: Path | str,
    compatibility: Mapping[str, Any],
    expected_stage1_request: Mapping[str, Any] | None,
    global_audit: ReusableGlobalAuditArtifact,
    plan: Any,
    producer_identity: str,
    owner_producer_identity: str,
    global_audit_producer_identity: str,
) -> tuple[
    ReusableProductionStage1ClusterPreflightArtifact,
    "ReusableClusterPreflightStateBundle",
] | None:
    try:
        return load_reusable_assembled_preflight(
            store_root=store_root,
            compatibility=compatibility,
            expected_stage1_request=expected_stage1_request,
            global_audit=global_audit,
            plan=plan,
            producer_identity=producer_identity,
            owner_producer_identity=owner_producer_identity,
            global_audit_producer_identity=(
                global_audit_producer_identity
            ),
        )
    except FileNotFoundError:
        return None
    except (OSError, RuntimeError, ValueError) as exc:
        _quarantine_invalid_artifact(
            store_root=store_root,
            artifact_directory="assembled_contexts",
            scientific_key_value=scientific_key(
                compatibility,
                expected_schema=ASSEMBLED_COMPATIBILITY_SCHEMA,
            ),
            failure=exc,
        )
        return None


class _ReusableStateMap(Mapping[str, Any]):
    def __init__(
        self,
        *,
        preflight: ReusableProductionStage1ClusterPreflightArtifact,
    ) -> None:
        self.preflight = preflight
        self.owner_order = tuple(
            preflight.audit["physical_scope_order"]
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.owner_order)

    def __len__(self) -> int:
        return len(self.owner_order)

    def __getitem__(self, owner: str) -> Any:
        from .production_stage1_scope_scheduler import Stage1ScopePlan

        del Stage1ScopePlan
        raise TypeError(
            "load reusable clustered state through its owner manifest "
            "and canonical request"
        )

    def manifest_path_for_owner(self, owner: str) -> Path:
        return self.preflight.owner_manifest_path(owner)


@dataclass(frozen=True)
class ReusableClusterPreflightStateBundle:
    preflight: ReusableProductionStage1ClusterPreflightArtifact
    plan: Any
    states: Mapping[str, Any] = field(init=False)
    manifest: Mapping[str, Any] = field(init=False)
    root: Path = field(init=False)
    _adapter_lock: threading.RLock = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan",
            _normalize_stage1_scope_plan(self.plan),
        )
        owner_order = list(
            self.preflight.audit["physical_scope_order"]
        )
        plan_scientific_content_sha256 = _require_sha256(
            getattr(self.plan, "scientific_content_sha256", None),
            label="reusable state plan",
        )
        if [
            scope.scope_id for scope in self.plan.physical_scopes
        ] != owner_order:
            raise ValueError(
                "reusable state plan changed its physical owner order"
            )
        body = {
            "schema_version": (
                REUSABLE_STATE_BUNDLE_REFERENCE_SCHEMA
            ),
            "status": "complete",
            "plan_scientific_content_sha256": (
                plan_scientific_content_sha256
            ),
            "assembled_preflight_scientific_content_sha256": (
                self.preflight.identity()[
                    "path_neutral_scientific_content_sha256"
                ]
            ),
            "physical_owner_scope_order": owner_order,
            "physical_owner_count": len(owner_order),
            "owner_state_payloads_embedded": False,
            "cluster_refit_performed": False,
        }
        object.__setattr__(
            self,
            "manifest",
            MappingProxyType(
                {**body, "content_sha256": _sha256_json(body)}
            ),
        )
        object.__setattr__(self, "root", self.preflight.root)
        object.__setattr__(
            self,
            "_adapter_lock",
            threading.RLock(),
        )
        object.__setattr__(
            self,
            "states",
            _ReusableStateMap(preflight=self.preflight),
        )

    @property
    def content_sha256(self) -> str:
        return str(self.manifest["content_sha256"])

    @property
    def plan_scientific_content_sha256(self) -> str:
        return str(self.plan.scientific_content_sha256)

    def manifest_path_for_owner(self, scope_id: str) -> Path:
        owner = str(scope_id)
        if owner not in set(
            self.preflight.audit["physical_scope_order"]
        ):
            raise ValueError(
                "reusable state bundle lacks that physical owner"
            )
        from .role_neutral_embedding_group_execution import (
            RoleNeutralEmbeddingPhysicalGroupRequest,
            load_canonical_clustered_preflight_scope_state,
            seal_canonical_clustered_preflight_scope_state,
        )

        request = RoleNeutralEmbeddingPhysicalGroupRequest.from_plan(
            plan=self.plan,
            physical_owner_scope_id=owner,
        )
        store_root = self.preflight.root.parents[1]
        adapter_parent = (
            store_root
            / "canonical_state_adapters"
            / self.preflight.root.name
            / self.plan.scientific_content_sha256
        )
        target = adapter_parent / owner
        manifest_path = target / "cluster_state_manifest.json"
        with self._adapter_lock:
            if manifest_path.is_file() and not manifest_path.is_symlink():
                load_canonical_clustered_preflight_scope_state(
                    manifest_path=manifest_path,
                    preflight=self.preflight,
                    request=request,
                )
                return manifest_path
            adapter_parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{owner}.attempt-",
                    dir=adapter_parent,
                )
            )
            try:
                handle = self.preflight._owners[owner]
                captured = handle.load_captured_state()
                seal_canonical_clustered_preflight_scope_state(
                    output_root=temporary / "state",
                    preflight=self.preflight,
                    request=request,
                    kmeans_state=captured["kmeans_state"],
                    svd_states=tuple(captured["svd_states"]),
                )
                os.replace(temporary / "state", target)
                _fsync_directory(adapter_parent)
                temporary.rmdir()
            except BaseException:
                # The noncanonical attempt remains recoverable and can never
                # be mistaken for a complete adapter.
                raise
        return manifest_path


def load_reusable_owner_cluster_state(
    *,
    manifest_path: Path | str,
    preflight: ReusableProductionStage1ClusterPreflightArtifact,
    request: Any,
) -> Any:
    """Materialize one standard state object from a reusable owner artifact."""

    supplied = Path(manifest_path)
    if supplied.name != OWNER_TERMINAL or not supplied.is_absolute():
        raise ValueError(
            "reusable owner state manifest path is invalid"
        )
    owner = str(request.physical_owner.scope_id)
    expected = preflight.owner_manifest_path(owner)
    if supplied.resolve(strict=True) != expected:
        raise ValueError(
            "reusable owner state manifest belongs to another owner"
        )
    handle = preflight._owners[owner]
    scope = handle.load_scope_audit()
    captured = handle.load_captured_state()
    from .role_neutral_embedding_group_execution import (
        AuthenticatedClusteredPreflightScopeState,
        _canonical_preflight_scope_binding,
        _normalize_cluster_state_arrays,
    )

    binding, bound_scope = _canonical_preflight_scope_binding(
        preflight=preflight,
        request=request,
        provider_cache_identity=None,
    )
    if bound_scope != scope:
        raise ValueError(
            "reusable owner state scope differs from assembled preflight"
        )
    arrays, state_metadata = _normalize_cluster_state_arrays(
        scope=scope,
        kmeans_state=captured["kmeans_state"],
        svd_states=tuple(captured["svd_states"]),
    )
    body = {
        "schema_version": (
            "production_canonical_clustered_preflight_scope_state_v2"
        ),
        "status": "complete",
        "group_request_content_sha256": request.content_sha256,
        "plan_scientific_content_sha256": (
            request.plan_scientific_content_sha256
        ),
        "physical_owner_scope_id": owner,
        "fit_row_ids": list(request.physical_owner.fit_row_ids),
        "fit_row_order_fingerprint": scope[
            "fit_row_order_fingerprint"
        ],
        "canonical_group_seed": int(
            request.physical_owner.scope_seed
        ),
        "cluster_scientific_configuration_sha256": _sha256_json(
            state_metadata["cluster_scientific_configuration"]
        ),
        "preflight_binding": binding,
        "state_metadata": state_metadata,
        "array_order": sorted(arrays),
        "arrays": {
            name: {
                "relative_path": (
                    f"reusable-owner://{handle.scientific_key}/{name}"
                ),
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": _sha256_json(
                    {
                        "dtype": array.dtype.str,
                        "shape": list(array.shape),
                        "bytes_sha256": hashlib.sha256(
                            array.tobytes(order="C")
                        ).hexdigest(),
                    }
                ),
                "size_bytes": int(array.nbytes),
            }
            for name, array in arrays.items()
        },
        "state_origin": "canonical_clustered_preflight_no_refit_v1",
        "executable_serialization_used": False,
        "pickle_joblib_or_npz_used": False,
    }
    manifest = {**body, "content_sha256": _sha256_json(body)}
    return AuthenticatedClusteredPreflightScopeState(
        root=handle.root,
        manifest=manifest,
        arrays=arrays,
        scope_record=_ReusableLazyScope(
            preflight=preflight,
            owner=owner,
        ),
    )


@dataclass(frozen=True)
class _ReusableLazyScope(Mapping[str, Any]):
    preflight: ReusableProductionStage1ClusterPreflightArtifact
    owner: str

    def _snapshot(self) -> dict[str, Any]:
        return self.preflight.logical_scope_record(
            self.owner,
            include_concepts=True,
        )

    def __getitem__(self, key: str) -> Any:
        return self._snapshot()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(tuple(self._snapshot()))

    def __len__(self) -> int:
        return len(self._snapshot())

    def __deepcopy__(self, _memo: dict[int, Any]) -> "_ReusableLazyScope":
        return self


def publish_reusable_preflight_references(
    *,
    preflight_output_root: Path | str,
    state_output_root: Path | str,
    artifact: ReusableProductionStage1ClusterPreflightArtifact,
    state_bundle: ReusableClusterPreflightStateBundle,
) -> tuple[
    ReusableProductionStage1ClusterPreflightArtifact,
    ReusableClusterPreflightStateBundle,
]:
    """Publish small phase-local locator manifests, never owner payload copies."""

    preflight_root = Path(preflight_output_root)
    state_root = Path(state_output_root)
    for root in (preflight_root, state_root):
        if not root.is_absolute() or root.exists() or root.is_symlink():
            raise ValueError(
                "reusable preflight reference output must be fresh and absolute"
            )
        root.mkdir(parents=True)
    identity = artifact.identity()
    reference_body = {
        "schema_version": REUSABLE_PREFLIGHT_REFERENCE_SCHEMA,
        "status": "complete",
        "assembled_terminal_path": str(artifact.manifest_path),
        "assembled_terminal_content_sha256": (
            artifact._owners[
                artifact.audit["physical_scope_order"][0]
            ].terminal.get("content_sha256")
        ),
        "assembled_scientific_key": artifact.manifest_path.parent.name,
        "assembled_scientific_content_sha256": identity[
            "path_neutral_scientific_content_sha256"
        ],
        "portable_cluster_preflight_reference": copy.deepcopy(
            dict(artifact.reference)
        ),
        "locator_is_operational_not_scientific": True,
        "owner_payloads_copied": False,
    }
    # Bind the actual assembled terminal rather than the first-owner terminal.
    assembled_terminal = _read_json(
        artifact.manifest_path,
        label="assembled preflight terminal",
    )
    reference_body["assembled_terminal_content_sha256"] = (
        assembled_terminal["content_sha256"]
    )
    _write_new_json(
        preflight_root / REFERENCE_MANIFEST,
        {
            **reference_body,
            "content_sha256": _sha256_json(reference_body),
        },
    )
    state_body = {
        "schema_version": REUSABLE_STATE_BUNDLE_REFERENCE_SCHEMA,
        "status": "complete",
        "preflight_reference_relative_path": (
            Path("..")
            / preflight_root.name
            / REFERENCE_MANIFEST
        ).as_posix(),
        "assembled_terminal_path": str(artifact.manifest_path),
        "assembled_scientific_content_sha256": identity[
            "path_neutral_scientific_content_sha256"
        ],
        "plan_scientific_content_sha256": (
            state_bundle.plan_scientific_content_sha256
        ),
        "physical_owner_scope_order": list(
            artifact.audit["physical_scope_order"]
        ),
        "owner_manifest_paths": {
            owner: str(artifact.owner_manifest_path(owner))
            for owner in artifact.audit["physical_scope_order"]
        },
        "owner_payloads_copied": False,
        "cluster_refit_performed": False,
        "locators_are_operational_not_scientific": True,
    }
    _write_new_json(
        state_root / STATE_BUNDLE_MANIFEST,
        {**state_body, "content_sha256": _sha256_json(state_body)},
    )
    return artifact, state_bundle


def is_reusable_preflight_reference(
    manifest_path: Path | str,
) -> bool:
    path = Path(manifest_path)
    if not path.is_file() or path.is_symlink():
        return False
    try:
        return (
            _read_json(
                path,
                label="clustered preflight manifest",
            ).get("schema_version")
            == REUSABLE_PREFLIGHT_REFERENCE_SCHEMA
        )
    except (OSError, ValueError):
        return False


def load_reusable_preflight_reference(
    *,
    manifest_path: Path | str,
    expected_stage1_request: Mapping[str, Any] | None,
    plan: Any,
    producer_identity: str,
) -> ReusableProductionStage1ClusterPreflightArtifact:
    path = Path(manifest_path).resolve(strict=True)
    reference = _read_json(
        path,
        label="reusable preflight reference",
    )
    _manifest_body(reference)
    assembled_path = Path(str(reference.get("assembled_terminal_path")))
    if (
        reference.get("schema_version")
        != REUSABLE_PREFLIGHT_REFERENCE_SCHEMA
        or reference.get("status") != "complete"
        or not assembled_path.is_absolute()
        or assembled_path.name != ASSEMBLED_TERMINAL
        or reference.get(
            "locator_is_operational_not_scientific"
        )
        is not True
        or reference.get("owner_payloads_copied") is not False
    ):
        raise ValueError(
            "reusable preflight reference is invalid"
        )
    assembled = _read_json(
        assembled_path,
        label="referenced assembled preflight",
    )
    compatibility = assembled.get("compatibility")
    global_locator = _read_json(
        assembled_path.parent / "global_audit_locator.json",
        label="referenced global audit locator",
    )
    store_root = assembled_path.parents[2]
    global_terminal_path = (
        store_root
        / "global_audits"
        / str(global_locator["global_scientific_key"])
        / GLOBAL_TERMINAL
    ).resolve(strict=True)
    global_terminal = _read_json(
        global_terminal_path,
        label="referenced global audit terminal",
    )
    global_artifact = load_reusable_global_audit(
        store_root=store_root,
        compatibility=global_terminal["compatibility"],
        producer_identity=assembled[
            "global_audit_producer_identity"
        ],
    )
    artifact, _bundle = load_reusable_assembled_preflight(
        store_root=store_root,
        compatibility=compatibility,
        expected_stage1_request=expected_stage1_request,
        global_audit=global_artifact,
        plan=plan,
        producer_identity=producer_identity,
        owner_producer_identity=assembled[
            "owner_producer_identity"
        ],
        global_audit_producer_identity=assembled[
            "global_audit_producer_identity"
        ],
    )
    if (
        artifact.identity()[
            "path_neutral_scientific_content_sha256"
        ]
        != reference["assembled_scientific_content_sha256"]
        or dict(artifact.reference)
        != reference["portable_cluster_preflight_reference"]
    ):
        raise ValueError(
            "reusable preflight reference scientific binding changed"
        )
    return artifact


def load_reusable_state_bundle_reference(
    *,
    manifest_path: Path | str,
    preflight: ReusableProductionStage1ClusterPreflightArtifact,
    plan: Any,
) -> ReusableClusterPreflightStateBundle:
    path = Path(manifest_path).resolve(strict=True)
    value = _read_json(
        path,
        label="reusable preflight state bundle reference",
    )
    _manifest_body(value)
    owner_order = [
        scope.scope_id for scope in plan.physical_scopes
    ]
    if (
        value.get("schema_version")
        != REUSABLE_STATE_BUNDLE_REFERENCE_SCHEMA
        or value.get("status") != "complete"
        or value.get("plan_scientific_content_sha256")
        != plan.scientific_content_sha256
        or value.get("physical_owner_scope_order") != owner_order
        or value.get("owner_manifest_paths")
        != {
            owner: str(preflight.owner_manifest_path(owner))
            for owner in owner_order
        }
        or value.get("owner_payloads_copied") is not False
        or value.get("cluster_refit_performed") is not False
        or value.get(
            "locators_are_operational_not_scientific"
        )
        is not True
        or value.get("preflight_reference_relative_path")
        != (Path("..") / "cluster_preflight" / REFERENCE_MANIFEST).as_posix()
    ):
        raise ValueError(
            "reusable preflight state bundle reference changed"
        )
    return ReusableClusterPreflightStateBundle(
        preflight=preflight,
        plan=plan,
    )


@dataclass(frozen=True)
class ReusablePreflightAcceptance:
    """Authenticated small reopen index plus its lazy scientific context."""

    root: Path
    terminal: Mapping[str, Any]
    prepared_context: Any
    effective_profile_path: Path
    preflight: ReusableProductionStage1ClusterPreflightArtifact
    state_bundle: ReusableClusterPreflightStateBundle
    authentication_mode: str
    authentication_seconds: float
    payload_bytes_read: int
    global_audit_authentication_mode: str
    global_audit_authentication_seconds: float
    global_audit_payload_bytes_read: int


def _accepted_context_key(selector: Mapping[str, Any]) -> str:
    value = _json_copy(
        selector,
        label="reusable preflight accepted-input selector",
    )
    if (
        value.get("schema_version")
        != "production_stage1_preflight_accepted_input_selector_v2"
        or value.get("content_sha256")
        != _sha256_json(
            {
                key: child
                for key, child in value.items()
                if key != "content_sha256"
            }
        )
    ):
        raise ValueError(
            "reusable preflight accepted-input selector is invalid"
        )
    return _sha256_json(value)


def publish_reusable_preflight_acceptance(
    *,
    store_root: Path | str,
    selector: Mapping[str, Any],
    artifact: ReusableProductionStage1ClusterPreflightArtifact,
    prepared_context_manifest_path: Path | str,
    producer_identity: str,
    owner_producer_identity: str,
    global_audit_producer_identity: str,
) -> ReusablePreflightAcceptance:
    """Publish one immutable small capability after cold preflight sealing."""

    from .prepared_stage1_context import (
        load_prepared_stage1_context,
    )

    store = _store_root(store_root)
    key = _accepted_context_key(selector)
    target = store / "accepted_contexts" / key
    if target.is_dir() and not target.is_symlink():
        return load_reusable_preflight_acceptance(
            store_root=store,
            selector=selector,
            producer_identity=producer_identity,
            owner_producer_identity=owner_producer_identity,
            global_audit_producer_identity=(
                global_audit_producer_identity
            ),
        )
    if target.exists() or target.is_symlink():
        raise ValueError(
            "reusable preflight accepted-context target is invalid"
        )
    source_context = load_prepared_stage1_context(
        prepared_context_manifest_path
    )
    source_profile_path = Path(
        str(
            source_context.execution_locators[
                "stage1_build_options"
            ]["config_path"]
        )
    ).resolve(strict=True)
    exact_request = source_context.execution_locators[
        "exact_stage1_request"
    ]
    plan_value = exact_request.get("stage1_scope_plan")
    if not isinstance(plan_value, Mapping):
        raise ValueError(
            "accepted prepared context lacks its canonical scope plan"
        )
    plan = stage1_scope_plan_from_mapping(plan_value)
    if artifact.audit.get("physical_scope_order") != [
        scope.scope_id for scope in plan.physical_scopes
    ]:
        raise ValueError(
            "accepted reusable preflight differs from its canonical plan"
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{key}.attempt-",
            dir=target.parent,
        )
    )
    try:
        copied_context_root = (
            temporary / "prepared_stage1_context"
        )
        shutil.copytree(
            source_context.root,
            copied_context_root,
            copy_function=shutil.copy2,
        )
        shutil.copy2(
            source_profile_path,
            temporary / "effective_stage1_profile.json",
        )
        copied_context = load_prepared_stage1_context(
            copied_context_root
            / "prepared_stage1_context_manifest.json"
        )
        if (
            copied_context.content_root_sha256
            != source_context.content_root_sha256
        ):
            raise RuntimeError(
                "accepted prepared context changed while copying"
            )
        files = [
            _file_registration(
                path,
                root=temporary,
                kind="prepared_stage1_context_small_payload",
            )
            for path in sorted(copied_context_root.rglob("*"))
            if path.is_file()
        ]
        files.append(
            _file_registration(
                temporary / "effective_stage1_profile.json",
                root=temporary,
                kind="effective_stage1_profile_small_payload",
            )
        )
        identity = artifact.identity()
        body = {
            "schema_version": REUSABLE_ACCEPTED_CONTEXT_SCHEMA,
            "status": "complete",
            "accepted_input_selector": _json_copy(
                selector,
                label="accepted input selector",
            ),
            "accepted_input_selector_key": key,
            "assembled_scientific_key": (
                artifact.manifest_path.parent.name
            ),
            "assembled_terminal_content_sha256": _read_json(
                artifact.manifest_path,
                label="accepted assembled terminal",
            )["content_sha256"],
            "assembled_scientific_content_sha256": identity[
                "path_neutral_scientific_content_sha256"
            ],
            "prepared_context_manifest_relative_path": (
                "prepared_stage1_context/"
                "prepared_stage1_context_manifest.json"
            ),
            "prepared_context_scientific_content_root_sha256": (
                source_context.content_root_sha256
            ),
            "effective_profile_relative_path": (
                "effective_stage1_profile.json"
            ),
            "preflight_plan_content_sha256": (
                preflight_scope_plan_projection(plan)[
                    "content_sha256"
                ]
            ),
            "producer_identity": str(producer_identity),
            "owner_producer_identity": str(
                owner_producer_identity
            ),
            "global_audit_producer_identity": str(
                global_audit_producer_identity
            ),
            "files": files,
            "assembled_and_owner_payloads_copied": False,
            "prepared_context_payload_is_small": True,
            "operational_paths_in_selector": False,
            "stage2_identity_in_selector": False,
        }
        _write_new_json(
            temporary / ACCEPTED_CONTEXT_TERMINAL,
            {**body, "content_sha256": _sha256_json(body)},
        )
        _close_tree_read_only(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.chmod(0o700)
            for path in temporary.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_reusable_preflight_acceptance(
        store_root=store,
        selector=selector,
        producer_identity=producer_identity,
        owner_producer_identity=owner_producer_identity,
        global_audit_producer_identity=(
            global_audit_producer_identity
        ),
    )


def load_reusable_preflight_acceptance(
    *,
    store_root: Path | str,
    selector: Mapping[str, Any],
    producer_identity: str,
    owner_producer_identity: str,
    global_audit_producer_identity: str,
) -> ReusablePreflightAcceptance:
    """Reopen an unchanged preflight from manifests/stats, never cohort data."""

    from .prepared_stage1_context import (
        load_prepared_stage1_context,
    )

    started = time.perf_counter()
    store = _store_root(store_root)
    key = _accepted_context_key(selector)
    root = (store / "accepted_contexts" / key).resolve(strict=True)
    terminal_path = root / ACCEPTED_CONTEXT_TERMINAL
    start_probe = _authentication_probe_ctime_ns(store)
    terminal_sha, terminal_size = _stable_file_sha256(
        terminal_path
    )
    terminal = _read_json(
        terminal_path,
        label="reusable preflight accepted-context terminal",
    )
    _manifest_body(terminal)
    assembled_key = str(
        terminal.get("assembled_scientific_key", "")
    )
    try:
        _require_sha256(
            assembled_key,
            label="accepted assembled scientific key",
        )
    except ValueError:
        assembled_key = ""
    assembled_path = (
        store
        / "assembled_contexts"
        / assembled_key
        / ASSEMBLED_TERMINAL
    )
    if (
        terminal.get("schema_version")
        != REUSABLE_ACCEPTED_CONTEXT_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("accepted_input_selector")
        != _json_copy(selector, label="accepted input selector")
        or terminal.get("accepted_input_selector_key") != key
        or terminal.get("producer_identity")
        != str(producer_identity)
        or terminal.get("owner_producer_identity")
        != str(owner_producer_identity)
        or terminal.get("global_audit_producer_identity")
        != str(global_audit_producer_identity)
        or terminal.get("assembled_and_owner_payloads_copied")
        is not False
        or terminal.get("prepared_context_payload_is_small")
        is not True
        or terminal.get("operational_paths_in_selector") is not False
        or terminal.get("stage2_identity_in_selector") is not False
        or not assembled_key
        or "assembled_terminal_path" in terminal
    ):
        raise ValueError(
            "reusable preflight accepted-context terminal is incompatible"
        )
    proof = _load_fast_proof(
        store_root=store,
        artifact_kind="accepted_context",
        scientific_key=key,
        artifact_root=root,
        terminal_content_sha256=terminal["content_sha256"],
        producer_identity=str(producer_identity),
        schema_identity=REUSABLE_ACCEPTED_CONTEXT_SCHEMA,
    )
    (
        _registrations,
        payload_bytes,
        authenticated_bytes,
    ) = _validate_registered_tree(
        root=root,
        terminal_name=ACCEPTED_CONTEXT_TERMINAL,
        files=terminal["files"],
        read_payload_bytes=proof is None,
    )
    if proof is None:
        authenticated_bytes[ACCEPTED_CONTEXT_TERMINAL] = (
            terminal_sha,
            terminal_size,
        )
    context_path = root / str(
        terminal["prepared_context_manifest_relative_path"]
    )
    effective_profile_path = root / str(
        terminal["effective_profile_relative_path"]
    )
    if (
        effective_profile_path.name
        != "effective_stage1_profile.json"
        or not effective_profile_path.is_file()
        or effective_profile_path.is_symlink()
    ):
        raise ValueError(
            "accepted effective Stage 1 profile is invalid"
        )
    prepared_context = load_prepared_stage1_context(context_path)
    if (
        prepared_context.content_root_sha256
        != terminal[
            "prepared_context_scientific_content_root_sha256"
        ]
    ):
        raise ValueError(
            "accepted prepared-context scientific identity changed"
        )
    exact_request = prepared_context.execution_locators[
        "exact_stage1_request"
    ]
    plan = stage1_scope_plan_from_mapping(
        exact_request["stage1_scope_plan"]
    )
    if (
        preflight_scope_plan_projection(plan)["content_sha256"]
        != terminal["preflight_plan_content_sha256"]
    ):
        raise ValueError(
            "accepted preflight scope plan changed"
        )
    assembled = _read_json(
        assembled_path,
        label="accepted assembled preflight terminal",
    )
    global_locator = _read_json(
        assembled_path.parent / "global_audit_locator.json",
        label="accepted global audit locator",
    )
    global_terminal = _read_json(
        (
            store
            / "global_audits"
            / str(global_locator["global_scientific_key"])
            / GLOBAL_TERMINAL
        ).resolve(strict=True),
        label="accepted global audit terminal",
    )
    global_audit = load_reusable_global_audit(
        store_root=store,
        compatibility=global_terminal["compatibility"],
        producer_identity=global_audit_producer_identity,
    )
    preflight, state_bundle = load_reusable_assembled_preflight(
        store_root=store,
        compatibility=assembled["compatibility"],
        expected_stage1_request=None,
        global_audit=global_audit,
        plan=plan,
        producer_identity=producer_identity,
        owner_producer_identity=owner_producer_identity,
        global_audit_producer_identity=(
            global_audit_producer_identity
        ),
    )
    if (
        preflight.manifest_path != assembled_path
        or assembled["content_sha256"]
        != terminal["assembled_terminal_content_sha256"]
        or preflight.identity()[
            "path_neutral_scientific_content_sha256"
        ]
        != terminal["assembled_scientific_content_sha256"]
    ):
        raise ValueError(
            "accepted assembled preflight binding changed"
        )
    if proof is None:
        _publish_optional_full_auth_proof(
            store_root=store,
            artifact_kind="accepted_context",
            scientific_key=key,
            artifact_root=root,
            terminal_content_sha256=terminal["content_sha256"],
            artifact_scientific_content_sha256=(
                terminal[
                    "prepared_context_scientific_content_root_sha256"
                ]
            ),
            producer_identity=str(producer_identity),
            schema_identity=REUSABLE_ACCEPTED_CONTEXT_SCHEMA,
            full_authentication_start_probe_ctime_ns=int(
                start_probe
            ),
            authenticated_byte_inventory=authenticated_bytes,
        )
    return ReusablePreflightAcceptance(
        root=root,
        terminal=MappingProxyType(copy.deepcopy(terminal)),
        prepared_context=prepared_context,
        effective_profile_path=effective_profile_path,
        preflight=preflight,
        state_bundle=state_bundle,
        authentication_mode=(
            "prior_proof_stat_continuity"
            if proof is not None
            else "full_byte_reauthentication"
        ),
        authentication_seconds=time.perf_counter() - started,
        payload_bytes_read=terminal_size + payload_bytes,
        global_audit_authentication_mode=(
            global_audit.authentication_mode
        ),
        global_audit_authentication_seconds=(
            global_audit.authentication_seconds
        ),
        global_audit_payload_bytes_read=(
            global_audit.payload_bytes_read
        ),
    )


def try_load_reusable_preflight_acceptance(
    *,
    store_root: Path | str,
    selector: Mapping[str, Any],
    producer_identity: str,
    owner_producer_identity: str,
    global_audit_producer_identity: str,
) -> ReusablePreflightAcceptance | None:
    """Open one accepted index or preserve/quarantine it before fallback."""

    try:
        return load_reusable_preflight_acceptance(
            store_root=store_root,
            selector=selector,
            producer_identity=producer_identity,
            owner_producer_identity=owner_producer_identity,
            global_audit_producer_identity=(
                global_audit_producer_identity
            ),
        )
    except FileNotFoundError:
        return None
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _quarantine_invalid_artifact(
            store_root=store_root,
            artifact_directory="accepted_contexts",
            scientific_key_value=_accepted_context_key(selector),
            failure=exc,
        )
        return None


__all__ = [
    "ACCEPTED_CONTEXT_TERMINAL",
    "ASSEMBLED_TERMINAL",
    "ASSEMBLED_COMPATIBILITY_SCHEMA",
    "GLOBAL_TERMINAL",
    "GLOBAL_AUDIT_COMPATIBILITY_SCHEMA",
    "OWNER_TERMINAL",
    "OWNER_COMPATIBILITY_SCHEMA",
    "REFERENCE_MANIFEST",
    "REUSABLE_ASSEMBLED_ARTIFACT_SCHEMA",
    "REUSABLE_ACCEPTED_CONTEXT_SCHEMA",
    "REUSABLE_GLOBAL_AUDIT_SCHEMA",
    "REUSABLE_OWNER_ARTIFACT_SCHEMA",
    "REUSABLE_PREFLIGHT_REFERENCE_SCHEMA",
    "REUSABLE_STATE_BUNDLE_REFERENCE_SCHEMA",
    "ReusableClusterPreflightStateBundle",
    "ReusablePreflightAcceptance",
    "ReusableGlobalAuditArtifact",
    "ReusableOwnerArtifact",
    "ReusableProductionStage1ClusterPreflightArtifact",
    "assembled_compatibility",
    "captured_state_from_authenticated_canonical_state",
    "is_reusable_preflight_reference",
    "load_reusable_assembled_preflight",
    "load_reusable_global_audit",
    "load_reusable_owner_artifact",
    "load_reusable_owner_cluster_state",
    "load_reusable_preflight_reference",
    "load_reusable_preflight_acceptance",
    "load_reusable_state_bundle_reference",
    "owner_compatibility",
    "publish_reusable_preflight_references",
    "publish_reusable_preflight_acceptance",
    "scientific_key",
    "seal_reusable_assembled_preflight",
    "seal_reusable_global_audit",
    "seal_reusable_global_audit_from_authenticated_legacy",
    "seal_reusable_owner_artifact",
    "try_load_reusable_assembled_preflight",
    "try_load_reusable_global_audit",
    "try_load_reusable_owner_artifact",
    "try_load_reusable_preflight_acceptance",
    "preflight_scope_plan_projection",
    "stage1_scope_plan_from_mapping",
]
