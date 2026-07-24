"""Closed per-scope fragments and canonical merge for legacy Stage 1.

The production Stage 1 scope scheduler owns process isolation, GPU assignment,
and sealed-attempt resume.  This module owns the publication boundary between
those attempts and the legacy component finalizer:

* one scope emits one immutable fragment;
* every artifact declares its eventual component-relative destination;
* all fragments are reopened and authenticated before a merge tree is created;
* duplicate and ancestor/descendant destination collisions are rejected; and
* the merge manifest is written last after canonical-order accumulator output.

Scientific component indexes are intentionally represented as per-scope
accumulator payloads here.  The legacy component finalizer consumes those
payloads only after this module has proved complete canonical scope coverage.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from .production_stage1_scope_scheduler import Stage1ScopePlan, Stage1ScopeSpec

LEGACY_STAGE1_SCOPE_ACCUMULATOR_SCHEMA = "production_legacy_stage1_scope_accumulator_v1"
LEGACY_STAGE1_SCOPE_FRAGMENT_SCHEMA = "production_legacy_stage1_scope_fragment_v1"
LEGACY_STAGE1_FRAGMENT_MERGE_SCHEMA = "production_legacy_stage1_fragment_merge_v1"
LEGACY_STAGE1_FRAGMENT_MERGE_ACCUMULATORS_SCHEMA = (
    "production_legacy_stage1_fragment_merge_accumulators_v1"
)

_FRAGMENT_MANIFEST_NAME = "fragment_manifest.json"
_ACCUMULATOR_NAME = "scope_accumulator.json"
_ARTIFACT_DIRECTORY_NAME = "artifacts"
_MERGE_MANIFEST_NAME = "merge_manifest.json"
_MERGE_ACCUMULATORS_NAME = "scope_accumulators.json"
_HEX = frozenset("0123456789abcdef")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"fragment path is not a regular file: {path}")
        if int(before.st_nlink) != 1:
            raise ValueError(f"fragment files cannot be hard linked: {path}")
        digest = hashlib.sha256()
        total = 0
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            total += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if tuple(getattr(before, field) for field in identity_fields) != tuple(
        getattr(after, field) for field in identity_fields
    ) or total != int(after.st_size):
        raise RuntimeError(f"fragment file changed while hashing: {path}")
    named = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(named.st_mode)
        or int(named.st_nlink) != 1
        or (int(named.st_dev), int(named.st_ino)) != (int(after.st_dev), int(after.st_ino))
    ):
        raise RuntimeError(f"fragment path was substituted while hashing: {path}")
    return digest.hexdigest(), int(after.st_size)


def _read_regular_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ValueError(f"{label} must be one singly-linked regular file")
        if int(before.st_size) > int(maximum_bytes):
            raise ValueError(f"{label} is unexpectedly large")
        payload = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            payload.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if tuple(getattr(before, field) for field in identity_fields) != tuple(
        getattr(after, field) for field in identity_fields
    ) or len(payload) != int(after.st_size):
        raise RuntimeError(f"{label} changed while reading")
    named = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(named.st_mode)
        or int(named.st_nlink) != 1
        or (int(named.st_dev), int(named.st_ino)) != (int(after.st_dev), int(after.st_ino))
    ):
        raise RuntimeError(f"{label} path was substituted while reading")
    return bytes(payload)


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _closed_json(value: Any, *, path: str = "value") -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if not key or key in output:
                raise ValueError(f"{path} has an empty or colliding key")
            output[key] = _closed_json(child, path=f"{path}.{key}")
        return output
    if isinstance(value, (tuple, list)):
        return [_closed_json(child, path=f"{path}[{index}]") for index, child in enumerate(value)]
    if isinstance(value, np.generic):
        return _closed_json(value.item(), path=path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    payload = _read_regular_file(path, label=label)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite constant {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"durability target is not a directory: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_regular_file(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ValueError(f"durability target is not a singly-linked regular file: {path}")
        os.fsync(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if tuple(getattr(before, field) for field in identity_fields) != tuple(
        getattr(after, field) for field in identity_fields
    ):
        raise RuntimeError(f"durability target changed while syncing: {path}")


def durably_sync_legacy_stage1_tree(root: Path | str) -> None:
    """Fsync every closed-tree payload, then every directory bottom-up."""

    tree = _regular_directory(Path(root), label="legacy Stage 1 durability root")
    files: list[Path] = []
    directories: list[Path] = [tree]
    for path in tree.rglob("*"):
        observed = os.stat(path, follow_symlinks=False)
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError("legacy Stage 1 durability tree cannot contain symlinks")
        if stat.S_ISDIR(observed.st_mode):
            directories.append(path)
        elif stat.S_ISREG(observed.st_mode) and int(observed.st_nlink) == 1:
            files.append(path)
        else:
            raise ValueError(
                "legacy Stage 1 durability tree contains a special or multiply-linked file"
            )
    for path in sorted(files):
        _fsync_regular_file(path)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(path)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable fragment file already exists: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_relative_path(value: str, *, label: str) -> str:
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} is not one canonical relative path: {text!r}")
    return text


def _regular_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be one regular directory")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{label} path must be canonical")
    return resolved


def _artifact_inventory(fragment_root: Path) -> list[dict[str, Any]]:
    artifacts = fragment_root / _ARTIFACT_DIRECTORY_NAME
    _regular_directory(artifacts, label="fragment artifact directory")
    rows: list[dict[str, Any]] = []
    for path in sorted(artifacts.rglob("*")):
        if path.is_symlink():
            raise ValueError("legacy Stage 1 fragments cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("legacy Stage 1 fragment contains a special file")
        physical_relative = path.relative_to(fragment_root).as_posix()
        merge_relative = path.relative_to(artifacts).as_posix()
        _canonical_relative_path(physical_relative, label="fragment artifact physical path")
        _canonical_relative_path(merge_relative, label="fragment artifact merge path")
        digest, size = _sha256_file(path)
        rows.append(
            {
                "fragment_relative_path": physical_relative,
                "merge_relative_path": merge_relative,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    if not rows:
        raise ValueError("legacy Stage 1 scope fragment has no artifacts")
    merge_paths = [str(row["merge_relative_path"]) for row in rows]
    if len(merge_paths) != len(set(merge_paths)):
        raise ValueError("legacy Stage 1 fragment has duplicate merge paths")
    return rows


def _root_tree_inventory(root: Path) -> tuple[set[str], set[str]]:
    before = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("legacy Stage 1 fragment tree root is not a directory")
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        observed = os.stat(path, follow_symlinks=False)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError("legacy Stage 1 fragment tree cannot contain symlinks")
        if stat.S_ISREG(observed.st_mode):
            if int(observed.st_nlink) != 1:
                raise ValueError("legacy Stage 1 fragment tree cannot contain hard links")
            files.add(relative)
        elif stat.S_ISDIR(observed.st_mode):
            directories.add(relative)
        else:
            raise ValueError("legacy Stage 1 fragment tree contains a special file")
    after = os.stat(root, follow_symlinks=False)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if tuple(getattr(before, field) for field in identity_fields) != tuple(
        getattr(after, field) for field in identity_fields
    ):
        raise RuntimeError("legacy Stage 1 fragment tree changed during inventory")
    return files, directories


def _parent_directories(relative_files: Sequence[str]) -> set[str]:
    directories: set[str] = set()
    for relative in relative_files:
        path = PurePosixPath(relative)
        for parent in path.parents:
            if parent.as_posix() != ".":
                directories.add(parent.as_posix())
    return directories


@dataclass(frozen=True)
class LegacyStage1ScopeFragment:
    root: Path
    manifest_path: Path
    scope_id: str
    manifest_content_sha256: str
    accumulator: Mapping[str, Any]
    artifacts: tuple[Mapping[str, Any], ...]

    def identity(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "scope_id": self.scope_id,
            "manifest_content_sha256": self.manifest_content_sha256,
            "artifact_count": len(self.artifacts),
        }


def _resolve_fragment_scope_authority(
    *,
    plan: Stage1ScopePlan | None,
    scope_authority: Stage1ScopeSpec | None,
    plan_content_sha256: str | None,
    scope_id: str,
) -> tuple[Stage1ScopeSpec, str]:
    """Resolve a complete parent plan or one closed worker-scope authority."""

    if (plan is None) == (scope_authority is None):
        raise TypeError("supply exactly one of the complete plan or one scope authority")
    if plan is not None:
        if not isinstance(plan, Stage1ScopePlan):
            raise TypeError("plan must be a Stage1ScopePlan")
        if plan_content_sha256 is not None:
            raise TypeError("plan_content_sha256 is implicit when the complete plan is supplied")
        return plan.scope(str(scope_id)), plan.content_sha256
    if not isinstance(scope_authority, Stage1ScopeSpec):
        raise TypeError("scope_authority must be a Stage1ScopeSpec")
    if scope_authority.scope_id != str(scope_id):
        raise ValueError("scope authority belongs to another scope")
    return scope_authority, _require_sha256(
        plan_content_sha256,
        label="plan_content_sha256",
    )


def seal_legacy_stage1_scope_fragment(
    *,
    fragment_root: Path | str,
    plan: Stage1ScopePlan | None = None,
    scope_authority: Stage1ScopeSpec | None = None,
    plan_content_sha256: str | None = None,
    scope_id: str,
    stage1_request_sha256: str,
    scope_attempt_request_sha256: str,
    accumulator: Mapping[str, Any],
) -> LegacyStage1ScopeFragment:
    """Seal an existing ``artifacts/`` tree and write its manifest last."""

    scope, resolved_plan_sha256 = _resolve_fragment_scope_authority(
        plan=plan,
        scope_authority=scope_authority,
        plan_content_sha256=plan_content_sha256,
        scope_id=scope_id,
    )
    root = _regular_directory(Path(fragment_root), label="fragment root")
    if set(path.name for path in root.iterdir()) != {_ARTIFACT_DIRECTORY_NAME}:
        raise ValueError("unsealed fragment root must contain only its artifacts directory")
    request_sha = _require_sha256(stage1_request_sha256, label="stage1_request_sha256")
    attempt_sha = _require_sha256(
        scope_attempt_request_sha256,
        label="scope_attempt_request_sha256",
    )
    payload = _closed_json(dict(accumulator), path="scope_accumulator")
    accumulator_body = {
        "schema_version": LEGACY_STAGE1_SCOPE_ACCUMULATOR_SCHEMA,
        "scope_id": scope.scope_id,
        "scope_kind": scope.scope_kind,
        "canonical_index": scope.canonical_index,
        "payload": payload,
    }
    accumulator_value = {
        **accumulator_body,
        "content_sha256": _sha256_json(accumulator_body),
    }
    accumulator_path = root / _ACCUMULATOR_NAME
    _write_new_json(accumulator_path, accumulator_value)
    accumulator_sha, accumulator_size = _sha256_file(accumulator_path)
    artifacts = _artifact_inventory(root)
    body = {
        "schema_version": LEGACY_STAGE1_SCOPE_FRAGMENT_SCHEMA,
        "status": "complete",
        "plan_content_sha256": resolved_plan_sha256,
        "stage1_request_sha256": request_sha,
        "scope_attempt_request_sha256": attempt_sha,
        "scope": scope.as_dict(),
        "heldout_labels_supplied": False,
        "accumulator": {
            "relative_path": _ACCUMULATOR_NAME,
            "sha256": accumulator_sha,
            "size_bytes": accumulator_size,
            "content_sha256": accumulator_value["content_sha256"],
        },
        "artifacts": artifacts,
    }
    manifest = {**body, "content_sha256": _sha256_json(body)}
    # Every artifact and its directory entry must reach stable storage before
    # the sole terminal marker can become visible.
    durably_sync_legacy_stage1_tree(root)
    # This file is the sole terminal marker and is always written last.
    _write_new_json(root / _FRAGMENT_MANIFEST_NAME, manifest)
    return validate_legacy_stage1_scope_fragment(
        fragment_root=root,
        plan=plan,
        scope_authority=scope_authority,
        plan_content_sha256=plan_content_sha256,
        scope_id=scope.scope_id,
        stage1_request_sha256=request_sha,
        scope_attempt_request_sha256=attempt_sha,
    )


def validate_legacy_stage1_scope_fragment(
    *,
    fragment_root: Path | str,
    plan: Stage1ScopePlan | None = None,
    scope_authority: Stage1ScopeSpec | None = None,
    plan_content_sha256: str | None = None,
    scope_id: str,
    stage1_request_sha256: str,
    scope_attempt_request_sha256: str,
) -> LegacyStage1ScopeFragment:
    """Reopen and authenticate one complete scope fragment."""

    scope, resolved_plan_sha256 = _resolve_fragment_scope_authority(
        plan=plan,
        scope_authority=scope_authority,
        plan_content_sha256=plan_content_sha256,
        scope_id=scope_id,
    )
    request_sha = _require_sha256(stage1_request_sha256, label="stage1_request_sha256")
    attempt_sha = _require_sha256(
        scope_attempt_request_sha256,
        label="scope_attempt_request_sha256",
    )
    root = _regular_directory(Path(fragment_root), label="fragment root")
    if set(path.name for path in root.iterdir()) != {
        _ARTIFACT_DIRECTORY_NAME,
        _ACCUMULATOR_NAME,
        _FRAGMENT_MANIFEST_NAME,
    }:
        raise ValueError("sealed fragment root has missing or unregistered entries")
    manifest_path = root / _FRAGMENT_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("scope fragment terminal manifest is absent")
    manifest = _read_json(manifest_path, label="scope fragment terminal manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("scope fragment terminal manifest must be one object")
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    if (
        set(manifest)
        != {
            "schema_version",
            "status",
            "plan_content_sha256",
            "stage1_request_sha256",
            "scope_attempt_request_sha256",
            "scope",
            "heldout_labels_supplied",
            "accumulator",
            "artifacts",
            "content_sha256",
        }
        or manifest.get("schema_version") != LEGACY_STAGE1_SCOPE_FRAGMENT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("plan_content_sha256") != resolved_plan_sha256
        or manifest.get("stage1_request_sha256") != request_sha
        or manifest.get("scope_attempt_request_sha256") != attempt_sha
        or manifest.get("scope") != scope.as_dict()
        or manifest.get("heldout_labels_supplied") is not False
        or manifest.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("scope fragment terminal manifest has an invalid binding")
    accumulator_registration = manifest.get("accumulator")
    if (
        not isinstance(accumulator_registration, Mapping)
        or set(accumulator_registration)
        != {"relative_path", "sha256", "size_bytes", "content_sha256"}
        or accumulator_registration.get("relative_path") != _ACCUMULATOR_NAME
    ):
        raise ValueError("scope fragment accumulator registration is invalid")
    accumulator_path = root / _ACCUMULATOR_NAME
    accumulator_sha, accumulator_size = _sha256_file(accumulator_path)
    accumulator = _read_json(accumulator_path, label="scope fragment accumulator")
    if not isinstance(accumulator, Mapping):
        raise ValueError("scope fragment accumulator must be one object")
    accumulator_body = {
        key: copy.deepcopy(value) for key, value in accumulator.items() if key != "content_sha256"
    }
    if (
        set(accumulator)
        != {
            "schema_version",
            "scope_id",
            "scope_kind",
            "canonical_index",
            "payload",
            "content_sha256",
        }
        or accumulator.get("schema_version") != LEGACY_STAGE1_SCOPE_ACCUMULATOR_SCHEMA
        or accumulator.get("scope_id") != scope.scope_id
        or accumulator.get("scope_kind") != scope.scope_kind
        or accumulator.get("canonical_index") != scope.canonical_index
        or accumulator.get("content_sha256") != _sha256_json(accumulator_body)
        or dict(accumulator_registration)
        != {
            "relative_path": _ACCUMULATOR_NAME,
            "sha256": accumulator_sha,
            "size_bytes": accumulator_size,
            "content_sha256": accumulator.get("content_sha256"),
        }
    ):
        raise ValueError("scope fragment accumulator changed or was substituted")
    artifacts = _artifact_inventory(root)
    if manifest.get("artifacts") != artifacts:
        raise ValueError("scope fragment artifact inventory changed")
    expected_files = {
        _FRAGMENT_MANIFEST_NAME,
        _ACCUMULATOR_NAME,
        *(str(row["fragment_relative_path"]) for row in artifacts),
    }
    observed_files, observed_directories = _root_tree_inventory(root)
    if observed_files != expected_files or observed_directories != _parent_directories(
        tuple(expected_files)
    ):
        raise ValueError("scope fragment contains unregistered entries")
    return LegacyStage1ScopeFragment(
        root=root,
        manifest_path=manifest_path,
        scope_id=scope.scope_id,
        manifest_content_sha256=str(manifest["content_sha256"]),
        accumulator=copy.deepcopy(dict(accumulator)),
        artifacts=tuple(copy.deepcopy(artifacts)),
    )


def _validate_collision_free_paths(
    rows: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    ordered = sorted(
        (
            PurePosixPath(str(registration["merge_relative_path"])),
            scope_id,
        )
        for scope_id, registration in rows
    )
    for (previous, previous_scope), (path, scope_id) in zip(ordered, ordered[1:], strict=False):
        if path == previous or previous in path.parents:
            raise ValueError(
                "legacy fragment merge path collision: "
                f"{previous_scope}:{previous.as_posix()} and "
                f"{scope_id}:{path.as_posix()}"
            )


def _fragment_identity_rows(
    *,
    plan: Stage1ScopePlan,
    stage1_request_sha256: str,
    fragment_roots_by_scope: Mapping[str, Path | str],
    scope_attempt_request_sha256_by_scope: Mapping[str, str],
    require_production_coverage: bool,
) -> tuple[list[LegacyStage1ScopeFragment], list[dict[str, Any]]]:
    expected = tuple(scope.scope_id for scope in plan.scopes)
    if set(fragment_roots_by_scope) != set(expected):
        missing = sorted(set(expected) - set(fragment_roots_by_scope))
        extra = sorted(set(fragment_roots_by_scope) - set(expected))
        raise ValueError(
            f"legacy fragment scope coverage differs: missing={missing}, extra={extra}"
        )
    if set(scope_attempt_request_sha256_by_scope) != set(expected):
        raise ValueError("scope attempt request identities have incomplete coverage")
    if require_production_coverage:
        kind_counts = {
            kind: sum(scope.scope_kind == kind for scope in plan.scopes)
            for kind in ("full_outer", "exact_inner", "cumulative_spent")
        }
        if kind_counts != {
            "full_outer": 5,
            "exact_inner": 25,
            "cumulative_spent": 10,
        }:
            raise ValueError("production legacy fragment merge requires exact 5/25/10 coverage")
    fragments: list[LegacyStage1ScopeFragment] = []
    identities: list[dict[str, Any]] = []
    artifact_rows: list[tuple[str, Mapping[str, Any]]] = []
    for scope in plan.scopes:
        fragment = validate_legacy_stage1_scope_fragment(
            fragment_root=fragment_roots_by_scope[scope.scope_id],
            plan=plan,
            scope_id=scope.scope_id,
            stage1_request_sha256=stage1_request_sha256,
            scope_attempt_request_sha256=scope_attempt_request_sha256_by_scope[scope.scope_id],
        )
        fragments.append(fragment)
        identities.append(
            {
                **fragment.identity(),
                "scope": scope.as_dict(),
                "scope_attempt_request_sha256": (
                    scope_attempt_request_sha256_by_scope[scope.scope_id]
                ),
            }
        )
        artifact_rows.extend((scope.scope_id, registration) for registration in fragment.artifacts)
    _validate_collision_free_paths(artifact_rows)
    return fragments, identities


def merge_legacy_stage1_scope_fragments(
    *,
    plan: Stage1ScopePlan,
    stage1_request_sha256: str,
    fragment_roots_by_scope: Mapping[str, Path | str],
    scope_attempt_request_sha256_by_scope: Mapping[str, str],
    destination_root: Path | str,
    require_production_coverage: bool = True,
) -> Mapping[str, Any]:
    """Validate every fragment, then atomically publish one collision-safe merge."""

    request_sha = _require_sha256(stage1_request_sha256, label="stage1_request_sha256")
    destination = Path(destination_root)
    if not destination.is_absolute():
        raise ValueError("legacy fragment merge destination must be absolute")
    if destination.is_symlink() or destination.exists():
        raise FileExistsError("legacy fragment merge destination must be fresh")
    parent = _regular_directory(destination.parent, label="merge destination parent")
    if parent != destination.parent:
        raise ValueError("legacy fragment merge parent must be canonical")

    # This is deliberately completed before mkdir/mkdtemp: no aggregate output
    # exists until the complete canonical fragment set has authenticated.
    fragments, fragment_identities = _fragment_identity_rows(
        plan=plan,
        stage1_request_sha256=request_sha,
        fragment_roots_by_scope=fragment_roots_by_scope,
        scope_attempt_request_sha256_by_scope=(scope_attempt_request_sha256_by_scope),
        require_production_coverage=require_production_coverage,
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.merging-", dir=parent))
    try:
        copied: list[dict[str, Any]] = []
        for scope, fragment in zip(plan.scopes, fragments, strict=True):
            for registration in fragment.artifacts:
                relative = _canonical_relative_path(
                    str(registration["merge_relative_path"]),
                    label="merge destination",
                )
                source = fragment.root / str(registration["fragment_relative_path"])
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    raise RuntimeError(f"validated merge collision reached copy phase: {relative}")
                shutil.copyfile(source, target, follow_symlinks=False)
                _fsync_regular_file(target)
                digest, size = _sha256_file(target)
                if digest != registration["sha256"] or size != int(registration["size_bytes"]):
                    raise RuntimeError(f"legacy fragment bytes changed during merge: {relative}")
                copied.append(
                    {
                        "scope_id": scope.scope_id,
                        "relative_path": relative,
                        "sha256": digest,
                        "size_bytes": size,
                    }
                )
        accumulator_body = {
            "schema_version": (LEGACY_STAGE1_FRAGMENT_MERGE_ACCUMULATORS_SCHEMA),
            "plan_content_sha256": plan.content_sha256,
            "stage1_request_sha256": request_sha,
            "canonical_scope_order": [scope.scope_id for scope in plan.scopes],
            "scope_count": len(fragments),
            "scopes": [
                {
                    "scope": scope.as_dict(),
                    "fragment_manifest_content_sha256": (fragment.manifest_content_sha256),
                    "accumulator": copy.deepcopy(dict(fragment.accumulator)),
                }
                for scope, fragment in zip(plan.scopes, fragments, strict=True)
            ],
        }
        accumulator_value = {
            **accumulator_body,
            "content_sha256": _sha256_json(accumulator_body),
        }
        accumulator_path = temporary / _MERGE_ACCUMULATORS_NAME
        _write_new_json(accumulator_path, accumulator_value)
        accumulator_sha, accumulator_size = _sha256_file(accumulator_path)
        body = {
            "schema_version": LEGACY_STAGE1_FRAGMENT_MERGE_SCHEMA,
            "status": "complete",
            "plan_content_sha256": plan.content_sha256,
            "stage1_request_sha256": request_sha,
            "canonical_scope_order": [scope.scope_id for scope in plan.scopes],
            "scope_count": len(fragments),
            "production_coverage_required": bool(require_production_coverage),
            "fragments": fragment_identities,
            "copied_files": copied,
            "scope_accumulators": {
                "relative_path": _MERGE_ACCUMULATORS_NAME,
                "sha256": accumulator_sha,
                "size_bytes": accumulator_size,
                "content_sha256": accumulator_value["content_sha256"],
            },
            "heldout_labels_supplied_to_workers": False,
        }
        manifest = {**body, "content_sha256": _sha256_json(body)}
        # The terminal manifest is meaningful only after all copied artifacts,
        # the canonical accumulator, and their directory entries are durable.
        durably_sync_legacy_stage1_tree(temporary)
        # Terminal publication record is the final file written.
        _write_new_json(temporary / _MERGE_MANIFEST_NAME, manifest)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_legacy_stage1_fragment_merge(
        plan=plan,
        stage1_request_sha256=request_sha,
        fragment_roots_by_scope=fragment_roots_by_scope,
        scope_attempt_request_sha256_by_scope=(scope_attempt_request_sha256_by_scope),
        destination_root=destination,
        require_production_coverage=require_production_coverage,
    )


def validate_legacy_stage1_fragment_merge(
    *,
    plan: Stage1ScopePlan,
    stage1_request_sha256: str,
    fragment_roots_by_scope: Mapping[str, Path | str],
    scope_attempt_request_sha256_by_scope: Mapping[str, str],
    destination_root: Path | str,
    require_production_coverage: bool = True,
) -> Mapping[str, Any]:
    """Fresh read-only validation of a terminal fragment merge."""

    request_sha = _require_sha256(stage1_request_sha256, label="stage1_request_sha256")
    fragments, fragment_identities = _fragment_identity_rows(
        plan=plan,
        stage1_request_sha256=request_sha,
        fragment_roots_by_scope=fragment_roots_by_scope,
        scope_attempt_request_sha256_by_scope=(scope_attempt_request_sha256_by_scope),
        require_production_coverage=require_production_coverage,
    )
    root = _regular_directory(Path(destination_root), label="fragment merge root")
    manifest_path = root / _MERGE_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("legacy fragment merge terminal manifest is absent")
    manifest = _read_json(manifest_path, label="legacy fragment merge manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("legacy fragment merge manifest must be one object")
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    if (
        set(manifest)
        != {
            "schema_version",
            "status",
            "plan_content_sha256",
            "stage1_request_sha256",
            "canonical_scope_order",
            "scope_count",
            "production_coverage_required",
            "fragments",
            "copied_files",
            "scope_accumulators",
            "heldout_labels_supplied_to_workers",
            "content_sha256",
        }
        or manifest.get("schema_version") != LEGACY_STAGE1_FRAGMENT_MERGE_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("plan_content_sha256") != plan.content_sha256
        or manifest.get("stage1_request_sha256") != request_sha
        or manifest.get("canonical_scope_order") != [scope.scope_id for scope in plan.scopes]
        or manifest.get("scope_count") != len(plan.scopes)
        or manifest.get("production_coverage_required") is not bool(require_production_coverage)
        or manifest.get("fragments") != fragment_identities
        or manifest.get("heldout_labels_supplied_to_workers") is not False
        or manifest.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("legacy fragment merge manifest has an invalid binding")
    copied = manifest.get("copied_files")
    if not isinstance(copied, list):
        raise ValueError("legacy fragment merge copied-file inventory is invalid")
    expected_copied: list[dict[str, Any]] = []
    expected_files = {_MERGE_MANIFEST_NAME, _MERGE_ACCUMULATORS_NAME}
    for scope, fragment in zip(plan.scopes, fragments, strict=True):
        for registration in fragment.artifacts:
            relative = str(registration["merge_relative_path"])
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"merged legacy artifact is absent: {relative}")
            digest, size = _sha256_file(path)
            expected = {
                "scope_id": scope.scope_id,
                "relative_path": relative,
                "sha256": digest,
                "size_bytes": size,
            }
            if digest != registration["sha256"] or size != int(registration["size_bytes"]):
                raise ValueError(f"merged legacy artifact changed: {relative}")
            expected_copied.append(expected)
            expected_files.add(relative)
    if copied != expected_copied:
        raise ValueError("legacy fragment merge file order or identity changed")
    accumulator_path = root / _MERGE_ACCUMULATORS_NAME
    accumulator_sha, accumulator_size = _sha256_file(accumulator_path)
    accumulator = _read_json(accumulator_path, label="legacy fragment merge accumulator")
    if not isinstance(accumulator, Mapping):
        raise ValueError("legacy fragment merge accumulator must be one object")
    accumulator_body = {
        key: copy.deepcopy(value) for key, value in accumulator.items() if key != "content_sha256"
    }
    if (
        set(accumulator)
        != {
            "schema_version",
            "plan_content_sha256",
            "stage1_request_sha256",
            "canonical_scope_order",
            "scope_count",
            "scopes",
            "content_sha256",
        }
        or accumulator.get("schema_version") != LEGACY_STAGE1_FRAGMENT_MERGE_ACCUMULATORS_SCHEMA
        or accumulator.get("plan_content_sha256") != plan.content_sha256
        or accumulator.get("stage1_request_sha256") != request_sha
        or accumulator.get("canonical_scope_order") != [scope.scope_id for scope in plan.scopes]
        or accumulator.get("scope_count") != len(fragments)
        or accumulator.get("content_sha256") != _sha256_json(accumulator_body)
        or manifest.get("scope_accumulators")
        != {
            "relative_path": _MERGE_ACCUMULATORS_NAME,
            "sha256": accumulator_sha,
            "size_bytes": accumulator_size,
            "content_sha256": accumulator.get("content_sha256"),
        }
    ):
        raise ValueError("legacy fragment merge accumulator changed")
    observed_files, observed_directories = _root_tree_inventory(root)
    if observed_files != expected_files or observed_directories != _parent_directories(
        tuple(expected_files)
    ):
        raise ValueError("legacy fragment merge contains unregistered entries")
    return copy.deepcopy(dict(manifest))


def validate_legacy_stage1_fragment_merge_from_path(
    *,
    plan: Stage1ScopePlan,
    stage1_request_sha256: str,
    destination_root: Path | str,
    require_production_coverage: bool = True,
) -> Mapping[str, Any]:
    """Authenticate a terminal merge using only its path and immutable request.

    The terminal manifest is not treated as an authority by itself.  Its
    fragment paths and attempt identities are merely capabilities used to
    reopen all canonical source fragments; the regular merge validator then
    derives the expected manifest and copied-file inventory from those
    authenticated fragments.
    """

    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError("plan must be a Stage1ScopePlan")
    request_sha = _require_sha256(stage1_request_sha256, label="stage1_request_sha256")
    root = _regular_directory(Path(destination_root), label="fragment merge root")
    manifest = _read_json(
        root / _MERGE_MANIFEST_NAME,
        label="legacy fragment merge terminal manifest",
    )
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    rows = manifest.get("fragments")
    if (
        manifest.get("schema_version") != LEGACY_STAGE1_FRAGMENT_MERGE_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("plan_content_sha256") != plan.content_sha256
        or manifest.get("stage1_request_sha256") != request_sha
        or manifest.get("content_sha256") != _sha256_json(body)
        or not isinstance(rows, list)
        or len(rows) != len(plan.scopes)
    ):
        raise ValueError("legacy fragment merge path does not contain a bound terminal manifest")

    expected_identity_fields = {
        "root",
        "manifest_path",
        "scope_id",
        "manifest_content_sha256",
        "artifact_count",
        "scope",
        "scope_attempt_request_sha256",
    }
    fragment_roots: dict[str, Path] = {}
    attempt_request_hashes: dict[str, str] = {}
    for scope, row in zip(plan.scopes, rows, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_identity_fields
            or row.get("scope_id") != scope.scope_id
            or row.get("scope") != scope.as_dict()
            or isinstance(row.get("artifact_count"), bool)
            or not isinstance(row.get("artifact_count"), int)
            or int(row["artifact_count"]) < 1
        ):
            raise ValueError("legacy fragment merge path has a malformed fragment capability")
        fragment_root = Path(str(row["root"]))
        if (
            not fragment_root.is_absolute()
            or fragment_root.resolve(strict=True) != fragment_root
            or str(row["manifest_path"]) != str(fragment_root / _FRAGMENT_MANIFEST_NAME)
        ):
            raise ValueError("legacy fragment merge path has a noncanonical fragment capability")
        _require_sha256(
            row.get("manifest_content_sha256"),
            label=f"{scope.scope_id} fragment manifest content SHA-256",
        )
        attempt_sha = _require_sha256(
            row.get("scope_attempt_request_sha256"),
            label=f"{scope.scope_id} attempt request SHA-256",
        )
        fragment_roots[scope.scope_id] = fragment_root
        attempt_request_hashes[scope.scope_id] = attempt_sha

    return validate_legacy_stage1_fragment_merge(
        plan=plan,
        stage1_request_sha256=request_sha,
        fragment_roots_by_scope=fragment_roots,
        scope_attempt_request_sha256_by_scope=attempt_request_hashes,
        destination_root=root,
        require_production_coverage=require_production_coverage,
    )


__all__ = [
    "LEGACY_STAGE1_FRAGMENT_MERGE_ACCUMULATORS_SCHEMA",
    "LEGACY_STAGE1_FRAGMENT_MERGE_SCHEMA",
    "LEGACY_STAGE1_SCOPE_ACCUMULATOR_SCHEMA",
    "LEGACY_STAGE1_SCOPE_FRAGMENT_SCHEMA",
    "LegacyStage1ScopeFragment",
    "durably_sync_legacy_stage1_tree",
    "merge_legacy_stage1_scope_fragments",
    "seal_legacy_stage1_scope_fragment",
    "validate_legacy_stage1_fragment_merge",
    "validate_legacy_stage1_fragment_merge_from_path",
    "validate_legacy_stage1_scope_fragment",
]
