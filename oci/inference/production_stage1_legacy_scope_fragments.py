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

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
)
from .lossless_stage1_evidence_catalog import (
    NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
)
from .production_stage1_scope_scheduler import Stage1ScopePlan, Stage1ScopeSpec

LEGACY_STAGE1_SCOPE_ACCUMULATOR_SCHEMA = "production_legacy_stage1_scope_accumulator_v1"
LEGACY_STAGE1_SCOPE_FRAGMENT_SCHEMA = "production_legacy_stage1_scope_fragment_v1"
LEGACY_STAGE1_FRAGMENT_MERGE_SCHEMA = "production_legacy_stage1_fragment_merge_v1"
LEGACY_STAGE1_FRAGMENT_MERGE_ACCUMULATORS_SCHEMA = (
    "production_legacy_stage1_fragment_merge_accumulators_v1"
)
LEGACY_STAGE1_ROLE_NEUTRAL_PHYSICAL_FIT_SCHEMA = (
    "production_legacy_stage1_role_neutral_physical_fit_v2"
)
LEGACY_STAGE1_LOGICAL_EVIDENCE_VIEW_SCHEMA = (
    "production_legacy_stage1_logical_evidence_view_v2"
)
LEGACY_STAGE1_ROLE_NEUTRAL_BINDING_SET_SCHEMA = (
    "production_legacy_stage1_role_neutral_binding_set_v2"
)
LEGACY_STAGE1_ROLE_NEUTRAL_PHYSICAL_PAYLOAD_SCHEMA = (
    "production_legacy_stage1_role_neutral_physical_payload_v2"
)
LEGACY_STAGE1_LOGICAL_VIEW_ARTIFACT_SCHEMA = (
    "production_legacy_stage1_logical_view_artifact_v2"
)
LEGACY_STAGE1_ROLE_NEUTRAL_PERSISTED_SET_SCHEMA = (
    "production_legacy_stage1_role_neutral_persisted_set_v2"
)
LEGACY_STAGE1_FIT_ONLY_FAMILY_SEAL_SCHEMA = (
    "production_legacy_stage1_fit_only_family_seal_v2"
)

_FRAGMENT_MANIFEST_NAME = "fragment_manifest.json"
_ACCUMULATOR_NAME = "scope_accumulator.json"
_ARTIFACT_DIRECTORY_NAME = "artifacts"
_MERGE_MANIFEST_NAME = "merge_manifest.json"
_MERGE_ACCUMULATORS_NAME = "scope_accumulators.json"
_ROLE_NEUTRAL_MANIFEST_NAME = "role_neutral_binding_set.json"
_ROLE_NEUTRAL_PHYSICAL_DIRECTORY = "physical_fit_payloads"
_ROLE_NEUTRAL_LOGICAL_DIRECTORY = "logical_views"
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


def build_role_neutral_fit_only_family_seal(
    *,
    plan: Stage1ScopePlan,
    physical_owner_scope_id: str,
    family: str,
    evidence_payload: Mapping[str, Any],
    producer_identity_sha256: str,
    configuration_identity_sha256: str,
    fit_state_artifact_sha256: str,
) -> dict[str, Any]:
    """Seal one family's fit-only result before any logical-view transform.

    This is the worker-facing contract. A native producer must call it only
    after its fit-side state and evidence payload are immutable and before it
    receives registered held-out text. The parent accepts only the resulting
    closed record; an opaque audit digest is insufficient.
    """

    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError("plan must be a Stage1ScopePlan")
    owner = plan.scope(str(physical_owner_scope_id))
    if plan.physical_owner(owner.scope_id).scope_id != owner.scope_id:
        raise ValueError("fit-only family seal must belong to a physical owner")
    family_name = str(family)
    if family_name not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError("fit-only family seal names an inactive family")
    closed = _closed_json(
        dict(evidence_payload),
        path=f"{owner.scope_id}.{family_name}.evidence_payload",
    )
    if (
        not isinstance(closed, dict)
        or set(closed)
        != {"schema_version", "family", "architecture_evidence"}
        or closed.get("schema_version")
        != NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION
        or closed.get("family") != family_name
        or not isinstance(closed.get("architecture_evidence"), list)
        or not closed["architecture_evidence"]
    ):
        raise ValueError("fit-only family seal requires one nonempty native payload")
    producer_id = _require_sha256(
        producer_identity_sha256,
        label=f"{family_name} producer identity SHA-256",
    )
    configuration_id = _require_sha256(
        configuration_identity_sha256,
        label=f"{family_name} configuration identity SHA-256",
    )
    fit_state_id = _require_sha256(
        fit_state_artifact_sha256,
        label=f"{family_name} fit-state artifact SHA-256",
    )
    payload_id = _sha256_json(closed)
    scope = owner.as_dict()
    events = [
        {
            "sequence": 1,
            "event": "fit_completed",
            "fit_state_artifact_sha256": fit_state_id,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
        },
        {
            "sequence": 2,
            "event": "fit_family_artifact_sealed",
            "fit_state_artifact_sha256": fit_state_id,
            "evidence_payload_sha256": payload_id,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
        },
    ]
    body = {
        "schema_version": LEGACY_STAGE1_FIT_ONLY_FAMILY_SEAL_SCHEMA,
        "plan_scientific_content_sha256": plan.scientific_content_sha256,
        "physical_owner_scope_id": owner.scope_id,
        "physical_owner_scope_sha256": scope["scope_sha256"],
        "family": family_name,
        "fit_row_ids": list(owner.fit_row_ids),
        "fit_row_order_fingerprint": scope["fit_row_order_fingerprint"],
        "canonical_group_seed": int(owner.scope_seed),
        "producer_identity_sha256": producer_id,
        "configuration_identity_sha256": configuration_id,
        "fit_state_artifact_sha256": fit_state_id,
        "evidence_payload_sha256": payload_id,
        "evidence_payload": closed,
        "event_order": events,
        "logical_view_transform_started": False,
        "registered_heldout_text_accessed": False,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _validate_role_neutral_fit_only_family_seal(
    value: Mapping[str, Any],
    *,
    plan: Stage1ScopePlan,
    owner: Stage1ScopeSpec,
    family: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("fit-only family seal must be a mapping")
    seal = copy.deepcopy(dict(value))
    expected = build_role_neutral_fit_only_family_seal(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
        family=family,
        evidence_payload=seal.get("evidence_payload") or {},
        producer_identity_sha256=seal.get("producer_identity_sha256"),
        configuration_identity_sha256=seal.get(
            "configuration_identity_sha256"
        ),
        fit_state_artifact_sha256=seal.get("fit_state_artifact_sha256"),
    )
    if seal != expected:
        raise ValueError(
            f"fit-only family seal has an invalid event binding: "
            f"{owner.scope_id}/{family}"
        )
    return seal


def build_role_neutral_physical_fit_artifact(
    *,
    plan: Stage1ScopePlan,
    physical_owner_scope_id: str,
    fit_artifact_sha256: str,
    family_fit_artifact_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Describe the smallest reusable scientific result of one physical fit.

    This record deliberately excludes a logical scope's held-out transformation
    and evidence-view bytes.  Those differ between exact-inner and
    cumulative-review purposes even when their fitted rows and seed are
    identical.  A producer may publish this record only when every one of the
    ten family artifacts is derived from fit rows alone.
    """

    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError("plan must be a Stage1ScopePlan")
    owner = plan.scope(str(physical_owner_scope_id))
    if plan.physical_owner(owner.scope_id).scope_id != owner.scope_id:
        raise ValueError("role-neutral fit artifact must belong to a physical owner")
    fit_sha = _require_sha256(
        fit_artifact_sha256,
        label="role-neutral fit artifact SHA-256",
    )
    if (
        not isinstance(family_fit_artifact_sha256, Mapping)
        or set(family_fit_artifact_sha256) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET
    ):
        missing = sorted(
            ACTIVE_STAGE1_CONCEPT_FAMILY_SET
            - set(family_fit_artifact_sha256 or {})
        )
        extra = sorted(
            set(family_fit_artifact_sha256 or {})
            - ACTIVE_STAGE1_CONCEPT_FAMILY_SET
        )
        raise ValueError(
            "role-neutral physical fit must register exactly all ten evidence "
            f"families; missing={missing}, extra={extra}"
        )
    family_ids = {
        family: _require_sha256(
            family_fit_artifact_sha256[family],
            label=f"{family} fit artifact SHA-256",
        )
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    scope = owner.as_dict()
    body = {
        "schema_version": LEGACY_STAGE1_ROLE_NEUTRAL_PHYSICAL_FIT_SCHEMA,
        "plan_scientific_content_sha256": plan.scientific_content_sha256,
        "physical_owner_scope_id": owner.scope_id,
        "physical_owner_scope_sha256": scope["scope_sha256"],
        "fit_row_order_fingerprint": scope["fit_row_order_fingerprint"],
        "fit_row_set_fingerprint": _sha256_json(sorted(owner.fit_row_ids)),
        "fit_row_count": owner.fit_row_count,
        "canonical_group_seed": int(owner.scope_seed),
        "fit_artifact_sha256": fit_sha,
        "family_fit_artifact_sha256": family_ids,
        "fit_input_policy": "fit_row_id_text_treatment_outcome_only_v1",
        "heldout_text_accessed_during_fit": False,
        "heldout_labels_accessed_during_fit": False,
        "oracle_fields_accessed_during_fit": False,
        "logical_view_bytes_included": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _validate_role_neutral_physical_fit_artifact(
    value: Mapping[str, Any],
    *,
    plan: Stage1ScopePlan,
    owner: Stage1ScopeSpec,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("role-neutral physical fit artifact must be a mapping")
    artifact = copy.deepcopy(dict(value))
    expected_fields = {
        "schema_version",
        "plan_scientific_content_sha256",
        "physical_owner_scope_id",
        "physical_owner_scope_sha256",
        "fit_row_order_fingerprint",
        "fit_row_set_fingerprint",
        "fit_row_count",
        "canonical_group_seed",
        "fit_artifact_sha256",
        "family_fit_artifact_sha256",
        "fit_input_policy",
        "heldout_text_accessed_during_fit",
        "heldout_labels_accessed_during_fit",
        "oracle_fields_accessed_during_fit",
        "logical_view_bytes_included",
        "content_sha256",
    }
    body = {
        key: copy.deepcopy(child)
        for key, child in artifact.items()
        if key != "content_sha256"
    }
    scope = owner.as_dict()
    families = artifact.get("family_fit_artifact_sha256")
    if (
        set(artifact) != expected_fields
        or artifact.get("schema_version")
        != LEGACY_STAGE1_ROLE_NEUTRAL_PHYSICAL_FIT_SCHEMA
        or artifact.get("plan_scientific_content_sha256")
        != plan.scientific_content_sha256
        or artifact.get("physical_owner_scope_id") != owner.scope_id
        or artifact.get("physical_owner_scope_sha256") != scope["scope_sha256"]
        or artifact.get("fit_row_order_fingerprint")
        != scope["fit_row_order_fingerprint"]
        or artifact.get("fit_row_set_fingerprint")
        != _sha256_json(sorted(owner.fit_row_ids))
        or artifact.get("fit_row_count") != owner.fit_row_count
        or artifact.get("canonical_group_seed") != owner.scope_seed
        or artifact.get("fit_input_policy")
        != "fit_row_id_text_treatment_outcome_only_v1"
        or artifact.get("heldout_text_accessed_during_fit") is not False
        or artifact.get("heldout_labels_accessed_during_fit") is not False
        or artifact.get("oracle_fields_accessed_during_fit") is not False
        or artifact.get("logical_view_bytes_included") is not False
        or not isinstance(families, Mapping)
        or set(families) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET
        or artifact.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("role-neutral physical fit artifact has an invalid binding")
    _require_sha256(
        artifact.get("fit_artifact_sha256"),
        label="role-neutral fit artifact SHA-256",
    )
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        _require_sha256(
            families[family],
            label=f"{family} fit artifact SHA-256",
        )
    return artifact


def build_role_neutral_logical_evidence_bindings(
    *,
    plan: Stage1ScopePlan,
    physical_fit_artifacts_by_owner: Mapping[str, Mapping[str, Any]],
    logical_view_artifact_sha256_by_scope: Mapping[str, str],
) -> dict[str, Any]:
    """Bind distinct logical evidence views to shared all-ten physical fits.

    Equality is asserted only for the fit-side family artifacts.  Each logical
    context must supply its own view artifact identity, and cross-purpose
    aliases are forbidden from claiming the same view bytes.
    """

    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError("plan must be a Stage1ScopePlan")
    owner_ids = tuple(scope.scope_id for scope in plan.physical_scopes)
    logical_ids = tuple(scope.scope_id for scope in plan.scopes)
    if set(physical_fit_artifacts_by_owner) != set(owner_ids):
        raise ValueError("role-neutral physical fit coverage is incomplete")
    if set(logical_view_artifact_sha256_by_scope) != set(logical_ids):
        raise ValueError("logical evidence-view artifact coverage is incomplete")

    physical_rows: list[dict[str, Any]] = []
    physical_by_owner: dict[str, dict[str, Any]] = {}
    for owner in plan.physical_scopes:
        physical = _validate_role_neutral_physical_fit_artifact(
            physical_fit_artifacts_by_owner[owner.scope_id],
            plan=plan,
            owner=owner,
        )
        physical_rows.append(physical)
        physical_by_owner[owner.scope_id] = physical

    view_ids = {
        scope_id: _require_sha256(
            logical_view_artifact_sha256_by_scope[scope_id],
            label=f"{scope_id} logical evidence-view SHA-256",
        )
        for scope_id in logical_ids
    }
    logical_rows: list[dict[str, Any]] = []
    for logical in plan.scopes:
        owner = plan.physical_owner(logical.scope_id)
        physical = physical_by_owner[owner.scope_id]
        if (
            tuple(logical.fit_row_ids)
            != tuple(owner.fit_row_ids)
            or logical.scope_seed != owner.scope_seed
        ):
            raise RuntimeError("logical context no longer matches its physical fit")
        if logical.scope_kind == "cumulative_spent":
            view_input_policy = "sealed_row_ids_only_no_sealed_text_or_labels_v1"
        elif logical.scope_kind in {"exact_inner", "full_outer"}:
            view_input_policy = "heldout_row_id_and_text_no_labels_v1"
        else:  # pragma: no cover - Stage1ScopeSpec is already closed.
            raise ValueError("unsupported logical Stage 1 purpose")
        if (
            logical.scope_id != owner.scope_id
            and logical.scope_kind != owner.scope_kind
            and view_ids[logical.scope_id] == view_ids[owner.scope_id]
        ):
            raise ValueError(
                "cross-purpose logical views cannot claim byte-identical artifacts"
            )
        logical_scope = logical.as_dict()
        body = {
            "schema_version": LEGACY_STAGE1_LOGICAL_EVIDENCE_VIEW_SCHEMA,
            "plan_scientific_content_sha256": plan.scientific_content_sha256,
            "logical_scope_id": logical.scope_id,
            "logical_scope_sha256": logical_scope["scope_sha256"],
            "logical_purpose": logical.scope_kind,
            "logical_heldout_row_order_fingerprint": logical_scope[
                "heldout_row_order_fingerprint"
            ],
            "view_input_policy": view_input_policy,
            "logical_view_artifact_sha256": view_ids[logical.scope_id],
            "physical_owner_scope_id": owner.scope_id,
            "physical_fit_content_sha256": physical["content_sha256"],
            "family_fit_artifact_sha256": copy.deepcopy(
                physical["family_fit_artifact_sha256"]
            ),
            "reuses_physical_fit": logical.scope_id != owner.scope_id,
            "logical_view_artifact_claimed_equal_to_owner": (
                view_ids[logical.scope_id] == view_ids[owner.scope_id]
            ),
            "heldout_labels_supplied_to_view": False,
        }
        logical_rows.append({**body, "content_sha256": _sha256_json(body)})

    top_body = {
        "schema_version": LEGACY_STAGE1_ROLE_NEUTRAL_BINDING_SET_SCHEMA,
        "plan_scientific_content_sha256": plan.scientific_content_sha256,
        "canonical_logical_scope_order": list(logical_ids),
        "physical_owner_scope_order": list(owner_ids),
        "logical_scope_count": len(logical_rows),
        "physical_fit_count": len(physical_rows),
        "deduplicated_fit_count": len(logical_rows) - len(physical_rows),
        "physical_fits": physical_rows,
        "logical_views": logical_rows,
        "all_ten_family_fit_artifact_ids_equal_within_group": True,
        "cross_purpose_logical_view_equality_claimed": False,
        "heldout_labels_supplied": False,
    }
    return {**top_body, "content_sha256": _sha256_json(top_body)}


def validate_role_neutral_logical_evidence_bindings(
    value: Mapping[str, Any],
    *,
    plan: Stage1ScopePlan,
) -> dict[str, Any]:
    """Freshly validate the closed physical-fit/logical-view binding set."""

    if not isinstance(value, Mapping):
        raise TypeError("role-neutral binding set must be a mapping")
    manifest = copy.deepcopy(dict(value))
    body = {
        key: copy.deepcopy(child)
        for key, child in manifest.items()
        if key != "content_sha256"
    }
    expected_fields = {
        "schema_version",
        "plan_scientific_content_sha256",
        "canonical_logical_scope_order",
        "physical_owner_scope_order",
        "logical_scope_count",
        "physical_fit_count",
        "deduplicated_fit_count",
        "physical_fits",
        "logical_views",
        "all_ten_family_fit_artifact_ids_equal_within_group",
        "cross_purpose_logical_view_equality_claimed",
        "heldout_labels_supplied",
        "content_sha256",
    }
    physical_rows = manifest.get("physical_fits")
    logical_rows = manifest.get("logical_views")
    if (
        set(manifest) != expected_fields
        or manifest.get("schema_version")
        != LEGACY_STAGE1_ROLE_NEUTRAL_BINDING_SET_SCHEMA
        or manifest.get("plan_scientific_content_sha256")
        != plan.scientific_content_sha256
        or manifest.get("canonical_logical_scope_order")
        != [scope.scope_id for scope in plan.scopes]
        or manifest.get("physical_owner_scope_order")
        != [scope.scope_id for scope in plan.physical_scopes]
        or manifest.get("logical_scope_count") != len(plan.scopes)
        or manifest.get("physical_fit_count") != len(plan.physical_scopes)
        or manifest.get("deduplicated_fit_count")
        != len(plan.scopes) - len(plan.physical_scopes)
        or manifest.get("all_ten_family_fit_artifact_ids_equal_within_group")
        is not True
        or manifest.get("cross_purpose_logical_view_equality_claimed") is not False
        or manifest.get("heldout_labels_supplied") is not False
        or not isinstance(physical_rows, list)
        or not isinstance(logical_rows, list)
        or manifest.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("role-neutral binding set has an invalid envelope")

    physical_by_owner: dict[str, dict[str, Any]] = {}
    for owner, physical in zip(plan.physical_scopes, physical_rows, strict=True):
        validated = _validate_role_neutral_physical_fit_artifact(
            physical,
            plan=plan,
            owner=owner,
        )
        physical_by_owner[owner.scope_id] = validated
    if len(physical_rows) != len(plan.physical_scopes):
        raise ValueError("role-neutral physical fit coverage changed")

    expected_view_fields = {
        "schema_version",
        "plan_scientific_content_sha256",
        "logical_scope_id",
        "logical_scope_sha256",
        "logical_purpose",
        "logical_heldout_row_order_fingerprint",
        "view_input_policy",
        "logical_view_artifact_sha256",
        "physical_owner_scope_id",
        "physical_fit_content_sha256",
        "family_fit_artifact_sha256",
        "reuses_physical_fit",
        "logical_view_artifact_claimed_equal_to_owner",
        "heldout_labels_supplied_to_view",
        "content_sha256",
    }
    view_by_scope: dict[str, Mapping[str, Any]] = {}
    for logical, row in zip(plan.scopes, logical_rows, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("logical evidence view is malformed")
        row_body = {
            key: copy.deepcopy(child)
            for key, child in row.items()
            if key != "content_sha256"
        }
        owner = plan.physical_owner(logical.scope_id)
        physical = physical_by_owner[owner.scope_id]
        expected_policy = (
            "sealed_row_ids_only_no_sealed_text_or_labels_v1"
            if logical.scope_kind == "cumulative_spent"
            else "heldout_row_id_and_text_no_labels_v1"
        )
        scope = logical.as_dict()
        if (
            set(row) != expected_view_fields
            or row.get("schema_version")
            != LEGACY_STAGE1_LOGICAL_EVIDENCE_VIEW_SCHEMA
            or row.get("plan_scientific_content_sha256")
            != plan.scientific_content_sha256
            or row.get("logical_scope_id") != logical.scope_id
            or row.get("logical_scope_sha256") != scope["scope_sha256"]
            or row.get("logical_purpose") != logical.scope_kind
            or row.get("logical_heldout_row_order_fingerprint")
            != scope["heldout_row_order_fingerprint"]
            or row.get("view_input_policy") != expected_policy
            or row.get("physical_owner_scope_id") != owner.scope_id
            or row.get("physical_fit_content_sha256")
            != physical["content_sha256"]
            or row.get("family_fit_artifact_sha256")
            != physical["family_fit_artifact_sha256"]
            or row.get("reuses_physical_fit")
            is not (logical.scope_id != owner.scope_id)
            or row.get("heldout_labels_supplied_to_view") is not False
            or row.get("content_sha256") != _sha256_json(row_body)
        ):
            raise ValueError(
                f"logical evidence view has an invalid binding: {logical.scope_id}"
            )
        _require_sha256(
            row.get("logical_view_artifact_sha256"),
            label=f"{logical.scope_id} logical evidence-view SHA-256",
        )
        view_by_scope[logical.scope_id] = row
    if len(logical_rows) != len(plan.scopes):
        raise ValueError("logical evidence-view coverage changed")
    for owner, members in plan.physical_scope_groups:
        owner_view = view_by_scope[owner.scope_id]
        for logical in members:
            view = view_by_scope[logical.scope_id]
            if (
                view["family_fit_artifact_sha256"]
                != owner_view["family_fit_artifact_sha256"]
            ):
                raise ValueError("logical alias changed a family fit artifact")
            claimed_equal = (
                view["logical_view_artifact_sha256"]
                == owner_view["logical_view_artifact_sha256"]
            )
            if view.get("logical_view_artifact_claimed_equal_to_owner") is not claimed_equal:
                raise ValueError("logical view equality claim differs from its bytes")
            if (
                logical.scope_kind != owner.scope_kind
                and logical.scope_id != owner.scope_id
                and claimed_equal
            ):
                raise ValueError(
                    "cross-purpose logical views cannot share one artifact identity"
                )
    return manifest


def persist_role_neutral_logical_evidence_bindings(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
    family_fit_seal_by_physical_owner: Mapping[
        str,
        Mapping[str, Mapping[str, Any]],
    ],
    logical_source_artifact_sha256_by_scope: Mapping[str, str],
) -> dict[str, Any]:
    """Persist fit-side payloads once and publish authenticated logical views.

    The caller supplies exactly one closed fit-only family seal for every
    family of every physical owner. No alias payload is accepted: equivalent
    logical scopes prove equality by referencing the same immutable physical
    family artifacts. Logical views contain only purpose-specific metadata and
    references.
    """

    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError("plan must be a Stage1ScopePlan")
    destination = Path(root)
    if not destination.is_absolute():
        raise ValueError("role-neutral binding root must be absolute")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("role-neutral binding root must be fresh")
    parent = _regular_directory(
        destination.parent,
        label="role-neutral binding parent",
    )
    if parent != destination.parent:
        raise ValueError("role-neutral binding parent must be canonical")
    owner_ids = tuple(scope.scope_id for scope in plan.physical_scopes)
    logical_ids = tuple(scope.scope_id for scope in plan.scopes)
    expected_owners = set(owner_ids)
    expected_logical = set(logical_ids)
    if (
        not isinstance(family_fit_seal_by_physical_owner, Mapping)
        or set(family_fit_seal_by_physical_owner) != expected_owners
    ):
        raise ValueError(
            "physical-owner fit-only family-seal coverage differs from the "
            "physical plan"
        )
    if (
        not isinstance(logical_source_artifact_sha256_by_scope, Mapping)
        or set(logical_source_artifact_sha256_by_scope) != expected_logical
    ):
        raise ValueError(
            "logical source artifact coverage differs from the logical plan"
        )

    normalized_seals: dict[str, dict[str, dict[str, Any]]] = {}
    source_ids: dict[str, str] = {}
    for owner in plan.physical_scopes:
        owner_id = owner.scope_id
        raw_seals = family_fit_seal_by_physical_owner[owner_id]
        if (
            not isinstance(raw_seals, Mapping)
            or set(raw_seals) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET
        ):
            raise ValueError(
                f"{owner_id} does not contain exactly ten fit-only family seals"
            )
        normalized_seals[owner_id] = {
            family: _validate_role_neutral_fit_only_family_seal(
                raw_seals[family],
                plan=plan,
                owner=owner,
                family=family,
            )
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        }
    for scope_id in logical_ids:
        source_ids[scope_id] = _require_sha256(
            logical_source_artifact_sha256_by_scope[scope_id],
            label=f"{scope_id} logical source artifact SHA-256",
        )

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.role-neutral-",
            dir=parent,
        )
    )
    try:
        physical_dir = temporary / _ROLE_NEUTRAL_PHYSICAL_DIRECTORY
        logical_dir = temporary / _ROLE_NEUTRAL_LOGICAL_DIRECTORY
        physical_dir.mkdir(parents=False, exist_ok=False)
        logical_dir.mkdir(parents=False, exist_ok=False)
        physical_artifacts: dict[str, Mapping[str, Any]] = {}
        physical_registrations: list[dict[str, Any]] = []
        logical_view_hashes: dict[str, str] = {}
        logical_registrations: list[dict[str, Any]] = []

        for owner, _members in plan.physical_scope_groups:
            owner_seals = normalized_seals[owner.scope_id]
            family_payload_ids = {
                family: owner_seals[family]["evidence_payload_sha256"]
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            }
            family_fit_ids = {
                family: owner_seals[family]["content_sha256"]
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            }
            owner_scope = owner.as_dict()
            physical_body = {
                "schema_version": (
                    LEGACY_STAGE1_ROLE_NEUTRAL_PHYSICAL_PAYLOAD_SCHEMA
                ),
                "plan_scientific_content_sha256": (
                    plan.scientific_content_sha256
                ),
                "physical_owner_scope_id": owner.scope_id,
                "physical_owner_scope_sha256": owner_scope["scope_sha256"],
                "fit_row_ids": list(owner.fit_row_ids),
                "fit_row_order_fingerprint": owner_scope[
                    "fit_row_order_fingerprint"
                ],
                "fit_row_set_fingerprint": _sha256_json(
                    sorted(owner.fit_row_ids)
                ),
                "canonical_group_seed": int(owner.scope_seed),
                "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
                "family_evidence_payload_sha256": family_payload_ids,
                "family_fit_artifact_sha256": family_fit_ids,
                "family_fit_seals": owner_seals,
                "heldout_text_included": False,
                "heldout_labels_included": False,
                "logical_view_metadata_included": False,
            }
            physical_payload = {
                **physical_body,
                "content_sha256": _sha256_json(physical_body),
            }
            physical_path = physical_dir / f"{owner.scope_id}.json"
            _write_new_json(physical_path, physical_payload)
            physical_sha, physical_size = _sha256_file(physical_path)
            physical_artifacts[owner.scope_id] = (
                build_role_neutral_physical_fit_artifact(
                    plan=plan,
                    physical_owner_scope_id=owner.scope_id,
                    fit_artifact_sha256=physical_sha,
                    family_fit_artifact_sha256=family_fit_ids,
                )
            )
            physical_registrations.append(
                {
                    "physical_owner_scope_id": owner.scope_id,
                    "relative_path": physical_path.relative_to(
                        temporary
                    ).as_posix(),
                    "sha256": physical_sha,
                    "size_bytes": physical_size,
                    "content_sha256": physical_payload["content_sha256"],
                }
            )

        physical_registration_by_owner = {
            row["physical_owner_scope_id"]: row
            for row in physical_registrations
        }
        for logical in plan.scopes:
            owner = plan.physical_owner(logical.scope_id)
            source_id = source_ids[logical.scope_id]
            physical_payload = physical_registration_by_owner[owner.scope_id]
            family_fit_ids = physical_artifacts[owner.scope_id][
                "family_fit_artifact_sha256"
            ]
            policy = (
                "sealed_row_ids_only_no_sealed_text_or_labels_v1"
                if logical.scope_kind == "cumulative_spent"
                else "heldout_row_id_and_text_no_labels_v1"
            )
            scope = logical.as_dict()
            view_body = {
                "schema_version": LEGACY_STAGE1_LOGICAL_VIEW_ARTIFACT_SCHEMA,
                "plan_scientific_content_sha256": (
                    plan.scientific_content_sha256
                ),
                "logical_scope_id": logical.scope_id,
                "logical_scope_sha256": scope["scope_sha256"],
                "logical_purpose": logical.scope_kind,
                "logical_heldout_row_ids": list(logical.heldout_row_ids),
                "logical_heldout_row_order_fingerprint": scope[
                    "heldout_row_order_fingerprint"
                ],
                "view_input_policy": policy,
                "logical_source_artifact_sha256": source_id,
                "physical_owner_scope_id": owner.scope_id,
                "physical_payload": copy.deepcopy(physical_payload),
                "family_fit_artifact_sha256": copy.deepcopy(family_fit_ids),
                "event": "logical_view_reference_published",
                "published_after_all_family_fit_seals": True,
                "logical_view_transform_performed": False,
                "registered_heldout_text_accessed": False,
                "reuses_physical_fit": logical.scope_id != owner.scope_id,
                "heldout_labels_supplied": False,
            }
            view = {**view_body, "content_sha256": _sha256_json(view_body)}
            view_path = logical_dir / f"{logical.scope_id}.json"
            _write_new_json(view_path, view)
            view_sha, view_size = _sha256_file(view_path)
            logical_view_hashes[logical.scope_id] = view_sha
            logical_registrations.append(
                {
                    "logical_scope_id": logical.scope_id,
                    "relative_path": view_path.relative_to(temporary).as_posix(),
                    "sha256": view_sha,
                    "size_bytes": view_size,
                    "content_sha256": view["content_sha256"],
                }
            )

        scientific_bindings = build_role_neutral_logical_evidence_bindings(
            plan=plan,
            physical_fit_artifacts_by_owner=physical_artifacts,
            logical_view_artifact_sha256_by_scope=logical_view_hashes,
        )
        terminal_body = {
            "schema_version": LEGACY_STAGE1_ROLE_NEUTRAL_PERSISTED_SET_SCHEMA,
            "status": "complete",
            "plan_scientific_content_sha256": (
                plan.scientific_content_sha256
            ),
            "logical_scope_count": len(plan.scopes),
            "physical_fit_count": len(plan.physical_scopes),
            "deduplicated_fit_count": (
                len(plan.scopes) - len(plan.physical_scopes)
            ),
            "physical_payloads": physical_registrations,
            "logical_views": logical_registrations,
            "scientific_bindings": scientific_bindings,
            "payload_bytes_persisted_once_per_physical_fit": True,
            "logical_views_are_reference_only": True,
        }
        terminal = {
            **terminal_body,
            "content_sha256": _sha256_json(terminal_body),
        }
        durably_sync_legacy_stage1_tree(temporary)
        _write_new_json(temporary / _ROLE_NEUTRAL_MANIFEST_NAME, terminal)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_persisted_role_neutral_logical_evidence_bindings(
        root=destination,
        plan=plan,
    )


def validate_persisted_role_neutral_logical_evidence_bindings(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
) -> dict[str, Any]:
    """Reopen every persisted payload and logical view from a fresh path."""

    tree = _regular_directory(
        Path(root),
        label="persisted role-neutral binding root",
    )
    terminal = _read_json(
        tree / _ROLE_NEUTRAL_MANIFEST_NAME,
        label="persisted role-neutral binding manifest",
    )
    body = {
        key: copy.deepcopy(value)
        for key, value in terminal.items()
        if key != "content_sha256"
    }
    expected_fields = {
        "schema_version",
        "status",
        "plan_scientific_content_sha256",
        "logical_scope_count",
        "physical_fit_count",
        "deduplicated_fit_count",
        "physical_payloads",
        "logical_views",
        "scientific_bindings",
        "payload_bytes_persisted_once_per_physical_fit",
        "logical_views_are_reference_only",
        "content_sha256",
    }
    physical_rows = terminal.get("physical_payloads")
    logical_rows = terminal.get("logical_views")
    if (
        set(terminal) != expected_fields
        or terminal.get("schema_version")
        != LEGACY_STAGE1_ROLE_NEUTRAL_PERSISTED_SET_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("plan_scientific_content_sha256")
        != plan.scientific_content_sha256
        or terminal.get("logical_scope_count") != len(plan.scopes)
        or terminal.get("physical_fit_count") != len(plan.physical_scopes)
        or terminal.get("deduplicated_fit_count")
        != len(plan.scopes) - len(plan.physical_scopes)
        or terminal.get("payload_bytes_persisted_once_per_physical_fit")
        is not True
        or terminal.get("logical_views_are_reference_only") is not True
        or not isinstance(physical_rows, list)
        or not isinstance(logical_rows, list)
        or terminal.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("persisted role-neutral binding manifest is invalid")

    physical_artifacts: dict[str, Mapping[str, Any]] = {}
    expected_files = {_ROLE_NEUTRAL_MANIFEST_NAME}
    expected_physical_fields = {
        "physical_owner_scope_id",
        "relative_path",
        "sha256",
        "size_bytes",
        "content_sha256",
    }
    for owner, registration in zip(
        plan.physical_scopes,
        physical_rows,
        strict=True,
    ):
        if (
            not isinstance(registration, Mapping)
            or set(registration) != expected_physical_fields
            or registration.get("physical_owner_scope_id") != owner.scope_id
        ):
            raise ValueError("persisted physical payload registration is invalid")
        relative = _canonical_relative_path(
            str(registration["relative_path"]),
            label="persisted physical payload path",
        )
        if relative != f"{_ROLE_NEUTRAL_PHYSICAL_DIRECTORY}/{owner.scope_id}.json":
            raise ValueError("persisted physical payload path is noncanonical")
        path = tree / relative
        digest, size = _sha256_file(path)
        payload = _read_json(path, label=f"{owner.scope_id} physical payload")
        payload_body = {
            key: copy.deepcopy(value)
            for key, value in payload.items()
            if key != "content_sha256"
        }
        expected_payload_fields = {
            "schema_version",
            "plan_scientific_content_sha256",
            "physical_owner_scope_id",
            "physical_owner_scope_sha256",
            "fit_row_ids",
            "fit_row_order_fingerprint",
            "fit_row_set_fingerprint",
            "canonical_group_seed",
            "architecture_order",
            "family_evidence_payload_sha256",
            "family_fit_artifact_sha256",
            "family_fit_seals",
            "heldout_text_included",
            "heldout_labels_included",
            "logical_view_metadata_included",
            "content_sha256",
        }
        seals = payload.get("family_fit_seals")
        family_payload_ids = payload.get("family_evidence_payload_sha256")
        family_fit_ids = payload.get("family_fit_artifact_sha256")
        scope = owner.as_dict()
        if (
            set(payload) != expected_payload_fields
            or payload.get("schema_version")
            != LEGACY_STAGE1_ROLE_NEUTRAL_PHYSICAL_PAYLOAD_SCHEMA
            or payload.get("plan_scientific_content_sha256")
            != plan.scientific_content_sha256
            or payload.get("physical_owner_scope_id") != owner.scope_id
            or payload.get("physical_owner_scope_sha256")
            != scope["scope_sha256"]
            or payload.get("fit_row_ids") != list(owner.fit_row_ids)
            or payload.get("fit_row_order_fingerprint")
            != scope["fit_row_order_fingerprint"]
            or payload.get("fit_row_set_fingerprint")
            != _sha256_json(sorted(owner.fit_row_ids))
            or payload.get("canonical_group_seed") != owner.scope_seed
            or payload.get("architecture_order")
            != list(ACTIVE_STAGE1_CONCEPT_FAMILIES)
            or not isinstance(seals, Mapping)
            or set(seals) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET
            or not isinstance(family_payload_ids, Mapping)
            or set(family_payload_ids) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET
            or not isinstance(family_fit_ids, Mapping)
            or set(family_fit_ids) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET
            or any(
                family_payload_ids[family]
                != _validate_role_neutral_fit_only_family_seal(
                    seals[family],
                    plan=plan,
                    owner=owner,
                    family=family,
                )["evidence_payload_sha256"]
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            )
            or any(
                family_fit_ids[family]
                != seals[family].get("content_sha256")
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            )
            or payload.get("heldout_text_included") is not False
            or payload.get("heldout_labels_included") is not False
            or payload.get("logical_view_metadata_included") is not False
            or payload.get("content_sha256") != _sha256_json(payload_body)
            or dict(registration)
            != {
                "physical_owner_scope_id": owner.scope_id,
                "relative_path": relative,
                "sha256": digest,
                "size_bytes": size,
                "content_sha256": payload.get("content_sha256"),
            }
        ):
            raise ValueError(f"persisted physical payload changed: {owner.scope_id}")
        physical_artifacts[owner.scope_id] = (
            build_role_neutral_physical_fit_artifact(
                plan=plan,
                physical_owner_scope_id=owner.scope_id,
                fit_artifact_sha256=digest,
                family_fit_artifact_sha256=family_fit_ids,
            )
        )
        expected_files.add(relative)
    if len(physical_rows) != len(plan.physical_scopes):
        raise ValueError("persisted physical payload coverage changed")

    logical_view_hashes: dict[str, str] = {}
    expected_logical_fields = {
        "logical_scope_id",
        "relative_path",
        "sha256",
        "size_bytes",
        "content_sha256",
    }
    physical_registration_by_owner = {
        str(row["physical_owner_scope_id"]): row for row in physical_rows
    }
    for logical, registration in zip(plan.scopes, logical_rows, strict=True):
        if (
            not isinstance(registration, Mapping)
            or set(registration) != expected_logical_fields
            or registration.get("logical_scope_id") != logical.scope_id
        ):
            raise ValueError("persisted logical view registration is invalid")
        relative = _canonical_relative_path(
            str(registration["relative_path"]),
            label="persisted logical view path",
        )
        if relative != f"{_ROLE_NEUTRAL_LOGICAL_DIRECTORY}/{logical.scope_id}.json":
            raise ValueError("persisted logical view path is noncanonical")
        path = tree / relative
        digest, size = _sha256_file(path)
        view = _read_json(path, label=f"{logical.scope_id} logical view")
        view_body = {
            key: copy.deepcopy(value)
            for key, value in view.items()
            if key != "content_sha256"
        }
        expected_view_fields = {
            "schema_version",
            "plan_scientific_content_sha256",
            "logical_scope_id",
            "logical_scope_sha256",
            "logical_purpose",
            "logical_heldout_row_ids",
            "logical_heldout_row_order_fingerprint",
            "view_input_policy",
            "logical_source_artifact_sha256",
            "physical_owner_scope_id",
            "physical_payload",
            "family_fit_artifact_sha256",
            "event",
            "published_after_all_family_fit_seals",
            "logical_view_transform_performed",
            "registered_heldout_text_accessed",
            "reuses_physical_fit",
            "heldout_labels_supplied",
            "content_sha256",
        }
        owner = plan.physical_owner(logical.scope_id)
        physical_path = tree / str(
            physical_registration_by_owner[owner.scope_id]["relative_path"]
        )
        physical_payload = _read_json(
            physical_path,
            label=f"{owner.scope_id} physical payload for logical view",
        )
        expected_policy = (
            "sealed_row_ids_only_no_sealed_text_or_labels_v1"
            if logical.scope_kind == "cumulative_spent"
            else "heldout_row_id_and_text_no_labels_v1"
        )
        scope = logical.as_dict()
        if (
            set(view) != expected_view_fields
            or view.get("schema_version")
            != LEGACY_STAGE1_LOGICAL_VIEW_ARTIFACT_SCHEMA
            or view.get("plan_scientific_content_sha256")
            != plan.scientific_content_sha256
            or view.get("logical_scope_id") != logical.scope_id
            or view.get("logical_scope_sha256") != scope["scope_sha256"]
            or view.get("logical_purpose") != logical.scope_kind
            or view.get("logical_heldout_row_ids")
            != list(logical.heldout_row_ids)
            or view.get("logical_heldout_row_order_fingerprint")
            != scope["heldout_row_order_fingerprint"]
            or view.get("view_input_policy") != expected_policy
            or _require_sha256(
                view.get("logical_source_artifact_sha256"),
                label=f"{logical.scope_id} logical source SHA-256",
            )
            != view.get("logical_source_artifact_sha256")
            or view.get("physical_owner_scope_id") != owner.scope_id
            or view.get("physical_payload")
            != physical_registration_by_owner[owner.scope_id]
            or view.get("family_fit_artifact_sha256")
            != physical_payload["family_fit_artifact_sha256"]
            or view.get("event") != "logical_view_reference_published"
            or view.get("published_after_all_family_fit_seals") is not True
            or view.get("logical_view_transform_performed") is not False
            or view.get("registered_heldout_text_accessed") is not False
            or view.get("reuses_physical_fit")
            is not (logical.scope_id != owner.scope_id)
            or view.get("heldout_labels_supplied") is not False
            or view.get("content_sha256") != _sha256_json(view_body)
            or dict(registration)
            != {
                "logical_scope_id": logical.scope_id,
                "relative_path": relative,
                "sha256": digest,
                "size_bytes": size,
                "content_sha256": view.get("content_sha256"),
            }
        ):
            raise ValueError(f"persisted logical view changed: {logical.scope_id}")
        logical_view_hashes[logical.scope_id] = digest
        expected_files.add(relative)
    if len(logical_rows) != len(plan.scopes):
        raise ValueError("persisted logical view coverage changed")

    expected_scientific = build_role_neutral_logical_evidence_bindings(
        plan=plan,
        physical_fit_artifacts_by_owner=physical_artifacts,
        logical_view_artifact_sha256_by_scope=logical_view_hashes,
    )
    if terminal.get("scientific_bindings") != expected_scientific:
        raise ValueError("persisted scientific role-neutral bindings changed")
    observed_files, observed_directories = _root_tree_inventory(tree)
    if observed_files != expected_files or observed_directories != _parent_directories(
        tuple(expected_files)
    ):
        raise ValueError("persisted role-neutral binding tree has unregistered entries")
    return copy.deepcopy(dict(terminal))


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
        outer_folds = {scope.outer_fold for scope in plan.scopes}
        outer_count = len(outer_folds)
        exact_counts = {
            outer_fold: sum(
                scope.scope_kind == "exact_inner"
                and scope.outer_fold == outer_fold
                for scope in plan.scopes
            )
            for outer_fold in outer_folds
        }
        if (
            outer_count < 2
            or kind_counts["full_outer"] != outer_count
            or not exact_counts
            or len(set(exact_counts.values())) != 1
            or next(iter(exact_counts.values())) < 2
            or kind_counts["exact_inner"] != sum(exact_counts.values())
            or kind_counts["cumulative_spent"]
            != outer_count * int(plan.review_rounds)
        ):
            raise ValueError(
                "production legacy fragment merge coverage does not match "
                "the authenticated fold/review plan"
            )
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
    "LEGACY_STAGE1_FIT_ONLY_FAMILY_SEAL_SCHEMA",
    "LEGACY_STAGE1_LOGICAL_EVIDENCE_VIEW_SCHEMA",
    "LEGACY_STAGE1_LOGICAL_VIEW_ARTIFACT_SCHEMA",
    "LEGACY_STAGE1_ROLE_NEUTRAL_BINDING_SET_SCHEMA",
    "LEGACY_STAGE1_ROLE_NEUTRAL_PERSISTED_SET_SCHEMA",
    "LEGACY_STAGE1_ROLE_NEUTRAL_PHYSICAL_PAYLOAD_SCHEMA",
    "LEGACY_STAGE1_ROLE_NEUTRAL_PHYSICAL_FIT_SCHEMA",
    "LEGACY_STAGE1_SCOPE_ACCUMULATOR_SCHEMA",
    "LEGACY_STAGE1_SCOPE_FRAGMENT_SCHEMA",
    "LegacyStage1ScopeFragment",
    "build_role_neutral_fit_only_family_seal",
    "build_role_neutral_logical_evidence_bindings",
    "build_role_neutral_physical_fit_artifact",
    "durably_sync_legacy_stage1_tree",
    "merge_legacy_stage1_scope_fragments",
    "persist_role_neutral_logical_evidence_bindings",
    "seal_legacy_stage1_scope_fragment",
    "validate_legacy_stage1_fragment_merge",
    "validate_legacy_stage1_fragment_merge_from_path",
    "validate_persisted_role_neutral_logical_evidence_bindings",
    "validate_role_neutral_logical_evidence_bindings",
    "validate_legacy_stage1_scope_fragment",
]
