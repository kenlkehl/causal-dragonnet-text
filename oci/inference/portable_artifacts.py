"""Path-neutral artifact DAG manifests and fail-closed checkpoint adoption."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from .portable_identity import canonical_json, identity_sha256


PORTABLE_ARTIFACT_MANIFEST = "portable_scientific_artifact_manifest_v1"
PORTABLE_ARTIFACT_LOCATOR = "portable_scientific_artifact_locator_v2"
PORTABLE_ADOPTION_ATTESTATION = "portable_checkpoint_adoption_attestation_v3"
PORTABLE_PHASE_BINDING = "portable_workflow_phase_binding_v1"
PORTABLE_SCIENTIFIC_CONTENT_DESCRIPTOR = (
    "portable_scientific_content_descriptor_v2"
)
SCIENTIFIC_COMPATIBILITY_VERSION = "portable_all_evidence_compatibility_v1"
MANIFEST_NAME = "artifact_manifest.json"
LOCATOR_NAME = "artifact_locator.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELATIVE = re.compile(r"^(?!/)(?!.*(?:^|/)\.\.(?:/|$)).+$")
_PAYLOAD_PATH_TOKEN = "$portable_payload_path"
_OMITTED_EXTERNAL_LOCATOR_TOKEN = "$omitted_external_locator"
COMPLETE_PAYLOAD_TREE = "complete_payload_tree_v1"
REGISTERED_PAYLOAD_PATHS_ONLY = "registered_payload_paths_only_v1"
_PAYLOAD_INVENTORY_POLICIES = frozenset(
    {COMPLETE_PAYLOAD_TREE, REGISTERED_PAYLOAD_PATHS_ONLY}
)

CHECKPOINT_ARTIFACT_KINDS = frozenset(
    {
        "prepared_cohort",
        "embedding_cache",
        "clustered_preflight",
        "prepared_stage1_context",
        "physical_scope_fit",
        "logical_scope_bindings",
        "tfidf_component",
        "neural_query_component",
        "stage1_handoff",
        "stage2_response_component",
        "stage2_extraction_component",
        "stage2_review_component",
        "frozen_prediction",
        "oracle_evaluation",
        "stage2_canary",
        "stage2_fold",
        "row_map",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _safe_path_boundaries(
    *,
    root: Path,
    relative_path: str,
    label: str,
) -> dict[str, tuple[int, ...]]:
    """Authenticate every lexical path component without following symlinks."""

    root = Path(root)
    relative = _normalize_relative(relative_path)
    try:
        root_state = os.lstat(root)
    except OSError as exc:
        raise FileNotFoundError(f"{label} root is missing: {root}") from exc
    if stat.S_ISLNK(root_state.st_mode) or not stat.S_ISDIR(root_state.st_mode):
        raise ValueError(f"{label} root must be a non-symlink directory")
    identities = {str(root): _stat_identity(root_state)}
    parts = Path(relative).parts
    cursor = root
    for ordinal, part in enumerate(parts):
        cursor = cursor / part
        try:
            state = os.lstat(cursor)
        except OSError as exc:
            raise FileNotFoundError(f"{label} is missing: {cursor}") from exc
        is_leaf = ordinal == len(parts) - 1
        if stat.S_ISLNK(state.st_mode):
            raise ValueError(f"{label} cannot traverse a symlink: {cursor}")
        if is_leaf:
            if not stat.S_ISREG(state.st_mode) or int(state.st_nlink) != 1:
                raise ValueError(
                    f"{label} must end at a private regular file with one hard link"
                )
        elif not stat.S_ISDIR(state.st_mode):
            raise ValueError(
                f"{label} ancestor must be a non-symlink directory: {cursor}"
            )
        identities[str(cursor)] = _stat_identity(state)
    return identities


def _safe_file_hash_with_identity(
    path: Path,
    *,
    label: str,
) -> tuple[str, int, tuple[int, ...]]:
    """Authenticate a non-symlink, single-link regular file through one fd."""

    try:
        before_path = os.lstat(path)
    except OSError as exc:
        raise FileNotFoundError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(before_path.st_mode)
        or not stat.S_ISREG(before_path.st_mode)
        or int(before_path.st_nlink) != 1
    ):
        raise ValueError(f"{label} must be a non-symlink regular file with one hard link")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    size = 0
    try:
        before_fd = os.fstat(descriptor)
        if _stat_identity(before_fd) != _stat_identity(before_path):
            raise RuntimeError(f"{label} changed while being opened")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(path)
    expected = _stat_identity(before_path)
    if (
        _stat_identity(before_fd) != expected
        or _stat_identity(after_fd) != expected
        or _stat_identity(after_path) != expected
        or size != int(before_path.st_size)
    ):
        raise RuntimeError(f"{label} changed while being authenticated")
    return digest.hexdigest(), size, expected


def _safe_file_hash(path: Path, *, label: str) -> tuple[str, int]:
    digest, size, _identity = _safe_file_hash_with_identity(path, label=label)
    return digest, size


def _safe_read_with_identity(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, tuple[int, ...]]:
    """Read one private regular file from a stable no-follow descriptor."""

    try:
        before_path = os.lstat(path)
    except OSError as exc:
        raise FileNotFoundError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(before_path.st_mode)
        or not stat.S_ISREG(before_path.st_mode)
        or int(before_path.st_nlink) != 1
    ):
        raise ValueError(f"{label} must be a non-symlink regular file with one hard link")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before_fd = os.fstat(descriptor)
        expected = _stat_identity(before_path)
        if _stat_identity(before_fd) != expected:
            raise RuntimeError(f"{label} changed while being opened")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(path)
    payload = b"".join(blocks)
    if (
        _stat_identity(after_fd) != expected
        or _stat_identity(after_path) != expected
        or len(payload) != int(before_path.st_size)
    ):
        raise RuntimeError(f"{label} changed while being read")
    return payload, expected


def _safe_read(path: Path, *, label: str) -> bytes:
    payload, _identity = _safe_read_with_identity(path, label=label)
    return payload


def _atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(
        dict(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _normalize_relative(value: str) -> str:
    normalized = Path(value).as_posix()
    if (
        not value
        or value != normalized
        or _RELATIVE.fullmatch(value) is None
        or value in {MANIFEST_NAME, LOCATOR_NAME}
    ):
        raise ValueError(f"unsafe or reserved artifact payload path: {value!r}")
    return value


@dataclass(frozen=True)
class PayloadRegistration:
    relative_path: str
    size_bytes: int
    sha256: str
    media_type: str

    def __post_init__(self) -> None:
        _normalize_relative(self.relative_path)
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("payload size must be a nonnegative integer")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("payload SHA-256 is invalid")
        if not isinstance(self.media_type, str) or not self.media_type:
            raise ValueError("payload media type is required")


@dataclass(frozen=True)
class ArtifactCompatibility:
    dataset_identity: str
    split_identity: str
    row_order_identity: str
    model_identities: Mapping[str, str]
    prompt_identities: Mapping[str, str]
    configuration_identity: str
    seed_identity: str
    producer_code_identity: str
    runtime_compatibility_class: str

    def __post_init__(self) -> None:
        for label, value in (
            ("dataset_identity", self.dataset_identity),
            ("split_identity", self.split_identity),
            ("row_order_identity", self.row_order_identity),
            ("configuration_identity", self.configuration_identity),
            ("seed_identity", self.seed_identity),
            ("producer_code_identity", self.producer_code_identity),
        ):
            if _SHA256.fullmatch(str(value)) is None:
                raise ValueError(f"{label} must be one lowercase SHA-256")
        for collection_name, collection in (
            ("model_identities", self.model_identities),
            ("prompt_identities", self.prompt_identities),
        ):
            for name, digest in collection.items():
                if not name or _SHA256.fullmatch(str(digest)) is None:
                    raise ValueError(f"{collection_name} contains an invalid identity")
        if not self.runtime_compatibility_class:
            raise ValueError("runtime compatibility class is required")

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "model_identities": dict(sorted(self.model_identities.items())),
            "prompt_identities": dict(sorted(self.prompt_identities.items())),
        }

    @property
    def key(self) -> str:
        return identity_sha256(self.as_dict())


@dataclass(frozen=True)
class ValidatedPortableArtifact:
    root: Path
    payload_root: Path
    manifest_path: Path
    locator_path: Path
    manifest: Mapping[str, Any]
    payloads: tuple[PayloadRegistration, ...]
    stat_inventory: Mapping[str, tuple[int, ...]]

    @property
    def artifact_id(self) -> str:
        return str(self.manifest["artifact_id"])

    @property
    def compatibility_key(self) -> str:
        return str(self.manifest["compatibility_key"])

    @property
    def phase_binding(self) -> Mapping[str, Any] | None:
        value = self.manifest.get("workflow_phase_binding")
        return None if value is None else dict(value)

    @property
    def artifact_metadata(self) -> Mapping[str, Any]:
        value = self.manifest.get("artifact_metadata")
        return {} if value is None else dict(value)

    @property
    def payload_inventory_policy(self) -> str:
        return str(
            self.manifest.get("payload_inventory_policy", COMPLETE_PAYLOAD_TREE)
        )


def _closed_artifact_metadata(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("portable artifact metadata must be one mapping")
    try:
        normalized = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TypeError("portable artifact metadata must be closed finite JSON") from exc
    if not isinstance(normalized, dict) or any(
        not isinstance(key, str) or not key for key in normalized
    ):
        raise ValueError("portable artifact metadata keys must be nonempty strings")

    def reject_absolute_locator(item: Any) -> None:
        if isinstance(item, Mapping):
            for child in item.values():
                reject_absolute_locator(child)
            return
        if isinstance(item, list):
            for child in item:
                reject_absolute_locator(child)
            return
        if isinstance(item, str) and Path(item).is_absolute():
            raise ValueError(
                "portable artifact metadata cannot contain absolute locators"
            )

    reject_absolute_locator(normalized)
    return normalized


def _encode_phase_result_value(value: Any, *, payload_root: Path) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _encode_phase_result_value(item, payload_root=payload_root)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _encode_phase_result_value(item, payload_root=payload_root)
            for item in value
        ]
    if isinstance(value, str) and Path(value).is_absolute():
        supplied = Path(value)
        if supplied.is_symlink():
            raise ValueError("workflow phase result cannot bind a symlink locator")
        try:
            resolved = supplied.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                "workflow phase result contains an absent absolute locator"
            ) from exc
        try:
            relative = resolved.relative_to(payload_root)
        except ValueError:
            # External execution locators are intentionally excluded from the
            # path-neutral scientific artifact. Downstream phase consumers may
            # use only registered payload locators after adoption.
            return {_OMITTED_EXTERNAL_LOCATOR_TOKEN: True}
        return {
            _PAYLOAD_PATH_TOKEN: (
                "." if relative == Path(".") else relative.as_posix()
            )
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not (
            float("-inf") < value < float("inf")
        ):
            raise ValueError("workflow phase result contains a non-finite value")
        return value
    raise TypeError(
        "workflow phase result must contain only JSON-compatible scalar values"
    )


def _phase_binding_body(
    *,
    phase: str,
    result: Mapping[str, Any],
    payload_root: Path,
    payloads: Sequence[PayloadRegistration],
) -> dict[str, Any]:
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("workflow phase binding requires a nonempty phase")
    if not isinstance(result, Mapping):
        raise TypeError("workflow phase binding requires one result mapping")
    terminal_files = result.get("terminal_files")
    if (
        not isinstance(terminal_files, list)
        or any(not isinstance(value, str) for value in terminal_files)
        or len(terminal_files) != len(set(terminal_files))
    ):
        raise ValueError(
            "workflow phase binding requires a unique terminal_files list"
        )
    registered = {row.relative_path for row in payloads}
    terminal_relative: list[str] = []
    for raw in terminal_files:
        supplied = Path(raw)
        if not supplied.is_absolute() or supplied.is_symlink():
            raise ValueError(
                "workflow phase terminal files must be absolute non-symlink payloads"
            )
        resolved = supplied.resolve(strict=True)
        try:
            relative = resolved.relative_to(payload_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                "workflow phase terminal file escapes the portable payload root"
            ) from exc
        if relative not in registered:
            raise ValueError(
                "workflow phase terminal file is absent from the payload inventory"
            )
        terminal_relative.append(relative)
    return {
        "schema_version": PORTABLE_PHASE_BINDING,
        "phase": phase,
        "result_template": _encode_phase_result_value(
            dict(result),
            payload_root=payload_root,
        ),
        "terminal_payload_paths": terminal_relative,
    }


def _validate_phase_binding(
    value: Any,
    *,
    payloads: Sequence[PayloadRegistration],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("portable workflow phase binding must be one object")
    required = {
        "schema_version",
        "phase",
        "result_template",
        "terminal_payload_paths",
        "content_sha256",
    }
    if set(value) != required:
        raise ValueError("portable workflow phase binding is not closed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("schema_version") != PORTABLE_PHASE_BINDING
        or not isinstance(value.get("phase"), str)
        or not str(value["phase"]).strip()
        or not isinstance(value.get("result_template"), Mapping)
        or value.get("content_sha256") != identity_sha256(body)
    ):
        raise ValueError("portable workflow phase binding is invalid")
    registered = {row.relative_path for row in payloads}
    terminal = value.get("terminal_payload_paths")
    if (
        not isinstance(terminal, list)
        or any(not isinstance(item, str) for item in terminal)
        or len(terminal) != len(set(terminal))
        or any(item not in registered for item in terminal)
    ):
        raise ValueError(
            "portable workflow phase binding has invalid terminal payloads"
        )

    def validate_template(item: Any) -> None:
        if isinstance(item, Mapping):
            if set(item) == {_PAYLOAD_PATH_TOKEN}:
                raw = item[_PAYLOAD_PATH_TOKEN]
                if not isinstance(raw, str):
                    raise ValueError("portable payload token is invalid")
                if raw != ".":
                    _normalize_relative(raw)
                    if raw not in registered and not any(
                        path.startswith(f"{raw}/") for path in registered
                    ):
                        raise ValueError(
                            "portable payload token is absent from the inventory"
                        )
                return
            if set(item) == {_OMITTED_EXTERNAL_LOCATOR_TOKEN}:
                if item[_OMITTED_EXTERNAL_LOCATOR_TOKEN] is not True:
                    raise ValueError("omitted external locator token is invalid")
                return
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError(
                        "portable workflow phase result keys must be strings"
                    )
                validate_template(child)
            return
        if isinstance(item, list):
            for child in item:
                validate_template(child)
            return
        if item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, float) and not (
                float("-inf") < item < float("inf")
            ):
                raise ValueError(
                    "portable workflow phase result contains a non-finite value"
                )
            return
        raise ValueError("portable workflow phase result template is invalid")

    validate_template(value["result_template"])
    return dict(value)


def _scientific_content_descriptor(
    *,
    payloads: Sequence[PayloadRegistration],
    phase_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Describe ordered logical payload roles without physical locators.

    Relative payload IDs are part of the producer's logical artifact schema,
    not deployment locators.  Binding them in producer order prevents two
    different file roles from being exchanged merely because their byte
    multisets happen to be equal.  Absolute payload/control roots remain in
    the locator record only.

    The workflow result template is execution metadata.  It may contain GPU
    inventory, worker assignments, UUIDs, and other machine-local values, so
    it is authenticated by the locator but deliberately excluded from the
    scientific content root.  Scientific scalar claims belong in closed
    ``artifact_metadata`` or in registered payload bytes.
    """

    ordered_payloads = [
        {
            "logical_relative_id": row.relative_path,
            "size_bytes": int(row.size_bytes),
            "sha256": row.sha256,
            "media_type": row.media_type,
        }
        for row in payloads
    ]

    phase_claims: Mapping[str, Any] | None = None
    if phase_binding is not None:
        binding = _validate_phase_binding(
            phase_binding,
            payloads=payloads,
        )
        phase_claims = {
            "phase": str(binding["phase"]),
            "terminal_payload_logical_ids": list(
                binding["terminal_payload_paths"]
            ),
            "operational_result_template_included": False,
        }
    body = {
        "schema_version": PORTABLE_SCIENTIFIC_CONTENT_DESCRIPTOR,
        "payload_count": len(payloads),
        "ordered_logical_payloads": ordered_payloads,
        "workflow_phase_claims": phase_claims,
        "absolute_filesystem_locators_included": False,
    }
    return {**body, "content_sha256": identity_sha256(body)}


def materialize_portable_phase(
    artifact: ValidatedPortableArtifact,
    *,
    expected_phase: str,
) -> Mapping[str, Any]:
    """Materialize one authenticated phase result from portable payload refs."""

    assert_validated_artifact_unchanged(artifact)
    binding = artifact.phase_binding
    if binding is None:
        raise ValueError(
            "portable artifact has no authenticated workflow phase binding"
        )
    binding = _validate_phase_binding(binding, payloads=artifact.payloads)
    if binding["phase"] != expected_phase:
        raise ValueError("portable artifact is bound to a different workflow phase")

    def decode(item: Any) -> Any:
        if isinstance(item, Mapping):
            if set(item) == {_PAYLOAD_PATH_TOKEN}:
                relative = str(item[_PAYLOAD_PATH_TOKEN])
                target = (
                    artifact.payload_root
                    if relative == "."
                    else artifact.payload_root / relative
                )
                return str(target.resolve(strict=True))
            if set(item) == {_OMITTED_EXTERNAL_LOCATOR_TOKEN}:
                return None
            return {str(key): decode(child) for key, child in item.items()}
        if isinstance(item, list):
            return [decode(child) for child in item]
        return item

    result = decode(binding["result_template"])
    if not isinstance(result, Mapping):
        raise RuntimeError("portable phase binding decoded to a non-mapping result")
    artifacts = [
        {
            "path": str(
                (artifact.payload_root / row.relative_path).resolve(strict=True)
            ),
            "relative_path": row.relative_path,
            "sha256": row.sha256,
            "size_bytes": row.size_bytes,
        }
        for row in artifact.payloads
    ]
    expected_terminal = [
        str((artifact.payload_root / relative).resolve(strict=True))
        for relative in binding["terminal_payload_paths"]
    ]
    if list(result.get("terminal_files") or ()) != expected_terminal:
        raise ValueError(
            "portable phase result terminal files differ from its binding"
        )
    return {
        "phase": expected_phase,
        "attempt_dir": str(artifact.payload_root),
        "result": dict(result),
        "artifacts": artifacts,
        "portable_artifact_id": artifact.artifact_id,
    }


def _expected_tree_directories(relative_files: set[str]) -> set[str]:
    expected: set[str] = set()
    for relative in relative_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected.add(parent.as_posix())
            parent = parent.parent
    return expected


def _safe_tree_inventory(
    root: Path,
    *,
    label: str,
) -> tuple[set[str], set[str]]:
    """Inspect every subtree entry and reject links and special files."""

    tree = Path(root)
    try:
        root_state = os.lstat(tree)
    except OSError as exc:
        raise FileNotFoundError(f"{label} is missing: {tree}") from exc
    if stat.S_ISLNK(root_state.st_mode) or not stat.S_ISDIR(root_state.st_mode):
        raise ValueError(f"{label} must be a non-symlink directory")
    files: set[str] = set()
    directories: set[str] = set()
    for path in tree.rglob("*"):
        state = os.lstat(path)
        relative = path.relative_to(tree).as_posix()
        if stat.S_ISLNK(state.st_mode):
            raise ValueError(f"{label} cannot contain symlinks: {relative}")
        if stat.S_ISDIR(state.st_mode):
            directories.add(relative)
            continue
        if not stat.S_ISREG(state.st_mode):
            raise ValueError(f"{label} cannot contain special files: {relative}")
        if int(state.st_nlink) != 1:
            raise ValueError(f"{label} cannot contain hard-linked files: {relative}")
        files.add(relative)
    return files, directories


def _validate_artifact_tree_boundaries(
    *,
    root: Path,
    payload_root: Path,
    payloads: Sequence[PayloadRegistration],
    payload_inventory_policy: str,
    reject_extra_files: bool,
) -> None:
    """Validate closed control trees and safe payload subtree boundaries."""

    registered_payloads = {row.relative_path for row in payloads}
    control_files, control_directories = _safe_tree_inventory(
        root,
        label="portable artifact control tree",
    )
    expected_control = {MANIFEST_NAME, LOCATOR_NAME}
    if payload_root == root:
        expected_control |= registered_payloads
    if reject_extra_files:
        expected_directories = _expected_tree_directories(expected_control)
        if (
            control_files != expected_control
            or control_directories != expected_directories
        ):
            raise ValueError(
                "portable artifact control tree contains missing or "
                "unregistered entries; "
                f"missing_files={sorted(expected_control - control_files)}, "
                f"extra_files={sorted(control_files - expected_control)}, "
                "missing_directories="
                f"{sorted(expected_directories - control_directories)}, "
                "extra_directories="
                f"{sorted(control_directories - expected_directories)}"
            )
    if payload_root == root:
        return

    observed_payloads, observed_directories = _safe_tree_inventory(
        payload_root,
        label="portable artifact payload tree",
    )
    if not registered_payloads.issubset(observed_payloads):
        raise ValueError(
            "portable payload tree is missing registered files; "
            f"missing={sorted(registered_payloads - observed_payloads)}"
        )
    if (
        reject_extra_files
        and payload_inventory_policy == COMPLETE_PAYLOAD_TREE
    ):
        expected_directories = _expected_tree_directories(
            registered_payloads
        )
        if (
            observed_payloads != registered_payloads
            or observed_directories != expected_directories
        ):
            raise ValueError(
                "portable payload tree contains missing or unregistered "
                "entries; "
                f"missing_files={sorted(registered_payloads - observed_payloads)}, "
                f"extra_files={sorted(observed_payloads - registered_payloads)}, "
                "missing_directories="
                f"{sorted(expected_directories - observed_directories)}, "
                "extra_directories="
                f"{sorted(observed_directories - expected_directories)}"
            )


def _portable_artifact_stat_inventory(
    *,
    root: Path,
    payload_root: Path,
    payloads: Sequence[PayloadRegistration],
    expected_identities: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, tuple[int, ...]]:
    paths = [root / MANIFEST_NAME, root / LOCATOR_NAME]
    paths.extend(payload_root / row.relative_path for row in payloads)
    inventory: dict[str, tuple[int, ...]] = {}
    expected = dict(expected_identities or {})
    for path in paths:
        is_control = path.parent == root and path.name in {
            MANIFEST_NAME,
            LOCATOR_NAME,
        }
        if not is_control:
            _safe_path_boundaries(
                root=payload_root,
                relative_path=path.relative_to(payload_root).as_posix(),
                label="portable artifact inventory path",
            )
        state = os.lstat(path)
        identity = _stat_identity(state)
        if (
            stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or int(state.st_nlink) != 1
        ):
            raise ValueError(
                "portable artifact inventory contains a non-private regular file"
            )
        key = str(path.resolve(strict=True))
        if key in expected and expected[key] != identity:
            raise RuntimeError(
                "portable artifact changed after its bytes were authenticated"
            )
        inventory[key] = identity
    if expected and set(expected) != set(inventory):
        raise RuntimeError(
            "portable artifact authenticated identity inventory is incomplete"
        )
    return inventory


def assert_validated_artifact_unchanged(
    artifact: ValidatedPortableArtifact,
) -> None:
    """Reuse a full-byte-authenticated handle inside one trusted process."""

    if not isinstance(artifact, ValidatedPortableArtifact):
        raise TypeError("validated portable artifact handle is required")
    try:
        current = _portable_artifact_stat_inventory(
            root=artifact.root,
            payload_root=artifact.payload_root,
            payloads=artifact.payloads,
        )
        _validate_artifact_tree_boundaries(
            root=artifact.root,
            payload_root=artifact.payload_root,
            payloads=artifact.payloads,
            payload_inventory_policy=artifact.payload_inventory_policy,
            reject_extra_files=True,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "portable artifact changed after full-byte authentication"
        ) from exc
    if current != dict(artifact.stat_inventory):
        raise RuntimeError("portable artifact changed after full-byte authentication")


def publish_portable_artifact(
    *,
    root: Path,
    artifact_kind: str,
    artifact_schema: str,
    compatibility: ArtifactCompatibility,
    upstream_artifact_ids: Sequence[str],
    payload_paths: Sequence[str],
    media_types: Mapping[str, str] | None = None,
    workflow_phase: str | None = None,
    workflow_phase_result: Mapping[str, Any] | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
    scientific_compatibility_version: str = SCIENTIFIC_COMPATIBILITY_VERSION,
) -> ValidatedPortableArtifact:
    """Seal an already-written payload tree without copying its payloads."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("portable artifact root must be an existing symlink-free directory")
    if artifact_kind not in CHECKPOINT_ARTIFACT_KINDS:
        raise ValueError(f"unsupported checkpoint artifact kind {artifact_kind!r}")
    if not artifact_schema:
        raise ValueError("artifact schema is required")
    if scientific_compatibility_version != SCIENTIFIC_COMPATIBILITY_VERSION:
        raise ValueError("scientific artifact schema downgrade or substitution is forbidden")
    if (root / MANIFEST_NAME).exists() or (root / LOCATOR_NAME).exists():
        raise FileExistsError("portable artifact is already sealed")
    upstream = tuple(str(value) for value in upstream_artifact_ids)
    if len(upstream) != len(set(upstream)) or any(
        _SHA256.fullmatch(value) is None for value in upstream
    ):
        raise ValueError("upstream artifact IDs must be unique lowercase SHA-256 values")
    normalized_paths = tuple(_normalize_relative(value) for value in payload_paths)
    if not normalized_paths or len(normalized_paths) != len(set(normalized_paths)):
        raise ValueError("payload inventory must be nonempty and contain unique paths")
    payloads: list[PayloadRegistration] = []
    media = dict(media_types or {})
    for relative in normalized_paths:
        path = root / relative
        before_boundaries = _safe_path_boundaries(
            root=root,
            relative_path=relative,
            label=f"artifact payload {relative}",
        )
        digest, size = _safe_file_hash(
            path,
            label=f"artifact payload {relative}",
        )
        if (
            _safe_path_boundaries(
                root=root,
                relative_path=relative,
                label=f"artifact payload {relative}",
            )
            != before_boundaries
        ):
            raise RuntimeError(
                f"artifact payload path changed while authenticating: {relative}"
            )
        payloads.append(
            PayloadRegistration(
                relative_path=relative,
                size_bytes=size,
                sha256=digest,
                media_type=media.get(relative, "application/octet-stream"),
            )
        )
    compatibility_payload = compatibility.as_dict()
    metadata = _closed_artifact_metadata(artifact_metadata)
    if (workflow_phase is None) != (workflow_phase_result is None):
        raise ValueError(
            "workflow_phase and workflow_phase_result must be supplied together"
        )
    phase_binding: Mapping[str, Any] | None = None
    if workflow_phase is not None and workflow_phase_result is not None:
        phase_body = _phase_binding_body(
            phase=workflow_phase,
            result=workflow_phase_result,
            payload_root=root.resolve(strict=True),
            payloads=payloads,
        )
        phase_binding = {
            **phase_body,
            "content_sha256": identity_sha256(phase_body),
        }
    body = {
        "schema_version": PORTABLE_ARTIFACT_MANIFEST,
        "artifact_kind": artifact_kind,
        "artifact_schema": artifact_schema,
        "scientific_compatibility_version": scientific_compatibility_version,
        "upstream_artifact_ids": list(upstream),
        "compatibility": compatibility_payload,
        "compatibility_key": compatibility.key,
        "payloads": [asdict(value) for value in payloads],
        "payload_inventory_policy": COMPLETE_PAYLOAD_TREE,
    }
    if metadata is not None:
        body["artifact_metadata"] = metadata
    if phase_binding is not None:
        body["workflow_phase_binding"] = phase_binding
    body["scientific_content_descriptor"] = (
        _scientific_content_descriptor(
            payloads=payloads,
            phase_binding=phase_binding,
        )
    )
    content_root = identity_sha256(_manifest_content_body(body))
    manifest = {
        **body,
        "content_root": content_root,
        "artifact_id": content_root,
    }
    _atomic_json_new(root / MANIFEST_NAME, manifest)
    locator_body = {
        "schema_version": PORTABLE_ARTIFACT_LOCATOR,
        "artifact_id": content_root,
        "root": str(root.resolve(strict=True)),
        "payload_root": str(root.resolve(strict=True)),
        "manifest_relative_path": MANIFEST_NAME,
        "payload_relative_paths": list(normalized_paths),
    }
    if phase_binding is not None:
        locator_body["operational_phase_binding_content_sha256"] = (
            phase_binding["content_sha256"]
        )
    locator = {**locator_body, "content_sha256": identity_sha256(locator_body)}
    _atomic_json_new(root / LOCATOR_NAME, locator)
    return validate_portable_artifact(root)


def publish_portable_reference_artifact(
    *,
    control_root: Path,
    payload_root: Path,
    artifact_kind: str,
    artifact_schema: str,
    compatibility: ArtifactCompatibility,
    upstream_artifact_ids: Sequence[str],
    payload_paths: Sequence[str],
    media_types: Mapping[str, str] | None = None,
    expected_payload_identities: Mapping[str, tuple[str, int]] | None = None,
    process_authenticated_stat_inventory: (
        Mapping[str, Sequence[int]] | None
    ) = None,
    workflow_phase: str | None = None,
    workflow_phase_result: Mapping[str, Any] | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
    payload_inventory_policy: str = COMPLETE_PAYLOAD_TREE,
    scientific_compatibility_version: str = SCIENTIFIC_COMPATIBILITY_VERSION,
) -> ValidatedPortableArtifact:
    """Seal existing payload bytes through a small no-copy control artifact.

    No payload byte is copied or modified. The path-neutral manifest contains
    only ordered sizes/hashes; the physical payload locator is kept in the
    separate locator record. The default policy requires ``payload_paths`` to
    account for the complete referenced tree. ``registered_paths_only`` is for
    immutable records that intentionally share a parent directory; every
    claimed byte remains authenticated, while sibling bytes belong to other
    independently indexed artifacts.

    A producer in the same trusted process may supply the exact stat inventory
    captured when those hashes were written or authenticated. This avoids a
    redundant full read while still checking the inventory before and after
    publication. A fresh process never receives this nonserializable handle
    and therefore reopens and hashes every registered byte.
    """

    control = Path(control_root)
    payload = Path(payload_root)
    if control.exists() or control.is_symlink():
        raise FileExistsError("portable reference control root must be absent")
    if payload.is_symlink() or not payload.is_dir():
        raise ValueError("portable reference payload root must be a real directory")
    payload = payload.resolve(strict=True)
    if artifact_kind not in CHECKPOINT_ARTIFACT_KINDS:
        raise ValueError(f"unsupported checkpoint artifact kind {artifact_kind!r}")
    if not artifact_schema:
        raise ValueError("artifact schema is required")
    if scientific_compatibility_version != SCIENTIFIC_COMPATIBILITY_VERSION:
        raise ValueError("scientific artifact schema downgrade or substitution is forbidden")
    inventory_policy = str(payload_inventory_policy)
    if inventory_policy not in _PAYLOAD_INVENTORY_POLICIES:
        raise ValueError("portable payload inventory policy is unsupported")
    upstream = tuple(str(value) for value in upstream_artifact_ids)
    if len(upstream) != len(set(upstream)) or any(
        _SHA256.fullmatch(value) is None for value in upstream
    ):
        raise ValueError("upstream artifact IDs must be unique lowercase SHA-256 values")
    normalized_paths = tuple(_normalize_relative(value) for value in payload_paths)
    if not normalized_paths or len(normalized_paths) != len(set(normalized_paths)):
        raise ValueError("payload inventory must be nonempty and contain unique paths")
    media = dict(media_types or {})
    expected_identities = dict(expected_payload_identities or {})
    if expected_identities and set(expected_identities) != set(normalized_paths):
        raise ValueError(
            "expected reference payload identities must cover the exact inventory"
        )
    trusted_stats = (
        None
        if process_authenticated_stat_inventory is None
        else {
            str(relative): tuple(int(value) for value in identity)
            for relative, identity in process_authenticated_stat_inventory.items()
        }
    )
    if trusted_stats is not None and (
        not expected_identities
        or set(trusted_stats) != set(normalized_paths)
    ):
        raise ValueError(
            "process-authenticated stat inventory requires exact payload "
            "identities and complete path coverage"
        )
    registrations: list[PayloadRegistration] = []
    publication_authentication_cache: dict[
        str, tuple[tuple[int, ...], str, int]
    ] = {}
    for relative in normalized_paths:
        path = payload / relative
        before_boundaries = _safe_path_boundaries(
            root=payload,
            relative_path=relative,
            label=f"portable reference payload {relative}",
        )
        resolved = path.resolve(strict=True)
        if resolved != path:
            raise ValueError(
                "portable reference payload path must be lexical and symlink-free"
            )
        if trusted_stats is None:
            digest, size, authenticated_identity = (
                _safe_file_hash_with_identity(
                    path,
                    label=f"portable reference payload {relative}",
                )
            )
        else:
            state = os.lstat(path)
            authenticated_identity = _stat_identity(state)
            if (
                stat.S_ISLNK(state.st_mode)
                or not stat.S_ISREG(state.st_mode)
                or int(state.st_nlink) != 1
                or authenticated_identity != trusted_stats[relative]
            ):
                raise RuntimeError(
                    "process-authenticated portable payload changed: "
                    f"{relative}"
                )
            digest, size = expected_identities[relative]
        after_boundaries = _safe_path_boundaries(
            root=payload,
            relative_path=relative,
            label=f"portable reference payload {relative}",
        )
        if (
            after_boundaries != before_boundaries
            or after_boundaries[str(path)] != authenticated_identity
        ):
            raise RuntimeError(
                "portable reference payload path changed while "
                f"authenticating: {relative}"
            )
        if expected_identities and expected_identities[relative] != (
            digest,
            size,
        ):
            raise ValueError(
                f"portable reference payload differs from its producer registration: {relative}"
            )
        registrations.append(
            PayloadRegistration(
                relative_path=relative,
                size_bytes=size,
                sha256=digest,
                media_type=media.get(relative, "application/octet-stream"),
            )
        )
        publication_authentication_cache[str(resolved)] = (
            authenticated_identity,
            digest,
            size,
        )
    observed_payloads, observed_directories = _safe_tree_inventory(
        payload,
        label="portable reference payload tree",
    )
    registered_payloads = set(normalized_paths)
    if not registered_payloads.issubset(observed_payloads):
        raise ValueError(
            "portable reference payload tree is missing registered files"
        )
    if inventory_policy == COMPLETE_PAYLOAD_TREE:
        expected_directories = _expected_tree_directories(
            registered_payloads
        )
        if (
            observed_payloads != registered_payloads
            or observed_directories != expected_directories
        ):
            raise ValueError(
                "portable reference inventory must cover the complete payload "
                "tree without extra entries"
            )
    compatibility_payload = compatibility.as_dict()
    metadata = _closed_artifact_metadata(artifact_metadata)
    if (workflow_phase is None) != (workflow_phase_result is None):
        raise ValueError(
            "workflow_phase and workflow_phase_result must be supplied together"
        )
    phase_binding: Mapping[str, Any] | None = None
    if workflow_phase is not None and workflow_phase_result is not None:
        phase_body = _phase_binding_body(
            phase=workflow_phase,
            result=workflow_phase_result,
            payload_root=payload,
            payloads=registrations,
        )
        phase_binding = {
            **phase_body,
            "content_sha256": identity_sha256(phase_body),
        }
    body = {
        "schema_version": PORTABLE_ARTIFACT_MANIFEST,
        "artifact_kind": artifact_kind,
        "artifact_schema": artifact_schema,
        "scientific_compatibility_version": scientific_compatibility_version,
        "upstream_artifact_ids": list(upstream),
        "compatibility": compatibility_payload,
        "compatibility_key": compatibility.key,
        "payloads": [asdict(value) for value in registrations],
        "payload_inventory_policy": inventory_policy,
    }
    if metadata is not None:
        body["artifact_metadata"] = metadata
    if phase_binding is not None:
        body["workflow_phase_binding"] = phase_binding
    body["scientific_content_descriptor"] = (
        _scientific_content_descriptor(
            payloads=registrations,
            phase_binding=phase_binding,
        )
    )
    content_root = identity_sha256(_manifest_content_body(body))
    control.mkdir(parents=True, exist_ok=False)
    manifest = {
        **body,
        "content_root": content_root,
        "artifact_id": content_root,
    }
    _atomic_json_new(
        control / MANIFEST_NAME,
        manifest,
    )
    locator_body = {
        "schema_version": PORTABLE_ARTIFACT_LOCATOR,
        "artifact_id": content_root,
        "root": str(control.resolve(strict=True)),
        "payload_root": str(payload),
        "manifest_relative_path": MANIFEST_NAME,
        "payload_relative_paths": list(normalized_paths),
        "payload_inventory_policy": inventory_policy,
    }
    if phase_binding is not None:
        locator_body["operational_phase_binding_content_sha256"] = (
            phase_binding["content_sha256"]
        )
    _atomic_json_new(
        control / LOCATOR_NAME,
        {**locator_body, "content_sha256": identity_sha256(locator_body)},
    )
    return validate_portable_artifact(
        control,
        expected_kind=artifact_kind,
        expected_compatibility_key=compatibility.key,
        expected_upstream_artifact_ids=upstream,
        payload_authentication_cache=publication_authentication_cache,
    )


def _manifest_content_body(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if "scientific_content_descriptor" in manifest:
        body = {
            key: manifest[key]
            for key in (
                "schema_version",
                "artifact_kind",
                "artifact_schema",
                "scientific_compatibility_version",
                "upstream_artifact_ids",
                "compatibility",
                "compatibility_key",
                "scientific_content_descriptor",
            )
        }
        if "artifact_metadata" in manifest:
            body["artifact_metadata"] = manifest["artifact_metadata"]
        return body
    body = {
        key: manifest[key]
        for key in (
            "schema_version",
            "artifact_kind",
            "artifact_schema",
            "scientific_compatibility_version",
            "upstream_artifact_ids",
            "compatibility",
            "compatibility_key",
            "payloads",
        )
    }
    if "workflow_phase_binding" in manifest:
        body["workflow_phase_binding"] = manifest["workflow_phase_binding"]
    if "payload_inventory_policy" in manifest:
        body["payload_inventory_policy"] = manifest["payload_inventory_policy"]
    if "artifact_metadata" in manifest:
        body["artifact_metadata"] = manifest["artifact_metadata"]
    return body


def validate_portable_artifact(
    source: Path,
    *,
    expected_kind: str | None = None,
    expected_compatibility_key: str | None = None,
    expected_upstream_artifact_ids: Sequence[str] | None = None,
    reject_extra_files: bool = True,
    payload_authentication_cache: MutableMapping[
        str, tuple[tuple[int, ...], str, int]
    ]
    | None = None,
) -> ValidatedPortableArtifact:
    """Freshly reopen every registered byte and validate the path-neutral root."""

    source = Path(source)
    root = source.parent if source.name == MANIFEST_NAME else source
    if root.is_symlink():
        raise ValueError("portable artifact root cannot be a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("portable artifact source must be a directory or manifest")
    manifest_path = root / MANIFEST_NAME
    locator_path = root / LOCATOR_NAME
    authenticated_identities: dict[str, tuple[int, ...]] = {}
    manifest_bytes, manifest_identity = _safe_read_with_identity(
        manifest_path,
        label="portable artifact manifest",
    )
    authenticated_identities[str(manifest_path.resolve(strict=True))] = (
        manifest_identity
    )
    manifest = _strict_json_bytes(
        manifest_bytes,
        label="portable artifact manifest",
    )
    required_manifest = {
        "schema_version",
        "artifact_kind",
        "artifact_schema",
        "scientific_compatibility_version",
        "upstream_artifact_ids",
        "compatibility",
        "compatibility_key",
        "payloads",
        "content_root",
        "artifact_id",
    }
    optional_manifest = {
        "workflow_phase_binding",
        "payload_inventory_policy",
        "artifact_metadata",
        "scientific_content_descriptor",
    }
    if not required_manifest.issubset(manifest) or not set(manifest).issubset(
        required_manifest | optional_manifest
    ):
        raise ValueError("portable artifact manifest has missing or extra fields")
    if (
        manifest["schema_version"] != PORTABLE_ARTIFACT_MANIFEST
        or manifest["scientific_compatibility_version"]
        != SCIENTIFIC_COMPATIBILITY_VERSION
    ):
        raise ValueError("portable artifact schema downgrade or substitution is forbidden")
    if manifest["artifact_kind"] not in CHECKPOINT_ARTIFACT_KINDS:
        raise ValueError("portable artifact kind is unsupported")
    if expected_kind is not None and manifest["artifact_kind"] != expected_kind:
        raise ValueError("portable artifact kind is incompatible with the requested phase")
    inventory_policy = str(
        manifest.get("payload_inventory_policy", COMPLETE_PAYLOAD_TREE)
    )
    if inventory_policy not in _PAYLOAD_INVENTORY_POLICIES:
        raise ValueError("portable artifact payload inventory policy is invalid")
    if "artifact_metadata" in manifest:
        if _closed_artifact_metadata(manifest["artifact_metadata"]) != manifest[
            "artifact_metadata"
        ]:
            raise ValueError("portable artifact metadata is not canonical")
    compatibility_raw = manifest["compatibility"]
    if not isinstance(compatibility_raw, Mapping):
        raise ValueError("portable artifact compatibility payload is invalid")
    compatibility = ArtifactCompatibility(**dict(compatibility_raw))
    if manifest["compatibility_key"] != compatibility.key:
        raise ValueError("portable artifact compatibility key is invalid")
    if (
        expected_compatibility_key is not None
        and manifest["compatibility_key"] != expected_compatibility_key
    ):
        raise ValueError("checkpoint is scientifically or runtime incompatible")
    upstream = manifest["upstream_artifact_ids"]
    if not isinstance(upstream, list) or len(upstream) != len(set(upstream)) or any(
        _SHA256.fullmatch(str(value)) is None for value in upstream
    ):
        raise ValueError("portable artifact upstream DAG is invalid")
    if (
        expected_upstream_artifact_ids is not None
        and tuple(upstream) != tuple(expected_upstream_artifact_ids)
    ):
        raise ValueError("checkpoint upstream dependencies do not match the new request")
    payload_rows = manifest["payloads"]
    if not isinstance(payload_rows, list) or not payload_rows:
        raise ValueError("portable artifact payload inventory is empty")
    payloads = tuple(PayloadRegistration(**dict(row)) for row in payload_rows)
    if len({row.relative_path for row in payloads}) != len(payloads):
        raise ValueError("portable artifact payload paths are duplicated")
    if "workflow_phase_binding" in manifest:
        _validate_phase_binding(
            manifest["workflow_phase_binding"],
            payloads=payloads,
        )
    if "scientific_content_descriptor" in manifest:
        expected_descriptor = _scientific_content_descriptor(
            payloads=payloads,
            phase_binding=manifest.get("workflow_phase_binding"),
        )
        if manifest["scientific_content_descriptor"] != expected_descriptor:
            raise ValueError(
                "portable artifact scientific content descriptor is invalid"
            )
    locator_bytes, locator_identity = _safe_read_with_identity(
        locator_path,
        label="portable artifact locator",
    )
    authenticated_identities[str(locator_path.resolve(strict=True))] = (
        locator_identity
    )
    locator = _strict_json_bytes(
        locator_bytes,
        label="portable artifact locator",
    )
    locator_keys = (
        "schema_version",
        "artifact_id",
        "root",
        "payload_root",
        "manifest_relative_path",
        "payload_relative_paths",
    )
    if "payload_inventory_policy" in locator:
        locator_keys = (*locator_keys, "payload_inventory_policy")
    if "operational_phase_binding_content_sha256" in locator:
        locator_keys = (
            *locator_keys,
            "operational_phase_binding_content_sha256",
        )
    locator_body = {
        key: locator.get(key)
        for key in locator_keys
    }
    if set(locator) != {*locator_body, "content_sha256"}:
        raise ValueError("portable artifact locator has missing or extra fields")
    raw_payload_root = locator_body["payload_root"]
    if not isinstance(raw_payload_root, str):
        raise ValueError("portable artifact payload locator is invalid")
    supplied_payload_root = Path(raw_payload_root)
    if supplied_payload_root.is_symlink():
        raise ValueError("portable artifact payload root cannot be a symlink")
    payload_root = supplied_payload_root.resolve(strict=True)
    if not payload_root.is_dir():
        raise ValueError("portable artifact payload root is not a directory")
    if (
        locator_body["schema_version"] != PORTABLE_ARTIFACT_LOCATOR
        or locator_body["artifact_id"] != manifest["artifact_id"]
        or locator_body["root"] != str(root)
        or locator_body["payload_root"] != str(payload_root)
        or locator_body["manifest_relative_path"] != MANIFEST_NAME
        or locator_body["payload_relative_paths"]
        != [row.relative_path for row in payloads]
        or locator_body.get("payload_inventory_policy", COMPLETE_PAYLOAD_TREE)
        != inventory_policy
        or (
            "operational_phase_binding_content_sha256" in locator_body
        )
        != ("workflow_phase_binding" in manifest)
        or locator_body.get("operational_phase_binding_content_sha256")
        != (
            manifest["workflow_phase_binding"]["content_sha256"]
            if "workflow_phase_binding" in manifest
            else None
        )
        or locator.get("content_sha256") != identity_sha256(locator_body)
    ):
        raise ValueError("portable artifact locator is absent, stale, or substituted")
    for row in payloads:
        path = payload_root / row.relative_path
        before_boundaries = _safe_path_boundaries(
            root=payload_root,
            relative_path=row.relative_path,
            label=f"portable artifact payload {row.relative_path}",
        )
        resolved = path.resolve(strict=True)
        if resolved != path:
            raise ValueError(
                "portable artifact payload path must be lexical and symlink-free"
            )
        cache_key = str(resolved)
        state_identity = before_boundaries[str(path)]
        cached = (
            None
            if payload_authentication_cache is None
            else payload_authentication_cache.get(cache_key)
        )
        if cached is not None:
            cached_state, observed_hash, observed_size = cached
            if cached_state != state_identity:
                raise RuntimeError(
                    f"portable artifact payload changed within trust boundary: "
                    f"{row.relative_path}"
                )
            authenticated_identity = cached_state
        else:
            (
                observed_hash,
                observed_size,
                authenticated_identity,
            ) = _safe_file_hash_with_identity(
                path,
                label=(
                    f"portable artifact payload {row.relative_path}"
                ),
            )
            if authenticated_identity != state_identity:
                raise RuntimeError(
                    f"portable artifact payload changed while authenticating: "
                    f"{row.relative_path}"
                )
            if payload_authentication_cache is not None:
                payload_authentication_cache[cache_key] = (
                    authenticated_identity,
                    observed_hash,
                    observed_size,
                )
        after_boundaries = _safe_path_boundaries(
            root=payload_root,
            relative_path=row.relative_path,
            label=f"portable artifact payload {row.relative_path}",
        )
        if (
            after_boundaries != before_boundaries
            or after_boundaries[str(path)] != authenticated_identity
        ):
            raise RuntimeError(
                f"portable artifact payload path changed while authenticating: "
                f"{row.relative_path}"
            )
        authenticated_identities[cache_key] = authenticated_identity
        if (observed_hash, observed_size) != (row.sha256, row.size_bytes):
            raise ValueError(f"portable artifact payload changed: {row.relative_path}")
    body = _manifest_content_body(manifest)
    content_root = identity_sha256(body)
    if (
        manifest["content_root"] != content_root
        or manifest["artifact_id"] != content_root
    ):
        raise ValueError("portable artifact content root is invalid")
    if locator_body["artifact_id"] != content_root:
        raise ValueError("portable artifact locator content root changed")
    _validate_artifact_tree_boundaries(
        root=root,
        payload_root=payload_root,
        payloads=payloads,
        payload_inventory_policy=inventory_policy,
        reject_extra_files=reject_extra_files,
    )
    final_stat_inventory = _portable_artifact_stat_inventory(
        root=root,
        payload_root=payload_root,
        payloads=payloads,
        expected_identities=authenticated_identities,
    )
    return ValidatedPortableArtifact(
        root=root,
        payload_root=payload_root,
        manifest_path=manifest_path,
        locator_path=locator_path,
        manifest=manifest,
        payloads=payloads,
        stat_inventory=final_stat_inventory,
    )


def _artifact_control_attestation_claims(
    artifact: ValidatedPortableArtifact,
) -> dict[str, Any]:
    """Bind one handle to its exact operational manifest and locator bytes."""

    assert_validated_artifact_unchanged(artifact)
    manifest_sha256, manifest_size, manifest_identity = (
        _safe_file_hash_with_identity(
            artifact.manifest_path,
            label="portable artifact manifest for adoption",
        )
    )
    locator_sha256, locator_size, locator_identity = (
        _safe_file_hash_with_identity(
            artifact.locator_path,
            label="portable artifact locator for adoption",
        )
    )
    expected_inventory = dict(artifact.stat_inventory)
    if (
        expected_inventory.get(str(artifact.manifest_path))
        != manifest_identity
        or expected_inventory.get(str(artifact.locator_path))
        != locator_identity
    ):
        raise RuntimeError(
            "portable artifact controls changed before adoption"
        )
    assert_validated_artifact_unchanged(artifact)
    phase_binding = artifact.phase_binding
    operational_binding_sha256 = (
        None
        if phase_binding is None
        else phase_binding.get("content_sha256")
    )
    if (
        operational_binding_sha256 is not None
        and _SHA256.fullmatch(str(operational_binding_sha256)) is None
    ):
        raise ValueError(
            "portable artifact operational phase binding is invalid"
        )
    return {
        "producer_manifest_sha256": manifest_sha256,
        "producer_manifest_size_bytes": manifest_size,
        "producer_locator_sha256": locator_sha256,
        "producer_locator_size_bytes": locator_size,
        "producer_operational_phase_binding_content_sha256": (
            operational_binding_sha256
        ),
    }


def adopt_checkpoint(
    *,
    source: Path,
    attestation_root: Path,
    consumer_request_sha256: str,
    expected_kind: str | None = None,
    expected_compatibility_key: str | None = None,
    expected_upstream_artifact_ids: Sequence[str] | None = None,
    validated_artifact: ValidatedPortableArtifact | None = None,
) -> Mapping[str, Any]:
    """Authenticate and adopt one complete checkpoint into a fresh request.

    Adoption records an immutable reference; it never merges/copies payload
    trees and has no force or loose-digest bypass.
    """

    if _SHA256.fullmatch(str(consumer_request_sha256)) is None:
        raise ValueError("consumer request identity must be one lowercase SHA-256")
    if validated_artifact is None:
        artifact = validate_portable_artifact(
            source,
            expected_kind=expected_kind,
            expected_compatibility_key=expected_compatibility_key,
            expected_upstream_artifact_ids=expected_upstream_artifact_ids,
        )
    else:
        artifact = validated_artifact
        requested_root = (
            Path(source).parent
            if Path(source).name == MANIFEST_NAME
            else Path(source)
        ).resolve(strict=True)
        if requested_root != artifact.root:
            raise ValueError("validated checkpoint handle does not match its source")
        assert_validated_artifact_unchanged(artifact)
        if (
            expected_kind is not None
            and artifact.manifest["artifact_kind"] != expected_kind
        ):
            raise ValueError(
                "portable artifact kind is incompatible with the requested phase"
            )
        if (
            expected_compatibility_key is not None
            and artifact.compatibility_key != expected_compatibility_key
        ):
            raise ValueError("checkpoint is scientifically or runtime incompatible")
        if (
            expected_upstream_artifact_ids is not None
            and tuple(artifact.manifest["upstream_artifact_ids"])
            != tuple(expected_upstream_artifact_ids)
        ):
            raise ValueError(
                "checkpoint upstream dependencies do not match the new request"
            )
    control_claims = _artifact_control_attestation_claims(artifact)
    stable_body = {
        "schema_version": PORTABLE_ADOPTION_ATTESTATION,
        "producer_artifact_id": artifact.artifact_id,
        "producer_artifact_kind": artifact.manifest["artifact_kind"],
        "producer_compatibility_key": artifact.compatibility_key,
        "producer_content_root": artifact.manifest["content_root"],
        "producer_locator": str(artifact.locator_path),
        **control_claims,
        "consumer_request_sha256": consumer_request_sha256,
        "validated_upstream_artifact_ids": list(
            artifact.manifest["upstream_artifact_ids"]
        ),
        "validation_policy": (
            "fresh_full_byte_and_exact_control_inventory_no_force_v3"
        ),
    }
    target_root = Path(attestation_root)
    if target_root.exists() and (target_root.is_symlink() or not target_root.is_dir()):
        raise ValueError("adoption attestation root must be a symlink-free directory")
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / f"{artifact.artifact_id}.adoption.json"
    if target.exists():
        try:
            observed = validate_checkpoint_adoption(
                attestation_path=target,
                artifact=artifact,
                consumer_request_sha256=consumer_request_sha256,
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "existing checkpoint adoption attestation conflicts"
            ) from exc
        observed_stable_body = {
            key: value
            for key, value in observed.items()
            if key not in {"content_sha256", "recorded_at"}
        }
        if observed_stable_body != stable_body:
            raise ValueError(
                "existing checkpoint adoption attestation conflicts"
            )
        return observed
    body = {
        **stable_body,
        "recorded_at": _utc_now(),
    }
    attestation = {
        **body,
        "content_sha256": identity_sha256(body),
    }
    _atomic_json_new(target, attestation)
    reopened = _strict_json_bytes(
        _safe_read(target, label="new checkpoint adoption attestation"),
        label="new checkpoint adoption attestation",
    )
    if reopened != attestation:
        raise RuntimeError("checkpoint adoption attestation changed after publication")
    return reopened


def validate_checkpoint_adoption(
    *,
    attestation_path: Path,
    artifact: ValidatedPortableArtifact,
    consumer_request_sha256: str,
) -> Mapping[str, Any]:
    """Freshly validate one producer-to-consumer adoption attestation."""

    if _SHA256.fullmatch(str(consumer_request_sha256)) is None:
        raise ValueError("consumer request identity must be one lowercase SHA-256")
    control_claims = _artifact_control_attestation_claims(artifact)
    value = _strict_json_bytes(
        _safe_read(
            Path(attestation_path),
            label="checkpoint adoption attestation",
        ),
        label="checkpoint adoption attestation",
    )
    required = {
        "schema_version",
        "producer_artifact_id",
        "producer_artifact_kind",
        "producer_compatibility_key",
        "producer_content_root",
        "producer_locator",
        "producer_manifest_sha256",
        "producer_manifest_size_bytes",
        "producer_locator_sha256",
        "producer_locator_size_bytes",
        "producer_operational_phase_binding_content_sha256",
        "consumer_request_sha256",
        "validated_upstream_artifact_ids",
        "validation_policy",
        "content_sha256",
        "recorded_at",
    }
    body = {
        key: item
        for key, item in value.items()
        if key != "content_sha256"
    }
    if (
        set(value) != required
        or value.get("schema_version") != PORTABLE_ADOPTION_ATTESTATION
        or value.get("producer_artifact_id") != artifact.artifact_id
        or value.get("producer_artifact_kind")
        != artifact.manifest["artifact_kind"]
        or value.get("producer_compatibility_key") != artifact.compatibility_key
        or value.get("producer_content_root") != artifact.manifest["content_root"]
        or value.get("producer_locator") != str(artifact.locator_path)
        or any(
            value.get(key) != expected
            for key, expected in control_claims.items()
        )
        or value.get("consumer_request_sha256") != consumer_request_sha256
        or value.get("validated_upstream_artifact_ids")
        != list(artifact.manifest["upstream_artifact_ids"])
        or value.get("validation_policy")
        != "fresh_full_byte_and_exact_control_inventory_no_force_v3"
        or value.get("content_sha256") != identity_sha256(body)
        or not isinstance(value.get("recorded_at"), str)
        or not str(value["recorded_at"]).strip()
    ):
        raise ValueError("checkpoint adoption attestation is invalid")
    return value


def relocate_portable_artifact(
    *,
    source: Path,
    target_root: Path,
) -> ValidatedPortableArtifact:
    """Relocate a complete artifact while preserving its path-neutral root."""

    artifact = validate_portable_artifact(source)
    target = Path(target_root)
    if target.exists() or target.is_symlink():
        raise FileExistsError("portable artifact relocation target must be absent")
    target.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.relocation_",
            dir=target.parent,
        )
    )
    try:
        for payload in artifact.payloads:
            destination = attempt / payload.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                artifact.payload_root / payload.relative_path,
                destination,
            )
            observed_hash, observed_size = _safe_file_hash(
                destination,
                label=f"relocated payload {payload.relative_path}",
            )
            if (observed_hash, observed_size) != (
                payload.sha256,
                payload.size_bytes,
            ):
                raise RuntimeError("portable artifact relocation changed payload bytes")
        shutil.copyfile(artifact.manifest_path, attempt / MANIFEST_NAME)
        locator_body = {
            "schema_version": PORTABLE_ARTIFACT_LOCATOR,
            "artifact_id": artifact.artifact_id,
            "root": str(target.resolve()),
            "payload_root": str(target.resolve()),
            "manifest_relative_path": MANIFEST_NAME,
            "payload_relative_paths": [
                payload.relative_path for payload in artifact.payloads
            ],
        }
        if "payload_inventory_policy" in artifact.manifest:
            locator_body["payload_inventory_policy"] = (
                artifact.payload_inventory_policy
            )
        if artifact.phase_binding is not None:
            locator_body[
                "operational_phase_binding_content_sha256"
            ] = artifact.phase_binding["content_sha256"]
        _atomic_json_new(
            attempt / LOCATOR_NAME,
            {**locator_body, "content_sha256": identity_sha256(locator_body)},
        )
        os.rename(attempt, target)
    except BaseException:
        # Preserve the failed attempt for audit; never merge it into the target.
        raise
    relocated = validate_portable_artifact(target)
    if relocated.artifact_id != artifact.artifact_id:
        raise RuntimeError("portable artifact content root changed during relocation")
    return relocated


def write_dense_npy_new(path: Path, array: Any) -> PayloadRegistration:
    """Write one dense mmap-safe array without pickle or compression."""

    import numpy as np

    value = np.asarray(array)
    if value.dtype.hasobject:
        raise ValueError("portable dense arrays cannot use object dtype")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    # np.save accepts a file handle and therefore cannot silently add a suffix.
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    digest, size = _safe_file_hash(path, label="new dense NPY payload")
    return PayloadRegistration(path.name, size, digest, "application/x-npy")


def write_table_parquet_new(path: Path, table: Any) -> PayloadRegistration:
    """Write one typed table payload; pickle is never an accepted fallback."""

    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table)
    table.to_parquet(path, index=False)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest, size = _safe_file_hash(path, label="new Parquet payload")
    return PayloadRegistration(
        path.name,
        size,
        digest,
        "application/vnd.apache.parquet",
    )


__all__ = [
    "ArtifactCompatibility",
    "CHECKPOINT_ARTIFACT_KINDS",
    "COMPLETE_PAYLOAD_TREE",
    "LOCATOR_NAME",
    "MANIFEST_NAME",
    "PORTABLE_ADOPTION_ATTESTATION",
    "PORTABLE_ARTIFACT_LOCATOR",
    "PORTABLE_ARTIFACT_MANIFEST",
    "PORTABLE_PHASE_BINDING",
    "REGISTERED_PAYLOAD_PATHS_ONLY",
    "PayloadRegistration",
    "SCIENTIFIC_COMPATIBILITY_VERSION",
    "ValidatedPortableArtifact",
    "adopt_checkpoint",
    "assert_validated_artifact_unchanged",
    "materialize_portable_phase",
    "publish_portable_artifact",
    "publish_portable_reference_artifact",
    "relocate_portable_artifact",
    "validate_checkpoint_adoption",
    "validate_portable_artifact",
    "write_dense_npy_new",
    "write_table_parquet_new",
]
