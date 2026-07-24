"""Read-only authenticated Stage 1 bundle loader for hierarchical discovery.

All ten exact-inner and cumulative-spent family adapters now feed the production
root graph.  This module revalidates the root bundle, canonical scope indexes,
lossless catalogs, typed family artifacts, and native descriptors before any
inputs are returned to the hierarchy runner.  Final one-shot E2E certification
is tracked separately from this implemented validation substrate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
)
from .production_stage1_bundle import (
    STAGE1_BUNDLE_MANIFEST_SCHEMA,
    STAGE1_BUNDLE_REQUEST_SCHEMA,
    STAGE1_COMPONENT_MANIFEST_SCHEMA,
    STAGE1_CUMULATIVE_ALL_TEN_ROOT_INDEX_SCHEMA,
    STAGE1_EMBEDDING_CLUSTER_FIT_IDENTITY_SCHEMA,
    STAGE1_EMBEDDING_CLUSTER_FIT_INDEX_SCHEMA,
    STAGE1_EXACT_INNER_ROOT_INDEX_SCHEMA,
    STAGE1_MATCHED_PAIR_PROOF_SCHEMA,
    STAGE1_RAW_EVIDENCE_SIDECAR_SCHEMA,
    STAGE1_SCOPE_INDEX_SCHEMA,
    _HEX_SHA256,
    _sha256_json,
    _registry_scopes,
    _source_identity,
    _validate_embedding_cluster_fit_identity,
    validate_embedding_cluster_feasibility_audit,
    validate_htr_input_nontruncation_audit,
)
from .production_embedding_cache_builder import (
    validate_published_production_embedding_cache,
)
from .production_stage1_hierarchy_contract import (
    validate_production_stage1_hierarchy_contract_identity,
    validate_production_stage1_hierarchy_request_bindings,
)
from .stage1_exact_inner_evidence import (
    CanonicalStage1SplitRegistry,
    validate_exact_inner_stage1_evidence_bundle,
)
from .stage1_cumulative_spent_evidence import (
    validate_cumulative_spent_stage1_evidence_bundle,
)
from .tfidf_topic_discovery import row_set_fingerprint

STAGE1_EXACT_INNER_INDEX_SCHEMA = "production_stage1_exact_inner_evidence_index_v1"
STAGE1_HIERARCHY_INPUTS_SCHEMA = "authenticated_stage1_hierarchy_inputs_v4"


@dataclass(frozen=True)
class _StableFileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    stat_identity: tuple[int, int, int, int, int]


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting ambiguous duplicate keys."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key!r}")
        result[key] = value
    return result


def _absolute_without_resolution(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _open_directory_no_symlinks(path: Path | str, *, label: str) -> tuple[int, Path]:
    """Open an absolute directory one component at a time without following links."""

    absolute = _absolute_without_resolution(path)
    if not absolute.is_absolute():
        raise ValueError(f"{label} must be absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError(f"{label} contains an unsafe path component")
            next_descriptor = os.open(
                component,
                flags | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            raise ValueError(f"{label} must be a directory")
        return descriptor, absolute
    except Exception:
        os.close(descriptor)
        raise


def _safe_relative_parts(value: Any, *, label: str) -> tuple[str, ...]:
    relative = Path(str(value or ""))
    if (
        not str(relative)
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} must be a non-traversing relative path")
    return tuple(relative.parts)


class _BundleRootCapability:
    """Descriptor-anchored, no-symlink view of one authenticated bundle root."""

    def __init__(self, root: Path | str) -> None:
        descriptor, absolute = _open_directory_no_symlinks(root, label="bundle root")
        self.path = absolute
        self._descriptor = descriptor
        self._lock = threading.RLock()
        self._closed = False

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                os.close(self._descriptor)
                self._closed = True

    def __del__(self) -> None:  # pragma: no cover - best-effort process cleanup
        try:
            self.close()
        except Exception:
            pass

    def _duplicate_root(self) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("bundle-root capability is closed")
            return os.dup(self._descriptor)

    def _open_relative(
        self,
        relative_path: Any,
        *,
        label: str,
        directory: bool,
    ) -> tuple[int, Path, tuple[str, ...]]:
        parts = _safe_relative_parts(relative_path, label=label)
        descriptor = self._duplicate_root()
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        try:
            for component in parts[:-1]:
                next_descriptor = os.open(
                    component,
                    directory_flags | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
            final_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
            if directory:
                final_flags |= os.O_DIRECTORY
            final_descriptor = os.open(
                parts[-1],
                final_flags | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = final_descriptor
            opened = os.fstat(descriptor)
            expected = stat.S_ISDIR(opened.st_mode) if directory else stat.S_ISREG(opened.st_mode)
            if not expected:
                raise ValueError(
                    f"{label} must be a {'directory' if directory else 'regular file'}"
                )
            return descriptor, self.path.joinpath(*parts), parts
        except Exception:
            os.close(descriptor)
            raise

    def snapshot(self, relative_path: Any, *, label: str) -> _StableFileSnapshot:
        descriptor, diagnostic_path, _parts = self._open_relative(
            relative_path,
            label=label,
            directory=False,
        )
        return _snapshot_open_descriptor(
            descriptor,
            path=diagnostic_path,
            label=label,
        )

    def sha256(
        self,
        relative_path: Any,
        *,
        label: str,
    ) -> tuple[str, tuple[int, int, int, int, int], Path]:
        descriptor, diagnostic_path, _parts = self._open_relative(
            relative_path,
            label=label,
            directory=False,
        )
        digest, identity = _hash_open_descriptor(descriptor, label=label)
        return digest, identity, diagnostic_path

    def directory_exists(self, relative_path: Any, *, label: str) -> bool:
        try:
            descriptor, _path, _parts = self._open_relative(
                relative_path,
                label=label,
                directory=True,
            )
        except (OSError, ValueError):
            return False
        os.close(descriptor)
        return True

    def walk_regular_files(self, relative_path: Any, *, label: str) -> tuple[str, ...]:
        descriptor, _path, _parts = self._open_relative(
            relative_path,
            label=label,
            directory=True,
        )

        def visit(directory_descriptor: int, prefix: tuple[str, ...]) -> list[str]:
            rows: list[str] = []
            for name in sorted(os.listdir(directory_descriptor)):
                if name in {"", ".", ".."}:
                    raise ValueError(f"{label} contains an unsafe directory entry")
                item_stat = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
                if stat.S_ISLNK(item_stat.st_mode):
                    raise ValueError(f"{label} cannot contain symlinks")
                if stat.S_ISDIR(item_stat.st_mode):
                    child = os.open(
                        name,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_descriptor,
                    )
                    try:
                        rows.extend(visit(child, (*prefix, name)))
                    finally:
                        os.close(child)
                elif stat.S_ISREG(item_stat.st_mode):
                    rows.append(Path(*prefix, name).as_posix())
                else:
                    raise ValueError(f"{label} contains a non-regular filesystem entry")
            return rows

        try:
            return tuple(visit(descriptor, ()))
        finally:
            os.close(descriptor)


def _snapshot_open_descriptor(
    descriptor: int,
    *,
    path: Path,
    label: str,
) -> _StableFileSnapshot:
    """Read and close one already safely opened regular-file descriptor."""

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if before_identity != after_identity or len(payload) != after_identity[2]:
        raise RuntimeError(f"{label} changed while its bytes were being authenticated")
    return _StableFileSnapshot(
        path=path,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        stat_identity=after_identity,
    )


def _hash_open_descriptor(
    descriptor: int,
    *,
    label: str,
) -> tuple[str, tuple[int, int, int, int, int]]:
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        digest = hashlib.sha256()
        observed_size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            observed_size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if before_identity != after_identity or observed_size != after_identity[2]:
        raise RuntimeError(f"{label} changed while it was being authenticated")
    return digest.hexdigest(), after_identity


def _read_stable_file_snapshot(path: Path, *, label: str) -> _StableFileSnapshot:
    """Read one regular file once and retain the exact authenticated bytes."""

    requested = _absolute_without_resolution(path)
    parent_descriptor, parent = _open_directory_no_symlinks(
        requested.parent,
        label=f"{label} parent directory",
    )
    try:
        descriptor = os.open(
            requested.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened as a regular non-symlink file") from exc
    finally:
        os.close(parent_descriptor)
    return _snapshot_open_descriptor(descriptor, path=parent / requested.name, label=label)


def _read_stable_file_sha256_no_symlinks(
    path: Path | str,
    *,
    label: str,
) -> tuple[str, tuple[int, int, int, int, int], Path]:
    requested = _absolute_without_resolution(path)
    parent_descriptor, parent = _open_directory_no_symlinks(
        requested.parent,
        label=f"{label} parent directory",
    )
    try:
        descriptor = os.open(
            requested.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)
    digest, identity = _hash_open_descriptor(descriptor, label=label)
    return digest, identity, parent / requested.name


def _load_json_snapshot(snapshot: _StableFileSnapshot, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(snapshot.payload, object_pairs_hook=_strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid JSON: {snapshot.path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object: {snapshot.path}")
    return value


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    return _load_json_snapshot(
        _read_stable_file_snapshot(path, label=label),
        label=label,
    )


def _root_capability(root: Path | _BundleRootCapability) -> tuple[_BundleRootCapability, bool]:
    if isinstance(root, _BundleRootCapability):
        return root, False
    return _BundleRootCapability(root), True


def _inside(
    root: Path | _BundleRootCapability,
    relative_path: Any,
    *,
    label: str,
    directory: bool = False,
) -> Path:
    """Resolve a diagnostic path only after descriptor-relative safe opening."""

    capability, temporary = _root_capability(root)
    try:
        descriptor, path, _parts = capability._open_relative(
            relative_path,
            label=label,
            directory=directory,
        )
        os.close(descriptor)
        return path
    finally:
        if temporary:
            capability.close()


def _registered_file_snapshot(
    root: Path | _BundleRootCapability,
    value: Any,
    *,
    label: str,
) -> _StableFileSnapshot:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} registration must be an object")
    capability, temporary = _root_capability(root)
    try:
        snapshot = capability.snapshot(value.get("relative_path"), label=label)
    finally:
        if temporary:
            capability.close()
    try:
        size = int(value.get("size", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} registration has an invalid size") from exc
    digest = str(value.get("sha256") or "")
    if (
        size != snapshot.stat_identity[2]
        or _HEX_SHA256.fullmatch(digest) is None
        or snapshot.sha256 != digest
    ):
        raise ValueError(f"{label} registered bytes changed")
    return snapshot


def _registered_file(
    root: Path | _BundleRootCapability,
    value: Any,
    *,
    label: str,
) -> Path:
    return _registered_file_snapshot(root, value, label=label).path


def _registered_file_hash(
    root: Path | _BundleRootCapability,
    value: Any,
    *,
    label: str,
) -> tuple[Path, str, tuple[int, int, int, int, int]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} registration must be an object")
    capability, temporary = _root_capability(root)
    try:
        digest, identity, path = capability.sha256(value.get("relative_path"), label=label)
    finally:
        if temporary:
            capability.close()
    try:
        size = int(value.get("size", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} registration has an invalid size") from exc
    if (
        size != identity[2]
        or _HEX_SHA256.fullmatch(str(value.get("sha256") or "")) is None
        or digest != value.get("sha256")
    ):
        raise ValueError(f"{label} registered bytes changed")
    return path, digest, identity


def _registered_json(
    root: Path | _BundleRootCapability,
    value: Any,
    *,
    label: str,
) -> tuple[Path, Mapping[str, Any], _StableFileSnapshot]:
    snapshot = _registered_file_snapshot(root, value, label=label)
    return snapshot.path, _load_json_snapshot(snapshot, label=label), snapshot


def _sha_registry(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError(f"{label} must cover exactly all ten architecture families")
    result = {str(key): str(raw) for key, raw in value.items()}
    if any(_HEX_SHA256.fullmatch(raw) is None for raw in result.values()):
        raise ValueError(f"{label} contains a malformed SHA-256")
    return {family: result[family] for family in ACTIVE_STAGE1_CONCEPT_FAMILIES}


def _contract_registry(wrapper_registry: Mapping[str, Any]) -> CanonicalStage1SplitRegistry:
    try:
        registry = CanonicalStage1SplitRegistry.build(
            dataset_row_ids=tuple(range(int(wrapper_registry["dataset_row_count"]))),
            outer_heldout_row_ids={
                int(row["outer_fold"]): tuple(map(int, row["heldout_row_ids"]))
                for row in wrapper_registry["outer_folds"]
            },
            inner_fold_count=len(wrapper_registry["outer_folds"][0]["inner_folds"]),
            inner_seed_base=int(wrapper_registry.get("inner_seed_base", 51_000)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "bundle split registry cannot instantiate the exact-inner contract"
        ) from exc
    observed = {
        (int(outer["outer_fold"]), int(inner["inner_fold"])): (
            tuple(map(int, inner["fit_row_ids"])),
            tuple(map(int, inner["heldout_row_ids"])),
        )
        for outer in wrapper_registry["outer_folds"]
        for inner in outer["inner_folds"]
    }
    expected = {
        (outer.outer_fold, inner.inner_fold): (inner.fit_row_ids, inner.heldout_row_ids)
        for outer in registry.outer_splits
        for inner in outer.inner_splits
    }
    if observed != expected:
        raise ValueError("bundle split registry drifts from the exact-inner contract")
    return registry


def _strict_json_lines(
    snapshot: _StableFileSnapshot, *, label: str
) -> tuple[Mapping[str, Any], ...]:
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_strict_json_object)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{label} line {line_number} is invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} line {line_number} must contain one object")
        rows.append(value)
    return tuple(rows)


def _validate_raw_sidecar_snapshot(
    snapshot: _StableFileSnapshot,
    *,
    registration: Mapping[str, Any],
    scope: Mapping[str, Any],
    split_registry_content_sha256: str,
) -> Mapping[str, Any]:
    if snapshot.stat_identity[2] != int(
        registration.get("size", -1)
    ) or snapshot.sha256 != registration.get("sha256"):
        raise RuntimeError("raw evidence sidecar registered bytes changed")
    payload = _load_json_snapshot(snapshot, label="legacy raw evidence sidecar")
    body = dict(payload)
    declared = body.pop("content_sha256", None)
    if not _HEX_SHA256.fullmatch(str(declared or "")) or _sha256_json(body) != declared:
        raise RuntimeError("raw evidence sidecar content hash is invalid")
    expected = {
        "schema_version": STAGE1_RAW_EVIDENCE_SIDECAR_SCHEMA,
        "scope_id": str(scope["scope_id"]),
        "outer_fold": int(scope["outer_fold"]),
        "inner_fold": (None if scope.get("inner_fold") is None else int(scope["inner_fold"])),
        "fit_row_fingerprint": row_set_fingerprint(scope["fit_row_ids"]),
        "heldout_row_fingerprint": row_set_fingerprint(scope["heldout_row_ids"]),
        "split_registry_content_sha256": split_registry_content_sha256,
        "prompt_grounding_allowed": False,
        "raw_drillback_requires_authenticated_id": True,
    }
    mismatched = [
        key for key, expected_value in expected.items() if payload.get(key) != expected_value
    ]
    if mismatched or registration.get("content_sha256") != declared:
        raise RuntimeError(f"raw evidence sidecar scope binding is invalid; fields={mismatched}")
    proofs = payload.get("matched_pair_subproducer_proofs")
    subproducers = proofs.get("subproducers") if isinstance(proofs, Mapping) else None
    if (
        not isinstance(proofs, Mapping)
        or proofs.get("schema_version") != STAGE1_MATCHED_PAIR_PROOF_SCHEMA
        or proofs.get("scope_id") != str(scope["scope_id"])
        or proofs.get("all_required_subproducers_succeeded") is not True
        or not isinstance(subproducers, Mapping)
        or set(subproducers) != {"bow", "htr"}
        or proofs.get("content_sha256") != _sha256_json(subproducers or {})
        or any(
            not isinstance(row, Mapping)
            or row.get("schema_version") != STAGE1_MATCHED_PAIR_PROOF_SCHEMA
            or row.get("subproducer") != name
            or row.get("success") is not True
            or not isinstance(row.get("output_columns"), list)
            or not row.get("output_columns")
            or not _HEX_SHA256.fullmatch(str(row.get("model_artifact_sha256") or ""))
            or not _HEX_SHA256.fullmatch(str(row.get("fit_execution_sha256") or ""))
            for name, row in (subproducers or {}).items()
        )
    ):
        raise RuntimeError("raw evidence sidecar lacks separate matched-pair proofs")
    return payload


def _validate_legacy_scope_lineage_snapshots(
    *,
    handoff_snapshot: _StableFileSnapshot,
    scope_index: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_content_sha256: str,
    bundle_root: _BundleRootCapability,
    legacy_component_root: Path,
) -> None:
    if (
        scope_index.get("schema_version") != STAGE1_SCOPE_INDEX_SCHEMA
        or scope_index.get("split_registry_content_sha256") != registry_content_sha256
        or not isinstance(scope_index.get("scopes"), list)
    ):
        raise ValueError("legacy exact-scope index has an invalid registry binding")
    indexed_scopes = {
        str(row.get("scope_id")): row for row in scope_index["scopes"] if isinstance(row, Mapping)
    }
    if len(indexed_scopes) != len(scope_index["scopes"]):
        raise ValueError("legacy exact-scope index contains duplicates or malformed rows")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in _strict_json_lines(handoff_snapshot, label="legacy handoff"):
        outer_fold = int(row["outer_fold"])
        inner_fold = row.get("inner_fold")
        scope_id = (
            f"outer_{outer_fold:03d}_inner_{int(inner_fold):03d}"
            if inner_fold is not None
            else f"outer_{outer_fold:03d}_full"
        )
        if scope_id in rows:
            raise ValueError(f"duplicate legacy scope {scope_id}")
        if row.get("evidence_reused_from_fold_key") is not None:
            raise ValueError("legacy exact-inner evidence was reused instead of refit")
        if not row.get("evidence_scope_fit_was_executed"):
            raise ValueError("legacy evidence scope lacks an executed-fit attestation")
        if row.get("heldout_labels_supplied_to_evidence_builder") is not False:
            raise ValueError("legacy evidence scope received held-out labels")
        if (
            row.get("lossless_concept_catalog_projection") is not True
            or row.get("prompt_compactor_used") is not False
        ):
            raise ValueError("legacy evidence scope did not use the lossless projection")
        if row.get("split_registry_content_sha256") != registry_content_sha256:
            raise ValueError("legacy evidence scope has the wrong split-registry binding")
        rows[scope_id] = row
    expected = {str(scope["scope_id"]): scope for scope in _registry_scopes(registry)}
    if set(rows) != set(expected) or set(indexed_scopes) != set(expected):
        raise ValueError("legacy handoff does not match the canonical scope registry")
    for scope_id, scope in expected.items():
        row = rows[scope_id]
        indexed = indexed_scopes[scope_id]
        is_inner = scope["inner_fold"] is not None
        expected_scope = "candidate_consistency_inner_train" if is_inner else "full_outer_train"
        expected_fold_key = (
            int(scope["outer_fold"]) * 1000 + int(scope["inner_fold"])
            if is_inner
            else int(scope["outer_fold"])
        )
        if (
            row.get("scope") != expected_scope
            or int(row.get("fold_key", 0)) != expected_fold_key
            or int(row.get("n_rows", 0)) != len(scope["fit_row_ids"])
            or (is_inner and int(row.get("heldout_rows", 0)) != len(scope["heldout_row_ids"]))
            or list(map(int, row.get("fit_row_ids") or ())) != scope["fit_row_ids"]
            or list(map(int, row.get("heldout_row_ids") or ())) != scope["heldout_row_ids"]
            or row.get("fit_row_fingerprint") != row_set_fingerprint(scope["fit_row_ids"])
            or row.get("heldout_row_fingerprint") != row_set_fingerprint(scope["heldout_row_ids"])
        ):
            raise ValueError(f"legacy scope lineage mismatch: {scope_id}")
        raw_registration = indexed.get("raw_evidence_sidecar")
        if not isinstance(raw_registration, Mapping):
            raise ValueError(f"legacy scope lacks a raw evidence sidecar: {scope_id}")
        relative_parts = _safe_relative_parts(
            raw_registration.get("relative_path"),
            label=f"legacy raw sidecar {scope_id}",
        )
        sidecar_snapshot = _registered_file_snapshot(
            bundle_root,
            {
                **raw_registration,
                "relative_path": (legacy_component_root / Path(*relative_parts)).as_posix(),
            },
            label=f"legacy raw sidecar {scope_id}",
        )
        sidecar = _validate_raw_sidecar_snapshot(
            sidecar_snapshot,
            registration=raw_registration,
            scope=scope,
            split_registry_content_sha256=registry_content_sha256,
        )
        if row.get("raw_evidence_sidecar_sha256") != raw_registration.get("sha256") or indexed.get(
            "matched_pair_subproducer_proofs_sha256"
        ) != (sidecar.get("matched_pair_subproducer_proofs") or {}).get("content_sha256"):
            raise ValueError(f"legacy scope raw sidecar linkage mismatch: {scope_id}")


def _validate_embedding_cluster_fit_index_snapshot(
    *,
    index: Mapping[str, Any],
    index_snapshot: _StableFileSnapshot,
    legacy_scope_index: Mapping[str, Any],
    cluster_audit: Mapping[str, Any],
    request_sha256: str,
    registry_content_sha256: str,
    bundle_root: _BundleRootCapability,
    legacy_component_root: Path,
) -> None:
    """Reopen and semantically authenticate every fitted cluster identity."""

    index_fields = {
        "schema_version",
        "request_sha256",
        "split_registry_content_sha256",
        "preflight_audit_content_sha256",
        "scope_count",
        "full_outer_scope_count",
        "exact_inner_scope_count",
        "cumulative_spent_scope_count",
        "scope_order",
        "all_actual_identities_equal_preflight",
        "scopes",
        "content_sha256",
    }
    body = {key: copy.deepcopy(value) for key, value in index.items() if key != "content_sha256"}
    preflight_scopes = cluster_audit.get("scopes")
    expected_order = cluster_audit.get("scope_order")
    rows = index.get("scopes")
    if (
        not isinstance(index, Mapping)
        or set(index) != index_fields
        or index.get("schema_version") != STAGE1_EMBEDDING_CLUSTER_FIT_INDEX_SCHEMA
        or index.get("request_sha256") != request_sha256
        or index.get("split_registry_content_sha256") != registry_content_sha256
        or index.get("preflight_audit_content_sha256")
        != cluster_audit.get("content_sha256")
        or index.get("content_sha256") != _sha256_json(body)
        or index.get("all_actual_identities_equal_preflight") is not True
        or not isinstance(preflight_scopes, list)
        or not isinstance(expected_order, list)
        or not isinstance(rows, list)
        or index.get("scope_order") != expected_order
        or [row.get("scope_id") for row in rows if isinstance(row, Mapping)]
        != expected_order
        or int(index.get("scope_count", -1)) != len(expected_order)
    ):
        raise ValueError("cluster-fit index has an invalid closed binding or scope order")
    expected_counts = {
        "full_outer_scope_count": sum(
            row.get("scope_kind") == "full_outer" for row in preflight_scopes
        ),
        "exact_inner_scope_count": sum(
            row.get("scope_kind") == "exact_inner" for row in preflight_scopes
        ),
        "cumulative_spent_scope_count": sum(
            row.get("scope_kind") == "cumulative_spent" for row in preflight_scopes
        ),
    }
    if any(int(index.get(key, -1)) != value for key, value in expected_counts.items()):
        raise ValueError("cluster-fit index scope-kind counts changed")
    preflight_by_scope = {
        str(row.get("scope_id")): row
        for row in preflight_scopes
        if isinstance(row, Mapping)
    }
    if len(preflight_by_scope) != len(preflight_scopes) or set(preflight_by_scope) != set(
        expected_order
    ):
        raise ValueError("cluster preflight scope identities are duplicated or incomplete")

    component_registration = legacy_scope_index.get("embedding_cluster_fit_index")
    expected_relative = (
        legacy_component_root / "embedding_cluster_fit_index.json"
    ).as_posix()
    if (
        not isinstance(component_registration, Mapping)
        or component_registration.get("relative_path") != "embedding_cluster_fit_index.json"
        or int(component_registration.get("size", -1)) != index_snapshot.stat_identity[2]
        or component_registration.get("sha256") != index_snapshot.sha256
        or index_snapshot.path
        != bundle_root.path / Path(expected_relative)
    ):
        raise ValueError("legacy scope index substituted its cluster-fit index")

    row_fields = {"scope_id", "scope_kind", "identity_sha256", "record"}
    record_fields = {
        "schema_version",
        "scope_id",
        "scope_kind",
        "preflight_identity_sha256",
        "actual_identity",
        "actual_equals_preflight",
        "content_sha256",
    }
    observed: set[str] = set()
    for row, scope_id in zip(rows, expected_order, strict=True):
        if not isinstance(row, Mapping) or set(row) != row_fields:
            raise ValueError("cluster-fit index contains a malformed scope row")
        expected_scope = preflight_by_scope[str(scope_id)]
        preflight_identity = expected_scope.get("cluster_fit_identity")
        fit_rows = tuple(
            map(
                int,
                (
                    preflight_identity.get("fit_row_ids")
                    if isinstance(preflight_identity, Mapping)
                    else ()
                )
                or (),
            )
        )
        expected_identity = _validate_embedding_cluster_fit_identity(
            preflight_identity,
            scope_id=str(scope_id),
            fit_row_ids=fit_rows,
        )
        if (
            str(scope_id) in observed
            or row.get("scope_id") != scope_id
            or row.get("scope_kind") != expected_scope.get("scope_kind")
            or row.get("identity_sha256") != expected_identity["content_sha256"]
        ):
            raise ValueError("cluster-fit index reordered or substituted a scope")
        observed.add(str(scope_id))
        registration = row.get("record")
        expected_record_relative = (
            f"embedding_cluster_fit_records/{scope_id}.json"
        )
        if (
            not isinstance(registration, Mapping)
            or set(registration) != {"relative_path", "size", "sha256"}
            or registration.get("relative_path") != expected_record_relative
        ):
            raise ValueError("cluster-fit record registration is not canonical")
        _path, record, _snapshot = _registered_json(
            bundle_root,
            {
                **registration,
                "relative_path": (
                    legacy_component_root / expected_record_relative
                ).as_posix(),
            },
            label=f"cluster-fit identity record {scope_id}",
        )
        record_body = {
            key: copy.deepcopy(value)
            for key, value in record.items()
            if key != "content_sha256"
        }
        try:
            actual_identity = _validate_embedding_cluster_fit_identity(
                record.get("actual_identity"),
                scope_id=str(scope_id),
                fit_row_ids=fit_rows,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"cluster-fit record has an invalid identity: {scope_id}"
            ) from exc
        if (
            set(record) != record_fields
            or record.get("schema_version")
            != STAGE1_EMBEDDING_CLUSTER_FIT_IDENTITY_SCHEMA
            or record.get("scope_id") != scope_id
            or record.get("scope_kind") != expected_scope.get("scope_kind")
            or record.get("preflight_identity_sha256")
            != expected_identity["content_sha256"]
            or record.get("actual_equals_preflight") is not True
            or record.get("content_sha256") != _sha256_json(record_body)
            or actual_identity != expected_identity
        ):
            raise ValueError(
                f"cluster-fit record differs from accepted preflight: {scope_id}"
            )
    if observed != set(map(str, expected_order)):
        raise ValueError("cluster-fit index does not cover every preflight scope")


def _validate_component(
    bundle_root: _BundleRootCapability,
    *,
    request_sha256: str,
    name: str,
    registration: Any,
) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    if not isinstance(registration, Mapping):
        raise ValueError(f"{name} component registration must be an object")
    component_parts = _safe_relative_parts(
        registration.get("relative_path"),
        label=f"{name} component",
    )
    component_relative = Path(*component_parts).as_posix()
    if not bundle_root.directory_exists(component_relative, label=f"{name} component"):
        raise ValueError(f"{name} component must be a symlink-free directory")
    manifest_snapshot = bundle_root.snapshot(
        f"{component_relative}/component_manifest.json",
        label=f"{name} component manifest",
    )
    if manifest_snapshot.sha256 != registration.get("manifest_sha256"):
        raise ValueError(f"{name} component manifest bytes changed")
    manifest = _load_json_snapshot(manifest_snapshot, label=f"{name} component manifest")
    body = dict(manifest)
    declared = body.pop("content_sha256", None)
    if (
        manifest.get("schema_version") != STAGE1_COMPONENT_MANIFEST_SCHEMA
        or manifest.get("request_sha256") != request_sha256
        or manifest.get("component") != name
        or _HEX_SHA256.fullmatch(str(declared or "")) is None
        or _sha256_json(body) != declared
        or registration.get("content_sha256") != declared
        or not isinstance(manifest.get("files"), list)
    ):
        raise ValueError(f"{name} component manifest identity is invalid")
    inventory = manifest["files"]
    expected_paths: set[str] = set()
    registrations: dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(inventory):
        if not isinstance(raw_row, Mapping) or set(raw_row) != {
            "relative_path",
            "size",
            "sha256",
        }:
            raise ValueError(f"{name} component inventory row {index} is invalid")
        relative_parts = _safe_relative_parts(
            raw_row.get("relative_path"),
            label=f"{name} component inventory row {index}",
        )
        relative_name = Path(*relative_parts).as_posix()
        if relative_name == "component_manifest.json" or relative_name in expected_paths:
            raise ValueError(f"{name} component inventory paths are invalid or duplicated")
        expected_paths.add(relative_name)
        registrations[relative_name] = raw_row
    observed_paths = set(
        bundle_root.walk_regular_files(component_relative, label=f"{name} component")
    )
    observed_paths.discard("component_manifest.json")
    if observed_paths != expected_paths:
        raise RuntimeError(f"authenticated component file set changed: {name}")
    for relative_name, raw_row in registrations.items():
        digest, file_identity, _path = bundle_root.sha256(
            f"{component_relative}/{relative_name}",
            label=f"{name} component file {relative_name}",
        )
        if file_identity[2] != int(raw_row.get("size", -1)) or digest != raw_row.get("sha256"):
            raise RuntimeError(f"authenticated component file changed: {name}/{relative_name}")
    return component_parts, copy.deepcopy(dict(manifest))


@dataclass(frozen=True)
class AuthenticatedStage1HierarchyInputs:
    bundle_manifest_path: Path
    bundle_sha256: str
    request_sha256: str
    hierarchical_discovery_contract_identity: Mapping[str, Any]
    dataset_path: Path
    stage1_config_path: Path
    split_registry_path: Path
    primary_splits_path: Path
    legacy_handoff_path: Path
    legacy_scope_index_path: Path
    embedding_cluster_fit_index_path: Path
    tfidf_handoff_path: Path
    neural_query_artifact_index_path: Path
    exact_inner_evidence_index_path: Path
    embedding_cache_dir: Path
    neural_query_full_outer_artifacts: tuple[tuple[int, Path, str], ...]
    _bundle_root_capability: _BundleRootCapability = field(repr=False, compare=False)
    _bundle_manifest_snapshot: _StableFileSnapshot = field(repr=False, compare=False)
    _bundle_manifest_json: Mapping[str, Any] = field(repr=False, compare=False)
    _registered_snapshots: Mapping[str, _StableFileSnapshot] = field(
        repr=False,
        compare=False,
    )
    _registered_json_values: Mapping[str, Mapping[str, Any]] = field(
        repr=False,
        compare=False,
    )
    _component_roots: Mapping[str, tuple[str, ...]] = field(repr=False, compare=False)
    _component_manifests: Mapping[str, Mapping[str, Any]] = field(
        repr=False,
        compare=False,
    )
    _embedding_cache_capability: _BundleRootCapability = field(repr=False, compare=False)

    def _authenticated_manifest(self) -> Mapping[str, Any]:
        value = copy.deepcopy(dict(self._bundle_manifest_json))
        body = dict(value)
        declared = body.pop("bundle_sha256", None)
        if declared != self.bundle_sha256 or _sha256_json(body) != self.bundle_sha256:
            raise RuntimeError("retained Stage 1 bundle manifest differs from its binding")
        return value

    def _authenticated_registered_snapshot(self, key: str) -> _StableFileSnapshot:
        try:
            snapshot = self._registered_snapshots[key]
        except KeyError as exc:
            raise ValueError(f"Stage 1 input has no retained registered snapshot: {key}") from exc
        return snapshot

    def _authenticated_registered_json(self, key: str) -> Mapping[str, Any]:
        try:
            value = self._registered_json_values[key]
        except KeyError as exc:
            raise ValueError(f"Stage 1 input has no retained registered JSON: {key}") from exc
        return copy.deepcopy(dict(value))

    def compatibility_cli_arguments(self) -> tuple[str, ...]:
        """Return historical handoff arguments for diagnostics only.

        These paths are authenticated, but the low-level hierarchy currently
        uses them to refit an independently generated accumulated-spent
        schedule.  They are therefore not a production Stage-1-to-hierarchy
        handoff and must not be executed as one.
        """

        arguments = [
            "--dataset",
            str(self.dataset_path),
            "--legacy-handoff",
            str(self.legacy_handoff_path),
            "--resealed-tfidf-handoff",
            str(self.tfidf_handoff_path),
            "--primary-splits",
            str(self.primary_splits_path),
            "--review-stage1-config",
            str(self.stage1_config_path),
            "--review-embedding-cache-dir",
            str(self.embedding_cache_dir),
            "--require-neural-query-moments",
        ]
        for fold, path, digest in self.neural_query_full_outer_artifacts:
            arguments.extend(["--neural-query-moment-artifact", f"{fold}={path}::{digest}"])
        return tuple(arguments)

    def hierarchy_cli_arguments(self) -> tuple[str, ...]:
        raise RuntimeError(
            "the authenticated compatibility handoffs do not bind the hierarchy's "
            "accumulated-spent catalogs; load_production_stage1_hierarchy_handoff "
            "and use its canonical catalog provider"
        )

    def as_dict(self) -> dict[str, Any]:
        hierarchical_discovery_contract_identity = (
            validate_production_stage1_hierarchy_contract_identity(
                self.hierarchical_discovery_contract_identity
            )
        )
        body = {
            "schema_version": STAGE1_HIERARCHY_INPUTS_SCHEMA,
            "bundle_manifest_path": str(self.bundle_manifest_path),
            "bundle_sha256": self.bundle_sha256,
            "request_sha256": self.request_sha256,
            "hierarchical_discovery_contract_identity_sha256": (
                hierarchical_discovery_contract_identity["content_sha256"]
            ),
            "dataset_path": str(self.dataset_path),
            "stage1_config_path": str(self.stage1_config_path),
            "split_registry_path": str(self.split_registry_path),
            "primary_splits_path": str(self.primary_splits_path),
            "legacy_handoff_path": str(self.legacy_handoff_path),
            "legacy_scope_index_path": str(self.legacy_scope_index_path),
            "embedding_cluster_fit_index_path": str(
                self.embedding_cluster_fit_index_path
            ),
            "tfidf_handoff_path": str(self.tfidf_handoff_path),
            "neural_query_artifact_index_path": str(self.neural_query_artifact_index_path),
            "exact_inner_evidence_index_path": str(self.exact_inner_evidence_index_path),
            "embedding_cache_dir": str(self.embedding_cache_dir),
            "neural_query_full_outer_artifacts": [
                {"outer_fold": fold, "path": str(path), "sha256": digest}
                for fold, path, digest in self.neural_query_full_outer_artifacts
            ],
            "compatibility_cli_arguments": list(self.compatibility_cli_arguments()),
            "compatibility_cli_arguments_executable_as_production_handoff": False,
            "canonical_accumulated_spent_catalog_provider_required": True,
            "production_hierarchy_ready": False,
            "manual_digest_approval_required": False,
        }
        return {**body, "content_sha256": _sha256_json(body)}


def load_authenticated_stage1_bundle_for_hierarchy(
    manifest_path: Path | str,
) -> AuthenticatedStage1HierarchyInputs:
    """Authenticate a complete all-ten bundle and return hierarchy input paths."""

    requested = _absolute_without_resolution(Path(manifest_path))
    root_capability = _BundleRootCapability(requested.parent)
    try:
        manifest_snapshot = root_capability.snapshot(
            requested.name,
            label="Stage 1 bundle manifest",
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Stage 1 bundle manifest must not be a symlink and must be a regular file"
        ) from exc
    requested = manifest_snapshot.path
    manifest = _load_json_snapshot(manifest_snapshot, label="Stage 1 bundle manifest")
    body = dict(manifest)
    bundle_sha256 = str(body.pop("bundle_sha256", ""))
    if (
        manifest.get("schema_version") != STAGE1_BUNDLE_MANIFEST_SCHEMA
        or manifest.get("manual_digest_approval_required") is not False
        or _HEX_SHA256.fullmatch(
            str(manifest.get("hierarchical_discovery_contract_identity_sha256") or "")
        )
        is None
        or _HEX_SHA256.fullmatch(bundle_sha256) is None
        or _sha256_json(body) != bundle_sha256
    ):
        raise ValueError("Stage 1 bundle manifest identity is invalid")
    request_sha256 = str(manifest.get("request_sha256") or "")
    if _HEX_SHA256.fullmatch(request_sha256) is None:
        raise ValueError("Stage 1 bundle request SHA-256 is invalid")

    required_files = (
        "immutable_build_request",
        "stage1_config",
        "split_registry",
        "primary_splits",
        "row_registry",
        "legacy_handoff",
        "embedding_cluster_fit_index",
        "tfidf_handoff",
        "neural_query_artifact_index",
        "exact_inner_evidence_index",
    )
    registered_snapshots = {
        key: _registered_file_snapshot(root_capability, manifest.get(key), label=key)
        for key in required_files
    }
    paths = {key: snapshot.path for key, snapshot in registered_snapshots.items()}
    registered_json_values: dict[str, Mapping[str, Any]] = {}
    request = _load_json_snapshot(
        registered_snapshots["immutable_build_request"],
        label="immutable build request",
    )
    registered_json_values["immutable_build_request"] = request
    request_body = dict(request)
    declared_request_sha256 = str(request_body.pop("request_sha256", ""))
    if (
        request.get("schema_version") != STAGE1_BUNDLE_REQUEST_SCHEMA
        or declared_request_sha256 != request_sha256
        or _sha256_json(request_body) != request_sha256
    ):
        raise ValueError("immutable Stage 1 build request identity is invalid")
    hierarchical_discovery_contract_identity = (
        validate_production_stage1_hierarchy_request_bindings(request)
    )
    if (
        manifest.get("hierarchical_discovery_contract_identity_sha256")
        != hierarchical_discovery_contract_identity["content_sha256"]
    ):
        raise ValueError(
            "Stage 1 root manifest changed its hierarchical discovery contract binding"
        )
    effective_config = request.get("effective_stage1_config")
    dataset_request = request.get("dataset")
    htr_model_request = request.get("htr_model")
    htr_audit = request.get("htr_input_nontruncation_audit")
    cluster_audit = request.get("embedding_cluster_feasibility_audit")
    security = request.get("security")
    if (
        not isinstance(effective_config, Mapping)
        or not isinstance(dataset_request, Mapping)
        or not isinstance(htr_model_request, Mapping)
        or htr_model_request.get("sentence_encoder_unfrozen") is not True
        or _HEX_SHA256.fullmatch(str(htr_model_request.get("tree_sha256") or "")) is None
        or not isinstance(htr_audit, Mapping)
        or not isinstance(security, Mapping)
        or security.get("htr_source_word_truncation_allowed") is not False
        or security.get("htr_tokenizer_truncation_allowed") is not False
    ):
        raise ValueError("Stage 1 request lacks its HTR no-truncation contract")
    validate_htr_input_nontruncation_audit(
        htr_audit,
        config=effective_config,
        expected_rows=int(dataset_request.get("row_count", -1)),
        expected_htr_model_tree_sha256=str(htr_model_request["tree_sha256"]),
    )
    if not isinstance(cluster_audit, Mapping):
        raise ValueError("Stage 1 request lacks its clustered-embedding feasibility audit")
    exact_status = request.get("exact_inner_contract") or {}
    if (
        exact_status.get("contract_module_available") is not True
        or exact_status.get("registry_matches_contract") is not True
        or (
            (exact_status.get("family_adapter_gate") or {}).get("candidate_bundle_build_ready")
            is not True
            and (exact_status.get("family_adapter_gate") or {}).get("production_execution_ready")
            is not True
        )
    ):
        raise ValueError("Stage 1 request was not built with production-ready exact-inner adapters")
    behavior = request.get("behavior_identity")
    if not isinstance(behavior, Mapping):
        raise ValueError("Stage 1 request has no complete behavior identity")
    behavior_body = dict(behavior)
    behavior_sha = behavior_body.pop("content_sha256", None)
    if (
        _HEX_SHA256.fullmatch(str(behavior_sha or "")) is None
        or _sha256_json(behavior_body) != behavior_sha
    ):
        raise ValueError("Stage 1 behavior identity content hash is invalid")
    if _source_identity() != behavior:
        raise ValueError("current Stage 1 behavior dependencies differ from the sealed request")

    components = manifest.get("components")
    if not isinstance(components, Mapping) or set(components) != {
        "legacy_all_source",
        "tfidf",
        "neural_query",
    }:
        raise ValueError("Stage 1 bundle component set is incomplete")
    component_roots: dict[str, tuple[str, ...]] = {}
    component_manifests: dict[str, Mapping[str, Any]] = {}
    for name, registration in components.items():
        component_root, component_manifest = _validate_component(
            root_capability,
            request_sha256=request_sha256,
            name=str(name),
            registration=registration,
        )
        component_roots[str(name)] = component_root
        component_manifests[str(name)] = component_manifest

    wrapper_registry = _load_json_snapshot(
        registered_snapshots["split_registry"],
        label="split registry",
    )
    registered_json_values["split_registry"] = wrapper_registry
    if _sha256_json(wrapper_registry) != request.get("split_registry_content_sha256"):
        raise ValueError("Stage 1 split registry does not match the immutable request")
    contract_registry = _contract_registry(wrapper_registry)
    if (
        exact_status.get("contract_registry_content_sha256") != contract_registry.content_sha256
        or exact_status.get("contract_registry") != contract_registry.as_dict()
    ):
        raise ValueError("immutable request changed its exact-inner registry")

    legacy_component_root = Path(*component_roots["legacy_all_source"])
    legacy_handoff_relative = Path(
        *_safe_relative_parts(
            manifest["legacy_handoff"].get("relative_path"),
            label="legacy handoff",
        )
    )
    try:
        legacy_handoff_relative.relative_to(legacy_component_root)
    except ValueError as exc:
        raise ValueError("legacy handoff is outside its authenticated component") from exc
    legacy_scope_component_name = "exact_scope_index.json"
    legacy_inventory = component_manifests["legacy_all_source"].get("files")
    legacy_scope_registration = next(
        (
            row
            for row in legacy_inventory
            if isinstance(row, Mapping) and row.get("relative_path") == legacy_scope_component_name
        ),
        None,
    )
    if legacy_scope_registration is None:
        raise ValueError("legacy component has no registered exact-scope index")
    legacy_scope_root_registration = {
        **legacy_scope_registration,
        "relative_path": (legacy_component_root / legacy_scope_component_name).as_posix(),
    }
    legacy_scope_snapshot = _registered_file_snapshot(
        root_capability,
        legacy_scope_root_registration,
        label="legacy exact-scope index",
    )
    legacy_scope_index_path = legacy_scope_snapshot.path
    registered_snapshots["legacy_scope_index"] = legacy_scope_snapshot
    legacy_scope_index = _load_json_snapshot(
        legacy_scope_snapshot,
        label="legacy exact-scope index",
    )
    registered_json_values["legacy_scope_index"] = legacy_scope_index
    _validate_legacy_scope_lineage_snapshots(
        handoff_snapshot=registered_snapshots["legacy_handoff"],
        scope_index=legacy_scope_index,
        registry=wrapper_registry,
        registry_content_sha256=str(request.get("split_registry_content_sha256") or ""),
        bundle_root=root_capability,
        legacy_component_root=legacy_component_root,
    )
    cluster_fit_index = _load_json_snapshot(
        registered_snapshots["embedding_cluster_fit_index"],
        label="embedding cluster-fit index",
    )
    registered_json_values["embedding_cluster_fit_index"] = cluster_fit_index

    exact_index = _load_json_snapshot(
        registered_snapshots["exact_inner_evidence_index"],
        label="exact-inner evidence index",
    )
    registered_json_values["exact_inner_evidence_index"] = exact_index
    index_body = dict(exact_index)
    index_sha = index_body.pop("content_sha256", None)
    exact_index_schema = exact_index.get("schema_version")
    if (
        exact_index_schema
        not in {STAGE1_EXACT_INNER_INDEX_SCHEMA, STAGE1_EXACT_INNER_ROOT_INDEX_SCHEMA}
        or _HEX_SHA256.fullmatch(str(index_sha or "")) is None
        or _sha256_json(index_body) != index_sha
        or exact_index.get("split_registry_content_sha256")
        != request.get("split_registry_content_sha256")
        or exact_index.get("contract_registry_content_sha256") != contract_registry.content_sha256
        or exact_index.get("contract_registry") != contract_registry.as_dict()
    ):
        raise ValueError("exact-inner evidence index identity is invalid")
    producer_hashes = None
    full_outer_hashes = None
    if exact_index_schema == STAGE1_EXACT_INNER_INDEX_SCHEMA:
        producer_hashes = _sha_registry(
            exact_index.get("producer_identity_sha256_by_family"),
            label="producer identity registry",
        )
        full_outer_hashes = _sha_registry(
            exact_index.get("full_outer_payload_sha256_by_family"),
            label="full-outer payload registry",
        )
    elif (
        tuple(exact_index.get("architecture_order") or ()) != ACTIVE_STAGE1_CONCEPT_FAMILIES
        or exact_index.get("scope_identity_registries_are_local") is not True
    ):
        raise ValueError("exact-inner v2 index lacks scope-local identity registries")
    scopes = exact_index.get("scopes")
    if not isinstance(scopes, list):
        raise ValueError("exact-inner evidence index has no scopes")
    by_scope = {
        (int(row.get("outer_fold", 0)), int(row.get("inner_fold", 0))): row
        for row in scopes
        if isinstance(row, Mapping)
    }
    expected_scopes = {
        (outer.outer_fold, inner.inner_fold)
        for outer in contract_registry.outer_splits
        for inner in outer.inner_splits
    }
    if set(by_scope) != expected_scopes or len(by_scope) != len(scopes):
        raise ValueError("exact-inner evidence index scope coverage is incomplete")
    for scope_key in sorted(expected_scopes):
        registration = by_scope[scope_key]
        scope_producer_hashes = producer_hashes
        scope_full_outer_hashes = full_outer_hashes
        if exact_index_schema == STAGE1_EXACT_INNER_ROOT_INDEX_SCHEMA:
            scope_producer_hashes = _sha_registry(
                registration.get("producer_identity_sha256_by_family"),
                label=f"producer identity registry {scope_key[0]}/{scope_key[1]}",
            )
            scope_full_outer_hashes = _sha_registry(
                registration.get("full_outer_payload_sha256_by_family"),
                label=f"full-outer payload registry {scope_key[0]}/{scope_key[1]}",
            )
            _catalog_path, catalog_raw, _catalog_snapshot = _registered_json(
                root_capability,
                registration.get("catalog"),
                label=f"exact-inner catalog {scope_key[0]}/{scope_key[1]}",
            )
            catalog_identity = {
                key: copy.deepcopy(catalog_raw.get(key))
                for key in (
                    "schema_version",
                    "outer_fold",
                    "scope",
                    "inner_fold",
                    "split_fingerprint",
                    "atoms",
                    "non_grounding_numerical_summaries",
                )
            }
            if (
                int(catalog_raw.get("outer_fold", 0)) != scope_key[0]
                or int(catalog_raw.get("inner_fold", 0)) != scope_key[1]
                or catalog_raw.get("scope") != "inner_train"
                or catalog_raw.get("catalog_sha256") != registration.get("catalog_sha256")
                or _sha256_json(catalog_identity) != catalog_raw.get("catalog_sha256")
            ):
                raise ValueError("exact-inner catalog registration changed its identity")
        assert scope_producer_hashes is not None and scope_full_outer_hashes is not None
        _artifact_path, bundle, _artifact_snapshot = _registered_json(
            root_capability,
            registration,
            label=f"exact-inner evidence {scope_key[0]}/{scope_key[1]}",
        )
        if (
            int(bundle.get("outer_fold", 0)) != scope_key[0]
            or int(bundle.get("inner_fold", 0)) != scope_key[1]
            or bundle.get("data_projection_sha256") != registration.get("data_projection_sha256")
        ):
            raise ValueError("exact-inner evidence registration changed its scope or projection")
        validate_exact_inner_stage1_evidence_bundle(
            bundle,
            registry=contract_registry,
            expected_data_projection_sha256=str(registration.get("data_projection_sha256") or ""),
            expected_producer_identity_sha256_by_family=scope_producer_hashes,
            full_outer_payload_sha256_by_family=scope_full_outer_hashes,
        )

    if exact_index_schema == STAGE1_EXACT_INNER_ROOT_INDEX_SCHEMA:
        cumulative_snapshot = _registered_file_snapshot(
            root_capability,
            manifest.get("cumulative_all_ten_root_index"),
            label="cumulative all-ten root index",
        )
        registered_snapshots["cumulative_all_ten_root_index"] = cumulative_snapshot
        cumulative_index = _load_json_snapshot(
            cumulative_snapshot,
            label="cumulative all-ten root index",
        )
        registered_json_values["cumulative_all_ten_root_index"] = cumulative_index
        cumulative_body = dict(cumulative_index)
        cumulative_content_sha256 = cumulative_body.pop("content_sha256", None)
        cumulative_scopes = cumulative_index.get("scopes")
        exact_registration = cumulative_index.get("exact_inner_evidence_index")
        expected_cumulative_index_keys = {
            "schema_version",
            "request_sha256",
            "split_registry_content_sha256",
            "schedule_sha256",
            "architecture_order",
            "exact_inner_evidence_index",
            "scopes",
            "manual_digest_approval_required",
        }
        if (
            set(cumulative_body) != expected_cumulative_index_keys
            or cumulative_index.get("schema_version") != STAGE1_CUMULATIVE_ALL_TEN_ROOT_INDEX_SCHEMA
            or cumulative_index.get("request_sha256") != request_sha256
            or cumulative_index.get("split_registry_content_sha256")
            != request.get("split_registry_content_sha256")
            or cumulative_index.get("schedule_sha256")
            != (request.get("hierarchy_spent_evidence_contract") or {}).get("schedule_sha256")
            or tuple(cumulative_index.get("architecture_order") or ())
            != ACTIVE_STAGE1_CONCEPT_FAMILIES
            or cumulative_index.get("manual_digest_approval_required") is not False
            or _HEX_SHA256.fullmatch(str(cumulative_content_sha256 or "")) is None
            or _sha256_json(cumulative_body) != cumulative_content_sha256
            or not isinstance(cumulative_scopes, list)
            or not isinstance(exact_registration, Mapping)
            or exact_registration.get("relative_path")
            != manifest["exact_inner_evidence_index"].get("relative_path")
            or exact_registration.get("size") != manifest["exact_inner_evidence_index"].get("size")
            or exact_registration.get("sha256")
            != registered_snapshots["exact_inner_evidence_index"].sha256
        ):
            raise ValueError("cumulative all-ten root index identity is invalid")
        from .production_stage1_hierarchy_handoff import (
            CanonicalHierarchySpentSchedule,
        )

        hierarchy_contract = request.get("hierarchy_spent_evidence_contract") or {}
        schedule = CanonicalHierarchySpentSchedule.build(
            registry=contract_registry,
            review_rounds=int(hierarchy_contract.get("review_rounds", 0)),
        )
        if schedule.schedule_sha256 != cumulative_index.get("schedule_sha256"):
            raise ValueError("cumulative all-ten root index changed its canonical schedule")
        hierarchy_snapshot = _registered_file_snapshot(
            root_capability,
            manifest.get("hierarchy_spent_evidence_index"),
            label="hierarchy spent evidence index",
        )
        hierarchy_index = _load_json_snapshot(
            hierarchy_snapshot,
            label="hierarchy spent evidence index",
        )
        hierarchy_body = dict(hierarchy_index)
        hierarchy_content_sha256 = hierarchy_body.pop("content_sha256", None)
        hierarchy_scopes = hierarchy_index.get("scopes")
        if (
            _HEX_SHA256.fullmatch(str(hierarchy_content_sha256 or "")) is None
            or _sha256_json(hierarchy_body) != hierarchy_content_sha256
            or hierarchy_index.get("request_sha256") != request_sha256
            or hierarchy_index.get("schedule_sha256") != schedule.schedule_sha256
            or not isinstance(hierarchy_scopes, list)
        ):
            raise ValueError("hierarchy spent index is invalid at the cumulative root boundary")
        hierarchy_by_scope = {
            (int(row.get("outer_fold", 0)), int(row.get("context_epoch", -1))): row
            for row in hierarchy_scopes
            if isinstance(row, Mapping)
        }
        expected_schedule_by_scope = {
            (scope.outer_fold, scope.context_epoch): scope for scope in schedule.scopes
        }
        if set(hierarchy_by_scope) != set(expected_schedule_by_scope) or len(
            hierarchy_by_scope
        ) != len(hierarchy_scopes):
            raise ValueError("hierarchy spent index differs from the canonical schedule")
        seen_cumulative_scopes: set[tuple[int, int]] = set()
        expected_scope_row_keys = {
            "scope_id",
            "outer_fold",
            "context_epoch",
            "provider_inner_fold",
            "split_fingerprint",
            "typed_bundle",
            "typed_bundle_sha256",
            "catalog",
            "catalog_sha256",
            "proof_bundle",
        }
        for scope_row in cumulative_scopes:
            if not isinstance(scope_row, Mapping) or set(scope_row) != expected_scope_row_keys:
                raise ValueError("cumulative all-ten root scope is not a closed schema")
            outer_fold = int(scope_row.get("outer_fold", 0))
            context_epoch = int(scope_row.get("context_epoch", -1))
            scope_key = (outer_fold, context_epoch)
            canonical_scope = expected_schedule_by_scope.get(scope_key)
            hierarchy_row = hierarchy_by_scope.get(scope_key)
            if (
                scope_key in seen_cumulative_scopes
                or canonical_scope is None
                or not isinstance(hierarchy_row, Mapping)
                or scope_row.get("scope_id") != canonical_scope.scope_id
                or int(scope_row.get("provider_inner_fold", 0))
                != canonical_scope.provider_inner_fold
                or scope_row.get("split_fingerprint") != canonical_scope.split_fingerprint
                or hierarchy_row.get("scope_id") != canonical_scope.scope_id
                or tuple(map(int, hierarchy_row.get("spent_row_ids") or ()))
                != canonical_scope.spent_row_ids
                or tuple(map(int, hierarchy_row.get("sealed_row_ids") or ()))
                != canonical_scope.sealed_row_ids
                or hierarchy_row.get("split_fingerprint") != canonical_scope.split_fingerprint
                or hierarchy_row.get("catalog") != scope_row.get("catalog")
                or hierarchy_row.get("proof_bundle") != scope_row.get("proof_bundle")
                or hierarchy_row.get("catalog_sha256") != scope_row.get("catalog_sha256")
            ):
                raise ValueError("cumulative all-ten scope differs from the canonical root graph")
            seen_cumulative_scopes.add(scope_key)
            _bundle_path, cumulative_bundle, bundle_snapshot = _registered_json(
                root_capability,
                scope_row.get("typed_bundle"),
                label=f"cumulative all-ten typed bundle {outer_fold}/{context_epoch}",
            )
            producer_registry = _sha_registry(
                cumulative_bundle.get("producer_identity_sha256_by_family"),
                label=f"cumulative producer registry {outer_fold}/{context_epoch}",
            )
            if (
                cumulative_bundle.get("bundle_sha256") != scope_row.get("typed_bundle_sha256")
                or bundle_snapshot.sha256 != (scope_row.get("typed_bundle") or {}).get("sha256")
                or cumulative_bundle.get("scope_id") != scope_row.get("scope_id")
                or int(cumulative_bundle.get("provider_inner_fold", 0))
                != int(scope_row.get("provider_inner_fold", 0))
                or cumulative_bundle.get("split_scope_fingerprint")
                != scope_row.get("split_fingerprint")
                or tuple(map(int, cumulative_bundle.get("spent_row_ids") or ()))
                != canonical_scope.spent_row_ids
                or tuple(map(int, cumulative_bundle.get("sealed_row_ids") or ()))
                != canonical_scope.sealed_row_ids
            ):
                raise ValueError("cumulative typed bundle changed its root registration")
            validate_cumulative_spent_stage1_evidence_bundle(
                cumulative_bundle,
                expected_request_sha256=request_sha256,
                expected_schedule_sha256=str(cumulative_index.get("schedule_sha256") or ""),
                expected_scope_id=str(scope_row.get("scope_id") or ""),
                expected_split_scope_fingerprint=str(scope_row.get("split_fingerprint") or ""),
                expected_spent_row_ids=canonical_scope.spent_row_ids,
                expected_sealed_row_ids=canonical_scope.sealed_row_ids,
                expected_data_projection_sha256=str(
                    cumulative_bundle.get("data_projection_sha256") or ""
                ),
                expected_producer_identity_sha256_by_family=producer_registry,
            )
            _catalog_path, catalog_raw, _catalog_snapshot = _registered_json(
                root_capability,
                scope_row.get("catalog"),
                label=f"cumulative catalog {outer_fold}/{context_epoch}",
            )
            _proof_path, proof_raw, _proof_snapshot = _registered_json(
                root_capability,
                scope_row.get("proof_bundle"),
                label=f"cumulative proof bundle {outer_fold}/{context_epoch}",
            )
            if (
                catalog_raw.get("catalog_sha256") != scope_row.get("catalog_sha256")
                or proof_raw.get("catalog_sha256") != scope_row.get("catalog_sha256")
                or proof_raw.get("scope_id") != canonical_scope.scope_id
            ):
                raise ValueError("cumulative catalog/proof registrations differ from root index")
            typed_artifacts = cumulative_bundle.get("family_artifacts")
            proof_rows = proof_raw.get("family_proofs")
            if (
                not isinstance(typed_artifacts, list)
                or not isinstance(proof_rows, list)
                or tuple(
                    str(row.get("family")) for row in typed_artifacts if isinstance(row, Mapping)
                )
                != ACTIVE_STAGE1_CONCEPT_FAMILIES
                or tuple(str(row.get("family")) for row in proof_rows if isinstance(row, Mapping))
                != ACTIVE_STAGE1_CONCEPT_FAMILIES
            ):
                raise ValueError("cumulative typed/proof family rows are incomplete")
            for family, typed_artifact, proof_row in zip(
                ACTIVE_STAGE1_CONCEPT_FAMILIES,
                typed_artifacts,
                proof_rows,
            ):
                _descriptor_path, descriptor, _descriptor_snapshot = _registered_json(
                    root_capability,
                    proof_row.get("model_artifact"),
                    label=(f"cumulative native descriptor {outer_fold}/{context_epoch}/{family}"),
                )
                if (
                    descriptor.get("scope_id") != canonical_scope.scope_id
                    or descriptor.get("family") != family
                    or descriptor.get("typed_family_artifact_sha256")
                    != typed_artifact.get("artifact_sha256")
                    or descriptor.get("producer_identity_sha256")
                    != typed_artifact.get("producer_identity_sha256")
                ):
                    raise ValueError(
                        "cumulative native descriptor differs from its typed family artifact"
                    )
        if set(seen_cumulative_scopes) != set(expected_schedule_by_scope):
            raise ValueError("cumulative all-ten root index has incomplete scope coverage")

    coverage = manifest.get("coverage")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("all_ten_families_nonzero_in_every_scope") is not True
        or tuple(coverage.get("required_families") or ()) != ACTIVE_STAGE1_CONCEPT_FAMILIES
    ):
        raise ValueError("Stage 1 bundle lacks all-ten architecture coverage")

    dataset_registration = request.get("dataset")
    if not isinstance(dataset_registration, Mapping):
        raise ValueError("immutable request has no dataset registration")
    dataset_sha, _dataset_stat, dataset_path = _read_stable_file_sha256_no_symlinks(
        Path(str(dataset_registration.get("path") or "")),
        label="registered cohort",
    )
    if dataset_sha != dataset_registration.get("sha256"):
        raise ValueError("current cohort bytes differ from the Stage 1 bundle")
    embedding_registration = request.get("embedding_cache")
    if not isinstance(embedding_registration, Mapping):
        raise ValueError("immutable request has no embedding-cache registration")
    embedding_cache_capability = _BundleRootCapability(
        Path(str(embedding_registration.get("path") or ""))
    )
    embedding_cache_dir = embedding_cache_capability.path
    expected_cache_identity = embedding_registration.get("identity")
    if not isinstance(expected_cache_identity, Mapping):
        raise ValueError("immutable request has no embedding-cache byte identity")
    required_cache_identity = {
        "provider": "spent_only_frozen_chunk_embedding_cache_v2",
        "cache_snapshot_authentication": "streamed_private_fd_sha256_v1",
        "chunk_text_storage": "private_fd_pread_lazy_row_decode_v1",
        "embeddings_path_backed": False,
        "private_snapshot_embedding_mmap": True,
        "future_row_text_decoded": False,
        "novel_text_encoding_allowed": False,
    }
    if any(
        expected_cache_identity.get(key) != value for key, value in required_cache_identity.items()
    ):
        raise ValueError("registered embedding cache identity policy is invalid")
    cache_hash_fields = {
        "metadata.json": "metadata_sha256",
        "chunk_embeddings.npy": "embeddings_sha256",
        "offsets.npy": "offsets_sha256",
        "chunk_texts.jsonl": "chunk_texts_sha256",
    }
    for filename, identity_key in cache_hash_fields.items():
        digest, _file_stat, _path = embedding_cache_capability.sha256(
            filename,
            label=f"embedding cache {filename}",
        )
        if digest != expected_cache_identity.get(identity_key):
            raise ValueError("registered embedding cache failed authentication")
    metadata_snapshot = embedding_cache_capability.snapshot(
        "metadata.json",
        label="embedding cache metadata",
    )
    metadata = _load_json_snapshot(metadata_snapshot, label="embedding cache metadata")
    if int(metadata.get("num_samples", -1)) != int(expected_cache_identity.get("row_count", -2)):
        raise ValueError("registered embedding cache failed authentication")
    adapter_gate = exact_status.get("family_adapter_gate") or {}
    if adapter_gate.get("candidate_bundle_build_ready") is True:
        expected_build_identity = embedding_registration.get("production_cache_build_identity")
        effective_config = request.get("effective_stage1_config")
        try:
            embedding_config = effective_config["architecture"]["multi_model_forest"][
                "embedding_contrast"
            ]
            text_column = effective_config["text_column"]
            sentence_model_name = embedding_config["model_name"]
            chunk_configuration = {
                "chunk_size_words": embedding_config["chunk_size_words"],
                "chunk_overlap_words": embedding_config["chunk_overlap_words"],
                "max_chunks": embedding_config["max_chunks"],
                "chunk_selection": embedding_config["chunk_selection"],
                "normalize_embeddings": embedding_config["normalize_embeddings"],
                "max_seq_length": embedding_config["max_seq_length"],
            }
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "candidate Stage 1 request lacks its closed embedding-cache configuration"
            ) from exc
        if not isinstance(expected_build_identity, Mapping):
            raise ValueError(
                "candidate Stage 1 request lacks a production embedding-cache build identity"
            )
        validated_build_identity = validate_published_production_embedding_cache(
            cache_dir=embedding_cache_dir,
            dataset_path=dataset_path,
            text_column=text_column,
            sentence_model_name=sentence_model_name,
            chunk_configuration=chunk_configuration,
        )
        if (
            validated_build_identity != expected_build_identity
            or validated_build_identity.get("provider_identity") != expected_cache_identity
        ):
            raise ValueError(
                "production embedding-cache build identity differs from current inputs"
            )
    try:
        validate_embedding_cluster_feasibility_audit(
            cluster_audit,
            config=effective_config,
            registry=wrapper_registry,
            registry_content_sha256=str(request.get("split_registry_content_sha256") or ""),
            embedding_cache_identity=expected_cache_identity,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Stage 1 clustered-embedding feasibility audit is missing or invalid"
        ) from exc
    _validate_embedding_cluster_fit_index_snapshot(
        index=cluster_fit_index,
        index_snapshot=registered_snapshots["embedding_cluster_fit_index"],
        legacy_scope_index=legacy_scope_index,
        cluster_audit=cluster_audit,
        request_sha256=request_sha256,
        registry_content_sha256=str(
            request.get("split_registry_content_sha256") or ""
        ),
        bundle_root=root_capability,
        legacy_component_root=legacy_component_root,
    )

    query_index = _load_json_snapshot(
        registered_snapshots["neural_query_artifact_index"],
        label="neural-query artifact index",
    )
    registered_json_values["neural_query_artifact_index"] = query_index
    if (
        query_index.get("schema_version") != STAGE1_SCOPE_INDEX_SCHEMA
        or query_index.get("split_registry_content_sha256")
        != request.get("split_registry_content_sha256")
        or not isinstance(query_index.get("scopes"), list)
    ):
        raise ValueError("neural-query artifact index is not registry-bound")
    query_component_root = Path(*component_roots["neural_query"])
    query_artifacts: list[tuple[int, Path, str]] = []
    seen_query_folds: set[int] = set()
    for row in query_index["scopes"]:
        if not isinstance(row, Mapping) or row.get("inner_fold") is not None:
            continue
        fold = int(row.get("outer_fold", 0))
        query_relative = (
            query_component_root
            / Path(
                *_safe_relative_parts(
                    row.get("path"),
                    label="neural-query artifact",
                )
            )
        ).as_posix()
        digest = str(row.get("sha256") or "")
        artifact_snapshot = root_capability.snapshot(
            query_relative,
            label="neural-query artifact",
        )
        if fold < 1 or fold in seen_query_folds or artifact_snapshot.sha256 != digest:
            raise ValueError("full-outer neural-query artifact registration is invalid")
        seen_query_folds.add(fold)
        query_artifacts.append((fold, artifact_snapshot.path, digest))
    expected_outer = {outer.outer_fold for outer in contract_registry.outer_splits}
    if seen_query_folds != expected_outer:
        raise ValueError("neural-query artifacts do not cover every outer fold")

    return AuthenticatedStage1HierarchyInputs(
        bundle_manifest_path=requested,
        bundle_sha256=bundle_sha256,
        request_sha256=request_sha256,
        hierarchical_discovery_contract_identity=(hierarchical_discovery_contract_identity),
        dataset_path=dataset_path,
        stage1_config_path=paths["stage1_config"],
        split_registry_path=paths["split_registry"],
        primary_splits_path=paths["primary_splits"],
        legacy_handoff_path=paths["legacy_handoff"],
        legacy_scope_index_path=legacy_scope_index_path,
        embedding_cluster_fit_index_path=paths["embedding_cluster_fit_index"],
        tfidf_handoff_path=paths["tfidf_handoff"],
        neural_query_artifact_index_path=paths["neural_query_artifact_index"],
        exact_inner_evidence_index_path=paths["exact_inner_evidence_index"],
        embedding_cache_dir=embedding_cache_dir,
        neural_query_full_outer_artifacts=tuple(sorted(query_artifacts)),
        _bundle_root_capability=root_capability,
        _bundle_manifest_snapshot=manifest_snapshot,
        _bundle_manifest_json=copy.deepcopy(dict(manifest)),
        _registered_snapshots=dict(registered_snapshots),
        _registered_json_values={
            key: copy.deepcopy(dict(value)) for key, value in registered_json_values.items()
        },
        _component_roots=dict(component_roots),
        _component_manifests={
            key: copy.deepcopy(dict(value)) for key, value in component_manifests.items()
        },
        _embedding_cache_capability=embedding_cache_capability,
    )


__all__ = [
    "AuthenticatedStage1HierarchyInputs",
    "STAGE1_EXACT_INNER_INDEX_SCHEMA",
    "STAGE1_HIERARCHY_INPUTS_SCHEMA",
    "load_authenticated_stage1_bundle_for_hierarchy",
]
