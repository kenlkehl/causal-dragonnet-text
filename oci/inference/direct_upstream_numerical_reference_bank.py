"""Reference-only final numerical banks over role-neutral Stage 1 artifacts.

The legacy :mod:`direct_upstream_numerical_manifest` contract authenticates
four materialized, combined matrices.  The role-neutral Stage 1 path already
persists every native numerical payload once, so copying those values into
four more ``.npy`` files would be both expensive and scientifically
unnecessary.  This module is the reference-layout extension of that contract:

* exact-inner primary transforms are the sole outer-train OOF source;
* the full-outer primary transform is the sole outer-held-out source;
* every numerical payload remains in its authenticated producer tree;
* the published artifact contains JSON references and indexes only; and
* estimator-facing matrices are assembled in memory, on demand.

Variable-coordinate lexical banks are not truncated.  They use a complete
permutation-invariant signed-order representation whose width is derived from
the authenticated source artifacts.  A presence coordinate accompanies every
rank so padding cannot be confused with an observed zero.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
    BOW_NUISANCE,
    BOW_R_LOSS,
    DIRECT_UPSTREAM_NUMERICAL_CHANNEL,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HETEROGENEITY_AXIS,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    OUTCOME_AXIS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
    TREATMENT_AXIS,
)
from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .production_neural_query_binary_layout import validate_npy_array_set
from .production_role_neutral_stage2_handoff import (
    validate_authenticated_prepared_projection_binding,
    validate_authenticated_role_neutral_stage2_runtime_binding,
)
from .production_stage1_role_neutral_execution import (
    ROLE_NEUTRAL_COMPONENT_DIRECTORY,
    ROLE_NEUTRAL_EXECUTION_MANIFEST,
    validate_role_neutral_stage1_execution,
)
from .production_stage1_scope_scheduler import Stage1ScopePlan, Stage1ScopeSpec


DIRECT_NUMERICAL_REFERENCE_MANIFEST_SCHEMA = (
    "direct_upstream_numerical_manifest_v2_role_neutral_references"
)
DIRECT_NUMERICAL_REFERENCE_LOCATOR_SCHEMA = (
    "direct_upstream_numerical_reference_locator_v1"
)
DIRECT_NUMERICAL_REFERENCE_BANK_ID = (
    "authenticated_role_neutral_direct_numerical_reference_bank_v1"
)
DIRECT_NUMERICAL_REFERENCE_MANIFEST = "direct_upstream_numerical_manifest.json"
DIRECT_NUMERICAL_REFERENCE_LOCATOR = "locator_attestation.json"

OUTER_TRAIN_OOF_SCOPE = "outer_train_exact_inner_oof"
OUTER_HELDOUT_SCOPE = "outer_heldout_full_outer_transform"
REVIEW_GATE_SCOPE = "cumulative_review_gate_transform"
_ROW_SCOPES = frozenset(
    {OUTER_TRAIN_OOF_SCOPE, OUTER_HELDOUT_SCOPE, REVIEW_GATE_SCOPE}
)

CALIBRATED_SOURCE_BANK = "calibrated_source"
RAW_FEATURE_BANK = "raw_feature"
_BANK_KINDS = frozenset({CALIBRATED_SOURCE_BANK, RAW_FEATURE_BANK})

EXACT_NAMED_ALIGNMENT = "exact_named_coordinate_v1"
COMPLETE_SIGNED_ORDER_ALIGNMENT = "complete_signed_order_with_presence_v1"
COMPLETE_ABSOLUTE_ORDER_ALIGNMENT = "complete_absolute_order_with_presence_v1"
_ALIGNMENTS = frozenset(
    {
        EXACT_NAMED_ALIGNMENT,
        COMPLETE_SIGNED_ORDER_ALIGNMENT,
        COMPLETE_ABSOLUTE_ORDER_ALIGNMENT,
    }
)

DENSE_NPY_PAYLOAD = "dense_npy_reference_v1"
SPARSE_CSR_PAYLOAD = "sparse_csr_npy_references_v1"
_PAYLOAD_KINDS = frozenset({DENSE_NPY_PAYLOAD, SPARSE_CSR_PAYLOAD})

_COMPONENT_FAMILIES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
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
)
_FAMILY_COMPONENT = {
    family: component
    for component, families in _COMPONENT_FAMILIES.items()
    for family in families
}
_ROLES = frozenset(
    {
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    }
)
_AXES = frozenset({TREATMENT_AXIS, OUTCOME_AXIS, HETEROGENEITY_AXIS})
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


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _strict_object(
    pairs: Sequence[tuple[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{label} contains duplicate key {key!r}")
        result[key] = value
    return result


def _stable_private_file(path: Path, *, label: str) -> tuple[bytes, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one regular file")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise ValueError(f"{label} must be private regular data")
    payload = path.read_bytes()
    after = os.lstat(path)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        int(getattr(before, field)) != int(getattr(after, field))
        for field in identity_fields
    ) or len(payload) != int(after.st_size):
        raise RuntimeError(f"{label} changed while it was read")
    return payload, hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    payload, digest = _stable_private_file(path, label=label)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=lambda pairs: _strict_object(
                pairs,
                label=label,
            ),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    if "content_sha256" in value and value["content_sha256"] != _sha256_json(body):
        raise ValueError(f"{label} content identity changed")
    return value, digest


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_root(value: Path | str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{label} must be canonical and absolute")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_dir():
        raise ValueError(f"{label} must be one real directory")
    return resolved


def _tree_stat_inventory(root: Path) -> tuple[tuple[Any, ...], ...]:
    """Guard a fully authenticated in-process handle without rehashing bytes."""

    rows: list[tuple[Any, ...]] = []
    for path in (
        root,
        *sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()),
    ):
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("authenticated numerical source tree contains a symlink")
        rows.append(
            (
                "." if path == root else path.relative_to(root).as_posix(),
                int(metadata.st_dev),
                int(metadata.st_ino),
                int(metadata.st_mode),
                int(metadata.st_nlink),
                int(metadata.st_size),
                int(metadata.st_mtime_ns),
                int(metadata.st_ctime_ns),
            )
        )
    return tuple(rows)


def _relative_path(root: Path, raw: Any, *, label: str) -> Path:
    relative = PurePosixPath(str(raw))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} has a noncanonical relative path")
    path = root.joinpath(*relative.parts)
    resolved = path.resolve(strict=True)
    if path.is_symlink() or resolved != path:
        raise ValueError(f"{label} path is invalid")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its component root") from exc
    return path


def _validated_registration(
    component_root: Path,
    value: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], Path]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} registration must be an object")
    registration = copy.deepcopy(dict(value))
    path = _relative_path(
        component_root,
        registration.get("relative_path"),
        label=label,
    )
    payload, digest = _stable_private_file(path, label=label)
    if (
        digest
        != _require_sha256(
            registration.get("sha256"),
            label=f"{label} bytes",
        )
        or len(payload) != registration.get("size_bytes")
    ):
        raise ValueError(f"{label} differs from its source registration")
    return registration, path


class _AuthenticatedNumericalPayloadCache:
    """One-trust-boundary cache of authenticated numerical payloads.

    Manifest construction is the byte-authentication pass.  Each temporary
    mmap is copied into a read-only in-memory array and closed immediately, so
    authenticating many logical contexts does not retain one file descriptor
    per payload.  The arrays are then guarded by their exact file/stat
    identities, and ordinary fold/gate consumers never reopen, rehash, or
    repeat a whole-array finite scan.
    """

    def __init__(self) -> None:
        self._arrays: dict[tuple[Any, ...], np.ndarray] = {}
        self._path_keys: dict[Path, tuple[Any, ...]] = {}
        self._file_stats: dict[Path, tuple[int, ...]] = {}
        self._prepared_blocks: dict[str, np.ndarray | sparse.csr_matrix] = {}
        self._neural_array_sets: dict[
            tuple[Path, str],
            tuple[Mapping[str, Any], Mapping[str, np.ndarray]],
        ] = {}
        self._neural_array_set_keys_by_root: dict[Path, tuple[Path, str]] = {}
        self._byte_authenticated_file_count = 0
        self._externally_authenticated_file_count = 0
        self._unique_payload_file_count = 0
        self._unique_neural_array_set_count = 0
        self._npy_open_count = 0
        self._ordinary_materialization_file_open_count = 0

    @staticmethod
    def _close_memmap(array: np.ndarray) -> None:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()

    @classmethod
    def _detach_memmap(cls, array: np.ndarray) -> np.ndarray:
        try:
            detached = np.array(array, copy=True, order="K", subok=False)
        finally:
            cls._close_memmap(array)
        detached.setflags(write=False)
        return detached

    def validate_neural_array_set_once(
        self,
        root: Path,
        *,
        expected_inventory: Mapping[str, Any] | None,
    ) -> tuple[Mapping[str, Any], Mapping[str, np.ndarray]]:
        inventory_identity = _sha256_json(
            None if expected_inventory is None else dict(expected_inventory)
        )
        key = (root, inventory_identity)
        previous = self._neural_array_set_keys_by_root.get(root)
        if previous is not None and previous != key:
            raise ValueError(
                "neural-query array set was referenced with different inventory"
            )
        cached = self._neural_array_sets.get(key)
        if cached is not None:
            return cached
        descriptor, arrays = validate_npy_array_set(
            root,
            expected_order=("gate_row_ids", "feature_values"),
            expected_inventory=expected_inventory,
        )
        try:
            detached = {
                name: self._detach_memmap(array)
                for name, array in arrays.items()
            }
        finally:
            for array in arrays.values():
                self._close_memmap(array)
        retained = (descriptor, detached)
        self._neural_array_sets[key] = retained
        self._neural_array_set_keys_by_root[root] = key
        self._unique_neural_array_set_count += 1
        return retained

    @staticmethod
    def _stat_identity(path: Path) -> tuple[int, ...]:
        metadata = os.lstat(path)
        if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_nlink) != 1:
            raise ValueError("authenticated numerical payload is no longer private data")
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_mode),
            int(metadata.st_nlink),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
        )

    @staticmethod
    def _registration_key(
        component_root: Path,
        registration: Any,
        *,
        label: str,
    ) -> tuple[tuple[Any, ...], Path]:
        if not isinstance(registration, Mapping):
            raise ValueError(f"{label} registration must be an object")
        path = _relative_path(
            component_root,
            registration.get("relative_path"),
            label=label,
        )
        shape_raw = registration.get("shape")
        if not isinstance(shape_raw, (list, tuple)):
            raise ValueError(f"{label} registration lacks an array shape")
        shape = tuple(int(value) for value in shape_raw)
        dtype = str(registration.get("dtype"))
        sha256 = _require_sha256(
            registration.get("sha256"),
            label=f"{label} bytes",
        )
        size_bytes = registration.get("size_bytes")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, (int, np.integer))
            or int(size_bytes) < 1
            or not dtype
            or not shape
        ):
            raise ValueError(f"{label} registration metadata is malformed")
        return (
            (
                path,
                sha256,
                int(size_bytes),
                dtype,
                shape,
            ),
            path,
        )

    @staticmethod
    def _normalized_registration(
        registration: Mapping[str, Any],
        *,
        shape: tuple[int, ...],
    ) -> dict[str, Any]:
        normalized = {
            "relative_path": str(registration["relative_path"]),
            "sha256": str(registration["sha256"]),
            "size_bytes": int(registration["size_bytes"]),
            "dtype": str(registration["dtype"]),
            "shape": list(shape),
        }
        content = registration.get("content_sha256")
        if content is not None:
            normalized["content_sha256"] = _require_sha256(
                content,
                label="numerical payload content",
            )
        return normalized

    def _register_path(
        self,
        *,
        key: tuple[Any, ...],
        path: Path,
        array: np.ndarray,
        externally_authenticated: bool,
        label: str,
    ) -> None:
        previous = self._path_keys.get(path)
        if previous is not None and previous != key:
            raise ValueError(f"{label} reused one path with different metadata")
        if key in self._arrays:
            return
        if (
            array.dtype.hasobject
            or tuple(array.shape) != key[4]
            or array.dtype.str != key[3]
            or array.ndim not in {1, 2}
            or not np.isfinite(np.asarray(array)).all()
        ):
            raise ValueError(f"{label} array metadata or finite-value contract changed")
        array.setflags(write=False)
        self._arrays[key] = array
        self._path_keys[path] = key
        self._file_stats[path] = self._stat_identity(path)
        self._unique_payload_file_count += 1
        if externally_authenticated:
            self._externally_authenticated_file_count += 1
        else:
            self._byte_authenticated_file_count += 1

    def authenticate_array(
        self,
        component_root: Path,
        registration: Any,
        *,
        label: str,
    ) -> tuple[dict[str, Any], np.ndarray]:
        key, path = self._registration_key(
            component_root,
            registration,
            label=label,
        )
        previous = self._path_keys.get(path)
        if previous is not None and previous != key:
            raise ValueError(f"{label} reused one path with different metadata")
        cached = self._arrays.get(key)
        if cached is not None:
            assert isinstance(registration, Mapping)
            return (
                self._normalized_registration(registration, shape=key[4]),
                cached,
            )
        registered, authenticated_path = _validated_registration(
            component_root,
            registration,
            label=label,
        )
        if authenticated_path != path:
            raise RuntimeError(f"{label} locator changed during authentication")
        try:
            mapped = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{label} is not one safe NPY array") from exc
        array = self._detach_memmap(mapped)
        self._npy_open_count += 1
        self._register_path(
            key=key,
            path=path,
            array=array,
            externally_authenticated=False,
            label=label,
        )
        return (
            self._normalized_registration(registered, shape=key[4]),
            array,
        )

    def adopt_externally_authenticated_array(
        self,
        component_root: Path,
        registration: Mapping[str, Any],
        array: np.ndarray,
        *,
        label: str,
    ) -> None:
        """Retain an mmap already fully checked by a closed layout validator."""

        key, path = self._registration_key(
            component_root,
            registration,
            label=label,
        )
        previous = self._path_keys.get(path)
        if previous is not None and previous != key:
            raise ValueError(f"{label} reused one path with different metadata")
        if key in self._arrays:
            return
        self._register_path(
            key=key,
            path=path,
            array=array,
            externally_authenticated=True,
            label=label,
        )

    def _lookup_array(
        self,
        component_root: Path,
        registration: Mapping[str, Any],
        *,
        label: str,
    ) -> np.ndarray:
        key, path = self._registration_key(
            component_root,
            registration,
            label=label,
        )
        if self._path_keys.get(path) != key or key not in self._arrays:
            raise RuntimeError(
                f"{label} was not retained from the fresh authentication pass"
            )
        return self._arrays[key]

    @staticmethod
    def _block_key(
        projection: Mapping[str, Any],
        block: Mapping[str, Any],
    ) -> str:
        return _sha256_json(
            {
                "source_transform_scope_id": projection[
                    "source_transform_scope_id"
                ],
                "source_component": block["source_component"],
                "payload_kind": block["payload_kind"],
                "files": block["files"],
                "csr_shape": block.get("csr_shape"),
            }
        )

    def prepare_block(
        self,
        *,
        execution_root: Path,
        projection: Mapping[str, Any],
        block: Mapping[str, Any],
    ) -> np.ndarray | sparse.csr_matrix:
        key = self._block_key(projection, block)
        cached = self._prepared_blocks.get(key)
        if cached is not None:
            return cached
        component_root = (
            execution_root
            / ROLE_NEUTRAL_COMPONENT_DIRECTORY
            / str(projection["source_transform_scope_id"])
            / str(block["source_component"])
        )
        kind = str(block["payload_kind"])
        if kind == DENSE_NPY_PAYLOAD:
            if len(block["files"]) != 1:
                raise ValueError("dense numerical block must reference exactly one file")
            array = self._lookup_array(
                component_root,
                block["files"][0],
                label="direct numerical dense payload",
            )
            matrix = np.asarray(array)
            prepared: np.ndarray | sparse.csr_matrix = (
                matrix.reshape(-1, 1) if matrix.ndim == 1 else matrix
            )
        elif kind == SPARSE_CSR_PAYLOAD:
            arrays: dict[str, np.ndarray] = {}
            for registration in block["files"]:
                array = self._lookup_array(
                    component_root,
                    registration,
                    label="direct numerical sparse payload",
                )
                stem = PurePosixPath(str(registration["relative_path"])).stem
                arrays[stem.rsplit("_", 1)[-1]] = array
            if set(arrays) != {"data", "indices", "indptr"}:
                raise ValueError("direct numerical sparse payload coverage changed")
            prepared = sparse.csr_matrix(
                (
                    np.asarray(arrays["data"], dtype=np.float64),
                    np.asarray(arrays["indices"], dtype=np.int64),
                    np.asarray(arrays["indptr"], dtype=np.int64),
                ),
                shape=tuple(int(value) for value in block["csr_shape"]),
            )
            if not prepared.has_sorted_indices or not np.isfinite(prepared.data).all():
                raise ValueError("direct numerical sparse payload changed")
            prepared.data.setflags(write=False)
            prepared.indices.setflags(write=False)
            prepared.indptr.setflags(write=False)
        else:
            raise ValueError("direct numerical block has an unsupported payload kind")
        if prepared.shape != (
            int(block["row_count"]),
            int(block["source_column_count"]),
        ):
            raise ValueError("direct numerical payload shape changed")
        self._prepared_blocks[key] = prepared
        return prepared

    def release_authentication_buffers(self) -> None:
        """Drop arrays no longer needed after every block is prepared."""

        self._arrays.clear()
        self._neural_array_sets.clear()

    def prepared_block(
        self,
        *,
        projection: Mapping[str, Any],
        block: Mapping[str, Any],
    ) -> np.ndarray | sparse.csr_matrix:
        key = self._block_key(projection, block)
        try:
            return self._prepared_blocks[key]
        except KeyError as exc:
            raise RuntimeError(
                "ordinary numerical access attempted to reopen an unaudited payload"
            ) from exc

    def validate_guarded_files(self) -> None:
        for path, expected in self._file_stats.items():
            if self._stat_identity(path) != expected:
                raise ValueError("referenced role-neutral numerical payload changed")

    def audit_counters(self) -> Mapping[str, Any]:
        return {
            "schema_version": "direct_numerical_payload_cache_audit_v1",
            "unique_payload_file_count": self._unique_payload_file_count,
            "unique_prepared_block_count": len(self._prepared_blocks),
            "byte_authenticated_payload_file_count": (
                self._byte_authenticated_file_count
            ),
            "externally_authenticated_payload_file_count": (
                self._externally_authenticated_file_count
            ),
            "unique_neural_query_array_set_count": (
                self._unique_neural_array_set_count
            ),
            "npy_open_count_during_fresh_audit": self._npy_open_count,
            "ordinary_materialization_payload_file_open_count": (
                self._ordinary_materialization_file_open_count
            ),
            "payload_handles_reused_in_process": True,
        }


def _array_reference(
    component_root: Path,
    registration: Any,
    *,
    label: str,
    payload_cache: _AuthenticatedNumericalPayloadCache | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    if payload_cache is not None:
        return payload_cache.authenticate_array(
            component_root,
            registration,
            label=label,
        )
    registered, path = _validated_registration(
        component_root,
        registration,
        label=label,
    )
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is not one safe NPY array") from exc
    shape = tuple(int(value) for value in registered.get("shape") or ())
    if (
        array.dtype.hasobject
        or tuple(array.shape) != shape
        or array.dtype.str != registered.get("dtype")
        or array.ndim not in {1, 2}
        or not np.isfinite(np.asarray(array)).all()
    ):
        raise ValueError(f"{label} array metadata or finite-value contract changed")
    normalized = {
        "relative_path": registered["relative_path"],
        "sha256": registered["sha256"],
        "size_bytes": int(registered["size_bytes"]),
        "dtype": registered["dtype"],
        "shape": list(shape),
    }
    content = registered.get("content_sha256")
    if content is not None:
        normalized["content_sha256"] = _require_sha256(
            content,
            label=f"{label} numerical content",
        )
    return normalized, array


def _rows(values: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be an ordered row-ID sequence")
    result: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError(f"{label} must contain integer row IDs")
        row_id = int(value)
        if row_id < 0:
            raise ValueError(f"{label} cannot contain negative row IDs")
        result.append(row_id)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{label} must be nonempty and unique")
    return tuple(result)


def _scope_rows(plan: Stage1ScopePlan, outer_fold: int) -> tuple[
    Stage1ScopeSpec,
    tuple[Stage1ScopeSpec, ...],
    tuple[Stage1ScopeSpec, ...],
]:
    full = tuple(
        scope
        for scope in plan.scopes
        if scope.outer_fold == int(outer_fold) and scope.scope_kind == "full_outer"
    )
    exact = tuple(
        sorted(
            (
                scope
                for scope in plan.scopes
                if scope.outer_fold == int(outer_fold)
                and scope.scope_kind == "exact_inner"
            ),
            key=lambda scope: int(scope.inner_fold or -1),
        )
    )
    cumulative = tuple(
        sorted(
            (
                scope
                for scope in plan.scopes
                if scope.outer_fold == int(outer_fold)
                and scope.scope_kind == "cumulative_spent"
            ),
            key=lambda scope: int(scope.context_epoch or 0),
        )
    )
    if len(full) != 1 or len(exact) < 2:
        raise ValueError(f"outer fold {outer_fold} lacks full/exact-inner contexts")
    train = full[0].fit_row_ids
    flattened = tuple(row for scope in exact for row in scope.heldout_row_ids)
    if len(flattened) != len(set(flattened)) or set(flattened) != set(train):
        raise ValueError(f"outer fold {outer_fold} exact-inner OOF coverage changed")
    if any(scope.fit_row_ids != tuple(row for row in train if row not in set(scope.heldout_row_ids)) for scope in exact):
        raise ValueError(f"outer fold {outer_fold} exact-inner lineage is not complementary")
    return full[0], exact, cumulative


def _terminal(component_root: Path, *, component: str) -> dict[str, Any]:
    terminal, _digest = _read_json(
        component_root / "execution_manifest.json",
        label=f"{component} execution terminal",
    )
    if (
        terminal.get("status") != "complete"
        or terminal.get("registered_heldout_labels_accessed") is not False
        or terminal.get("oracle_fields_accessed") is not False
    ):
        raise ValueError(f"{component} terminal is incomplete or label/oracle contaminated")
    return terminal


def _logical_view(
    *,
    component_root: Path,
    terminal: Mapping[str, Any],
    source_scope: Stage1ScopeSpec,
    family: str,
) -> dict[str, Any]:
    registrations = terminal.get("logical_views")
    if not isinstance(registrations, list):
        raise ValueError("component terminal lacks logical-view registrations")
    component = _FAMILY_COMPONENT.get(family)
    single_family_registration = (
        component is not None
        and _COMPONENT_FAMILIES.get(component) == (family,)
    )
    matches = [
        row
        for row in registrations
        if isinstance(row, Mapping)
        and row.get("logical_scope_id") == source_scope.scope_id
        and (
            row.get("family") == family
            or (
                single_family_registration
                and row.get("family") is None
            )
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{source_scope.scope_id}/{family} has no unique logical view"
        )
    registration, path = _validated_registration(
        component_root,
        matches[0],
        label=f"{source_scope.scope_id}/{family} logical view",
    )
    view, _digest = _read_json(
        path,
        label=f"{source_scope.scope_id}/{family} logical view",
    )
    if (
        view.get("content_sha256") != registration.get("content_sha256")
        or view.get("logical_scope_id") != source_scope.scope_id
        or view.get("family") != family
        or view.get("physical_owner_scope_id") != source_scope.scope_id
        or view.get("logical_transform_performed") is not True
        or view.get("registered_heldout_labels_accessed") is not False
    ):
        raise ValueError(
            f"{source_scope.scope_id}/{family} primary transform binding changed"
        )
    return view


def _axis_role_for_name(
    family: str,
    name: str,
) -> tuple[str, tuple[str, ...], str]:
    lowered = name.lower()
    if family == BOW_NUISANCE:
        if lowered.endswith("::treatment_nuisance"):
            return (
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                (TREATMENT_AXIS,),
                RAW_FEATURE_BANK,
            )
        if lowered.endswith("::outcome_nuisance"):
            return (
                OUTCOME_NUISANCE_FEATURE_ROLE,
                (OUTCOME_AXIS,),
                RAW_FEATURE_BANK,
            )
    if family == BOW_R_LOSS:
        return (
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            (HETEROGENEITY_AXIS,),
            CALIBRATED_SOURCE_BANK,
        )
    if family == HTR_NEURAL:
        if lowered == "htr_nuisance::e_hat":
            return (
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                (TREATMENT_AXIS,),
                RAW_FEATURE_BANK,
            )
        if lowered == "htr_nuisance::m_hat":
            return (
                OUTCOME_NUISANCE_FEATURE_ROLE,
                (OUTCOME_AXIS,),
                RAW_FEATURE_BANK,
            )
        if lowered.startswith("htr_effect::"):
            return (
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                (HETEROGENEITY_AXIS,),
                CALIBRATED_SOURCE_BANK,
            )
    if family == TFIDF_TOPICS:
        if lowered.startswith("treatment::"):
            return (
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                (TREATMENT_AXIS,),
                RAW_FEATURE_BANK,
            )
        if lowered.startswith("outcome::"):
            return (
                OUTCOME_NUISANCE_FEATURE_ROLE,
                (OUTCOME_AXIS,),
                RAW_FEATURE_BANK,
            )
    if family == NEURAL_QUERY_MOMENTS:
        if lowered.startswith("neural_query_treatment_"):
            return (
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                (TREATMENT_AXIS,),
                RAW_FEATURE_BANK,
            )
        if lowered.startswith("neural_query_outcome_"):
            return (
                OUTCOME_NUISANCE_FEATURE_ROLE,
                (OUTCOME_AXIS,),
                RAW_FEATURE_BANK,
            )
    return (
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        (HETEROGENEITY_AXIS,),
        RAW_FEATURE_BANK,
    )


def _dense_block(
    *,
    component_root: Path,
    registration: Any,
    component: str,
    family: str,
    alignment_group: str,
    source_names: Sequence[str] | None = None,
    source_kinds: Sequence[str] | None = None,
    roles: Sequence[str] | None = None,
    axes: Sequence[Sequence[str]] | None = None,
    bank_kinds: Sequence[str] | None = None,
    alignment_mode: str = EXACT_NAMED_ALIGNMENT,
    payload_cache: _AuthenticatedNumericalPayloadCache | None = None,
) -> dict[str, Any]:
    normalized, array = _array_reference(
        component_root,
        registration,
        label=f"{component}/{family}/{alignment_group}",
        payload_cache=payload_cache,
    )
    matrix = np.asarray(array)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    width = int(matrix.shape[1])
    registered_columns = (
        tuple(str(value) for value in registration.get("columns") or ())
        if isinstance(registration, Mapping)
        else ()
    )
    names = tuple(source_names or registered_columns)
    if len(names) != width or any(not value for value in names):
        raise ValueError(f"{family}/{alignment_group} has incomplete column names")
    if source_kinds is None or roles is None or axes is None or bank_kinds is None:
        derived = tuple(_axis_role_for_name(family, name) for name in names)
        if source_kinds is None:
            source_kinds = tuple(family for _ in names)
        if roles is None:
            roles = tuple(row[0] for row in derived)
        if axes is None:
            axes = tuple(row[1] for row in derived)
        if bank_kinds is None:
            bank_kinds = tuple(row[2] for row in derived)
    kinds = tuple(str(value) for value in source_kinds)
    role_values = tuple(str(value) for value in roles)
    axis_values = tuple(tuple(str(axis) for axis in value) for value in axes)
    banks = tuple(str(value) for value in bank_kinds)
    if not (
        len(kinds)
        == len(role_values)
        == len(axis_values)
        == len(banks)
        == width
    ):
        raise ValueError(f"{family}/{alignment_group} metadata is not column-aligned")
    if (
        set(role_values) - _ROLES
        or set(banks) - _BANK_KINDS
        or any(not value or set(value) - _AXES for value in axis_values)
        or alignment_mode not in _ALIGNMENTS
    ):
        raise ValueError(f"{family}/{alignment_group} has unsupported routing metadata")
    return {
        "payload_kind": DENSE_NPY_PAYLOAD,
        "source_component": component,
        "source_family": family,
        "alignment_group": alignment_group,
        "alignment_mode": alignment_mode,
        "row_count": int(matrix.shape[0]),
        "source_column_count": width,
        "source_names": list(names),
        "source_kinds": list(kinds),
        "consumer_roles": list(role_values),
        "observable_axes": [list(value) for value in axis_values],
        "bank_kinds": list(banks),
        "files": [normalized],
    }


def _embedding_config(component_root: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    metadata, _digest = _read_json(
        component_root / "fit_state" / "metadata.json",
        label="embedding fit-state metadata",
    )
    config = metadata.get("scientific_configuration")
    if not isinstance(config, Mapping):
        raise ValueError("embedding fit-state lacks scientific configuration")
    vocabulary_registration = metadata.get("semantic_vocabulary")
    registered, path = _validated_registration(
        component_root,
        vocabulary_registration,
        label="embedding semantic vocabulary",
    )
    vocabulary, _vocab_digest = _read_json(
        path,
        label="embedding semantic vocabulary",
    )
    terms = tuple(str(value) for value in vocabulary.get("terms") or ())
    if (
        not terms
        or len(terms) != len(set(terms))
        or vocabulary.get("content_sha256")
        != registered.get("content_sha256")
    ):
        raise ValueError("embedding semantic vocabulary changed")
    return copy.deepcopy(dict(config)), terms


def _embedding_blocks(
    *,
    component_root: Path,
    view: Mapping[str, Any],
    family: str,
    expected_rows: tuple[int, ...],
    payload_cache: _AuthenticatedNumericalPayloadCache | None = None,
) -> tuple[dict[str, Any], ...]:
    exact_metadata, _digest = _read_json(
        component_root / "exact_transforms" / "metadata.json",
        label="embedding exact-transform metadata",
    )
    if (
        exact_metadata.get("content_sha256")
        != view.get("exact_transform_content_sha256")
        or _rows(
            exact_metadata.get("heldout_row_ids"),
            label="embedding heldout rows",
        )
        != expected_rows
        or exact_metadata.get("registered_heldout_labels_accessed") is not False
    ):
        raise ValueError("embedding exact-transform row lineage changed")
    registrations = view.get("prediction_artifacts")
    if not isinstance(registrations, list):
        raise ValueError("embedding logical view lacks prediction artifacts")
    by_name = {
        PurePosixPath(str(row.get("relative_path"))).stem: row
        for row in registrations
        if isinstance(row, Mapping)
    }
    config, vocabulary = _embedding_config(component_root)
    transform = exact_metadata.get("transform_metadata")
    if not isinstance(transform, Mapping):
        raise ValueError("embedding exact-transform metadata is malformed")
    if family == EMBEDDING_WHOLE_COHORT:
        names = tuple(str(value) for value in transform.get("whole_contrast_names") or ())
        contrast_rows = config.get("contrasts")
        if (
            not isinstance(contrast_rows, list)
            or tuple(str(row.get("name")) for row in contrast_rows) != names
        ):
            raise ValueError("embedding contrast metadata changed")
        roles: list[str] = []
        axes: list[tuple[str, ...]] = []
        banks: list[str] = []
        for contrast in contrast_rows:
            family_name = str(contrast.get("contrast_family"))
            source = str(contrast.get("target_source"))
            if source == "fit_treatment":
                roles.append(PROPENSITY_NUISANCE_FEATURE_ROLE)
                axes.append((TREATMENT_AXIS,))
                banks.append(RAW_FEATURE_BANK)
            elif source == "fit_outcome":
                roles.append(OUTCOME_NUISANCE_FEATURE_ROLE)
                axes.append((OUTCOME_AXIS,))
                banks.append(RAW_FEATURE_BANK)
            elif family_name == "marginal_confounder_average":
                roles.append(OUTCOME_NUISANCE_FEATURE_ROLE)
                axes.append((TREATMENT_AXIS, OUTCOME_AXIS))
                banks.append(RAW_FEATURE_BANK)
            else:
                roles.append(UNCALIBRATED_EFFECT_MODIFIER_ROLE)
                axes.append((HETEROGENEITY_AXIS,))
                banks.append(RAW_FEATURE_BANK)
        block = _dense_block(
            component_root=component_root,
            registration=by_name.get("heldout_whole_patient_scores"),
            component="embeddings",
            family=family,
            alignment_group="whole_contrast_scores",
            source_names=names,
            source_kinds=tuple(
                f"embedding_whole_cohort::{row.get('contrast_family')}"
                for row in contrast_rows
            ),
            roles=roles,
            axes=axes,
            bank_kinds=banks,
            payload_cache=payload_cache,
        )
        return (block,)
    if family == EMBEDDING_CLUSTERED:
        output: list[dict[str, Any]] = []
        distances = by_name.get("heldout_cluster_distances")
        if distances is None:
            raise ValueError("cluster-local embeddings lack distance coordinates")
        distance_width = int((distances.get("shape") or [0, 0])[1])
        output.append(
            _dense_block(
                component_root=component_root,
                registration=distances,
                component="embeddings",
                family=family,
                alignment_group="cluster_distance",
                source_names=tuple(
                    f"cluster_distance::{index:06d}"
                    for index in range(distance_width)
                ),
                source_kinds=("embedding_cluster_distance",) * distance_width,
                roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,) * distance_width,
                axes=((TREATMENT_AXIS, OUTCOME_AXIS, HETEROGENEITY_AXIS),)
                * distance_width,
                bank_kinds=(RAW_FEATURE_BANK,) * distance_width,
                alignment_mode=COMPLETE_SIGNED_ORDER_ALIGNMENT,
                payload_cache=payload_cache,
            )
        )
        projection_rows = transform.get("cluster_svd_projections")
        if not isinstance(projection_rows, list):
            raise ValueError("cluster-local embedding SVD metadata changed")
        for projection in projection_rows:
            if not isinstance(projection, Mapping):
                raise ValueError("cluster-local embedding SVD metadata is malformed")
            key = str(projection.get("array_key"))
            registration = by_name.get(key)
            width = int(projection.get("component_count", -1))
            group = f"cluster_svd::{projection.get('family_key')}"
            output.append(
                _dense_block(
                    component_root=component_root,
                    registration=registration,
                    component="embeddings",
                    family=family,
                    alignment_group=group,
                    source_names=tuple(
                        f"{group}::{index:06d}" for index in range(width)
                    ),
                    source_kinds=("embedding_cluster_svd_projection",) * width,
                    roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,) * width,
                    axes=((HETEROGENEITY_AXIS,),) * width,
                    bank_kinds=(RAW_FEATURE_BANK,) * width,
                    alignment_mode=COMPLETE_ABSOLUTE_ORDER_ALIGNMENT,
                    payload_cache=payload_cache,
                )
            )
        return tuple(output)
    if family != TFIDF_SEMANTIC_RETRIEVAL:
        raise ValueError(f"unsupported embedding family {family!r}")
    shape = tuple(int(value) for value in transform.get("lexical_csr_shape") or ())
    if shape != (len(expected_rows), len(vocabulary)):
        raise ValueError("lexical semantic-retrieval CSR shape changed")
    files: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for suffix in ("data", "indices", "indptr"):
        key = f"heldout_lexical_csr_{suffix}"
        normalized, array = _array_reference(
            component_root,
            by_name.get(key),
            label=f"embedding lexical CSR {suffix}",
            payload_cache=payload_cache,
        )
        files.append(normalized)
        arrays[suffix] = array
    matrix = sparse.csr_matrix(
        (
            np.asarray(arrays["data"], dtype=np.float64),
            np.asarray(arrays["indices"], dtype=np.int64),
            np.asarray(arrays["indptr"], dtype=np.int64),
        ),
        shape=shape,
    )
    if not matrix.has_sorted_indices or not np.isfinite(matrix.data).all():
        raise ValueError("lexical semantic-retrieval CSR payload changed")
    return (
        {
            "payload_kind": SPARSE_CSR_PAYLOAD,
            "source_component": "embeddings",
            "source_family": family,
            "alignment_group": "complete_semantic_vocabulary",
            "alignment_mode": COMPLETE_SIGNED_ORDER_ALIGNMENT,
            "row_count": shape[0],
            "source_column_count": shape[1],
            "source_names": list(vocabulary),
            "source_kinds": ["tfidf_semantic_retrieval_coordinate"] * shape[1],
            "consumer_roles": [UNCALIBRATED_EFFECT_MODIFIER_ROLE] * shape[1],
            "observable_axes": [[HETEROGENEITY_AXIS]] * shape[1],
            "bank_kinds": [RAW_FEATURE_BANK] * shape[1],
            "csr_shape": list(shape),
            "files": files,
        },
    )


def _neural_query_block(
    *,
    component_root: Path,
    view: Mapping[str, Any],
    expected_rows: tuple[int, ...],
    payload_cache: _AuthenticatedNumericalPayloadCache | None = None,
) -> dict[str, Any]:
    artifact = view.get("prediction_artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("neural-query primary view lacks its prediction artifact")
    relative = PurePosixPath(str(artifact.get("relative_path")))
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("neural-query prediction directory is noncanonical")
    root = component_root.joinpath(*relative.parts)
    if payload_cache is None:
        descriptor, arrays = validate_npy_array_set(
            root,
            expected_order=("gate_row_ids", "feature_values"),
            expected_inventory=artifact.get("array_inventory"),
        )
    else:
        descriptor, arrays = payload_cache.validate_neural_array_set_once(
            root,
            expected_inventory=artifact.get("array_inventory"),
        )
    row_ids = tuple(
        int(value)
        for value in np.asarray(arrays["gate_row_ids"], dtype=np.int64).tolist()
    )
    if row_ids != expected_rows or artifact.get("heldout_labels_present") is not False:
        raise ValueError("neural-query prediction row order changed")
    values = np.asarray(arrays["feature_values"])
    names = tuple(str(value) for value in artifact.get("feature_names") or ())
    kinds = tuple(str(value) for value in artifact.get("feature_kinds") or ())
    roles = tuple(str(value) for value in artifact.get("feature_roles") or ())
    if (
        values.shape != (len(expected_rows), len(names))
        or len(kinds) != len(names)
        or len(roles) != len(names)
        or set(roles) - _ROLES
        or not np.isfinite(values).all()
    ):
        raise ValueError("neural-query feature metadata changed")
    index, index_sha = _read_json(
        root / "index.json",
        label="neural-query prediction array index",
    )
    if (
        index_sha != descriptor.get("index_sha256")
        or index.get("content_sha256") != descriptor.get("content_sha256")
    ):
        raise ValueError("neural-query prediction array index changed")
    inventory = {
        str(row["name"]): row
        for row in index.get("arrays") or ()
        if isinstance(row, Mapping)
    }
    feature = inventory.get("feature_values")
    if feature is None:
        raise ValueError("neural-query feature-value registration is missing")
    file_registration = {
        "relative_path": (
            relative / str(feature["relative_path"])
        ).as_posix(),
        "sha256": feature["file_sha256"],
        "size_bytes": int(feature["file_size_bytes"]),
        "dtype": feature["dtype"],
        "shape": list(feature["shape"]),
        "content_sha256": feature["content_sha256"],
    }
    if payload_cache is not None:
        payload_cache.adopt_externally_authenticated_array(
            component_root,
            file_registration,
            np.asarray(arrays["feature_values"]),
            label="neural-query feature-value payload",
        )
    axes = []
    banks = []
    for role in roles:
        if role == PROPENSITY_NUISANCE_FEATURE_ROLE:
            axes.append((TREATMENT_AXIS,))
        elif role == OUTCOME_NUISANCE_FEATURE_ROLE:
            axes.append((OUTCOME_AXIS,))
        else:
            axes.append((HETEROGENEITY_AXIS,))
        banks.append(RAW_FEATURE_BANK)
    return {
        "payload_kind": DENSE_NPY_PAYLOAD,
        "source_component": "neural_query",
        "source_family": NEURAL_QUERY_MOMENTS,
        "alignment_group": "neural_query_moments",
        "alignment_mode": EXACT_NAMED_ALIGNMENT,
        "row_count": len(expected_rows),
        "source_column_count": len(names),
        "source_names": list(names),
        "source_kinds": list(kinds),
        "consumer_roles": list(roles),
        "observable_axes": [list(value) for value in axes],
        "bank_kinds": banks,
        "files": [file_registration],
        "array_set_index_sha256": descriptor["index_sha256"],
        "array_set_content_sha256": descriptor["content_sha256"],
    }


def _blocks_for_family(
    *,
    component_root: Path,
    terminal: Mapping[str, Any],
    source_scope: Stage1ScopeSpec,
    logical_scope: Stage1ScopeSpec,
    family: str,
    payload_cache: _AuthenticatedNumericalPayloadCache | None = None,
) -> tuple[dict[str, Any], ...]:
    view = _logical_view(
        component_root=component_root,
        terminal=terminal,
        source_scope=source_scope,
        family=family,
    )
    expected_rows = source_scope.heldout_row_ids
    registered_rows = view.get("logical_heldout_row_ids")
    if registered_rows is not None and _rows(
        registered_rows,
        label=f"{source_scope.scope_id}/{family} heldout rows",
    ) != expected_rows:
        raise ValueError(f"{source_scope.scope_id}/{family} heldout row order changed")
    if expected_rows != logical_scope.heldout_row_ids:
        raise ValueError(
            f"{logical_scope.scope_id}/{family} physical transform rows "
            "do not equal the requested logical heldout rows"
        )
    component = _FAMILY_COMPONENT[family]
    if component == "embeddings":
        return _embedding_blocks(
            component_root=component_root,
            view=view,
            family=family,
            expected_rows=expected_rows,
            payload_cache=payload_cache,
        )
    if component == "neural_query":
        return (
            _neural_query_block(
                component_root=component_root,
                view=view,
                expected_rows=expected_rows,
                payload_cache=payload_cache,
            ),
        )
    if component == "matched_pair":
        artifacts = view.get("prediction_artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {"bow", "htr"}:
            raise ValueError("matched-pair prediction artifact coverage changed")
        return tuple(
            _dense_block(
                component_root=component_root,
                registration=artifacts[subproducer],
                component=component,
                family=family,
                alignment_group=f"matched_pair::{subproducer}",
                payload_cache=payload_cache,
            )
            for subproducer in ("bow", "htr")
        )
    artifact = view.get("prediction_artifact")
    if artifact is None:
        raise ValueError(f"{source_scope.scope_id}/{family} lacks numerical predictions")
    alignment = (
        COMPLETE_SIGNED_ORDER_ALIGNMENT
        if family == TFIDF_ORPHAN_NGRAMS
        else EXACT_NAMED_ALIGNMENT
    )
    return (
        _dense_block(
            component_root=component_root,
            registration=artifact,
            component=component,
            family=family,
            alignment_group=family,
            alignment_mode=alignment,
            payload_cache=payload_cache,
        ),
    )


def _projection_record(
    *,
    execution_root: Path,
    plan: Stage1ScopePlan,
    logical_scope: Stage1ScopeSpec,
    row_scope: str,
    payload_cache: _AuthenticatedNumericalPayloadCache | None = None,
) -> dict[str, Any]:
    if row_scope not in _ROW_SCOPES:
        raise ValueError("unknown numerical row scope")
    source_scope = plan.physical_owner(logical_scope.scope_id)
    exact_fit_order_equal = (
        source_scope.fit_row_ids == logical_scope.fit_row_ids
    )
    if (
        len(source_scope.fit_row_ids) != len(logical_scope.fit_row_ids)
        or set(source_scope.fit_row_ids) != set(logical_scope.fit_row_ids)
    ):
        raise ValueError("logical/physical numerical fit membership changed")
    blocks: list[dict[str, Any]] = []
    terminals: dict[str, dict[str, Any]] = {}
    for component in _COMPONENT_FAMILIES:
        component_root = (
            execution_root
            / ROLE_NEUTRAL_COMPONENT_DIRECTORY
            / source_scope.scope_id
            / component
        )
        if component_root != component_root.resolve(strict=True):
            raise ValueError("component root locator was substituted")
        terminals[component] = _terminal(component_root, component=component)
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        component = _FAMILY_COMPONENT[family]
        component_root = (
            execution_root
            / ROLE_NEUTRAL_COMPONENT_DIRECTORY
            / source_scope.scope_id
            / component
        )
        blocks.extend(
            _blocks_for_family(
                component_root=component_root,
                terminal=terminals[component],
                source_scope=source_scope,
                logical_scope=logical_scope,
                family=family,
                payload_cache=payload_cache,
            )
        )
    if set(block["source_family"] for block in blocks) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError("numerical projection does not cover all ten native families")
    body = {
        "outer_fold": int(logical_scope.outer_fold),
        "row_scope": row_scope,
        "inner_fold": (
            int(logical_scope.inner_fold)
            if logical_scope.inner_fold is not None
            else None
        ),
        "context_epoch": (
            int(logical_scope.context_epoch)
            if logical_scope.context_epoch is not None
            else None
        ),
        "logical_scope_id": logical_scope.scope_id,
        "logical_scope_sha256": logical_scope.as_dict()["scope_sha256"],
        "source_transform_scope_id": source_scope.scope_id,
        "source_transform_scope_sha256": source_scope.as_dict()["scope_sha256"],
        "logical_purpose": logical_scope.scope_kind,
        "physical_owner_scope_id": source_scope.scope_id,
        "fit_row_ids": list(logical_scope.fit_row_ids),
        "source_transform_fit_row_ids": list(source_scope.fit_row_ids),
        "heldout_row_ids": list(logical_scope.heldout_row_ids),
        "fit_row_order_sha256": _sha256_json(list(logical_scope.fit_row_ids)),
        "source_transform_fit_row_order_sha256": _sha256_json(
            list(source_scope.fit_row_ids)
        ),
        "heldout_row_order_sha256": _sha256_json(list(logical_scope.heldout_row_ids)),
        "blocks": blocks,
        "logical_and_physical_fit_rows_equal": exact_fit_order_equal,
        "logical_and_physical_fit_row_membership_equal": True,
        "logical_and_physical_fit_row_order_equal": exact_fit_order_equal,
        "physical_owner_row_order_retained": True,
        "source_transform_rows_equal_logical_heldout_rows": True,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _coordinate_schema(
    projections: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    by_group: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for projection in projections:
        for block in projection["blocks"]:
            key = (str(block["source_family"]), str(block["alignment_group"]))
            by_group.setdefault(key, []).append(block)
    coordinates: list[dict[str, Any]] = []
    for (family, group), blocks in by_group.items():
        modes = {str(block["alignment_mode"]) for block in blocks}
        if len(modes) != 1:
            raise ValueError(f"{family}/{group} alignment mode changed across scopes")
        mode = next(iter(modes))
        if mode == EXACT_NAMED_ALIGNMENT:
            schemas = {
                _sha256_json(
                    {
                        "names": block["source_names"],
                        "kinds": block["source_kinds"],
                        "roles": block["consumer_roles"],
                        "axes": block["observable_axes"],
                        "banks": block["bank_kinds"],
                    }
                )
                for block in blocks
            }
            if len(schemas) != 1:
                raise ValueError(
                    f"{family}/{group} exact coordinate schema changed across scopes"
                )
            first = blocks[0]
            for index, name in enumerate(first["source_names"]):
                identity = {
                    "source_family": family,
                    "alignment_group": group,
                    "alignment_mode": mode,
                    "coordinate_name": str(name),
                    "source_kind": str(first["source_kinds"][index]),
                    "consumer_role": str(first["consumer_roles"][index]),
                    "observable_axes": list(first["observable_axes"][index]),
                    "bank_kind": str(first["bank_kinds"][index]),
                    "source_column_index": index,
                    "statistic_kind": "exact_named_coordinate",
                    "statistic_rank": None,
                    "source_coordinate_identity_preserved": True,
                    "concept_grounding_allowed": False,
                }
                digest = _sha256_json(identity)
                coordinates.append(
                    {
                        **identity,
                        "coordinate_id": f"ref.{len(coordinates):06d}.{digest[:12]}",
                        "coordinate_identity_sha256": digest,
                    }
                )
            continue
        widths = {int(block["source_column_count"]) for block in blocks}
        maximum = max(widths)
        if maximum < 1:
            raise ValueError(f"{family}/{group} has no complete-order coordinates")
        first = blocks[0]
        roles = {
            str(value)
            for block in blocks
            for value in block["consumer_roles"]
        }
        axes = {
            tuple(value)
            for block in blocks
            for value in block["observable_axes"]
        }
        banks = {
            str(value)
            for block in blocks
            for value in block["bank_kinds"]
        }
        if (
            len(roles) != 1
            or len(axes) != 1
            or len(banks) != 1
        ):
            raise ValueError(
                f"{family}/{group} complete-order coordinates have mixed routing"
            )
        role = next(iter(roles))
        axis = list(next(iter(axes)))
        bank = next(iter(banks))
        for statistic in ("value", "presence"):
            for rank in range(1, maximum + 1):
                identity = {
                    "source_family": family,
                    "alignment_group": group,
                    "alignment_mode": mode,
                    "coordinate_name": (
                        f"{family}::{group}::{statistic}_rank_{rank:06d}"
                    ),
                    "source_kind": (
                        f"{family}_complete_permutation_invariant_{statistic}"
                    ),
                    "consumer_role": role,
                    "observable_axes": axis,
                    "bank_kind": bank,
                    "source_column_index": None,
                    "statistic_kind": (
                        "signed_descending_order"
                        if statistic == "value"
                        and mode == COMPLETE_SIGNED_ORDER_ALIGNMENT
                        else (
                            "absolute_descending_order"
                            if statistic == "value"
                            else "coordinate_presence"
                        )
                    ),
                    "statistic_rank": rank,
                    "source_coordinate_identity_preserved": False,
                    "concept_grounding_allowed": False,
                }
                digest = _sha256_json(identity)
                coordinates.append(
                    {
                        **identity,
                        "coordinate_id": f"ref.{len(coordinates):06d}.{digest[:12]}",
                        "coordinate_identity_sha256": digest,
                    }
                )
    if not coordinates:
        raise ValueError("direct numerical reference manifest has no coordinates")
    return tuple(coordinates)


def _manifest_body(
    *,
    execution_root: Path,
    plan: Stage1ScopePlan,
    execution_manifest: Mapping[str, Any],
    payload_cache: _AuthenticatedNumericalPayloadCache | None = None,
) -> dict[str, Any]:
    fresh = validate_role_neutral_stage1_execution(
        root=execution_root,
        plan=plan,
    )
    if dict(execution_manifest) != fresh:
        raise ValueError(
            "supplied execution manifest differs from fresh path-only validation"
        )
    outer_folds = tuple(
        sorted(
            {
                int(scope.outer_fold)
                for scope in plan.scopes
                if scope.scope_kind == "full_outer"
            }
        )
    )
    projections: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for outer_fold in outer_folds:
        full, exact, cumulative = _scope_rows(plan, outer_fold)
        exact_projections = [
            _projection_record(
                execution_root=execution_root,
                plan=plan,
                logical_scope=scope,
                row_scope=OUTER_TRAIN_OOF_SCOPE,
                payload_cache=payload_cache,
            )
            for scope in exact
        ]
        full_projection = _projection_record(
            execution_root=execution_root,
            plan=plan,
            logical_scope=full,
            row_scope=OUTER_HELDOUT_SCOPE,
            payload_cache=payload_cache,
        )
        review_projections = [
            _projection_record(
                execution_root=execution_root,
                plan=plan,
                logical_scope=scope,
                row_scope=REVIEW_GATE_SCOPE,
                payload_cache=payload_cache,
            )
            for scope in cumulative
        ]
        projections.extend(exact_projections)
        projections.append(full_projection)
        projections.extend(review_projections)
        fold_by_row = {
            row_id: int(scope.inner_fold or -1)
            for scope in exact
            for row_id in scope.heldout_row_ids
        }
        meta_ids = tuple(fold_by_row[row_id] for row_id in full.fit_row_ids)
        if any(value < 1 for value in meta_ids):
            raise ValueError("exact-inner fold assignment is incomplete")
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "outer_train_row_ids": list(full.fit_row_ids),
                "outer_heldout_row_ids": list(full.heldout_row_ids),
                "meta_inner_fold_ids": list(meta_ids),
                "exact_inner_scope_ids": [scope.scope_id for scope in exact],
                "full_outer_scope_id": full.scope_id,
                "cumulative_review_scope_ids": [
                    scope.scope_id for scope in cumulative
                ],
                "outer_train_rows_covered_exactly_once_by_exact_inner": True,
                "outer_heldout_rows_from_full_outer_transform_only": True,
            }
        )
    coordinates = _coordinate_schema(projections)
    family_coordinates = {
        family: [
            row["coordinate_id"]
            for row in coordinates
            if row["source_family"] == family
        ]
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    if any(not values for values in family_coordinates.values()):
        raise ValueError("direct numerical reference coordinate coverage is not all-ten")
    body = {
        "schema_version": DIRECT_NUMERICAL_REFERENCE_MANIFEST_SCHEMA,
        "channel": DIRECT_UPSTREAM_NUMERICAL_CHANNEL,
        "producer": DIRECT_NUMERICAL_REFERENCE_BANK_ID,
        "plan_scientific_content_sha256": plan.scientific_content_sha256,
        "source_execution_content_sha256": _require_sha256(
            fresh["content_sha256"],
            label="source execution",
        ),
        "native_family_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
        "outer_folds": fold_rows,
        "projections": projections,
        "coordinates": list(coordinates),
        "family_coverage": [
            {
                "source_family": family,
                "coordinate_ids": family_coordinates[family],
                "numerical_payload_reference_count": sum(
                    1
                    for projection in projections
                    for block in projection["blocks"]
                    if block["source_family"] == family
                ),
                "nonempty_native_numerical_family": True,
            }
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        ],
        "outer_train_oof_source_policy": "exact_inner_primary_transforms_only",
        "outer_heldout_source_policy": "full_outer_primary_transform_only",
        "whole_cohort_and_cluster_local_embeddings_independently_represented": True,
        "all_source_bytes_freshly_authenticated": True,
        "source_numerical_payloads_copied": False,
        "combined_npy_payloads_persisted": False,
        "fit_or_refit_performed": False,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "execution_locator_in_scientific_identity": False,
    }
    return body


def _projection_key(value: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        int(value["outer_fold"]),
        str(value["row_scope"]),
        str(value["logical_scope_id"]),
    )


@dataclass(frozen=True)
class MaterializedRoleNeutralNumericalMatrix:
    row_ids: tuple[int, ...]
    coordinate_ids: tuple[str, ...]
    names: tuple[str, ...]
    source_families: tuple[str, ...]
    source_kinds: tuple[str, ...]
    consumer_roles: tuple[str, ...]
    observable_axes: tuple[tuple[str, ...], ...]
    bank_kinds: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.coordinate_ids)
        if not (
            count
            == len(self.names)
            == len(self.source_families)
            == len(self.source_kinds)
            == len(self.consumer_roles)
            == len(self.observable_axes)
            == len(self.bank_kinds)
        ):
            raise ValueError("materialized numerical metadata is not column-aligned")
        matrix = np.asarray(self.values)
        if matrix.shape != (len(self.row_ids), count) or not np.isfinite(matrix).all():
            raise ValueError("materialized numerical matrix is malformed")
        if (
            matrix.dtype == np.dtype(np.float64)
            and matrix.flags.c_contiguous
            and matrix.flags.owndata
        ):
            frozen = matrix
        else:
            frozen = np.array(matrix, dtype=np.float64, copy=True, order="C")
        frozen.setflags(write=False)
        object.__setattr__(self, "values", frozen)


def _materialize_referenced_projections(
    *,
    bank: "AuthenticatedRoleNeutralDirectNumericalBank",
    projections: Sequence[Mapping[str, Any]],
    requested_rows: tuple[int, ...],
    bank_kinds: Iterable[str] | None,
    consumer_roles: Iterable[str] | None,
    source_families: Iterable[str] | None,
) -> MaterializedRoleNeutralNumericalMatrix:
    """Materialize selected references once in caller-specified row order."""

    bank.verify_authenticated_content()
    bank._require_prepared_projection_binding()
    coordinates = tuple(bank.manifest["coordinates"])
    banks = _BANK_KINDS if bank_kinds is None else frozenset(bank_kinds)
    roles = _ROLES if consumer_roles is None else frozenset(consumer_roles)
    families = (
        ACTIVE_STAGE1_CONCEPT_FAMILY_SET
        if source_families is None
        else frozenset(source_families)
    )
    if not banks or set(banks) - _BANK_KINDS:
        raise ValueError("materialization names unsupported bank kinds")
    if not roles or set(roles) - _ROLES:
        raise ValueError("materialization names unsupported consumer roles")
    if not families or set(families) - ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError("materialization names unsupported source families")
    selected = tuple(
        row
        for row in coordinates
        if row["bank_kind"] in banks
        and row["consumer_role"] in roles
        and row["source_family"] in families
    )
    if not selected:
        raise ValueError("materialization selected no numerical coordinates")
    position = {row_id: index for index, row_id in enumerate(requested_rows)}
    output = np.empty((len(requested_rows), len(selected)), dtype=np.float64)
    filled = np.zeros((len(requested_rows), len(selected)), dtype=bool)
    selected_position = {
        row["coordinate_id"]: index for index, row in enumerate(selected)
    }
    coordinates_by_group: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for coordinate in selected:
        key = (
            str(coordinate["source_family"]),
            str(coordinate["alignment_group"]),
        )
        coordinates_by_group.setdefault(key, []).append(coordinate)

    for projection in projections:
        projection_rows = tuple(int(value) for value in projection["heldout_row_ids"])
        selected_projection_positions = tuple(
            index
            for index, row_id in enumerate(projection_rows)
            if row_id in position
        )
        if not selected_projection_positions:
            continue
        selected_projection_rows = tuple(
            projection_rows[index] for index in selected_projection_positions
        )
        target_rows = [position[row_id] for row_id in selected_projection_rows]
        for block in projection["blocks"]:
            key = (
                str(block["source_family"]),
                str(block["alignment_group"]),
            )
            group_coordinates = coordinates_by_group.get(key)
            if not group_coordinates:
                continue
            if block["payload_kind"] == DENSE_NPY_PAYLOAD:
                complete_matrix = bank._payload_cache.prepared_block(
                    projection=projection,
                    block=block,
                )
            elif block["payload_kind"] == SPARSE_CSR_PAYLOAD:
                cached_sparse = bank._payload_cache.prepared_block(
                    projection=projection,
                    block=block,
                )
                if not sparse.isspmatrix_csr(cached_sparse):
                    raise RuntimeError("authenticated sparse numerical block changed type")
                complete_matrix = cached_sparse.toarray()
            else:  # pragma: no cover - manifest validation rejects this.
                raise RuntimeError("unsupported numerical payload kind")
            if complete_matrix.shape != (
                len(projection_rows),
                int(block["source_column_count"]),
            ):
                raise ValueError("referenced numerical matrix shape changed")
            matrix = np.asarray(
                complete_matrix[list(selected_projection_positions), :],
                dtype=np.float64,
            )
            mode = str(block["alignment_mode"])
            if mode == EXACT_NAMED_ALIGNMENT:
                source_by_name = {
                    str(name): index
                    for index, name in enumerate(block["source_names"])
                }
                for coordinate in group_coordinates:
                    source_index = source_by_name.get(
                        str(coordinate["coordinate_name"])
                    )
                    if source_index is None:
                        raise ValueError("exact numerical coordinate disappeared")
                    column = selected_position[coordinate["coordinate_id"]]
                    if filled[target_rows, column].any():
                        raise ValueError("numerical projection filled one row twice")
                    output[target_rows, column] = matrix[:, source_index]
                    filled[target_rows, column] = True
                continue
            width = max(
                int(row["statistic_rank"])
                for row in group_coordinates
                if row["statistic_rank"] is not None
            )
            values = (
                np.sort(matrix, axis=1, kind="stable")[:, ::-1]
                if mode == COMPLETE_SIGNED_ORDER_ALIGNMENT
                else np.sort(np.abs(matrix), axis=1, kind="stable")[:, ::-1]
            )
            padded = np.zeros((len(selected_projection_rows), width), dtype=np.float64)
            presence = np.zeros_like(padded)
            padded[:, : values.shape[1]] = values
            presence[:, : values.shape[1]] = 1.0
            for coordinate in group_coordinates:
                rank = int(coordinate["statistic_rank"]) - 1
                source_values = (
                    presence[:, rank]
                    if coordinate["statistic_kind"] == "coordinate_presence"
                    else padded[:, rank]
                )
                column = selected_position[coordinate["coordinate_id"]]
                if filled[target_rows, column].any():
                    raise ValueError("numerical projection filled one row twice")
                output[target_rows, column] = source_values
                filled[target_rows, column] = True
    if not filled.all():
        missing = int(np.size(filled) - np.count_nonzero(filled))
        raise ValueError(
            f"direct numerical row/coordinate assembly is incomplete ({missing} cells)"
        )
    bank.verify_authenticated_content()
    return MaterializedRoleNeutralNumericalMatrix(
        row_ids=requested_rows,
        coordinate_ids=tuple(row["coordinate_id"] for row in selected),
        names=tuple(row["coordinate_name"] for row in selected),
        source_families=tuple(row["source_family"] for row in selected),
        source_kinds=tuple(row["source_kind"] for row in selected),
        consumer_roles=tuple(row["consumer_role"] for row in selected),
        observable_axes=tuple(tuple(row["observable_axes"]) for row in selected),
        bank_kinds=tuple(row["bank_kind"] for row in selected),
        values=output,
    )


@dataclass(frozen=True)
class RoleNeutralDirectForestBlocks:
    train_row_ids: tuple[int, ...]
    heldout_row_ids: tuple[int, ...]
    effect_names: tuple[str, ...]
    control_names: tuple[str, ...]
    effect_train_values: np.ndarray
    effect_heldout_values: np.ndarray
    control_train_values: np.ndarray
    control_heldout_values: np.ndarray
    reference_manifest_content_sha256: str


class RoleNeutralGateOnlyNumericalView:
    """One cumulative-fit, label-free gate projection with no context OOF half."""

    def __init__(
        self,
        *,
        bank: "AuthenticatedRoleNeutralDirectNumericalBank",
        projection: Mapping[str, Any],
        outer_fold: int,
        context_epoch: int,
        spent_row_ids: tuple[int, ...],
        gate_row_ids: tuple[int, ...],
    ) -> None:
        self._bank = bank
        self._projection = copy.deepcopy(dict(projection))
        self.outer_fold = int(outer_fold)
        self.context_epoch = int(context_epoch)
        self.spent_row_ids = spent_row_ids
        self.gate_row_ids = gate_row_ids
        self.available_prediction_row_ids = tuple(
            int(value) for value in projection["heldout_row_ids"]
        )
        self.context_oof_available = False
        self.fit_or_refit_performed = False

    def identity(self) -> Mapping[str, Any]:
        self._bank.verify_authenticated_content()
        body = {
            "schema_version": "role_neutral_gate_only_numerical_view_v1",
            "reference_manifest_content_sha256": self._bank.manifest[
                "content_sha256"
            ],
            "outer_fold": self.outer_fold,
            "context_epoch": self.context_epoch,
            "logical_scope_id": self._projection["logical_scope_id"],
            "source_transform_scope_id": self._projection[
                "source_transform_scope_id"
            ],
            "spent_row_ids": list(self.spent_row_ids),
            "gate_row_ids": list(self.gate_row_ids),
            "available_prediction_row_ids": list(
                self.available_prediction_row_ids
            ),
            "gate_fit_row_provenance": list(self.spent_row_ids),
            "gate_values_are_cumulative_fit_label_free_transforms": True,
            "context_oof_available": False,
            "conditional_context_gate_view_claimed": False,
            "fit_or_refit_performed": False,
            "registered_gate_labels_accessed": False,
        }
        return {**body, "content_sha256": _sha256_json(body)}

    def materialize(
        self,
        *,
        bank_kinds: Iterable[str] | None = None,
        consumer_roles: Iterable[str] | None = None,
        source_families: Iterable[str] | None = None,
    ) -> MaterializedRoleNeutralNumericalMatrix:
        """Materialize only the precommitted next-gate rows."""

        return _materialize_referenced_projections(
            bank=self._bank,
            projections=(self._projection,),
            requested_rows=self.gate_row_ids,
            bank_kinds=bank_kinds,
            consumer_roles=consumer_roles,
            source_families=source_families,
        )

    def aligned_conditional_values(self, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "gate-only role-neutral reference views contain no spent-context "
            "inner-OOF values and cannot satisfy a conditional gate contract"
        )

    def context_oof_values(self) -> Any:
        raise RuntimeError(
            "spent-context inner-OOF numerical values were not produced by "
            "the role-neutral Stage1 graph"
        )


class RoleNeutralDirectNumericalFoldView:
    """One authenticated outer-fold view over immutable producer arrays."""

    def __init__(
        self,
        *,
        bank: "AuthenticatedRoleNeutralDirectNumericalBank",
        outer_fold: int,
        train_row_ids: tuple[int, ...],
        heldout_row_ids: tuple[int, ...],
        meta_inner_fold_ids: tuple[int, ...],
    ) -> None:
        self._bank = bank
        self.outer_fold = int(outer_fold)
        self.train_row_ids = train_row_ids
        self.heldout_row_ids = heldout_row_ids
        self.meta_inner_fold_ids = meta_inner_fold_ids

    @property
    def cache_key(self) -> str:
        return _sha256_json(
            {
                "reference_manifest_content_sha256": (
                    self._bank.manifest["content_sha256"]
                ),
                "outer_fold": self.outer_fold,
                "train_row_ids": list(self.train_row_ids),
                "heldout_row_ids": list(self.heldout_row_ids),
                "meta_inner_fold_ids": list(self.meta_inner_fold_ids),
            }
        )

    def verify_authenticated_content(self) -> None:
        self._bank.verify_authenticated_content()

    def _selected_projections(self, scope: str) -> tuple[Mapping[str, Any], ...]:
        if scope == OUTER_TRAIN_OOF_SCOPE:
            return tuple(
                row
                for row in self._bank.manifest["projections"]
                if row["outer_fold"] == self.outer_fold
                and row["row_scope"] == OUTER_TRAIN_OOF_SCOPE
            )
        if scope == OUTER_HELDOUT_SCOPE:
            rows = tuple(
                row
                for row in self._bank.manifest["projections"]
                if row["outer_fold"] == self.outer_fold
                and row["row_scope"] == OUTER_HELDOUT_SCOPE
            )
            if len(rows) != 1:
                raise ValueError("full-outer numerical projection is not unique")
            return rows
        raise ValueError("final fold view supports only OOF and outer-heldout scopes")

    def materialize(
        self,
        *,
        scope: str,
        bank_kinds: Iterable[str] | None = None,
        consumer_roles: Iterable[str] | None = None,
        source_families: Iterable[str] | None = None,
    ) -> MaterializedRoleNeutralNumericalMatrix:
        """Assemble one in-memory matrix without persisting a combined payload."""

        requested_rows = (
            self.train_row_ids
            if scope == OUTER_TRAIN_OOF_SCOPE
            else (
                self.heldout_row_ids
                if scope == OUTER_HELDOUT_SCOPE
                else ()
            )
        )
        if not requested_rows:
            raise ValueError("final fold view supports only OOF and outer-heldout scopes")
        return _materialize_referenced_projections(
            bank=self._bank,
            projections=self._selected_projections(scope),
            requested_rows=requested_rows,
            bank_kinds=bank_kinds,
            consumer_roles=consumer_roles,
            source_families=source_families,
        )

    def forest_blocks(self) -> RoleNeutralDirectForestBlocks:
        """Materialize role-routed arrays accepted by a strict final forest."""

        effect_roles = (UNCALIBRATED_EFFECT_MODIFIER_ROLE,)
        control_roles = (
            PROPENSITY_NUISANCE_FEATURE_ROLE,
            OUTCOME_NUISANCE_FEATURE_ROLE,
        )
        effect_train = self.materialize(
            scope=OUTER_TRAIN_OOF_SCOPE,
            consumer_roles=effect_roles,
        )
        effect_heldout = self.materialize(
            scope=OUTER_HELDOUT_SCOPE,
            consumer_roles=effect_roles,
        )
        control_train = self.materialize(
            scope=OUTER_TRAIN_OOF_SCOPE,
            consumer_roles=control_roles,
        )
        control_heldout = self.materialize(
            scope=OUTER_HELDOUT_SCOPE,
            consumer_roles=control_roles,
        )
        if (
            effect_train.coordinate_ids != effect_heldout.coordinate_ids
            or control_train.coordinate_ids != control_heldout.coordinate_ids
        ):
            raise ValueError("train/heldout direct numerical coordinate schemas differ")
        return RoleNeutralDirectForestBlocks(
            train_row_ids=self.train_row_ids,
            heldout_row_ids=self.heldout_row_ids,
            effect_names=tuple(
                f"role_neutral_effect__{index:06d}"
                for index in range(len(effect_train.names))
            ),
            control_names=tuple(
                f"role_neutral_control__{index:06d}"
                for index in range(len(control_train.names))
            ),
            effect_train_values=effect_train.values,
            effect_heldout_values=effect_heldout.values,
            control_train_values=control_train.values,
            control_heldout_values=control_heldout.values,
            reference_manifest_content_sha256=self._bank.manifest[
                "content_sha256"
            ],
        )


class AuthenticatedRoleNeutralDirectNumericalBank:
    """Guarded, path-backed producer for every final outer-fold bank."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        execution_root: Path,
        plan: Stage1ScopePlan,
        manifest: Mapping[str, Any],
        payload_cache: _AuthenticatedNumericalPayloadCache,
    ) -> None:
        self.manifest_path = manifest_path
        self.execution_root = execution_root
        self.plan = plan
        self.manifest = copy.deepcopy(dict(manifest))
        self._payload_cache = payload_cache
        self._prepared_projection_binding: Any | None = None
        self._prepared_projection_binding_payload: dict[str, Any] | None = None
        self._runtime_binding: Any | None = None
        self._runtime_binding_payload: dict[str, Any] | None = None
        self._manifest_bytes_sha256 = _stable_private_file(
            manifest_path,
            label="direct numerical reference manifest",
        )[1]
        payload_cache.validate_guarded_files()
        before = _tree_stat_inventory(execution_root)
        for projection in self.manifest["projections"]:
            for block in projection["blocks"]:
                payload_cache.prepare_block(
                    execution_root=execution_root,
                    projection=projection,
                    block=block,
                )
        payload_cache.release_authentication_buffers()
        payload_cache.validate_guarded_files()
        after = _tree_stat_inventory(execution_root)
        if before != after:
            raise RuntimeError(
                "role-neutral numerical source graph changed while handles were retained"
            )
        self._authenticated_execution_stat_inventory = after

    def identity(self) -> Mapping[str, Any]:
        self.verify_authenticated_content()
        return {
            "provider": DIRECT_NUMERICAL_REFERENCE_BANK_ID,
            "manifest_content_sha256": self.manifest["content_sha256"],
            "plan_scientific_content_sha256": self.plan.scientific_content_sha256,
            "source_execution_content_sha256": self.manifest[
                "source_execution_content_sha256"
            ],
            "meta_inner_assignments": "exact_inner_primary_transforms_v1",
            "outer_heldout_transform": "full_outer_primary_transform_v1",
            "all_ten_native_families": True,
            "combined_npy_payloads_persisted": False,
            "heldout_labels_accepted": False,
        }

    def payload_cache_audit(self) -> Mapping[str, Any]:
        """Report counters proving ordinary access performs no payload reopen."""

        self.verify_authenticated_content()
        return dict(self._payload_cache.audit_counters())

    def verify_authenticated_content(self) -> None:
        value, digest = _read_json(
            self.manifest_path,
            label="direct numerical reference manifest",
        )
        if (
            digest != self._manifest_bytes_sha256
            or value != self.manifest
            or value.get("content_sha256")
            != _sha256_json(
                {
                    key: copy.deepcopy(child)
                    for key, child in value.items()
                    if key != "content_sha256"
                }
            )
        ):
            raise ValueError("direct numerical reference manifest changed")
        if (
            _tree_stat_inventory(self.execution_root)
            != self._authenticated_execution_stat_inventory
        ):
            raise ValueError("referenced role-neutral numerical source graph changed")
        self._payload_cache.validate_guarded_files()

    def bind_prepared_projection(
        self,
        binding: Any,
    ) -> "AuthenticatedRoleNeutralDirectNumericalBank":
        """Bind once to the provider-authenticated complete prepared cohort."""

        self.verify_authenticated_content()
        payload = dict(
            validate_authenticated_prepared_projection_binding(
                binding,
                expected_plan_scientific_content_sha256=(
                    self.plan.scientific_content_sha256
                ),
                expected_source_execution_content_sha256=self.manifest[
                    "source_execution_content_sha256"
                ],
            )
        )
        all_rows = {
            int(row_id)
            for scope in self.plan.scopes
            for row_id in (*scope.fit_row_ids, *scope.heldout_row_ids)
        }
        proof_rows = payload.get("physical_owner_projection_proofs")
        if not isinstance(proof_rows, list):
            raise ValueError("prepared projection binding lacks physical-owner proofs")
        owner_proofs: dict[str, str] = {}
        for row in proof_rows:
            if not isinstance(row, Mapping):
                raise ValueError("prepared projection owner proof is malformed")
            scope_id = str(row.get("physical_owner_scope_id"))
            if scope_id in owner_proofs:
                raise ValueError("prepared projection repeats a physical-owner proof")
            owner_proofs[scope_id] = _require_sha256(
                row.get("projection_proof_content_sha256"),
                label="prepared projection owner proof",
            )
        expected_owners = {scope.scope_id for scope in self.plan.physical_scopes}
        if (
            set(owner_proofs) != expected_owners
            or payload.get("row_count") != len(all_rows)
            or payload.get("all_physical_fit_projections_verified") is not True
            or payload.get("raw_text_persisted") is not False
            or payload.get("raw_treatment_persisted") is not False
            or payload.get("raw_outcome_persisted") is not False
            or payload.get("text_truncation_applied") is not False
        ):
            raise ValueError(
                "prepared projection binding is incomplete, persisted raw values, "
                "or applied text truncation"
            )
        _require_sha256(
            payload.get("content_sha256"),
            label="prepared projection binding",
        )
        if self._prepared_projection_binding_payload is not None:
            if payload != self._prepared_projection_binding_payload:
                raise RuntimeError(
                    "direct numerical bank was already bound to another "
                    "prepared cohort projection"
                )
            return self
        self._prepared_projection_binding = binding
        self._prepared_projection_binding_payload = copy.deepcopy(payload)
        return self

    def _require_prepared_projection_binding(self) -> Mapping[str, Any]:
        binding = self._prepared_projection_binding
        expected = self._prepared_projection_binding_payload
        if binding is None or expected is None:
            raise RuntimeError(
                "direct numerical consumption requires bind_prepared_projection() "
                "with the provider-issued complete prepared-cohort proof"
            )
        current = dict(
            validate_authenticated_prepared_projection_binding(
                binding,
                expected_plan_scientific_content_sha256=(
                    self.plan.scientific_content_sha256
                ),
                expected_source_execution_content_sha256=self.manifest[
                    "source_execution_content_sha256"
                ],
            )
        )
        if current != expected:
            raise RuntimeError("prepared-cohort projection binding changed after binding")
        return current

    def bind_runtime_authorization(
        self,
        binding: Any,
    ) -> "AuthenticatedRoleNeutralDirectNumericalBank":
        """Bind the provider-issued dataset/row/meta authority once."""

        projection = self._require_prepared_projection_binding()
        payload = dict(
            validate_authenticated_role_neutral_stage2_runtime_binding(
                binding,
                expected_plan_scientific_content_sha256=(
                    self.plan.scientific_content_sha256
                ),
                expected_source_execution_content_sha256=self.manifest[
                    "source_execution_content_sha256"
                ],
            )
        )
        if (
            payload.get("prepared_projection_binding_content_sha256")
            != projection["content_sha256"]
            or payload.get("runner_dataset_artifact_sha256")
            != projection["prepared_cohort_artifact_sha256"]
            or payload.get("row_map_sha256") != projection["row_map_sha256"]
            or payload.get("per_fold_text_treatment_outcome_rehash_required")
            is not False
        ):
            raise ValueError(
                "direct runtime authorization differs from the prepared "
                "projection binding"
            )
        if self._runtime_binding_payload is not None:
            if payload != self._runtime_binding_payload:
                raise RuntimeError(
                    "direct numerical bank was already bound to another runtime"
                )
            return self
        self._runtime_binding = binding
        self._runtime_binding_payload = copy.deepcopy(payload)
        return self

    def _require_runtime_authorization(self) -> Any:
        binding = self._runtime_binding
        expected = self._runtime_binding_payload
        if binding is None or expected is None:
            raise RuntimeError(
                "direct numerical consumption requires "
                "bind_runtime_authorization() with the provider-issued "
                "dataset/row/meta binding"
            )
        current = dict(
            validate_authenticated_role_neutral_stage2_runtime_binding(
                binding,
                expected_plan_scientific_content_sha256=(
                    self.plan.scientific_content_sha256
                ),
                expected_source_execution_content_sha256=self.manifest[
                    "source_execution_content_sha256"
                ],
            )
        )
        if current != expected:
            raise RuntimeError("direct runtime authorization changed after binding")
        return binding

    def prepared_projection_binding_identity(self) -> Mapping[str, Any]:
        self.verify_authenticated_content()
        payload = self._require_prepared_projection_binding()
        return {
            "schema_version": "bound_direct_numerical_prepared_projection_v1",
            "reference_manifest_content_sha256": self.manifest["content_sha256"],
            "prepared_projection_binding_content_sha256": payload[
                "content_sha256"
            ],
            "prepared_cohort_artifact_sha256": payload[
                "prepared_cohort_artifact_sha256"
            ],
            "row_map_sha256": payload["row_map_sha256"],
            "all_physical_fit_projections_verified": True,
            "text_truncation_applied": False,
        }

    def _fold(self, outer_fold: int) -> Mapping[str, Any]:
        matches = [
            row
            for row in self.manifest["outer_folds"]
            if row["outer_fold"] == int(outer_fold)
        ]
        if len(matches) != 1:
            raise ValueError("requested outer fold is absent from numerical manifest")
        return matches[0]

    def get_meta_inner_fold_ids(
        self,
        *,
        outer_fold: int,
        exact_outer_train_row_ids: Sequence[int],
    ) -> tuple[int, ...]:
        """Return plan-derived assignments aligned to the caller's exact order."""

        self.verify_authenticated_content()
        self._require_runtime_authorization()
        fold = self._fold(outer_fold)
        stored_rows = tuple(int(value) for value in fold["outer_train_row_ids"])
        stored_ids = tuple(int(value) for value in fold["meta_inner_fold_ids"])
        requested = _rows(
            tuple(exact_outer_train_row_ids),
            label="exact outer-train row IDs",
        )
        if set(requested) != set(stored_rows):
            raise ValueError("outer-train rows differ from the Stage1 exact-inner plan")
        by_row = dict(zip(stored_rows, stored_ids))
        return tuple(by_row[row_id] for row_id in requested)

    def fold_view(
        self,
        *,
        outer_fold: int,
        exact_outer_train_row_ids: Sequence[int],
        exact_outer_heldout_row_ids: Sequence[int],
    ) -> RoleNeutralDirectNumericalFoldView:
        self.verify_authenticated_content()
        self._require_runtime_authorization()
        fold = self._fold(outer_fold)
        train = _rows(
            tuple(exact_outer_train_row_ids),
            label="exact outer-train row IDs",
        )
        heldout = _rows(
            tuple(exact_outer_heldout_row_ids),
            label="exact outer-heldout row IDs",
        )
        if set(train) != set(fold["outer_train_row_ids"]):
            raise ValueError("outer-train rows differ from direct numerical manifest")
        if set(heldout) != set(fold["outer_heldout_row_ids"]):
            raise ValueError("outer-heldout rows differ from direct numerical manifest")
        meta = self.get_meta_inner_fold_ids(
            outer_fold=outer_fold,
            exact_outer_train_row_ids=train,
        )
        return RoleNeutralDirectNumericalFoldView(
            bank=self,
            outer_fold=int(outer_fold),
            train_row_ids=train,
            heldout_row_ids=heldout,
            meta_inner_fold_ids=meta,
        )

    def get_gate_only_view(
        self,
        *,
        outer_fold: int,
        context_epoch: int,
        exact_spent_row_ids: Sequence[int],
        exact_gate_row_ids: Sequence[int],
    ) -> RoleNeutralGateOnlyNumericalView:
        """Open a cumulative primary transform without claiming context OOF.

        The requested gate must be exactly the next precommitted inner
        partition.  For an early epoch the physical transform may contain
        later sealed partitions too; those rows remain unopened by
        :meth:`RoleNeutralGateOnlyNumericalView.materialize`.
        """

        self.verify_authenticated_content()
        self._require_runtime_authorization()
        epoch = int(context_epoch)
        matches = [
            row
            for row in self.manifest["projections"]
            if row["outer_fold"] == int(outer_fold)
            and row["row_scope"] == REVIEW_GATE_SCOPE
            and row["context_epoch"] == epoch
        ]
        if len(matches) != 1:
            raise ValueError("requested cumulative numerical projection is absent")
        projection = matches[0]
        spent = _rows(
            tuple(exact_spent_row_ids),
            label="exact cumulative spent row IDs",
        )
        gate = _rows(
            tuple(exact_gate_row_ids),
            label="exact next-gate row IDs",
        )
        if spent != tuple(int(value) for value in projection["fit_row_ids"]):
            raise ValueError("gate-only view changed cumulative spent row order")
        expected_inner = (
            int(self.plan.initial_training_partitions) + epoch + 1
        )
        inner_matches = [
            scope
            for scope in self.plan.scopes
            if scope.outer_fold == int(outer_fold)
            and scope.scope_kind == "exact_inner"
            and scope.inner_fold == expected_inner
        ]
        if len(inner_matches) != 1 or gate != inner_matches[0].heldout_row_ids:
            raise ValueError("gate-only view changed the precommitted next partition")
        available = tuple(
            int(value) for value in projection["heldout_row_ids"]
        )
        if not set(gate).issubset(available) or set(spent) & set(available):
            raise ValueError("cumulative gate projection row lineage changed")
        return RoleNeutralGateOnlyNumericalView(
            bank=self,
            projection=projection,
            outer_fold=int(outer_fold),
            context_epoch=epoch,
            spent_row_ids=spent,
            gate_row_ids=gate,
        )

    def prepare_hierarchy_gate_contract(
        self,
        *,
        outer_fold: int,
        context_epoch: int,
        exact_spent_row_ids: Sequence[int],
        exact_gate_row_ids: Sequence[int],
        catalog: Any,
    ) -> Any:
        """Bind discovery to one already-fit cumulative numerical projection.

        This is metadata-only: it validates the exact gate scope and emits no
        row values, conditional fit, cache materialization, or text/label hash.
        """

        from .approved_hierarchical_discovery_agent import (
            AuthenticatedReferenceOnlyDirectNumericalContract,
        )
        from .lossless_stage1_evidence_catalog import (
            RoleNeutralEvidenceCatalog,
        )

        if type(catalog) is not RoleNeutralEvidenceCatalog:
            raise TypeError(
                "hierarchy gate contract requires the exact semantic catalog"
            )
        opened = self.get_gate_only_view(
            outer_fold=outer_fold,
            context_epoch=context_epoch,
            exact_spent_row_ids=exact_spent_row_ids,
            exact_gate_row_ids=exact_gate_row_ids,
        )
        projection = opened._projection
        runtime = self._require_prepared_projection_binding()
        runtime_binding = self._runtime_binding_payload
        if runtime_binding is None:
            raise RuntimeError(
                "hierarchy gate contract lacks runtime authorization"
            )
        family_coordinates = {
            str(row["source_family"]): tuple(
                str(value) for value in row["coordinate_ids"]
            )
            for row in self.manifest["family_coverage"]
        }
        if set(family_coordinates) != set(ACTIVE_STAGE1_CONCEPT_FAMILIES):
            raise RuntimeError(
                "hierarchy gate contract lost all-ten coordinate coverage"
            )
        return AuthenticatedReferenceOnlyDirectNumericalContract.create(
            outer_fold=int(outer_fold),
            context_epoch=int(context_epoch),
            plan_scientific_content_sha256=(
                self.plan.scientific_content_sha256
            ),
            source_execution_content_sha256=self.manifest[
                "source_execution_content_sha256"
            ],
            reference_manifest_content_sha256=self.manifest[
                "content_sha256"
            ],
            runtime_binding_content_sha256=runtime_binding[
                "content_sha256"
            ],
            provider_identity_sha256=runtime[
                "provider_identity_sha256"
            ],
            spent_row_ids=opened.spent_row_ids,
            gate_row_ids=opened.gate_row_ids,
            catalog=catalog,
            family_coordinate_ids=family_coordinates,
            projection_content_sha256=projection["content_sha256"],
        )

    def produce(
        self,
        *,
        outer_fold: int,
        outer_train_row_ids: Sequence[int],
        outer_train_texts: Sequence[str],
        outer_train_treatment: Sequence[float],
        outer_train_outcome: Sequence[float],
        outer_heldout_row_ids: Sequence[int],
        outer_heldout_texts: Sequence[str],
        meta_inner_fold_ids: Sequence[int],
    ) -> RoleNeutralDirectNumericalFoldView:
        """Validate supplied outer-fit text/T/Y against the sealed fit proof.

        This compatibility surface performs no fit.  It canonicalizes the
        caller's row order and proves the exact observable outer-fit projection
        is the one authenticated by the Stage 1 provider.
        """

        runtime_binding = self._require_runtime_authorization()
        view = self.fold_view(
            outer_fold=outer_fold,
            exact_outer_train_row_ids=outer_train_row_ids,
            exact_outer_heldout_row_ids=outer_heldout_row_ids,
        )
        runtime_binding.authorize_final_fold_shapes(
            outer_fold=int(outer_fold),
            exact_outer_train_row_ids=outer_train_row_ids,
            exact_outer_heldout_row_ids=outer_heldout_row_ids,
            exact_meta_inner_fold_ids=meta_inner_fold_ids,
            outer_train_text_count=len(outer_train_texts),
            outer_train_treatment_count=len(outer_train_treatment),
            outer_train_outcome_count=len(outer_train_outcome),
            outer_heldout_text_count=len(outer_heldout_texts),
            runner_dataset_artifact_sha256=self._runtime_binding_payload[
                "runner_dataset_artifact_sha256"
            ],
        )
        return view

    def get_gate_source_view(self, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "reference-only Stage1 has gate-side cumulative transforms but lacks "
            "the spent-context inner-OOF predictions required by "
            "GateSourceSignalView; no provenance is fabricated"
        )

    def get_gate_feature_bank_view(self, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "reference-only Stage1 has gate-side cumulative transforms but lacks "
            "the spent-context inner-OOF predictions required by "
            "GateFeatureBankView; no provenance is fabricated"
        )


def publish_role_neutral_direct_numerical_reference_bank(
    *,
    root: Path | str,
    execution_root: Path | str,
    plan: Stage1ScopePlan,
    execution_manifest: Mapping[str, Any],
) -> AuthenticatedRoleNeutralDirectNumericalBank:
    """Publish JSON-only references to every final numerical source."""

    destination = Path(root)
    if not destination.is_absolute():
        raise ValueError("direct numerical reference root must be absolute")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("direct numerical reference root must be fresh")
    parent = destination.parent
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise ValueError("direct numerical reference parent must be canonical")
    source = _canonical_root(execution_root, label="role-neutral execution root")
    payload_cache = _AuthenticatedNumericalPayloadCache()
    body = _manifest_body(
        execution_root=source,
        plan=plan,
        execution_manifest=execution_manifest,
        payload_cache=payload_cache,
    )
    manifest = {**body, "content_sha256": _sha256_json(body)}
    destination.mkdir(exist_ok=False)
    manifest_path = destination / DIRECT_NUMERICAL_REFERENCE_MANIFEST
    _write_new_json(manifest_path, manifest)
    manifest_bytes, manifest_sha = _stable_private_file(
        manifest_path,
        label="direct numerical reference manifest",
    )
    source_manifest_path = source / ROLE_NEUTRAL_EXECUTION_MANIFEST
    source_bytes, source_sha = _stable_private_file(
        source_manifest_path,
        label="role-neutral execution manifest",
    )
    locator_body = {
        "schema_version": DIRECT_NUMERICAL_REFERENCE_LOCATOR_SCHEMA,
        "scientific_manifest": {
            "relative_path": DIRECT_NUMERICAL_REFERENCE_MANIFEST,
            "sha256": manifest_sha,
            "size_bytes": len(manifest_bytes),
            "content_sha256": manifest["content_sha256"],
        },
        "source_execution": {
            "absolute_root_locator": str(source),
            "manifest_relative_path": ROLE_NEUTRAL_EXECUTION_MANIFEST,
            "manifest_sha256": source_sha,
            "manifest_size_bytes": len(source_bytes),
            "manifest_content_sha256": execution_manifest["content_sha256"],
        },
        "references_only": True,
        "source_numerical_payloads_copied": False,
    }
    locator = {
        **locator_body,
        "content_sha256": _sha256_json(locator_body),
    }
    _write_new_json(
        destination / DIRECT_NUMERICAL_REFERENCE_LOCATOR,
        locator,
    )
    return AuthenticatedRoleNeutralDirectNumericalBank(
        manifest_path=manifest_path,
        execution_root=source,
        plan=plan,
        manifest=manifest,
        payload_cache=payload_cache,
    )


def load_role_neutral_direct_numerical_reference_bank(
    *,
    manifest_path: Path | str,
    plan: Stage1ScopePlan,
    execution_root: Path | str | None = None,
) -> AuthenticatedRoleNeutralDirectNumericalBank:
    """Freshly authenticate JSON references and every retained source byte."""

    supplied = Path(manifest_path)
    if (
        not supplied.is_absolute()
        or supplied.is_symlink()
        or supplied.name != DIRECT_NUMERICAL_REFERENCE_MANIFEST
    ):
        raise ValueError("direct numerical manifest path must be canonical and absolute")
    path = supplied.resolve(strict=True)
    root = path.parent
    if {child.name for child in root.iterdir()} != {
        DIRECT_NUMERICAL_REFERENCE_MANIFEST,
        DIRECT_NUMERICAL_REFERENCE_LOCATOR,
    }:
        raise ValueError("direct numerical reference root contains payload copies or extras")
    manifest, manifest_sha = _read_json(
        path,
        label="direct numerical reference manifest",
    )
    locator, _locator_sha = _read_json(
        root / DIRECT_NUMERICAL_REFERENCE_LOCATOR,
        label="direct numerical reference locator",
    )
    if (
        manifest.get("schema_version") != DIRECT_NUMERICAL_REFERENCE_MANIFEST_SCHEMA
        or manifest.get("channel") != DIRECT_UPSTREAM_NUMERICAL_CHANNEL
        or manifest.get("source_numerical_payloads_copied") is not False
        or manifest.get("combined_npy_payloads_persisted") is not False
        or manifest.get("fit_or_refit_performed") is not False
        or manifest.get("registered_heldout_labels_accessed") is not False
        or manifest.get("oracle_fields_accessed") is not False
        or locator.get("schema_version") != DIRECT_NUMERICAL_REFERENCE_LOCATOR_SCHEMA
        or locator.get("references_only") is not True
        or locator.get("source_numerical_payloads_copied") is not False
    ):
        raise ValueError("direct numerical reference contract is invalid")
    scientific = locator.get("scientific_manifest")
    source_reference = locator.get("source_execution")
    if (
        not isinstance(scientific, Mapping)
        or scientific.get("relative_path") != DIRECT_NUMERICAL_REFERENCE_MANIFEST
        or scientific.get("sha256") != manifest_sha
        or scientific.get("size_bytes") != path.stat().st_size
        or scientific.get("content_sha256") != manifest.get("content_sha256")
        or not isinstance(source_reference, Mapping)
    ):
        raise ValueError("direct numerical locator changed its scientific manifest")
    source = _canonical_root(
        (
            execution_root
            if execution_root is not None
            else Path(str(source_reference.get("absolute_root_locator")))
        ),
        label="role-neutral execution root",
    )
    source_manifest_path = source / ROLE_NEUTRAL_EXECUTION_MANIFEST
    source_payload, source_sha = _stable_private_file(
        source_manifest_path,
        label="role-neutral execution manifest",
    )
    execution, _digest = _read_json(
        source_manifest_path,
        label="role-neutral execution manifest",
    )
    if (
        source_reference.get("manifest_relative_path")
        != ROLE_NEUTRAL_EXECUTION_MANIFEST
        or source_reference.get("manifest_sha256") != source_sha
        or source_reference.get("manifest_size_bytes") != len(source_payload)
        or source_reference.get("manifest_content_sha256")
        != execution.get("content_sha256")
        or execution.get("content_sha256")
        != manifest.get("source_execution_content_sha256")
    ):
        raise ValueError("referenced role-neutral execution changed")
    payload_cache = _AuthenticatedNumericalPayloadCache()
    rebuilt_body = _manifest_body(
        execution_root=source,
        plan=plan,
        execution_manifest=execution,
        payload_cache=payload_cache,
    )
    rebuilt = {**rebuilt_body, "content_sha256": _sha256_json(rebuilt_body)}
    if rebuilt != manifest:
        raise ValueError("direct numerical source references changed")
    return AuthenticatedRoleNeutralDirectNumericalBank(
        manifest_path=path,
        execution_root=source,
        plan=plan,
        manifest=manifest,
        payload_cache=payload_cache,
    )


__all__ = [
    "AuthenticatedRoleNeutralDirectNumericalBank",
    "CALIBRATED_SOURCE_BANK",
    "COMPLETE_ABSOLUTE_ORDER_ALIGNMENT",
    "COMPLETE_SIGNED_ORDER_ALIGNMENT",
    "DENSE_NPY_PAYLOAD",
    "DIRECT_NUMERICAL_REFERENCE_BANK_ID",
    "DIRECT_NUMERICAL_REFERENCE_LOCATOR",
    "DIRECT_NUMERICAL_REFERENCE_LOCATOR_SCHEMA",
    "DIRECT_NUMERICAL_REFERENCE_MANIFEST",
    "DIRECT_NUMERICAL_REFERENCE_MANIFEST_SCHEMA",
    "EXACT_NAMED_ALIGNMENT",
    "MaterializedRoleNeutralNumericalMatrix",
    "OUTER_HELDOUT_SCOPE",
    "OUTER_TRAIN_OOF_SCOPE",
    "RAW_FEATURE_BANK",
    "REVIEW_GATE_SCOPE",
    "RoleNeutralDirectForestBlocks",
    "RoleNeutralGateOnlyNumericalView",
    "RoleNeutralDirectNumericalFoldView",
    "SPARSE_CSR_PAYLOAD",
    "load_role_neutral_direct_numerical_reference_bank",
    "publish_role_neutral_direct_numerical_reference_bank",
]
