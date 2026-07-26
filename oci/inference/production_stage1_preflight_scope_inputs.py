"""Row-restricted inputs for one clustered-embedding preflight scope.

The clustered preflight is label-dependent.  A process evaluating one scope
therefore receives only that scope's fit text/labels and a capability that
refuses every non-fit cache row.  All scopes share one immutable, read-only
embedding cache.  Scope artifacts contain only row/view metadata: embedding
arrays and chunk texts are never copied into them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import (
    AppliedInferenceConfig,
    EmbeddingContrastDiscoveryConfig,
    ExperimentConfig,
)
from .embedding_native_proof_capture import LOGICAL_FROZEN_EMBEDDING_CACHE_URI
from .production_stage1_config_wire import (
    production_stage1_effective_config_payload,
)
from .production_stage1_legacy_scope_adapter import (
    _closed_tree_inventory,
    _file_registration,
    _read_exact_parquet,
    _read_json,
    _validate_registration,
    _write_json,
    _write_parquet,
)
from .review_spent_evidence_provider import (
    BoundSpentFrozenChunkEmbeddingProvider,
    SpentOnlyFrozenChunkEmbeddingCache,
)

PREFLIGHT_SCOPE_INPUT_SCHEMA = "production_stage1_preflight_scope_input_v4"
PREFLIGHT_SCOPE_INPUT_SET_SCHEMA = "production_stage1_preflight_scope_input_set_v4"
PREFLIGHT_ONE_SCOPE_AUTHORITY_SCHEMA = "production_stage1_preflight_one_scope_authority_v1"
PREFLIGHT_SHARED_CACHE_REFERENCE_SCHEMA = (
    "production_stage1_preflight_shared_embedding_cache_reference_v1"
)
PREFLIGHT_SCOPED_CACHE_VIEW_SCHEMA = (
    "production_stage1_preflight_scoped_embedding_cache_view_v1"
)
PREFLIGHT_SCOPE_INPUT_MANIFEST = "preflight_scope_input_manifest.json"
PREFLIGHT_SCOPE_INPUT_SET_MANIFEST = "preflight_scope_input_set_manifest.json"
PREFLIGHT_SHARED_CACHE_REFERENCE = "shared_embedding_cache_reference.json"

_CONFIG_FILE = "effective_config.json"
_SEMANTIC_WITNESS_CONFIG_FILE = "semantic_witness_scientific_config.json"
_SCOPE_AUTHORITY_FILE = "one_scope_authority.json"
_MODELING_FILE = "fit_only_modeling.parquet"
_HEX = frozenset("0123456789abcdef")
_CACHE_FILES = (
    "metadata.json",
    "chunk_embeddings.npy",
    "offsets.npy",
    "chunk_texts.jsonl",
)
_CACHE_DIGEST_FIELD = {
    "metadata.json": "metadata_sha256",
    "chunk_embeddings.npy": "embeddings_sha256",
    "offsets.npy": "offsets_sha256",
    "chunk_texts.jsonl": "chunk_texts_sha256",
}
_SHARED_CACHE_HANDLES: dict[str, SpentOnlyFrozenChunkEmbeddingCache] = {}


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


def _scope_value(scope: Mapping[str, Any], key: str) -> Any:
    if key not in scope:
        raise ValueError(f"preflight scope lacks {key}")
    return scope[key]


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


def _source_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    """Match ``SpentOnlyFrozenChunkEmbeddingCache._file_stats``."""

    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _authenticated_cache_identity(cache: Any) -> Mapping[str, Any]:
    getter = getattr(cache, "authenticated_snapshot_identity", None)
    if not callable(getter):
        raise TypeError(
            "shared preflight cache must expose an already-authenticated identity"
        )
    identity = getter()
    if not isinstance(identity, Mapping):
        raise TypeError("shared preflight cache identity must be one mapping")
    return copy.deepcopy(dict(identity))


def _validated_line_spans(
    value: Any,
    *,
    row_count: int,
    file_size: int,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != int(row_count):
        raise ValueError(
            "shared preflight cache line-span index does not match row count"
        )
    spans: list[tuple[int, int]] = []
    expected_start = 0
    for raw in value:
        if (
            not isinstance(raw, (list, tuple))
            or len(raw) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, (int, np.integer))
                for item in raw
            )
        ):
            raise ValueError("shared preflight cache line-span index is malformed")
        start, stop = map(int, raw)
        if start != expected_start or stop <= start or stop > int(file_size):
            raise ValueError(
                "shared preflight cache line-span index is not contiguous"
            )
        spans.append((start, stop))
        expected_start = stop
    if expected_start != int(file_size):
        raise ValueError(
            "shared preflight cache line-span index does not cover its exact file"
        )
    return tuple(spans)


def _cache_file_rows(
    *,
    cache_dir: Path,
    logical_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    observed = {child.name for child in cache_dir.iterdir()}
    if observed != set(_CACHE_FILES):
        raise ValueError(
            "shared preflight embedding cache must contain exactly four files"
        )
    rows: list[dict[str, Any]] = []
    for name in _CACHE_FILES:
        path = cache_dir / name
        state = os.lstat(path)
        digest_field = _CACHE_DIGEST_FIELD[name]
        if (
            stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or int(state.st_nlink) != 1
            or digest_field not in logical_identity
        ):
            raise ValueError(
                f"shared preflight embedding cache file is invalid: {name}"
            )
        rows.append(
            {
                "name": name,
                "size_bytes": int(state.st_size),
                "stat_identity": list(_stat_identity(state)),
                "sha256": _require_sha256(
                    logical_identity[digest_field],
                    label=f"shared preflight {name} SHA",
                ),
            }
        )
    return tuple(rows)


def _build_shared_cache_reference(
    *,
    embedding_cache: Any,
    embedding_cache_identity: Mapping[str, Any],
    global_embedding_cache_path: Path,
) -> dict[str, Any]:
    logical_identity = json.loads(_canonical_json(dict(embedding_cache_identity)))
    if _authenticated_cache_identity(embedding_cache) != logical_identity:
        raise ValueError(
            "shared preflight cache differs from its authenticated logical identity"
        )
    supplied = Path(global_embedding_cache_path)
    cache_dir = Path(embedding_cache.cache_dir)
    if (
        not supplied.is_absolute()
        or not cache_dir.is_absolute()
        or supplied.is_symlink()
        or cache_dir.is_symlink()
        or supplied.resolve(strict=True) != cache_dir.resolve(strict=True)
    ):
        raise ValueError(
            "shared preflight cache locator differs from its authenticated handle"
        )
    cache_dir = cache_dir.resolve(strict=True)
    cache_state = os.lstat(cache_dir)
    if stat.S_ISLNK(cache_state.st_mode) or not stat.S_ISDIR(cache_state.st_mode):
        raise ValueError("shared preflight cache root is invalid")
    file_rows = _cache_file_rows(
        cache_dir=cache_dir,
        logical_identity=logical_identity,
    )

    operator_proof = getattr(
        embedding_cache,
        "operator_trusted_read_proof",
        None,
    )
    if operator_proof is not None:
        from .operator_trusted_embedding_cache_reader import (
            validate_operator_trusted_cache_read_proof,
        )

        validated_proof = validate_operator_trusted_cache_read_proof(
            operator_proof,
            cache_dir=cache_dir,
        )
        if validated_proof["provider_identity"] != logical_identity:
            raise ValueError(
                "shared preflight operator proof has another logical cache identity"
            )
        reader_mode = "operator_trusted_stat_continuity_v1"
        proof_payload: Mapping[str, Any] | None = validated_proof
    else:
        source_stats = getattr(embedding_cache, "_file_stats", None)
        if not isinstance(source_stats, Mapping) or set(source_stats) != set(
            _CACHE_FILES
        ):
            raise TypeError(
                "shared preflight cache lacks a reusable authenticated stat inventory"
            )
        for name in _CACHE_FILES:
            if tuple(source_stats[name]) != _source_stat_identity(
                os.lstat(cache_dir / name)
            ):
                raise RuntimeError(
                    "shared preflight cache changed after parent authentication: "
                    f"{name}"
                )
        reader_mode = "parent_authenticated_stat_continuity_v1"
        proof_payload = None

    raw_spans = getattr(embedding_cache, "_line_spans", None)
    chunk_size = next(
        row["size_bytes"] for row in file_rows if row["name"] == "chunk_texts.jsonl"
    )
    row_count = int(logical_identity.get("row_count", -1))
    spans = _validated_line_spans(
        raw_spans,
        row_count=row_count,
        file_size=int(chunk_size),
    )
    body = {
        "schema_version": PREFLIGHT_SHARED_CACHE_REFERENCE_SCHEMA,
        "reader_mode": reader_mode,
        "cache_dir": str(cache_dir),
        "cache_dir_stat_identity": list(_stat_identity(cache_state)),
        "cache_files": list(file_rows),
        "logical_identity": logical_identity,
        "logical_identity_sha256": _sha256_json(logical_identity),
        "chunk_text_line_spans": [list(span) for span in spans],
        "operator_trusted_read_proof": (
            None
            if proof_payload is None
            else json.loads(_canonical_json(dict(proof_payload)))
        ),
        "one_physical_cache_shared_across_scopes": True,
        "embedding_arrays_copied_into_scope_inputs": False,
        "chunk_texts_copied_into_scope_inputs": False,
        "treatment_or_outcome_supplied": False,
        "payload_bytes_reauthenticated_during_publication": False,
        "global_release_certified": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _validate_shared_cache_reference(
    *,
    path: Path | str,
    expected_content_sha256: str | None = None,
    expected_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reference_path = Path(path).absolute()
    if (
        reference_path.name != PREFLIGHT_SHARED_CACHE_REFERENCE
        or reference_path.is_symlink()
        or not reference_path.is_file()
    ):
        raise ValueError("shared preflight cache reference path is invalid")
    reference = _read_json(
        reference_path,
        label="shared preflight cache reference",
    )
    required = {
        "schema_version",
        "reader_mode",
        "cache_dir",
        "cache_dir_stat_identity",
        "cache_files",
        "logical_identity",
        "logical_identity_sha256",
        "chunk_text_line_spans",
        "operator_trusted_read_proof",
        "one_physical_cache_shared_across_scopes",
        "embedding_arrays_copied_into_scope_inputs",
        "chunk_texts_copied_into_scope_inputs",
        "treatment_or_outcome_supplied",
        "payload_bytes_reauthenticated_during_publication",
        "global_release_certified",
        "content_sha256",
    }
    body = {
        key: copy.deepcopy(value)
        for key, value in reference.items()
        if key != "content_sha256"
    }
    logical_identity = reference.get("logical_identity")
    if (
        set(reference) != required
        or reference.get("schema_version")
        != PREFLIGHT_SHARED_CACHE_REFERENCE_SCHEMA
        or reference.get("reader_mode")
        not in {
            "operator_trusted_stat_continuity_v1",
            "parent_authenticated_stat_continuity_v1",
        }
        or reference.get("content_sha256") != _sha256_json(body)
        or (
            expected_content_sha256 is not None
            and reference.get("content_sha256") != expected_content_sha256
        )
        or not isinstance(logical_identity, Mapping)
        or reference.get("logical_identity_sha256")
        != _sha256_json(logical_identity)
        or reference.get("one_physical_cache_shared_across_scopes") is not True
        or reference.get("embedding_arrays_copied_into_scope_inputs") is not False
        or reference.get("chunk_texts_copied_into_scope_inputs") is not False
        or reference.get("treatment_or_outcome_supplied") is not False
        or reference.get("payload_bytes_reauthenticated_during_publication")
        is not False
        or reference.get("global_release_certified") is not False
    ):
        raise ValueError("shared preflight cache reference is invalid")
    _require_sha256(
        reference["content_sha256"],
        label="shared preflight cache reference SHA",
    )
    _require_sha256(
        reference["logical_identity_sha256"],
        label="shared preflight cache logical identity SHA",
    )
    cache_dir = Path(str(reference.get("cache_dir") or ""))
    if (
        not cache_dir.is_absolute()
        or cache_dir.is_symlink()
        or not cache_dir.is_dir()
        or cache_dir.resolve(strict=True) != cache_dir
        or _stat_identity(os.lstat(cache_dir))
        != tuple(reference.get("cache_dir_stat_identity") or ())
    ):
        raise ValueError("shared preflight cache root changed")
    rows = reference.get("cache_files")
    if not isinstance(rows, list) or len(rows) != len(_CACHE_FILES):
        raise ValueError("shared preflight cache file inventory is incomplete")
    observed_names = {child.name for child in cache_dir.iterdir()}
    if observed_names != set(_CACHE_FILES):
        raise ValueError("shared preflight cache file inventory changed")
    for expected_name, row in zip(_CACHE_FILES, rows):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"name", "size_bytes", "stat_identity", "sha256"}
            or row.get("name") != expected_name
            or not isinstance(row.get("size_bytes"), int)
            or int(row["size_bytes"]) <= 0
            or not isinstance(row.get("stat_identity"), list)
            or len(row["stat_identity"]) != 7
            or _stat_identity(os.lstat(cache_dir / expected_name))
            != tuple(row["stat_identity"])
            or int(os.lstat(cache_dir / expected_name).st_size)
            != int(row["size_bytes"])
            or row.get("sha256")
            != logical_identity.get(_CACHE_DIGEST_FIELD[expected_name])
        ):
            raise ValueError(
                "shared preflight cache file inventory changed: "
                f"{expected_name}"
            )
        _require_sha256(
            row["sha256"],
            label=f"shared preflight {expected_name} SHA",
        )
    row_count = int(logical_identity.get("row_count", -1))
    chunk_size = int(rows[-1]["size_bytes"])
    _validated_line_spans(
        reference.get("chunk_text_line_spans"),
        row_count=row_count,
        file_size=chunk_size,
    )
    proof = reference.get("operator_trusted_read_proof")
    if reference["reader_mode"] == "operator_trusted_stat_continuity_v1":
        if not isinstance(proof, Mapping):
            raise ValueError(
                "shared preflight operator-trusted reference lacks its proof"
            )
        from .operator_trusted_embedding_cache_reader import (
            validate_operator_trusted_cache_read_proof,
        )

        validated_proof = validate_operator_trusted_cache_read_proof(
            proof,
            cache_dir=cache_dir,
        )
        if validated_proof["provider_identity"] != logical_identity:
            raise ValueError(
                "shared preflight operator proof logical identity changed"
            )
    elif proof is not None:
        raise ValueError(
            "parent-authenticated shared preflight reference contains an "
            "operator proof"
        )
    if expected_reference is not None and reference != dict(expected_reference):
        raise ValueError(
            "existing shared preflight cache reference differs from this request"
        )
    return copy.deepcopy(reference)


class _ParentAuthenticatedSharedEmbeddingCache(
    SpentOnlyFrozenChunkEmbeddingCache
):
    """Direct read-only cache handle guarded by a parent-authenticated stat set."""

    def __init__(self, reference: Mapping[str, Any]) -> None:
        from .operator_trusted_embedding_cache_reader import (
            _load_readonly_mmap,
            _open_readonly_nofollow,
        )

        cache_dir = Path(str(reference["cache_dir"]))
        if not hasattr(os, "O_NOFOLLOW"):
            raise RuntimeError("shared preflight cache requires POSIX O_NOFOLLOW")
        root_flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_DIRECTORY"):
            root_flags |= os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            root_flags |= os.O_CLOEXEC
        root_fd = os.open(cache_dir, root_flags)
        rows = {
            str(row["name"]): row
            for row in reference["cache_files"]
        }
        try:
            if _stat_identity(os.fstat(root_fd)) != tuple(
                reference["cache_dir_stat_identity"]
            ):
                raise ValueError(
                    "shared preflight cache root changed while opening"
                )
            handles = {
                name: _open_readonly_nofollow(
                    root_fd=root_fd,
                    name=name,
                    expected_stat=tuple(rows[name]["stat_identity"]),
                )
                for name in _CACHE_FILES
            }
        except BaseException:
            os.close(root_fd)
            raise
        self.cache_dir = cache_dir
        self._cache_root_fd = root_fd
        self._snapshot_files = handles
        self._shared_reference = copy.deepcopy(dict(reference))
        metadata_size = int(rows["metadata.json"]["size_bytes"])
        metadata_bytes = os.pread(
            handles["metadata.json"].fileno(),
            metadata_size,
            0,
        )
        if len(metadata_bytes) != metadata_size:
            raise RuntimeError("shared preflight cache metadata ended unexpectedly")
        try:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "shared preflight cache metadata is invalid JSON"
            ) from exc
        if not isinstance(metadata, dict):
            raise ValueError("shared preflight cache metadata must be one object")
        self._metadata = metadata
        self._embeddings = _load_readonly_mmap(
            handles["chunk_embeddings.npy"],
            name="chunk_embeddings",
        )
        self._offsets = _load_readonly_mmap(
            handles["offsets.npy"],
            name="offsets",
        )
        row_count = int(metadata.get("num_samples", -1))
        hidden_size = int(metadata.get("hidden_size", -1))
        if (
            row_count < 1
            or self._embeddings.ndim != 2
            or self._offsets.ndim != 1
            or len(self._offsets) != row_count + 1
            or not np.issubdtype(self._offsets.dtype, np.integer)
            or int(self._offsets[-1]) != int(self._embeddings.shape[0])
            or hidden_size != int(self._embeddings.shape[1])
        ):
            raise ValueError("shared preflight cache arrays are inconsistent")
        logical = reference["logical_identity"]
        if (
            int(logical.get("row_count", -1)) != row_count
            or int(logical.get("chunk_count", -1))
            != int(self._embeddings.shape[0])
        ):
            raise ValueError(
                "shared preflight cache shape differs from logical identity"
            )
        self._chunk_text_snapshot = handles["chunk_texts.jsonl"]
        self._line_spans = _validated_line_spans(
            reference["chunk_text_line_spans"],
            row_count=row_count,
            file_size=int(rows["chunk_texts.jsonl"]["size_bytes"]),
        )
        self._identity = copy.deepcopy(dict(logical))
        self._assert_shared_files_unchanged()

    def _assert_shared_files_unchanged(self) -> None:
        reference = self._shared_reference
        if _stat_identity(os.fstat(self._cache_root_fd)) != tuple(
            reference["cache_dir_stat_identity"]
        ):
            raise RuntimeError("shared preflight cache root changed during use")
        rows = {
            str(row["name"]): row
            for row in reference["cache_files"]
        }
        for name in _CACHE_FILES:
            expected = tuple(rows[name]["stat_identity"])
            try:
                descriptor_state = _stat_identity(
                    os.fstat(self._snapshot_files[name].fileno())
                )
                path_state = _stat_identity(os.lstat(self.cache_dir / name))
            except OSError as exc:
                raise RuntimeError(
                    f"shared preflight cache path changed during use: {name}"
                ) from exc
            if descriptor_state != expected or path_state != expected:
                raise RuntimeError(
                    f"shared preflight cache file changed during use: {name}"
                )
        if {child.name for child in self.cache_dir.iterdir()} != set(
            _CACHE_FILES
        ):
            raise RuntimeError(
                "shared preflight cache inventory changed during use"
            )

    def authenticated_snapshot_identity(self) -> Mapping[str, Any]:
        self._assert_shared_files_unchanged()
        return copy.deepcopy(self._identity)

    def identity(self) -> Mapping[str, Any]:
        self._assert_shared_files_unchanged()
        return copy.deepcopy(self._identity)


def _load_shared_cache(
    reference: Mapping[str, Any],
) -> SpentOnlyFrozenChunkEmbeddingCache:
    content_sha = str(reference["content_sha256"])
    cached = _SHARED_CACHE_HANDLES.get(content_sha)
    if cached is not None:
        if _authenticated_cache_identity(cached) != reference["logical_identity"]:
            raise RuntimeError(
                "memoized shared preflight cache identity changed"
            )
        return cached
    if reference["reader_mode"] == "operator_trusted_stat_continuity_v1":
        from .operator_trusted_embedding_cache_reader import (
            OperatorTrustedSpentOnlyFrozenChunkEmbeddingCache,
        )

        cache = OperatorTrustedSpentOnlyFrozenChunkEmbeddingCache(
            reference["cache_dir"],
            proof=reference["operator_trusted_read_proof"],
            authenticated_line_spans=reference["chunk_text_line_spans"],
        )
    else:
        cache = _ParentAuthenticatedSharedEmbeddingCache(reference)
    if _authenticated_cache_identity(cache) != reference["logical_identity"]:
        raise ValueError("opened shared preflight cache has another identity")
    _SHARED_CACHE_HANDLES[content_sha] = cache
    return cache


class ScopedEmbeddingView:
    """A logical cache facade that refuses every row outside one fit scope."""

    def __init__(
        self,
        *,
        shared_cache: SpentOnlyFrozenChunkEmbeddingCache,
        logical_identity: Mapping[str, Any],
        allowed_row_ids: Sequence[int],
        shared_reference_content_sha256: str,
    ) -> None:
        allowed = tuple(map(int, allowed_row_ids))
        if (
            not allowed
            or len(allowed) != len(set(allowed))
            or min(allowed) < 0
            or max(allowed) >= int(shared_cache.row_count)
        ):
            raise ValueError("scoped embedding view row authority is invalid")
        if _authenticated_cache_identity(shared_cache) != dict(logical_identity):
            raise ValueError(
                "scoped embedding view logical identity differs from shared cache"
            )
        self._cache = shared_cache
        self.cache_dir = shared_cache.cache_dir
        self._logical_identity = copy.deepcopy(dict(logical_identity))
        self._allowed_row_ids = frozenset(allowed)
        self._allowed_row_order = allowed
        self.shared_reference_content_sha256 = _require_sha256(
            shared_reference_content_sha256,
            label="scoped embedding view shared-reference SHA",
        )
        self._metadata = shared_cache._metadata
        self._embeddings = shared_cache._embeddings
        self._offsets = shared_cache._offsets

    @property
    def row_count(self) -> int:
        return int(self._cache.row_count)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._metadata)

    @property
    def allowed_row_ids(self) -> tuple[int, ...]:
        return self._allowed_row_order

    def identity(self) -> Mapping[str, Any]:
        if _authenticated_cache_identity(self._cache) != self._logical_identity:
            raise RuntimeError("shared embedding cache identity changed during use")
        return copy.deepcopy(self._logical_identity)

    def authenticated_snapshot_identity(self) -> Mapping[str, Any]:
        return self.identity()

    def _cached_chunks(self, row_id: int) -> tuple[str, ...]:
        value = int(row_id)
        if value not in self._allowed_row_ids:
            raise ValueError(
                "scoped embedding view refuses a non-fit row"
            )
        return self._cache._cached_chunks(value)

    def bind_spent(
        self,
        row_ids: Sequence[int],
        texts: Sequence[str],
    ) -> BoundSpentFrozenChunkEmbeddingProvider:
        requested = tuple(map(int, row_ids))
        if not set(requested).issubset(self._allowed_row_ids):
            raise ValueError(
                "scoped embedding view refuses a non-fit row"
            )
        physical = self._cache.bind_spent(
            requested,
            tuple(texts),
        )
        return BoundSpentFrozenChunkEmbeddingProvider(
            cache=self,
            row_ids=physical.row_ids,
            cached_by_row=physical.cached_by_row,
            token_bounded_row_ids=physical.token_bounded_row_ids,
        )


def _private_config_payload(
    *,
    config: AppliedInferenceConfig,
    forbidden_paths: Sequence[Path],
) -> dict[str, Any]:
    # Runtime receives the physical inputs as separately authenticated
    # capabilities.  Keeping neutral URIs here makes the scientific
    # configuration independent of an attempt/recovery location.
    modeling_path = "production://private-preflight/fit-only-modeling-v1"
    cache_path = LOGICAL_FROZEN_EMBEDDING_CACHE_URI

    def rewrite(value: Any, *, key: str | None = None) -> Any:
        if key == "dataset_path":
            return modeling_path
        if key == "cache_dir":
            return cache_path
        if key == "external_corpus_cache_dirs":
            return []
        if isinstance(value, Mapping):
            return {
                str(child_key): rewrite(child_value, key=str(child_key))
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [rewrite(child) for child in value]
        return copy.deepcopy(value)

    payload = rewrite(production_stage1_effective_config_payload(config))
    serialized = _canonical_json(payload)
    forbidden = tuple(str(path.resolve(strict=False)) for path in forbidden_paths)
    if any(value in serialized for value in forbidden):
        raise ValueError("preflight scope configuration exposes a prepared cohort or global cache")
    return payload


@dataclass(frozen=True)
class AuthenticatedPreflightScopeInput:
    root: Path
    manifest: Mapping[str, Any]
    modeling_data: pd.DataFrame
    config: AppliedInferenceConfig
    scope_authority: Mapping[str, Any]
    scope: Mapping[str, Any]
    embedding_cache: ScopedEmbeddingView
    shared_cache_reference_path: Path
    shared_cache_reference: Mapping[str, Any]
    semantic_witness_scientific_config: Any

    @property
    def manifest_path(self) -> Path:
        return self.root / PREFLIGHT_SCOPE_INPUT_MANIFEST

    @property
    def scope_id(self) -> str:
        return str(self.scope["scope_id"])

    def worker_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "production_stage1_preflight_worker_payload_v2",
            "scope_id": self.scope_id,
            "manifest_path": str(self.manifest_path),
            "manifest_content_sha256": str(self.manifest["content_sha256"]),
            "shared_cache_reference_path": str(
                self.shared_cache_reference_path
            ),
            "shared_cache_reference_content_sha256": str(
                self.shared_cache_reference["content_sha256"]
            ),
        }


@dataclass(frozen=True)
class AuthenticatedPreflightScopeInputSet:
    root: Path
    manifest: Mapping[str, Any]
    scopes: Mapping[str, AuthenticatedPreflightScopeInput]
    shared_cache_reference_path: Path
    shared_cache_reference: Mapping[str, Any]

    def worker_payloads(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(scope.worker_payload() for scope in self.scopes.values())

    def identity(self) -> dict[str, Any]:
        manifest_registration = _file_registration(
            self.root / PREFLIGHT_SCOPE_INPUT_SET_MANIFEST,
            self.root,
        )
        attempt_root = self.root.parent / f".{self.root.name}.scope_attempts"
        attempts = (
            sorted(entry.name for entry in os.scandir(attempt_root))
            if attempt_root.is_dir() and not attempt_root.is_symlink()
            else []
        )
        body = {
            "schema_version": "production_stage1_preflight_scope_input_set_identity_v2",
            "root": str(self.root),
            "manifest_path": str(self.root / PREFLIGHT_SCOPE_INPUT_SET_MANIFEST),
            "manifest": manifest_registration,
            "manifest_content_sha256": str(self.manifest["content_sha256"]),
            "scope_order": list(self.scopes),
            "scope_manifest_content_sha256": {
                scope_id: str(scope.manifest["content_sha256"])
                for scope_id, scope in self.scopes.items()
            },
            "shared_cache_reference": _file_registration(
                self.shared_cache_reference_path,
                self.root,
            ),
            "shared_cache_reference_content_sha256": str(
                self.shared_cache_reference["content_sha256"]
            ),
            "per_scope_embedding_arrays_copied": False,
            "per_scope_chunk_texts_copied": False,
            "attempt_root": str(attempt_root),
            "preserved_incomplete_attempts": attempts,
            "scope_inputs_outside_terminal_scientific_artifact": True,
        }
        return {**body, "content_sha256": _sha256_json(body)}


def _write_scope(
    *,
    root: Path,
    modeling_data: pd.DataFrame,
    config: AppliedInferenceConfig,
    embedding_cache_identity: Mapping[str, Any],
    shared_cache_reference_content_sha256: str,
    registry_content_sha256: str,
    scope: Mapping[str, Any],
    forbidden_paths: Sequence[Path],
    semantic_witness_scientific_config: Any,
) -> None:
    root.mkdir(parents=True, exist_ok=False)
    fit_rows = tuple(map(int, _scope_value(scope, "fit_row_ids")))
    row_count = len(modeling_data)
    if not fit_rows or min(fit_rows) < 0 or max(fit_rows) >= row_count:
        raise ValueError("preflight scope fit rows are invalid")
    private = pd.DataFrame(
        {
            config.text_column: np.full(row_count, "", dtype=object),
            config.treatment_column: np.full(row_count, np.nan, dtype=float),
            config.outcome_column: np.full(row_count, np.nan, dtype=float),
        }
    )
    private.loc[
        list(fit_rows),
        [
            config.text_column,
            config.treatment_column,
            config.outcome_column,
        ],
    ] = modeling_data.iloc[list(fit_rows)][
        [
            config.text_column,
            config.treatment_column,
            config.outcome_column,
        ]
    ].to_numpy(
        copy=True
    )
    _write_parquet(root / _MODELING_FILE, private)
    _write_json(
        root / _CONFIG_FILE,
        _private_config_payload(
            config=config,
            forbidden_paths=forbidden_paths,
        ),
    )
    from .review_spent_evidence_provider import (
        SemanticWitnessScientificConfig,
    )

    if (
        type(semantic_witness_scientific_config)
        is not SemanticWitnessScientificConfig
    ):
        raise TypeError(
            "preflight scope input requires one closed semantic-witness "
            "scientific config"
        )
    _write_json(
        root / _SEMANTIC_WITNESS_CONFIG_FILE,
        semantic_witness_scientific_config.as_dict(),
    )
    authority_body = {
        "schema_version": PREFLIGHT_ONE_SCOPE_AUTHORITY_SCHEMA,
        "registry_content_sha256": registry_content_sha256,
        "dataset_row_count": row_count,
        "scope": copy.deepcopy(dict(scope)),
        "scope_binding_sha256": _sha256_json(
            {
                "registry_content_sha256": registry_content_sha256,
                "scope": scope,
            }
        ),
        "authorized_scope_count": 1,
        "other_scope_definitions_supplied": False,
        "other_scope_row_identities_supplied": False,
    }
    _write_json(
        root / _SCOPE_AUTHORITY_FILE,
        {
            **authority_body,
            "content_sha256": _sha256_json(authority_body),
        },
    )
    files = {
        "effective_config": _file_registration(root / _CONFIG_FILE, root),
        "semantic_witness_scientific_config": _file_registration(
            root / _SEMANTIC_WITNESS_CONFIG_FILE,
            root,
        ),
        "one_scope_authority": _file_registration(
            root / _SCOPE_AUTHORITY_FILE,
            root,
        ),
        "fit_only_modeling": _file_registration(root / _MODELING_FILE, root),
    }
    cache_view = {
        "schema_version": PREFLIGHT_SCOPED_CACHE_VIEW_SCHEMA,
        "shared_cache_reference_content_sha256": _require_sha256(
            shared_cache_reference_content_sha256,
            label="shared preflight cache reference SHA",
        ),
        "logical_identity": json.loads(
            _canonical_json(dict(embedding_cache_identity))
        ),
        "logical_identity_sha256": _sha256_json(embedding_cache_identity),
        "allowed_row_ids": list(fit_rows),
        "allowed_row_order_sha256": _sha256_json(list(fit_rows)),
        "allowed_row_count": len(fit_rows),
        "peer_row_access_allowed": False,
        "embedding_array_payload_count": 0,
        "chunk_text_payload_count": 0,
    }
    body = {
        "schema_version": PREFLIGHT_SCOPE_INPUT_SCHEMA,
        "scope": copy.deepcopy(dict(scope)),
        "scope_binding_sha256": _sha256_json(
            {
                "registry_content_sha256": registry_content_sha256,
                "scope": scope,
            }
        ),
        "registry_content_sha256": registry_content_sha256,
        "row_count": row_count,
        "columns": [
            config.text_column,
            config.treatment_column,
            config.outcome_column,
        ],
        "files": files,
        "embedding_cache_view": cache_view,
        "semantic_witness_scientific_config_sha256": (
            semantic_witness_scientific_config.identity_sha256
        ),
        "nonfit_text_supplied": False,
        "nonfit_labels_supplied": False,
        "global_cache_path_supplied": False,
        "source_dataset_path_supplied": False,
    }
    _write_json(
        root / PREFLIGHT_SCOPE_INPUT_MANIFEST,
        {**body, "content_sha256": _sha256_json(body)},
    )


def publish_preflight_scope_inputs(
    *,
    output_root: Path | str,
    modeling_data: pd.DataFrame,
    config: AppliedInferenceConfig,
    embedding_cache: Any,
    embedding_cache_identity: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_content_sha256: str,
    scopes: Sequence[Mapping[str, Any]],
    source_dataset_path: Path,
    global_embedding_cache_path: Path,
    semantic_witness_scientific_config: Any,
) -> AuthenticatedPreflightScopeInputSet:
    """Recoverably publish one fit-only capability per canonical scope."""

    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("preflight scope-input root must be absolute")
    canonical_scopes = tuple(json.loads(_canonical_json(dict(scope))) for scope in scopes)
    scope_ids = [str(scope.get("scope_id") or "") for scope in canonical_scopes]
    if (
        not scope_ids
        or any(not value for value in scope_ids)
        or len(scope_ids) != len(set(scope_ids))
    ):
        raise ValueError("preflight scope IDs must be unique and nonempty")
    if _sha256_json(registry) != str(registry_content_sha256):
        raise ValueError("preflight parent registry differs from its content identity")
    shared_reference = _build_shared_cache_reference(
        embedding_cache=embedding_cache,
        embedding_cache_identity=embedding_cache_identity,
        global_embedding_cache_path=global_embedding_cache_path,
    )
    shared_reference_path = root / PREFLIGHT_SHARED_CACHE_REFERENCE
    terminal_manifest = root / PREFLIGHT_SCOPE_INPUT_SET_MANIFEST
    if terminal_manifest.is_file():
        return validate_preflight_scope_input_set(
            root=root,
            expected_scopes=canonical_scopes,
            expected_registry_content_sha256=registry_content_sha256,
            parent_modeling_data=modeling_data,
            parent_config=config,
            parent_embedding_cache=embedding_cache,
            parent_embedding_cache_identity=embedding_cache_identity,
            expected_shared_cache_reference=shared_reference,
            expected_semantic_witness_scientific_config=(
                semantic_witness_scientific_config
            ),
            forbidden_paths=(source_dataset_path, global_embedding_cache_path),
        )
    if root.is_symlink():
        raise ValueError("preflight scope-input root cannot be a symlink")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=True)
    if root.resolve(strict=True) != root:
        raise ValueError("preflight scope-input root is not canonical")
    allowed_entries = {
        *scope_ids,
        PREFLIGHT_SHARED_CACHE_REFERENCE,
    }
    observed_entries = {entry.name for entry in os.scandir(root)}
    if not observed_entries.issubset(allowed_entries):
        raise ValueError("incomplete preflight scope-input root contains unknown entries")
    if shared_reference_path.exists():
        _validate_shared_cache_reference(
            path=shared_reference_path,
            expected_content_sha256=str(shared_reference["content_sha256"]),
            expected_reference=shared_reference,
        )
    else:
        _write_json(shared_reference_path, shared_reference)
    attempt_root = root.parent / f".{root.name}.scope_attempts"
    if attempt_root.is_symlink():
        raise ValueError("preflight scope-input attempt root cannot be a symlink")
    attempt_root.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for scope in canonical_scopes:
        scope_id = str(scope["scope_id"])
        scope_root = root / scope_id
        if scope_root.exists():
            completed = validate_preflight_scope_input(
                manifest_path=scope_root / PREFLIGHT_SCOPE_INPUT_MANIFEST,
                expected_scope_id=scope_id,
                expected_registry_content_sha256=registry_content_sha256,
                parent_modeling_data=modeling_data,
                parent_config=config,
                parent_embedding_cache=embedding_cache,
                parent_embedding_cache_identity=embedding_cache_identity,
                shared_cache_reference_path=shared_reference_path,
                expected_shared_cache_reference_content_sha256=str(
                    shared_reference["content_sha256"]
                ),
                expected_semantic_witness_scientific_config=(
                    semantic_witness_scientific_config
                ),
                forbidden_paths=(
                    source_dataset_path,
                    global_embedding_cache_path,
                ),
            )
            if completed.scope != scope:
                raise ValueError("completed preflight scope input belongs to another scope")
        else:
            attempt = Path(
                tempfile.mkdtemp(
                    prefix=f"{scope_id}.attempt-",
                    dir=attempt_root,
                )
            )
            temporary = attempt / "scope_input"
            _write_scope(
                root=temporary,
                modeling_data=modeling_data,
                config=config,
                embedding_cache_identity=embedding_cache_identity,
                shared_cache_reference_content_sha256=str(
                    shared_reference["content_sha256"]
                ),
                registry_content_sha256=registry_content_sha256,
                scope=scope,
                forbidden_paths=(source_dataset_path, global_embedding_cache_path),
                semantic_witness_scientific_config=(
                    semantic_witness_scientific_config
                ),
            )
            completed = validate_preflight_scope_input(
                manifest_path=temporary / PREFLIGHT_SCOPE_INPUT_MANIFEST,
                expected_scope_id=scope_id,
                expected_registry_content_sha256=registry_content_sha256,
                parent_modeling_data=modeling_data,
                parent_config=config,
                parent_embedding_cache=embedding_cache,
                parent_embedding_cache_identity=embedding_cache_identity,
                shared_cache_reference_path=shared_reference_path,
                expected_shared_cache_reference_content_sha256=str(
                    shared_reference["content_sha256"]
                ),
                expected_semantic_witness_scientific_config=(
                    semantic_witness_scientific_config
                ),
                forbidden_paths=(
                    source_dataset_path,
                    global_embedding_cache_path,
                ),
            )
            if completed.scope != scope:
                raise ValueError("new preflight scope input belongs to another scope")
            os.replace(temporary, scope_root)
            attempt.rmdir()
            descriptor = os.open(
                root,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        rows.append(
            {
                "scope_id": scope_id,
                "manifest": _file_registration(
                    scope_root / PREFLIGHT_SCOPE_INPUT_MANIFEST,
                    root,
                ),
            }
        )
    body = {
        "schema_version": PREFLIGHT_SCOPE_INPUT_SET_SCHEMA,
        "registry_content_sha256": registry_content_sha256,
        "scope_order": scope_ids,
        "scope_count": len(scope_ids),
        "scopes": rows,
        "shared_embedding_cache_reference": _file_registration(
            shared_reference_path,
            root,
        ),
        "shared_embedding_cache_reference_content_sha256": str(
            shared_reference["content_sha256"]
        ),
        "one_scope_per_worker_payload": True,
        "one_physical_cache_shared_across_scopes": True,
        "per_scope_embedding_arrays_copied": False,
        "per_scope_chunk_texts_copied": False,
    }
    _write_json(
        terminal_manifest,
        {**body, "content_sha256": _sha256_json(body)},
    )
    descriptor = os.open(
        root.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return validate_preflight_scope_input_set(
        root=root,
        expected_scopes=canonical_scopes,
        expected_registry_content_sha256=registry_content_sha256,
        parent_modeling_data=modeling_data,
        parent_config=config,
        parent_embedding_cache=embedding_cache,
        parent_embedding_cache_identity=embedding_cache_identity,
        expected_shared_cache_reference=shared_reference,
        expected_semantic_witness_scientific_config=(
            semantic_witness_scientific_config
        ),
        forbidden_paths=(source_dataset_path, global_embedding_cache_path),
    )


def validate_preflight_scope_input(
    *,
    manifest_path: Path | str,
    expected_scope_id: str,
    expected_manifest_content_sha256: str | None = None,
    expected_registry_content_sha256: str | None = None,
    parent_modeling_data: pd.DataFrame | None = None,
    parent_config: AppliedInferenceConfig | None = None,
    parent_embedding_cache: Any | None = None,
    parent_embedding_cache_identity: Mapping[str, Any] | None = None,
    shared_cache_reference_path: Path | str | None = None,
    expected_shared_cache_reference_content_sha256: str | None = None,
    expected_semantic_witness_scientific_config: Any | None = None,
    forbidden_paths: Sequence[Path] = (),
) -> AuthenticatedPreflightScopeInput:
    path = Path(manifest_path).absolute()
    root = path.parent
    if (
        path.name != PREFLIGHT_SCOPE_INPUT_MANIFEST
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
    ):
        raise ValueError("preflight scope-input manifest path is invalid")
    manifest = _read_json(path, label="preflight scope-input manifest")
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    required = {
        "schema_version",
        "scope",
        "scope_binding_sha256",
        "registry_content_sha256",
        "row_count",
        "columns",
        "files",
        "embedding_cache_view",
        "semantic_witness_scientific_config_sha256",
        "nonfit_text_supplied",
        "nonfit_labels_supplied",
        "global_cache_path_supplied",
        "source_dataset_path_supplied",
        "content_sha256",
    }
    scope = manifest.get("scope")
    if (
        set(manifest) != required
        or manifest.get("schema_version") != PREFLIGHT_SCOPE_INPUT_SCHEMA
        or not isinstance(scope, Mapping)
        or scope.get("scope_id") != expected_scope_id
        or manifest.get("content_sha256") != _sha256_json(body)
        or (
            expected_manifest_content_sha256 is not None
            and manifest.get("content_sha256") != expected_manifest_content_sha256
        )
        or manifest.get("nonfit_text_supplied") is not False
        or manifest.get("nonfit_labels_supplied") is not False
        or manifest.get("global_cache_path_supplied") is not False
        or manifest.get("source_dataset_path_supplied") is not False
    ):
        raise ValueError("preflight scope-input manifest is invalid")
    _require_sha256(
        manifest.get("content_sha256"),
        label="preflight scope-input content_sha256",
    )
    registry_sha = _require_sha256(
        manifest.get("registry_content_sha256"),
        label="preflight scope-input registry SHA",
    )
    if (
        expected_registry_content_sha256 is not None
        and registry_sha != expected_registry_content_sha256
    ):
        raise ValueError("preflight scope-input registry changed")
    if manifest.get("scope_binding_sha256") != _sha256_json(
        {"registry_content_sha256": registry_sha, "scope": scope}
    ):
        raise ValueError("preflight scope-input binding changed")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "effective_config",
        "semantic_witness_scientific_config",
        "one_scope_authority",
        "fit_only_modeling",
    }:
        raise ValueError("preflight scope-input files are incomplete")
    paths = {
        key: _validate_registration(root, registration, label=key)
        for key, registration in files.items()
    }
    columns = manifest.get("columns")
    if (
        not isinstance(columns, list)
        or len(columns) != 3
        or any(not isinstance(value, str) or not value for value in columns)
        or len(set(columns)) != 3
    ):
        raise ValueError("preflight scope-input columns are invalid")
    config_payload = _read_json(paths["effective_config"], label="preflight config")
    config = ExperimentConfig.from_dict({"applied_inference": config_payload}).applied_inference
    raw_embedding = (
        (config_payload.get("architecture") or {}).get("multi_model_forest") or {}
    ).get("embedding_contrast")
    if not isinstance(raw_embedding, Mapping):
        raise ValueError("preflight scope-input config lacks its embedding configuration")
    # The production wrapper already validated this effective configuration.
    # Restore its exact embedding block after the legacy config constructor's
    # compatibility normalization, which can otherwise disable it.
    restored_embedding = EmbeddingContrastDiscoveryConfig(**raw_embedding)
    config.architecture.multi_model_forest.embedding_contrast = restored_embedding
    config.architecture.multi_model_agentic_forest.embedding_contrast = copy.deepcopy(
        restored_embedding
    )
    from .review_spent_evidence_provider import (
        SemanticWitnessScientificConfig,
    )

    semantic_witness_scientific_config = (
        SemanticWitnessScientificConfig.from_mapping(
            _read_json(
                paths["semantic_witness_scientific_config"],
                label="preflight semantic-witness scientific config",
            ),
            label="preflight semantic-witness scientific config",
        )
    )
    if (
        manifest.get("semantic_witness_scientific_config_sha256")
        != semantic_witness_scientific_config.identity_sha256
    ):
        raise ValueError(
            "preflight semantic-witness scientific config identity changed"
        )
    if expected_semantic_witness_scientific_config is not None:
        if (
            type(expected_semantic_witness_scientific_config)
            is not SemanticWitnessScientificConfig
            or expected_semantic_witness_scientific_config.as_dict()
            != semantic_witness_scientific_config.as_dict()
        ):
            raise ValueError(
                "preflight semantic-witness scientific config differs from "
                "its parent request"
            )
    if columns != [
        config.text_column,
        config.treatment_column,
        config.outcome_column,
    ]:
        raise ValueError("preflight scope-input config columns changed")
    authority = _read_json(
        paths["one_scope_authority"],
        label="preflight one-scope authority",
    )
    authority_body = {
        key: copy.deepcopy(value) for key, value in authority.items() if key != "content_sha256"
    }
    authority_fields = {
        "schema_version",
        "registry_content_sha256",
        "dataset_row_count",
        "scope",
        "scope_binding_sha256",
        "authorized_scope_count",
        "other_scope_definitions_supplied",
        "other_scope_row_identities_supplied",
        "content_sha256",
    }
    if (
        set(authority) != authority_fields
        or authority.get("schema_version") != PREFLIGHT_ONE_SCOPE_AUTHORITY_SCHEMA
        or authority.get("registry_content_sha256") != registry_sha
        or authority.get("scope") != scope
        or authority.get("scope_binding_sha256") != manifest.get("scope_binding_sha256")
        or authority.get("authorized_scope_count") != 1
        or authority.get("other_scope_definitions_supplied") is not False
        or authority.get("other_scope_row_identities_supplied") is not False
        or authority.get("content_sha256") != _sha256_json(authority_body)
    ):
        raise ValueError("preflight one-scope authority changed")
    modeling = _read_exact_parquet(
        paths["fit_only_modeling"],
        expected_columns=columns,
        label="preflight fit-only modeling data",
    )
    row_count = int(manifest["row_count"])
    if authority.get("dataset_row_count") != row_count:
        raise ValueError("preflight one-scope authority row count changed")
    fit_rows = tuple(map(int, scope.get("fit_row_ids") or ()))
    if len(modeling) != row_count or not fit_rows:
        raise ValueError("preflight scope-input row coverage changed")
    nonfit = sorted(set(range(row_count)) - set(fit_rows))
    if (
        not bool(
            modeling.iloc[list(fit_rows)][config.text_column]
            .map(lambda value: isinstance(value, str) and bool(value))
            .all()
        )
        or modeling.iloc[list(fit_rows)][[config.treatment_column, config.outcome_column]]
        .isna()
        .any()
        .any()
        or modeling.iloc[nonfit][config.text_column].map(bool).any()
        or modeling.iloc[nonfit][[config.treatment_column, config.outcome_column]]
        .notna()
        .any()
        .any()
    ):
        raise ValueError("preflight scope-input contains nonfit data or missing fit data")
    cache_view = manifest.get("embedding_cache_view")
    view_fields = {
        "schema_version",
        "shared_cache_reference_content_sha256",
        "logical_identity",
        "logical_identity_sha256",
        "allowed_row_ids",
        "allowed_row_order_sha256",
        "allowed_row_count",
        "peer_row_access_allowed",
        "embedding_array_payload_count",
        "chunk_text_payload_count",
    }
    if (
        not isinstance(cache_view, Mapping)
        or set(cache_view) != view_fields
        or cache_view.get("schema_version")
        != PREFLIGHT_SCOPED_CACHE_VIEW_SCHEMA
        or cache_view.get("allowed_row_ids") != list(fit_rows)
        or cache_view.get("allowed_row_order_sha256")
        != _sha256_json(list(fit_rows))
        or cache_view.get("allowed_row_count") != len(fit_rows)
        or cache_view.get("peer_row_access_allowed") is not False
        or cache_view.get("embedding_array_payload_count") != 0
        or cache_view.get("chunk_text_payload_count") != 0
        or not isinstance(cache_view.get("logical_identity"), Mapping)
        or cache_view.get("logical_identity_sha256")
        != _sha256_json(cache_view["logical_identity"])
    ):
        raise ValueError("preflight scoped embedding-cache view is invalid")
    reference_sha = _require_sha256(
        cache_view["shared_cache_reference_content_sha256"],
        label="preflight scoped shared-cache reference SHA",
    )
    if (
        expected_shared_cache_reference_content_sha256 is not None
        and reference_sha
        != expected_shared_cache_reference_content_sha256
    ):
        raise ValueError(
            "preflight scoped embedding-cache reference changed"
        )
    reference_path = (
        root.parent / PREFLIGHT_SHARED_CACHE_REFERENCE
        if shared_cache_reference_path is None
        else Path(shared_cache_reference_path).absolute()
    )
    shared_reference = _validate_shared_cache_reference(
        path=reference_path,
        expected_content_sha256=reference_sha,
    )
    if (
        shared_reference["logical_identity"]
        != cache_view["logical_identity"]
        or int(shared_reference["logical_identity"].get("row_count", -1))
        != row_count
    ):
        raise ValueError(
            "preflight scoped view differs from its shared embedding cache"
        )
    if parent_embedding_cache_identity is not None and (
        cache_view["logical_identity"]
        != dict(parent_embedding_cache_identity)
    ):
        raise ValueError(
            "preflight scoped view logical identity changed"
        )
    if parent_embedding_cache is None:
        shared_cache = _load_shared_cache(shared_reference)
    else:
        if (
            Path(parent_embedding_cache.cache_dir).resolve(strict=True)
            != Path(shared_reference["cache_dir"])
            or _authenticated_cache_identity(parent_embedding_cache)
            != shared_reference["logical_identity"]
        ):
            raise ValueError(
                "preflight shared cache differs from its parent handle"
            )
        shared_cache = parent_embedding_cache
    cache = ScopedEmbeddingView(
        shared_cache=shared_cache,
        logical_identity=cache_view["logical_identity"],
        allowed_row_ids=fit_rows,
        shared_reference_content_sha256=reference_sha,
    )
    expected_files = {
        PREFLIGHT_SCOPE_INPUT_MANIFEST,
        *(str(value["relative_path"]) for value in files.values()),
    }
    observed_files, observed_directories = _closed_tree_inventory(
        root,
        label="preflight scope input",
    )
    expected_directories = {
        Path(value).parent.as_posix()
        for value in expected_files
        if Path(value).parent.as_posix() != "."
    }
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("preflight scope input contains unregistered entries")
    if forbidden_paths:
        serialized = b"".join(
            (root / relative).read_bytes()
            for relative in sorted(observed_files)
            if not relative.endswith((".npy", ".parquet"))
        )
        for forbidden in forbidden_paths:
            if str(forbidden.resolve(strict=False)).encode("utf-8") in serialized:
                raise ValueError("preflight scope input exposes a forbidden path")
    if parent_modeling_data is not None:
        if parent_config is None:
            raise ValueError("parent config is required with parent modeling data")
        expected = parent_modeling_data.iloc[list(fit_rows)][columns]
        actual = modeling.iloc[list(fit_rows)][columns]
        if actual.to_dict("records") != expected.to_dict("records"):
            raise ValueError("preflight scope input differs from parent fit rows")
    return AuthenticatedPreflightScopeInput(
        root=root,
        manifest=copy.deepcopy(manifest),
        modeling_data=modeling,
        config=config,
        scope_authority=authority,
        scope=copy.deepcopy(dict(scope)),
        embedding_cache=cache,
        shared_cache_reference_path=reference_path,
        shared_cache_reference=shared_reference,
        semantic_witness_scientific_config=(
            semantic_witness_scientific_config
        ),
    )


def validate_preflight_scope_input_set(
    *,
    root: Path | str,
    expected_scopes: Sequence[Mapping[str, Any]],
    expected_registry_content_sha256: str,
    parent_modeling_data: pd.DataFrame | None = None,
    parent_config: AppliedInferenceConfig | None = None,
    parent_embedding_cache: Any | None = None,
    parent_embedding_cache_identity: Mapping[str, Any] | None = None,
    expected_shared_cache_reference: Mapping[str, Any] | None = None,
    expected_semantic_witness_scientific_config: Any | None = None,
    forbidden_paths: Sequence[Path] = (),
) -> AuthenticatedPreflightScopeInputSet:
    set_root = Path(root).absolute()
    if set_root.is_symlink() or not set_root.is_dir() or set_root.resolve(strict=True) != set_root:
        raise ValueError("preflight scope-input set root is invalid")
    manifest = _read_json(
        set_root / PREFLIGHT_SCOPE_INPUT_SET_MANIFEST,
        label="preflight scope-input set manifest",
    )
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    required = {
        "schema_version",
        "registry_content_sha256",
        "scope_order",
        "scope_count",
        "scopes",
        "shared_embedding_cache_reference",
        "shared_embedding_cache_reference_content_sha256",
        "one_scope_per_worker_payload",
        "one_physical_cache_shared_across_scopes",
        "per_scope_embedding_arrays_copied",
        "per_scope_chunk_texts_copied",
        "content_sha256",
    }
    expected = tuple(json.loads(_canonical_json(dict(scope))) for scope in expected_scopes)
    expected_order = [str(scope["scope_id"]) for scope in expected]
    rows = manifest.get("scopes")
    if (
        set(manifest) != required
        or manifest.get("schema_version") != PREFLIGHT_SCOPE_INPUT_SET_SCHEMA
        or manifest.get("registry_content_sha256") != expected_registry_content_sha256
        or manifest.get("scope_order") != expected_order
        or manifest.get("scope_count") != len(expected)
        or manifest.get("one_scope_per_worker_payload") is not True
        or manifest.get("one_physical_cache_shared_across_scopes") is not True
        or manifest.get("per_scope_embedding_arrays_copied") is not False
        or manifest.get("per_scope_chunk_texts_copied") is not False
        or manifest.get("content_sha256") != _sha256_json(body)
        or not isinstance(rows, list)
        or len(rows) != len(expected)
    ):
        raise ValueError("preflight scope-input set manifest is invalid")
    shared_reference_path = _validate_registration(
        set_root,
        manifest["shared_embedding_cache_reference"],
        label="shared preflight cache reference",
    )
    if shared_reference_path.name != PREFLIGHT_SHARED_CACHE_REFERENCE:
        raise ValueError("shared preflight cache reference layout changed")
    shared_reference = _validate_shared_cache_reference(
        path=shared_reference_path,
        expected_content_sha256=str(
            manifest["shared_embedding_cache_reference_content_sha256"]
        ),
        expected_reference=expected_shared_cache_reference,
    )
    authenticated: dict[str, AuthenticatedPreflightScopeInput] = {}
    for scope, row in zip(expected, rows):
        scope_id = str(scope["scope_id"])
        if (
            not isinstance(row, Mapping)
            or set(row) != {"scope_id", "manifest"}
            or row.get("scope_id") != scope_id
        ):
            raise ValueError("preflight scope-input set row changed")
        child_manifest = _validate_registration(
            set_root,
            row["manifest"],
            label=f"{scope_id} preflight manifest",
        )
        child = validate_preflight_scope_input(
            manifest_path=child_manifest,
            expected_scope_id=scope_id,
            expected_registry_content_sha256=expected_registry_content_sha256,
            parent_modeling_data=parent_modeling_data,
            parent_config=parent_config,
            parent_embedding_cache=parent_embedding_cache,
            parent_embedding_cache_identity=parent_embedding_cache_identity,
            shared_cache_reference_path=shared_reference_path,
            expected_shared_cache_reference_content_sha256=str(
                shared_reference["content_sha256"]
            ),
            expected_semantic_witness_scientific_config=(
                expected_semantic_witness_scientific_config
            ),
            forbidden_paths=forbidden_paths,
        )
        if child.scope != scope:
            raise ValueError("preflight scope-input set scope changed")
        authenticated[scope_id] = child
    expected_files = {
        PREFLIGHT_SCOPE_INPUT_SET_MANIFEST,
        PREFLIGHT_SHARED_CACHE_REFERENCE,
    }
    expected_directories: set[str] = set()
    for scope_id, child in authenticated.items():
        child_files, child_directories = _closed_tree_inventory(
            child.root,
            label=f"{scope_id} preflight scope input",
        )
        expected_directories.add(scope_id)
        expected_files.update(f"{scope_id}/{relative}" for relative in child_files)
        expected_directories.update(f"{scope_id}/{relative}" for relative in child_directories)
    observed_files, observed_directories = _closed_tree_inventory(
        set_root,
        label="preflight scope-input set",
    )
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("preflight scope-input set contains unregistered entries")
    return AuthenticatedPreflightScopeInputSet(
        root=set_root,
        manifest=copy.deepcopy(manifest),
        scopes=authenticated,
        shared_cache_reference_path=shared_reference_path,
        shared_cache_reference=shared_reference,
    )


__all__ = [
    "AuthenticatedPreflightScopeInput",
    "AuthenticatedPreflightScopeInputSet",
    "PREFLIGHT_ONE_SCOPE_AUTHORITY_SCHEMA",
    "PREFLIGHT_SCOPED_CACHE_VIEW_SCHEMA",
    "PREFLIGHT_SHARED_CACHE_REFERENCE",
    "PREFLIGHT_SHARED_CACHE_REFERENCE_SCHEMA",
    "PREFLIGHT_SCOPE_INPUT_MANIFEST",
    "PREFLIGHT_SCOPE_INPUT_SET_MANIFEST",
    "ScopedEmbeddingView",
    "publish_preflight_scope_inputs",
    "validate_preflight_scope_input",
    "validate_preflight_scope_input_set",
]
