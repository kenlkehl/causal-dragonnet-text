"""Row-restricted inputs for clustered-embedding preflight scopes.

The clustered preflight is label-dependent.  One authenticated text-only
cohort block is stored once, while each scope carries a compact ordered
treatment/outcome projection for only its fit rows.  Validation reconstructs a
fit-only worker frame and refuses every non-fit modeling or embedding row.
No process-global cache retains cohort labels, frames, or embedding providers.
"""

from __future__ import annotations

import copy
import hashlib
import io
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

PREFLIGHT_SCOPE_INPUT_SCHEMA = "production_stage1_preflight_scope_input_v8"
PREFLIGHT_SCOPE_INPUT_SET_SCHEMA = "production_stage1_preflight_scope_input_set_v8"
PREFLIGHT_ONE_SCOPE_AUTHORITY_SCHEMA = "production_stage1_preflight_one_scope_authority_v1"
PREFLIGHT_SHARED_CACHE_REFERENCE_SCHEMA = (
    "production_stage1_preflight_shared_embedding_cache_reference_v1"
)
PREFLIGHT_SCOPED_CACHE_VIEW_SCHEMA = (
    "production_stage1_preflight_scoped_embedding_cache_view_v2"
)
PREFLIGHT_SCOPED_CACHE_METADATA_SCHEMA = (
    "production_stage1_preflight_scoped_embedding_cache_metadata_v1"
)
PREFLIGHT_SAFE_CACHE_SCIENTIFIC_METADATA_SCHEMA = (
    "production_stage1_preflight_safe_cache_scientific_metadata_v1"
)
PREFLIGHT_EMBEDDING_ROW_BLOCK_SCHEMA = (
    "production_stage1_preflight_embedding_row_block_v1"
)
PREFLIGHT_EMBEDDING_ROW_STORE_SCHEMA = (
    "production_stage1_preflight_embedding_row_store_v1"
)
PREFLIGHT_SHARED_MODELING_REFERENCE_SCHEMA = (
    "production_stage1_preflight_shared_text_reference_v1"
)
PREFLIGHT_SHARED_MODELING_VIEW_SCHEMA = (
    "production_stage1_preflight_text_and_fit_labels_view_v1"
)
PREFLIGHT_SCOPE_LABEL_PROJECTION_SCHEMA = (
    "production_stage1_preflight_fit_label_projection_v1"
)
PREFLIGHT_SCOPE_INPUT_MANIFEST = "preflight_scope_input_manifest.json"
PREFLIGHT_SCOPE_INPUT_SET_MANIFEST = "preflight_scope_input_set_manifest.json"
PREFLIGHT_SHARED_CACHE_REFERENCE = "shared_embedding_cache_reference.json"
PREFLIGHT_SHARED_MODELING_REFERENCE = "shared_text_reference.json"
PREFLIGHT_EMBEDDING_ROW_STORE_MANIFEST = (
    "shared_embedding_row_store_manifest.json"
)

_CONFIG_FILE = "effective_config.json"
_SEMANTIC_WITNESS_CONFIG_FILE = "semantic_witness_scientific_config.json"
_SCOPE_AUTHORITY_FILE = "one_scope_authority.json"
_SHARED_MODELING_FILE = "shared_text.parquet"
_LABEL_PROJECTION_FILE = "fit_label_projection.parquet"
_EMBEDDING_ROW_DIRECTORY = "shared_embedding_rows"
_GLOBAL_ROW_ID_COLUMN = "__production_global_row_id"
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
_SHARED_CACHE_HANDLES: set[str] = set()
_SHARED_MODELING_HANDLES: dict[
    str,
    tuple[Path, tuple[int, ...], Mapping[str, Any]],
] = {}


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


def _validated_fit_rows(
    value: Any,
    *,
    row_count: int,
    label: str,
) -> tuple[int, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(
            isinstance(row_id, (bool, np.bool_))
            or not isinstance(row_id, (int, np.integer))
            for row_id in value
        )
    ):
        raise ValueError(f"{label} must be one nonempty integer row sequence")
    rows = tuple(int(row_id) for row_id in value)
    if len(rows) != len(set(rows)):
        raise ValueError(f"{label} contains duplicate global row IDs")
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, (int, np.integer))
        or int(row_count) < 1
        or min(rows) < 0
        or max(rows) >= int(row_count)
    ):
        raise ValueError(f"{label} contains an out-of-range global row ID")
    return rows


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


def _ordered_range_sha256(row_count: int) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    for row_id in range(int(row_count)):
        if row_id:
            digest.update(b",")
        digest.update(str(row_id).encode("ascii"))
    digest.update(b"]")
    return digest.hexdigest()


def _modeling_columns(config: AppliedInferenceConfig) -> list[str]:
    columns = [
        config.text_column,
        config.treatment_column,
        config.outcome_column,
    ]
    if (
        len(set(columns)) != len(columns)
        or _GLOBAL_ROW_ID_COLUMN in columns
    ):
        raise ValueError(
            "preflight modeling columns collide with each other or the "
            "global row-ID column"
        )
    return columns


def _shared_modeling_table(
    *,
    modeling_data: pd.DataFrame,
    text_column: str,
) -> pd.DataFrame:
    row_count = len(modeling_data)
    if row_count < 1:
        raise ValueError("shared preflight text block cannot be empty")
    if modeling_data.columns.tolist().count(text_column) != 1:
        raise ValueError(
            "shared preflight modeling source lacks its exact text column"
        )
    table = modeling_data.iloc[list(range(row_count))][[text_column]].copy(deep=True)
    table.reset_index(drop=True, inplace=True)
    table.insert(
        0,
        _GLOBAL_ROW_ID_COLUMN,
        np.arange(row_count, dtype=np.int64),
    )
    return table


def _validate_shared_modeling_frame(
    frame: pd.DataFrame,
    *,
    reference: Mapping[str, Any],
) -> None:
    row_count = reference.get("row_count")
    text_column = reference.get("text_column")
    stored_columns = reference.get("stored_columns")
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 1
        or not isinstance(text_column, str)
        or not text_column
        or text_column == _GLOBAL_ROW_ID_COLUMN
        or stored_columns != [_GLOBAL_ROW_ID_COLUMN, text_column]
        or list(frame.columns) != stored_columns
        or len(frame) != row_count
    ):
        raise ValueError("shared preflight text block shape changed")
    raw_row_ids = frame[_GLOBAL_ROW_ID_COLUMN].tolist()
    if any(
        isinstance(row_id, (bool, np.bool_))
        or not isinstance(row_id, (int, np.integer))
        for row_id in raw_row_ids
    ):
        raise ValueError(
            "shared preflight text block has a noninteger global row ID"
        )
    row_ids = tuple(int(row_id) for row_id in raw_row_ids)
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(
            "shared preflight text block has duplicate global row IDs"
        )
    if row_ids != tuple(range(row_count)):
        raise ValueError(
            "shared preflight text block global row order changed"
        )
    if frame[text_column].map(lambda value: not isinstance(value, str)).any():
        raise ValueError("shared preflight text block contains non-text values")


def _shared_modeling_values_match(
    *,
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    columns: Sequence[str],
) -> bool:
    return (
        len(observed) == len(expected)
        and observed[list(columns)].to_dict("records")
        == expected[list(columns)].to_dict("records")
    )


def _fit_modeling_content_sha256(
    *,
    modeling_data: pd.DataFrame,
    fit_rows: Sequence[int],
    columns: Sequence[str],
) -> str:
    selected = modeling_data.iloc[list(fit_rows)][list(columns)]
    rows: list[list[Any]] = []
    for row_id, values in zip(fit_rows, selected.itertuples(index=False, name=None)):
        text, treatment, outcome = values
        if not isinstance(text, str) or not text:
            raise ValueError(
                "preflight fit modeling identity requires nonempty text"
            )
        normalized_labels: list[int | float | bool] = []
        for value in (treatment, outcome):
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bool):
                normalized_labels.append(value)
            elif isinstance(value, int):
                normalized_labels.append(value)
            elif isinstance(value, float) and bool(np.isfinite(value)):
                normalized_labels.append(value)
            else:
                raise ValueError(
                    "preflight fit modeling identity requires finite "
                    "numeric labels"
                )
        rows.append(
            [
                int(row_id),
                text,
                normalized_labels[0],
                normalized_labels[1],
            ]
        )
    return _sha256_json(
        {
            "schema_version": "production_stage1_preflight_fit_modeling_identity_v1",
            "columns": [_GLOBAL_ROW_ID_COLUMN, *columns],
            "rows": rows,
        }
    )


def _fit_label_projection(
    *,
    modeling_data: pd.DataFrame,
    fit_rows: Sequence[int],
    columns: Sequence[str],
) -> pd.DataFrame:
    projection = (
        modeling_data.iloc[list(fit_rows)][list(columns[1:])]
        .reset_index(drop=True)
        .copy(deep=True)
    )
    projection.insert(
        0,
        _GLOBAL_ROW_ID_COLUMN,
        np.asarray(tuple(map(int, fit_rows)), dtype=np.int64),
    )
    return projection


def _build_shared_modeling_reference(
    *,
    block_path: Path,
    root: Path,
    modeling_data: pd.DataFrame,
    config: AppliedInferenceConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = _modeling_columns(config)
    expected = _shared_modeling_table(
        modeling_data=modeling_data,
        text_column=columns[0],
    )
    if block_path.exists() or block_path.is_symlink():
        if block_path.is_symlink() or not block_path.is_file():
            raise ValueError("shared preflight text block path is invalid")
        authenticated_state = _stat_identity(os.lstat(block_path))
        observed = _read_exact_parquet(
            block_path,
            expected_columns=[_GLOBAL_ROW_ID_COLUMN, columns[0]],
            label="shared preflight text block",
        )
        if not _shared_modeling_values_match(
            observed=observed,
            expected=expected,
            columns=[_GLOBAL_ROW_ID_COLUMN, columns[0]],
        ):
            raise ValueError(
                "existing shared preflight text block differs from "
                "the prepared cohort"
            )
    else:
        _write_parquet(block_path, expected)
        authenticated_state = _stat_identity(os.lstat(block_path))
        observed = expected
    registration = _file_registration(block_path, root)
    if _stat_identity(os.lstat(block_path)) != authenticated_state:
        raise RuntimeError(
            "shared preflight text block changed while publishing"
        )
    body = {
        "schema_version": PREFLIGHT_SHARED_MODELING_REFERENCE_SCHEMA,
        "text_block": registration,
        "row_count": len(expected),
        "global_row_id_column": _GLOBAL_ROW_ID_COLUMN,
        "text_column": columns[0],
        "stored_columns": [_GLOBAL_ROW_ID_COLUMN, columns[0]],
        "ordered_row_identity_sha256": _ordered_range_sha256(len(expected)),
        "complete_text_cohort_stored_once": True,
        "treatment_or_outcome_stored": False,
        "per_scope_text_payload_count": 0,
        "worker_api_restricts_nonfit_rows": True,
        "source_dataset_path_supplied": False,
    }
    reference = {**body, "content_sha256": _sha256_json(body)}
    _validate_shared_modeling_frame(observed, reference=reference)
    _SHARED_MODELING_HANDLES[str(reference["content_sha256"])] = (
        block_path,
        authenticated_state,
        copy.deepcopy(reference),
    )
    return reference, observed


def _registered_shared_modeling_path(
    *,
    root: Path,
    registration: Mapping[str, Any],
) -> Path:
    if (
        not isinstance(registration, Mapping)
        or set(registration) != {"relative_path", "sha256", "size_bytes"}
        or registration.get("relative_path") != _SHARED_MODELING_FILE
    ):
        raise ValueError(
            "shared preflight text-block registration is not closed"
        )
    _require_sha256(
        registration.get("sha256"),
        label="shared preflight text-block SHA",
    )
    size = registration.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(
            "shared preflight text-block size is invalid"
        )
    path = root / _SHARED_MODELING_FILE
    if path.is_symlink() or not path.is_file():
        raise ValueError("shared preflight text block is absent")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(
            "shared preflight text block escapes its artifact"
        ) from exc
    state = os.lstat(path)
    if (
        not stat.S_ISREG(state.st_mode)
        or int(state.st_nlink) != 1
        or int(state.st_size) != size
    ):
        raise ValueError("shared preflight text block stat changed")
    return path


def _validate_shared_modeling_reference(
    *,
    path: Path | str,
    expected_content_sha256: str | None = None,
    expected_reference: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    reference_path = Path(path).absolute()
    if (
        reference_path.name != PREFLIGHT_SHARED_MODELING_REFERENCE
        or reference_path.is_symlink()
        or not reference_path.is_file()
    ):
        raise ValueError("shared preflight modeling reference path is invalid")
    reference = _read_json(
        reference_path,
        label="shared preflight modeling reference",
    )
    required = {
        "schema_version",
        "text_block",
        "row_count",
        "global_row_id_column",
        "text_column",
        "stored_columns",
        "ordered_row_identity_sha256",
        "complete_text_cohort_stored_once",
        "treatment_or_outcome_stored",
        "per_scope_text_payload_count",
        "worker_api_restricts_nonfit_rows",
        "source_dataset_path_supplied",
        "content_sha256",
    }
    body = {
        key: copy.deepcopy(value)
        for key, value in reference.items()
        if key != "content_sha256"
    }
    row_count = reference.get("row_count")
    text_column = reference.get("text_column")
    if (
        set(reference) != required
        or reference.get("schema_version")
        != PREFLIGHT_SHARED_MODELING_REFERENCE_SCHEMA
        or reference.get("content_sha256") != _sha256_json(body)
        or (
            expected_content_sha256 is not None
            and reference.get("content_sha256")
            != expected_content_sha256
        )
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 1
        or not isinstance(text_column, str)
        or not text_column
        or text_column == _GLOBAL_ROW_ID_COLUMN
        or reference.get("global_row_id_column")
        != _GLOBAL_ROW_ID_COLUMN
        or reference.get("stored_columns")
        != [_GLOBAL_ROW_ID_COLUMN, text_column]
        or reference.get("ordered_row_identity_sha256")
        != _ordered_range_sha256(row_count)
        or reference.get("complete_text_cohort_stored_once") is not True
        or reference.get("treatment_or_outcome_stored") is not False
        or reference.get("per_scope_text_payload_count") != 0
        or reference.get("worker_api_restricts_nonfit_rows") is not True
        or reference.get("source_dataset_path_supplied") is not False
    ):
        raise ValueError("shared preflight modeling reference is invalid")
    content_sha = _require_sha256(
        reference.get("content_sha256"),
        label="shared preflight modeling-reference SHA",
    )
    if expected_reference is not None and reference != dict(expected_reference):
        raise ValueError(
            "existing shared preflight modeling reference differs from "
            "this request"
        )
    root = reference_path.parent
    block_path = _registered_shared_modeling_path(
        root=root,
        registration=reference["text_block"],
    )
    current_stat = _stat_identity(os.lstat(block_path))
    cached = _SHARED_MODELING_HANDLES.get(content_sha)
    if cached is not None:
        cached_path, cached_stat, cached_reference = cached
        if (
            cached_path == block_path
            and cached_stat == current_stat
            and cached_reference == reference
        ):
            authenticated_path = block_path
        else:
            authenticated_path = _validate_registration(
                root,
                reference["text_block"],
                label="shared preflight text block",
            )
    else:
        authenticated_path = _validate_registration(
            root,
            reference["text_block"],
            label="shared preflight text block",
        )
    frame = _read_exact_parquet(
        authenticated_path,
        expected_columns=reference["stored_columns"],
        label="shared preflight text block",
    )
    _validate_shared_modeling_frame(frame, reference=reference)
    after_stat = _stat_identity(os.lstat(authenticated_path))
    if after_stat != current_stat:
        raise RuntimeError(
            "shared preflight text block changed while authenticating"
        )
    _SHARED_MODELING_HANDLES[content_sha] = (
        authenticated_path,
        after_stat,
        copy.deepcopy(reference),
    )
    return copy.deepcopy(reference), frame


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
    # Process-global memoization records only the authenticated content
    # identity.  In particular, it must not retain the cache locator, open
    # provider, arrays, or the reference document that can reopen them.
    _SHARED_CACHE_HANDLES.add(content_sha)
    return cache


def _safe_cache_scientific_metadata(
    *,
    shared_cache: SpentOnlyFrozenChunkEmbeddingCache,
    logical_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the closed, path-neutral cache science needed by workers."""

    source = shared_cache.metadata
    try:
        hidden_size = int(source["hidden_size"])
        chunk_size_words = int(source["chunk_size_words"])
        chunk_overlap_words = int(source["chunk_overlap_words"])
        max_chunks = int(source["max_chunks"])
        chunk_selection = str(source.get("chunk_selection") or "last")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "shared cache metadata lacks its scientific chunk configuration"
        ) from exc
    if (
        hidden_size < 1
        or chunk_size_words < 1
        or chunk_overlap_words < 0
        or chunk_overlap_words >= chunk_size_words
        or max_chunks < 1
        or chunk_selection not in {"first", "last"}
    ):
        raise ValueError(
            "shared cache metadata has an invalid scientific chunk configuration"
        )
    source_metadata_sha256 = _require_sha256(
        logical_identity.get("metadata_sha256"),
        label="scoped embedding source metadata SHA",
    )
    return {
        "schema_version": PREFLIGHT_SAFE_CACHE_SCIENTIFIC_METADATA_SCHEMA,
        "source_metadata_sha256": source_metadata_sha256,
        "hidden_size": hidden_size,
        "chunk_size_words": chunk_size_words,
        "chunk_overlap_words": chunk_overlap_words,
        "max_chunks": max_chunks,
        "chunk_selection": chunk_selection,
        "normalize_embeddings": bool(source.get("normalize_embeddings", False)),
        "max_seq_length": (
            None
            if source.get("max_seq_length") is None
            else int(source["max_seq_length"])
        ),
        "chunk_cap_nonbinding": source.get("chunk_cap_nonbinding"),
        "semantic_truncation_allowed": source.get(
            "semantic_truncation_allowed"
        ),
        "tokenizer_truncation_allowed": source.get(
            "tokenizer_truncation_allowed"
        ),
        "production_provenance_included": False,
        "operational_execution_metadata_included": False,
    }


def _validate_safe_cache_scientific_metadata(
    value: Any,
    *,
    expected_source_metadata_sha256: str | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "source_metadata_sha256",
        "hidden_size",
        "chunk_size_words",
        "chunk_overlap_words",
        "max_chunks",
        "chunk_selection",
        "normalize_embeddings",
        "max_seq_length",
        "chunk_cap_nonbinding",
        "semantic_truncation_allowed",
        "tokenizer_truncation_allowed",
        "production_provenance_included",
        "operational_execution_metadata_included",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(
            "safe cache scientific metadata is not closed"
        )
    metadata = copy.deepcopy(dict(value))
    source_sha = _require_sha256(
        metadata.get("source_metadata_sha256"),
        label="safe cache source metadata SHA",
    )
    try:
        hidden_size = int(metadata["hidden_size"])
        chunk_size_words = int(metadata["chunk_size_words"])
        chunk_overlap_words = int(metadata["chunk_overlap_words"])
        max_chunks = int(metadata["max_chunks"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "safe cache scientific metadata has invalid numeric fields"
        ) from exc
    max_seq_length = metadata["max_seq_length"]
    if (
        metadata["schema_version"]
        != PREFLIGHT_SAFE_CACHE_SCIENTIFIC_METADATA_SCHEMA
        or (
            expected_source_metadata_sha256 is not None
            and source_sha != expected_source_metadata_sha256
        )
        or hidden_size < 1
        or chunk_size_words < 1
        or chunk_overlap_words < 0
        or chunk_overlap_words >= chunk_size_words
        or max_chunks < 1
        or metadata["chunk_selection"] not in {"first", "last"}
        or type(metadata["normalize_embeddings"]) is not bool
        or (
            max_seq_length is not None
            and (
                isinstance(max_seq_length, bool)
                or not isinstance(max_seq_length, int)
                or max_seq_length < 1
            )
        )
        or metadata["chunk_cap_nonbinding"] is not True
        or metadata["semantic_truncation_allowed"] is not False
        or metadata["tokenizer_truncation_allowed"] is not False
        or metadata["production_provenance_included"] is not False
        or metadata["operational_execution_metadata_included"] is not False
    ):
        raise ValueError(
            "safe cache scientific metadata is invalid"
        )
    return metadata


def _scoped_cache_metadata_from_counts(
    *,
    scientific_metadata: Mapping[str, Any],
    global_row_count: int,
    allowed_row_ids: Sequence[int],
    chunk_counts_by_row: Mapping[int, int],
) -> dict[str, Any]:
    base = _validate_safe_cache_scientific_metadata(
        scientific_metadata
    )
    allowed = tuple(map(int, allowed_row_ids))
    allowed_set = frozenset(allowed)
    if (
        isinstance(global_row_count, bool)
        or not isinstance(global_row_count, int)
        or global_row_count < 1
        or not allowed
        or len(allowed) != len(allowed_set)
        or min(allowed) < 0
        or max(allowed) >= global_row_count
        or set(map(int, chunk_counts_by_row)) != set(allowed)
        or any(
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or int(count) < 1
            for count in chunk_counts_by_row.values()
        )
    ):
        raise ValueError(
            "scoped cache chunk-count authority is invalid"
        )
    scoped_counts = [
        int(chunk_counts_by_row[row_id])
        if row_id in allowed_set
        else 0
        for row_id in range(global_row_count)
    ]
    fit_chunks = int(sum(scoped_counts))
    return {
        "schema_version": PREFLIGHT_SCOPED_CACHE_METADATA_SCHEMA,
        "source_metadata_sha256": base["source_metadata_sha256"],
        "num_samples": global_row_count,
        "hidden_size": int(base["hidden_size"]),
        "total_chunks": fit_chunks,
        "chunk_counts": scoped_counts,
        "chunk_size_words": int(base["chunk_size_words"]),
        "chunk_overlap_words": int(base["chunk_overlap_words"]),
        "max_chunks": int(base["max_chunks"]),
        "chunk_selection": str(base["chunk_selection"]),
        "normalize_embeddings": bool(base["normalize_embeddings"]),
        "max_seq_length": base["max_seq_length"],
        "uncapped_total_chunks": fit_chunks,
        "uncapped_chunk_counts_sha256": _sha256_json(scoped_counts),
        "chunk_cap_nonbinding": True,
        "semantic_truncation_allowed": False,
        "tokenizer_truncation_allowed": False,
        "allowed_row_count": len(allowed),
        "nonfit_row_count": global_row_count - len(allowed),
        "nonfit_chunk_count": 0,
        "nonfit_chunk_counts_zeroed": True,
        "production_provenance_included": False,
        "operational_execution_metadata_included": False,
    }


def _scoped_cache_metadata(
    *,
    shared_cache: SpentOnlyFrozenChunkEmbeddingCache,
    logical_identity: Mapping[str, Any],
    allowed_row_ids: Sequence[int],
) -> dict[str, Any]:
    """Project cache metadata to the closed fit-row worker contract."""

    source = shared_cache.metadata
    global_rows = int(shared_cache.row_count)
    source_counts = source.get("chunk_counts")
    if (
        not isinstance(source_counts, list)
        or len(source_counts) != global_rows
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in source_counts
        )
    ):
        raise ValueError(
            "shared cache metadata lacks exact per-row chunk counts"
        )
    allowed = tuple(map(int, allowed_row_ids))
    return _scoped_cache_metadata_from_counts(
        scientific_metadata=_safe_cache_scientific_metadata(
            shared_cache=shared_cache,
            logical_identity=logical_identity,
        ),
        global_row_count=global_rows,
        allowed_row_ids=allowed,
        chunk_counts_by_row={
            row_id: int(source_counts[row_id])
            for row_id in allowed
        },
    )


class ScopedEmbeddingView:
    """A row-remapped cache facade containing only one owner's fit rows."""

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
        self._logical_identity = copy.deepcopy(dict(logical_identity))
        self._allowed_row_ids = frozenset(allowed)
        self._allowed_row_order = allowed
        self.shared_reference_content_sha256 = _require_sha256(
            shared_reference_content_sha256,
            label="scoped embedding view shared-reference SHA",
        )
        global_rows = int(shared_cache.row_count)
        ordered_allowed = set(allowed)
        offsets = np.zeros(global_rows + 1, dtype=np.int64)
        matrices: list[np.ndarray] = []
        cached_by_row: dict[int, tuple[str, ...]] = {}
        cursor = 0
        for row_id in range(global_rows):
            offsets[row_id] = cursor
            if row_id in ordered_allowed:
                start = int(shared_cache._offsets[row_id])
                stop = int(shared_cache._offsets[row_id + 1])
                matrix = np.array(
                    shared_cache._embeddings[start:stop],
                    dtype=np.float32,
                    copy=True,
                    order="C",
                )
                matrix.setflags(write=False)
                matrices.append(matrix)
                cursor += len(matrix)
                cached_by_row[row_id] = tuple(
                    shared_cache._cached_chunks(row_id)
                )
            offsets[row_id + 1] = cursor
        if not matrices:
            raise RuntimeError(
                "scoped embedding view contains no fit-row embeddings"
            )
        embeddings = np.concatenate(matrices, axis=0)
        embeddings.setflags(write=False)
        offsets.setflags(write=False)
        metadata = _scoped_cache_metadata(
            shared_cache=shared_cache,
            logical_identity=logical_identity,
            allowed_row_ids=allowed,
        )
        self.cache_dir = Path("production-scoped-embedding-cache")
        self._metadata = metadata
        self._embeddings = embeddings
        self._offsets = offsets
        self._cached_by_row = cached_by_row
        # Deliberately retain no reference to the shared cache or its full
        # arrays.  The worker capability cannot accidentally index a held-out
        # embedding even through private implementation attributes.
        self._shared_cache_retained = False

    @classmethod
    def from_authorized_row_blocks(
        cls,
        *,
        logical_identity: Mapping[str, Any],
        global_row_count: int,
        allowed_row_ids: Sequence[int],
        shared_reference_content_sha256: str,
        scientific_metadata: Mapping[str, Any],
        matrices_by_row: Mapping[int, np.ndarray],
        chunks_by_row: Mapping[int, Sequence[str]],
    ) -> "ScopedEmbeddingView":
        """Build a worker view without ever opening the global cache."""

        allowed = tuple(map(int, allowed_row_ids))
        allowed_set = frozenset(allowed)
        if (
            isinstance(global_row_count, bool)
            or not isinstance(global_row_count, int)
            or global_row_count < 1
            or not allowed
            or len(allowed) != len(allowed_set)
            or min(allowed) < 0
            or max(allowed) >= global_row_count
            or set(map(int, matrices_by_row)) != set(allowed)
            or set(map(int, chunks_by_row)) != set(allowed)
        ):
            raise ValueError(
                "authorized embedding-row projection is invalid"
            )
        base = _validate_safe_cache_scientific_metadata(
            scientific_metadata,
            expected_source_metadata_sha256=str(
                logical_identity.get("metadata_sha256") or ""
            ),
        )
        hidden_size = int(base["hidden_size"])
        offsets = np.zeros(global_row_count + 1, dtype=np.int64)
        matrices: list[np.ndarray] = []
        cached_by_row: dict[int, tuple[str, ...]] = {}
        chunk_counts_by_row: dict[int, int] = {}
        cursor = 0
        for row_id in range(global_row_count):
            offsets[row_id] = cursor
            if row_id in allowed_set:
                matrix = np.asarray(matrices_by_row[row_id])
                chunks = tuple(chunks_by_row[row_id])
                if (
                    matrix.dtype != np.dtype(np.float32)
                    or matrix.ndim != 2
                    or matrix.shape[0] < 1
                    or matrix.shape[1] != hidden_size
                    or len(chunks) != int(matrix.shape[0])
                    or not all(isinstance(chunk, str) for chunk in chunks)
                ):
                    raise ValueError(
                        "authorized embedding-row block has invalid shape "
                        "or chunk alignment"
                    )
                copied = np.array(
                    matrix,
                    dtype=np.float32,
                    copy=True,
                    order="C",
                )
                copied.setflags(write=False)
                matrices.append(copied)
                cached_by_row[row_id] = chunks
                chunk_counts_by_row[row_id] = len(chunks)
                cursor += len(copied)
            offsets[row_id + 1] = cursor
        embeddings = np.concatenate(matrices, axis=0)
        embeddings.setflags(write=False)
        offsets.setflags(write=False)
        metadata = _scoped_cache_metadata_from_counts(
            scientific_metadata=base,
            global_row_count=global_row_count,
            allowed_row_ids=allowed,
            chunk_counts_by_row=chunk_counts_by_row,
        )
        view = cls.__new__(cls)
        view._logical_identity = copy.deepcopy(dict(logical_identity))
        view._allowed_row_ids = allowed_set
        view._allowed_row_order = allowed
        view.shared_reference_content_sha256 = _require_sha256(
            shared_reference_content_sha256,
            label="scoped embedding view shared-reference SHA",
        )
        view.cache_dir = Path("production-scoped-embedding-row-blocks")
        view._metadata = metadata
        view._embeddings = embeddings
        view._offsets = offsets
        view._cached_by_row = cached_by_row
        view._shared_cache_retained = False
        return view

    @property
    def row_count(self) -> int:
        return int(len(self._offsets) - 1)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._metadata)

    @property
    def allowed_row_ids(self) -> tuple[int, ...]:
        return self._allowed_row_order

    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._logical_identity)

    def authenticated_snapshot_identity(self) -> Mapping[str, Any]:
        return self.identity()

    def _cached_chunks(self, row_id: int) -> tuple[str, ...]:
        value = int(row_id)
        if value not in self._allowed_row_ids:
            raise ValueError(
                "scoped embedding view refuses a non-fit row"
            )
        try:
            return self._cached_by_row[value]
        except KeyError as exc:
            raise ValueError(
                "scoped embedding view refuses a non-fit row"
            ) from exc

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
        physical = SpentOnlyFrozenChunkEmbeddingCache.bind_spent(
            self,
            requested,
            tuple(texts),
        )
        return BoundSpentFrozenChunkEmbeddingProvider(
            cache=self,
            row_ids=physical.row_ids,
            cached_by_row=physical.cached_by_row,
            token_bounded_row_ids=physical.token_bounded_row_ids,
        )


def _write_npy(path: Path, values: np.ndarray) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(
            f"immutable embedding-row block already exists: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npy",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
        descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _embedding_array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _embedding_row_scientific_sha256(
    *,
    row_id: int,
    matrix: np.ndarray,
    chunks: Sequence[str],
    token_bounded_reconciliation_used: bool,
) -> str:
    array = np.ascontiguousarray(matrix)
    exact_chunks = tuple(chunks)
    if (
        array.dtype != np.dtype(np.float32)
        or array.ndim != 2
        or array.shape[0] < 1
        or len(exact_chunks) != int(array.shape[0])
        or not all(isinstance(chunk, str) for chunk in exact_chunks)
        or type(token_bounded_reconciliation_used) is not bool
    ):
        raise ValueError(
            "embedding-row scientific projection is invalid"
        )
    body = {
        "schema_version": (
            "spent_only_frozen_embedding_row_scientific_digest_v1"
        ),
        "row_id": int(row_id),
        "ordered_chunk_texts_sha256": _sha256_json(list(exact_chunks)),
        "ordered_chunk_count": len(exact_chunks),
        "embedding_array_sha256": _embedding_array_digest(array),
        "embedding_dtype": array.dtype.str,
        "embedding_shape": list(array.shape),
        "token_bounded_reconciliation_used": (
            token_bounded_reconciliation_used
        ),
    }
    return _sha256_json(body)


def _embedding_row_paths(
    *,
    set_root: Path,
    row_id: int,
    scientific_sha256: str,
) -> tuple[Path, Path]:
    digest = _require_sha256(
        scientific_sha256,
        label="embedding-row scientific SHA",
    )
    stem = f"row_{int(row_id):012d}_{digest}"
    directory = set_root / _EMBEDDING_ROW_DIRECTORY
    return directory / f"{stem}.npy", directory / f"{stem}.chunks.json"


def _embedding_row_record_body(
    *,
    row_id: int,
    source_row_scientific_sha256: str,
    token_bounded_reconciliation_used: bool,
    matrix: np.ndarray,
    embedding_registration: Mapping[str, Any],
    chunks_registration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_EMBEDDING_ROW_BLOCK_SCHEMA,
        "global_row_id": int(row_id),
        "source_row_scientific_sha256": _require_sha256(
            source_row_scientific_sha256,
            label="embedding-row source scientific SHA",
        ),
        "token_bounded_reconciliation_used": bool(
            token_bounded_reconciliation_used
        ),
        "embedding_dtype": np.asarray(matrix).dtype.str,
        "embedding_shape": list(np.asarray(matrix).shape),
        "chunk_count": int(np.asarray(matrix).shape[0]),
        "embedding_block": copy.deepcopy(dict(embedding_registration)),
        "chunk_text_block": copy.deepcopy(dict(chunks_registration)),
    }


def _read_registered_bytes_once(
    root: Path,
    registration: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Path, bytes]:
    """Authenticate and return one row block with one filesystem read."""

    if set(registration) != {
        "relative_path",
        "sha256",
        "size_bytes",
    }:
        raise ValueError(f"{label} registration is not closed")
    relative_text = str(registration["relative_path"])
    relative = Path(relative_text)
    if (
        not relative_text
        or relative.is_absolute()
        or relative.as_posix() != relative_text
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} registration escapes its descriptor")
    resolved_root = root.resolve(strict=True)
    path = root / relative
    try:
        path.resolve(strict=True).relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} file escapes its descriptor") from exc
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or int(before.st_size)
            != int(registration["size_bytes"])
        ):
            raise ValueError(f"{label} file metadata changed")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        observed_size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
            observed_size += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = os.lstat(path)
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(named)
        or observed_size != int(registration["size_bytes"])
        or digest.hexdigest() != registration["sha256"]
    ):
        raise ValueError(f"{label} file changed")
    return path, b"".join(chunks)


def _json_object_from_authenticated_bytes(
    payload: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    def reject_duplicates(
        pairs: Sequence[tuple[str, Any]],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(
                    f"{label} contains duplicate key {key!r}"
                )
            output[key] = value
        return output

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda raw: (
                (_ for _ in ()).throw(
                    ValueError(
                        f"{label} contains non-finite constant {raw}"
                    )
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one object")
    return value


def _publish_embedding_row_store(
    *,
    set_root: Path,
    embedding_cache: SpentOnlyFrozenChunkEmbeddingCache,
    embedding_cache_identity: Mapping[str, Any],
    shared_cache_reference_content_sha256: str,
    modeling_data: pd.DataFrame,
    text_column: str,
    required_row_ids: Sequence[int],
) -> dict[str, Any]:
    """Materialize each required embedding row once at the trusted boundary."""

    row_count = int(embedding_cache.row_count)
    required = _validated_fit_rows(
        list(required_row_ids),
        row_count=row_count,
        label="shared embedding-row store rows",
    )
    if tuple(required) != tuple(sorted(required)):
        raise ValueError(
            "shared embedding-row store rows must be in global order"
        )
    if len(modeling_data) != row_count:
        raise ValueError(
            "shared embedding-row store differs from its prepared cohort"
        )
    if _authenticated_cache_identity(embedding_cache) != dict(
        embedding_cache_identity
    ):
        raise ValueError(
            "shared embedding-row store cache identity changed"
        )
    store_manifest_path = (
        set_root / PREFLIGHT_EMBEDDING_ROW_STORE_MANIFEST
    )
    if store_manifest_path.is_file():
        return _validate_embedding_row_store(
            set_root=set_root,
            manifest_path=store_manifest_path,
            expected_row_ids=required,
            expected_row_count=row_count,
            expected_embedding_cache_identity=embedding_cache_identity,
            expected_shared_cache_reference_content_sha256=(
                shared_cache_reference_content_sha256
            ),
            parent_embedding_cache=embedding_cache,
            parent_modeling_data=modeling_data,
            text_column=text_column,
        )
    row_directory = set_root / _EMBEDDING_ROW_DIRECTORY
    if row_directory.is_symlink():
        raise ValueError(
            "shared embedding-row directory cannot be a symlink"
        )
    row_directory.mkdir(exist_ok=True)
    if row_directory.resolve(strict=True) != row_directory:
        raise ValueError(
            "shared embedding-row directory is not canonical"
        )
    texts = tuple(
        str(value)
        for value in modeling_data.iloc[list(required)][text_column]
    )
    bound = embedding_cache.bind_spent(required, texts)
    source_digests = bound.exact_row_scientific_digests()
    token_bounded = frozenset(
        map(int, bound.token_bounded_row_ids)
    )
    records: list[dict[str, Any]] = []
    for row_id, source_digest in zip(
        required,
        source_digests,
        strict=True,
    ):
        matrix = np.ascontiguousarray(
            bound.chunk_matrix(row_id),
            dtype=np.float32,
        )
        chunks = tuple(bound.chunk_texts((row_id,))[0])
        reconciled = row_id in token_bounded
        if source_digest != _embedding_row_scientific_sha256(
            row_id=row_id,
            matrix=matrix,
            chunks=chunks,
            token_bounded_reconciliation_used=reconciled,
        ):
            raise RuntimeError(
                "embedding-row source digest changed during publication"
            )
        embedding_path, chunks_path = _embedding_row_paths(
            set_root=set_root,
            row_id=row_id,
            scientific_sha256=source_digest,
        )
        if not embedding_path.exists() and not embedding_path.is_symlink():
            _write_npy(embedding_path, matrix)
        chunks_body = {
            "schema_version": PREFLIGHT_EMBEDDING_ROW_BLOCK_SCHEMA,
            "global_row_id": row_id,
            "source_row_scientific_sha256": source_digest,
            "token_bounded_reconciliation_used": reconciled,
            "chunks": list(chunks),
        }
        if not chunks_path.exists() and not chunks_path.is_symlink():
            _write_json(
                chunks_path,
                {
                    **chunks_body,
                    "content_sha256": _sha256_json(chunks_body),
                },
                compact=True,
            )
        row_body = _embedding_row_record_body(
            row_id=row_id,
            source_row_scientific_sha256=source_digest,
            token_bounded_reconciliation_used=reconciled,
            matrix=matrix,
            embedding_registration=_file_registration(
                embedding_path,
                set_root,
            ),
            chunks_registration=_file_registration(
                chunks_path,
                set_root,
            ),
        )
        records.append(
            {
                **row_body,
                "content_sha256": _sha256_json(row_body),
            }
        )
    scientific_metadata = _safe_cache_scientific_metadata(
        shared_cache=embedding_cache,
        logical_identity=embedding_cache_identity,
    )
    body = {
        "schema_version": PREFLIGHT_EMBEDDING_ROW_STORE_SCHEMA,
        "source_cache_reference_content_sha256": _require_sha256(
            shared_cache_reference_content_sha256,
            label="embedding-row store source-cache reference SHA",
        ),
        "source_cache_logical_identity": json.loads(
            _canonical_json(dict(embedding_cache_identity))
        ),
        "source_cache_logical_identity_sha256": _sha256_json(
            embedding_cache_identity
        ),
        "scientific_metadata": scientific_metadata,
        "dataset_row_count": row_count,
        "required_row_ids": list(required),
        "required_row_order_sha256": _sha256_json(list(required)),
        "required_row_count": len(required),
        "rows": records,
        "row_record_order_sha256": _sha256_json(
            [row["content_sha256"] for row in records]
        ),
        "content_addressed_row_blocks": True,
        "each_embedding_row_materialized_once": True,
        "per_scope_embedding_payload_copies": 0,
        "global_cache_locator_included": False,
        "production_provenance_included": False,
    }
    manifest = {**body, "content_sha256": _sha256_json(body)}
    _write_json(store_manifest_path, manifest)
    return _validate_embedding_row_store(
        set_root=set_root,
        manifest_path=store_manifest_path,
        expected_row_ids=required,
        expected_row_count=row_count,
        expected_embedding_cache_identity=embedding_cache_identity,
        expected_shared_cache_reference_content_sha256=(
            shared_cache_reference_content_sha256
        ),
        parent_embedding_cache=embedding_cache,
        parent_modeling_data=modeling_data,
        text_column=text_column,
    )


def _load_embedding_row_record(
    *,
    set_root: Path,
    record: Mapping[str, Any],
    expected_row_id: int,
    expected_hidden_size: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    required = {
        "schema_version",
        "global_row_id",
        "source_row_scientific_sha256",
        "token_bounded_reconciliation_used",
        "embedding_dtype",
        "embedding_shape",
        "chunk_count",
        "embedding_block",
        "chunk_text_block",
        "content_sha256",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise ValueError(
            "embedding-row registration is not closed"
        )
    body = {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key != "content_sha256"
    }
    source_digest = _require_sha256(
        record.get("source_row_scientific_sha256"),
        label="embedding-row source scientific SHA",
    )
    reconciled = record.get("token_bounded_reconciliation_used")
    shape = record.get("embedding_shape")
    if (
        record.get("schema_version")
        != PREFLIGHT_EMBEDDING_ROW_BLOCK_SCHEMA
        or record.get("global_row_id") != int(expected_row_id)
        or type(reconciled) is not bool
        or record.get("embedding_dtype") != np.dtype(np.float32).str
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in shape
        )
        or shape[0] < 1
        or shape[1] != int(expected_hidden_size)
        or record.get("chunk_count") != shape[0]
        or record.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("embedding-row registration is invalid")
    embedding_path, chunks_path = _embedding_row_paths(
        set_root=set_root,
        row_id=expected_row_id,
        scientific_sha256=source_digest,
    )
    (
        registered_embedding,
        embedding_bytes,
    ) = _read_registered_bytes_once(
        set_root,
        record["embedding_block"],
        label=f"embedding row {expected_row_id} array",
    )
    (
        registered_chunks,
        chunks_bytes,
    ) = _read_registered_bytes_once(
        set_root,
        record["chunk_text_block"],
        label=f"embedding row {expected_row_id} chunks",
    )
    if (
        registered_embedding != embedding_path
        or registered_chunks != chunks_path
    ):
        raise ValueError(
            "embedding-row content-addressed layout changed"
        )
    try:
        matrix = np.load(
            io.BytesIO(embedding_bytes),
            allow_pickle=False,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "embedding-row array is invalid"
        ) from exc
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.dtype != np.dtype(np.float32)
        or list(matrix.shape) != shape
        or matrix.ndim != 2
        or not bool(np.isfinite(matrix).all())
    ):
        raise ValueError(
            "embedding-row array shape, dtype, or values changed"
        )
    chunk_payload = _json_object_from_authenticated_bytes(
        chunks_bytes,
        label=f"embedding row {expected_row_id} chunks",
    )
    chunk_required = {
        "schema_version",
        "global_row_id",
        "source_row_scientific_sha256",
        "token_bounded_reconciliation_used",
        "chunks",
        "content_sha256",
    }
    chunk_body = {
        key: copy.deepcopy(value)
        for key, value in chunk_payload.items()
        if key != "content_sha256"
    }
    chunks = chunk_payload.get("chunks")
    if (
        set(chunk_payload) != chunk_required
        or chunk_payload.get("schema_version")
        != PREFLIGHT_EMBEDDING_ROW_BLOCK_SCHEMA
        or chunk_payload.get("global_row_id") != expected_row_id
        or chunk_payload.get("source_row_scientific_sha256")
        != source_digest
        or chunk_payload.get("token_bounded_reconciliation_used")
        is not reconciled
        or not isinstance(chunks, list)
        or len(chunks) != shape[0]
        or not all(isinstance(chunk, str) for chunk in chunks)
        or chunk_payload.get("content_sha256")
        != _sha256_json(chunk_body)
    ):
        raise ValueError("embedding-row chunk block changed")
    copied = np.array(
        matrix,
        dtype=np.float32,
        copy=True,
        order="C",
    )
    if source_digest != _embedding_row_scientific_sha256(
        row_id=expected_row_id,
        matrix=copied,
        chunks=chunks,
        token_bounded_reconciliation_used=reconciled,
    ):
        raise ValueError(
            "embedding-row scientific content changed"
        )
    copied.setflags(write=False)
    return copied, tuple(chunks)


def _validate_embedding_row_store(
    *,
    set_root: Path,
    manifest_path: Path,
    expected_row_ids: Sequence[int],
    expected_row_count: int,
    expected_embedding_cache_identity: Mapping[str, Any],
    expected_shared_cache_reference_content_sha256: str,
    parent_embedding_cache: SpentOnlyFrozenChunkEmbeddingCache | None = None,
    parent_modeling_data: pd.DataFrame | None = None,
    text_column: str | None = None,
) -> dict[str, Any]:
    root = Path(set_root).absolute()
    path = Path(manifest_path).absolute()
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
        or path != root / PREFLIGHT_EMBEDDING_ROW_STORE_MANIFEST
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError(
            "shared embedding-row store manifest path is invalid"
        )
    manifest = _read_json(
        path,
        label="shared embedding-row store manifest",
    )
    required_fields = {
        "schema_version",
        "source_cache_reference_content_sha256",
        "source_cache_logical_identity",
        "source_cache_logical_identity_sha256",
        "scientific_metadata",
        "dataset_row_count",
        "required_row_ids",
        "required_row_order_sha256",
        "required_row_count",
        "rows",
        "row_record_order_sha256",
        "content_addressed_row_blocks",
        "each_embedding_row_materialized_once",
        "per_scope_embedding_payload_copies",
        "global_cache_locator_included",
        "production_provenance_included",
        "content_sha256",
    }
    body = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "content_sha256"
    }
    expected_rows = tuple(map(int, expected_row_ids))
    rows = manifest.get("rows")
    logical_identity = manifest.get("source_cache_logical_identity")
    if (
        set(manifest) != required_fields
        or manifest.get("schema_version")
        != PREFLIGHT_EMBEDDING_ROW_STORE_SCHEMA
        or manifest.get("source_cache_reference_content_sha256")
        != expected_shared_cache_reference_content_sha256
        or not isinstance(logical_identity, Mapping)
        or dict(logical_identity)
        != dict(expected_embedding_cache_identity)
        or manifest.get("source_cache_logical_identity_sha256")
        != _sha256_json(logical_identity)
        or manifest.get("dataset_row_count") != expected_row_count
        or manifest.get("required_row_ids") != list(expected_rows)
        or manifest.get("required_row_order_sha256")
        != _sha256_json(list(expected_rows))
        or manifest.get("required_row_count") != len(expected_rows)
        or not isinstance(rows, list)
        or len(rows) != len(expected_rows)
        or manifest.get("row_record_order_sha256")
        != _sha256_json(
            [
                row.get("content_sha256")
                if isinstance(row, Mapping)
                else None
                for row in rows
            ]
        )
        or manifest.get("content_addressed_row_blocks") is not True
        or manifest.get("each_embedding_row_materialized_once") is not True
        or manifest.get("per_scope_embedding_payload_copies") != 0
        or manifest.get("global_cache_locator_included") is not False
        or manifest.get("production_provenance_included") is not False
        or manifest.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError(
            "shared embedding-row store manifest is invalid"
        )
    scientific_metadata = _validate_safe_cache_scientific_metadata(
        manifest.get("scientific_metadata"),
        expected_source_metadata_sha256=str(
            logical_identity.get("metadata_sha256") or ""
        ),
    )
    observed_digests: list[str] = []
    for row_id, record in zip(expected_rows, rows, strict=True):
        _matrix, _chunks = _load_embedding_row_record(
            set_root=root,
            record=record,
            expected_row_id=row_id,
            expected_hidden_size=int(scientific_metadata["hidden_size"]),
        )
        observed_digests.append(
            str(record["source_row_scientific_sha256"])
        )
    if parent_embedding_cache is not None:
        if (
            parent_modeling_data is None
            or text_column is None
            or len(parent_modeling_data) != expected_row_count
            or _authenticated_cache_identity(parent_embedding_cache)
            != dict(expected_embedding_cache_identity)
        ):
            raise ValueError(
                "trusted embedding-row publication parent changed"
            )
        texts = tuple(
            str(value)
            for value in parent_modeling_data.iloc[
                list(expected_rows)
            ][text_column]
        )
        bound = parent_embedding_cache.bind_spent(
            expected_rows,
            texts,
        )
        if tuple(observed_digests) != tuple(
            bound.exact_row_scientific_digests()
        ):
            raise ValueError(
                "embedding-row store differs from its trusted parent cache"
            )
    return copy.deepcopy(manifest)


def _load_scoped_embedding_row_blocks(
    *,
    set_root: Path,
    cache_view: Mapping[str, Any],
    global_row_count: int,
    fit_rows: Sequence[int],
) -> ScopedEmbeddingView:
    records = cache_view["row_blocks"]
    scientific_metadata = _validate_safe_cache_scientific_metadata(
        cache_view["scientific_metadata"],
        expected_source_metadata_sha256=str(
            cache_view["logical_identity"].get("metadata_sha256") or ""
        ),
    )
    matrices: dict[int, np.ndarray] = {}
    chunks: dict[int, tuple[str, ...]] = {}
    for row_id, record in zip(fit_rows, records, strict=True):
        matrix, row_chunks = _load_embedding_row_record(
            set_root=set_root,
            record=record,
            expected_row_id=int(row_id),
            expected_hidden_size=int(scientific_metadata["hidden_size"]),
        )
        matrices[int(row_id)] = matrix
        chunks[int(row_id)] = row_chunks
    return ScopedEmbeddingView.from_authorized_row_blocks(
        logical_identity=cache_view["logical_identity"],
        global_row_count=global_row_count,
        allowed_row_ids=fit_rows,
        shared_reference_content_sha256=str(
            cache_view["source_cache_reference_content_sha256"]
        ),
        scientific_metadata=scientific_metadata,
        matrices_by_row=matrices,
        chunks_by_row=chunks,
    )


def _private_config_payload(
    *,
    config: AppliedInferenceConfig,
    forbidden_paths: Sequence[Path],
) -> dict[str, Any]:
    # Runtime receives the physical inputs as separately authenticated
    # capabilities.  Keeping neutral URIs here makes the scientific
    # configuration independent of an attempt/recovery location.
    modeling_path = (
        "production://private-preflight/shared-text-fit-label-view-v1"
    )
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
    semantic_witness_scientific_config: Any

    @property
    def manifest_path(self) -> Path:
        return self.root / PREFLIGHT_SCOPE_INPUT_MANIFEST

    @property
    def scope_id(self) -> str:
        return str(self.scope["scope_id"])


@dataclass(frozen=True)
class AuthenticatedPreflightScopeCapability:
    """Parent-held lightweight pointer to one sealed worker capability."""

    root: Path
    manifest: Mapping[str, Any]
    scope: Mapping[str, Any]

    @property
    def manifest_path(self) -> Path:
        return self.root / PREFLIGHT_SCOPE_INPUT_MANIFEST

    @property
    def scope_id(self) -> str:
        return str(self.scope["scope_id"])


@dataclass(frozen=True)
class AuthenticatedPreflightScopeInputSet:
    root: Path
    manifest: Mapping[str, Any]
    scopes: Mapping[str, AuthenticatedPreflightScopeCapability]
    shared_cache_reference_path: Path
    shared_cache_reference: Mapping[str, Any]
    shared_modeling_reference_path: Path
    shared_modeling_reference: Mapping[str, Any]
    embedding_row_store_manifest_path: Path
    embedding_row_store_manifest: Mapping[str, Any]

    def worker_payloads(self) -> tuple[Mapping[str, Any], ...]:
        payloads: list[dict[str, Any]] = []
        for scope in self.scopes.values():
            payloads.append(
                {
                    "schema_version": (
                        "production_stage1_preflight_worker_payload_v4"
                    ),
                    "scope_id": scope.scope_id,
                    "manifest_path": str(scope.manifest_path),
                    "manifest_content_sha256": str(
                        scope.manifest["content_sha256"]
                    ),
                }
            )
        return tuple(payloads)

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
            "schema_version": "production_stage1_preflight_scope_input_set_identity_v5",
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
            "embedding_row_store_manifest": _file_registration(
                self.embedding_row_store_manifest_path,
                self.root,
            ),
            "embedding_row_store_content_sha256": str(
                self.embedding_row_store_manifest["content_sha256"]
            ),
            "shared_embedding_row_block_count": int(
                self.embedding_row_store_manifest["required_row_count"]
            ),
            "shared_embedding_row_store_bytes": sum(
                int(record[registration]["size_bytes"])
                for record in self.embedding_row_store_manifest["rows"]
                for registration in (
                    "embedding_block",
                    "chunk_text_block",
                )
            ),
            "worker_global_cache_locator_supplied": False,
            "worker_embedding_source": (
                "authenticated_fit_only_content_addressed_row_blocks_v1"
            ),
            "per_scope_embedding_arrays_copied": False,
            "per_scope_chunk_texts_copied": False,
            "per_scope_full_cohort_modeling_copied": False,
            "shared_modeling_reference": _file_registration(
                self.shared_modeling_reference_path,
                self.root,
            ),
            "shared_modeling_reference_content_sha256": str(
                self.shared_modeling_reference["content_sha256"]
            ),
            "shared_text_rows": int(
                self.shared_modeling_reference["row_count"]
            ),
            "shared_text_bytes": int(
                self.shared_modeling_reference["text_block"]["size_bytes"]
            ),
            "per_scope_fit_index_rows": {
                scope_id: int(
                    scope.manifest["shared_modeling_view"]["allowed_row_count"]
                )
                for scope_id, scope in self.scopes.items()
            },
            "per_scope_label_projection_bytes": {
                scope_id: int(
                    scope.manifest["files"]["fit_label_projection"][
                        "size_bytes"
                    ]
                )
                for scope_id, scope in self.scopes.items()
            },
            "attempt_root": str(attempt_root),
            "preserved_incomplete_attempts": attempts,
            "scope_inputs_outside_terminal_scientific_artifact": True,
        }
        return {**body, "content_sha256": _sha256_json(body)}


def _write_scope(
    *,
    root: Path,
    row_count: int,
    config: AppliedInferenceConfig,
    embedding_cache_identity: Mapping[str, Any],
    shared_cache_reference_content_sha256: str,
    embedding_row_store_scientific_metadata: Mapping[str, Any],
    embedding_row_blocks: Sequence[Mapping[str, Any]],
    fit_modeling_content_sha256: str,
    fit_label_projection: pd.DataFrame,
    registry_content_sha256: str,
    scope: Mapping[str, Any],
    forbidden_paths: Sequence[Path],
    semantic_witness_scientific_config: Any,
) -> None:
    root.mkdir(parents=True, exist_ok=False)
    fit_rows = _validated_fit_rows(
        _scope_value(scope, "fit_row_ids"),
        row_count=row_count,
        label="preflight scope fit rows",
    )
    modeling_columns = _modeling_columns(config)
    expected_label_columns = [
        _GLOBAL_ROW_ID_COLUMN,
        config.treatment_column,
        config.outcome_column,
    ]
    if (
        list(fit_label_projection.columns) != expected_label_columns
        or len(fit_label_projection) != len(fit_rows)
        or fit_label_projection[_GLOBAL_ROW_ID_COLUMN].tolist()
        != list(fit_rows)
    ):
        raise ValueError(
            "preflight fit-label projection differs from its row capability"
        )
    _write_parquet(
        root / _LABEL_PROJECTION_FILE,
        fit_label_projection,
    )
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
        "fit_label_projection": _file_registration(
            root / _LABEL_PROJECTION_FILE,
            root,
        ),
    }
    row_blocks = tuple(
        json.loads(_canonical_json(dict(record)))
        for record in embedding_row_blocks
    )
    if (
        len(row_blocks) != len(fit_rows)
        or tuple(
            int(record.get("global_row_id", -1))
            for record in row_blocks
        )
        != fit_rows
    ):
        raise ValueError(
            "preflight embedding-row blocks differ from fit-row authority"
        )
    scientific_metadata = _validate_safe_cache_scientific_metadata(
        embedding_row_store_scientific_metadata,
        expected_source_metadata_sha256=str(
            embedding_cache_identity.get("metadata_sha256") or ""
        ),
    )
    cache_view = {
        "schema_version": PREFLIGHT_SCOPED_CACHE_VIEW_SCHEMA,
        "source_cache_reference_content_sha256": _require_sha256(
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
        "scientific_metadata": scientific_metadata,
        "row_blocks": list(row_blocks),
        "row_blocks_content_sha256": _sha256_json(
            [record["content_sha256"] for record in row_blocks]
        ),
        "peer_row_access_allowed": False,
        "embedding_array_payload_count": 0,
        "chunk_text_payload_count": 0,
        "authorized_embedding_row_block_count": len(row_blocks),
        "authorized_chunk_text_row_block_count": len(row_blocks),
        "content_addressed_blocks_shared_across_scopes": True,
        "global_cache_locator_supplied": False,
        "full_cache_provider_supplied": False,
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
        "columns": modeling_columns,
        "shared_modeling_view": {
            "schema_version": PREFLIGHT_SHARED_MODELING_VIEW_SCHEMA,
            "fit_modeling_content_sha256": _require_sha256(
                fit_modeling_content_sha256,
                label="preflight fit-modeling content SHA",
            ),
            "dataset_row_count": row_count,
            "allowed_row_ids": list(fit_rows),
            "allowed_row_order_sha256": _sha256_json(list(fit_rows)),
            "allowed_row_count": len(fit_rows),
            "peer_row_access_allowed": False,
            "per_scope_text_payload_count": 0,
            "fit_label_projection_count": 1,
            "fit_label_projection_schema": (
                PREFLIGHT_SCOPE_LABEL_PROJECTION_SCHEMA
            ),
            "nonfit_labels_stored": False,
            "nonfit_rows_returned_by_worker_api": False,
        },
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


def _validated_scoped_cache_view(
    value: Any,
    *,
    fit_rows: Sequence[int],
    expected_embedding_cache_identity: Mapping[str, Any] | None = None,
    expected_shared_cache_reference_content_sha256: str | None = None,
    expected_row_records: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = tuple(map(int, fit_rows))
    fields = {
        "schema_version",
        "source_cache_reference_content_sha256",
        "logical_identity",
        "logical_identity_sha256",
        "allowed_row_ids",
        "allowed_row_order_sha256",
        "allowed_row_count",
        "scientific_metadata",
        "row_blocks",
        "row_blocks_content_sha256",
        "peer_row_access_allowed",
        "embedding_array_payload_count",
        "chunk_text_payload_count",
        "authorized_embedding_row_block_count",
        "authorized_chunk_text_row_block_count",
        "content_addressed_blocks_shared_across_scopes",
        "global_cache_locator_supplied",
        "full_cache_provider_supplied",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(
            "preflight scoped embedding-cache view is not closed"
        )
    view = copy.deepcopy(dict(value))
    logical_identity = view.get("logical_identity")
    row_blocks = view.get("row_blocks")
    reference_sha = _require_sha256(
        view.get("source_cache_reference_content_sha256"),
        label="preflight scoped source-cache reference SHA",
    )
    if (
        view.get("schema_version") != PREFLIGHT_SCOPED_CACHE_VIEW_SCHEMA
        or view.get("allowed_row_ids") != list(rows)
        or view.get("allowed_row_order_sha256")
        != _sha256_json(list(rows))
        or view.get("allowed_row_count") != len(rows)
        or view.get("peer_row_access_allowed") is not False
        or view.get("embedding_array_payload_count") != 0
        or view.get("chunk_text_payload_count") != 0
        or view.get("authorized_embedding_row_block_count")
        != len(rows)
        or view.get("authorized_chunk_text_row_block_count")
        != len(rows)
        or view.get("content_addressed_blocks_shared_across_scopes")
        is not True
        or view.get("global_cache_locator_supplied") is not False
        or view.get("full_cache_provider_supplied") is not False
        or not isinstance(logical_identity, Mapping)
        or view.get("logical_identity_sha256")
        != _sha256_json(logical_identity)
        or not isinstance(row_blocks, list)
        or len(row_blocks) != len(rows)
        or [
            int(record.get("global_row_id", -1))
            if isinstance(record, Mapping)
            else -1
            for record in row_blocks
        ]
        != list(rows)
        or view.get("row_blocks_content_sha256")
        != _sha256_json(
            [
                record.get("content_sha256")
                if isinstance(record, Mapping)
                else None
                for record in row_blocks
            ]
        )
    ):
        raise ValueError(
            "preflight scoped embedding-cache view is invalid"
        )
    _validate_safe_cache_scientific_metadata(
        view.get("scientific_metadata"),
        expected_source_metadata_sha256=str(
            logical_identity.get("metadata_sha256") or ""
        ),
    )
    if (
        expected_embedding_cache_identity is not None
        and dict(logical_identity)
        != dict(expected_embedding_cache_identity)
    ):
        raise ValueError(
            "preflight scoped view logical identity changed"
        )
    if (
        expected_shared_cache_reference_content_sha256 is not None
        and reference_sha
        != expected_shared_cache_reference_content_sha256
    ):
        raise ValueError(
            "preflight scoped source-cache reference changed"
        )
    if expected_row_records is not None:
        expected_records = [
            dict(expected_row_records[row_id])
            for row_id in rows
        ]
        if row_blocks != expected_records:
            raise ValueError(
                "preflight scoped row blocks differ from their shared store"
            )
    return view


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
    required_embedding_rows = tuple(
        sorted(
            {
                row_id
                for scope in canonical_scopes
                for row_id in _validated_fit_rows(
                    scope["fit_row_ids"],
                    row_count=len(modeling_data),
                    label=(
                        f"{scope['scope_id']} preflight embedding rows"
                    ),
                )
            }
        )
    )
    if _sha256_json(registry) != str(registry_content_sha256):
        raise ValueError("preflight parent registry differs from its content identity")
    shared_reference = _build_shared_cache_reference(
        embedding_cache=embedding_cache,
        embedding_cache_identity=embedding_cache_identity,
        global_embedding_cache_path=global_embedding_cache_path,
    )
    shared_reference_path = root / PREFLIGHT_SHARED_CACHE_REFERENCE
    shared_modeling_reference_path = (
        root / PREFLIGHT_SHARED_MODELING_REFERENCE
    )
    shared_modeling_path = root / _SHARED_MODELING_FILE
    embedding_row_store_manifest_path = (
        root / PREFLIGHT_EMBEDDING_ROW_STORE_MANIFEST
    )
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
        PREFLIGHT_SHARED_MODELING_REFERENCE,
        PREFLIGHT_EMBEDDING_ROW_STORE_MANIFEST,
        _EMBEDDING_ROW_DIRECTORY,
        _SHARED_MODELING_FILE,
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
    if (
        shared_modeling_reference_path.exists()
        and not shared_modeling_path.exists()
    ):
        raise ValueError(
            "shared preflight modeling reference exists without its block"
        )
    shared_modeling_reference, _shared_modeling_frame = (
        _build_shared_modeling_reference(
            block_path=shared_modeling_path,
            root=root,
            modeling_data=modeling_data,
            config=config,
        )
    )
    if shared_modeling_reference_path.exists():
        _validate_shared_modeling_reference(
            path=shared_modeling_reference_path,
            expected_content_sha256=str(
                shared_modeling_reference["content_sha256"]
            ),
            expected_reference=shared_modeling_reference,
        )
    else:
        _write_json(
            shared_modeling_reference_path,
            shared_modeling_reference,
        )
        _validate_shared_modeling_reference(
            path=shared_modeling_reference_path,
            expected_content_sha256=str(
                shared_modeling_reference["content_sha256"]
            ),
            expected_reference=shared_modeling_reference,
        )
    embedding_row_store_manifest = _publish_embedding_row_store(
        set_root=root,
        embedding_cache=embedding_cache,
        embedding_cache_identity=embedding_cache_identity,
        shared_cache_reference_content_sha256=str(
            shared_reference["content_sha256"]
        ),
        modeling_data=modeling_data,
        text_column=config.text_column,
        required_row_ids=required_embedding_rows,
    )
    embedding_rows_by_id = {
        int(record["global_row_id"]): record
        for record in embedding_row_store_manifest["rows"]
    }
    attempt_root = root.parent / f".{root.name}.scope_attempts"
    if attempt_root.is_symlink():
        raise ValueError("preflight scope-input attempt root cannot be a symlink")
    attempt_root.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for scope in canonical_scopes:
        scope_id = str(scope["scope_id"])
        scope_root = root / scope_id
        if scope_root.exists():
            completed = _validate_preflight_scope_capability(
                manifest_path=scope_root / PREFLIGHT_SCOPE_INPUT_MANIFEST,
                expected_scope=scope,
                expected_registry_content_sha256=registry_content_sha256,
                shared_modeling=_shared_modeling_frame,
                shared_modeling_reference=shared_modeling_reference,
                embedding_row_records=embedding_rows_by_id,
                expected_embedding_cache_identity=embedding_cache_identity,
                expected_shared_cache_reference_content_sha256=str(
                    shared_reference["content_sha256"]
                ),
                parent_modeling_data=modeling_data,
                parent_config=config,
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
                row_count=len(modeling_data),
                config=config,
                embedding_cache_identity=embedding_cache_identity,
                shared_cache_reference_content_sha256=str(
                    shared_reference["content_sha256"]
                ),
                embedding_row_store_scientific_metadata=(
                    embedding_row_store_manifest["scientific_metadata"]
                ),
                embedding_row_blocks=tuple(
                    embedding_rows_by_id[row_id]
                    for row_id in _validated_fit_rows(
                        scope["fit_row_ids"],
                        row_count=len(modeling_data),
                        label=f"{scope_id} preflight fit rows",
                    )
                ),
                fit_modeling_content_sha256=_fit_modeling_content_sha256(
                    modeling_data=modeling_data,
                    fit_rows=_validated_fit_rows(
                        scope["fit_row_ids"],
                        row_count=len(modeling_data),
                        label=f"{scope_id} preflight fit rows",
                    ),
                    columns=_modeling_columns(config),
                ),
                fit_label_projection=_fit_label_projection(
                    modeling_data=modeling_data,
                    fit_rows=_validated_fit_rows(
                        scope["fit_row_ids"],
                        row_count=len(modeling_data),
                        label=f"{scope_id} preflight fit rows",
                    ),
                    columns=_modeling_columns(config),
                ),
                registry_content_sha256=registry_content_sha256,
                scope=scope,
                forbidden_paths=(source_dataset_path, global_embedding_cache_path),
                semantic_witness_scientific_config=(
                    semantic_witness_scientific_config
                ),
            )
            completed = _validate_preflight_scope_capability(
                manifest_path=temporary / PREFLIGHT_SCOPE_INPUT_MANIFEST,
                expected_scope=scope,
                expected_registry_content_sha256=registry_content_sha256,
                shared_modeling=_shared_modeling_frame,
                shared_modeling_reference=shared_modeling_reference,
                embedding_row_records=embedding_rows_by_id,
                expected_embedding_cache_identity=embedding_cache_identity,
                expected_shared_cache_reference_content_sha256=str(
                    shared_reference["content_sha256"]
                ),
                parent_modeling_data=modeling_data,
                parent_config=config,
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
        "shared_embedding_row_store_manifest": _file_registration(
            embedding_row_store_manifest_path,
            root,
        ),
        "shared_embedding_row_store_content_sha256": str(
            embedding_row_store_manifest["content_sha256"]
        ),
        "shared_embedding_row_block_count": len(
            embedding_row_store_manifest["rows"]
        ),
        "shared_embedding_row_blocks_materialized_once": True,
        "worker_global_cache_locator_supplied": False,
        "shared_modeling_reference": _file_registration(
            shared_modeling_reference_path,
            root,
        ),
        "shared_modeling_reference_content_sha256": str(
            shared_modeling_reference["content_sha256"]
        ),
        "one_scope_per_worker_payload": True,
        "one_physical_cache_shared_across_scopes": True,
        "per_scope_embedding_arrays_copied": False,
        "per_scope_chunk_texts_copied": False,
        "per_scope_full_cohort_modeling_copied": False,
        "one_shared_text_block_across_scopes": True,
        "shared_text_contains_labels": False,
        "per_scope_text_payload_count": 0,
        "per_scope_fit_label_projection_count": 1,
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
        expected_shared_modeling_reference=shared_modeling_reference,
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
    embedding_row_store_root: Path | str | None = None,
    shared_modeling_reference_path: Path | str | None = None,
    expected_shared_modeling_reference_content_sha256: str | None = None,
    expected_semantic_witness_scientific_config: Any | None = None,
    forbidden_paths: Sequence[Path] = (),
) -> AuthenticatedPreflightScopeInput:
    if shared_cache_reference_path is not None:
        raise ValueError(
            "scope workers must not receive a global cache reference path"
        )
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
        "shared_modeling_view",
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
        "fit_label_projection",
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
    if _GLOBAL_ROW_ID_COLUMN in columns:
        raise ValueError(
            "preflight scope-input modeling column collides with its row-ID column"
        )
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
    row_count = manifest.get("row_count")
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 1
    ):
        raise ValueError("preflight scope-input row count changed")
    if authority.get("dataset_row_count") != row_count:
        raise ValueError("preflight one-scope authority row count changed")
    fit_rows = _validated_fit_rows(
        scope.get("fit_row_ids"),
        row_count=row_count,
        label="preflight scope-input fit rows",
    )
    modeling_view = manifest.get("shared_modeling_view")
    modeling_view_fields = {
        "schema_version",
        "fit_modeling_content_sha256",
        "dataset_row_count",
        "allowed_row_ids",
        "allowed_row_order_sha256",
        "allowed_row_count",
        "peer_row_access_allowed",
        "per_scope_text_payload_count",
        "fit_label_projection_count",
        "fit_label_projection_schema",
        "nonfit_labels_stored",
        "nonfit_rows_returned_by_worker_api",
    }
    if (
        not isinstance(modeling_view, Mapping)
        or set(modeling_view) != modeling_view_fields
        or modeling_view.get("schema_version")
        != PREFLIGHT_SHARED_MODELING_VIEW_SCHEMA
        or modeling_view.get("dataset_row_count") != row_count
        or modeling_view.get("allowed_row_ids") != list(fit_rows)
        or modeling_view.get("allowed_row_order_sha256")
        != _sha256_json(list(fit_rows))
        or modeling_view.get("allowed_row_count") != len(fit_rows)
        or modeling_view.get("peer_row_access_allowed") is not False
        or modeling_view.get("per_scope_text_payload_count") != 0
        or modeling_view.get("fit_label_projection_count") != 1
        or modeling_view.get("fit_label_projection_schema")
        != PREFLIGHT_SCOPE_LABEL_PROJECTION_SCHEMA
        or modeling_view.get("nonfit_labels_stored") is not False
        or modeling_view.get("nonfit_rows_returned_by_worker_api") is not False
    ):
        raise ValueError("preflight shared modeling-view authority changed")
    fit_modeling_sha = _require_sha256(
        modeling_view.get("fit_modeling_content_sha256"),
        label="preflight fit-modeling content SHA",
    )
    modeling_reference_path = (
        root.parent / PREFLIGHT_SHARED_MODELING_REFERENCE
        if shared_modeling_reference_path is None
        else Path(shared_modeling_reference_path).absolute()
    )
    shared_modeling_reference, shared_modeling = (
        _validate_shared_modeling_reference(
            path=modeling_reference_path,
            expected_content_sha256=(
                expected_shared_modeling_reference_content_sha256
            ),
        )
    )
    if (
        shared_modeling_reference["row_count"] != row_count
        or shared_modeling_reference["text_column"] != columns[0]
    ):
        raise ValueError(
            "preflight shared modeling view differs from its cohort block"
        )
    selected_text = shared_modeling.iloc[list(fit_rows)][
        [_GLOBAL_ROW_ID_COLUMN, columns[0]]
    ].copy(deep=True)
    if selected_text[_GLOBAL_ROW_ID_COLUMN].tolist() != list(fit_rows):
        raise ValueError("preflight shared text fit row order changed")
    label_columns = [
        _GLOBAL_ROW_ID_COLUMN,
        columns[1],
        columns[2],
    ]
    fit_labels = _read_exact_parquet(
        paths["fit_label_projection"],
        expected_columns=label_columns,
        label="preflight fit-label projection",
    )
    raw_label_row_ids = fit_labels[_GLOBAL_ROW_ID_COLUMN].tolist()
    if any(
        isinstance(row_id, (bool, np.bool_))
        or not isinstance(row_id, (int, np.integer))
        for row_id in raw_label_row_ids
    ):
        raise ValueError(
            "preflight fit-label projection has a noninteger global row ID"
        )
    label_row_ids = tuple(map(int, raw_label_row_ids))
    if len(label_row_ids) != len(set(label_row_ids)):
        raise ValueError(
            "preflight fit-label projection has duplicate global row IDs"
        )
    if label_row_ids != fit_rows:
        raise ValueError("preflight fit-label projection row order changed")
    fit_values = pd.DataFrame(
        {
            columns[0]: selected_text[columns[0]].to_numpy(copy=True),
            columns[1]: fit_labels[columns[1]].to_numpy(copy=True),
            columns[2]: fit_labels[columns[2]].to_numpy(copy=True),
        }
    )
    if (
        len(fit_values) != len(fit_rows)
        or not bool(
            fit_values[config.text_column]
            .map(lambda value: isinstance(value, str) and bool(value))
            .all()
        )
        or fit_values[[config.treatment_column, config.outcome_column]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "preflight shared modeling view has missing fit data"
        )
    modeling = pd.DataFrame(
        {
            config.text_column: np.full(row_count, "", dtype=object),
            config.treatment_column: np.full(row_count, np.nan, dtype=float),
            config.outcome_column: np.full(row_count, np.nan, dtype=float),
        }
    )
    modeling.loc[list(fit_rows), columns] = fit_values.to_numpy(copy=True)
    if (
        _fit_modeling_content_sha256(
            modeling_data=modeling,
            fit_rows=fit_rows,
            columns=columns,
        )
        != fit_modeling_sha
    ):
        raise ValueError(
            "preflight shared text or fit-label content changed"
        )
    del shared_modeling
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
    cache_view = _validated_scoped_cache_view(
        manifest.get("embedding_cache_view"),
        fit_rows=fit_rows,
        expected_embedding_cache_identity=(
            parent_embedding_cache_identity
        ),
        expected_shared_cache_reference_content_sha256=(
            expected_shared_cache_reference_content_sha256
        ),
    )
    if (
        int(cache_view["logical_identity"].get("row_count", -1))
        != row_count
    ):
        raise ValueError(
            "preflight scoped view row count changed"
        )
    row_store_root = (
        root.parent
        if embedding_row_store_root is None
        else Path(embedding_row_store_root).absolute()
    )
    if (
        row_store_root.is_symlink()
        or not row_store_root.is_dir()
        or row_store_root.resolve(strict=True) != row_store_root
    ):
        raise ValueError(
            "preflight embedding-row capability root is invalid"
        )
    cache = _load_scoped_embedding_row_blocks(
        set_root=row_store_root,
        cache_view=cache_view,
        global_row_count=row_count,
        fit_rows=fit_rows,
    )
    if parent_embedding_cache is not None:
        if (
            parent_embedding_cache_identity is None
            or _authenticated_cache_identity(parent_embedding_cache)
            != dict(parent_embedding_cache_identity)
        ):
            raise ValueError(
                "trusted parent embedding cache identity changed"
            )
        parent_bound = parent_embedding_cache.bind_spent(
            fit_rows,
            tuple(
                modeling.iloc[list(fit_rows)][
                    config.text_column
                ].tolist()
            ),
        )
        observed_digests = tuple(
            record["source_row_scientific_sha256"]
            for record in cache_view["row_blocks"]
        )
        if observed_digests != tuple(
            parent_bound.exact_row_scientific_digests()
        ):
            raise ValueError(
                "preflight embedding-row capability differs from "
                "its trusted parent"
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
        actual = fit_values
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
        semantic_witness_scientific_config=(
            semantic_witness_scientific_config
        ),
    )


def _validate_preflight_scope_capability(
    *,
    manifest_path: Path,
    expected_scope: Mapping[str, Any],
    expected_registry_content_sha256: str,
    shared_modeling: pd.DataFrame,
    shared_modeling_reference: Mapping[str, Any],
    embedding_row_records: Mapping[int, Mapping[str, Any]],
    expected_embedding_cache_identity: Mapping[str, Any],
    expected_shared_cache_reference_content_sha256: str,
    parent_config: AppliedInferenceConfig | None,
    parent_modeling_data: pd.DataFrame | None,
    expected_semantic_witness_scientific_config: Any | None,
    forbidden_paths: Sequence[Path],
) -> AuthenticatedPreflightScopeCapability:
    """Authenticate a scope index without materializing its embedding rows."""

    path = Path(manifest_path).absolute()
    root = path.parent
    if (
        path.name != PREFLIGHT_SCOPE_INPUT_MANIFEST
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
    ):
        raise ValueError(
            "preflight scope capability manifest path is invalid"
        )
    manifest = _read_json(
        path,
        label="preflight scope capability manifest",
    )
    required = {
        "schema_version",
        "scope",
        "scope_binding_sha256",
        "registry_content_sha256",
        "row_count",
        "columns",
        "shared_modeling_view",
        "files",
        "embedding_cache_view",
        "semantic_witness_scientific_config_sha256",
        "nonfit_text_supplied",
        "nonfit_labels_supplied",
        "global_cache_path_supplied",
        "source_dataset_path_supplied",
        "content_sha256",
    }
    body = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "content_sha256"
    }
    scope = json.loads(_canonical_json(dict(expected_scope)))
    registry_sha = _require_sha256(
        expected_registry_content_sha256,
        label="preflight scope capability registry SHA",
    )
    if (
        set(manifest) != required
        or manifest.get("schema_version") != PREFLIGHT_SCOPE_INPUT_SCHEMA
        or manifest.get("scope") != scope
        or manifest.get("registry_content_sha256") != registry_sha
        or manifest.get("scope_binding_sha256")
        != _sha256_json(
            {
                "registry_content_sha256": registry_sha,
                "scope": scope,
            }
        )
        or manifest.get("nonfit_text_supplied") is not False
        or manifest.get("nonfit_labels_supplied") is not False
        or manifest.get("global_cache_path_supplied") is not False
        or manifest.get("source_dataset_path_supplied") is not False
        or manifest.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError(
            "preflight scope capability manifest is invalid"
        )
    row_count = manifest.get("row_count")
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count != int(shared_modeling_reference["row_count"])
    ):
        raise ValueError(
            "preflight scope capability row count changed"
        )
    fit_rows = _validated_fit_rows(
        scope.get("fit_row_ids"),
        row_count=row_count,
        label="preflight scope capability fit rows",
    )
    files = manifest.get("files")
    expected_file_keys = {
        "effective_config",
        "semantic_witness_scientific_config",
        "one_scope_authority",
        "fit_label_projection",
    }
    if not isinstance(files, Mapping) or set(files) != expected_file_keys:
        raise ValueError(
            "preflight scope capability files are incomplete"
        )
    paths = {
        key: _validate_registration(root, registration, label=key)
        for key, registration in files.items()
    }
    observed_files, observed_directories = _closed_tree_inventory(
        root,
        label="preflight scope capability",
    )
    expected_files = {
        PREFLIGHT_SCOPE_INPUT_MANIFEST,
        *(str(value["relative_path"]) for value in files.values()),
    }
    expected_directories = {
        Path(value).parent.as_posix()
        for value in expected_files
        if Path(value).parent.as_posix() != "."
    }
    if (
        observed_files != expected_files
        or observed_directories != expected_directories
    ):
        raise ValueError(
            "preflight scope capability contains unregistered entries"
        )
    config_payload = _read_json(
        paths["effective_config"],
        label="preflight scope capability config",
    )
    config = ExperimentConfig.from_dict(
        {"applied_inference": config_payload}
    ).applied_inference
    columns = manifest.get("columns")
    if (
        not isinstance(columns, list)
        or columns
        != [
            config.text_column,
            config.treatment_column,
            config.outcome_column,
        ]
    ):
        raise ValueError(
            "preflight scope capability columns changed"
        )
    if parent_config is not None and config_payload != _private_config_payload(
        config=parent_config,
        forbidden_paths=forbidden_paths,
    ):
        raise ValueError(
            "preflight scope capability config differs from its parent"
        )
    authority = _read_json(
        paths["one_scope_authority"],
        label="preflight scope capability authority",
    )
    authority_body = {
        key: copy.deepcopy(value)
        for key, value in authority.items()
        if key != "content_sha256"
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
        or authority.get("schema_version")
        != PREFLIGHT_ONE_SCOPE_AUTHORITY_SCHEMA
        or authority.get("registry_content_sha256") != registry_sha
        or authority.get("dataset_row_count") != row_count
        or authority.get("scope") != scope
        or authority.get("scope_binding_sha256")
        != manifest["scope_binding_sha256"]
        or authority.get("authorized_scope_count") != 1
        or authority.get("other_scope_definitions_supplied") is not False
        or authority.get("other_scope_row_identities_supplied") is not False
        or authority.get("content_sha256")
        != _sha256_json(authority_body)
    ):
        raise ValueError(
            "preflight scope capability authority changed"
        )
    from .review_spent_evidence_provider import (
        SemanticWitnessScientificConfig,
    )

    semantic = SemanticWitnessScientificConfig.from_mapping(
        _read_json(
            paths["semantic_witness_scientific_config"],
            label="preflight scope capability semantic config",
        ),
        label="preflight scope capability semantic config",
    )
    if (
        manifest.get("semantic_witness_scientific_config_sha256")
        != semantic.identity_sha256
    ):
        raise ValueError(
            "preflight scope capability semantic config changed"
        )
    if (
        expected_semantic_witness_scientific_config is not None
        and (
            type(expected_semantic_witness_scientific_config)
            is not SemanticWitnessScientificConfig
            or expected_semantic_witness_scientific_config.as_dict()
            != semantic.as_dict()
        )
    ):
        raise ValueError(
            "preflight scope capability semantic config differs "
            "from its parent"
        )
    modeling_view = manifest.get("shared_modeling_view")
    modeling_fields = {
        "schema_version",
        "fit_modeling_content_sha256",
        "dataset_row_count",
        "allowed_row_ids",
        "allowed_row_order_sha256",
        "allowed_row_count",
        "peer_row_access_allowed",
        "per_scope_text_payload_count",
        "fit_label_projection_count",
        "fit_label_projection_schema",
        "nonfit_labels_stored",
        "nonfit_rows_returned_by_worker_api",
    }
    if (
        not isinstance(modeling_view, Mapping)
        or set(modeling_view) != modeling_fields
        or modeling_view.get("schema_version")
        != PREFLIGHT_SHARED_MODELING_VIEW_SCHEMA
        or modeling_view.get("dataset_row_count") != row_count
        or modeling_view.get("allowed_row_ids") != list(fit_rows)
        or modeling_view.get("allowed_row_order_sha256")
        != _sha256_json(list(fit_rows))
        or modeling_view.get("allowed_row_count") != len(fit_rows)
        or modeling_view.get("peer_row_access_allowed") is not False
        or modeling_view.get("per_scope_text_payload_count") != 0
        or modeling_view.get("fit_label_projection_count") != 1
        or modeling_view.get("fit_label_projection_schema")
        != PREFLIGHT_SCOPE_LABEL_PROJECTION_SCHEMA
        or modeling_view.get("nonfit_labels_stored") is not False
        or modeling_view.get("nonfit_rows_returned_by_worker_api")
        is not False
    ):
        raise ValueError(
            "preflight scope capability modeling view changed"
        )
    label_columns = [
        _GLOBAL_ROW_ID_COLUMN,
        config.treatment_column,
        config.outcome_column,
    ]
    fit_labels = _read_exact_parquet(
        paths["fit_label_projection"],
        expected_columns=label_columns,
        label="preflight scope capability fit labels",
    )
    if (
        fit_labels[_GLOBAL_ROW_ID_COLUMN].tolist() != list(fit_rows)
        or fit_labels[
            [config.treatment_column, config.outcome_column]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "preflight scope capability fit labels changed"
        )
    selected_text = shared_modeling.iloc[list(fit_rows)][
        [shared_modeling_reference["text_column"]]
    ].reset_index(drop=True)
    fit_projection = pd.DataFrame(
        {
            config.text_column: selected_text[
                shared_modeling_reference["text_column"]
            ].to_numpy(copy=True),
            config.treatment_column: fit_labels[
                config.treatment_column
            ].to_numpy(copy=True),
            config.outcome_column: fit_labels[
                config.outcome_column
            ].to_numpy(copy=True),
        }
    )
    identity_frame = pd.DataFrame(
        {
            config.text_column: np.full(row_count, "", dtype=object),
            config.treatment_column: np.full(
                row_count,
                np.nan,
                dtype=float,
            ),
            config.outcome_column: np.full(
                row_count,
                np.nan,
                dtype=float,
            ),
        }
    )
    identity_frame.loc[list(fit_rows), columns] = (
        fit_projection.to_numpy(copy=True)
    )
    if modeling_view.get(
        "fit_modeling_content_sha256"
    ) != _fit_modeling_content_sha256(
        modeling_data=identity_frame,
        fit_rows=fit_rows,
        columns=columns,
    ):
        raise ValueError(
            "preflight scope capability fit modeling content changed"
        )
    if parent_modeling_data is not None:
        expected_values = parent_modeling_data.iloc[
            list(fit_rows)
        ][columns]
        if (
            fit_projection.to_dict("records")
            != expected_values.to_dict("records")
        ):
            raise ValueError(
                "preflight scope capability differs from its parent rows"
            )
    _validated_scoped_cache_view(
        manifest.get("embedding_cache_view"),
        fit_rows=fit_rows,
        expected_embedding_cache_identity=(
            expected_embedding_cache_identity
        ),
        expected_shared_cache_reference_content_sha256=(
            expected_shared_cache_reference_content_sha256
        ),
        expected_row_records=embedding_row_records,
    )
    if forbidden_paths:
        serialized = b"".join(
            (root / relative).read_bytes()
            for relative in sorted(observed_files)
            if not relative.endswith((".npy", ".parquet"))
        )
        for forbidden in forbidden_paths:
            if (
                str(forbidden.resolve(strict=False)).encode("utf-8")
                in serialized
            ):
                raise ValueError(
                    "preflight scope capability exposes a forbidden path"
                )
    return AuthenticatedPreflightScopeCapability(
        root=root,
        manifest=copy.deepcopy(manifest),
        scope=copy.deepcopy(scope),
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
    expected_shared_modeling_reference: Mapping[str, Any] | None = None,
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
        "shared_embedding_row_store_manifest",
        "shared_embedding_row_store_content_sha256",
        "shared_embedding_row_block_count",
        "shared_embedding_row_blocks_materialized_once",
        "worker_global_cache_locator_supplied",
        "shared_modeling_reference",
        "shared_modeling_reference_content_sha256",
        "one_scope_per_worker_payload",
        "one_physical_cache_shared_across_scopes",
        "per_scope_embedding_arrays_copied",
        "per_scope_chunk_texts_copied",
        "per_scope_full_cohort_modeling_copied",
        "one_shared_text_block_across_scopes",
        "shared_text_contains_labels",
        "per_scope_text_payload_count",
        "per_scope_fit_label_projection_count",
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
        or manifest.get("shared_embedding_row_blocks_materialized_once")
        is not True
        or manifest.get("worker_global_cache_locator_supplied") is not False
        or manifest.get("per_scope_embedding_arrays_copied") is not False
        or manifest.get("per_scope_chunk_texts_copied") is not False
        or manifest.get("per_scope_full_cohort_modeling_copied") is not False
        or manifest.get("one_shared_text_block_across_scopes") is not True
        or manifest.get("shared_text_contains_labels") is not False
        or manifest.get("per_scope_text_payload_count") != 0
        or manifest.get("per_scope_fit_label_projection_count") != 1
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
    source_row_count = int(
        shared_reference["logical_identity"].get("row_count", -1)
    )
    if source_row_count < 1:
        raise ValueError(
            "shared preflight cache reference has an invalid row count"
        )
    required_embedding_rows = tuple(
        sorted(
            {
                row_id
                for scope in expected
                for row_id in _validated_fit_rows(
                    scope["fit_row_ids"],
                    row_count=source_row_count,
                    label=(
                        f"{scope['scope_id']} preflight embedding rows"
                    ),
                )
            }
        )
    )
    if manifest.get("shared_embedding_row_block_count") != len(
        required_embedding_rows
    ):
        raise ValueError(
            "preflight scope-input set row-block count changed"
        )
    embedding_row_store_manifest_path = _validate_registration(
        set_root,
        manifest["shared_embedding_row_store_manifest"],
        label="shared preflight embedding-row store manifest",
    )
    if (
        embedding_row_store_manifest_path.name
        != PREFLIGHT_EMBEDDING_ROW_STORE_MANIFEST
    ):
        raise ValueError(
            "shared preflight embedding-row store layout changed"
        )
    embedding_row_store_manifest = _validate_embedding_row_store(
        set_root=set_root,
        manifest_path=embedding_row_store_manifest_path,
        expected_row_ids=required_embedding_rows,
        expected_row_count=source_row_count,
        expected_embedding_cache_identity=(
            shared_reference["logical_identity"]
        ),
        expected_shared_cache_reference_content_sha256=str(
            shared_reference["content_sha256"]
        ),
        parent_embedding_cache=(
            parent_embedding_cache
            if parent_modeling_data is not None
            and parent_config is not None
            else None
        ),
        parent_modeling_data=parent_modeling_data,
        text_column=(
            None if parent_config is None else parent_config.text_column
        ),
    )
    if (
        embedding_row_store_manifest["content_sha256"]
        != manifest["shared_embedding_row_store_content_sha256"]
    ):
        raise ValueError(
            "shared preflight embedding-row store identity changed"
        )
    embedding_row_records = {
        int(record["global_row_id"]): record
        for record in embedding_row_store_manifest["rows"]
    }
    shared_modeling_reference_path = _validate_registration(
        set_root,
        manifest["shared_modeling_reference"],
        label="shared preflight modeling reference",
    )
    if (
        shared_modeling_reference_path.name
        != PREFLIGHT_SHARED_MODELING_REFERENCE
    ):
        raise ValueError(
            "shared preflight modeling reference layout changed"
        )
    shared_modeling_reference, shared_modeling = (
        _validate_shared_modeling_reference(
            path=shared_modeling_reference_path,
            expected_content_sha256=str(
                manifest["shared_modeling_reference_content_sha256"]
            ),
            expected_reference=expected_shared_modeling_reference,
        )
    )
    if parent_modeling_data is not None:
        if parent_config is None:
            raise ValueError(
                "parent config is required with parent modeling data"
            )
        parent_table = _shared_modeling_table(
            modeling_data=parent_modeling_data,
            text_column=parent_config.text_column,
        )
        if not _shared_modeling_values_match(
            observed=shared_modeling,
            expected=parent_table,
            columns=shared_modeling_reference["stored_columns"],
        ):
            raise ValueError(
                "shared preflight text block differs from its parent cohort"
            )
    authenticated: dict[str, AuthenticatedPreflightScopeCapability] = {}
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
        child = _validate_preflight_scope_capability(
            manifest_path=child_manifest,
            expected_scope=scope,
            expected_registry_content_sha256=expected_registry_content_sha256,
            shared_modeling=shared_modeling,
            shared_modeling_reference=shared_modeling_reference,
            embedding_row_records=embedding_row_records,
            expected_embedding_cache_identity=(
                shared_reference["logical_identity"]
            ),
            expected_shared_cache_reference_content_sha256=str(
                shared_reference["content_sha256"]
            ),
            parent_modeling_data=parent_modeling_data,
            parent_config=parent_config,
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
        PREFLIGHT_SHARED_MODELING_REFERENCE,
        PREFLIGHT_EMBEDDING_ROW_STORE_MANIFEST,
        _SHARED_MODELING_FILE,
        *(
            str(record[registration]["relative_path"])
            for record in embedding_row_store_manifest["rows"]
            for registration in (
                "embedding_block",
                "chunk_text_block",
            )
        ),
    }
    expected_directories: set[str] = {_EMBEDDING_ROW_DIRECTORY}
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
        shared_modeling_reference_path=shared_modeling_reference_path,
        shared_modeling_reference=shared_modeling_reference,
        embedding_row_store_manifest_path=(
            embedding_row_store_manifest_path
        ),
        embedding_row_store_manifest=embedding_row_store_manifest,
    )


__all__ = [
    "AuthenticatedPreflightScopeInput",
    "AuthenticatedPreflightScopeCapability",
    "AuthenticatedPreflightScopeInputSet",
    "PREFLIGHT_EMBEDDING_ROW_BLOCK_SCHEMA",
    "PREFLIGHT_EMBEDDING_ROW_STORE_MANIFEST",
    "PREFLIGHT_EMBEDDING_ROW_STORE_SCHEMA",
    "PREFLIGHT_ONE_SCOPE_AUTHORITY_SCHEMA",
    "PREFLIGHT_SAFE_CACHE_SCIENTIFIC_METADATA_SCHEMA",
    "PREFLIGHT_SCOPE_LABEL_PROJECTION_SCHEMA",
    "PREFLIGHT_SCOPED_CACHE_METADATA_SCHEMA",
    "PREFLIGHT_SCOPED_CACHE_VIEW_SCHEMA",
    "PREFLIGHT_SHARED_CACHE_REFERENCE",
    "PREFLIGHT_SHARED_CACHE_REFERENCE_SCHEMA",
    "PREFLIGHT_SHARED_MODELING_REFERENCE",
    "PREFLIGHT_SHARED_MODELING_REFERENCE_SCHEMA",
    "PREFLIGHT_SHARED_MODELING_VIEW_SCHEMA",
    "PREFLIGHT_SCOPE_INPUT_MANIFEST",
    "PREFLIGHT_SCOPE_INPUT_SET_MANIFEST",
    "ScopedEmbeddingView",
    "publish_preflight_scope_inputs",
    "validate_preflight_scope_input",
    "validate_preflight_scope_input_set",
]
