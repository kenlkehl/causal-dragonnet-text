"""Complete, authenticated Stage 2 aggregation of native HTR token evidence.

The native token-attention package remains the scientific source of truth.
This module reads its already-sealed columnar arrays without mmap, derives one
readable semantic occurrence for every eligible non-special raw token, and
builds a compact aggregate/reverse-index layer.  No raw token array is copied
into the handoff or an LLM prompt.

The previous catalog projection was based on the bounded human-readable
summaries stored beside each chunk.  Those summaries are useful for inspection
but are intentionally not complete.  This version therefore derives the
Stage 2 semantic domain directly from the complete raw token sidecars.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .all_evidence_discovery_interfaces import (
    HETEROGENEITY_AXIS,
    HTR_NEURAL,
    OUTCOME_AXIS,
    TREATMENT_AXIS,
    canonical_json,
)
from .all_evidence_fusion import LEGACY_ALL_SOURCE
from .htr_native_proof_capture import _array_sha256
from .htr_stage2_semantic_aggregation import (
    _load_npy,
    _read_json,
    _validate_source_payload,
    _write_json,
    _write_npy,
)


HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA = (
    "production_htr_stage2_complete_semantic_aggregate_payload_v2"
)
HTR_STAGE2_AGGREGATE_BATCH_SCHEMA = (
    "production_htr_stage2_complete_semantic_aggregate_batch_v2"
)
HTR_STAGE2_CROSS_FOLD_AGGREGATE_SCHEMA = (
    "production_htr_stage2_complete_cross_fold_semantic_aggregate_v2"
)
HTR_STAGE2_MODEL_AGGREGATE_SCHEMA = (
    "production_htr_stage2_complete_model_facing_semantic_aggregate_v2"
)
HTR_STAGE2_FOLD_AGGREGATE_SCHEMA = (
    "production_htr_stage2_complete_fold_local_semantic_aggregate_v2"
)
HTR_STAGE2_REVERSE_INDEX_SCHEMA = (
    "production_htr_stage2_complete_semantic_reverse_index_v2"
)
HTR_STAGE2_SCOPE_MANIFEST_SCHEMA = (
    "production_htr_stage2_complete_semantic_scope_manifest_v2"
)
HTR_STAGE2_STORE_MANIFEST_SCHEMA = (
    "production_htr_stage2_complete_semantic_store_manifest_v2"
)
HTR_STAGE2_ARCHITECTURE_CHUNK_SCHEMA = (
    "production_htr_stage2_authenticated_architecture_chunk_reference_v2"
)
HTR_STAGE2_NORMALIZATION_SCHEMA = (
    "htr_complete_raw_token_nfkc_casefold_whitespace_alnum_v2"
)
HTR_STAGE2_CROSS_FOLD_CONSOLIDATION_SCHEMA = (
    "same_stage_objective_normalized_raw_token_across_disjoint_oof_folds_v2"
)
HTR_STAGE2_CONTEXT_POLICY_SCHEMA = (
    "highest_hierarchical_then_token_then_raw_coordinate_char_window_v2"
)
HTR_STAGE2_BATCHING_SCHEMA = (
    "canonical_stage_objective_byte_token_and_member_bounded_batches_v2"
)
HTR_STAGE2_RAW_OCCURRENCE_PARTITION_SCHEMA = (
    "padding_then_special_then_readable_then_non_readable_exhaustive_v2"
)

DEFAULT_MODEL_FACING_BATCH_BYTES = 28_000
DEFAULT_MODEL_FACING_TOKEN_UPPER_BOUND = 28_000
DEFAULT_MODEL_FACING_AGGREGATES_PER_BATCH = 3
DEFAULT_CONTEXT_WINDOWS_PER_AGGREGATE = 3
DEFAULT_CONTEXT_CHARACTER_RADIUS = 80

_SHA256_HEX = frozenset("0123456789abcdef")
_STAGE_ORDER = {"nuisance": 0, "effect_modifier": 1}
_REQUIRED_RAW_COLUMNS = frozenset(
    {
        "fit_note_position",
        "fit_row_id",
        "chunk_index",
        "token_position",
        "token_id",
        "decoded_token_text_utf8",
        "decoded_token_text_byte_offsets",
        "char_start",
        "char_end",
        "is_special_token",
        "is_padding",
        "token_attention",
        "chunk_attention",
        "hierarchical_attention_score",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite numeric data")
    return result


def normalize_htr_complete_readable_token(value: str) -> tuple[str, str]:
    """Return the deterministic semantic key for one raw tokenizer token."""

    if not isinstance(value, str):
        raise TypeError("HTR decoded token text must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    kind = (
        "continuation_subword"
        if normalized.startswith("##")
        else "whole_or_initial_token"
    )
    return normalized, kind


def _is_readable_normalized_token(value: str) -> bool:
    return bool(value) and any(character.isalnum() for character in value)


def _array_path(root: Path, name: str) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name)
    ):
        raise ValueError("HTR raw array name is unsafe")
    path = root / f"{name}.npy"
    if path.is_symlink() or not path.is_file() or path.parent.resolve(strict=True) != root:
        raise ValueError("HTR raw array path is not a canonical regular file")
    return path


def _load_source_array(
    *,
    root: Path,
    registration: Mapping[str, Any],
    expected_length: int | None,
) -> np.ndarray:
    if set(registration) != {"array", "content_sha256", "dtype", "shape"}:
        raise ValueError("HTR raw column registration is not a closed schema")
    name = str(registration["array"])
    expected_shape = registration["shape"]
    if (
        not isinstance(expected_shape, list)
        or len(expected_shape) != 1
        or isinstance(expected_shape[0], bool)
        or not isinstance(expected_shape[0], int)
        or int(expected_shape[0]) < 0
        or (
            expected_length is not None
            and int(expected_shape[0]) != int(expected_length)
        )
    ):
        raise ValueError("HTR raw column shape is invalid")
    path = _array_path(root, name)
    # Deliberately bounded/non-mmap for Python 3.14 shared-filesystem safety.
    value = np.load(path, allow_pickle=False, mmap_mode=None)
    if (
        value.dtype.hasobject
        or value.ndim != 1
        or value.dtype.str != registration["dtype"]
        or list(value.shape) != expected_shape
        or _array_sha256(value) != registration["content_sha256"]
    ):
        raise ValueError(f"HTR raw array does not authenticate: {name}")
    return np.ascontiguousarray(value)


def _decode_utf8_occurrences(
    payload: np.ndarray,
    offsets: np.ndarray,
    occurrence_indices: np.ndarray,
) -> tuple[str, ...]:
    if (
        payload.dtype != np.dtype(np.uint8)
        or offsets.dtype != np.dtype(np.int64)
        or payload.ndim != 1
        or offsets.ndim != 1
        or offsets.size < 1
        or int(offsets[0]) != 0
        or int(offsets[-1]) != int(payload.size)
        or np.any(offsets[1:] < offsets[:-1])
        or occurrence_indices.ndim != 1
        or (
            occurrence_indices.size
            and (
                int(occurrence_indices.min()) < 0
                or int(occurrence_indices.max()) + 1 >= offsets.size
            )
        )
    ):
        raise ValueError("HTR decoded-token UTF-8 table is malformed")
    raw = payload.tobytes()
    try:
        return tuple(
            raw[int(offsets[index]) : int(offsets[index + 1])].decode("utf-8")
            for index in occurrence_indices.tolist()
        )
    except UnicodeDecodeError as exc:
        raise ValueError("HTR decoded-token UTF-8 data is invalid") from exc


@dataclass(frozen=True)
class _BatchIdentity:
    batch_index: int
    stage: str
    objective: str
    fold: int
    raw_start: int
    raw_count: int
    content_sha256: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.stage, self.objective, self.fold)


@dataclass(frozen=True)
class _LoadedBatch:
    identity: _BatchIdentity
    registration: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    decoded_text_by_token_id: Mapping[int, str]
    normalized_key_by_token_id: Mapping[int, tuple[str, str] | None]


@dataclass
class _MutableSummary:
    occurrence_count: int
    unique_notes: set[int]
    unique_chunks: set[tuple[int, int]]
    token_values: list[np.ndarray]
    chunk_values: list[np.ndarray]
    hierarchical_values: list[np.ndarray]
    note_token_max: dict[int, float]
    note_chunk_max: dict[int, float]
    note_hierarchical_max: dict[int, float]
    display_variants: set[str]
    contexts: list[dict[str, Any]]


@dataclass(frozen=True)
class _ComputedAggregation:
    fold_records: tuple[Mapping[str, Any], ...]
    cross_records: tuple[Mapping[str, Any], ...]
    reverse_raw_occurrence_index: np.ndarray
    aggregate_offsets: np.ndarray
    source_batches: tuple[Mapping[str, Any], ...]
    raw_partition: Mapping[str, Any]
    stage_objective_fold_counts: tuple[Mapping[str, Any], ...]
    source_tokenizer_identity_sha256: str
    normalization_checks: Mapping[str, Any]


@dataclass(frozen=True)
class HtrSemanticAggregationResult:
    payload: Mapping[str, Any]
    scope_manifest: Mapping[str, Any]
    scope_manifest_path: Path


def _ordered_fold_batches(package: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_batches = package.get("fold_batches")
    if not isinstance(raw_batches, list) or not raw_batches:
        raise ValueError("HTR complete semantic source has no fold batches")
    batches = tuple(copy.deepcopy(dict(row)) for row in raw_batches)
    ordered = tuple(
        sorted(
            batches,
            key=lambda row: (
                _STAGE_ORDER.get(str(row.get("stage")), 99),
                str(row.get("objective")),
                int(row.get("fold", -1)),
            ),
        )
    )
    if batches != ordered:
        raise ValueError("HTR raw fold batches are not in canonical order")
    tokenizer_hashes: set[str] = set()
    for batch in ordered:
        columns = batch.get("columns")
        tokenizer = batch.get("tokenizer_identity")
        if (
            not isinstance(columns, Mapping)
            or set(columns) != _REQUIRED_RAW_COLUMNS
            or not isinstance(tokenizer, Mapping)
            or batch.get("raw_occurrence_order")
            != "fit_note_position_then_chunk_index_then_token_position_v1"
            or batch.get("decoded_token_text_encoding")
            != "concatenated_utf8_with_offsets_v1"
        ):
            raise ValueError("HTR raw fold batch lacks its complete columnar schema")
        tokenizer_hashes.add(_sha256_json(tokenizer))
    if len(tokenizer_hashes) != 1:
        raise ValueError("HTR fold batches do not share one fitted tokenizer identity")
    return ordered


def _architecture_chunk_lookup(
    evidence: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int, int, int], Mapping[str, Any]]:
    result: dict[tuple[str, str, int, int, int], Mapping[str, Any]] = {}
    for row in evidence:
        key = (
            str(row["stage"]),
            str(row["objective"]),
            int(row["fold"]),
            int(row["fit_note_position"]),
            int(row["chunk_index"]),
        )
        if key in result:
            raise ValueError("HTR source duplicates an architecture chunk coordinate")
        result[key] = row
    return result


def _load_fold_batch(
    *,
    root: Path,
    raw_batch: Mapping[str, Any],
    identity: _BatchIdentity,
    decoded_text_by_token_id: dict[int, str],
) -> _LoadedBatch:
    columns = raw_batch["columns"]
    count = int(identity.raw_count)
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(_REQUIRED_RAW_COLUMNS):
        expected = (
            None
            if name == "decoded_token_text_utf8"
            else count + 1
            if name == "decoded_token_text_byte_offsets"
            else count
        )
        arrays[name] = _load_source_array(
            root=root,
            registration=columns[name],
            expected_length=expected,
        )
    token_ids = arrays["token_id"]
    unique_ids, first_indices = np.unique(token_ids, return_index=True)
    decoded = _decode_utf8_occurrences(
        arrays["decoded_token_text_utf8"],
        arrays["decoded_token_text_byte_offsets"],
        first_indices,
    )
    normalized: dict[int, tuple[str, str] | None] = {}
    for raw_id, text in zip(unique_ids.tolist(), decoded, strict=True):
        token_id = int(raw_id)
        prior = decoded_text_by_token_id.get(token_id)
        if prior is not None and prior != text:
            raise ValueError("one fitted tokenizer ID has multiple decoded texts")
        decoded_text_by_token_id[token_id] = text
        semantic_key = normalize_htr_complete_readable_token(text)
        normalized[token_id] = (
            semantic_key
            if _is_readable_normalized_token(semantic_key[0])
            else None
        )
    return _LoadedBatch(
        identity=identity,
        registration=raw_batch,
        arrays=arrays,
        decoded_text_by_token_id={
            int(raw_id): text
            for raw_id, text in zip(unique_ids.tolist(), decoded, strict=True)
        },
        normalized_key_by_token_id=normalized,
    )


def _token_semantic_keys(
    loaded: _LoadedBatch,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, str] | None, ...]]:
    token_ids = loaded.arrays["token_id"]
    unique_ids, inverse = np.unique(token_ids, return_inverse=True)
    keys = tuple(
        loaded.normalized_key_by_token_id[int(token_id)]
        for token_id in unique_ids.tolist()
    )
    return unique_ids, inverse, keys


def _context_window(
    *,
    text: str,
    start: int,
    end: int,
    radius: int,
) -> str:
    left = max(0, int(start) - int(radius))
    right = min(len(text), int(end) + int(radius))
    if left > 0:
        whitespace = text.find(" ", left, int(start))
        if whitespace >= 0:
            left = whitespace + 1
    if right < len(text):
        whitespace = text.rfind(" ", int(end), right)
        if whitespace >= int(end):
            right = whitespace
    value = " ".join(text[left:right].split())
    if not value:
        value = text[int(start) : int(end)]
    return value


def _summary(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("HTR aggregate attention summary is invalid")
    ordered = tuple(float(value) for value in values.tolist())
    total = math.fsum(ordered)
    return {
        "count": len(ordered),
        "sum": total,
        "mean": total / len(ordered),
        "max": max(ordered),
    }


def _note_max_summary(note_ids: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    if (
        note_ids.shape != values.shape
        or note_ids.ndim != 1
        or note_ids.size < 1
    ):
        raise ValueError("HTR note-level maximum inputs are misaligned")
    boundaries = np.flatnonzero(
        np.r_[True, note_ids[1:] != note_ids[:-1]]
    )
    maxima = np.maximum.reduceat(values, boundaries)
    ordered = tuple(float(value) for value in maxima.tolist())
    total = math.fsum(ordered)
    return {
        "note_count": len(ordered),
        "sum_of_note_maxima": total,
        "mean_of_note_maxima": total / len(ordered),
        "max_of_note_maxima": max(ordered),
    }


def _attention_summaries(
    *,
    note_ids: np.ndarray,
    token: np.ndarray,
    chunk: np.ndarray,
    hierarchical: np.ndarray,
) -> dict[str, Any]:
    return {
        "token_attention": {
            "occurrence_weighted": _summary(token),
            "note_level_max": _note_max_summary(note_ids, token),
        },
        "chunk_attention": {
            "occurrence_weighted": _summary(chunk),
            "note_level_max": _note_max_summary(note_ids, chunk),
        },
        "hierarchical_attention_score": {
            "occurrence_weighted": _summary(hierarchical),
            "note_level_max": _note_max_summary(note_ids, hierarchical),
        },
    }


def _merge_metric_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    occurrence_rows = [
        row["attention_summaries"][metric]["occurrence_weighted"]
        for row in rows
    ]
    note_rows = [
        row["attention_summaries"][metric]["note_level_max"]
        for row in rows
    ]
    occurrence_count = sum(int(row["count"]) for row in occurrence_rows)
    occurrence_sum = math.fsum(float(row["sum"]) for row in occurrence_rows)
    note_count = sum(int(row["note_count"]) for row in note_rows)
    note_sum = math.fsum(float(row["sum_of_note_maxima"]) for row in note_rows)
    return {
        "occurrence_weighted": {
            "count": occurrence_count,
            "sum": occurrence_sum,
            "mean": occurrence_sum / occurrence_count,
            "max": max(float(row["max"]) for row in occurrence_rows),
        },
        "note_level_max": {
            "note_count": note_count,
            "sum_of_note_maxima": note_sum,
            "mean_of_note_maxima": note_sum / note_count,
            "max_of_note_maxima": max(
                float(row["max_of_note_maxima"]) for row in note_rows
            ),
        },
    }


def _batch_coordinate_checks(
    *,
    loaded: _LoadedBatch,
    chunk_lookup: Mapping[
        tuple[str, str, int, int, int],
        Mapping[str, Any],
    ],
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    arrays = loaded.arrays
    count = int(loaded.identity.raw_count)
    note = arrays["fit_note_position"]
    row_id = arrays["fit_row_id"]
    chunk = arrays["chunk_index"]
    token_position = arrays["token_position"]
    token_attention = arrays["token_attention"]
    chunk_attention = arrays["chunk_attention"]
    hierarchical = arrays["hierarchical_attention_score"]
    special = arrays["is_special_token"].astype(bool, copy=False)
    padding = arrays["is_padding"].astype(bool, copy=False)
    char_start = arrays["char_start"]
    char_end = arrays["char_end"]
    if (
        count < 1
        or not all(value.ndim == 1 for value in arrays.values())
        or not np.isfinite(token_attention).all()
        or not np.isfinite(chunk_attention).all()
        or not np.isfinite(hierarchical).all()
        or np.any(token_attention < 0.0)
        or np.any(chunk_attention < 0.0)
        or np.any(char_start < 0)
        or np.any(char_end < char_start)
        or not np.allclose(
            hierarchical,
            token_attention * chunk_attention,
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ValueError("HTR raw attention columns are invalid")
    if np.any(note[1:] < note[:-1]):
        raise ValueError("HTR raw note order is not canonical")
    same_note = note[1:] == note[:-1]
    if np.any(chunk[1:][same_note] < chunk[:-1][same_note]):
        raise ValueError("HTR raw chunk order is not canonical")
    same_chunk = same_note & (chunk[1:] == chunk[:-1])
    if np.any(token_position[1:][same_chunk] <= token_position[:-1][same_chunk]):
        raise ValueError("HTR raw token positions are not strictly ordered")

    chunk_starts = np.flatnonzero(
        np.r_[
            True,
            (note[1:] != note[:-1]) | (chunk[1:] != chunk[:-1]),
        ]
    )
    chunk_stops = np.r_[chunk_starts[1:], count]
    token_sums = np.add.reduceat(token_attention, chunk_starts)
    if not np.allclose(token_sums, 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("HTR raw token weights do not normalize by chunk")
    first_chunk_weights = chunk_attention[chunk_starts]
    if np.any(
        np.maximum.reduceat(chunk_attention, chunk_starts)
        - np.minimum.reduceat(chunk_attention, chunk_starts)
        > 1e-15
    ):
        raise ValueError("HTR chunk weight changes within one token inventory")

    note_chunk_positions = note[chunk_starts]
    note_starts = np.flatnonzero(
        np.r_[True, note_chunk_positions[1:] != note_chunk_positions[:-1]]
    )
    if not np.allclose(
        np.add.reduceat(first_chunk_weights, note_starts),
        1.0,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("HTR chunk weights do not normalize by note")

    expected_note_positions = tuple(
        int(value) for value in loaded.registration["fit_note_positions"]
    )
    expected_row_ids = tuple(
        int(value) for value in loaded.registration["fit_row_ids"]
    )
    observed_note_positions = tuple(
        int(value) for value in note[chunk_starts[note_starts]].tolist()
    )
    observed_row_ids = tuple(
        int(value) for value in row_id[chunk_starts[note_starts]].tolist()
    )
    if (
        observed_note_positions != expected_note_positions
        or observed_row_ids != expected_row_ids
    ):
        raise ValueError("HTR raw note coordinates differ from the fold manifest")

    stage, objective, fold = loaded.identity.key
    for start, stop in zip(chunk_starts.tolist(), chunk_stops.tolist(), strict=True):
        key = (
            stage,
            objective,
            fold,
            int(note[start]),
            int(chunk[start]),
        )
        source = chunk_lookup.get(key)
        if source is None:
            raise ValueError("HTR raw chunk lacks an authenticated chunk record")
        text = str(source["chunk_text"])
        if (
            int(row_id[start]) != int(source["fit_row_id"])
            or not np.all(row_id[start:stop] == row_id[start])
            or int(token_position[start]) != 0
            or int(token_position[stop - 1]) != stop - start - 1
            or int(char_end[start:stop].max(initial=0)) > len(text)
            or not math.isclose(
                float(first_chunk_weights[
                    int(np.searchsorted(chunk_starts, start))
                ]),
                float(source["attention"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("HTR raw token coordinates do not join to the chunk")

    raw_special_count = int(np.count_nonzero(special))
    raw_padding_count = int(np.count_nonzero(padding))
    raw_special_mass = math.fsum(
        float(value) for value in token_attention[special].tolist()
    )
    if (
        raw_special_count
        != int(loaded.registration["special_token_occurrence_count"])
        or raw_padding_count
        != int(loaded.registration["padding_occurrence_count"])
        or not math.isclose(
            raw_special_mass,
            float(loaded.registration["special_token_attention_mass"]),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    ):
        raise ValueError("HTR raw special/padding accounting changed")
    return {
        "chunk_group_count": len(chunk_starts),
        "note_group_count": len(note_starts),
        "token_normalization_valid": True,
        "chunk_normalization_valid": True,
        "hierarchical_product_valid": True,
        "coordinate_join_valid": True,
    }


def _unique_chunk_count(note: np.ndarray, chunk: np.ndarray) -> int:
    if note.shape != chunk.shape or note.ndim != 1 or note.size < 1:
        raise ValueError("HTR unique-chunk inputs are invalid")
    return 1 + int(
        np.count_nonzero(
            (note[1:] != note[:-1]) | (chunk[1:] != chunk[:-1])
        )
    )


def _best_contexts(
    *,
    loaded: _LoadedBatch,
    source_indices: np.ndarray,
    chunk_lookup: Mapping[
        tuple[str, str, int, int, int],
        Mapping[str, Any],
    ],
    limit: int,
    radius: int,
) -> list[dict[str, Any]]:
    arrays = loaded.arrays
    if source_indices.size < 1:
        return []
    order = np.lexsort(
        (
            source_indices,
            arrays["token_position"][source_indices],
            arrays["chunk_index"][source_indices],
            arrays["fit_row_id"][source_indices],
            -arrays["token_attention"][source_indices],
            -arrays["hierarchical_attention_score"][source_indices],
        )
    )
    stage, objective, fold = loaded.identity.key
    contexts: list[dict[str, Any]] = []
    for source_index in source_indices[order[:limit]].tolist():
        note_position = int(arrays["fit_note_position"][source_index])
        chunk_index = int(arrays["chunk_index"][source_index])
        source = chunk_lookup[
            (stage, objective, fold, note_position, chunk_index)
        ]
        start = int(arrays["char_start"][source_index])
        end = int(arrays["char_end"][source_index])
        chunk_text = str(source["chunk_text"])
        body = {
            "schema_version": HTR_STAGE2_ARCHITECTURE_CHUNK_SCHEMA,
            "fold": fold,
            "fit_note_position": note_position,
            "fit_row_id": int(arrays["fit_row_id"][source_index]),
            "chunk_index": chunk_index,
            "chunk_text_sha256": str(source["chunk_sha256"]),
            "token_position": int(arrays["token_position"][source_index]),
            "char_start": start,
            "char_end": end,
            "display_text": _context_window(
                text=chunk_text,
                start=start,
                end=end,
                radius=radius,
            ),
            "exact_focus_text": chunk_text[start:end],
            "token_attention": float(
                arrays["token_attention"][source_index]
            ),
            "chunk_attention": float(
                arrays["chunk_attention"][source_index]
            ),
            "hierarchical_attention_score": float(
                arrays["hierarchical_attention_score"][source_index]
            ),
            "source_raw_occurrence_index": (
                int(loaded.identity.raw_start) + int(source_index)
            ),
        }
        contexts.append({**body, "content_sha256": _sha256_json(body)})
    return contexts


def _local_aggregate_rows(
    *,
    loaded: _LoadedBatch,
    key_to_dynamic: dict[tuple[str, str, str, str], int],
    dynamic_to_key: list[tuple[str, str, str, str]],
    variants_by_dynamic: dict[int, set[str]],
    chunk_lookup: Mapping[
        tuple[str, str, int, int, int],
        Mapping[str, Any],
    ],
    context_limit: int,
    context_radius: int,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    Mapping[str, Any],
]:
    arrays = loaded.arrays
    unique_ids, inverse, semantic_keys = _token_semantic_keys(loaded)
    dynamic_for_unique = np.full(unique_ids.size, -1, dtype=np.int32)
    stage, objective, fold = loaded.identity.key
    for position, (token_id, semantic_key) in enumerate(
        zip(unique_ids.tolist(), semantic_keys, strict=True)
    ):
        if semantic_key is None:
            continue
        normalized, wordpiece_kind = semantic_key
        cross_key = (stage, objective, normalized, wordpiece_kind)
        dynamic = key_to_dynamic.get(cross_key)
        if dynamic is None:
            dynamic = len(dynamic_to_key)
            key_to_dynamic[cross_key] = dynamic
            dynamic_to_key.append(cross_key)
        dynamic_for_unique[position] = dynamic
        variants_by_dynamic.setdefault(dynamic, set()).add(
            loaded.decoded_text_by_token_id[int(token_id)]
        )
    dynamic_by_occurrence = dynamic_for_unique[inverse]
    special = arrays["is_special_token"].astype(bool, copy=False)
    padding = arrays["is_padding"].astype(bool, copy=False)
    valid_offset = arrays["char_end"] > arrays["char_start"]
    eligible = (
        (~special)
        & (~padding)
        & valid_offset
        & (dynamic_by_occurrence >= 0)
    )
    special_only = special & (~padding)
    non_readable = (~padding) & (~special) & (~eligible)
    partition = {
        "raw_token_occurrence_count": int(loaded.identity.raw_count),
        "readable_token_occurrence_count": int(np.count_nonzero(eligible)),
        "special_token_occurrence_count": int(np.count_nonzero(special_only)),
        "padding_occurrence_count": int(np.count_nonzero(padding)),
        "special_and_padding_overlap_count": int(
            np.count_nonzero(special & padding)
        ),
        "non_readable_token_occurrence_count": int(
            np.count_nonzero(non_readable)
        ),
        "special_token_attention_mass": math.fsum(
            float(value)
            for value in arrays["token_attention"][special].tolist()
        ),
        "padding_token_attention_mass": math.fsum(
            float(value)
            for value in arrays["token_attention"][padding].tolist()
        ),
        "non_readable_token_attention_mass": math.fsum(
            float(value)
            for value in arrays["token_attention"][non_readable].tolist()
        ),
    }
    exclusive_total = (
        partition["readable_token_occurrence_count"]
        + partition["special_token_occurrence_count"]
        + partition["padding_occurrence_count"]
        + partition["non_readable_token_occurrence_count"]
    )
    if exclusive_total != loaded.identity.raw_count:
        raise RuntimeError("HTR raw occurrence partition is not exhaustive")

    source_indices = np.flatnonzero(eligible)
    if source_indices.size < 1:
        raise ValueError("HTR fold batch has no readable raw token occurrence")
    dynamic = dynamic_by_occurrence[source_indices]
    order = np.argsort(dynamic, kind="stable")
    source_indices = source_indices[order]
    dynamic = dynamic[order]
    boundaries = np.flatnonzero(np.r_[True, dynamic[1:] != dynamic[:-1]])
    stops = np.r_[boundaries[1:], source_indices.size]
    rows: list[dict[str, Any]] = []
    for start, stop in zip(boundaries.tolist(), stops.tolist(), strict=True):
        selected = source_indices[start:stop]
        dynamic_id = int(dynamic[start])
        row_note = arrays["fit_row_id"][selected]
        row_chunk = arrays["chunk_index"][selected]
        token = arrays["token_attention"][selected]
        chunk_attention = arrays["chunk_attention"][selected]
        hierarchical = arrays["hierarchical_attention_score"][selected]
        cross_key = dynamic_to_key[dynamic_id]
        identity_body = {
            "schema_version": HTR_STAGE2_FOLD_AGGREGATE_SCHEMA,
            "stage": stage,
            "objective": objective,
            "fold": fold,
            "normalized_focus_text": cross_key[2],
            "wordpiece_kind": cross_key[3],
            "raw_occurrence_projection": (
                "one_complete_eligible_raw_token_occurrence_v2"
            ),
        }
        row = {
            **identity_body,
            "fold_aggregate_id": (
                f"htr_complete_fold_aggregate_{_sha256_json(identity_body)}"
            ),
            "occurrence_count": int(selected.size),
            "raw_token_occurrence_count": int(selected.size),
            "unique_note_count": int(np.unique(row_note).size),
            "unique_chunk_count": _unique_chunk_count(
                row_note,
                row_chunk,
            ),
            "attention_summaries": _attention_summaries(
                note_ids=row_note,
                token=token,
                chunk=chunk_attention,
                hierarchical=hierarchical,
            ),
            "display_text_variants": sorted(
                variants_by_dynamic[dynamic_id]
            ),
            "context_windows": _best_contexts(
                loaded=loaded,
                source_indices=selected,
                chunk_lookup=chunk_lookup,
                limit=context_limit,
                radius=context_radius,
            ),
            "context_window_policy": HTR_STAGE2_CONTEXT_POLICY_SCHEMA,
            "all_eligible_occurrences_included": True,
            "_dynamic_id": dynamic_id,
        }
        public = {key: value for key, value in row.items() if key != "_dynamic_id"}
        row["content_sha256"] = _sha256_json(public)
        rows.append(row)
    return rows, dynamic_by_occurrence, partition


def _merge_cross_fold_rows(
    *,
    dynamic_to_key: Sequence[tuple[str, str, str, str]],
    local_rows: Sequence[Mapping[str, Any]],
    context_limit: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, int],
]:
    rows_by_dynamic: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in local_rows:
        rows_by_dynamic[int(row["_dynamic_id"])].append(row)
    canonical_dynamic = sorted(
        rows_by_dynamic,
        key=lambda dynamic: (
            _STAGE_ORDER[dynamic_to_key[dynamic][0]],
            dynamic_to_key[dynamic][1],
            dynamic_to_key[dynamic][2],
            dynamic_to_key[dynamic][3],
        ),
    )
    cross_index_by_dynamic = {
        dynamic: index for index, dynamic in enumerate(canonical_dynamic)
    }
    cross_records: list[dict[str, Any]] = []
    public_local_records: list[dict[str, Any]] = []
    cross_offset = 0
    for dynamic in canonical_dynamic:
        stage, objective, normalized, wordpiece_kind = dynamic_to_key[dynamic]
        fold_rows = sorted(
            rows_by_dynamic[dynamic],
            key=lambda row: int(row["fold"]),
        )
        seen_folds: set[int] = set()
        local_offset = cross_offset
        fold_support: list[dict[str, Any]] = []
        variants: set[str] = set()
        context_candidates: list[Mapping[str, Any]] = []
        for row in fold_rows:
            fold = int(row["fold"])
            if fold in seen_folds:
                raise ValueError("HTR semantic source duplicates a fold aggregate")
            seen_folds.add(fold)
            public = {
                key: copy.deepcopy(value)
                for key, value in row.items()
                if key not in {"_dynamic_id", "content_sha256"}
            }
            public["reverse_index_start"] = local_offset
            public["reverse_index_count"] = int(row["occurrence_count"])
            public["content_sha256"] = _sha256_json(public)
            public_local_records.append(public)
            fold_support.append(
                {
                    "fold": fold,
                    "fold_aggregate_id": str(public["fold_aggregate_id"]),
                    "fold_aggregate_content_sha256": str(
                        public["content_sha256"]
                    ),
                    "occurrence_count": int(public["occurrence_count"]),
                    "raw_token_occurrence_count": int(
                        public["raw_token_occurrence_count"]
                    ),
                    "unique_note_count": int(public["unique_note_count"]),
                    "unique_chunk_count": int(public["unique_chunk_count"]),
                }
            )
            variants.update(map(str, public["display_text_variants"]))
            context_candidates.extend(public["context_windows"])
            local_offset += int(public["occurrence_count"])
        occurrence_count = sum(
            int(row["occurrence_count"]) for row in fold_rows
        )
        identity_body = {
            "schema_version": HTR_STAGE2_CROSS_FOLD_AGGREGATE_SCHEMA,
            "stage": stage,
            "objective": objective,
            "normalized_focus_text": normalized,
            "wordpiece_kind": wordpiece_kind,
            "cross_fold_consolidation": (
                HTR_STAGE2_CROSS_FOLD_CONSOLIDATION_SCHEMA
            ),
            "raw_occurrence_projection": (
                "one_complete_eligible_raw_token_occurrence_v2"
            ),
        }
        ordered_contexts = sorted(
            context_candidates,
            key=lambda row: (
                -float(row["hierarchical_attention_score"]),
                -float(row["token_attention"]),
                int(row["fold"]),
                int(row["fit_row_id"]),
                int(row["chunk_index"]),
                int(row["token_position"]),
                int(row["source_raw_occurrence_index"]),
            ),
        )[:context_limit]
        cross = {
            **identity_body,
            "aggregate_id": (
                f"htr_complete_cross_fold_aggregate_"
                f"{_sha256_json(identity_body)}"
            ),
            "occurrence_count": occurrence_count,
            "raw_token_occurrence_count": occurrence_count,
            "unique_note_count": sum(
                int(row["unique_note_count"]) for row in fold_rows
            ),
            "unique_chunk_count": sum(
                int(row["unique_chunk_count"]) for row in fold_rows
            ),
            "attention_summaries": {
                metric: _merge_metric_summaries(fold_rows, metric=metric)
                for metric in (
                    "token_attention",
                    "chunk_attention",
                    "hierarchical_attention_score",
                )
            },
            "fold_support": fold_support,
            "folds_retained_before_consolidation": sorted(seen_folds),
            "display_text_variants": sorted(variants),
            "display_text_variant_count": len(variants),
            "display_text_variant_content_sha256": _sha256_json(
                sorted(variants)
            ),
            "context_windows": ordered_contexts,
            "context_window_policy": HTR_STAGE2_CONTEXT_POLICY_SCHEMA,
            "reverse_index_start": cross_offset,
            "reverse_index_count": occurrence_count,
            "overlap_accounting": {
                "raw_overlapping_chunk_occurrence_count": occurrence_count,
                "unique_supporting_note_count": sum(
                    int(row["unique_note_count"]) for row in fold_rows
                ),
                "unique_supporting_chunk_count": sum(
                    int(row["unique_chunk_count"]) for row in fold_rows
                ),
                "note_level_maxima_separate_from_occurrence_summaries": True,
            },
            "hierarchical_attention_interpretation": (
                "ranking_heuristic_not_causal_attribution"
            ),
            "all_eligible_occurrences_included": True,
        }
        cross["content_sha256"] = _sha256_json(cross)
        cross_records.append(cross)
        cross_offset += occurrence_count
    return public_local_records, cross_records, cross_index_by_dynamic


def _compute_complete_aggregation(
    *,
    source_payload: Mapping[str, Any],
    source_array_store_root: Path,
    context_limit: int,
    context_radius: int,
) -> _ComputedAggregation:
    evidence, package = _validate_source_payload(source_payload)
    package_body = {
        key: copy.deepcopy(value)
        for key, value in package.items()
        if key != "content_sha256"
    }
    if (
        package.get("content_sha256") != _sha256_json(package_body)
        or package.get("top_k_applied_to_raw_inventory") is not False
    ):
        raise ValueError("HTR token package content identity is invalid")
    array_root = Path(source_array_store_root)
    if (
        not array_root.is_absolute()
        or array_root.is_symlink()
        or array_root.resolve(strict=True) != array_root
        or not array_root.is_dir()
    ):
        raise ValueError("HTR raw array store root is not canonical")
    raw_batches = _ordered_fold_batches(package)
    chunk_lookup = _architecture_chunk_lookup(evidence)
    decoded_text_by_token_id: dict[int, str] = {}
    key_to_dynamic: dict[tuple[str, str, str, str], int] = {}
    dynamic_to_key: list[tuple[str, str, str, str]] = []
    variants_by_dynamic: dict[int, set[str]] = {}
    local_rows: list[dict[str, Any]] = []
    partitions: list[dict[str, Any]] = []
    source_batches: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    identities: list[_BatchIdentity] = []
    raw_cursor = 0
    note_ids_by_stage_objective: dict[
        tuple[str, str],
        set[int],
    ] = defaultdict(set)

    for batch_index, raw_batch in enumerate(raw_batches):
        count = _positive_int(
            raw_batch.get("token_occurrence_count"),
            label="HTR fold raw-token count",
        )
        identity = _BatchIdentity(
            batch_index=batch_index,
            stage=str(raw_batch["stage"]),
            objective=str(raw_batch["objective"]),
            fold=int(raw_batch["fold"]),
            raw_start=raw_cursor,
            raw_count=count,
            content_sha256=_require_sha256(
                raw_batch.get("content_sha256"),
                label="HTR raw fold batch",
            ),
        )
        identities.append(identity)
        loaded = _load_fold_batch(
            root=array_root,
            raw_batch=raw_batch,
            identity=identity,
            decoded_text_by_token_id=decoded_text_by_token_id,
        )
        checks = _batch_coordinate_checks(
            loaded=loaded,
            chunk_lookup=chunk_lookup,
        )
        rows, _dynamic, partition = _local_aggregate_rows(
            loaded=loaded,
            key_to_dynamic=key_to_dynamic,
            dynamic_to_key=dynamic_to_key,
            variants_by_dynamic=variants_by_dynamic,
            chunk_lookup=chunk_lookup,
            context_limit=context_limit,
            context_radius=context_radius,
        )
        local_rows.extend(rows)
        partitions.append(dict(partition))
        normalization_rows.append(
            {
                "stage": identity.stage,
                "objective": identity.objective,
                "fold": identity.fold,
                **checks,
            }
        )
        note_key = (identity.stage, identity.objective)
        row_ids = {int(value) for value in raw_batch["fit_row_ids"]}
        if note_ids_by_stage_objective[note_key].intersection(row_ids):
            raise ValueError("HTR cross-fold validation-note coverage overlaps")
        note_ids_by_stage_objective[note_key].update(row_ids)
        source_batches.append(
            {
                "schema_version": (
                    "production_htr_stage2_raw_fold_batch_reference_v2"
                ),
                "source_batch_index": batch_index,
                "stage": identity.stage,
                "objective": identity.objective,
                "fold": identity.fold,
                "global_raw_occurrence_start": raw_cursor,
                "raw_token_occurrence_count": count,
                "source_batch_content_sha256": identity.content_sha256,
                "tokenizer_identity_content_sha256": _sha256_json(
                    raw_batch["tokenizer_identity"]
                ),
                "column_content_sha256": {
                    name: str(raw_batch["columns"][name]["content_sha256"])
                    for name in sorted(_REQUIRED_RAW_COLUMNS)
                },
                "raw_arrays_remain_in_source_fit_state": True,
            }
        )
        raw_cursor += count

    objective_group_count = len(note_ids_by_stage_objective)
    note_interpretation_count = int(package["note_interpretation_count"])
    if (
        raw_cursor != int(package["token_occurrence_count"])
        or objective_group_count < 2
        or note_interpretation_count % objective_group_count
        or any(
            len(row_ids)
            != note_interpretation_count // objective_group_count
            for row_ids in note_ids_by_stage_objective.values()
        )
        or len(
            {
                tuple(sorted(row_ids))
                for row_ids in note_ids_by_stage_objective.values()
            }
        )
        != 1
    ):
        raise ValueError("HTR raw batches do not provide exact OOF note coverage")

    fold_records, cross_records, cross_index_by_dynamic = (
        _merge_cross_fold_rows(
            dynamic_to_key=dynamic_to_key,
            local_rows=local_rows,
            context_limit=context_limit,
        )
    )
    expected_readable_count = sum(
        int(record["occurrence_count"]) for record in cross_records
    )
    aggregate_offsets = np.zeros(len(cross_records) + 1, dtype=np.uint64)
    if cross_records:
        aggregate_offsets[1:] = np.cumsum(
            np.asarray(
                [int(row["occurrence_count"]) for row in cross_records],
                dtype=np.uint64,
            )
        )
    if int(aggregate_offsets[-1]) != expected_readable_count:
        raise RuntimeError("HTR aggregate offset accounting changed")
    raw_index_dtype = (
        np.uint32 if raw_cursor <= np.iinfo(np.uint32).max else np.uint64
    )
    reverse_raw_index = np.empty(
        expected_readable_count,
        dtype=raw_index_dtype,
    )
    write_cursor = np.asarray(aggregate_offsets[:-1], dtype=np.uint64).copy()
    dynamic_to_cross = np.full(len(dynamic_to_key), -1, dtype=np.int32)
    for dynamic, cross_index in cross_index_by_dynamic.items():
        dynamic_to_cross[int(dynamic)] = int(cross_index)

    decoded_replay: dict[int, str] = {}
    for raw_batch, identity in zip(raw_batches, identities, strict=True):
        loaded = _load_fold_batch(
            root=array_root,
            raw_batch=raw_batch,
            identity=identity,
            decoded_text_by_token_id=decoded_replay,
        )
        unique_ids, inverse, semantic_keys = _token_semantic_keys(loaded)
        dynamic_for_unique = np.full(unique_ids.size, -1, dtype=np.int32)
        for position, semantic_key in enumerate(semantic_keys):
            if semantic_key is None:
                continue
            dynamic_for_unique[position] = key_to_dynamic[
                (
                    identity.stage,
                    identity.objective,
                    semantic_key[0],
                    semantic_key[1],
                )
            ]
        dynamic = dynamic_for_unique[inverse]
        special = loaded.arrays["is_special_token"].astype(bool, copy=False)
        padding = loaded.arrays["is_padding"].astype(bool, copy=False)
        eligible = (
            (~special)
            & (~padding)
            & (loaded.arrays["char_end"] > loaded.arrays["char_start"])
            & (dynamic >= 0)
        )
        source_indices = np.flatnonzero(eligible)
        cross = dynamic_to_cross[dynamic[source_indices]]
        if np.any(cross < 0):
            raise RuntimeError("HTR readable occurrence lacks a cross aggregate")
        order = np.argsort(cross, kind="stable")
        source_indices = source_indices[order]
        cross = cross[order]
        boundaries = np.flatnonzero(np.r_[True, cross[1:] != cross[:-1]])
        stops = np.r_[boundaries[1:], source_indices.size]
        for start, stop in zip(boundaries.tolist(), stops.tolist(), strict=True):
            cross_index = int(cross[start])
            target_start = int(write_cursor[cross_index])
            target_stop = target_start + int(stop - start)
            reverse_raw_index[target_start:target_stop] = (
                np.asarray(source_indices[start:stop], dtype=np.uint64)
                + int(identity.raw_start)
            ).astype(raw_index_dtype, copy=False)
            write_cursor[cross_index] = target_stop
    if not np.array_equal(write_cursor, aggregate_offsets[1:]):
        raise RuntimeError("HTR reverse index did not cover every aggregate")

    readable_count = sum(
        int(row["readable_token_occurrence_count"]) for row in partitions
    )
    special_only_count = sum(
        int(row["special_token_occurrence_count"]) for row in partitions
    )
    padding_count = sum(
        int(row["padding_occurrence_count"]) for row in partitions
    )
    non_readable_count = sum(
        int(row["non_readable_token_occurrence_count"])
        for row in partitions
    )
    raw_special_flag_count = sum(
        int(batch["special_token_occurrence_count"]) for batch in raw_batches
    )
    special_mass = math.fsum(
        float(row["special_token_attention_mass"]) for row in partitions
    )
    if (
        readable_count != expected_readable_count
        or readable_count
        + special_only_count
        + padding_count
        + non_readable_count
        != raw_cursor
        or raw_special_flag_count != int(package["special_token_occurrence_count"])
        or padding_count != int(package["padding_occurrence_count"])
        or not math.isclose(
            special_mass,
            float(package["special_token_attention_mass"]),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    ):
        raise ValueError("HTR complete raw occurrence accounting changed")

    partition_body = {
        "schema_version": HTR_STAGE2_RAW_OCCURRENCE_PARTITION_SCHEMA,
        "raw_token_occurrence_count": raw_cursor,
        "readable_token_occurrence_count": readable_count,
        "special_nonpadding_token_occurrence_count": special_only_count,
        "raw_special_flag_occurrence_count": raw_special_flag_count,
        "padding_occurrence_count": padding_count,
        "special_and_padding_overlap_count": sum(
            int(row["special_and_padding_overlap_count"]) for row in partitions
        ),
        "non_readable_token_occurrence_count": non_readable_count,
        "special_token_attention_mass": special_mass,
        "padding_token_attention_mass": math.fsum(
            float(row["padding_token_attention_mass"]) for row in partitions
        ),
        "non_readable_token_attention_mass": math.fsum(
            float(row["non_readable_token_attention_mass"])
            for row in partitions
        ),
        "partition_is_exhaustive_and_disjoint": True,
        "no_top_k_sampling_or_truncation": True,
    }
    raw_partition = {
        **partition_body,
        "content_sha256": _sha256_json(partition_body),
    }
    stage_counts = tuple(
        {
            "stage": identity.stage,
            "objective": identity.objective,
            "fold": identity.fold,
            "raw_token_occurrence_count": identity.raw_count,
            **{
                key: value
                for key, value in partition.items()
                if key != "raw_token_occurrence_count"
            },
        }
        for identity, partition in zip(identities, partitions, strict=True)
    )
    tokenizer_hashes = {
        str(row["tokenizer_identity_content_sha256"])
        for row in source_batches
    }
    if len(tokenizer_hashes) != 1:
        raise RuntimeError("HTR fitted tokenizer identity changed")
    normalization_checks = {
        "schema_version": (
            "production_htr_stage2_source_normalization_checks_v2"
        ),
        "batch_count": len(normalization_rows),
        "batches": normalization_rows,
        "all_raw_token_weights_normalized_by_chunk": True,
        "all_chunk_weights_normalized_by_note": True,
        "all_hierarchical_scores_equal_component_product": True,
        "all_raw_coordinates_join_to_authenticated_chunks": True,
        "exact_disjoint_oof_note_coverage": True,
    }
    return _ComputedAggregation(
        fold_records=tuple(fold_records),
        cross_records=tuple(cross_records),
        reverse_raw_occurrence_index=reverse_raw_index,
        aggregate_offsets=aggregate_offsets,
        source_batches=tuple(source_batches),
        raw_partition=raw_partition,
        stage_objective_fold_counts=stage_counts,
        source_tokenizer_identity_sha256=next(iter(tokenizer_hashes)),
        normalization_checks=normalization_checks,
    )


def _model_facing_aggregate(
    aggregate: Mapping[str, Any],
    *,
    cross_aggregate_reference: Mapping[str, Any],
) -> dict[str, Any]:
    fold_support = [
        {
            "fold": int(row["fold"]),
            "fold_aggregate_id": str(row["fold_aggregate_id"]),
            "fold_aggregate_content_sha256": _require_sha256(
                row["fold_aggregate_content_sha256"],
                label="HTR fold aggregate",
            ),
            "occurrence_count": int(row["occurrence_count"]),
            "raw_token_occurrence_count": int(
                row["raw_token_occurrence_count"]
            ),
            "unique_note_count": int(row["unique_note_count"]),
            "unique_chunk_count": int(row["unique_chunk_count"]),
        }
        for row in aggregate["fold_support"]
    ]
    body = {
        "schema_version": HTR_STAGE2_MODEL_AGGREGATE_SCHEMA,
        "aggregate_id": str(aggregate["aggregate_id"]),
        "source_aggregate_content_sha256": _require_sha256(
            aggregate["content_sha256"],
            label="HTR complete semantic aggregate",
        ),
        "stage": str(aggregate["stage"]),
        "objective": str(aggregate["objective"]),
        "normalized_focus_text": str(aggregate["normalized_focus_text"]),
        "wordpiece_kind": str(aggregate["wordpiece_kind"]),
        "semantic_occurrence_definition": (
            "every_eligible_non_special_raw_token_occurrence_v2"
        ),
        "occurrence_count": int(aggregate["occurrence_count"]),
        "raw_token_occurrence_count": int(
            aggregate["raw_token_occurrence_count"]
        ),
        "unique_note_count": int(aggregate["unique_note_count"]),
        "unique_chunk_count": int(aggregate["unique_chunk_count"]),
        "attention_summaries": copy.deepcopy(
            aggregate["attention_summaries"]
        ),
        "fold_support": fold_support,
        "display_text_variant_count": int(
            aggregate["display_text_variant_count"]
        ),
        "display_text_variant_content_sha256": _require_sha256(
            aggregate["display_text_variant_content_sha256"],
            label="HTR display-text variants",
        ),
        "display_text_variants_authenticated_reference": {
            "cross_fold_aggregate_inventory_content_sha256": _require_sha256(
                cross_aggregate_reference["content_sha256"],
                label="HTR cross-fold aggregate inventory",
            ),
            "aggregate_id": str(aggregate["aggregate_id"]),
            "complete_exact_variants_retained": True,
        },
        "context_windows": [
            {
                "display_text": str(row["display_text"]),
                "exact_focus_text": str(row["exact_focus_text"]),
                "token_attention": float(row["token_attention"]),
                "chunk_attention": float(row["chunk_attention"]),
                "hierarchical_attention_score": float(
                    row["hierarchical_attention_score"]
                ),
            }
            for row in aggregate["context_windows"]
        ],
        "architecture_chunk_schema_version": (
            HTR_STAGE2_ARCHITECTURE_CHUNK_SCHEMA
        ),
        "hierarchical_attention_interpretation": (
            "ranking_heuristic_not_causal_attribution"
        ),
        "complete_semantic_accounting": True,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _pack_batches(
    aggregates: Sequence[Mapping[str, Any]],
    *,
    raw_reference: Mapping[str, Any],
    reverse_index_reference: Mapping[str, Any],
    cross_aggregate_reference: Mapping[str, Any],
    max_bytes: int,
    max_token_upper_bound: int,
    max_aggregates: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for aggregate in aggregates:
        model = _model_facing_aggregate(
            aggregate,
            cross_aggregate_reference=cross_aggregate_reference,
        )
        grouped[(str(model["stage"]), str(model["objective"]))].append(model)
    batches: list[dict[str, Any]] = []
    sizes: list[int] = []
    bound = min(int(max_bytes), int(max_token_upper_bound))
    for (stage, objective), rows in sorted(
        grouped.items(),
        key=lambda item: (_STAGE_ORDER[item[0][0]], item[0][1]),
    ):
        ordered = sorted(rows, key=lambda row: str(row["aggregate_id"]))
        members_by_batch: list[list[Mapping[str, Any]]] = []
        current: list[Mapping[str, Any]] = []
        for row in ordered:
            candidate = [*current, row]
            probe = {
                "schema_version": HTR_STAGE2_AGGREGATE_BATCH_SCHEMA,
                "stage": stage,
                "objective": objective,
                "batch_index": 999999999,
                "batch_count": 999999999,
                "aggregate_count": len(candidate),
                "aggregates": candidate,
                "raw_evidence_reference": raw_reference,
                "reverse_index_reference": reverse_index_reference,
                "hierarchical_attention_interpretation": (
                    "ranking_heuristic_not_causal_attribution"
                ),
                "complete_semantic_aggregate_delivery": True,
                "content_sha256": "0" * 64,
            }
            size = len(canonical_json(probe).encode("utf-8"))
            if len(candidate) > max_aggregates or size > bound:
                if not current:
                    raise ValueError(
                        "one complete HTR semantic aggregate exceeds its "
                        "model-facing byte/token bound"
                    )
                members_by_batch.append(current)
                current = [row]
            else:
                current = candidate
        if current:
            members_by_batch.append(current)
        for batch_index, members in enumerate(members_by_batch, start=1):
            body = {
                "schema_version": HTR_STAGE2_AGGREGATE_BATCH_SCHEMA,
                "stage": stage,
                "objective": objective,
                "batch_index": batch_index,
                "batch_count": len(members_by_batch),
                "aggregate_count": len(members),
                "aggregates": members,
                "raw_evidence_reference": raw_reference,
                "reverse_index_reference": reverse_index_reference,
                "hierarchical_attention_interpretation": (
                    "ranking_heuristic_not_causal_attribution"
                ),
                "complete_semantic_aggregate_delivery": True,
            }
            batch = {**body, "content_sha256": _sha256_json(body)}
            size = len(canonical_json(batch).encode("utf-8"))
            if size > max_bytes or size > max_token_upper_bound:
                raise RuntimeError("complete HTR semantic batch exceeds its bound")
            batches.append(batch)
            sizes.append(size)
    return batches, sizes


def build_htr_semantic_aggregation_scope(
    *,
    root: Path | str,
    source_payload: Mapping[str, Any],
    source_array_store_root: Path | str,
    source_fit_seal_content_sha256: str,
    source_payload_content_sha256: str,
    source_fit_seal_locator: str,
    logical_scope_id: str,
    physical_owner_scope_id: str,
    outer_fold: int,
    context_epoch: int,
    scope_binding_sha256: str,
    max_model_facing_batch_bytes: int = DEFAULT_MODEL_FACING_BATCH_BYTES,
    max_model_facing_token_upper_bound: int = (
        DEFAULT_MODEL_FACING_TOKEN_UPPER_BOUND
    ),
    max_model_facing_aggregates_per_batch: int = (
        DEFAULT_MODEL_FACING_AGGREGATES_PER_BATCH
    ),
    context_windows_per_aggregate: int = (
        DEFAULT_CONTEXT_WINDOWS_PER_AGGREGATE
    ),
    context_character_radius: int = DEFAULT_CONTEXT_CHARACTER_RADIUS,
) -> HtrSemanticAggregationResult:
    target = Path(root)
    if not target.is_absolute():
        raise ValueError("HTR aggregate scope root must be absolute")
    if target.exists() or target.is_symlink():
        raise FileExistsError("HTR aggregate scope root must be fresh")
    if target.parent.resolve(strict=True) != target.parent:
        raise ValueError("HTR aggregate scope parent must be canonical")
    source_fit_sha = _require_sha256(
        source_fit_seal_content_sha256,
        label="HTR source fit seal",
    )
    source_payload_sha = _require_sha256(
        source_payload_content_sha256,
        label="HTR source evidence payload",
    )
    scope_binding = _require_sha256(
        scope_binding_sha256,
        label="HTR logical-scope binding",
    )
    if _sha256_json(source_payload) != source_payload_sha:
        raise ValueError("HTR source payload digest differs from its fit seal")
    if (
        not isinstance(source_fit_seal_locator, str)
        or not source_fit_seal_locator
        or Path(source_fit_seal_locator).is_absolute()
        or not logical_scope_id
        or not physical_owner_scope_id
    ):
        raise ValueError("HTR semantic scope identity is invalid")
    _positive_int(outer_fold, label="HTR aggregate outer fold")
    _nonnegative_int(context_epoch, label="HTR aggregate context epoch")
    max_bytes = _positive_int(
        max_model_facing_batch_bytes,
        label="HTR aggregate byte bound",
    )
    max_tokens = _positive_int(
        max_model_facing_token_upper_bound,
        label="HTR aggregate token upper bound",
    )
    max_aggregates = _positive_int(
        max_model_facing_aggregates_per_batch,
        label="HTR aggregates per batch",
    )
    context_limit = _positive_int(
        context_windows_per_aggregate,
        label="HTR context-window count",
    )
    context_radius = _positive_int(
        context_character_radius,
        label="HTR context character radius",
    )
    source_array_root = Path(source_array_store_root)
    computed = _compute_complete_aggregation(
        source_payload=source_payload,
        source_array_store_root=source_array_root,
        context_limit=context_limit,
        context_radius=context_radius,
    )
    package = source_payload["token_attention_evidence"]
    source_array_locator = (
        Path(source_fit_seal_locator).parent / "fit_state" / "arrays"
    ).as_posix()
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    try:
        reverse_root = staging / "reverse_index"
        reverse_root.mkdir()
        reverse_arrays = {
            "raw_occurrence_index_by_cross_aggregate": _write_npy(
                reverse_root / "raw_occurrence_index_by_cross_aggregate.npy",
                computed.reverse_raw_occurrence_index,
                root=staging,
            ),
            "cross_aggregate_offsets": _write_npy(
                reverse_root / "cross_aggregate_offsets.npy",
                computed.aggregate_offsets,
                root=staging,
            ),
        }
        reverse_body = {
            "schema_version": HTR_STAGE2_REVERSE_INDEX_SCHEMA,
            "source_payload_content_sha256": source_payload_sha,
            "source_fit_seal_content_sha256": source_fit_sha,
            "source_array_store_locator": source_array_locator,
            "scope_binding_sha256": scope_binding,
            "raw_occurrence_partition": computed.raw_partition,
            "eligible_readable_token_occurrence_count": int(
                computed.reverse_raw_occurrence_index.size
            ),
            "aggregated_readable_token_occurrence_count": int(
                computed.reverse_raw_occurrence_index.size
            ),
            "cross_fold_aggregate_count": len(computed.cross_records),
            "fold_local_aggregate_count": len(computed.fold_records),
            "source_chunk_record_count": int(
                package["chunk_interpretation_count"]
            ),
            "source_batches": list(computed.source_batches),
            "source_tokenizer_identity_content_sha256": (
                computed.source_tokenizer_identity_sha256
            ),
            "arrays": reverse_arrays,
            "occurrence_order": (
                "cross_aggregate_then_fold_then_global_raw_occurrence_v2"
            ),
            "every_eligible_readable_occurrence_accounted_exactly_once": True,
            "every_reverse_coordinate_resolves_to_authenticated_raw_source": True,
            "raw_token_arrays_copied": False,
        }
        reverse_manifest = {
            **reverse_body,
            "content_sha256": _sha256_json(reverse_body),
        }
        reverse_registration = _write_json(
            staging / "reverse_index_manifest.json",
            reverse_manifest,
            root=staging,
        )
        fold_body = {
            "schema_version": HTR_STAGE2_FOLD_AGGREGATE_SCHEMA,
            "scope_binding_sha256": scope_binding,
            "fold_local_aggregate_count": len(computed.fold_records),
            "fold_local_aggregates": list(computed.fold_records),
            "folds_kept_separate": True,
            "all_eligible_occurrences_included": True,
        }
        fold_payload = {
            **fold_body,
            "content_sha256": _sha256_json(fold_body),
        }
        fold_registration = _write_json(
            staging / "fold_local_aggregates.json",
            fold_payload,
            root=staging,
        )
        cross_body = {
            "schema_version": HTR_STAGE2_CROSS_FOLD_AGGREGATE_SCHEMA,
            "scope_binding_sha256": scope_binding,
            "cross_fold_aggregate_count": len(computed.cross_records),
            "cross_fold_aggregates": list(computed.cross_records),
            "fold_local_source_content_sha256": (
                fold_payload["content_sha256"]
            ),
            "cross_fold_consolidation": (
                HTR_STAGE2_CROSS_FOLD_CONSOLIDATION_SCHEMA
            ),
            "complete_semantic_aggregate_inventory": True,
            "no_semantic_aggregate_selected_away": True,
        }
        cross_payload = {
            **cross_body,
            "content_sha256": _sha256_json(cross_body),
        }
        cross_registration = _write_json(
            staging / "cross_fold_aggregates.json",
            cross_payload,
            root=staging,
        )
        raw_reference = {
            "source_fit_seal_locator": source_fit_seal_locator,
            "source_fit_seal_content_sha256": source_fit_sha,
            "source_payload_content_sha256": source_payload_sha,
            "source_array_store_locator": source_array_locator,
            "token_attention_package_content_sha256": _require_sha256(
                package["content_sha256"],
                label="HTR token-attention package",
            ),
            "tokenizer_identity_content_sha256": (
                computed.source_tokenizer_identity_sha256
            ),
            "token_occurrence_count": int(
                package["token_occurrence_count"]
            ),
            "chunk_interpretation_count": int(
                package["chunk_interpretation_count"]
            ),
            "special_token_occurrence_count": int(
                package["special_token_occurrence_count"]
            ),
            "special_token_attention_mass": float(
                package["special_token_attention_mass"]
            ),
            "padding_occurrence_count": int(
                package["padding_occurrence_count"]
            ),
            "raw_arrays_copied_to_handoff": False,
        }
        reverse_reference = {
            "reverse_index_manifest_relative_path": (
                reverse_registration["relative_path"]
            ),
            "reverse_index_manifest_content_sha256": (
                reverse_manifest["content_sha256"]
            ),
            "eligible_readable_token_occurrence_count": int(
                computed.reverse_raw_occurrence_index.size
            ),
            "complete_reverse_index": True,
        }
        cross_reference = {
            "relative_path": cross_registration["relative_path"],
            "content_sha256": cross_payload["content_sha256"],
            "cross_fold_aggregate_count": len(computed.cross_records),
            "complete_semantic_aggregate_inventory": True,
        }
        batches, _sizes = _pack_batches(
            computed.cross_records,
            raw_reference=raw_reference,
            reverse_index_reference=reverse_reference,
            cross_aggregate_reference=cross_reference,
            max_bytes=max_bytes,
            max_token_upper_bound=max_tokens,
            max_aggregates=max_aggregates,
        )
        architecture_evidence = sorted(
            [
                {
                    "atom_kind": "htr_semantic_aggregate_batch",
                    "source_kind": LEGACY_ALL_SOURCE,
                    "observable_axes": (
                        [TREATMENT_AXIS, OUTCOME_AXIS]
                        if batch["stage"] == "nuisance"
                        else [HETEROGENEITY_AXIS]
                    ),
                    "content": {
                        "architecture_encoder": (
                            "htr_token_attention_semantic_aggregation"
                        ),
                        "group": {
                            "stage": (
                                "nuisance"
                                if batch["stage"] == "nuisance"
                                else "effect"
                            ),
                            "meaning": batch["objective"],
                        },
                        "aggregate_batch": batch,
                    },
                }
                for batch in batches
            ],
            key=canonical_json,
        )
        batch_sizes = [
            len(
                canonical_json(
                    item["content"]["aggregate_batch"]
                ).encode("utf-8")
            )
            for item in architecture_evidence
        ]
        semantic_summary = {
            "schema_version": HTR_STAGE2_SCOPE_MANIFEST_SCHEMA,
            "normalization_rule": HTR_STAGE2_NORMALIZATION_SCHEMA,
            "raw_occurrence_partition": computed.raw_partition,
            "cross_fold_consolidation": (
                HTR_STAGE2_CROSS_FOLD_CONSOLIDATION_SCHEMA
            ),
            "architecture_chunk_schema_version": (
                HTR_STAGE2_ARCHITECTURE_CHUNK_SCHEMA
            ),
            "batching": HTR_STAGE2_BATCHING_SCHEMA,
            "raw_evidence_reference": raw_reference,
            "reverse_index_reference": reverse_reference,
            "cross_fold_aggregate_reference": cross_reference,
            "eligible_readable_token_occurrence_count": int(
                computed.reverse_raw_occurrence_index.size
            ),
            "aggregated_readable_token_occurrence_count": int(
                computed.reverse_raw_occurrence_index.size
            ),
            "non_readable_accounting_bucket": {
                "occurrence_count": int(
                    computed.raw_partition[
                        "non_readable_token_occurrence_count"
                    ]
                ),
                "attention_mass": float(
                    computed.raw_partition[
                        "non_readable_token_attention_mass"
                    ]
                ),
                "explicitly_accounted": True,
            },
            "special_token_accounting_bucket": {
                "occurrence_count": int(
                    package["special_token_occurrence_count"]
                ),
                "attention_mass": float(
                    package["special_token_attention_mass"]
                ),
                "excluded_from_readable_phrases": True,
                "retained_in_raw_authenticated_package": True,
            },
            "padding_accounting_bucket": {
                "occurrence_count": int(package["padding_occurrence_count"]),
                "explicitly_accounted": True,
            },
            "source_chunk_interpretation_count": int(
                package["chunk_interpretation_count"]
            ),
            "source_stage_objective_fold_counts": list(
                computed.stage_objective_fold_counts
            ),
            "source_normalization_checks": computed.normalization_checks,
            "fold_local_aggregate_count": len(computed.fold_records),
            "cross_fold_aggregate_count": len(computed.cross_records),
            "model_facing_batch_count": len(batches),
            "model_facing_bytes": sum(batch_sizes),
            "model_facing_batch_sizes_bytes": batch_sizes,
            "maximum_model_facing_batch_bytes": max(batch_sizes),
            "median_model_facing_batch_bytes": statistics.median(batch_sizes),
            "maximum_model_facing_token_upper_bound": max(batch_sizes),
            "configured_batch_byte_bound": max_bytes,
            "configured_batch_token_upper_bound": max_tokens,
            "configured_aggregates_per_batch_bound": max_aggregates,
            "maximum_aggregates_in_one_batch": max(
                int(batch["aggregate_count"]) for batch in batches
            ),
            "context_windows_per_aggregate": context_limit,
            "context_character_radius": context_radius,
            "one_atom_per_source_chunk_design_call_count": math.ceil(
                int(package["chunk_interpretation_count"]) / 2
            ),
            "planned_htr_interpretation_call_count": len(batches),
            "no_top_k_sampling_or_truncation": True,
            "every_semantic_aggregate_delivered_exactly_once": True,
            "every_eligible_raw_token_occurrence_accounted_exactly_once": True,
            "folds_separate_before_explicit_cross_fold_consolidation": True,
            "unique_note_prevalence_separate_from_overlap_occurrences": True,
            "raw_token_arrays_copied": False,
        }
        payload_body = {
            "schema_version": HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA,
            "family": HTR_NEURAL,
            "architecture_evidence": architecture_evidence,
            "semantic_aggregation": semantic_summary,
        }
        payload = {
            **payload_body,
            "content_sha256": _sha256_json(payload_body),
        }
        payload_registration = _write_json(
            staging / "model_facing_aggregate_payload.json",
            payload,
            root=staging,
        )
        scope_body = {
            "schema_version": HTR_STAGE2_SCOPE_MANIFEST_SCHEMA,
            "logical_scope_id": logical_scope_id,
            "physical_owner_scope_id": physical_owner_scope_id,
            "outer_fold": int(outer_fold),
            "context_epoch": int(context_epoch),
            "scope_binding_sha256": scope_binding,
            "source_fit_seal_locator": source_fit_seal_locator,
            "source_fit_seal_content_sha256": source_fit_sha,
            "source_payload_content_sha256": source_payload_sha,
            "source_array_store_locator": source_array_locator,
            "source_token_attention_package_content_sha256": (
                raw_reference["token_attention_package_content_sha256"]
            ),
            "model_facing_payload": payload_registration,
            "reverse_index_manifest": reverse_registration,
            "fold_local_aggregates": fold_registration,
            "cross_fold_aggregates": cross_registration,
            "array_file_count": len(reverse_arrays),
            "raw_token_arrays_copied": False,
            "all_source_chunks_accounted": True,
            "all_eligible_raw_token_occurrences_accounted_exactly_once": True,
            "no_top_k_sampling_or_truncation": True,
            "summary": semantic_summary,
        }
        scope_manifest = {
            **scope_body,
            "content_sha256": _sha256_json(scope_body),
        }
        _write_json(
            staging / "scope_manifest.json",
            scope_manifest,
            root=staging,
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return HtrSemanticAggregationResult(
        payload=payload,
        scope_manifest=scope_manifest,
        scope_manifest_path=target / "scope_manifest.json",
    )


def _read_closed_scope_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("HTR complete semantic scope manifest is missing")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("HTR complete semantic scope manifest is invalid") from exc
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    if (
        not isinstance(value, dict)
        or canonical_json(value).encode("utf-8") != payload
        or value.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("HTR complete semantic scope manifest is not canonical")
    return value


def validate_htr_semantic_aggregation_scope(
    *,
    root: Path | str,
    source_payload: Mapping[str, Any],
    source_array_store_root: Path | str,
    expected_source_fit_seal_content_sha256: str,
    expected_source_payload_content_sha256: str,
    expected_scope_binding_sha256: str,
) -> HtrSemanticAggregationResult:
    target = Path(root)
    if (
        not target.is_absolute()
        or target.is_symlink()
        or target.resolve(strict=True) != target
        or not target.is_dir()
    ):
        raise ValueError("HTR complete semantic scope root is not canonical")
    manifest_path = target / "scope_manifest.json"
    manifest = _read_closed_scope_manifest(manifest_path)
    expected_keys = {
        "schema_version",
        "logical_scope_id",
        "physical_owner_scope_id",
        "outer_fold",
        "context_epoch",
        "scope_binding_sha256",
        "source_fit_seal_locator",
        "source_fit_seal_content_sha256",
        "source_payload_content_sha256",
        "source_array_store_locator",
        "source_token_attention_package_content_sha256",
        "model_facing_payload",
        "reverse_index_manifest",
        "fold_local_aggregates",
        "cross_fold_aggregates",
        "array_file_count",
        "raw_token_arrays_copied",
        "all_source_chunks_accounted",
        "all_eligible_raw_token_occurrences_accounted_exactly_once",
        "no_top_k_sampling_or_truncation",
        "summary",
        "content_sha256",
    }
    source_fit_sha = _require_sha256(
        expected_source_fit_seal_content_sha256,
        label="expected HTR fit seal",
    )
    source_payload_sha = _require_sha256(
        expected_source_payload_content_sha256,
        label="expected HTR source payload",
    )
    scope_binding = _require_sha256(
        expected_scope_binding_sha256,
        label="expected HTR scope binding",
    )
    expected_array_locator = (
        Path(str(manifest.get("source_fit_seal_locator", ""))).parent
        / "fit_state"
        / "arrays"
    ).as_posix()
    summary = manifest.get("summary")
    if (
        set(manifest) != expected_keys
        or manifest.get("schema_version") != HTR_STAGE2_SCOPE_MANIFEST_SCHEMA
        or manifest.get("source_fit_seal_content_sha256") != source_fit_sha
        or manifest.get("source_payload_content_sha256")
        != source_payload_sha
        or manifest.get("scope_binding_sha256") != scope_binding
        or manifest.get("source_array_store_locator")
        != expected_array_locator
        or manifest.get("array_file_count") != 2
        or manifest.get("raw_token_arrays_copied") is not False
        or manifest.get("all_source_chunks_accounted") is not True
        or manifest.get(
            "all_eligible_raw_token_occurrences_accounted_exactly_once"
        )
        is not True
        or manifest.get("no_top_k_sampling_or_truncation") is not True
        or not isinstance(summary, Mapping)
        or summary.get("schema_version") != HTR_STAGE2_SCOPE_MANIFEST_SCHEMA
        or _sha256_json(source_payload) != source_payload_sha
    ):
        raise ValueError("HTR complete semantic scope identity changed")

    context_limit = _positive_int(
        summary.get("context_windows_per_aggregate"),
        label="HTR context-window count",
    )
    context_radius = _positive_int(
        summary.get("context_character_radius"),
        label="HTR context radius",
    )
    computed = _compute_complete_aggregation(
        source_payload=source_payload,
        source_array_store_root=Path(source_array_store_root),
        context_limit=context_limit,
        context_radius=context_radius,
    )
    reverse = _read_json(
        target / manifest["reverse_index_manifest"]["relative_path"],
        manifest["reverse_index_manifest"],
    )
    reverse_body = {
        key: copy.deepcopy(child)
        for key, child in reverse.items()
        if key != "content_sha256"
    }
    if (
        reverse.get("schema_version") != HTR_STAGE2_REVERSE_INDEX_SCHEMA
        or reverse.get("content_sha256") != _sha256_json(reverse_body)
        or reverse.get("source_payload_content_sha256")
        != source_payload_sha
        or reverse.get("source_fit_seal_content_sha256") != source_fit_sha
        or reverse.get("scope_binding_sha256") != scope_binding
        or reverse.get("source_array_store_locator")
        != expected_array_locator
        or reverse.get("raw_occurrence_partition")
        != computed.raw_partition
        or reverse.get("source_batches") != list(computed.source_batches)
        or reverse.get("source_tokenizer_identity_content_sha256")
        != computed.source_tokenizer_identity_sha256
        or reverse.get("raw_token_arrays_copied") is not False
        or reverse.get(
            "every_eligible_readable_occurrence_accounted_exactly_once"
        )
        is not True
        or reverse.get(
            "every_reverse_coordinate_resolves_to_authenticated_raw_source"
        )
        is not True
        or not isinstance(reverse.get("arrays"), Mapping)
        or set(reverse["arrays"])
        != {
            "raw_occurrence_index_by_cross_aggregate",
            "cross_aggregate_offsets",
        }
    ):
        raise ValueError("HTR complete semantic reverse manifest changed")
    observed_raw = _load_npy(
        target
        / reverse["arrays"][
            "raw_occurrence_index_by_cross_aggregate"
        ]["relative_path"],
        reverse["arrays"]["raw_occurrence_index_by_cross_aggregate"],
    )
    observed_offsets = _load_npy(
        target
        / reverse["arrays"]["cross_aggregate_offsets"]["relative_path"],
        reverse["arrays"]["cross_aggregate_offsets"],
    )
    if (
        not np.array_equal(
            observed_raw,
            computed.reverse_raw_occurrence_index,
        )
        or not np.array_equal(
            observed_offsets,
            computed.aggregate_offsets,
        )
        or observed_raw.size
        != int(
            reverse["eligible_readable_token_occurrence_count"]
        )
        or observed_raw.size
        != int(
            reverse["aggregated_readable_token_occurrence_count"]
        )
        or (
            observed_raw.size
            and (
                int(observed_raw.min()) < 0
                or int(observed_raw.max())
                >= int(computed.raw_partition["raw_token_occurrence_count"])
            )
        )
    ):
        raise ValueError("HTR complete semantic reverse index is incomplete")

    fold_payload = _read_json(
        target / manifest["fold_local_aggregates"]["relative_path"],
        manifest["fold_local_aggregates"],
    )
    fold_body = {
        key: copy.deepcopy(child)
        for key, child in fold_payload.items()
        if key != "content_sha256"
    }
    if (
        fold_payload.get("schema_version")
        != HTR_STAGE2_FOLD_AGGREGATE_SCHEMA
        or fold_payload.get("content_sha256") != _sha256_json(fold_body)
        or fold_payload.get("scope_binding_sha256") != scope_binding
        or fold_payload.get("fold_local_aggregate_count")
        != len(computed.fold_records)
        or fold_payload.get("fold_local_aggregates")
        != list(computed.fold_records)
        or fold_payload.get("folds_kept_separate") is not True
        or fold_payload.get("all_eligible_occurrences_included") is not True
    ):
        raise ValueError("HTR fold-local semantic aggregates changed")
    cross_payload = _read_json(
        target / manifest["cross_fold_aggregates"]["relative_path"],
        manifest["cross_fold_aggregates"],
    )
    cross_body = {
        key: copy.deepcopy(child)
        for key, child in cross_payload.items()
        if key != "content_sha256"
    }
    if (
        cross_payload.get("schema_version")
        != HTR_STAGE2_CROSS_FOLD_AGGREGATE_SCHEMA
        or cross_payload.get("content_sha256") != _sha256_json(cross_body)
        or cross_payload.get("scope_binding_sha256") != scope_binding
        or cross_payload.get("cross_fold_aggregate_count")
        != len(computed.cross_records)
        or cross_payload.get("cross_fold_aggregates")
        != list(computed.cross_records)
        or cross_payload.get("fold_local_source_content_sha256")
        != fold_payload["content_sha256"]
        or cross_payload.get("complete_semantic_aggregate_inventory")
        is not True
        or cross_payload.get("no_semantic_aggregate_selected_away")
        is not True
    ):
        raise ValueError("HTR cross-fold semantic aggregates changed")

    payload = _read_json(
        target / manifest["model_facing_payload"]["relative_path"],
        manifest["model_facing_payload"],
    )
    payload_body = {
        key: copy.deepcopy(child)
        for key, child in payload.items()
        if key != "content_sha256"
    }
    if (
        payload.get("schema_version") != HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA
        or payload.get("family") != HTR_NEURAL
        or payload.get("content_sha256") != _sha256_json(payload_body)
        or payload.get("semantic_aggregation") != summary
        or not isinstance(payload.get("architecture_evidence"), list)
        or not payload["architecture_evidence"]
    ):
        raise ValueError("HTR model-facing semantic payload changed")
    expected_batches, _sizes = _pack_batches(
        computed.cross_records,
        raw_reference=summary["raw_evidence_reference"],
        reverse_index_reference=summary["reverse_index_reference"],
        cross_aggregate_reference=summary[
            "cross_fold_aggregate_reference"
        ],
        max_bytes=int(summary["configured_batch_byte_bound"]),
        max_token_upper_bound=int(
            summary["configured_batch_token_upper_bound"]
        ),
        max_aggregates=int(
            summary["configured_aggregates_per_batch_bound"]
        ),
    )
    expected_architecture = sorted(
        [
            {
                "atom_kind": "htr_semantic_aggregate_batch",
                "source_kind": LEGACY_ALL_SOURCE,
                "observable_axes": (
                    [TREATMENT_AXIS, OUTCOME_AXIS]
                    if batch["stage"] == "nuisance"
                    else [HETEROGENEITY_AXIS]
                ),
                "content": {
                    "architecture_encoder": (
                        "htr_token_attention_semantic_aggregation"
                    ),
                    "group": {
                        "stage": (
                            "nuisance"
                            if batch["stage"] == "nuisance"
                            else "effect"
                        ),
                        "meaning": batch["objective"],
                    },
                    "aggregate_batch": batch,
                },
            }
            for batch in expected_batches
        ],
        key=canonical_json,
    )
    sizes = [
        len(
            canonical_json(item["content"]["aggregate_batch"]).encode(
                "utf-8"
            )
        )
        for item in expected_architecture
    ]
    if payload["architecture_evidence"] != expected_architecture:
        raise ValueError("HTR semantic aggregate delivery is altered")
    expected_summary_values = {
        "raw_occurrence_partition": computed.raw_partition,
        "normalization_rule": HTR_STAGE2_NORMALIZATION_SCHEMA,
        "cross_fold_consolidation": (
            HTR_STAGE2_CROSS_FOLD_CONSOLIDATION_SCHEMA
        ),
        "architecture_chunk_schema_version": (
            HTR_STAGE2_ARCHITECTURE_CHUNK_SCHEMA
        ),
        "batching": HTR_STAGE2_BATCHING_SCHEMA,
        "raw_evidence_reference": {
            "source_fit_seal_locator": manifest["source_fit_seal_locator"],
            "source_fit_seal_content_sha256": source_fit_sha,
            "source_payload_content_sha256": source_payload_sha,
            "source_array_store_locator": expected_array_locator,
            "token_attention_package_content_sha256": (
                source_payload["token_attention_evidence"][
                    "content_sha256"
                ]
            ),
            "tokenizer_identity_content_sha256": (
                computed.source_tokenizer_identity_sha256
            ),
            "token_occurrence_count": int(
                source_payload["token_attention_evidence"][
                    "token_occurrence_count"
                ]
            ),
            "chunk_interpretation_count": int(
                source_payload["token_attention_evidence"][
                    "chunk_interpretation_count"
                ]
            ),
            "special_token_occurrence_count": int(
                source_payload["token_attention_evidence"][
                    "special_token_occurrence_count"
                ]
            ),
            "special_token_attention_mass": float(
                source_payload["token_attention_evidence"][
                    "special_token_attention_mass"
                ]
            ),
            "padding_occurrence_count": int(
                source_payload["token_attention_evidence"][
                    "padding_occurrence_count"
                ]
            ),
            "raw_arrays_copied_to_handoff": False,
        },
        "reverse_index_reference": {
            "reverse_index_manifest_relative_path": manifest[
                "reverse_index_manifest"
            ]["relative_path"],
            "reverse_index_manifest_content_sha256": reverse[
                "content_sha256"
            ],
            "eligible_readable_token_occurrence_count": int(
                computed.reverse_raw_occurrence_index.size
            ),
            "complete_reverse_index": True,
        },
        "cross_fold_aggregate_reference": {
            "relative_path": manifest["cross_fold_aggregates"][
                "relative_path"
            ],
            "content_sha256": cross_payload["content_sha256"],
            "cross_fold_aggregate_count": len(computed.cross_records),
            "complete_semantic_aggregate_inventory": True,
        },
        "eligible_readable_token_occurrence_count": int(
            computed.reverse_raw_occurrence_index.size
        ),
        "aggregated_readable_token_occurrence_count": int(
            computed.reverse_raw_occurrence_index.size
        ),
        "source_chunk_interpretation_count": int(
            source_payload["token_attention_evidence"][
                "chunk_interpretation_count"
            ]
        ),
        "source_stage_objective_fold_counts": list(
            computed.stage_objective_fold_counts
        ),
        "source_normalization_checks": computed.normalization_checks,
        "non_readable_accounting_bucket": {
            "occurrence_count": int(
                computed.raw_partition[
                    "non_readable_token_occurrence_count"
                ]
            ),
            "attention_mass": float(
                computed.raw_partition[
                    "non_readable_token_attention_mass"
                ]
            ),
            "explicitly_accounted": True,
        },
        "special_token_accounting_bucket": {
            "occurrence_count": int(
                source_payload["token_attention_evidence"][
                    "special_token_occurrence_count"
                ]
            ),
            "attention_mass": float(
                source_payload["token_attention_evidence"][
                    "special_token_attention_mass"
                ]
            ),
            "excluded_from_readable_phrases": True,
            "retained_in_raw_authenticated_package": True,
        },
        "padding_accounting_bucket": {
            "occurrence_count": int(
                source_payload["token_attention_evidence"][
                    "padding_occurrence_count"
                ]
            ),
            "explicitly_accounted": True,
        },
        "fold_local_aggregate_count": len(computed.fold_records),
        "cross_fold_aggregate_count": len(computed.cross_records),
        "model_facing_batch_count": len(expected_batches),
        "model_facing_bytes": sum(sizes),
        "model_facing_batch_sizes_bytes": sizes,
        "maximum_model_facing_batch_bytes": max(sizes),
        "median_model_facing_batch_bytes": statistics.median(sizes),
        "planned_htr_interpretation_call_count": len(expected_batches),
        "no_top_k_sampling_or_truncation": True,
        "every_semantic_aggregate_delivered_exactly_once": True,
        "every_eligible_raw_token_occurrence_accounted_exactly_once": True,
        "raw_token_arrays_copied": False,
    }
    if any(summary.get(key) != value for key, value in expected_summary_values.items()):
        raise ValueError("HTR complete semantic scope summary changed")
    return HtrSemanticAggregationResult(
        payload=payload,
        scope_manifest=manifest,
        scope_manifest_path=manifest_path,
    )


def summarize_htr_call_plan(
    scope_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in scope_manifests]
    if not rows:
        raise ValueError("HTR call-plan summary requires at least one scope")
    summaries = [row.get("summary") for row in rows]
    if not all(isinstance(summary, Mapping) for summary in summaries):
        raise ValueError("HTR call-plan scope summaries are missing")
    sizes: list[int] = []
    for summary in summaries:
        scope_sizes = summary.get("model_facing_batch_sizes_bytes")
        if (
            not isinstance(scope_sizes, list)
            or len(scope_sizes)
            != int(summary["model_facing_batch_count"])
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in scope_sizes
            )
            or sum(scope_sizes) != int(summary["model_facing_bytes"])
            or max(scope_sizes)
            != int(summary["maximum_model_facing_batch_bytes"])
            or not math.isclose(
                float(statistics.median(scope_sizes)),
                float(summary["median_model_facing_batch_bytes"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or int(summary["maximum_aggregates_in_one_batch"])
            > int(summary["configured_aggregates_per_batch_bound"])
            or summary.get("no_top_k_sampling_or_truncation") is not True
            or summary.get(
                "every_eligible_raw_token_occurrence_accounted_exactly_once"
            )
            is not True
        ):
            raise ValueError("HTR complete semantic scope call plan is invalid")
        sizes.extend(scope_sizes)
    baseline = sum(
        int(summary["one_atom_per_source_chunk_design_call_count"])
        for summary in summaries
    )
    planned = sum(
        int(summary["planned_htr_interpretation_call_count"])
        for summary in summaries
    )
    if baseline < 1 or planned < 1:
        raise ValueError("HTR complete semantic call accounting is empty")
    return {
        "schema_version": "production_htr_stage2_call_plan_preflight_v2",
        "scope_count": len(rows),
        "raw_token_occurrence_count": sum(
            int(summary["raw_evidence_reference"]["token_occurrence_count"])
            for summary in summaries
        ),
        "raw_chunk_interpretation_count": sum(
            int(summary["source_chunk_interpretation_count"])
            for summary in summaries
        ),
        "raw_special_token_occurrence_count": sum(
            int(
                summary["special_token_accounting_bucket"][
                    "occurrence_count"
                ]
            )
            for summary in summaries
        ),
        "raw_special_token_attention_mass": math.fsum(
            float(
                summary["special_token_accounting_bucket"]["attention_mass"]
            )
            for summary in summaries
        ),
        "readable_token_occurrence_count": sum(
            int(summary["eligible_readable_token_occurrence_count"])
            for summary in summaries
        ),
        "non_readable_token_occurrence_count": sum(
            int(summary["non_readable_accounting_bucket"]["occurrence_count"])
            for summary in summaries
        ),
        "fold_local_aggregate_count": sum(
            int(summary["fold_local_aggregate_count"])
            for summary in summaries
        ),
        "cross_fold_aggregate_count": sum(
            int(summary["cross_fold_aggregate_count"])
            for summary in summaries
        ),
        "scope_source_stage_objective_fold_counts": [
            {
                "logical_scope_id": str(row["logical_scope_id"]),
                "physical_owner_scope_id": str(
                    row["physical_owner_scope_id"]
                ),
                "counts": summary["source_stage_objective_fold_counts"],
            }
            for row, summary in zip(rows, summaries, strict=True)
        ],
        "total_model_facing_bytes": sum(
            int(summary["model_facing_bytes"]) for summary in summaries
        ),
        "planned_htr_interpretation_call_count": planned,
        "one_atom_per_chunk_baseline_call_count": baseline,
        "call_reduction_fraction": 1.0 - (planned / baseline),
        "maximum_prompt_evidence_bytes": max(sizes),
        "median_prompt_evidence_bytes": statistics.median(sizes),
        "call_plan_on_order_of_hundreds_of_thousands": planned >= 100_000,
        "stage2_endpoint_launch_allowed": planned < 100_000,
        "raw_arrays_copied_to_model_facing_catalog": False,
        "no_top_k_sampling_or_truncation": True,
        "complete_raw_occurrence_partition_authenticated": True,
    }


__all__ = [
    "DEFAULT_CONTEXT_WINDOWS_PER_AGGREGATE",
    "DEFAULT_MODEL_FACING_AGGREGATES_PER_BATCH",
    "DEFAULT_MODEL_FACING_BATCH_BYTES",
    "DEFAULT_MODEL_FACING_TOKEN_UPPER_BOUND",
    "HTR_STAGE2_AGGREGATE_BATCH_SCHEMA",
    "HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA",
    "HTR_STAGE2_ARCHITECTURE_CHUNK_SCHEMA",
    "HTR_STAGE2_CROSS_FOLD_AGGREGATE_SCHEMA",
    "HTR_STAGE2_FOLD_AGGREGATE_SCHEMA",
    "HTR_STAGE2_MODEL_AGGREGATE_SCHEMA",
    "HTR_STAGE2_REVERSE_INDEX_SCHEMA",
    "HTR_STAGE2_SCOPE_MANIFEST_SCHEMA",
    "HTR_STAGE2_STORE_MANIFEST_SCHEMA",
    "HtrSemanticAggregationResult",
    "build_htr_semantic_aggregation_scope",
    "normalize_htr_complete_readable_token",
    "summarize_htr_call_plan",
    "validate_htr_semantic_aggregation_scope",
]
