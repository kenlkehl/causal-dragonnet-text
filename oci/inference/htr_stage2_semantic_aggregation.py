"""Authenticated semantic aggregation for complete HTR token-attention evidence.

The fitted HTR component remains the scientific source of truth.  This module
derives a compact, model-facing catalog from its deterministic readable-span
projection while retaining a complete reverse index to every contributing
chunk-local occurrence.  Raw token arrays are referenced by content identity;
they are never copied into this derived store or an LLM prompt.
"""

from __future__ import annotations

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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .all_evidence_discovery_interfaces import (
    HETEROGENEITY_AXIS,
    HTR_NEURAL,
    OUTCOME_AXIS,
    TREATMENT_AXIS,
    canonical_json,
)
from .all_evidence_fusion import LEGACY_ALL_SOURCE
from .htr_attention_evidence_schema import (
    ROLE_NEUTRAL_HTR_CHUNK_EVIDENCE_SCHEMA,
    ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA,
    ROLE_NEUTRAL_HTR_READABLE_SPAN_SCHEMA,
    ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_BATCH_SCHEMA,
    ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_PACKAGE_SCHEMA,
)

HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA = (
    "production_htr_stage2_semantic_aggregate_payload_v1"
)
HTR_STAGE2_AGGREGATE_BATCH_SCHEMA = (
    "production_htr_stage2_semantic_aggregate_batch_v1"
)
HTR_STAGE2_CROSS_FOLD_AGGREGATE_SCHEMA = (
    "production_htr_stage2_cross_fold_semantic_aggregate_v1"
)
HTR_STAGE2_MODEL_AGGREGATE_SCHEMA = (
    "production_htr_stage2_model_facing_semantic_aggregate_v1"
)
HTR_STAGE2_FOLD_AGGREGATE_SCHEMA = (
    "production_htr_stage2_fold_local_semantic_aggregate_v1"
)
HTR_STAGE2_REVERSE_INDEX_SCHEMA = (
    "production_htr_stage2_semantic_reverse_index_v1"
)
HTR_STAGE2_SCOPE_MANIFEST_SCHEMA = (
    "production_htr_stage2_semantic_scope_manifest_v1"
)
HTR_STAGE2_STORE_MANIFEST_SCHEMA = (
    "production_htr_stage2_semantic_store_manifest_v1"
)
HTR_STAGE2_NORMALIZATION_SCHEMA = (
    "htr_readable_focus_nfkc_casefold_whitespace_v1"
)
HTR_STAGE2_CROSS_FOLD_CONSOLIDATION_SCHEMA = (
    "same_stage_objective_normalized_focus_across_folds_v1"
)
HTR_STAGE2_CONTEXT_POLICY_SCHEMA = (
    "highest_hierarchical_then_token_then_coordinate_contexts_v1"
)
HTR_STAGE2_BATCHING_SCHEMA = (
    "canonical_stage_objective_member_byte_and_token_bounded_batches_v1"
)

DEFAULT_MODEL_FACING_BATCH_BYTES = 28_000
DEFAULT_MODEL_FACING_TOKEN_UPPER_BOUND = 28_000
DEFAULT_MODEL_FACING_AGGREGATES_PER_BATCH = 3
DEFAULT_CONTEXT_WINDOWS_PER_AGGREGATE = 3

_SHA256_HEX = frozenset("0123456789abcdef")
_STAGE_ORDER = {"nuisance": 0, "effect_modifier": 1}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _require_positive_int(value: Any, *, label: str) -> int:
    result = _require_nonnegative_int(value, label=label)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite numeric data")
    return result


def normalize_htr_readable_focus(value: str) -> tuple[str, str]:
    """Return the closed semantic key for one readable focus token/subword."""

    if not isinstance(value, str):
        raise TypeError("HTR readable focus text must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    wordpiece_kind = (
        "continuation_subword" if normalized.startswith("##") else "whole_or_initial_token"
    )
    return normalized, wordpiece_kind


def _summary(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("attention summary requires nonempty finite one-dimensional data")
    ordered = tuple(float(value) for value in values.tolist())
    total = math.fsum(ordered)
    return {
        "count": len(ordered),
        "sum": total,
        "mean": total / len(ordered),
        "max": max(ordered),
    }


def _note_max_summary(
    *,
    note_ids: np.ndarray,
    values: np.ndarray,
) -> dict[str, Any]:
    if note_ids.shape != values.shape or note_ids.ndim != 1 or note_ids.size < 1:
        raise ValueError("note-maximum summary inputs are misaligned")
    maxima: list[float] = []
    start = 0
    while start < note_ids.size:
        end = start + 1
        while end < note_ids.size and note_ids[end] == note_ids[start]:
            end += 1
        maxima.append(float(np.max(values[start:end])))
        start = end
    total = math.fsum(maxima)
    return {
        "note_count": len(maxima),
        "sum_of_note_maxima": total,
        "mean_of_note_maxima": total / len(maxima),
        "max_of_note_maxima": max(maxima),
    }


def _score_summaries(
    *,
    note_ids: np.ndarray,
    token_attention: np.ndarray,
    chunk_attention: np.ndarray,
    hierarchical_attention: np.ndarray,
) -> dict[str, Any]:
    return {
        "token_attention": {
            "occurrence_weighted": _summary(token_attention),
            "note_level_max": _note_max_summary(
                note_ids=note_ids,
                values=token_attention,
            ),
        },
        "chunk_attention": {
            "occurrence_weighted": _summary(chunk_attention),
            "note_level_max": _note_max_summary(
                note_ids=note_ids,
                values=chunk_attention,
            ),
        },
        "hierarchical_attention_score": {
            "occurrence_weighted": _summary(hierarchical_attention),
            "note_level_max": _note_max_summary(
                note_ids=note_ids,
                values=hierarchical_attention,
            ),
        },
    }


def _array_bytes(value: np.ndarray) -> bytes:
    if value.dtype.hasobject:
        raise ValueError("authenticated aggregate sidecars cannot contain object arrays")
    contiguous = np.ascontiguousarray(value)
    header = canonical_json(
        {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }
    ).encode("utf-8")
    return header + b"\0" + contiguous.tobytes(order="C")


def _write_npy(path: Path, value: np.ndarray, *, root: Path) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise FileExistsError("aggregate array target must be fresh")
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ValueError("aggregate arrays cannot use object dtype")
    np.save(path, array, allow_pickle=False)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    payload = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "array_content_sha256": _sha256_bytes(_array_bytes(array)),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def _load_npy(path: Path, registration: Mapping[str, Any]) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise ValueError("aggregate sidecar must be a regular file")
    payload = path.read_bytes()
    if (
        _sha256_bytes(payload) != registration.get("sha256")
        or len(payload) != registration.get("size_bytes")
    ):
        raise ValueError("aggregate sidecar bytes do not authenticate")
    # Deliberately bounded/non-mmap for Python 3.14 shared-filesystem safety.
    value = np.load(path, allow_pickle=False, mmap_mode=None)
    if (
        value.dtype.hasobject
        or value.dtype.str != registration.get("dtype")
        or list(value.shape) != registration.get("shape")
        or _sha256_bytes(_array_bytes(value))
        != registration.get("array_content_sha256")
    ):
        raise ValueError("aggregate sidecar array identity changed")
    return value


def _write_json(path: Path, value: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise FileExistsError("aggregate JSON target must be fresh")
    payload = canonical_json(value).encode("utf-8")
    path.write_bytes(payload)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "content_sha256": _sha256_json(value),
    }


def _read_json(path: Path, registration: Mapping[str, Any]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("aggregate JSON must be a regular file")
    payload = path.read_bytes()
    if (
        _sha256_bytes(payload) != registration.get("sha256")
        or len(payload) != registration.get("size_bytes")
    ):
        raise ValueError("aggregate JSON bytes do not authenticate")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("aggregate JSON is invalid") from exc
    if (
        not isinstance(value, dict)
        or canonical_json(value).encode("utf-8") != payload
        or _sha256_json(value) != registration.get("content_sha256")
    ):
        raise ValueError("aggregate JSON content does not authenticate")
    return value


def _utf8_table(values: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    payload = bytearray()
    offsets = [0]
    for value in values:
        encoded = value.encode("utf-8")
        payload.extend(encoded)
        offsets.append(len(payload))
    return (
        np.frombuffer(bytes(payload), dtype=np.uint8).copy(),
        np.asarray(offsets, dtype=np.int64),
    )


def _decode_utf8_table(payload: np.ndarray, offsets: np.ndarray) -> tuple[str, ...]:
    if (
        payload.dtype != np.dtype(np.uint8)
        or offsets.dtype != np.dtype(np.int64)
        or payload.ndim != 1
        or offsets.ndim != 1
        or offsets.size < 1
        or int(offsets[0]) != 0
        or int(offsets[-1]) != payload.size
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("aggregate UTF-8 table is malformed")
    raw = payload.tobytes()
    try:
        return tuple(
            raw[int(offsets[index]) : int(offsets[index + 1])].decode("utf-8")
            for index in range(offsets.size - 1)
        )
    except UnicodeDecodeError as exc:
        raise ValueError("aggregate UTF-8 table contains invalid text") from exc


def _validate_source_payload(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {
            "schema_version",
            "family",
            "architecture_evidence",
            "token_attention_evidence",
        }
        or payload.get("schema_version") != ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA
        or payload.get("family") != HTR_NEURAL
        or not isinstance(payload.get("architecture_evidence"), list)
        or not payload["architecture_evidence"]
        or not isinstance(payload.get("token_attention_evidence"), Mapping)
    ):
        raise ValueError("HTR aggregate source payload is incompatible")
    evidence = [dict(row) for row in payload["architecture_evidence"]]
    package = dict(payload["token_attention_evidence"])
    if (
        package.get("schema_version") != ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_PACKAGE_SCHEMA
        or package.get("sentence_pooling") != "token_attention"
        or package.get("effective_sentence_pooling") != "token_attention"
        or package.get("all_raw_token_occurrences_authenticated") is not True
        or package.get("all_chunk_occurrences_authenticated") is not True
        or package.get("exact_oof_note_coverage") is not True
        or package.get("fold_honest_validation_only_evidence") is not True
        or package.get("top_k_applied_to_raw_inventory") is not False
        or not isinstance(package.get("fold_batches"), list)
        or not package["fold_batches"]
    ):
        raise ValueError("HTR aggregate source lacks complete token-attention authentication")

    batch_keys: set[tuple[str, str, int]] = set()
    token_count = 0
    chunk_count = 0
    note_count = 0
    special_count = 0
    padding_count = 0
    special_mass: list[float] = []
    chunks_by_batch: dict[tuple[str, str, int], int] = {}
    for raw_batch in package["fold_batches"]:
        if not isinstance(raw_batch, Mapping):
            raise ValueError("HTR token-attention fold batch is malformed")
        batch = dict(raw_batch)
        stage = str(batch.get("stage") or "")
        objective = str(batch.get("objective") or "")
        fold = _require_positive_int(batch.get("fold"), label="HTR fold")
        key = (stage, objective, fold)
        if (
            key in batch_keys
            or batch.get("schema_version")
            != ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_BATCH_SCHEMA
            or stage not in _STAGE_ORDER
            or not objective
            or batch.get("sentence_pooling") != "token_attention"
            or batch.get("effective_sentence_pooling") != "token_attention"
            or not isinstance(batch.get("fold_honesty"), Mapping)
            or batch["fold_honesty"].get("evidence_rows")
            != "fold_validation_only"
            or batch["fold_honesty"].get(
                "fit_and_validation_rows_disjoint"
            )
            is not True
            or batch["fold_honesty"].get("generated_after_fit") is not True
            or batch["fold_honesty"].get(
                "validation_rows_used_for_model_fit"
            )
            is not False
            or batch.get("top_k_applied_to_raw_inventory") is not False
            or batch.get("all_overlapping_chunk_occurrences_retained") is not True
        ):
            raise ValueError("HTR token-attention fold batch identity changed")
        batch_keys.add(key)
        batch_tokens = _require_positive_int(
            batch.get("token_occurrence_count"),
            label="HTR fold token count",
        )
        batch_chunks = _require_positive_int(
            batch.get("chunk_count"),
            label="HTR fold chunk count",
        )
        batch_notes = _require_positive_int(
            batch.get("note_count"),
            label="HTR fold note count",
        )
        token_count += batch_tokens
        chunk_count += batch_chunks
        note_count += batch_notes
        special_count += _require_nonnegative_int(
            batch.get("special_token_occurrence_count"),
            label="HTR fold special-token count",
        )
        padding_count += _require_nonnegative_int(
            batch.get("padding_occurrence_count"),
            label="HTR fold padding count",
        )
        special_mass.append(
            _finite(
                batch.get("special_token_attention_mass"),
                label="HTR fold special-token mass",
            )
        )
        chunks_by_batch[key] = batch_chunks
    if (
        token_count != package.get("token_occurrence_count")
        or chunk_count != package.get("chunk_interpretation_count")
        or note_count != package.get("note_interpretation_count")
        or special_count != package.get("special_token_occurrence_count")
        or padding_count != package.get("padding_occurrence_count")
        or not math.isclose(
            math.fsum(special_mass),
            _finite(
                package.get("special_token_attention_mass"),
                label="HTR package special-token mass",
            ),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        or len(evidence) != chunk_count
    ):
        raise ValueError("HTR package counts differ from its authenticated fold batches")

    observed_chunks: dict[tuple[str, str, int], int] = defaultdict(int)
    observed_note_keys: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    for row in evidence:
        if (
            set(row)
            != {
                "witness_kind",
                "schema_version",
                "stage",
                "objective",
                "fold",
                "fit_note_position",
                "fit_row_id",
                "chunk_index",
                "chunk_text",
                "chunk_sha256",
                "attention",
                "readable_token_spans",
                "readable_span_policy",
                "token_inventory_content_sha256",
            }
            or row.get("witness_kind") != "complete_htr_chunk_attention"
            or row.get("schema_version") != ROLE_NEUTRAL_HTR_CHUNK_EVIDENCE_SCHEMA
        ):
            raise ValueError("HTR aggregate source chunk schema changed")
        stage = str(row.get("stage") or "")
        objective = str(row.get("objective") or "")
        fold = _require_positive_int(row.get("fold"), label="HTR chunk fold")
        key = (stage, objective, fold)
        if key not in batch_keys:
            raise ValueError("HTR chunk has no authenticated fold batch")
        text = row.get("chunk_text")
        if (
            not isinstance(text, str)
            or not text
            or _sha256_bytes(text.encode("utf-8")) != row.get("chunk_sha256")
            or not isinstance(row.get("readable_token_spans"), list)
            or not isinstance(row.get("readable_span_policy"), Mapping)
            or row["readable_span_policy"].get("complete_raw_inventory_retained")
            is not True
            or row["readable_span_policy"].get("special_tokens_excluded") is not True
            or row["readable_span_policy"].get(
                "overlapping_chunk_occurrences_retained"
            )
            is not True
        ):
            raise ValueError("HTR aggregate source chunk content changed")
        _finite(row.get("attention"), label="HTR chunk attention")
        _require_nonnegative_int(
            row.get("fit_note_position"),
            label="HTR chunk fit-note position",
        )
        fit_row_id = _require_nonnegative_int(
            row.get("fit_row_id"),
            label="HTR chunk fit-row ID",
        )
        _require_nonnegative_int(row.get("chunk_index"), label="HTR chunk index")
        _require_sha256(
            row.get("token_inventory_content_sha256"),
            label="HTR token inventory",
        )
        observed_chunks[key] += 1
        observed_note_keys[key].add(fit_row_id)
        focus_positions: set[int] = set()
        for rank, span in enumerate(row["readable_token_spans"], start=1):
            if (
                not isinstance(span, Mapping)
                or span.get("schema_version") != ROLE_NEUTRAL_HTR_READABLE_SPAN_SCHEMA
                or span.get("selection_rank") != rank
                or not isinstance(span.get("text"), str)
                or not span["text"]
                or not isinstance(span.get("focus_decoded_token_text"), str)
                or not span["focus_decoded_token_text"]
                or span.get("special_tokens_excluded_from_readable_projection")
                is not True
                or span.get("raw_special_token_mass_retained_in_sidecar") is not True
            ):
                raise ValueError("HTR readable-span source changed")
            focus_position = _require_nonnegative_int(
                span.get("focus_token_position"),
                label="HTR readable focus-token position",
            )
            if focus_position in focus_positions:
                raise ValueError("HTR readable spans duplicate one raw token occurrence")
            focus_positions.add(focus_position)
            token = _finite(span.get("token_attention"), label="HTR token attention")
            chunk = _finite(span.get("chunk_attention"), label="HTR span chunk attention")
            hierarchical = _finite(
                span.get("hierarchical_attention_score"),
                label="HTR hierarchical attention",
            )
            if not math.isclose(
                hierarchical,
                token * chunk,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("HTR hierarchical attention product changed")
    if observed_chunks != chunks_by_batch or any(
        len(observed_note_keys[key])
        != next(
            int(batch["note_count"])
            for batch in package["fold_batches"]
            if (
                str(batch["stage"]),
                str(batch["objective"]),
                int(batch["fold"]),
            )
            == key
        )
        for key in chunks_by_batch
    ):
        raise ValueError("HTR chunk coverage differs from fold-batch coverage")
    return evidence, package


@dataclass(frozen=True)
class HtrSemanticAggregationResult:
    payload: Mapping[str, Any]
    scope_manifest: Mapping[str, Any]
    scope_manifest_path: Path


def _ordered_indices(
    cross_index: np.ndarray,
    fold: np.ndarray,
    note: np.ndarray,
    chunk: np.ndarray,
    token_position: np.ndarray,
    source_index: np.ndarray,
    span_index: np.ndarray,
) -> np.ndarray:
    return np.lexsort(
        (
            span_index,
            source_index,
            token_position,
            chunk,
            note,
            fold,
            cross_index,
        )
    )


def _unique_pair_count(first: np.ndarray, second: np.ndarray) -> int:
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("unique-pair inputs are misaligned")
    if first.size == 0:
        return 0
    pairs = np.empty(
        first.size,
        dtype=[("first", first.dtype), ("second", second.dtype)],
    )
    pairs["first"] = first
    pairs["second"] = second
    return int(np.unique(pairs).size)


def _context_examples(
    *,
    indices: np.ndarray,
    hierarchical: np.ndarray,
    token: np.ndarray,
    chunk_attention: np.ndarray,
    note: np.ndarray,
    chunk: np.ndarray,
    token_position: np.ndarray,
    source_index: np.ndarray,
    span_index: np.ndarray,
    variant_index: np.ndarray,
    variants: Sequence[str],
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (int(index) for index in indices.tolist()),
        key=lambda index: (
            -float(hierarchical[index]),
            -float(token[index]),
            int(note[index]),
            int(chunk[index]),
            int(token_position[index]),
            int(source_index[index]),
            int(span_index[index]),
        ),
    )
    output: list[dict[str, Any]] = []
    seen_variants: set[int] = set()
    for index in ranked:
        display_index = int(variant_index[index])
        if display_index in seen_variants:
            continue
        seen_variants.add(display_index)
        text = variants[display_index]
        output.append(
            {
                "display_text": text,
                "display_text_sha256": _sha256_bytes(text.encode("utf-8")),
                "hierarchical_attention_score": float(hierarchical[index]),
                "token_attention": float(token[index]),
                "chunk_attention": float(chunk_attention[index]),
                "source_coordinate_sha256": _sha256_json(
                    {
                        "source_record_index": int(source_index[index]),
                        "readable_span_index": int(span_index[index]),
                        "fit_note_position": int(note[index]),
                        "chunk_index": int(chunk[index]),
                        "focus_token_position": int(token_position[index]),
                    }
                ),
            }
        )
        if len(output) == limit:
            break
    return output


def _model_facing_aggregate(
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one fully authenticated aggregate into compact prompt data."""

    fold_rows = aggregate.get("fold_local_summaries")
    context_rows = aggregate.get("context_windows")
    if not isinstance(fold_rows, list) or not isinstance(
        context_rows,
        list,
    ):
        raise ValueError("HTR semantic aggregate lacks its compact projection inputs")
    body = {
        "schema_version": HTR_STAGE2_MODEL_AGGREGATE_SCHEMA,
        "aggregate_id": str(aggregate["aggregate_id"]),
        "source_aggregate_content_sha256": _require_sha256(
            aggregate.get("content_sha256"),
            label="HTR source semantic aggregate",
        ),
        "stage": str(aggregate["stage"]),
        "objective": str(aggregate["objective"]),
        "normalized_focus_text": str(
            aggregate["normalized_focus_text"]
        ),
        "wordpiece_kind": str(aggregate["wordpiece_kind"]),
        "occurrence_count": int(aggregate["occurrence_count"]),
        "unique_note_count": int(aggregate["unique_note_count"]),
        "unique_chunk_count": int(aggregate["unique_chunk_count"]),
        "attention_summaries": aggregate["attention_summaries"],
        "fold_support": [
            {
                "fold": int(row["fold"]),
                "fold_aggregate_id": str(row["fold_aggregate_id"]),
                "fold_aggregate_content_sha256": _require_sha256(
                    row.get("content_sha256"),
                    label="HTR fold-local semantic aggregate",
                ),
                "occurrence_count": int(row["occurrence_count"]),
                "unique_note_count": int(row["unique_note_count"]),
                "unique_chunk_count": int(row["unique_chunk_count"]),
            }
            for row in fold_rows
        ],
        "display_text_variant_count": int(
            aggregate["display_text_variant_count"]
        ),
        "display_text_variant_indices_sha256": _require_sha256(
            aggregate.get("display_text_variant_indices_sha256"),
            label="HTR aggregate display-text variants",
        ),
        "context_windows": [
            {
                "display_text": str(row["display_text"]),
                "token_attention": float(row["token_attention"]),
                "chunk_attention": float(row["chunk_attention"]),
                "hierarchical_attention_score": float(
                    row["hierarchical_attention_score"]
                ),
            }
            for row in context_rows
        ],
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
    max_bytes: int,
    max_token_upper_bound: int,
    max_aggregates: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for raw_aggregate in aggregates:
        aggregate = _model_facing_aggregate(raw_aggregate)
        grouped[
            (
                str(aggregate["stage"]),
                str(aggregate["objective"]),
            )
        ].append(aggregate)
    batches: list[dict[str, Any]] = []
    sizes: list[int] = []
    for (stage, objective), rows in sorted(
        grouped.items(),
        key=lambda item: (
            _STAGE_ORDER[item[0][0]],
            item[0][1],
        ),
    ):
        ordered = sorted(rows, key=lambda row: str(row["aggregate_id"]))
        provisional: list[list[Mapping[str, Any]]] = []
        current: list[Mapping[str, Any]] = []
        for row in ordered:
            candidate = [*current, row]
            if len(candidate) > max_aggregates:
                provisional.append(current)
                current = [row]
                continue
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
            if size > min(max_bytes, max_token_upper_bound):
                if not current:
                    raise ValueError(
                        "one HTR semantic aggregate exceeds the configured "
                        "model-facing byte/token bound"
                    )
                provisional.append(current)
                current = [row]
            else:
                current = candidate
        if current:
            provisional.append(current)
        for batch_index, members in enumerate(provisional, start=1):
            body = {
                "schema_version": HTR_STAGE2_AGGREGATE_BATCH_SCHEMA,
                "stage": stage,
                "objective": objective,
                "batch_index": batch_index,
                "batch_count": len(provisional),
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
                raise RuntimeError("HTR semantic batch exceeds its byte/token bound")
            batches.append(batch)
            sizes.append(size)
    return batches, sizes


def build_htr_semantic_aggregation_scope(
    *,
    root: Path | str,
    source_payload: Mapping[str, Any],
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
) -> HtrSemanticAggregationResult:
    """Build one fresh, authenticated aggregate/reverse-index scope."""

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
    if (
        not isinstance(source_fit_seal_locator, str)
        or not source_fit_seal_locator
        or Path(source_fit_seal_locator).is_absolute()
        or not logical_scope_id
        or not physical_owner_scope_id
    ):
        raise ValueError("HTR aggregate source/scopes are invalid")
    _require_positive_int(outer_fold, label="HTR aggregate outer fold")
    _require_nonnegative_int(context_epoch, label="HTR aggregate context epoch")
    max_bytes = _require_positive_int(
        max_model_facing_batch_bytes,
        label="HTR aggregate batch byte bound",
    )
    max_tokens = _require_positive_int(
        max_model_facing_token_upper_bound,
        label="HTR aggregate batch token bound",
    )
    max_aggregates = _require_positive_int(
        max_model_facing_aggregates_per_batch,
        label="HTR aggregate batch aggregate-count bound",
    )
    context_limit = _require_positive_int(
        context_windows_per_aggregate,
        label="HTR aggregate context-window bound",
    )
    evidence, package = _validate_source_payload(source_payload)
    if _sha256_json(source_payload) != source_payload_sha:
        raise ValueError("HTR source payload digest differs from its fit seal")

    span_count = sum(len(row["readable_token_spans"]) for row in evidence)
    if span_count < 1:
        raise ValueError("HTR aggregate source has no eligible readable spans")
    source_count = len(evidence)

    source_record_index = np.empty(span_count, dtype=np.int64)
    readable_span_index = np.empty(span_count, dtype=np.int16)
    fold = np.empty(span_count, dtype=np.int16)
    fit_note_position = np.empty(span_count, dtype=np.int64)
    fit_row_id = np.empty(span_count, dtype=np.int64)
    chunk_index = np.empty(span_count, dtype=np.int32)
    focus_token_position = np.empty(span_count, dtype=np.int32)
    display_variant_index = np.empty(span_count, dtype=np.int64)
    cross_aggregate_index = np.empty(span_count, dtype=np.int64)
    fold_local_aggregate_index = np.full(span_count, -1, dtype=np.int64)
    token_attention = np.empty(span_count, dtype=np.float64)
    chunk_attention = np.empty(span_count, dtype=np.float64)
    hierarchical_attention = np.empty(span_count, dtype=np.float64)
    source_hashes = np.empty(source_count, dtype="S64")

    variant_by_text: dict[str, int] = {}
    variants: list[str] = []
    cross_key_by_occurrence: list[tuple[str, str, str, str]] = [
        ("", "", "", "")
    ] * span_count
    non_readable_count = 0
    cursor = 0
    for record_index, row in enumerate(evidence):
        source_hashes[record_index] = _sha256_json(row).encode("ascii")
        stage = str(row["stage"])
        objective = str(row["objective"])
        row_fold = int(row["fold"])
        for span_position, raw_span in enumerate(row["readable_token_spans"]):
            span = dict(raw_span)
            focus = str(span["focus_decoded_token_text"])
            normalized_focus, wordpiece_kind = normalize_htr_readable_focus(focus)
            if not normalized_focus:
                # Retained in the reverse index and explicit accounting bucket.
                non_readable_count += 1
                cross_key = (stage, objective, "", "non_readable")
            else:
                cross_key = (
                    stage,
                    objective,
                    normalized_focus,
                    wordpiece_kind,
                )
            display = str(span["text"])
            variant = variant_by_text.get(display)
            if variant is None:
                variant = len(variants)
                variant_by_text[display] = variant
                variants.append(display)
            source_record_index[cursor] = record_index
            readable_span_index[cursor] = span_position
            fold[cursor] = row_fold
            fit_note_position[cursor] = int(row["fit_note_position"])
            fit_row_id[cursor] = int(row["fit_row_id"])
            chunk_index[cursor] = int(row["chunk_index"])
            focus_token_position[cursor] = int(span["focus_token_position"])
            display_variant_index[cursor] = variant
            token_attention[cursor] = float(span["token_attention"])
            chunk_attention[cursor] = float(span["chunk_attention"])
            hierarchical_attention[cursor] = float(
                span["hierarchical_attention_score"]
            )
            cross_key_by_occurrence[cursor] = cross_key
            cursor += 1
    if cursor != span_count:
        raise RuntimeError("HTR readable-span preallocation accounting changed")

    readable_cross_keys = sorted(
        {
            key
            for key in cross_key_by_occurrence
            if key[3] != "non_readable"
        },
        key=lambda key: (
            _STAGE_ORDER[key[0]],
            key[1],
            key[2],
            key[3],
        ),
    )
    cross_index_by_key = {
        key: index for index, key in enumerate(readable_cross_keys)
    }
    for index, key in enumerate(cross_key_by_occurrence):
        cross_aggregate_index[index] = cross_index_by_key.get(key, -1)

    readable_mask = cross_aggregate_index >= 0
    readable_indices = np.flatnonzero(readable_mask)
    ordered_readable = readable_indices[
        _ordered_indices(
            cross_aggregate_index[readable_indices],
            fold[readable_indices],
            fit_row_id[readable_indices],
            chunk_index[readable_indices],
            focus_token_position[readable_indices],
            source_record_index[readable_indices],
            readable_span_index[readable_indices],
        )
    ]
    non_readable_indices = np.flatnonzero(~readable_mask)
    order = np.concatenate((ordered_readable, non_readable_indices))
    arrays = {
        "source_record_index": source_record_index[order],
        "readable_span_index": readable_span_index[order],
        "fold": fold[order],
        "fit_note_position": fit_note_position[order],
        "fit_row_id": fit_row_id[order],
        "chunk_index": chunk_index[order],
        "focus_token_position": focus_token_position[order],
        "display_variant_index": display_variant_index[order],
        "cross_aggregate_index": cross_aggregate_index[order],
        "fold_local_aggregate_index": fold_local_aggregate_index[order],
        "token_attention": token_attention[order],
        "chunk_attention": chunk_attention[order],
        "hierarchical_attention_score": hierarchical_attention[order],
    }

    fold_key_to_index: dict[tuple[int, int], int] = {}
    fold_records: list[dict[str, Any]] = []
    cross_records: list[dict[str, Any]] = []
    readable_total = int(ordered_readable.size)
    position = 0
    while position < readable_total:
        cross_id = int(arrays["cross_aggregate_index"][position])
        cross_end = position + 1
        while (
            cross_end < readable_total
            and int(arrays["cross_aggregate_index"][cross_end]) == cross_id
        ):
            cross_end += 1
        stage, objective, normalized_focus, wordpiece_kind = readable_cross_keys[
            cross_id
        ]
        fold_summaries: list[dict[str, Any]] = []
        local_position = position
        while local_position < cross_end:
            local_fold = int(arrays["fold"][local_position])
            local_end = local_position + 1
            while (
                local_end < cross_end
                and int(arrays["fold"][local_end]) == local_fold
            ):
                local_end += 1
            local_index = len(fold_records)
            fold_key_to_index[(cross_id, local_fold)] = local_index
            arrays["fold_local_aggregate_index"][
                local_position:local_end
            ] = local_index
            local_slice = slice(local_position, local_end)
            local_notes = arrays["fit_row_id"][local_slice]
            local_chunks = arrays["chunk_index"][local_slice]
            local_id_body = {
                "schema_version": HTR_STAGE2_FOLD_AGGREGATE_SCHEMA,
                "scope_binding_sha256": scope_binding,
                "stage": stage,
                "objective": objective,
                "fold": local_fold,
                "normalized_focus_text": normalized_focus,
                "wordpiece_kind": wordpiece_kind,
            }
            local_record = {
                **local_id_body,
                "fold_aggregate_id": (
                    f"htr_fold_aggregate_{_sha256_json(local_id_body)}"
                ),
                "occurrence_count": local_end - local_position,
                "unique_note_count": int(np.unique(local_notes).size),
                "unique_chunk_count": _unique_pair_count(
                    local_notes,
                    local_chunks,
                ),
                "attention_summaries": _score_summaries(
                    note_ids=local_notes,
                    token_attention=arrays["token_attention"][local_slice],
                    chunk_attention=arrays["chunk_attention"][local_slice],
                    hierarchical_attention=arrays[
                        "hierarchical_attention_score"
                    ][local_slice],
                ),
                "reverse_index_start": local_position,
                "reverse_index_count": local_end - local_position,
            }
            local_record["content_sha256"] = _sha256_json(local_record)
            fold_records.append(local_record)
            fold_summaries.append(local_record)
            local_position = local_end

        cross_slice = slice(position, cross_end)
        cross_notes = arrays["fit_row_id"][cross_slice]
        cross_chunks = arrays["chunk_index"][cross_slice]
        variant_refs = sorted(
            {
                int(value)
                for value in arrays["display_variant_index"][
                    cross_slice
                ].tolist()
            }
        )
        identity_body = {
            "schema_version": HTR_STAGE2_CROSS_FOLD_AGGREGATE_SCHEMA,
            "scope_binding_sha256": scope_binding,
            "stage": stage,
            "objective": objective,
            "normalized_focus_text": normalized_focus,
            "wordpiece_kind": wordpiece_kind,
            "cross_fold_consolidation": (
                HTR_STAGE2_CROSS_FOLD_CONSOLIDATION_SCHEMA
            ),
        }
        aggregate = {
            **identity_body,
            "aggregate_id": (
                f"htr_cross_fold_aggregate_{_sha256_json(identity_body)}"
            ),
            "occurrence_count": cross_end - position,
            "unique_note_count": int(np.unique(cross_notes).size),
            "unique_chunk_count": _unique_pair_count(
                cross_notes,
                cross_chunks,
            ),
            "attention_summaries": _score_summaries(
                note_ids=cross_notes,
                token_attention=arrays["token_attention"][cross_slice],
                chunk_attention=arrays["chunk_attention"][cross_slice],
                hierarchical_attention=arrays[
                    "hierarchical_attention_score"
                ][cross_slice],
            ),
            "fold_local_summaries": fold_summaries,
            "folds_retained_before_consolidation": sorted(
                int(row["fold"]) for row in fold_summaries
            ),
            "display_text_variant_count": len(variant_refs),
            "display_text_variant_indices_sha256": _sha256_json(variant_refs),
            "context_windows": _context_examples(
                indices=np.arange(position, cross_end, dtype=np.int64),
                hierarchical=arrays["hierarchical_attention_score"],
                token=arrays["token_attention"],
                chunk_attention=arrays["chunk_attention"],
                note=arrays["fit_row_id"],
                chunk=arrays["chunk_index"],
                token_position=arrays["focus_token_position"],
                source_index=arrays["source_record_index"],
                span_index=arrays["readable_span_index"],
                variant_index=arrays["display_variant_index"],
                variants=variants,
                limit=context_limit,
            ),
            "context_window_policy": HTR_STAGE2_CONTEXT_POLICY_SCHEMA,
            "reverse_index_start": position,
            "reverse_index_count": cross_end - position,
            "overlap_accounting": {
                "raw_overlapping_chunk_occurrence_count": cross_end - position,
                "unique_supporting_note_count": int(np.unique(cross_notes).size),
                "unique_supporting_chunk_count": _unique_pair_count(
                    cross_notes,
                    cross_chunks,
                ),
                "note_level_maxima_separate_from_occurrence_summaries": True,
            },
            "hierarchical_attention_interpretation": (
                "ranking_heuristic_not_causal_attribution"
            ),
        }
        aggregate["content_sha256"] = _sha256_json(aggregate)
        cross_records.append(aggregate)
        position = cross_end

    special_count = int(package["special_token_occurrence_count"])
    token_count = int(package["token_occurrence_count"])
    raw_nonselected_nonspecial = (
        token_count - special_count - int(package["padding_occurrence_count"]) - span_count
    )
    if raw_nonselected_nonspecial < 0:
        raise ValueError("readable span projection exceeds raw nonspecial token inventory")

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            dir=target.parent,
        )
    )
    try:
        reverse_root = staging / "reverse_index"
        reverse_root.mkdir()
        registrations: dict[str, dict[str, Any]] = {}
        for name, value in arrays.items():
            registrations[name] = _write_npy(
                reverse_root / f"{name}.npy",
                value,
                root=staging,
            )
        registrations["source_record_sha256"] = _write_npy(
            reverse_root / "source_record_sha256.npy",
            source_hashes,
            root=staging,
        )
        variant_utf8, variant_offsets = _utf8_table(variants)
        registrations["display_text_variant_utf8"] = _write_npy(
            reverse_root / "display_text_variant_utf8.npy",
            variant_utf8,
            root=staging,
        )
        registrations["display_text_variant_byte_offsets"] = _write_npy(
            reverse_root / "display_text_variant_byte_offsets.npy",
            variant_offsets,
            root=staging,
        )

        reverse_body = {
            "schema_version": HTR_STAGE2_REVERSE_INDEX_SCHEMA,
            "source_payload_content_sha256": source_payload_sha,
            "source_fit_seal_content_sha256": source_fit_sha,
            "scope_binding_sha256": scope_binding,
            "eligible_readable_span_occurrence_count": span_count,
            "aggregated_readable_span_occurrence_count": readable_total,
            "non_readable_accounting_bucket_count": non_readable_count,
            "cross_fold_aggregate_count": len(cross_records),
            "fold_local_aggregate_count": len(fold_records),
            "source_chunk_record_count": source_count,
            "display_text_variant_count": len(variants),
            "arrays": registrations,
            "occurrence_order": (
                "cross_aggregate_fold_row_chunk_token_source_span_then_"
                "non_readable_v1"
            ),
            "every_eligible_readable_span_accounted_exactly_once": True,
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
            "fold_local_aggregate_count": len(fold_records),
            "fold_local_aggregates": fold_records,
            "folds_kept_separate": True,
            "content_sha256": _sha256_json(fold_records),
        }
        fold_registration = _write_json(
            staging / "fold_local_aggregates.json",
            fold_body,
            root=staging,
        )
        cross_body = {
            "schema_version": HTR_STAGE2_CROSS_FOLD_AGGREGATE_SCHEMA,
            "scope_binding_sha256": scope_binding,
            "cross_fold_aggregate_count": len(cross_records),
            "cross_fold_aggregates": cross_records,
            "fold_local_source_content_sha256": (
                fold_registration["content_sha256"]
            ),
            "cross_fold_consolidation": (
                HTR_STAGE2_CROSS_FOLD_CONSOLIDATION_SCHEMA
            ),
            "complete_semantic_aggregate_inventory": True,
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
            "token_attention_package_content_sha256": _require_sha256(
                package.get("content_sha256"),
                label="HTR token-attention package",
            ),
            "token_occurrence_count": token_count,
            "chunk_interpretation_count": int(
                package["chunk_interpretation_count"]
            ),
            "special_token_occurrence_count": special_count,
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
            "eligible_readable_span_occurrence_count": span_count,
        }
        cross_reference = {
            "relative_path": cross_registration["relative_path"],
            "content_sha256": cross_registration["content_sha256"],
            "cross_fold_aggregate_count": len(cross_records),
            "complete_semantic_aggregate_inventory": True,
        }
        batches, batch_sizes = _pack_batches(
            cross_records,
            raw_reference=raw_reference,
            reverse_index_reference=reverse_reference,
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
        # Bind the exact prompt-size inventory to the same canonical delivery
        # order used by the catalog, not to the provisional packing order.
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
            "cross_fold_consolidation": (
                HTR_STAGE2_CROSS_FOLD_CONSOLIDATION_SCHEMA
            ),
            "batching": HTR_STAGE2_BATCHING_SCHEMA,
            "raw_evidence_reference": raw_reference,
            "reverse_index_reference": reverse_reference,
            "cross_fold_aggregate_reference": cross_reference,
            "eligible_readable_span_occurrence_count": span_count,
            "aggregated_readable_span_occurrence_count": readable_total,
            "non_readable_accounting_bucket_count": non_readable_count,
            "special_token_accounting_bucket": {
                "occurrence_count": special_count,
                "attention_mass": float(
                    package["special_token_attention_mass"]
                ),
                "excluded_from_readable_phrases": True,
                "retained_in_raw_authenticated_package": True,
            },
            "raw_nonselected_nonspecial_token_accounting_bucket": {
                "occurrence_count": raw_nonselected_nonspecial,
                "raw_inventory_retained": True,
                "model_facing_copy_emitted": False,
            },
            "source_chunk_interpretation_count": int(
                package["chunk_interpretation_count"]
            ),
            "source_stage_objective_fold_counts": sorted(
                [
                    {
                        "stage": str(batch["stage"]),
                        "objective": str(batch["objective"]),
                        "fold": int(batch["fold"]),
                        "token_occurrence_count": int(
                            batch["token_occurrence_count"]
                        ),
                        "chunk_interpretation_count": int(
                            batch["chunk_count"]
                        ),
                        "note_interpretation_count": int(
                            batch["note_count"]
                        ),
                        "special_token_occurrence_count": int(
                            batch["special_token_occurrence_count"]
                        ),
                        "special_token_attention_mass": float(
                            batch["special_token_attention_mass"]
                        ),
                        "padding_occurrence_count": int(
                            batch["padding_occurrence_count"]
                        ),
                    }
                    for batch in package["fold_batches"]
                ],
                key=canonical_json,
            ),
            "fold_local_aggregate_count": len(fold_records),
            "cross_fold_aggregate_count": len(cross_records),
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
            "one_atom_per_source_chunk_design_call_count": math.ceil(
                int(package["chunk_interpretation_count"]) / 2
            ),
            "planned_htr_interpretation_call_count": len(batches),
            "no_top_k_sampling_or_truncation": True,
            "every_semantic_aggregate_delivered_exactly_once": True,
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
            "source_token_attention_package_content_sha256": (
                raw_reference[
                    "token_attention_package_content_sha256"
                ]
            ),
            "model_facing_payload": payload_registration,
            "reverse_index_manifest": reverse_registration,
            "fold_local_aggregates": fold_registration,
            "cross_fold_aggregates": cross_registration,
            "array_file_count": len(registrations),
            "raw_token_arrays_copied": False,
            "all_source_chunks_accounted": True,
            "all_readable_span_occurrences_accounted_exactly_once": True,
            "no_top_k_sampling_or_truncation": True,
            "summary": semantic_summary,
        }
        scope_manifest = {
            **scope_body,
            "content_sha256": _sha256_json(scope_body),
        }
        scope_registration = _write_json(
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


def validate_htr_semantic_aggregation_scope(
    *,
    root: Path | str,
    source_payload: Mapping[str, Any],
    expected_source_fit_seal_content_sha256: str,
    expected_source_payload_content_sha256: str,
    expected_scope_binding_sha256: str,
) -> HtrSemanticAggregationResult:
    """Reopen a scope and prove complete source-to-aggregate accounting."""

    target = Path(root)
    if (
        not target.is_absolute()
        or target.is_symlink()
        or target.resolve(strict=True) != target
        or not target.is_dir()
    ):
        raise ValueError("HTR aggregate scope root is not canonical")
    manifest_path = target / "scope_manifest.json"
    manifest_payload = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("HTR aggregate scope manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or canonical_json(manifest).encode("utf-8") != manifest_payload
        or manifest.get("schema_version") != HTR_STAGE2_SCOPE_MANIFEST_SCHEMA
        or manifest.get("content_sha256")
        != _sha256_json(
            {
                key: child
                for key, child in manifest.items()
                if key != "content_sha256"
            }
        )
        or manifest.get("source_fit_seal_content_sha256")
        != _require_sha256(
            expected_source_fit_seal_content_sha256,
            label="expected HTR source fit seal",
        )
        or manifest.get("source_payload_content_sha256")
        != _require_sha256(
            expected_source_payload_content_sha256,
            label="expected HTR source payload",
        )
        or manifest.get("scope_binding_sha256")
        != _require_sha256(
            expected_scope_binding_sha256,
            label="expected HTR scope binding",
        )
    ):
        raise ValueError("HTR aggregate scope manifest identity is invalid")
    evidence, package = _validate_source_payload(source_payload)
    if _sha256_json(source_payload) != manifest["source_payload_content_sha256"]:
        raise ValueError("HTR aggregate source payload changed")

    reverse_registration = manifest.get("reverse_index_manifest")
    payload_registration = manifest.get("model_facing_payload")
    fold_registration = manifest.get("fold_local_aggregates")
    cross_registration = manifest.get("cross_fold_aggregates")
    if not all(
        isinstance(value, Mapping)
        for value in (
            reverse_registration,
            payload_registration,
            fold_registration,
            cross_registration,
        )
    ):
        raise ValueError("HTR aggregate scope registrations are incomplete")
    reverse = _read_json(
        target / str(reverse_registration["relative_path"]),
        reverse_registration,
    )
    payload = _read_json(
        target / str(payload_registration["relative_path"]),
        payload_registration,
    )
    fold_payload = _read_json(
        target / str(fold_registration["relative_path"]),
        fold_registration,
    )
    cross_payload = _read_json(
        target / str(cross_registration["relative_path"]),
        cross_registration,
    )
    if (
        reverse.get("schema_version") != HTR_STAGE2_REVERSE_INDEX_SCHEMA
        or payload.get("schema_version") != HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA
        or payload.get("family") != HTR_NEURAL
        or not isinstance(payload.get("semantic_aggregation"), Mapping)
        or fold_payload.get("schema_version")
        != HTR_STAGE2_FOLD_AGGREGATE_SCHEMA
        or cross_payload.get("schema_version")
        != HTR_STAGE2_CROSS_FOLD_AGGREGATE_SCHEMA
        or cross_payload.get("content_sha256")
        != _sha256_json(
            {
                key: child
                for key, child in cross_payload.items()
                if key != "content_sha256"
            }
        )
        or cross_payload.get("complete_semantic_aggregate_inventory")
        is not True
        or cross_payload.get("cross_fold_consolidation")
        != HTR_STAGE2_CROSS_FOLD_CONSOLIDATION_SCHEMA
        or cross_payload.get("fold_local_source_content_sha256")
        != fold_registration.get("content_sha256")
        or payload.get("content_sha256")
        != _sha256_json(
            {
                key: child
                for key, child in payload.items()
                if key != "content_sha256"
            }
        )
    ):
        raise ValueError("HTR aggregate payload schemas changed")
    summary = payload["semantic_aggregation"]
    array_registrations = reverse.get("arrays")
    if not isinstance(array_registrations, Mapping):
        raise ValueError("HTR aggregate reverse-index arrays are missing")
    loaded = {
        name: _load_npy(
            target / str(registration["relative_path"]),
            registration,
        )
        for name, registration in array_registrations.items()
    }
    variants = _decode_utf8_table(
        loaded["display_text_variant_utf8"],
        loaded["display_text_variant_byte_offsets"],
    )
    occurrence_names = {
        "source_record_index",
        "readable_span_index",
        "fold",
        "fit_note_position",
        "fit_row_id",
        "chunk_index",
        "focus_token_position",
        "display_variant_index",
        "cross_aggregate_index",
        "fold_local_aggregate_index",
        "token_attention",
        "chunk_attention",
        "hierarchical_attention_score",
    }
    occurrence_lengths = {loaded[name].shape for name in occurrence_names}
    expected_span_count = sum(
        len(row["readable_token_spans"]) for row in evidence
    )
    if (
        occurrence_lengths != {(expected_span_count,)}
        or reverse.get("eligible_readable_span_occurrence_count")
        != expected_span_count
        or len(loaded["source_record_sha256"]) != len(evidence)
        or len(variants) != reverse.get("display_text_variant_count")
    ):
        raise ValueError("HTR aggregate reverse-index coverage is incomplete")

    seen: set[tuple[int, int]] = set()
    readable_count = 0
    for position in range(expected_span_count):
        source_index = int(loaded["source_record_index"][position])
        span_index = int(loaded["readable_span_index"][position])
        coordinate = (source_index, span_index)
        if (
            coordinate in seen
            or source_index < 0
            or source_index >= len(evidence)
            or span_index < 0
            or span_index
            >= len(evidence[source_index]["readable_token_spans"])
        ):
            raise ValueError("HTR aggregate reverse index duplicates or substitutes a span")
        seen.add(coordinate)
        source = evidence[source_index]
        span = source["readable_token_spans"][span_index]
        if (
            loaded["source_record_sha256"][source_index].decode("ascii")
            != _sha256_json(source)
            or int(loaded["fold"][position]) != int(source["fold"])
            or int(loaded["fit_note_position"][position])
            != int(source["fit_note_position"])
            or int(loaded["fit_row_id"][position]) != int(source["fit_row_id"])
            or int(loaded["chunk_index"][position]) != int(source["chunk_index"])
            or int(loaded["focus_token_position"][position])
            != int(span["focus_token_position"])
            or variants[int(loaded["display_variant_index"][position])]
            != span["text"]
            or not math.isclose(
                float(loaded["token_attention"][position]),
                float(span["token_attention"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(loaded["chunk_attention"][position]),
                float(span["chunk_attention"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(loaded["hierarchical_attention_score"][position]),
                float(span["hierarchical_attention_score"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("HTR aggregate reverse index differs from source evidence")
        normalized, _kind = normalize_htr_readable_focus(
            str(span["focus_decoded_token_text"])
        )
        readable_count += int(bool(normalized))
    if (
        len(seen) != expected_span_count
        or readable_count
        != reverse.get("aggregated_readable_span_occurrence_count")
        or expected_span_count - readable_count
        != reverse.get("non_readable_accounting_bucket_count")
    ):
        raise ValueError("HTR readable-span accounting is not exact")

    batches = payload.get("architecture_evidence")
    if not isinstance(batches, list) or not batches:
        raise ValueError("HTR aggregate model-facing batches are empty")
    delivered_ids: list[str] = []
    batch_sizes: list[int] = []
    delivered_aggregates: list[dict[str, Any]] = []
    for item in batches:
        if (
            not isinstance(item, Mapping)
            or item.get("atom_kind") != "htr_semantic_aggregate_batch"
            or not isinstance(item.get("content"), Mapping)
            or not isinstance(item["content"].get("aggregate_batch"), Mapping)
        ):
            raise ValueError("HTR aggregate model-facing atom changed")
        batch = item["content"]["aggregate_batch"]
        body = {key: child for key, child in batch.items() if key != "content_sha256"}
        if (
            batch.get("schema_version") != HTR_STAGE2_AGGREGATE_BATCH_SCHEMA
            or batch.get("content_sha256") != _sha256_json(body)
            or batch.get("aggregate_count") != len(batch.get("aggregates") or ())
        ):
            raise ValueError("HTR aggregate batch does not authenticate")
        batch_sizes.append(len(canonical_json(batch).encode("utf-8")))
        for aggregate in batch["aggregates"]:
            aggregate_body = {
                key: child
                for key, child in aggregate.items()
                if key != "content_sha256"
            }
            if (
                aggregate.get("schema_version")
                != HTR_STAGE2_MODEL_AGGREGATE_SCHEMA
                or aggregate.get("content_sha256") != _sha256_json(aggregate_body)
                or aggregate.get("complete_semantic_accounting")
                is not True
                or aggregate.get(
                    "hierarchical_attention_interpretation"
                )
                != "ranking_heuristic_not_causal_attribution"
            ):
                raise ValueError(
                    "HTR model-facing semantic aggregate does not authenticate"
                )
            delivered_ids.append(str(aggregate["aggregate_id"]))
            delivered_aggregates.append(dict(aggregate))

    fold_records = fold_payload.get("fold_local_aggregates")
    cross_records = cross_payload.get("cross_fold_aggregates")
    if (
        not isinstance(fold_records, list)
        or fold_payload.get("fold_local_aggregate_count") != len(fold_records)
        or fold_payload.get("folds_kept_separate") is not True
        or fold_payload.get("content_sha256") != _sha256_json(fold_records)
        or not isinstance(cross_records, list)
        or cross_payload.get("cross_fold_aggregate_count")
        != len(cross_records)
    ):
        raise ValueError("HTR semantic aggregate inventories changed")

    def source_key(position: int) -> tuple[str, str, str, str]:
        source_index = int(loaded["source_record_index"][position])
        span_index = int(loaded["readable_span_index"][position])
        source = evidence[source_index]
        span = source["readable_token_spans"][span_index]
        normalized, kind = normalize_htr_readable_focus(
            str(span["focus_decoded_token_text"])
        )
        return (
            str(source["stage"]),
            str(source["objective"]),
            normalized,
            kind,
        )

    observed_fold_positions: list[int] = []
    folds_by_cross_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for local_index, record in enumerate(fold_records):
        if not isinstance(record, Mapping):
            raise ValueError("HTR fold-local aggregate is malformed")
        body = {
            key: child for key, child in record.items() if key != "content_sha256"
        }
        start = _require_nonnegative_int(
            record.get("reverse_index_start"),
            label="HTR fold-local reverse-index start",
        )
        count = _require_positive_int(
            record.get("reverse_index_count"),
            label="HTR fold-local reverse-index count",
        )
        end = start + count
        if (
            record.get("schema_version") != HTR_STAGE2_FOLD_AGGREGATE_SCHEMA
            or record.get("content_sha256") != _sha256_json(body)
            or end > readable_count
            or not np.all(
                loaded["fold_local_aggregate_index"][start:end]
                == local_index
            )
        ):
            raise ValueError("HTR fold-local aggregate does not authenticate")
        cross_indices = np.unique(
            loaded["cross_aggregate_index"][start:end]
        )
        observed_folds = np.unique(loaded["fold"][start:end])
        if cross_indices.size != 1 or observed_folds.size != 1:
            raise ValueError("HTR fold-local aggregate mixes folds or semantic keys")
        cross_index = int(cross_indices[0])
        stage, objective, normalized_focus, wordpiece_kind = source_key(start)
        if any(source_key(position) != source_key(start) for position in range(start, end)):
            raise ValueError("HTR fold-local aggregate mixes normalized spans")
        notes = loaded["fit_row_id"][start:end]
        chunks = loaded["chunk_index"][start:end]
        identity = {
            "schema_version": HTR_STAGE2_FOLD_AGGREGATE_SCHEMA,
            "scope_binding_sha256": manifest["scope_binding_sha256"],
            "stage": stage,
            "objective": objective,
            "fold": int(observed_folds[0]),
            "normalized_focus_text": normalized_focus,
            "wordpiece_kind": wordpiece_kind,
        }
        expected = {
            **identity,
            "fold_aggregate_id": (
                f"htr_fold_aggregate_{_sha256_json(identity)}"
            ),
            "occurrence_count": count,
            "unique_note_count": int(np.unique(notes).size),
            "unique_chunk_count": _unique_pair_count(notes, chunks),
            "attention_summaries": _score_summaries(
                note_ids=notes,
                token_attention=loaded["token_attention"][start:end],
                chunk_attention=loaded["chunk_attention"][start:end],
                hierarchical_attention=loaded[
                    "hierarchical_attention_score"
                ][start:end],
            ),
            "reverse_index_start": start,
            "reverse_index_count": count,
        }
        expected["content_sha256"] = _sha256_json(expected)
        if dict(record) != expected:
            raise ValueError("HTR fold-local aggregate differs from source occurrences")
        observed_fold_positions.extend(range(start, end))
        folds_by_cross_index[cross_index].append(dict(record))
    if sorted(observed_fold_positions) != list(range(readable_count)):
        raise ValueError("HTR fold-local aggregates omit or duplicate reverse-index rows")

    cross_by_start = sorted(
        cross_records,
        key=lambda row: int(row["reverse_index_start"]),
    )
    observed_cross_positions: list[int] = []
    expected_model_aggregates: dict[str, dict[str, Any]] = {}
    for expected_cross_index, aggregate in enumerate(cross_by_start):
        start = _require_nonnegative_int(
            aggregate.get("reverse_index_start"),
            label="HTR aggregate reverse-index start",
        )
        count = _require_positive_int(
            aggregate.get("reverse_index_count"),
            label="HTR aggregate reverse-index count",
        )
        end = start + count
        if (
            end > readable_count
            or not np.all(
                loaded["cross_aggregate_index"][start:end]
                == expected_cross_index
            )
        ):
            raise ValueError("HTR aggregate reverse-index slice changed")
        stage, objective, normalized_focus, wordpiece_kind = source_key(start)
        if any(source_key(position) != source_key(start) for position in range(start, end)):
            raise ValueError("HTR cross-fold aggregate mixes normalized spans")
        notes = loaded["fit_row_id"][start:end]
        chunks = loaded["chunk_index"][start:end]
        variant_refs = sorted(
            {
                int(value)
                for value in loaded["display_variant_index"][start:end].tolist()
            }
        )
        fold_summaries = sorted(
            folds_by_cross_index[expected_cross_index],
            key=lambda row: int(row["fold"]),
        )
        identity = {
            "schema_version": HTR_STAGE2_CROSS_FOLD_AGGREGATE_SCHEMA,
            "scope_binding_sha256": manifest["scope_binding_sha256"],
            "stage": stage,
            "objective": objective,
            "normalized_focus_text": normalized_focus,
            "wordpiece_kind": wordpiece_kind,
            "cross_fold_consolidation": (
                HTR_STAGE2_CROSS_FOLD_CONSOLIDATION_SCHEMA
            ),
        }
        expected = {
            **identity,
            "aggregate_id": (
                f"htr_cross_fold_aggregate_{_sha256_json(identity)}"
            ),
            "occurrence_count": count,
            "unique_note_count": int(np.unique(notes).size),
            "unique_chunk_count": _unique_pair_count(notes, chunks),
            "attention_summaries": _score_summaries(
                note_ids=notes,
                token_attention=loaded["token_attention"][start:end],
                chunk_attention=loaded["chunk_attention"][start:end],
                hierarchical_attention=loaded[
                    "hierarchical_attention_score"
                ][start:end],
            ),
            "fold_local_summaries": fold_summaries,
            "folds_retained_before_consolidation": [
                int(row["fold"]) for row in fold_summaries
            ],
            "display_text_variant_count": len(variant_refs),
            "display_text_variant_indices_sha256": _sha256_json(variant_refs),
            "context_windows": _context_examples(
                indices=np.arange(start, end, dtype=np.int64),
                hierarchical=loaded["hierarchical_attention_score"],
                token=loaded["token_attention"],
                chunk_attention=loaded["chunk_attention"],
                note=loaded["fit_row_id"],
                chunk=loaded["chunk_index"],
                token_position=loaded["focus_token_position"],
                source_index=loaded["source_record_index"],
                span_index=loaded["readable_span_index"],
                variant_index=loaded["display_variant_index"],
                variants=variants,
                limit=_require_positive_int(
                    summary.get(
                        "context_windows_per_aggregate"
                    ),
                    label="HTR aggregate context-window bound",
                ),
            ),
            "context_window_policy": HTR_STAGE2_CONTEXT_POLICY_SCHEMA,
            "reverse_index_start": start,
            "reverse_index_count": count,
            "overlap_accounting": {
                "raw_overlapping_chunk_occurrence_count": count,
                "unique_supporting_note_count": int(np.unique(notes).size),
                "unique_supporting_chunk_count": _unique_pair_count(notes, chunks),
                "note_level_maxima_separate_from_occurrence_summaries": True,
            },
            "hierarchical_attention_interpretation": (
                "ranking_heuristic_not_causal_attribution"
            ),
        }
        expected["content_sha256"] = _sha256_json(expected)
        if aggregate != expected:
            raise ValueError("HTR semantic aggregate differs from source occurrences")
        model_projection = _model_facing_aggregate(expected)
        expected_model_aggregates[
            str(model_projection["aggregate_id"])
        ] = model_projection
        observed_cross_positions.extend(range(start, end))
    if observed_cross_positions != list(range(readable_count)):
        raise ValueError("HTR cross-fold aggregates omit or duplicate occurrences")
    delivered_model_aggregates = {
        str(aggregate["aggregate_id"]): aggregate
        for aggregate in delivered_aggregates
    }
    if (
        len(delivered_model_aggregates) != len(delivered_aggregates)
        or delivered_model_aggregates != expected_model_aggregates
    ):
        raise ValueError(
            "HTR model-facing projection omits, duplicates, or alters an "
            "authenticated semantic aggregate"
        )

    if (
        len(delivered_ids) != len(set(delivered_ids))
        or len(delivered_ids) != summary.get("cross_fold_aggregate_count")
        or len(cross_records) != summary.get("cross_fold_aggregate_count")
        or summary.get("cross_fold_aggregate_reference")
        != {
            "relative_path": cross_registration["relative_path"],
            "content_sha256": cross_registration["content_sha256"],
            "cross_fold_aggregate_count": len(cross_records),
            "complete_semantic_aggregate_inventory": True,
        }
        or sum(
            int(
                aggregate["occurrence_count"]
            )
            for item in batches
            for aggregate in item["content"]["aggregate_batch"]["aggregates"]
        )
        != readable_count
        or summary.get("model_facing_batch_count") != len(batches)
        or summary.get("model_facing_bytes") != sum(batch_sizes)
        or summary.get("model_facing_batch_sizes_bytes") != batch_sizes
        or summary.get("maximum_model_facing_batch_bytes") != max(batch_sizes)
        or isinstance(
            summary.get("configured_aggregates_per_batch_bound"),
            bool,
        )
        or not isinstance(
            summary.get("configured_aggregates_per_batch_bound"),
            int,
        )
        or int(summary["configured_aggregates_per_batch_bound"]) < 1
        or summary.get("maximum_aggregates_in_one_batch")
        != max(
            int(item["content"]["aggregate_batch"]["aggregate_count"])
            for item in batches
        )
        or int(summary["maximum_aggregates_in_one_batch"])
        > int(summary["configured_aggregates_per_batch_bound"])
        or not math.isclose(
            float(summary.get("median_model_facing_batch_bytes")),
            float(statistics.median(batch_sizes)),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or summary.get("source_chunk_interpretation_count")
        != package["chunk_interpretation_count"]
        or summary.get("source_stage_objective_fold_counts")
        != sorted(
            [
                {
                    "stage": str(batch["stage"]),
                    "objective": str(batch["objective"]),
                    "fold": int(batch["fold"]),
                    "token_occurrence_count": int(
                        batch["token_occurrence_count"]
                    ),
                    "chunk_interpretation_count": int(
                        batch["chunk_count"]
                    ),
                    "note_interpretation_count": int(
                        batch["note_count"]
                    ),
                    "special_token_occurrence_count": int(
                        batch["special_token_occurrence_count"]
                    ),
                    "special_token_attention_mass": float(
                        batch["special_token_attention_mass"]
                    ),
                    "padding_occurrence_count": int(
                        batch["padding_occurrence_count"]
                    ),
                }
                for batch in package["fold_batches"]
            ],
            key=canonical_json,
        )
        or summary.get("special_token_accounting_bucket", {}).get(
            "occurrence_count"
        )
        != package["special_token_occurrence_count"]
    ):
        raise ValueError("HTR aggregate delivery/accounting summary changed")
    return HtrSemanticAggregationResult(
        payload=payload,
        scope_manifest=manifest,
        scope_manifest_path=manifest_path,
    )


def summarize_htr_call_plan(
    scope_manifests: Iterable[Mapping[str, Any]],
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
        ):
            raise ValueError("HTR scope batch-size inventory is invalid")
        sizes.extend(scope_sizes)
    old_calls = sum(
        int(summary["one_atom_per_source_chunk_design_call_count"])
        for summary in summaries
    )
    new_calls = sum(
        int(summary["planned_htr_interpretation_call_count"])
        for summary in summaries
    )
    if new_calls < 1 or old_calls < 1:
        raise ValueError("HTR call-plan accounting is invalid")
    return {
        "schema_version": "production_htr_stage2_call_plan_preflight_v1",
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
                summary["special_token_accounting_bucket"][
                    "attention_mass"
                ]
            )
            for summary in summaries
        ),
        "readable_span_occurrence_count": sum(
            int(summary["eligible_readable_span_occurrence_count"])
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
                "counts": summary[
                    "source_stage_objective_fold_counts"
                ],
            }
            for row, summary in zip(rows, summaries, strict=True)
        ],
        "total_model_facing_bytes": sum(
            int(summary["model_facing_bytes"])
            for summary in summaries
        ),
        "planned_htr_interpretation_call_count": new_calls,
        "one_atom_per_chunk_baseline_call_count": old_calls,
        "call_reduction_fraction": 1.0 - (new_calls / old_calls),
        "maximum_prompt_evidence_bytes": max(
            int(summary["maximum_model_facing_batch_bytes"])
            for summary in summaries
        ),
        "median_prompt_evidence_bytes": statistics.median(sizes),
        "call_plan_on_order_of_hundreds_of_thousands": new_calls >= 100_000,
        "stage2_endpoint_launch_allowed": new_calls < 100_000,
        "raw_arrays_copied_to_model_facing_catalog": False,
        "no_top_k_sampling_or_truncation": True,
    }


__all__ = [
    "DEFAULT_CONTEXT_WINDOWS_PER_AGGREGATE",
    "DEFAULT_MODEL_FACING_AGGREGATES_PER_BATCH",
    "DEFAULT_MODEL_FACING_BATCH_BYTES",
    "DEFAULT_MODEL_FACING_TOKEN_UPPER_BOUND",
    "HTR_STAGE2_AGGREGATE_BATCH_SCHEMA",
    "HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA",
    "HTR_STAGE2_BATCHING_SCHEMA",
    "HTR_STAGE2_CONTEXT_POLICY_SCHEMA",
    "HTR_STAGE2_CROSS_FOLD_AGGREGATE_SCHEMA",
    "HTR_STAGE2_CROSS_FOLD_CONSOLIDATION_SCHEMA",
    "HTR_STAGE2_FOLD_AGGREGATE_SCHEMA",
    "HTR_STAGE2_MODEL_AGGREGATE_SCHEMA",
    "HTR_STAGE2_NORMALIZATION_SCHEMA",
    "HTR_STAGE2_REVERSE_INDEX_SCHEMA",
    "HTR_STAGE2_SCOPE_MANIFEST_SCHEMA",
    "HTR_STAGE2_STORE_MANIFEST_SCHEMA",
    "HtrSemanticAggregationResult",
    "build_htr_semantic_aggregation_scope",
    "normalize_htr_readable_focus",
    "summarize_htr_call_plan",
    "validate_htr_semantic_aggregation_scope",
]
