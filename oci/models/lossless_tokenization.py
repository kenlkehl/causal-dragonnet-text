"""Fail-closed helpers for configuration-bounded model tokenization.

``max_length`` and ``max_chunks`` are resource/capacity settings.  They must
never silently select a prefix or suffix of a clinical note.  Callers in this
module therefore tokenize with ``truncation=False`` and reject a configured
capacity when it would bind.  The caller can then raise the relevant
configuration value or choose an explicitly lossless windowing architecture.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class SemanticTruncationError(ValueError):
    """A configured text capacity would discard part of an input."""


def _as_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _batch_size(texts: Any) -> int:
    if isinstance(texts, str):
        return 1
    try:
        return len(texts)
    except TypeError as exc:
        raise TypeError("tokenizer input must be one string or a sized batch") from exc


def _row_lengths(encoded: Mapping[str, Any], expected_rows: int) -> tuple[int, ...]:
    attention_mask = _as_python(encoded.get("attention_mask"))
    if attention_mask is not None:
        if expected_rows == 1 and (
            not isinstance(attention_mask, Sequence)
            or isinstance(attention_mask, (str, bytes))
            or not attention_mask
            or not isinstance(attention_mask[0], Sequence)
        ):
            rows = [attention_mask]
        else:
            rows = attention_mask
        try:
            lengths = tuple(sum(int(value) for value in row) for row in rows)
        except (TypeError, ValueError) as exc:
            raise ValueError("tokenizer returned an invalid attention_mask") from exc
        if len(lengths) == expected_rows:
            return lengths

    input_ids = _as_python(encoded.get("input_ids"))
    if input_ids is None:
        raise ValueError("tokenizer response omitted input_ids")
    if expected_rows == 1 and (
        not isinstance(input_ids, Sequence)
        or isinstance(input_ids, (str, bytes))
        or not input_ids
        or not isinstance(input_ids[0], Sequence)
    ):
        rows = [input_ids]
    else:
        rows = input_ids
    try:
        lengths = tuple(len(row) for row in rows)
    except TypeError as exc:
        raise ValueError("tokenizer returned invalid input_ids") from exc
    if len(lengths) != expected_rows:
        raise ValueError(
            "tokenizer response row count changed "
            f"({len(lengths)} != {expected_rows})"
        )
    return lengths


def tokenize_losslessly(
    tokenizer: Any,
    texts: Any,
    *,
    configured_max_length: int | None,
    context: str,
    padding: bool | str = False,
    **tokenizer_kwargs: Any,
) -> Any:
    """Tokenize complete inputs and fail if ``configured_max_length`` binds.

    ``padding="max_length"`` is supported after the same value has been
    supplied as ``configured_max_length``.  It pads shorter rows but still
    passes ``truncation=False`` and validates the resulting attention masks.
    """

    if "truncation" in tokenizer_kwargs or "max_length" in tokenizer_kwargs:
        raise TypeError(
            "tokenize_losslessly owns truncation and max_length arguments"
        )
    maximum: int | None
    if configured_max_length is None:
        maximum = None
    else:
        if (
            isinstance(configured_max_length, bool)
            or not isinstance(configured_max_length, int)
            or configured_max_length < 1
        ):
            raise ValueError("configured_max_length must be a positive integer")
        maximum = int(configured_max_length)

    call_kwargs = dict(tokenizer_kwargs)
    call_kwargs["padding"] = padding
    call_kwargs["truncation"] = False
    if padding == "max_length":
        if maximum is None:
            raise ValueError(
                "padding='max_length' requires configured_max_length"
            )
        call_kwargs["max_length"] = maximum

    encoded = tokenizer(texts, **call_kwargs)
    if not isinstance(encoded, Mapping):
        raise ValueError("tokenizer response must be a mapping")
    lengths = _row_lengths(encoded, _batch_size(texts))
    if maximum is not None:
        offenders = [
            (row_index, length)
            for row_index, length in enumerate(lengths)
            if length > maximum
        ]
        if offenders:
            preview = ", ".join(
                f"row {row_index}: {length} tokens"
                for row_index, length in offenders[:8]
            )
            suffix = (
                ""
                if len(offenders) <= 8
                else f", plus {len(offenders) - 8} more rows"
            )
            raise SemanticTruncationError(
                f"{context} exceeds configured_max_length={maximum}; "
                f"semantic truncation is forbidden ({preview}{suffix}). "
                "Increase the configured capacity or use a lossless windowing "
                "architecture."
            )
    return encoded


def required_overlapping_chunks(
    token_count: int,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """Return the exact chunks needed to cover a token sequence."""

    values = (token_count, chunk_size, chunk_overlap)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("token and chunk sizes must be integers")
    if token_count < 0:
        raise ValueError("token_count must be nonnegative")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
    if token_count <= chunk_size:
        return 1
    stride = chunk_size - chunk_overlap
    return 1 + (token_count - chunk_size + stride - 1) // stride


def require_nonbinding_chunk_capacity(
    token_count: int,
    *,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
    context: str,
) -> int:
    """Fail if a configured chunk cap cannot cover all supplied tokens."""

    if isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks < 1:
        raise ValueError("max_chunks must be a positive integer")
    required = required_overlapping_chunks(
        token_count,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if required > max_chunks:
        raise SemanticTruncationError(
            f"{context} requires {required} chunks but configured max_chunks="
            f"{max_chunks}; semantic truncation is forbidden. Increase the "
            "configured capacity."
        )
    return required
