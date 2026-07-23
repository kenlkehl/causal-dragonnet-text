"""Oracle-blind, auditable text preparation for production causal workflows."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

PREPARATION_SCHEMA = "production_modeling_text_preparation_v1"
MISSING_NOTE_MARKER = "[CLINICAL NOTE UNAVAILABLE]"
NEUTRAL_RUN_MARKER = "[REDACTED PATHOLOGICAL UNICODE RUN]"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return _sha_bytes(value.encode("utf-8"))


def _json_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=lambda item: item.item() if hasattr(item, "item") else str(item),
    )
    return _sha_text(payload)


def stable_file_sha256(path: Path) -> tuple[str, int]:
    """Hash a regular non-symlink file and fail if it changes during the read."""
    path = Path(path).resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    if identity(before) != identity(after) or size != after.st_size:
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest(), size


@dataclass(frozen=True)
class TextPreparationOptions:
    dataset_path: Path
    output_dir: Path
    unit_id_column: str
    text_column: str
    treatment_column: str
    outcome_column: str
    outcome_type: str = "binary"
    repeated_character_threshold: int = 1000
    empty_text_policy: str = "marker"
    repeated_character_policy: str = "marker"


def _qualifying_category(character: str) -> bool:
    return unicodedata.category(character)[:1] in {"P", "Z", "C"}


def prepare_text_value(text: Any, *, threshold: int) -> tuple[str, list[dict[str, Any]]]:
    original = "" if text is None or pd.isna(text) else str(text)
    if not original.strip():
        return MISSING_NOTE_MARKER, [{"kind": "empty_text", "start": 0, "count": len(original)}]
    output: list[str] = []
    audit: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(original):
        end = cursor + 1
        while end < len(original) and original[end] == original[cursor]:
            end += 1
        count = end - cursor
        character = original[cursor]
        if count >= threshold and _qualifying_category(character):
            output.append(NEUTRAL_RUN_MARKER)
            audit.append(
                {
                    "kind": "unicode_run",
                    "start": cursor,
                    "end": end,
                    "count": count,
                    "code_point": f"U+{ord(character):04X}",
                    "unicode_category": unicodedata.category(character),
                    "run_sha256": _sha_text(original[cursor:end]),
                }
            )
        else:
            output.append(original[cursor:end])
        cursor = end
    return "".join(output), audit


def _require_binary(series: pd.Series, label: str) -> None:
    if series.isna().any() or set(series.unique().tolist()) != {0, 1}:
        raise ValueError(f"{label} must be complete and contain exactly binary values 0 and 1")


def prepare_modeling_cohort(options: TextPreparationOptions) -> Mapping[str, Any]:
    """Project four configured columns, prepare text, and seal an audit manifest."""
    if options.outcome_type != "binary":
        raise ValueError("this production version supports only binary outcomes")
    if options.repeated_character_threshold < 1:
        raise ValueError("repeated_character_threshold must be positive")
    if options.empty_text_policy != "marker" or options.repeated_character_policy != "marker":
        raise ValueError("only the fail-closed marker policies are supported")
    columns = [
        options.unit_id_column,
        options.text_column,
        options.treatment_column,
        options.outcome_column,
    ]
    if len(set(columns)) != 4 or any(not str(value).strip() for value in columns):
        raise ValueError("the four configured modeling columns must be distinct and non-empty")
    source_sha, source_size = stable_file_sha256(options.dataset_path)
    frame = pd.read_parquet(options.dataset_path, columns=columns)
    ids = frame[options.unit_id_column]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("unit IDs must be complete and unique")
    _require_binary(frame[options.treatment_column], "treatment")
    _require_binary(frame[options.outcome_column], "outcome")
    before_nontext = frame.drop(columns=[options.text_column]).copy(deep=True)
    row_audits: list[dict[str, Any]] = []
    prepared_texts: list[str] = []
    for position, raw in enumerate(frame[options.text_column].tolist()):
        original = "" if raw is None or pd.isna(raw) else str(raw)
        prepared, transformations = prepare_text_value(
            raw, threshold=options.repeated_character_threshold
        )
        prepared_texts.append(prepared)
        row_audits.append(
            {
                "row_position": position,
                "unit_id": (
                    ids.iloc[position].item()
                    if hasattr(ids.iloc[position], "item")
                    else ids.iloc[position]
                ),
                "before_text_sha256": _sha_text(original),
                "after_text_sha256": _sha_text(prepared),
                "before_length": len(original),
                "after_length": len(prepared),
                "transformations": transformations,
            }
        )
    frame[options.text_column] = prepared_texts
    if not frame.drop(columns=[options.text_column]).equals(before_nontext):
        raise RuntimeError("non-text values or row order changed during preparation")
    root = Path(options.output_dir)
    if root.exists():
        raise ValueError("preparation output directory must be fresh")
    root.mkdir(parents=True)
    output_path = root / "modeling_cohort.parquet"
    frame.to_parquet(output_path, index=False)
    output_sha, output_size = stable_file_sha256(output_path)
    affected = [row["unit_id"] for row in row_audits if row["transformations"]]
    body = {
        "schema_version": PREPARATION_SCHEMA,
        "policy": {
            "identity": "neutral_marker_unicode_run_v1",
            "empty_text_policy": options.empty_text_policy,
            "repeated_character_policy": options.repeated_character_policy,
            "repeated_character_threshold": options.repeated_character_threshold,
            "missing_note_marker_sha256": _sha_text(MISSING_NOTE_MARKER),
            "run_marker_sha256": _sha_text(NEUTRAL_RUN_MARKER),
            "transformations_determined_from_text_only": True,
        },
        "columns": dict(zip(("unit_id", "text", "treatment", "outcome"), columns)),
        "row_count": len(frame),
        "affected_unit_ids": affected,
        "rows": row_audits,
        "source": {"path": str(Path(options.dataset_path).resolve()), "sha256": source_sha, "size_bytes": source_size},
        "output": {"path": str(output_path.resolve()), "sha256": output_sha, "size_bytes": output_size},
        "non_text_values_unchanged": True,
        "row_order_unchanged": True,
        "oracle_columns_decoded_or_materialized": False,
    }
    manifest = {**body, "content_sha256": _json_sha(body)}
    (root / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def verify_tokenizer_character_coverage(
    texts: Sequence[str],
    planners: Mapping[str, Callable[[str], Sequence[tuple[int, int]]]],
) -> Mapping[str, Any]:
    """Verify tokenizer-aware planners cover every character exactly once."""
    results: dict[str, Any] = {}
    for name, planner in planners.items():
        page_count = 0
        for row, text in enumerate(texts):
            spans = list(planner(text))
            expected = 0
            for start, end in spans:
                if start != expected or end <= start or end > len(text):
                    raise ValueError(f"{name} tokenizer coverage failed at row {row}")
                expected = end
            if expected != len(text) or (text and not spans):
                raise ValueError(f"{name} tokenizer omitted characters at row {row}")
            page_count += len(spans)
        results[name] = {"row_count": len(texts), "page_count": page_count, "complete": True}
    return results


__all__ = [
    "MISSING_NOTE_MARKER", "NEUTRAL_RUN_MARKER", "PREPARATION_SCHEMA",
    "TextPreparationOptions", "prepare_modeling_cohort", "prepare_text_value",
    "stable_file_sha256", "verify_tokenizer_character_coverage",
]
