#!/usr/bin/env python3
"""Sample and pre-embed synthetic notes into an external retrieval cache."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBMED_SCRIPT_DIR = REPO_ROOT / "scripts" / "pubmed_embeddings"
if str(PUBMED_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PUBMED_SCRIPT_DIR))

from embed_pubmed_corpus import EmbedConfig, build_pubmed_embedding_cache  # noqa: E402

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample synthetic notes from a parquet file and pre-embed them into "
            "an external chunk-cache for embedding contrast retrieval."
        )
    )
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-name", required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--text-column",
        required=True,
        help="Exact text column to sample; implicit column auto-detection is forbidden.",
    )
    parser.add_argument("--source-id-column", required=True)
    parser.add_argument(
        "--metadata-column",
        action="append",
        default=[],
        help="Optional metadata column copied into row_metadata.jsonl. May be repeated.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--device-ids",
        nargs="*",
        default=None,
        help="GPU ids to use, e.g. --device-ids 0 1 or --device-ids cpu.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rows-per-part", type=int, default=2500)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--chunk-size-words", type=int, required=True)
    parser.add_argument("--chunk-overlap-words", type=int, required=True)
    parser.add_argument("--max-chunks", type=int, required=True)
    parser.add_argument("--chunk-selection", choices=["first", "last"], required=True)
    normalization = parser.add_mutually_exclusive_group(required=True)
    normalization.add_argument(
        "--normalize-embeddings",
        dest="normalize_embeddings",
        action="store_true",
    )
    normalization.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
    )
    parser.add_argument(
        "--force-sample",
        action="store_true",
        help="Regenerate the sampled JSONL even if it already exists.",
    )
    parser.add_argument(
        "--force-embed",
        action="store_true",
        help="Rebuild embedding parts/cache instead of resuming or reusing.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write the sampled JSONL; do not embed.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_path = Path(args.input_parquet).expanduser()
    output_root = Path(args.output_root).expanduser()
    if (
        not isinstance(args.sample_name, str)
        or not args.sample_name.strip()
        or Path(args.sample_name).name != args.sample_name
    ):
        raise ValueError("--sample-name must be one non-empty path-free name")
    output_root.mkdir(parents=True, exist_ok=True)
    sample_path = output_root / f"{args.sample_name}.jsonl"
    sample_metadata_path = output_root / f"{args.sample_name}.metadata.json"
    metadata_columns = list(args.metadata_column)

    text_column = resolve_text_column(input_path, args.text_column)
    logger.info("Using synthetic text column: %s", text_column)
    if args.force_sample or not sample_path.exists():
        write_sample_jsonl(
            input_path=input_path,
            output_path=sample_path,
            metadata_path=sample_metadata_path,
            text_column=text_column,
            source_id_column=args.source_id_column,
            metadata_columns=metadata_columns,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    else:
        _assert_reusable_sample_configuration(
            metadata_path=sample_metadata_path,
            input_path=input_path,
            text_column=text_column,
            source_id_column=args.source_id_column,
            metadata_columns=metadata_columns,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        logger.info("Reusing existing sample JSONL: %s", sample_path)

    if args.prepare_only:
        print(sample_path)
        return

    cache_path = output_root / f"{args.sample_name}_embedding_cache"
    config = EmbedConfig(
        input_path=sample_path,
        output_cache_dir=cache_path,
        model_name=args.model_name,
        corpus_name=args.sample_name,
        text_column="text",
        source_id_column=args.source_id_column,
        metadata_columns=metadata_columns,
        batch_size=args.batch_size,
        rows_per_part=args.rows_per_part,
        max_seq_length=args.max_seq_length,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
        max_chunks=args.max_chunks,
        chunk_selection=args.chunk_selection,
        normalize_embeddings=args.normalize_embeddings,
        limit=None,
        force=args.force_embed,
    )
    final_cache = build_pubmed_embedding_cache(config, raw_device_ids=args.device_ids)
    print(final_cache)


def resolve_text_column(
    input_path: Path,
    requested: str,
) -> str:
    schema_names = parquet_column_names(input_path)
    if not requested or requested == "auto":
        raise ValueError("--text-column must name one exact column; 'auto' is forbidden")
    if requested not in schema_names:
        raise ValueError(
            f"Requested text column {requested!r} not found. " f"Available columns: {schema_names}"
        )
    return requested


def parquet_column_names(input_path: Path) -> List[str]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(input_path)
    return list(parquet_file.schema_arrow.names)


def write_sample_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    text_column: str,
    source_id_column: str,
    metadata_columns: Sequence[str],
    sample_size: int,
    seed: int,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if sample_size < 1:
        raise ValueError("--sample-size must be >= 1")
    parquet_file = pq.ParquetFile(input_path)
    schema_names = tuple(parquet_file.schema_arrow.names)
    required_columns = {text_column, source_id_column, *metadata_columns}
    missing_columns = sorted(required_columns - set(schema_names))
    if missing_columns:
        raise ValueError(
            f"configured sample columns are absent from the input parquet: {missing_columns}"
        )
    total_rows = int(parquet_file.metadata.num_rows)
    target_size = min(int(sample_size), total_rows)
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(total_rows, size=target_size, replace=False))
    columns = _columns_to_read(
        schema_names,
        text_column=text_column,
        source_id_column=source_id_column,
        metadata_columns=metadata_columns,
    )
    row_group_offsets = _row_group_offsets(parquet_file)
    logger.info(
        "Sampling %d/%d rows from %s into %s",
        target_size,
        total_rows,
        input_path,
        output_path,
    )
    written = 0
    selected_cursor = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for group_index in range(parquet_file.num_row_groups):
            group_start = row_group_offsets[group_index]
            group_end = row_group_offsets[group_index + 1]
            start_cursor = selected_cursor
            while selected_cursor < len(selected) and selected[selected_cursor] < group_end:
                selected_cursor += 1
            if selected_cursor == start_cursor:
                continue
            global_indices = selected[start_cursor:selected_cursor]
            local_indices = (global_indices - group_start).astype(np.int64)
            table = parquet_file.read_row_group(group_index, columns=columns)
            table = table.take(pa.array(local_indices))
            rows = table.to_pylist()
            for global_index, row in zip(global_indices, rows):
                text = str(row.get(text_column) or "").strip()
                if not text:
                    continue
                payload: Dict[str, Any] = {
                    "source_row_index": int(global_index),
                    "text": text,
                    "original_text_column": text_column,
                    "source_parquet": str(input_path),
                }
                for col in columns:
                    if col == text_column:
                        continue
                    payload[col] = _jsonable(row.get(col))
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1
    sample_metadata = {
        "input_parquet": str(input_path),
        "text_column": text_column,
        "source_id_column": source_id_column,
        "metadata_columns": list(metadata_columns),
        "sample_size_requested": int(sample_size),
        "sample_size_written": int(written),
        "total_rows": total_rows,
        "seed": int(seed),
        "output_path": str(output_path),
        "created_at": datetime.now().isoformat(),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(sample_metadata, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d sampled notes to %s", written, output_path)


def _assert_reusable_sample_configuration(
    *,
    metadata_path: Path,
    input_path: Path,
    text_column: str,
    source_id_column: str,
    metadata_columns: Sequence[str],
    sample_size: int,
    seed: int,
) -> None:
    if not metadata_path.is_file():
        raise RuntimeError(
            "existing sample lacks its configuration metadata; use a fresh output "
            "or explicitly rebuild with --force-sample"
        )
    with open(metadata_path, encoding="utf-8") as handle:
        observed = json.load(handle)
    expected = {
        "input_parquet": str(input_path),
        "text_column": text_column,
        "source_id_column": source_id_column,
        "metadata_columns": list(metadata_columns),
        "sample_size_requested": int(sample_size),
        "seed": int(seed),
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if not isinstance(observed, dict) or observed.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "existing synthetic-note sample is scientifically incompatible with "
            f"this request; mismatched fields: {mismatches}. Use a fresh output "
            "or --force-sample."
        )


def _columns_to_read(
    schema_names: Sequence[str],
    *,
    text_column: str,
    source_id_column: str,
    metadata_columns: Sequence[str],
) -> List[str]:
    available = set(schema_names)
    columns = [text_column]
    for col in [source_id_column, *metadata_columns]:
        if col and col in available and col not in columns:
            columns.append(col)
    return columns


def _row_group_offsets(parquet_file: Any) -> List[int]:
    offsets = [0]
    running = 0
    for group_index in range(parquet_file.num_row_groups):
        running += int(parquet_file.metadata.row_group(group_index).num_rows)
        offsets.append(running)
    return offsets


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "as_py"):
        return _jsonable(value.as_py())
    return value


if __name__ == "__main__":
    main()
