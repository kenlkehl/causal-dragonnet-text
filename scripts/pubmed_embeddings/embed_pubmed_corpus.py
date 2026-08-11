#!/usr/bin/env python3
"""Embed a PubMed JSONL corpus into an external retrieval cache.

The final output directory is directly consumable by
``EmbeddingContrastDiscoveryConfig.external_corpus_cache_dirs`` in the research
Stage 1 workflow.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import queue
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oci.models.concept_embedding_cache import (  # noqa: E402
    _coerce_embedding_matrix,
    _effective_max_seq_length,
    _get_sentence_transformer_tokenizer,
    load_sentence_transformer,
)
from oci.models.concept_embedding_utils import (  # noqa: E402
    chunk_text_words,
    split_text_to_token_chunks,
)
from oci.models.lossless_tokenization import SemanticTruncationError  # noqa: E402

LOSSLESS_EXTERNAL_EMBEDDING_POLICY_VERSION = "lossless_external_embedding_chunks_v1"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbedConfig:
    input_path: Path
    output_cache_dir: Path
    model_name: str
    corpus_name: str
    text_column: str
    source_id_column: Optional[str]
    metadata_columns: List[str]
    batch_size: int
    rows_per_part: int
    max_seq_length: Optional[int]
    chunk_size_words: int
    chunk_overlap_words: int
    max_chunks: int
    chunk_selection: str
    normalize_embeddings: bool
    limit: Optional[int]
    force: bool


@dataclass(frozen=True)
class PartSpec:
    index: int
    row_start: int
    row_end: int

    @property
    def name(self) -> str:
        return f"part_{self.index:05d}_{self.row_start:06d}_{self.row_end:06d}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed a PubMed JSONL corpus into a resumable chunk vector cache."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-cache-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--corpus-name", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--source-id-column", required=True)
    parser.add_argument(
        "--metadata-column",
        action="append",
        default=[],
        help=(
            "Metadata column to copy into row_metadata.jsonl. May be repeated. "
            "Defaults to PubMed-oriented metadata columns."
        ),
    )
    parser.add_argument(
        "--device-ids",
        nargs="*",
        default=None,
        help=(
            "GPU ids to use, e.g. --device-ids 0 1 or --device-ids 0,1. "
            "Omit to use all visible GPUs, or pass cpu for CPU."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rows-per-part", type=int, default=2500)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--chunk-size-words", type=int, required=True)
    parser.add_argument("--chunk-overlap-words", type=int, required=True)
    parser.add_argument("--max-chunks", type=int, required=True)
    parser.add_argument(
        "--chunk-selection",
        choices=["first", "last"],
        required=True,
        help=(
            "Cache-identity compatibility label. The cap is abort-only: no first/last "
            "selection is performed if max-chunks would bind."
        ),
    )
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def embed_config_from_args(args: argparse.Namespace) -> EmbedConfig:
    return EmbedConfig(
        input_path=Path(args.input).expanduser(),
        output_cache_dir=Path(args.output_cache_dir).expanduser(),
        model_name=args.model_name,
        corpus_name=args.corpus_name,
        text_column=args.text_column,
        source_id_column=args.source_id_column,
        metadata_columns=args.metadata_column or _default_pubmed_metadata_columns(),
        batch_size=args.batch_size,
        rows_per_part=args.rows_per_part,
        max_seq_length=args.max_seq_length,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
        max_chunks=args.max_chunks,
        chunk_selection=args.chunk_selection,
        normalize_embeddings=args.normalize_embeddings,
        limit=args.limit,
        force=args.force,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    config = embed_config_from_args(args)
    cache_path = build_pubmed_embedding_cache(config, raw_device_ids=args.device_ids)
    print(cache_path)


def build_pubmed_embedding_cache(
    config: EmbedConfig,
    *,
    raw_device_ids: Optional[Sequence[str]],
) -> Path:
    _validate_config(config)
    rows = _read_jsonl(config.input_path, limit=config.limit)
    if not rows:
        raise RuntimeError(f"No rows found in {config.input_path}")
    config.output_cache_dir.mkdir(parents=True, exist_ok=True)
    if not config.force:
        _assert_existing_configuration_is_reusable(config)
    if _final_cache_ready(config.output_cache_dir, expected_rows=len(rows)) and not config.force:
        logger.info("Reusing complete external embedding cache: %s", config.output_cache_dir)
        return config.output_cache_dir

    devices = _resolve_devices(raw_device_ids)
    logger.info(
        "Embedding %d PubMed records with %d device worker(s): %s",
        len(rows),
        len(devices),
        ", ".join(str(device) for device in devices),
    )
    parts = _part_specs(len(rows), config.rows_per_part)
    parts_dir = config.output_cache_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config.output_cache_dir / "build_config.json",
        {
            **_config_metadata(config),
            "input_path": str(config.input_path),
            "num_rows": len(rows),
            "num_parts": len(parts),
            "devices": [str(device) for device in devices],
            "updated_at": datetime.now().isoformat(),
        },
    )

    work_queue: "queue.Queue[PartSpec]" = queue.Queue()
    skipped = 0
    for part in parts:
        if _part_complete(parts_dir / part.name) and not config.force:
            skipped += 1
            continue
        work_queue.put(part)
    if skipped:
        logger.info("Skipping %d already completed part(s)", skipped)

    errors: List[BaseException] = []
    error_lock = threading.Lock()

    def _worker(device: Any) -> None:
        try:
            _process_parts_on_device(
                device=device,
                work_queue=work_queue,
                rows=rows,
                parts_dir=parts_dir,
                config=config,
            )
        except BaseException as exc:  # pragma: no cover - surfaced to caller
            with error_lock:
                errors.append(exc)
            raise

    if not work_queue.empty():
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(_worker, device) for device in devices]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    if errors:
        raise RuntimeError("At least one embedding worker failed") from errors[0]

    missing = [part.name for part in parts if not _part_complete(parts_dir / part.name)]
    if missing:
        raise RuntimeError(f"Embedding did not complete all parts; missing: {missing[:5]}")

    _merge_parts(config=config, rows=rows, parts=parts, parts_dir=parts_dir)
    return config.output_cache_dir


def _process_parts_on_device(
    *,
    device: Any,
    work_queue: "queue.Queue[PartSpec]",
    rows: List[Dict[str, Any]],
    parts_dir: Path,
    config: EmbedConfig,
) -> None:
    encoder = load_sentence_transformer(
        config.model_name,
        device=device,
        max_seq_length=config.max_seq_length,
    )
    tokenizer = _get_sentence_transformer_tokenizer(encoder)
    effective_max_seq_length = _effective_max_seq_length(
        encoder,
        config.max_seq_length,
    )
    while True:
        try:
            part = work_queue.get_nowait()
        except queue.Empty:
            return
        try:
            _encode_part(
                part=part,
                part_dir=parts_dir / part.name,
                rows=rows[part.row_start : part.row_end],
                global_row_start=part.row_start,
                encoder=encoder,
                tokenizer=tokenizer,
                effective_max_seq_length=effective_max_seq_length,
                device_name=str(device),
                config=config,
            )
        finally:
            work_queue.task_done()


def _encode_part(
    *,
    part: PartSpec,
    part_dir: Path,
    rows: List[Dict[str, Any]],
    global_row_start: int,
    encoder: Any,
    tokenizer: Any,
    effective_max_seq_length: Optional[int],
    device_name: str,
    config: EmbedConfig,
) -> None:
    part_dir.mkdir(parents=True, exist_ok=True)
    sample_chunks = _load_or_create_part_chunks(
        part_dir=part_dir,
        rows=rows,
        config=config,
        tokenizer=tokenizer,
        effective_max_seq_length=effective_max_seq_length,
    )
    chunk_counts = [len(chunks) for chunks in sample_chunks]
    total_chunks = int(sum(chunk_counts))
    if total_chunks <= 0:
        raise RuntimeError(f"No chunks generated for {part.name}")
    flat_chunks = [chunk for chunks in sample_chunks for chunk in chunks]
    embedding_dim = _embedding_dim(encoder, flat_chunks, config)
    emb_path = part_dir / "chunk_embeddings.npy"
    progress_path = part_dir / "progress.json"
    progress = _read_json(progress_path)
    encoded_until = 0 if config.force else int(progress.get("encoded_until", 0) or 0)
    encoded_until = min(max(encoded_until, 0), total_chunks)

    if emb_path.exists() and not config.force:
        emb_mmap = np.lib.format.open_memmap(str(emb_path), mode="r+")
        if emb_mmap.shape != (total_chunks, embedding_dim):
            raise RuntimeError(
                f"Existing part embedding shape mismatch for {part.name}: "
                f"{emb_mmap.shape} != {(total_chunks, embedding_dim)}"
            )
    else:
        emb_mmap = np.lib.format.open_memmap(
            str(emb_path),
            mode="w+",
            dtype=np.float16,
            shape=(total_chunks, embedding_dim),
        )
        encoded_until = 0

    _write_part_metadata(
        part_dir,
        part=part,
        rows=rows,
        global_row_start=global_row_start,
        total_chunks=total_chunks,
        embedding_dim=embedding_dim,
        device_name=device_name,
        config=config,
        status="encoding",
    )
    cursor = encoded_until
    logger.info(
        "%s on %s: encoding chunks %d/%d",
        part.name,
        device_name,
        cursor,
        total_chunks,
    )
    while cursor < total_chunks:
        end = min(cursor + config.batch_size, total_chunks)
        batch_chunks = flat_chunks[cursor:end]
        batch_embeddings = encoder.encode(
            batch_chunks,
            batch_size=len(batch_chunks),
            convert_to_numpy=True,
            normalize_embeddings=config.normalize_embeddings,
            show_progress_bar=False,
        )
        batch_embeddings, _ = _coerce_embedding_matrix(
            batch_embeddings,
            expected_rows=len(batch_chunks),
            expected_dim=embedding_dim,
        )
        emb_mmap[cursor:end] = batch_embeddings.astype(np.float16)
        cursor = end
        emb_mmap.flush()
        _write_json(
            progress_path,
            {
                "encoded_until": cursor,
                "total_chunks": total_chunks,
                "updated_at": datetime.now().isoformat(),
            },
        )

    emb_mmap.flush()
    _write_part_row_metadata(part_dir, rows, global_row_start, config)
    _write_part_metadata(
        part_dir,
        part=part,
        rows=rows,
        global_row_start=global_row_start,
        total_chunks=total_chunks,
        embedding_dim=embedding_dim,
        device_name=device_name,
        config=config,
        status="complete",
    )
    logger.info("%s complete on %s", part.name, device_name)


def _load_or_create_part_chunks(
    *,
    part_dir: Path,
    rows: List[Dict[str, Any]],
    config: EmbedConfig,
    tokenizer: Any,
    effective_max_seq_length: Optional[int],
) -> List[List[str]]:
    chunks_path = part_dir / "chunk_texts.jsonl"
    offsets_path = part_dir / "offsets.npy"
    if chunks_path.exists() and offsets_path.exists() and not config.force:
        chunks = _load_chunk_texts(chunks_path, expected_rows=len(rows))
        offsets = np.load(str(offsets_path))
        if len(offsets) == len(rows) + 1 and int(offsets[-1]) == sum(len(item) for item in chunks):
            return chunks

    texts = [str(row.get(config.text_column) or "") for row in rows]
    sample_chunks = [
        chunk_text_words(
            text,
            config.chunk_size_words,
            config.chunk_overlap_words,
            config.max_chunks,
            config.chunk_selection,
        )
        for text in texts
    ]
    if tokenizer is not None and effective_max_seq_length is not None:
        sample_chunks = [
            _token_bound_chunks(
                chunks,
                tokenizer=tokenizer,
                max_seq_length=effective_max_seq_length,
                chunk_overlap_tokens=config.chunk_overlap_words,
                max_chunks=config.max_chunks,
            )
            for chunks in sample_chunks
        ]
    sample_chunks = [chunks or [""] for chunks in sample_chunks]
    offsets = np.zeros(len(sample_chunks) + 1, dtype=np.int64)
    for idx, chunks in enumerate(sample_chunks):
        offsets[idx + 1] = offsets[idx] + len(chunks)
    np.save(str(offsets_path), offsets)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunks in sample_chunks:
            f.write(json.dumps({"chunks": chunks}, ensure_ascii=False) + "\n")
    return sample_chunks


def _token_bound_chunks(
    chunks: List[str],
    *,
    tokenizer: Any,
    max_seq_length: int,
    chunk_overlap_tokens: int,
    max_chunks: int,
) -> List[str]:
    """Token-bound every word chunk without discarding any resulting chunk."""

    split_chunks: List[str] = []
    for chunk in chunks:
        split_chunks.extend(
            split_text_to_token_chunks(
                chunk,
                tokenizer,
                max_seq_length=int(max_seq_length),
                chunk_overlap_tokens=int(chunk_overlap_tokens),
            )
        )
    if len(split_chunks) > max_chunks:
        raise SemanticTruncationError(
            "token-bounded external-corpus text requires "
            f"{len(split_chunks)} chunks but configured max_chunks={max_chunks}; "
            "semantic truncation is forbidden. Increase --max-chunks so the "
            "allocation bound is nonbinding."
        )
    return split_chunks or [""]


def _merge_parts(
    *,
    config: EmbedConfig,
    rows: List[Dict[str, Any]],
    parts: List[PartSpec],
    parts_dir: Path,
) -> None:
    part_meta = [_read_json(parts_dir / part.name / "part_metadata.json") for part in parts]
    embedding_dim = int(part_meta[0]["hidden_size"])
    total_chunks = int(sum(int(meta["total_chunks"]) for meta in part_meta))
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    cursor = 0
    for part in parts:
        local_offsets = np.load(str(parts_dir / part.name / "offsets.npy"))
        for local_idx in range(len(local_offsets) - 1):
            count = int(local_offsets[local_idx + 1] - local_offsets[local_idx])
            cursor += count
            offsets[part.row_start + local_idx + 1] = cursor
    if cursor != total_chunks:
        raise RuntimeError(f"Merged offset total {cursor} != total chunks {total_chunks}")

    logger.info(
        "Merging %d part(s): rows=%d chunks=%d hidden_size=%d",
        len(parts),
        len(rows),
        total_chunks,
        embedding_dim,
    )
    emb_tmp = config.output_cache_dir / "chunk_embeddings.npy.tmp"
    final_embeddings = np.lib.format.open_memmap(
        str(emb_tmp),
        mode="w+",
        dtype=np.float16,
        shape=(total_chunks, embedding_dim),
    )
    cursor = 0
    for part in parts:
        part_embeddings = np.load(
            str(parts_dir / part.name / "chunk_embeddings.npy"),
            mmap_mode="r",
        )
        end = cursor + int(part_embeddings.shape[0])
        final_embeddings[cursor:end] = part_embeddings
        cursor = end
    final_embeddings.flush()

    offsets_tmp = config.output_cache_dir / "offsets.npy.tmp"
    with open(offsets_tmp, "wb") as f:
        np.save(f, offsets)
    _concat_files(
        [parts_dir / part.name / "chunk_texts.jsonl" for part in parts],
        config.output_cache_dir / "chunk_texts.jsonl.tmp",
    )
    _write_global_row_metadata(
        config.output_cache_dir / "row_metadata.jsonl.tmp",
        rows,
        config=config,
    )

    os.replace(emb_tmp, config.output_cache_dir / "chunk_embeddings.npy")
    os.replace(offsets_tmp, config.output_cache_dir / "offsets.npy")
    os.replace(
        config.output_cache_dir / "chunk_texts.jsonl.tmp",
        config.output_cache_dir / "chunk_texts.jsonl",
    )
    os.replace(
        config.output_cache_dir / "row_metadata.jsonl.tmp",
        config.output_cache_dir / "row_metadata.jsonl",
    )
    metadata = {
        **_config_metadata(config),
        "corpus_name": config.corpus_name,
        "external_retrieval_corpus": True,
        "input_path": str(config.input_path),
        "text_column": config.text_column,
        "source_id_column": config.source_id_column,
        "metadata_columns": config.metadata_columns,
        "hidden_size": embedding_dim,
        "num_samples": len(rows),
        "total_chunks": total_chunks,
        "chunk_counts": np.diff(offsets).astype(int).tolist(),
        "actual_max_len": int(np.max(np.diff(offsets))) if len(offsets) > 1 else 0,
        "storage_format": "variable_length_chunks",
        "chunking_mode": "word_chunks_token_bounded",
        "dtype": "float16",
        "num_parts": len(parts),
        "created_at": datetime.now().isoformat(),
    }
    _write_json(config.output_cache_dir / "metadata.json", metadata)
    logger.info("External embedding cache ready: %s", config.output_cache_dir)


def _embedding_dim(encoder: Any, flat_chunks: List[str], config: EmbedConfig) -> int:
    dim = int(getattr(encoder, "get_sentence_embedding_dimension", lambda: 0)() or 0)
    if dim > 0:
        return dim
    probe = encoder.encode(
        [flat_chunks[0]],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize_embeddings,
        show_progress_bar=False,
    )
    _, dim = _coerce_embedding_matrix(probe, expected_rows=1)
    return dim


def _resolve_devices(raw_device_ids: Optional[Sequence[str]]) -> List[Any]:
    import torch

    tokens: List[str] = []
    for value in raw_device_ids or []:
        tokens.extend(part for part in str(value).split(",") if part.strip())
    if tokens and any(token.strip().lower() == "cpu" for token in tokens):
        return [torch.device("cpu")]
    if not tokens:
        if torch.cuda.is_available():
            return [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())]
        return [torch.device("cpu")]
    return [torch.device(f"cuda:{int(token)}") for token in tokens]


def _part_specs(num_rows: int, rows_per_part: int) -> List[PartSpec]:
    parts = []
    index = 0
    for start in range(0, num_rows, rows_per_part):
        end = min(start + rows_per_part, num_rows)
        parts.append(PartSpec(index=index, row_start=start, row_end=end))
        index += 1
    return parts


def _part_complete(part_dir: Path) -> bool:
    metadata = _read_json(part_dir / "part_metadata.json")
    return (
        metadata.get("status") == "complete"
        and (part_dir / "chunk_embeddings.npy").exists()
        and (part_dir / "offsets.npy").exists()
        and (part_dir / "chunk_texts.jsonl").exists()
        and (part_dir / "row_metadata.jsonl").exists()
    )


def _final_cache_ready(output_cache_dir: Path, expected_rows: int) -> bool:
    required = [
        output_cache_dir / "chunk_embeddings.npy",
        output_cache_dir / "offsets.npy",
        output_cache_dir / "chunk_texts.jsonl",
        output_cache_dir / "row_metadata.jsonl",
        output_cache_dir / "metadata.json",
    ]
    if not all(path.exists() for path in required):
        return False
    metadata = _read_json(output_cache_dir / "metadata.json")
    if int(metadata.get("num_samples", -1)) != int(expected_rows):
        return False
    offsets = np.load(str(output_cache_dir / "offsets.npy"))
    embeddings = np.load(str(output_cache_dir / "chunk_embeddings.npy"), mmap_mode="r")
    return len(offsets) == expected_rows + 1 and int(offsets[-1]) == int(embeddings.shape[0])


def _read_jsonl(path: Path, *, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            rows.append(payload)
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def _load_chunk_texts(path: Path, *, expected_rows: int) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append([str(chunk) for chunk in payload.get("chunks", [])])
    if len(rows) != expected_rows:
        raise RuntimeError(f"{path} has {len(rows)} chunk rows; expected {expected_rows}")
    return rows


def _write_part_metadata(
    part_dir: Path,
    *,
    part: PartSpec,
    rows: List[Dict[str, Any]],
    global_row_start: int,
    total_chunks: int,
    embedding_dim: int,
    device_name: str,
    config: EmbedConfig,
    status: str,
) -> None:
    _write_json(
        part_dir / "part_metadata.json",
        {
            **_config_metadata(config),
            "status": status,
            "part_index": part.index,
            "row_start": global_row_start,
            "row_end": global_row_start + len(rows),
            "num_samples": len(rows),
            "total_chunks": int(total_chunks),
            "hidden_size": int(embedding_dim),
            "device": device_name,
            "updated_at": datetime.now().isoformat(),
        },
    )


def _write_part_row_metadata(
    part_dir: Path,
    rows: List[Dict[str, Any]],
    global_row_start: int,
    config: EmbedConfig,
) -> None:
    _write_global_row_metadata(
        part_dir / "row_metadata.jsonl",
        rows,
        global_row_start,
        config=config,
    )


def _write_global_row_metadata(
    path: Path,
    rows: List[Dict[str, Any]],
    global_row_start: int = 0,
    *,
    config: EmbedConfig,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for local_idx, row in enumerate(rows):
            source_id = (
                _jsonable(row.get(config.source_id_column)) if config.source_id_column else None
            )
            f.write(
                json.dumps(
                    {
                        "row_index": int(global_row_start + local_idx),
                        "source_id": source_id,
                        "metadata": {
                            key: _jsonable(row.get(key))
                            for key in config.metadata_columns
                            if key in row
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _concat_files(paths: Iterable[Path], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as out:
        for path in paths:
            with open(path, encoding="utf-8") as src:
                for line in src:
                    out.write(line)


def _config_metadata(config: EmbedConfig) -> Dict[str, Any]:
    return {
        "lossless_embedding_policy_version": LOSSLESS_EXTERNAL_EMBEDDING_POLICY_VERSION,
        "sentence_model_name": config.model_name,
        "corpus_name": config.corpus_name,
        "text_column": config.text_column,
        "chunk_size_words": config.chunk_size_words,
        "chunk_overlap_words": config.chunk_overlap_words,
        "max_chunks": config.max_chunks,
        "chunk_selection": config.chunk_selection,
        "chunk_overflow_policy": "abort",
        "semantic_truncation_allowed": False,
        "normalize_embeddings": config.normalize_embeddings,
        "max_seq_length": config.max_seq_length,
        "precompute_batch_size": config.batch_size,
        "rows_per_part": config.rows_per_part,
        "source_id_column": config.source_id_column,
        "metadata_columns": config.metadata_columns,
        "input_row_limit": config.limit,
    }


def _assert_existing_configuration_is_reusable(config: EmbedConfig) -> None:
    """Reject old, lossy, or scientifically different partial/final caches."""

    expected = _config_metadata(config)
    scientific_keys = (
        "lossless_embedding_policy_version",
        "sentence_model_name",
        "corpus_name",
        "text_column",
        "chunk_size_words",
        "chunk_overlap_words",
        "max_chunks",
        "chunk_selection",
        "chunk_overflow_policy",
        "semantic_truncation_allowed",
        "normalize_embeddings",
        "max_seq_length",
        "source_id_column",
        "metadata_columns",
        "input_row_limit",
    )
    for path in (
        config.output_cache_dir / "build_config.json",
        config.output_cache_dir / "metadata.json",
    ):
        if not path.exists():
            continue
        observed = _read_json(path)
        mismatches = [key for key in scientific_keys if observed.get(key) != expected.get(key)]
        if mismatches:
            raise RuntimeError(
                f"{path} is not reusable under the configured lossless embedding "
                f"policy; mismatched fields: {mismatches}. Use a fresh output "
                "directory or explicitly rebuild with --force."
            )


def _default_pubmed_metadata_columns() -> List[str]:
    return [
        "pmid",
        "doi",
        "pmcid",
        "title",
        "journal",
        "year",
        "publication_types",
        "mesh_headings",
        "pubmed_url",
    ]


def _validate_config(config: EmbedConfig) -> None:
    for name in ("model_name", "corpus_name", "text_column"):
        value = getattr(config, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"--{name.replace('_', '-')} must be non-empty")
    if config.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if config.rows_per_part < 1:
        raise ValueError("--rows-per-part must be >= 1")
    if config.max_seq_length is None or config.max_seq_length < 1:
        raise ValueError("--max-seq-length must be >= 1")
    if config.chunk_size_words < 1:
        raise ValueError("--chunk-size-words must be >= 1")
    if config.chunk_overlap_words < 0:
        raise ValueError("--chunk-overlap-words must be >= 0")
    if config.chunk_overlap_words >= config.chunk_size_words:
        raise ValueError("--chunk-overlap-words must be smaller than --chunk-size-words")
    if config.max_chunks < 1:
        raise ValueError("--max-chunks must be >= 1")
    if config.chunk_selection not in {"first", "last"}:
        raise ValueError("--chunk-selection must be 'first' or 'last'")
    if not isinstance(config.normalize_embeddings, bool):
        raise TypeError("embedding normalization must be explicitly boolean")
    if config.limit is not None and config.limit < 1:
        raise ValueError("--limit must be >= 1 when supplied")
    if not config.input_path.exists():
        raise FileNotFoundError(config.input_path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


if __name__ == "__main__":
    main()
