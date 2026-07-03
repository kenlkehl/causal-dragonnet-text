"""Build a reusable embedding chunk cache for external retrieval corpora."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..models.concept_embedding_cache import ConceptEmbeddingCache

logger = logging.getLogger(__name__)


def build_embedding_chunk_cache(
    *,
    input_path: Path,
    text_column: str,
    output_cache_dir: Path,
    model_name: str,
    device: Optional[str] = None,
    batch_size: int = 16,
    max_seq_length: Optional[int] = 1024,
    chunk_size_words: int = 256,
    chunk_overlap_words: int = 64,
    max_chunks: int = 128,
    chunk_selection: str = "first",
    normalize_embeddings: bool = True,
    corpus_name: Optional[str] = None,
    source_id_column: Optional[str] = None,
    metadata_columns: Optional[List[str]] = None,
    force: bool = False,
) -> Path:
    """Embed an external text table into the cache format used by contrast retrieval."""
    frame = _read_table(input_path)
    if text_column not in frame.columns:
        raise ValueError(f"Text column {text_column!r} not found in {input_path}")
    texts = [str(text or "") for text in frame[text_column].fillna("")]
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    cache = ConceptEmbeddingCache(
        cache_dir=str(output_cache_dir),
        sentence_model_name=model_name,
        dataset_path=str(input_path),
        chunk_size_words=int(chunk_size_words),
        chunk_overlap_words=int(chunk_overlap_words),
        max_chunks=int(max_chunks),
        normalize_embeddings=bool(normalize_embeddings),
        chunk_selection=str(chunk_selection),
        max_seq_length=max_seq_length,
    )
    if force or not cache.is_valid(expected_num_samples=len(texts)):
        cache.precompute(
            texts,
            device=_torch_device_or_none(device),
            batch_size=int(batch_size),
        )
    else:
        logger.info("Reusing valid embedding chunk cache: %s", cache.cache_path)
    _write_row_metadata(
        cache.cache_path,
        frame=frame,
        text_column=text_column,
        source_id_column=source_id_column,
        metadata_columns=metadata_columns,
    )
    _update_cache_metadata(
        cache.cache_path,
        corpus_name=corpus_name or input_path.stem,
        text_column=text_column,
        source_id_column=source_id_column,
        metadata_columns=metadata_columns or _default_metadata_columns(frame, text_column),
        input_path=input_path,
    )
    return cache.cache_path


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv"}:
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(
        f"Unsupported input extension {suffix!r}; use parquet, csv, tsv, jsonl, or json."
    )


def _write_row_metadata(
    cache_path: Path,
    *,
    frame: pd.DataFrame,
    text_column: str,
    source_id_column: Optional[str],
    metadata_columns: Optional[List[str]],
) -> None:
    columns = metadata_columns or _default_metadata_columns(frame, text_column)
    columns = [col for col in columns if col in frame.columns and col != text_column]
    path = cache_path / "row_metadata.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for idx, row in frame.reset_index(drop=True).iterrows():
            payload: Dict[str, Any] = {"row_index": int(idx)}
            if source_id_column and source_id_column in frame.columns:
                payload["source_id"] = _jsonable(row[source_id_column])
            payload["metadata"] = {
                col: _jsonable(row[col]) for col in columns if col in frame.columns
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _default_metadata_columns(frame: pd.DataFrame, text_column: str) -> List[str]:
    preferred = [
        "pmid",
        "pmcid",
        "doi",
        "title",
        "journal",
        "year",
        "publication_year",
        "source",
        "url",
    ]
    return [col for col in preferred if col in frame.columns and col != text_column]


def _update_cache_metadata(
    cache_path: Path,
    *,
    corpus_name: str,
    text_column: str,
    source_id_column: Optional[str],
    metadata_columns: List[str],
    input_path: Path,
) -> None:
    path = cache_path / "metadata.json"
    metadata: Dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            metadata = json.load(f)
    metadata.update(
        {
            "corpus_name": corpus_name,
            "external_retrieval_corpus": True,
            "input_path": str(input_path),
            "text_column": text_column,
            "source_id_column": source_id_column,
            "metadata_columns": metadata_columns,
        }
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _torch_device_or_none(device: Optional[str]):
    if device is None or str(device).strip().lower() in {"", "auto"}:
        return None
    import torch

    return torch.device(str(device))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an external text chunk embedding cache for OCI retrieval."
    )
    parser.add_argument("--input", required=True, help="Input parquet/csv/tsv/jsonl/json file.")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-cache-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--chunk-size-words", type=int, default=256)
    parser.add_argument("--chunk-overlap-words", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=128)
    parser.add_argument("--chunk-selection", choices=["first", "last"], default="first")
    parser.add_argument("--no-normalize-embeddings", action="store_true")
    parser.add_argument("--corpus-name", default=None)
    parser.add_argument("--source-id-column", default=None)
    parser.add_argument("--metadata-column", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    cache_path = build_embedding_chunk_cache(
        input_path=Path(args.input),
        text_column=args.text_column,
        output_cache_dir=Path(args.output_cache_dir),
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
        max_chunks=args.max_chunks,
        chunk_selection=args.chunk_selection,
        normalize_embeddings=not args.no_normalize_embeddings,
        corpus_name=args.corpus_name,
        source_id_column=args.source_id_column,
        metadata_columns=args.metadata_column or None,
        force=args.force,
    )
    print(cache_path)


if __name__ == "__main__":
    main()
