"""Persistent cache for sentence-transformer chunk embeddings."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .concept_embedding_utils import chunk_text_words
from .hidden_state_cache import VariableLengthArray, VariableLengthMaskArray


logger = logging.getLogger(__name__)

_SENTENCE_TRANSFORMER_CACHE = {}


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def load_sentence_transformer(model_name: str, device: Optional[torch.device] = None):
    """Load a sentence-transformer model lazily."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for concept_embedding_cnn. "
            "Install the project dependency or run: pip install sentence-transformers"
        ) from exc
    model_device = str(device) if device is not None else None
    key = (model_name, model_device)
    if key not in _SENTENCE_TRANSFORMER_CACHE:
        _SENTENCE_TRANSFORMER_CACHE[key] = SentenceTransformer(
            model_name,
            device=model_device,
        )
    return _SENTENCE_TRANSFORMER_CACHE[key]


class ConceptEmbeddingCache:
    """Cache of sentence-transformer embeddings for overlapping text chunks.

    The cache intentionally exposes the same array properties as HiddenStateCache:
    each sample indexes to a variable-length matrix. In this cache the sequence
    axis is chunks rather than tokens, with shape (n_chunks, embedding_dim).
    """

    def __init__(
        self,
        cache_dir: str,
        sentence_model_name: str,
        dataset_path: str,
        chunk_size_words: int,
        chunk_overlap_words: int,
        max_chunks: int,
        normalize_embeddings: bool = True,
    ):
        self._cache_dir = Path(cache_dir)
        self._sentence_model_name = sentence_model_name
        self._dataset_path = dataset_path
        self._chunk_size_words = chunk_size_words
        self._chunk_overlap_words = chunk_overlap_words
        self._max_chunks = max_chunks
        self._normalize_embeddings = normalize_embeddings
        self._cache_hash = self.compute_cache_hash(
            sentence_model_name=sentence_model_name,
            dataset_path=dataset_path,
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
            max_chunks=max_chunks,
            normalize_embeddings=normalize_embeddings,
        )
        self._cache_path = self._cache_dir / f"cecnn_chunk_embeddings_{self._cache_hash}"
        self._flat_mmap = None
        self._offsets = None
        self._metadata = None
        self._hs_array = None
        self._mask_array = None

    @staticmethod
    def compute_cache_hash(
        sentence_model_name: str,
        dataset_path: str,
        chunk_size_words: int,
        chunk_overlap_words: int,
        max_chunks: int,
        normalize_embeddings: bool = True,
    ) -> str:
        key = "|".join(
            [
                sentence_model_name,
                os.path.abspath(dataset_path),
                f"words{chunk_size_words}",
                f"overlap{chunk_overlap_words}",
                f"max{max_chunks}",
                f"norm{int(normalize_embeddings)}",
            ]
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]

    @property
    def cache_hash(self) -> str:
        return self._cache_hash

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    @property
    def hidden_size(self) -> int:
        if self._metadata is None:
            self._load_metadata()
        return int(self._metadata["hidden_size"])

    @property
    def actual_max_len(self) -> int:
        if self._metadata is None:
            self._load_metadata()
        return int(self._metadata["actual_max_len"])

    @property
    def chunk_counts(self) -> Optional[List[int]]:
        if self._metadata is None:
            self._load_metadata()
        return self._metadata.get("chunk_counts")

    @property
    def cache_size_gb(self) -> float:
        if self._metadata is None:
            self._load_metadata()
        return (
            int(self._metadata["total_chunks"])
            * int(self._metadata["hidden_size"])
            * 2
            / 1e9
        )

    @property
    def hidden_states_array(self):
        if self._hs_array is None:
            self.open()
        return self._hs_array

    @property
    def attention_mask_array(self):
        if self._mask_array is None:
            self.open()
        return self._mask_array

    def _load_metadata(self) -> None:
        meta_path = self._cache_path / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Cache metadata not found: {meta_path}")
        with open(meta_path) as f:
            self._metadata = json.load(f)

    def is_valid(self, expected_num_samples: int) -> bool:
        try:
            meta_path = self._cache_path / "metadata.json"
            emb_path = self._cache_path / "chunk_embeddings.npy"
            offsets_path = self._cache_path / "offsets.npy"
            if not all(p.exists() for p in [meta_path, emb_path, offsets_path]):
                return False

            self._load_metadata()
            if self._metadata.get("cache_hash") != self._cache_hash:
                return False
            if self._metadata.get("num_samples") != expected_num_samples:
                return False
            if self._metadata.get("storage_format") != "variable_length_chunks":
                return False

            offsets = np.load(str(offsets_path))
            if len(offsets) != expected_num_samples + 1:
                return False
            embeddings = np.load(str(emb_path), mmap_mode="r")
            if embeddings.shape[0] != int(offsets[-1]):
                return False
            if embeddings.shape[1] != int(self._metadata["hidden_size"]):
                return False
            return True
        except Exception as exc:
            logger.warning("Concept embedding cache validation failed: %s", exc)
            return False

    def precompute(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
        batch_size: int = 128,
    ) -> None:
        """Precompute sentence-transformer embeddings for all text chunks."""
        num_samples = len(texts)
        logger.info(
            "Precomputing concept chunk embeddings: samples=%d, model=%s",
            num_samples,
            self._sentence_model_name,
        )

        sample_chunks = [
            chunk_text_words(
                text,
                self._chunk_size_words,
                self._chunk_overlap_words,
                self._max_chunks,
            )
            for text in texts
        ]
        chunk_counts = [len(chunks) for chunks in sample_chunks]
        total_chunks = int(sum(chunk_counts))
        offsets = np.zeros(num_samples + 1, dtype=np.int64)
        for i, count in enumerate(chunk_counts):
            offsets[i + 1] = offsets[i] + count

        flat_chunks = [chunk for chunks in sample_chunks for chunk in chunks]
        encoder = load_sentence_transformer(self._sentence_model_name, device=device)

        self._cache_path.mkdir(parents=True, exist_ok=True)
        emb_path = self._cache_path / "chunk_embeddings.npy"
        offsets_path = self._cache_path / "offsets.npy"
        np.save(str(offsets_path), offsets)

        effective_batch_size = max(1, int(batch_size))
        logger.info(
            "  Total chunks: %d across %d samples (mean %.1f, max %d)",
            total_chunks,
            num_samples,
            total_chunks / max(num_samples, 1),
            max(chunk_counts) if chunk_counts else 0,
        )
        logger.info("  Encoding chunks with batch_size=%d", effective_batch_size)

        emb_mmap = None
        embedding_dim = None
        cursor = 0
        while cursor < total_chunks:
            end = min(cursor + effective_batch_size, total_chunks)
            batch_chunks = flat_chunks[cursor:end]
            try:
                batch_embeddings = encoder.encode(
                    batch_chunks,
                    batch_size=len(batch_chunks),
                    convert_to_numpy=True,
                    normalize_embeddings=self._normalize_embeddings,
                    show_progress_bar=False,
                )
            except Exception as exc:
                if _is_cuda_oom(exc) and effective_batch_size > 1:
                    new_batch_size = max(1, effective_batch_size // 2)
                    logger.warning(
                        "CUDA OOM while encoding concept chunks; reducing "
                        "batch_size from %d to %d and retrying",
                        effective_batch_size,
                        new_batch_size,
                    )
                    effective_batch_size = new_batch_size
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

            batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
            if batch_embeddings.ndim == 1:
                batch_embeddings = batch_embeddings.reshape(1, -1)
            if batch_embeddings.ndim != 2:
                raise RuntimeError(
                    "Sentence transformer returned unexpected shape "
                    f"{batch_embeddings.shape}"
                )
            if batch_embeddings.shape[0] != len(batch_chunks):
                raise RuntimeError(
                    "Sentence transformer returned "
                    f"{batch_embeddings.shape[0]} embeddings for "
                    f"{len(batch_chunks)} chunks"
                )

            if emb_mmap is None:
                embedding_dim = int(batch_embeddings.shape[1])
                emb_mmap = np.lib.format.open_memmap(
                    str(emb_path),
                    mode="w+",
                    dtype=np.float16,
                    shape=(total_chunks, embedding_dim),
                )
            elif batch_embeddings.shape[1] != embedding_dim:
                raise RuntimeError(
                    "Sentence transformer embedding dimension changed from "
                    f"{embedding_dim} to {batch_embeddings.shape[1]}"
                )

            emb_mmap[cursor:end] = batch_embeddings.astype(np.float16)
            cursor = end

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if emb_mmap is None or embedding_dim is None:
            raise RuntimeError("No chunk embeddings were generated")
        emb_mmap.flush()
        logger.info(
            "  Encoded %d chunks; final batch_size=%d",
            total_chunks,
            effective_batch_size,
        )

        metadata = {
            "sentence_model_name": self._sentence_model_name,
            "hidden_size": embedding_dim,
            "num_samples": num_samples,
            "total_chunks": total_chunks,
            "chunk_counts": chunk_counts,
            "chunk_size_words": self._chunk_size_words,
            "chunk_overlap_words": self._chunk_overlap_words,
            "max_chunks": self._max_chunks,
            "normalize_embeddings": self._normalize_embeddings,
            "actual_max_len": max(chunk_counts) if chunk_counts else 0,
            "storage_format": "variable_length_chunks",
            "dataset_path": os.path.abspath(self._dataset_path),
            "cache_hash": self._cache_hash,
            "created_at": datetime.now().isoformat(),
            "dtype": "float16",
            "precompute_batch_size": effective_batch_size,
        }
        with open(self._cache_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        self._metadata = metadata
        logger.info(
            "Concept embedding cache created: %.3f GB, %d chunks",
            self.cache_size_gb,
            total_chunks,
        )

    def precompute_multi_gpu(
        self,
        texts: List[str],
        devices: List[torch.device],
        batch_size: int = 128,
    ) -> None:
        """Compatibility wrapper; sentence-transformer handles one device here."""
        device = devices[0] if devices else None
        self.precompute(texts, device=device, batch_size=batch_size)

    def open(self) -> None:
        if self._metadata is None:
            self._load_metadata()
        emb_path = self._cache_path / "chunk_embeddings.npy"
        offsets_path = self._cache_path / "offsets.npy"
        flat = np.load(str(emb_path), mmap_mode="r")
        offsets = np.load(str(offsets_path))
        self._flat_mmap = flat
        self._offsets = offsets
        self._hs_array = VariableLengthArray(flat, offsets)
        self._mask_array = VariableLengthMaskArray(offsets)

    def preload_to_ram(self) -> None:
        """Load memmaps into RAM for DataLoader workers."""
        if self._metadata is None:
            self._load_metadata()
        emb_path = self._cache_path / "chunk_embeddings.npy"
        offsets_path = self._cache_path / "offsets.npy"
        flat = np.load(str(emb_path))
        offsets = np.load(str(offsets_path))
        self._flat_mmap = flat
        self._offsets = offsets
        self._hs_array = VariableLengthArray(flat, offsets)
        self._mask_array = VariableLengthMaskArray(offsets)

    def load_batch(self, indices: List[int], device: torch.device):
        """Load and pad a batch, matching HiddenStateCache.load_batch."""
        self.open()
        arrays = [self._hs_array[int(i)] for i in indices]
        max_len = max(arr.shape[0] for arr in arrays)
        hidden_size = arrays[0].shape[-1]
        hs = np.zeros((len(arrays), max_len, hidden_size), dtype=np.float32)
        mask = np.zeros((len(arrays), max_len), dtype=np.float32)
        for i, arr in enumerate(arrays):
            length = arr.shape[0]
            hs[i, :length] = np.asarray(arr, dtype=np.float32)
            mask[i, :length] = 1.0
        return torch.from_numpy(hs).to(device), torch.from_numpy(mask).to(device)
