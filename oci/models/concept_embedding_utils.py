"""Utilities for concept-initialized sentence-chunk extractors."""

from __future__ import annotations

import re
from typing import List


_WORD_RE = re.compile(r"\S+")


def chunk_text_words(
    text: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
    max_chunks: int,
) -> List[str]:
    """Split text into overlapping word chunks.

    This intentionally uses simple whitespace word spans. Sentence-transformer
    tokenizers handle subword details after chunking, while this function keeps
    the cache key and chunk boundaries model-independent.
    """
    if chunk_size_words < 1:
        raise ValueError("chunk_size_words must be >= 1")
    if max_chunks < 1:
        raise ValueError("max_chunks must be >= 1")
    if chunk_overlap_words >= chunk_size_words:
        raise ValueError(
            "chunk_overlap_words must be smaller than chunk_size_words"
        )
    if chunk_overlap_words < 0:
        raise ValueError("chunk_overlap_words must be >= 0")

    words = [match.group(0) for match in _WORD_RE.finditer(text or "")]
    if not words:
        return [""]

    stride = chunk_size_words - chunk_overlap_words
    chunks: List[str] = []
    start = 0
    while start < len(words) and len(chunks) < max_chunks:
        chunk_words = words[start:start + chunk_size_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        start += stride

    return chunks or [""]
