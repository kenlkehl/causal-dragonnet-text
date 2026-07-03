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
    chunk_selection: str = "first",
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
    selection = str(chunk_selection).strip().lower()
    if selection not in {"first", "last"}:
        raise ValueError("chunk_selection must be 'first' or 'last'")

    words = [match.group(0) for match in _WORD_RE.finditer(text or "")]
    if not words:
        return [""]

    stride = chunk_size_words - chunk_overlap_words
    chunks: List[str] = []
    start = 0
    while start < len(words):
        chunk_words = words[start:start + chunk_size_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        start += stride

    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks] if selection == "first" else chunks[-max_chunks:]

    return chunks or [""]


def split_text_to_token_chunks(
    text: str,
    tokenizer,
    max_seq_length: int,
    chunk_overlap_tokens: int = 0,
) -> List[str]:
    """Split text into chunks whose encoded length fits max_seq_length tokens."""
    if max_seq_length < 1:
        raise ValueError("max_seq_length must be >= 1")
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens must be >= 0")

    raw_text = text or ""
    token_ids = _encode_without_special_tokens(tokenizer, raw_text)
    if not token_ids:
        return [""]

    content_limit = max(1, max_seq_length - _num_special_tokens(tokenizer))
    if len(token_ids) <= content_limit:
        return [raw_text]

    overlap = min(int(chunk_overlap_tokens), content_limit - 1)
    stride = max(1, content_limit - overlap)
    chunks: List[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + content_limit, len(token_ids))
        decoded = _decode_token_ids(tokenizer, token_ids[start:end]).strip()
        chunks.append(decoded)
        if end >= len(token_ids):
            break
        start += stride
    return chunks or [""]


def _encode_without_special_tokens(tokenizer, text: str) -> List[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _decode_token_ids(tokenizer, token_ids: List[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def _num_special_tokens(tokenizer) -> int:
    fn = getattr(tokenizer, "num_special_tokens_to_add", None)
    if callable(fn):
        try:
            return max(0, int(fn(pair=False)))
        except TypeError:
            return max(0, int(fn(False)))
    return 0
