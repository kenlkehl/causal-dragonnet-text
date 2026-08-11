"""Focused coverage for the HTR extractor retained by Stage 1."""

from __future__ import annotations

import pytest

from oci.models.hierarchical_transformer_extractor import (
    HierarchicalTransformerExtractor,
    split_text_into_word_chunks,
)
from oci.models.lossless_tokenization import SemanticTruncationError


SAMPLE_TEXTS = (
    "Patient has metastatic lung cancer and performance status one.",
    "Treatment began after progression on first-line therapy.",
)


def _hash_extractor(**overrides):
    options = {
        "sentence_encoder_model": "hash",
        "chunk_size_words": 5,
        "chunk_overlap_words": 1,
        "max_chunks": 8,
        "num_transformer_layers": 1,
        "num_attention_heads": 2,
        "transformer_dim": 32,
        "projection_dim": 16,
        "hash_embedding_dim": 32,
        "transformer_dropout": 0.0,
    }
    options.update(overrides)
    return HierarchicalTransformerExtractor(**options)


def test_hash_backend_forward_shape():
    output = _hash_extractor()(list(SAMPLE_TEXTS))
    assert tuple(output.shape) == (2, 16)


def test_attention_evidence_preserves_source_chunks():
    evidence = _hash_extractor().get_attention_evidence(
        list(SAMPLE_TEXTS),
        row_ids=[10, 11],
        fold=2,
        stage="nuisance",
        top_k=2,
    )
    assert evidence
    assert {row["row_id"] for row in evidence} == {10, 11}
    assert all(row["stage"] == "nuisance" for row in evidence)
    assert all(row["chunk_text"] for row in evidence)
    assert all(0.0 <= row["attention"] <= 1.0 for row in evidence)


def test_word_chunking_fails_closed_when_capacity_would_truncate():
    text = " ".join(f"word{index}" for index in range(12))
    with pytest.raises(SemanticTruncationError, match="semantic truncation is forbidden"):
        split_text_into_word_chunks(
            text,
            chunk_size_words=4,
            chunk_overlap_words=1,
            max_chunks=2,
        )


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        _hash_extractor(chunk_overlap_words=-1)
    with pytest.raises(ValueError, match="smaller than chunk_size_words"):
        _hash_extractor(chunk_overlap_words=5)
