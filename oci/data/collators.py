# oci/data/collators.py
"""Collator utilities for CDT data loading.

LLM extractors generally tokenize inside the extractor. Trainable CNN
extractors can use collate-time tokenization so DataLoader memory behavior is
controlled outside the model forward pass while still dynamically padding only
to each batch's required length.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def _collate_common_fields(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    texts = [item['text'] for item in batch]
    outcomes = torch.stack([item['outcome'] for item in batch])
    treatments = torch.stack([item['treatment'] for item in batch])
    text_ids = [item['text_id'] for item in batch]

    result = {
        'texts': texts,
        'outcome': outcomes,
        'treatment': treatments,
        'text_id': text_ids,
    }

    if 'explicit_feature_values' in batch[0]:
        result['explicit_feature_values'] = [
            item['explicit_feature_values'] for item in batch
        ]

    return result


class SimpleCNNTokenizingCollator:
    """Collate raw text samples into dynamically padded Simple CNN token tensors."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = _collate_common_fields(batch)
        input_ids, attention_mask = self.tokenizer.encode_batch(
            result['texts'],
            max_length=self.max_length,
        )
        result['input_ids'] = input_ids
        result['attention_mask'] = attention_mask
        return result


class HierarchicalCNNTokenizingCollator:
    """Collate raw text samples into hierarchical chunk token tensors."""

    def __init__(
        self,
        tokenizer,
        chunk_size: int,
        chunk_overlap: int,
        max_chunks: int,
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks

    @property
    def max_token_length(self) -> int:
        return self.chunk_size + max(0, self.max_chunks - 1) * (
            self.chunk_size - self.chunk_overlap
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        from ..models.text_chunking import chunk_token_ids, pad_and_batch_chunks

        result = _collate_common_fields(batch)

        batch_chunk_ids = []
        for text in result['texts']:
            token_ids = self.tokenizer.encode(text, max_length=self.max_token_length)
            chunks = chunk_token_ids(
                token_ids,
                self.chunk_size,
                self.chunk_overlap,
                self.max_chunks,
            )
            batch_chunk_ids.append(chunks)

        input_ids, attention_mask, chunk_mask = pad_and_batch_chunks(
            batch_chunk_ids,
            self.tokenizer.pad_token_id,
        )
        result['input_ids'] = input_ids
        result['attention_mask'] = attention_mask
        result['chunk_mask'] = chunk_mask
        return result


def create_trainable_cnn_collator(feature_extractor_type: str, feature_extractor) -> Optional[Callable]:
    """Return a tokenizing collator for trainable CNN extractors when possible."""
    tokenizer = getattr(feature_extractor, '_tokenizer', None)
    if tokenizer is None or not getattr(tokenizer, 'is_fitted', False):
        return None

    if feature_extractor_type == "simple_cnn":
        return SimpleCNNTokenizingCollator(
            tokenizer=tokenizer,
            max_length=getattr(feature_extractor, '_max_length'),
        )
    if feature_extractor_type == "hierarchical_cnn":
        return HierarchicalCNNTokenizingCollator(
            tokenizer=tokenizer,
            chunk_size=getattr(feature_extractor, '_chunk_size'),
            chunk_overlap=getattr(feature_extractor, '_chunk_overlap'),
            max_chunks=getattr(feature_extractor, '_max_chunks'),
        )
    return None


def create_collator(
    feature_extractor,
) -> Optional[Callable]:
    """Return a legacy collator for the given feature extractor.

    Args:
        feature_extractor: The model's feature extractor (already initialized)

    Returns:
        None. Specific training loops should opt in to tokenizing collators via
        ``create_trainable_cnn_collator``.
    """
    return None
