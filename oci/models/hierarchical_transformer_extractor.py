"""Hierarchical short-chunk transformer extractor.

This revives the older sentence-encoder + transformer-pooling idea and adapts
it to short overlapping word chunks. A pretrained encoder maps each chunk to a
vector, a small transformer with a learnable pool token aggregates chunks, and
attention from the pool token is exported as chunk-level evidence.
"""

import hashlib
import json
import logging
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gated_attention_pooling import GatedAttentionPooling, MultiHeadGatedAttentionPooling
from .lossless_tokenization import SemanticTruncationError

logger = logging.getLogger(__name__)
_TRANSFORMERS_ENCODER_INIT_LOCK = threading.Lock()
_SENTENCE_TRANSFORMER_INIT_LOCK = threading.Lock()
_LEGACY_BERT_MODEL_PREFIXES = ("prajjwal1/bert-",)
HTR_SENTENCE_ENCODER_TRAINING_AUDIT_SCHEMA = "htr_sentence_encoder_training_state_v1"
_HTR_ENVIRONMENT_OVERRIDES = (
    "OCI_HTR_ENCODER_BATCH_SIZE",
    "OCI_HTR_CHUNK_CACHE_MAX_ENTRIES",
    "OCI_HTR_TOKEN_CACHE_MAX_ENTRIES",
)


def _configured_activation(name: str) -> nn.Module:
    """Return one closed activation implementation.

    The exact GELU approximation is part of the name so a scientific
    constructor never inherits a framework default that could later change.
    """

    implementations = {
        "gelu_exact": lambda: nn.GELU(approximate="none"),
        "gelu_tanh": lambda: nn.GELU(approximate="tanh"),
        "relu": lambda: nn.ReLU(inplace=False),
        "silu": lambda: nn.SiLU(inplace=False),
        "tanh": nn.Tanh,
    }
    key = str(name)
    if key not in implementations:
        raise ValueError(
            "HTR activation must be one of: "
            + ", ".join(sorted(implementations))
        )
    return implementations[key]()


def split_text_into_word_chunks(
    text: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
    max_chunks: int,
) -> List[str]:
    """Split text into short overlapping word chunks.

    ``max_chunks`` is a configured capacity.  If it would bind, fail closed
    rather than selecting a prefix or tail of the clinical note.
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    if chunk_overlap_words < 0:
        raise ValueError("chunk_overlap_words must be nonnegative")
    if chunk_overlap_words >= chunk_size_words:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
    words = re.findall(r"\S+", str(text or ""))
    if not words:
        return [""]

    stride = chunk_size_words - chunk_overlap_words

    chunks = []
    start = 0
    while start < len(words):
        chunk_words = words[start : start + chunk_size_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        start += stride
    if len(chunks) > max_chunks:
        raise SemanticTruncationError(
            "HierarchicalTransformer note requires "
            f"{len(chunks)} chunks but configured max_chunks={max_chunks}; "
            "semantic truncation is forbidden. Increase max_chunks so the "
            "capacity is nonbinding."
        )
    if not chunks:
        raise RuntimeError("HierarchicalTransformer chunk planner produced no chunks")
    return chunks


def split_text_to_token_bounded_chunks(
    text: str,
    tokenizer: Any,
    max_chunk_length: int,
) -> List[str]:
    """Losslessly subdivide one word chunk to fit the encoder token limit.

    Word counts are only a heuristic for subword-token counts.  In particular,
    a long run of a non-whitespace character is one word but can be tens of
    thousands of tokenizer tokens.  This secondary planner partitions the
    original string without dropping any characters and verifies every
    resulting chunk with the same non-truncating tokenizer call used by HTR.

    Fast tokenizers provide offsets that let us make near-capacity partitions.
    Slow tokenizers fall back to exact character bisection.  Bisection may make
    more chunks, but it retains the source text exactly and always terminates
    unless the configured limit cannot encode even a single character.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for token-bounded HTR chunking")
    if max_chunk_length <= 0:
        raise ValueError("max_chunk_length must be positive")

    source = str(text or "")
    if _htr_tokenized_length(tokenizer, source) <= int(max_chunk_length):
        return [source]

    offset_chunks = _htr_chunks_from_fast_tokenizer_offsets(
        source,
        tokenizer,
        int(max_chunk_length),
    )
    candidates = offset_chunks if offset_chunks is not None else [source]
    bounded: List[str] = []
    for candidate in candidates:
        bounded.extend(
            _bisect_htr_chunk_to_token_limit(
                candidate,
                tokenizer,
                int(max_chunk_length),
            )
        )
    if "".join(bounded) != source:
        raise RuntimeError("HTR token-bounded chunk planner changed source text")
    return bounded or [""]


def _htr_tokenized_length(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        str(text or ""),
        padding=False,
        truncation=False,
    )
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise ValueError("HTR tokenizer response omitted input_ids")
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        if len(input_ids) != 1:
            raise ValueError("HTR tokenizer changed one chunk into multiple rows")
        input_ids = input_ids[0]
    return len(input_ids)


def _htr_num_special_tokens(tokenizer: Any) -> int:
    counter = getattr(tokenizer, "num_special_tokens_to_add", None)
    if callable(counter):
        try:
            return max(0, int(counter(pair=False)))
        except TypeError:
            return max(0, int(counter(False)))
    return _htr_tokenized_length(tokenizer, "")


def _htr_chunks_from_fast_tokenizer_offsets(
    text: str,
    tokenizer: Any,
    max_chunk_length: int,
) -> Optional[List[str]]:
    content_limit = int(max_chunk_length) - _htr_num_special_tokens(tokenizer)
    if content_limit < 1:
        return None
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        return None
    input_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if input_ids is None or offsets is None:
        return None
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        if len(input_ids) != 1 or len(offsets) != 1:
            return None
        input_ids = input_ids[0]
        offsets = offsets[0]
    if len(input_ids) <= content_limit or len(offsets) != len(input_ids):
        return None

    chunks: List[str] = []
    cursor = 0
    for token_start in range(content_limit, len(input_ids), content_limit):
        try:
            boundary = int(offsets[token_start][0])
        except (IndexError, TypeError, ValueError):
            return None
        if boundary <= cursor or boundary >= len(text):
            return None
        chunks.append(text[cursor:boundary])
        cursor = boundary
    chunks.append(text[cursor:])
    if any(chunk == "" for chunk in chunks) or "".join(chunks) != text:
        return None
    return chunks


def _bisect_htr_chunk_to_token_limit(
    text: str,
    tokenizer: Any,
    max_chunk_length: int,
) -> List[str]:
    pending = [str(text or "")]
    bounded: List[str] = []
    while pending:
        candidate = pending.pop()
        token_count = _htr_tokenized_length(tokenizer, candidate)
        if token_count <= int(max_chunk_length):
            bounded.append(candidate)
            continue
        if len(candidate) <= 1:
            raise SemanticTruncationError(
                "HTR max_chunk_length cannot encode one source character; "
                "lossless chunking is impossible and semantic truncation is forbidden "
                f"({token_count} > {int(max_chunk_length)})"
            )
        boundary = len(candidate) // 2
        left = candidate[:boundary]
        right = candidate[boundary:]
        # Stack order preserves the source order in the result.
        pending.append(right)
        pending.append(left)
    return bounded


def _htr_chunks_for_text(
    text: str,
    *,
    chunk_size_words: int,
    chunk_overlap_words: int,
    max_chunks: int,
    tokenizer: Optional[Any],
    max_chunk_length: int,
) -> List[str]:
    word_chunks = split_text_into_word_chunks(
        text,
        chunk_size_words,
        chunk_overlap_words,
        max_chunks,
    )
    if tokenizer is None:
        return word_chunks
    chunks = [
        bounded
        for word_chunk in word_chunks
        for bounded in split_text_to_token_bounded_chunks(
            word_chunk,
            tokenizer,
            max_chunk_length,
        )
    ]
    if len(chunks) > max_chunks:
        raise SemanticTruncationError(
            "HierarchicalTransformer token-bounded note requires "
            f"{len(chunks)} chunks but configured max_chunks={max_chunks}; "
            "semantic truncation is forbidden. Increase max_chunks so the "
            "capacity is nonbinding."
        )
    return chunks or [""]


class HierarchicalTransformerBatchPreprocessor:
    """CPU-only chunking/tokenization helper for DataLoader collators."""

    def __init__(
        self,
        *,
        tokenizer: Optional[Any],
        chunk_size_words: int,
        chunk_overlap_words: int,
        max_chunks: int,
        max_chunk_length: int,
        tokenize_for_transformers: bool,
        chunk_cache_max_entries: int,
        tokenization_cache_max_entries: int,
    ):
        self._tokenizer = tokenizer
        self._chunk_size_words = int(chunk_size_words)
        self._chunk_overlap_words = int(chunk_overlap_words)
        self._max_chunks = int(max_chunks)
        self._max_chunk_length = int(max_chunk_length)
        self._tokenize_for_transformers = bool(tokenize_for_transformers)
        self._chunk_cache_max_entries = int(chunk_cache_max_entries)
        self._tokenization_cache_max_entries = int(tokenization_cache_max_entries)
        self._chunk_cache: Dict[str, List[str]] = {}
        self._tokenization_cache: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}

    def __call__(self, texts: Sequence[str]) -> Dict[str, Any]:
        texts = [str(text or "") for text in texts]
        chunks = self._chunks_for_texts(texts)
        batch: Dict[str, Any] = {
            "texts": texts,
            "chunks": chunks,
        }
        if self._tokenize_for_transformers:
            flat_chunks = [chunk for row in chunks for chunk in row]
            encoded = self._tokenize_chunks(flat_chunks)
            batch["chunk_input_ids"] = encoded["input_ids"]
            batch["chunk_attention_mask"] = encoded["attention_mask"]
        return batch

    def _chunks_for_texts(self, texts: Sequence[str]) -> List[List[str]]:
        chunks_by_text: List[List[str]] = []
        for text in texts:
            key = str(text or "")
            chunks = self._chunk_cache.get(key)
            if chunks is None:
                chunks = _htr_chunks_for_text(
                    key,
                    chunk_size_words=self._chunk_size_words,
                    chunk_overlap_words=self._chunk_overlap_words,
                    max_chunks=self._max_chunks,
                    tokenizer=(
                        self._tokenizer if self._tokenize_for_transformers else None
                    ),
                    max_chunk_length=self._max_chunk_length,
                )
                if len(self._chunk_cache) < self._chunk_cache_max_entries:
                    self._chunk_cache[key] = chunks
            chunks_by_text.append(chunks)
        return chunks_by_text

    def _tokenize_chunks(self, chunks: Sequence[str]) -> Dict[str, torch.Tensor]:
        entries = [self._tokenize_one_chunk(chunk) for chunk in chunks]
        return _collate_tokenized_chunks(self._tokenizer, entries)

    def _tokenize_one_chunk(self, chunk: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if self._tokenizer is None:
            raise RuntimeError("Cannot tokenize chunks without a tokenizer")
        key = str(chunk or "")
        cached = self._tokenization_cache.get(key)
        if cached is not None:
            return cached

        encoded = self._tokenizer(
            key,
            padding=False,
            truncation=False,
        )
        input_ids = tuple(int(token_id) for token_id in encoded["input_ids"])
        attention_mask = tuple(int(mask_value) for mask_value in encoded["attention_mask"])
        if len(input_ids) > self._max_chunk_length:
            raise ValueError(
                "HTR tokenizer input exceeds max_chunk_length; semantic truncation is forbidden "
                f"({len(input_ids)} > {self._max_chunk_length})"
            )
        if len(self._tokenization_cache) < self._tokenization_cache_max_entries:
            self._tokenization_cache[key] = (input_ids, attention_mask)
        return input_ids, attention_mask


def _collate_tokenized_chunks(
    tokenizer: Any,
    entries: Sequence[Tuple[Tuple[int, ...], Tuple[int, ...]]],
) -> Dict[str, torch.Tensor]:
    if not entries:
        return {
            "input_ids": torch.zeros(0, 0, dtype=torch.long),
            "attention_mask": torch.zeros(0, 0, dtype=torch.long),
        }
    max_length = max(len(input_ids) for input_ids, _ in entries)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0) or 0
    input_ids_tensor = torch.full(
        (len(entries), max_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask_tensor = torch.zeros(
        len(entries),
        max_length,
        dtype=torch.long,
    )
    left_pad = getattr(tokenizer, "padding_side", "right") == "left"
    for row, (input_ids, attention_mask) in enumerate(entries):
        offset = max_length - len(input_ids) if left_pad else 0
        input_ids_tensor[row, offset : offset + len(input_ids)] = torch.as_tensor(
            input_ids,
            dtype=torch.long,
        )
        attention_mask_tensor[row, offset : offset + len(attention_mask)] = torch.as_tensor(
            attention_mask,
            dtype=torch.long,
        )
    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
    }


def _find_overlapping_word(
    words: Sequence[Tuple[str, int, int]],
    start: int,
    end: int,
) -> Optional[int]:
    best_idx: Optional[int] = None
    best_overlap = 0
    for idx, (_, word_start, word_end) in enumerate(words):
        overlap = min(end, word_end) - max(start, word_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx
    return best_idx


class _InterpretableTransformerLayer(nn.Module):
    """Transformer encoder layer that can return self-attention weights."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        activation: str,
        norm_style: str,
        layer_norm_eps: float,
        layer_norm_elementwise_affine: bool,
        layer_norm_bias: bool,
        attention_dropout: float,
        residual_dropout: float,
        feedforward_dropout: float,
        attention_bias: bool,
        feedforward_bias: bool,
    ):
        super().__init__()
        if norm_style not in {"pre_norm", "post_norm"}:
            raise ValueError("HTR transformer norm_style must be pre_norm or post_norm")
        self.norm_style = str(norm_style)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=attention_dropout,
            bias=attention_bias,
            batch_first=True,
        )
        self.linear1 = nn.Linear(
            d_model,
            dim_feedforward,
            bias=feedforward_bias,
        )
        self.linear2 = nn.Linear(
            dim_feedforward,
            d_model,
            bias=feedforward_bias,
        )
        norm_kwargs = {
            "eps": float(layer_norm_eps),
            "elementwise_affine": bool(layer_norm_elementwise_affine),
            "bias": bool(layer_norm_bias),
        }
        self.norm1 = nn.LayerNorm(d_model, **norm_kwargs)
        self.norm2 = nn.LayerNorm(d_model, **norm_kwargs)
        self.residual_dropout = nn.Dropout(residual_dropout, inplace=False)
        self.feedforward_dropout = nn.Dropout(
            feedforward_dropout,
            inplace=False,
        )
        self.activation = _configured_activation(activation)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.norm_style == "pre_norm":
            attention_input = self.norm1(x)
            attn_output, attn_weights = self.self_attn(
                attention_input,
                attention_input,
                attention_input,
                key_padding_mask=key_padding_mask,
                need_weights=return_attention,
                average_attn_weights=True,
            )
            x = x + self.residual_dropout(attn_output)
            feedforward_input = self.norm2(x)
            ff_output = self.linear2(
                self.feedforward_dropout(
                    self.activation(self.linear1(feedforward_input))
                )
            )
            x = x + self.residual_dropout(ff_output)
        else:
            attn_output, attn_weights = self.self_attn(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=return_attention,
                average_attn_weights=True,
            )
            x = self.norm1(x + self.residual_dropout(attn_output))
            ff_output = self.linear2(
                self.feedforward_dropout(self.activation(self.linear1(x)))
            )
            x = self.norm2(x + self.residual_dropout(ff_output))
        return x, attn_weights


class HierarchicalTransformerExtractor(nn.Module):
    """Short-chunk encoder followed by transformer pooling.

    Args:
        sentence_encoder_model: HuggingFace encoder name, or ``"hash"`` for a
            deterministic no-download backend useful for tests.
        freeze_sentence_encoder: Freeze HuggingFace encoder parameters.
        chunk_size_words: Words per chunk.
        chunk_overlap_words: Overlap between neighboring chunks.
        max_chunks: Maximum chunks per document.
        max_chunk_length: Max subword tokens per chunk for HuggingFace encoder.
        num_transformer_layers: Number of pooling transformer layers.
        num_attention_heads: Attention heads for pooling transformer.
        transformer_dim: Pooling transformer hidden size.
        transformer_dropout: Dropout used in pooling/projection layers.
        projection_dim: Final output dimension.
        hash_embedding_dim: Hash backend chunk vector dimension.
    """

    def __init__(
        self,
        sentence_encoder_model: str = "prajjwal1/bert-tiny",
        freeze_sentence_encoder: bool = False,
        chunk_size_words: int = 96,
        chunk_overlap_words: int = 24,
        max_chunks: int = 128,
        max_chunk_length: int = 128,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 4,
        transformer_dim: int = 256,
        transformer_dropout: float = 0.1,
        projection_dim: int = 128,
        hash_embedding_dim: int = 256,
        sentence_encoder_batch_size: int = 128,
        sentence_encoder_backend: str = "auto",
        sentence_pooling: str = "auto",
        normalize_sentence_embeddings: bool = True,
        trainable_sentence_encoder_layers: int = 0,
        role_attention: bool = False,
        w_attention_heads: int = 1,
        x_attention_heads: int = 1,
        transformer_feedforward_dim: Optional[int] = None,
        transformer_activation: str = "gelu_exact",
        transformer_norm_style: str = "post_norm",
        transformer_layer_norm_eps: float = 1e-5,
        transformer_layer_norm_elementwise_affine: bool = True,
        transformer_layer_norm_bias: bool = True,
        transformer_attention_dropout: Optional[float] = None,
        transformer_residual_dropout: Optional[float] = None,
        transformer_feedforward_dropout: Optional[float] = None,
        transformer_attention_bias: bool = True,
        transformer_feedforward_bias: bool = True,
        output_projection_depth: int = 1,
        output_projection_hidden_dim: Optional[int] = None,
        output_projection_activation: str = "gelu_exact",
        output_projection_dropout: Optional[float] = None,
        output_projection_hidden_layer_norm: bool = True,
        output_projection_final_layer_norm: bool = True,
        output_projection_bias: bool = True,
        pool_token_init_std: float = 0.02,
        positional_encoding_base: float = 10_000.0,
        environment_override_policy: str = "legacy_allow",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        if chunk_overlap_words < 0:
            raise ValueError("chunk_overlap_words must be nonnegative")
        if chunk_overlap_words >= chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
        if max_chunk_length <= 0:
            raise ValueError("max_chunk_length must be positive")

        self._device = device or torch.device("cpu")
        self._sentence_encoder_model = sentence_encoder_model
        self._freeze = freeze_sentence_encoder
        self._chunk_size_words = int(chunk_size_words)
        self._chunk_overlap_words = int(chunk_overlap_words)
        self._max_chunks = int(max_chunks)
        self._max_chunk_length = int(max_chunk_length)
        self._num_layers = int(num_transformer_layers)
        self._num_heads = int(num_attention_heads)
        self._transformer_dim = int(transformer_dim)
        self._dropout = float(transformer_dropout)
        self._projection_dim = int(projection_dim)
        self._hash_embedding_dim = int(hash_embedding_dim)
        environment_override_policy = str(environment_override_policy)
        if environment_override_policy not in {"forbid", "legacy_allow"}:
            raise ValueError(
                "environment_override_policy must be forbid or legacy_allow"
            )
        observed_environment_overrides = tuple(
            name for name in _HTR_ENVIRONMENT_OVERRIDES if name in os.environ
        )
        if (
            environment_override_policy == "forbid"
            and observed_environment_overrides
        ):
            raise RuntimeError(
                "typed HTR execution forbids environment overrides; unset "
                + ", ".join(observed_environment_overrides)
            )
        if environment_override_policy == "legacy_allow":
            env_batch_size = os.environ.get("OCI_HTR_ENCODER_BATCH_SIZE")
            if env_batch_size:
                sentence_encoder_batch_size = int(env_batch_size)
        if sentence_encoder_batch_size <= 0:
            raise ValueError("sentence_encoder_batch_size must be positive")
        sentence_encoder_backend = str(sentence_encoder_backend or "auto").lower()
        if sentence_encoder_backend not in {"auto", "sentence_transformers", "transformers"}:
            raise ValueError(
                "sentence_encoder_backend must be one of: auto, sentence_transformers, transformers"
            )
        sentence_pooling = str(sentence_pooling or "auto").lower()
        valid_pooling = {"auto", "cls", "last", "mean", "token_attention"}
        if sentence_pooling not in valid_pooling:
            raise ValueError(
                "sentence_pooling must be one of: auto, cls, last, mean, token_attention"
            )
        if trainable_sentence_encoder_layers < 0:
            raise ValueError("trainable_sentence_encoder_layers must be >= 0")
        if w_attention_heads < 1:
            raise ValueError("w_attention_heads must be >= 1")
        if x_attention_heads < 1:
            raise ValueError("x_attention_heads must be >= 1")
        feedforward_dim = (
            int(transformer_dim) * 4
            if transformer_feedforward_dim is None
            else int(transformer_feedforward_dim)
        )
        output_hidden_dim = (
            int(transformer_dim)
            if output_projection_hidden_dim is None
            else int(output_projection_hidden_dim)
        )
        attention_dropout = (
            float(transformer_dropout)
            if transformer_attention_dropout is None
            else float(transformer_attention_dropout)
        )
        residual_dropout = (
            float(transformer_dropout)
            if transformer_residual_dropout is None
            else float(transformer_residual_dropout)
        )
        feedforward_dropout = (
            float(transformer_dropout)
            if transformer_feedforward_dropout is None
            else float(transformer_feedforward_dropout)
        )
        output_dropout = (
            float(transformer_dropout)
            if output_projection_dropout is None
            else float(output_projection_dropout)
        )
        if feedforward_dim < 1 or output_hidden_dim < 1:
            raise ValueError("HTR feedforward/output hidden dimensions must be positive")
        if int(output_projection_depth) < 0:
            raise ValueError("HTR output_projection_depth must be nonnegative")
        if str(transformer_norm_style) not in {"pre_norm", "post_norm"}:
            raise ValueError("HTR transformer norm style is unsupported")
        _configured_activation(str(transformer_activation))
        _configured_activation(str(output_projection_activation))
        if (
            not math.isfinite(float(transformer_layer_norm_eps))
            or float(transformer_layer_norm_eps) <= 0.0
        ):
            raise ValueError("HTR layer-norm epsilon must be finite and positive")
        for name, value in (
            ("transformer_attention_dropout", attention_dropout),
            ("transformer_residual_dropout", residual_dropout),
            ("transformer_feedforward_dropout", feedforward_dropout),
            ("output_projection_dropout", output_dropout),
        ):
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"HTR {name} must be in [0, 1)")
        if (
            not math.isfinite(float(pool_token_init_std))
            or float(pool_token_init_std) < 0.0
        ):
            raise ValueError("HTR pool-token initialization std must be nonnegative")
        if (
            not math.isfinite(float(positional_encoding_base))
            or float(positional_encoding_base) <= 1.0
        ):
            raise ValueError("HTR positional-encoding base must exceed one")
        boolean_topology = {
            "transformer_layer_norm_elementwise_affine": (
                transformer_layer_norm_elementwise_affine
            ),
            "transformer_layer_norm_bias": transformer_layer_norm_bias,
            "transformer_attention_bias": transformer_attention_bias,
            "transformer_feedforward_bias": transformer_feedforward_bias,
            "output_projection_hidden_layer_norm": (
                output_projection_hidden_layer_norm
            ),
            "output_projection_final_layer_norm": (
                output_projection_final_layer_norm
            ),
            "output_projection_bias": output_projection_bias,
        }
        if any(type(value) is not bool for value in boolean_topology.values()):
            raise TypeError("HTR topology Boolean settings must be exact booleans")
        self._sentence_encoder_batch_size = int(sentence_encoder_batch_size)
        self._sentence_encoder_backend = sentence_encoder_backend
        self._sentence_pooling = sentence_pooling
        self._normalize_sentence_embeddings = bool(normalize_sentence_embeddings)
        self._trainable_sentence_encoder_layers = int(trainable_sentence_encoder_layers)
        self._role_attention = bool(role_attention)
        self._w_attention_heads = int(w_attention_heads)
        self._x_attention_heads = int(x_attention_heads)
        self._transformer_feedforward_dim = feedforward_dim
        self._transformer_activation = str(transformer_activation)
        self._transformer_norm_style = str(transformer_norm_style)
        self._transformer_layer_norm_eps = float(transformer_layer_norm_eps)
        self._transformer_layer_norm_elementwise_affine = bool(
            transformer_layer_norm_elementwise_affine
        )
        self._transformer_layer_norm_bias = bool(transformer_layer_norm_bias)
        self._transformer_attention_dropout = attention_dropout
        self._transformer_residual_dropout = residual_dropout
        self._transformer_feedforward_dropout = feedforward_dropout
        self._transformer_attention_bias = bool(transformer_attention_bias)
        self._transformer_feedforward_bias = bool(transformer_feedforward_bias)
        self._output_projection_depth = int(output_projection_depth)
        self._output_projection_hidden_dim = output_hidden_dim
        self._output_projection_activation = str(output_projection_activation)
        self._output_projection_dropout = output_dropout
        self._output_projection_hidden_layer_norm = bool(
            output_projection_hidden_layer_norm
        )
        self._output_projection_final_layer_norm = bool(
            output_projection_final_layer_norm
        )
        self._output_projection_bias = bool(output_projection_bias)
        self._pool_token_init_std = float(pool_token_init_std)
        self._positional_encoding_base = float(positional_encoding_base)
        self._environment_override_policy = environment_override_policy
        self._encoder_has_trainable_params = False
        self._hash_backend = str(sentence_encoder_model).lower() in {
            "hash",
            "hashed",
            "hashing",
            "test_hash",
        }
        if self._hash_backend and sentence_pooling == "token_attention":
            raise ValueError("token_attention pooling requires a transformer token encoder")

        self._tokenizer = None
        self._sentence_encoder = None
        self._sentence_transformer_encoder = None
        self._token_pooling: Optional[GatedAttentionPooling] = None
        self._w_token_pooling: Optional[MultiHeadGatedAttentionPooling] = None
        self._x_token_pooling: Optional[MultiHeadGatedAttentionPooling] = None
        self._w_chunk_pooling: Optional[MultiHeadGatedAttentionPooling] = None
        self._x_chunk_pooling: Optional[MultiHeadGatedAttentionPooling] = None
        self._resolved_sentence_encoder_path: Optional[str] = None
        self._sentence_dim = self._hash_embedding_dim if self._hash_backend else None
        self._encoder_initialized = self._hash_backend
        self._chunk_cache: Dict[str, List[str]] = {}
        self._tokenization_cache: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        self._chunk_cache_max_entries = int(
            os.environ.get("OCI_HTR_CHUNK_CACHE_MAX_ENTRIES", "100000")
            if environment_override_policy == "legacy_allow"
            else 100000
        )
        self._tokenization_cache_max_entries = int(
            os.environ.get("OCI_HTR_TOKEN_CACHE_MAX_ENTRIES", "200000")
            if environment_override_policy == "legacy_allow"
            else 200000
        )

        self._input_projection = nn.Linear(
            self._hash_embedding_dim if self._hash_backend else transformer_dim,
            transformer_dim,
        )
        if not self._hash_backend:
            # Replaced lazily once the encoder hidden size is known.
            self._input_projection = None

        self._pool_token = nn.Parameter(
            torch.randn(1, transformer_dim) * self._pool_token_init_std
        )
        self.register_buffer(
            "_positional_encoding",
            self._make_positional_encoding(
                max_chunks + 1,
                transformer_dim,
                base=self._positional_encoding_base,
            ),
        )
        self._transformer_layers = nn.ModuleList(
            [
                _InterpretableTransformerLayer(
                    d_model=transformer_dim,
                    nhead=num_attention_heads,
                    dim_feedforward=self._transformer_feedforward_dim,
                    activation=self._transformer_activation,
                    norm_style=self._transformer_norm_style,
                    layer_norm_eps=self._transformer_layer_norm_eps,
                    layer_norm_elementwise_affine=(
                        self._transformer_layer_norm_elementwise_affine
                    ),
                    layer_norm_bias=self._transformer_layer_norm_bias,
                    attention_dropout=self._transformer_attention_dropout,
                    residual_dropout=self._transformer_residual_dropout,
                    feedforward_dropout=self._transformer_feedforward_dropout,
                    attention_bias=self._transformer_attention_bias,
                    feedforward_bias=self._transformer_feedforward_bias,
                )
                for _ in range(num_transformer_layers)
            ]
        )
        output_layers: List[nn.Module] = []
        output_input_dim = int(transformer_dim)
        output_norm_kwargs = {
            "eps": self._transformer_layer_norm_eps,
            "elementwise_affine": self._transformer_layer_norm_elementwise_affine,
            "bias": self._transformer_layer_norm_bias,
        }
        for _ in range(self._output_projection_depth):
            output_layers.append(
                nn.Linear(
                    output_input_dim,
                    self._output_projection_hidden_dim,
                    bias=self._output_projection_bias,
                )
            )
            if self._output_projection_hidden_layer_norm:
                output_layers.append(
                    nn.LayerNorm(
                        self._output_projection_hidden_dim,
                        **output_norm_kwargs,
                    )
                )
            output_layers.append(
                _configured_activation(self._output_projection_activation)
            )
            output_layers.append(
                nn.Dropout(
                    self._output_projection_dropout,
                    inplace=False,
                )
            )
            output_input_dim = self._output_projection_hidden_dim
        output_layers.append(
            nn.Linear(
                output_input_dim,
                projection_dim,
                bias=self._output_projection_bias,
            )
        )
        if self._output_projection_final_layer_norm:
            output_layers.append(
                nn.LayerNorm(projection_dim, **output_norm_kwargs)
            )
        self._output_projection = nn.Sequential(*output_layers)
        if self._role_attention:
            self._w_chunk_pooling = MultiHeadGatedAttentionPooling(
                hidden_dim=transformer_dim,
                attention_dim=transformer_dim,
                num_heads=self._w_attention_heads,
            )
            self._x_chunk_pooling = MultiHeadGatedAttentionPooling(
                hidden_dim=transformer_dim,
                attention_dim=transformer_dim,
                num_heads=self._x_attention_heads,
            )
        self._last_chunks: List[List[str]] = []
        self._last_chunk_weights: Optional[torch.Tensor] = None
        self._last_role_chunk_weights: Dict[str, torch.Tensor] = {}
        self._last_token_weights_by_chunk: List[torch.Tensor] = []
        self._last_role_token_weights_by_chunk: Dict[str, List[torch.Tensor]] = {}
        self._capture_token_attention = False
        self._token_weight_capture_buffer: List[torch.Tensor] = []
        self._role_token_weight_capture_buffer: Dict[str, List[torch.Tensor]] = {}
        self.to(self._device)

    @property
    def output_dim(self) -> int:
        return self._projection_dim

    @property
    def has_role_features(self) -> bool:
        return self._role_attention

    @staticmethod
    def _make_positional_encoding(
        max_len: int,
        d_model: int,
        *,
        base: float = 10_000.0,
    ) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(float(base)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _ensure_encoder_initialized(self) -> None:
        if self._encoder_initialized:
            return
        if self._effective_sentence_encoder_backend() == "sentence_transformers":
            with _SENTENCE_TRANSFORMER_INIT_LOCK:
                if self._encoder_initialized:
                    return
                self._ensure_sentence_transformer_initialized()
                self._encoder_initialized = True
        else:
            with _TRANSFORMERS_ENCODER_INIT_LOCK:
                if self._encoder_initialized:
                    return
                self._ensure_transformers_initialized()
                self._encoder_initialized = True

    def _ensure_sentence_transformer_initialized(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence-transformers is required for this HTR encoder") from exc

        logger.info("Loading sentence-transformer chunk encoder: %s", self._sentence_encoder_model)
        self._sentence_transformer_encoder = SentenceTransformer(
            self._sentence_encoder_model,
            device=str(self._device),
            model_kwargs={"torch_dtype": torch.float32},
        )
        self._sentence_transformer_encoder.float()
        self._sentence_transformer_encoder.eval()
        self._sentence_dim = self._sentence_transformer_encoder.get_sentence_embedding_dimension()
        if self._sentence_dim is None:
            probe = self._sentence_transformer_encoder.encode(
                [""],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize_sentence_embeddings,
                show_progress_bar=False,
            )
            self._sentence_dim = int(np.asarray(probe).shape[-1])
        else:
            self._sentence_dim = int(self._sentence_dim)
        self._input_projection = nn.Linear(self._sentence_dim, self._transformer_dim).to(
            self._device
        )
        self._encoder_has_trainable_params = False

    def _ensure_transformers_initialized(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for hierarchical_transformer. "
                f"Failed to import AutoModel/AutoTokenizer from transformers: {exc}"
            ) from exc

        logger.info("Loading chunk encoder: %s", self._sentence_encoder_model)
        self._tokenizer = self._load_tokenizer(AutoTokenizer)
        if self._effective_sentence_pooling() == "last":
            self._tokenizer.padding_side = "left"
        self._sentence_encoder = self._load_transformers_model(AutoModel)
        self._sentence_encoder = self._sentence_encoder.to(self._device)
        self._sentence_dim = int(self._sentence_encoder.config.hidden_size)
        self._configure_sentence_encoder_training()
        self._input_projection = nn.Linear(self._sentence_dim, self._transformer_dim).to(
            self._device
        )
        self._ensure_token_pooling_initialized()
        logger.info(
            "Chunk encoder initialized with backend=%s pooling=%s hidden_dim=%s "
            "device=%s trainable_encoder_params=%s/%s",
            self._effective_sentence_encoder_backend(),
            self._effective_sentence_pooling(),
            self._sentence_dim,
            self._device,
            self._trainable_sentence_encoder_parameter_count(),
            self._total_sentence_encoder_parameter_count(),
        )

    def _load_tokenizer(self, auto_tokenizer_cls):
        if self._should_prefer_legacy_bert_loader():
            try:
                tokenizer = self._load_legacy_bert_tokenizer()
            except Exception as legacy_exc:
                logger.debug(
                    "Preferred legacy BERT tokenizer load failed for %s: %s",
                    self._sentence_encoder_model,
                    legacy_exc,
                )
            else:
                if tokenizer is not None:
                    return tokenizer
        try:
            return auto_tokenizer_cls.from_pretrained(
                self._sentence_encoder_model,
                use_fast=True,
            )
        except Exception as fast_exc:
            logger.warning(
                "Fast tokenizer load failed for %s (%s). Retrying with use_fast=False.",
                self._sentence_encoder_model,
                fast_exc,
            )
            try:
                return auto_tokenizer_cls.from_pretrained(
                    self._sentence_encoder_model,
                    use_fast=False,
                )
            except Exception as slow_exc:
                try:
                    tokenizer = self._load_legacy_bert_tokenizer()
                except Exception as legacy_exc:
                    raise RuntimeError(
                        "Could not load tokenizer for "
                        f"{self._sentence_encoder_model!r}. Install tokenizer conversion "
                        "dependencies with `pip install sentencepiece tiktoken`, or use a "
                        "BERT/WordPiece model with tokenizer files available locally. "
                        f"Fast tokenizer error: {fast_exc}. Slow tokenizer error: {slow_exc}. "
                        f"Legacy BERT tokenizer error: {legacy_exc}."
                    ) from legacy_exc
                if tokenizer is not None:
                    return tokenizer
                raise RuntimeError(
                    "Could not load tokenizer for "
                    f"{self._sentence_encoder_model!r}. Install tokenizer conversion "
                    "dependencies with `pip install sentencepiece tiktoken`, or use a "
                    "BERT/WordPiece model with tokenizer files available locally. "
                    f"Fast tokenizer error: {fast_exc}. Slow tokenizer error: {slow_exc}."
                ) from slow_exc

    def _load_transformers_model(self, auto_model_cls):
        if self._should_prefer_legacy_bert_loader():
            try:
                model = self._load_legacy_bert_model()
            except Exception as legacy_exc:
                logger.debug(
                    "Preferred legacy BERT model load failed for %s: %s",
                    self._sentence_encoder_model,
                    legacy_exc,
                )
            else:
                if model is not None:
                    return model
        try:
            return auto_model_cls.from_pretrained(self._sentence_encoder_model)
        except Exception as auto_exc:
            try:
                model = self._load_legacy_bert_model()
            except Exception as legacy_exc:
                raise RuntimeError(
                    "Could not load transformer chunk encoder for "
                    f"{self._sentence_encoder_model!r}. AutoModel error: {auto_exc}. "
                    f"Legacy BERT model error: {legacy_exc}."
                ) from legacy_exc
            if model is not None:
                return model
            raise

    def _load_legacy_bert_tokenizer(self):
        if not self._should_try_legacy_bert_loader():
            return None
        try:
            from transformers import BertTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for BertTokenizer fallback") from exc

        resolved_model = self._resolve_sentence_encoder_path()
        vocab_file = Path(resolved_model) / "vocab.txt"
        if not vocab_file.exists():
            raise FileNotFoundError(f"legacy BERT tokenizer fallback expected {vocab_file}")
        logger.info("Loading legacy BERT tokenizer from local snapshot: %s", resolved_model)
        return BertTokenizer.from_pretrained(resolved_model, local_files_only=True)

    def _load_legacy_bert_model(self):
        if not self._should_try_legacy_bert_loader():
            return None
        try:
            from transformers import BertConfig, BertForPreTraining, BertModel
        except ImportError as exc:
            raise ImportError("transformers is required for BertModel fallback") from exc

        resolved_model = self._resolve_sentence_encoder_path()
        logger.info("Loading legacy BERT model from local snapshot: %s", resolved_model)
        config = BertConfig.from_pretrained(resolved_model, local_files_only=True)
        if self._legacy_bert_checkpoint_has_pretraining_heads(resolved_model):
            pretraining_model = BertForPreTraining.from_pretrained(
                resolved_model,
                config=config,
                local_files_only=True,
            )
            return pretraining_model.bert
        return BertModel.from_pretrained(
            resolved_model,
            config=config,
            local_files_only=True,
        )

    def _should_try_legacy_bert_loader(self) -> bool:
        model_name = str(self._sentence_encoder_model).lower()
        if "bert" in model_name:
            return True
        model_path = Path(str(self._sentence_encoder_model)).expanduser()
        return model_path.exists() and (model_path / "vocab.txt").exists()

    def _should_prefer_legacy_bert_loader(self) -> bool:
        model_name = str(self._sentence_encoder_model).lower()
        if any(model_name.startswith(prefix) for prefix in _LEGACY_BERT_MODEL_PREFIXES):
            return True
        model_path = Path(str(self._sentence_encoder_model)).expanduser()
        return model_path.exists() and self._snapshot_looks_like_legacy_bert(model_path)

    @staticmethod
    def _snapshot_looks_like_legacy_bert(model_path: Path) -> bool:
        config_path = model_path / "config.json"
        vocab_path = model_path / "vocab.txt"
        if not config_path.exists() or not vocab_path.exists():
            return False
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        model_type = config.get("model_type")
        if model_type not in {None, "bert"}:
            return False
        return all(
            key in config for key in ("hidden_size", "num_hidden_layers", "num_attention_heads")
        )

    @staticmethod
    def _legacy_bert_checkpoint_has_pretraining_heads(resolved_model: str) -> bool:
        model_path = Path(resolved_model)
        for index_name in ("pytorch_model.bin.index.json", "model.safetensors.index.json"):
            index_path = model_path / index_name
            if not index_path.exists():
                continue
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            weight_map = index.get("weight_map", {})
            if any(str(key).startswith("cls.") for key in weight_map):
                return True

        safetensors_path = model_path / "model.safetensors"
        if safetensors_path.exists():
            try:
                from safetensors import safe_open

                with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
                    return any(str(key).startswith("cls.") for key in handle.keys())
            except Exception:
                return False

        bin_path = model_path / "pytorch_model.bin"
        if not bin_path.exists():
            return False
        try:
            try:
                state = torch.load(bin_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(bin_path, map_location="cpu")
        except Exception:
            return False
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            return False
        return any(str(key).startswith("cls.") for key in state)

    @staticmethod
    def _huggingface_offline() -> bool:
        for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
            value = os.environ.get(name)
            if value and value.lower() in {"1", "true", "yes", "on"}:
                return True
        return False

    def _resolve_sentence_encoder_path(self) -> str:
        if self._resolved_sentence_encoder_path is not None:
            return self._resolved_sentence_encoder_path

        model_path = Path(str(self._sentence_encoder_model)).expanduser()
        if model_path.exists():
            self._resolved_sentence_encoder_path = str(model_path)
            return self._resolved_sentence_encoder_path

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to resolve legacy BERT checkpoints"
            ) from exc

        self._resolved_sentence_encoder_path = snapshot_download(
            str(self._sentence_encoder_model),
            local_files_only=self._huggingface_offline(),
        )
        return self._resolved_sentence_encoder_path

    def _effective_sentence_encoder_backend(self) -> str:
        if self._hash_backend:
            return "hash"
        if self._sentence_pooling == "token_attention":
            return "transformers"
        if self._sentence_encoder_backend != "auto":
            backend = self._sentence_encoder_backend
        elif (
            self._freeze
            and self._trainable_sentence_encoder_layers == 0
            and self._prefers_sentence_transformers()
        ):
            backend = "sentence_transformers"
        else:
            backend = "transformers"
        if backend == "sentence_transformers" and (
            not self._freeze or self._trainable_sentence_encoder_layers > 0
        ):
            logger.info(
                "Using transformers backend because trainable sentence encoder parameters were requested"
            )
            return "transformers"
        return backend

    def _prefers_sentence_transformers(self) -> bool:
        model_name = str(self._sentence_encoder_model).lower()
        return model_name.startswith("sentence-transformers/") or (
            "qwen" in model_name and "embedding" in model_name
        )

    def _effective_sentence_pooling(self) -> str:
        if self._hash_backend:
            return "hash"
        if self._sentence_pooling != "auto":
            return self._sentence_pooling
        model_name = str(self._sentence_encoder_model).lower()
        if "qwen" in model_name and "embedding" in model_name:
            return "last"
        return "cls"

    def _ensure_token_pooling_initialized(self) -> None:
        if self._effective_sentence_pooling() != "token_attention":
            return
        if self._sentence_dim is None:
            raise RuntimeError("token_attention pooling requires an initialized encoder")
        if self._role_attention:
            if self._w_token_pooling is None:
                self._w_token_pooling = MultiHeadGatedAttentionPooling(
                    hidden_dim=int(self._sentence_dim),
                    attention_dim=int(self._transformer_dim),
                    num_heads=self._w_attention_heads,
                ).to(self._device)
            if self._x_token_pooling is None:
                self._x_token_pooling = MultiHeadGatedAttentionPooling(
                    hidden_dim=int(self._sentence_dim),
                    attention_dim=int(self._transformer_dim),
                    num_heads=self._x_attention_heads,
                ).to(self._device)
            return
        if self._token_pooling is None:
            self._token_pooling = GatedAttentionPooling(
                hidden_dim=int(self._sentence_dim),
                attention_dim=int(self._transformer_dim),
            ).to(self._device)

    def _configure_sentence_encoder_training(self) -> None:
        if self._sentence_encoder is None:
            return
        for param in self._sentence_encoder.parameters():
            param.requires_grad = not self._freeze
        self._encoder_has_trainable_params = not self._freeze
        if self._freeze and self._trainable_sentence_encoder_layers > 0:
            layers = self._find_encoder_layers()
            if layers:
                for layer in layers[-self._trainable_sentence_encoder_layers :]:
                    for param in layer.parameters():
                        param.requires_grad = True
                self._encoder_has_trainable_params = True
                logger.info(
                    "Unfroze last %d/%d chunk-encoder layer(s)",
                    min(self._trainable_sentence_encoder_layers, len(layers)),
                    len(layers),
                )
            else:
                logger.warning(
                    "Could not find encoder layer stack for %s; sentence encoder remains frozen",
                    self._sentence_encoder_model,
                )
        if not self._encoder_has_trainable_params:
            self._sentence_encoder.eval()

    def _find_encoder_layers(self) -> List[nn.Module]:
        if self._sentence_encoder is None:
            return []
        candidate_paths = [
            "encoder.layer",
            "model.layers",
            "layers",
            "transformer.h",
            "model.decoder.layers",
        ]
        for path in candidate_paths:
            module = self._sentence_encoder
            for name in path.split("."):
                if not hasattr(module, name):
                    module = None
                    break
                module = getattr(module, name)
            if isinstance(module, (nn.ModuleList, list, tuple)) and len(module) > 0:
                return list(module)
        return []

    def _total_sentence_encoder_parameter_count(self) -> int:
        if self._sentence_encoder is None:
            return 0
        return sum(param.numel() for param in self._sentence_encoder.parameters())

    def _trainable_sentence_encoder_parameter_count(self) -> int:
        if self._sentence_encoder is None:
            return 0
        return sum(
            param.numel() for param in self._sentence_encoder.parameters() if param.requires_grad
        )

    def sentence_encoder_training_audit(self) -> Dict[str, Any]:
        """Report the initialized encoder's observed trainability state.

        This is deliberately based on the live parameter objects, rather than
        the requested freeze flag.  The production HTR trainer consumes it
        after encoder initialization and before constructing its optimizer.
        """

        parameters = (
            tuple(self._sentence_encoder.parameters()) if self._sentence_encoder is not None else ()
        )
        total_tensors = len(parameters)
        trainable_tensors = sum(bool(param.requires_grad) for param in parameters)
        total_parameters = sum(param.numel() for param in parameters)
        trainable_parameters = sum(param.numel() for param in parameters if param.requires_grad)
        backend = self._effective_sentence_encoder_backend()
        return {
            "schema_version": HTR_SENTENCE_ENCODER_TRAINING_AUDIT_SCHEMA,
            "requested_freeze_sentence_encoder": bool(self._freeze),
            "encoder_initialized": bool(self._encoder_initialized),
            "effective_backend": backend,
            "sentence_encoder_present": self._sentence_encoder is not None,
            "sentence_encoder_parameter_tensors": int(total_tensors),
            "trainable_sentence_encoder_parameter_tensors": int(trainable_tensors),
            "sentence_encoder_parameters": int(total_parameters),
            "trainable_sentence_encoder_parameters": int(trainable_parameters),
            "all_sentence_encoder_parameters_trainable": bool(
                total_tensors > 0
                and trainable_tensors == total_tensors
                and trainable_parameters == total_parameters
            ),
            "hash_backend_without_sentence_encoder": bool(
                backend == "hash" and self._sentence_encoder is None
            ),
        }

    def _hash_chunk_embedding(self, chunk: str) -> torch.Tensor:
        vec = torch.zeros(self._hash_embedding_dim, dtype=torch.float32, device=self._device)
        for token in re.findall(r"\w+", chunk.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            idx = value % self._hash_embedding_dim
            sign = 1.0 if ((value >> 8) & 1) else -1.0
            vec[idx] += sign
        norm = torch.linalg.vector_norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def _last_token_pool(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_state[:, -1]
        sequence_lengths = attention_mask.sum(dim=1).long().clamp(min=1) - 1
        batch_size = last_hidden_state.shape[0]
        return last_hidden_state[
            torch.arange(batch_size, device=last_hidden_state.device),
            sequence_lengths,
        ]

    def _pool_sentence_output(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        pooling = self._effective_sentence_pooling()
        if self._role_attention and pooling == "token_attention":
            self._ensure_token_pooling_initialized()
            if self._w_token_pooling is None or self._x_token_pooling is None:
                raise RuntimeError("role token_attention pooling was not initialized")
            w_by_head, w_token_weights = self._w_token_pooling(
                last_hidden_state,
                attention_mask=attention_mask.to(last_hidden_state.dtype),
            )
            x_by_head, x_token_weights = self._x_token_pooling(
                last_hidden_state,
                attention_mask=attention_mask.to(last_hidden_state.dtype),
            )
            w_pooled = w_by_head.mean(dim=1)
            x_pooled = x_by_head.mean(dim=1)
            if self._normalize_sentence_embeddings:
                w_pooled = F.normalize(w_pooled, p=2, dim=1)
                x_pooled = F.normalize(x_pooled, p=2, dim=1)
            shared = 0.5 * (w_pooled + x_pooled)
            return (
                {
                    "features": shared,
                    "w_features": w_pooled,
                    "x_features": x_pooled,
                },
                {
                    "w": w_token_weights.mean(dim=1),
                    "x": x_token_weights.mean(dim=1),
                    "w_heads": w_token_weights,
                    "x_heads": x_token_weights,
                },
            )
        if pooling == "last":
            pooled = self._last_token_pool(last_hidden_state, attention_mask)
            token_weights = None
        elif pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            token_weights = None
        elif pooling == "token_attention":
            self._ensure_token_pooling_initialized()
            if self._token_pooling is None:
                raise RuntimeError("token_attention pooling was not initialized")
            pooled, token_weights = self._token_pooling(
                last_hidden_state,
                attention_mask=attention_mask.to(last_hidden_state.dtype),
            )
        else:
            pooled = last_hidden_state[:, 0, :]
            token_weights = None
        if self._normalize_sentence_embeddings:
            pooled = F.normalize(pooled, p=2, dim=1)
        if self._role_attention:
            return (
                {
                    "features": pooled,
                    "w_features": pooled,
                    "x_features": pooled,
                },
                None,
            )
        return pooled, token_weights

    def _capture_token_weights(
        self,
        token_weights,
        attention_mask: torch.Tensor,
    ) -> None:
        if not self._capture_token_attention or token_weights is None:
            return
        if isinstance(token_weights, dict):
            mask_cpu = attention_mask.detach().cpu()
            for role in ("w", "x"):
                role_weights = token_weights.get(role)
                if role_weights is None:
                    continue
                weights_cpu = role_weights.detach().cpu()
                target = self._role_token_weight_capture_buffer.setdefault(role, [])
                for row in range(weights_cpu.shape[0]):
                    valid_len = int(mask_cpu[row].sum().item())
                    target.append(weights_cpu[row, :valid_len].clone())
            return
        weights_cpu = token_weights.detach().cpu()
        mask_cpu = attention_mask.detach().cpu()
        for row in range(weights_cpu.shape[0]):
            valid_len = int(mask_cpu[row].sum().item())
            self._token_weight_capture_buffer.append(weights_cpu[row, :valid_len].clone())

    def _encode_chunks(
        self,
        chunks: Sequence[str],
        *,
        return_attention_tensors: bool = False,
    ):
        self._ensure_encoder_initialized()
        if self._hash_backend:
            embeddings = torch.stack([self._hash_chunk_embedding(chunk) for chunk in chunks])
            if self._role_attention:
                embeddings = {
                    "features": embeddings,
                    "w_features": embeddings,
                    "x_features": embeddings,
                }
            if return_attention_tensors:
                return embeddings, None
            return embeddings

        if self._effective_sentence_encoder_backend() == "sentence_transformers":
            embeddings = self._encode_chunks_with_sentence_transformer(chunks)
            if self._role_attention:
                embeddings = {
                    "features": embeddings,
                    "w_features": embeddings,
                    "x_features": embeddings,
                }
            if return_attention_tensors:
                return embeddings, None
            return embeddings
        return self._encode_chunks_with_transformers(
            chunks,
            return_attention_tensors=return_attention_tensors,
        )

    def _encode_chunks_with_sentence_transformer(self, chunks: Sequence[str]) -> torch.Tensor:
        if self._sentence_transformer_encoder is None:
            raise RuntimeError("Sentence-transformer encoder was not initialized")
        chunk_list = list(chunks)
        embeddings_by_batch = []
        for start in range(0, len(chunk_list), self._sentence_encoder_batch_size):
            batch_chunks = chunk_list[start : start + self._sentence_encoder_batch_size]
            embeddings = self._sentence_transformer_encoder.encode(
                batch_chunks,
                batch_size=len(batch_chunks),
                convert_to_numpy=True,
                normalize_embeddings=self._normalize_sentence_embeddings,
                show_progress_bar=False,
            )
            embeddings_by_batch.append(np.asarray(embeddings, dtype=np.float32))
        embeddings_np = np.concatenate(embeddings_by_batch, axis=0)
        return torch.as_tensor(embeddings_np, dtype=torch.float32, device=self._device)

    def _encode_chunks_with_transformers(
        self,
        chunks: Sequence[str],
        *,
        return_attention_tensors: bool = False,
    ):
        chunk_list = list(chunks)
        outputs_by_batch = []
        role_outputs_by_batch: Dict[str, List[torch.Tensor]] = {
            "features": [],
            "w_features": [],
            "x_features": [],
        }
        token_weights_by_batch: List[torch.Tensor] = []
        role_token_weights_by_batch: Dict[str, List[torch.Tensor]] = {"w": [], "x": []}
        input_ids_by_batch: List[torch.Tensor] = []
        attention_mask_by_batch: List[torch.Tensor] = []
        offset_mapping_by_batch: List[torch.Tensor] = []
        for start in range(0, len(chunk_list), self._sentence_encoder_batch_size):
            batch_chunks = chunk_list[start : start + self._sentence_encoder_batch_size]
            encoded = self._tokenize_chunks_for_transformers(
                batch_chunks,
                return_offsets_mapping=return_attention_tensors,
            )
            offset_mapping = encoded.pop("offset_mapping", None)
            input_ids = encoded["input_ids"].to(self._device)
            attention_mask = encoded["attention_mask"].to(self._device)
            if self._sentence_encoder is not None and not self._encoder_has_trainable_params:
                self._sentence_encoder.eval()
            with torch.set_grad_enabled(self._encoder_has_trainable_params):
                if not self._encoder_has_trainable_params:
                    with torch.no_grad():
                        outputs = self._sentence_encoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                else:
                    outputs = self._sentence_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
            pooled, token_weights = self._pool_sentence_output(
                outputs.last_hidden_state,
                attention_mask,
            )
            self._capture_token_weights(token_weights, attention_mask)
            if isinstance(pooled, dict):
                for key, value in pooled.items():
                    role_outputs_by_batch[key].append(value.float())
            else:
                outputs_by_batch.append(pooled.float())
            if return_attention_tensors:
                if isinstance(token_weights, dict):
                    for role in ("w", "x"):
                        role_weights = token_weights.get(role)
                        if role_weights is not None:
                            role_token_weights_by_batch[role].append(role_weights)
                    if token_weights.get("w") is not None and token_weights.get("x") is not None:
                        token_weights_by_batch.append(
                            0.5 * (token_weights["w"] + token_weights["x"])
                        )
                elif token_weights is not None:
                    token_weights_by_batch.append(token_weights)
                input_ids_by_batch.append(input_ids)
                attention_mask_by_batch.append(attention_mask)
                if offset_mapping is None:
                    offset_mapping = torch.full(
                        (input_ids.shape[0], input_ids.shape[1], 2),
                        -1,
                        dtype=torch.long,
                        device=self._device,
                    )
                else:
                    offset_mapping = offset_mapping.to(self._device)
                offset_mapping_by_batch.append(offset_mapping)
        if role_outputs_by_batch["features"]:
            embeddings = {
                key: torch.cat(value, dim=0)
                for key, value in role_outputs_by_batch.items()
                if value
            }
        else:
            embeddings = torch.cat(outputs_by_batch, dim=0)
        if not return_attention_tensors:
            return embeddings
        token_info: Dict[str, Any] = {
            "token_alpha": (
                self._pad_and_cat_token_batches(token_weights_by_batch, pad_value=0.0)
                if token_weights_by_batch
                else None
            ),
            "token_alpha_sources": token_weights_by_batch,
            "role_token_alpha": {
                role: (self._pad_and_cat_token_batches(values, pad_value=0.0) if values else None)
                for role, values in role_token_weights_by_batch.items()
            },
            "input_ids": self._pad_and_cat_token_batches(input_ids_by_batch, pad_value=0),
            "attention_mask": self._pad_and_cat_token_batches(attention_mask_by_batch, pad_value=0),
            "offset_mapping": self._pad_and_cat_offset_batches(offset_mapping_by_batch),
        }
        return embeddings, token_info

    def _encode_prepared_transformer_chunks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_attention_tensors: bool = False,
    ):
        input_ids = input_ids.to(self._device, non_blocking=True)
        attention_mask = attention_mask.to(self._device, non_blocking=True)
        if self._sentence_encoder is not None and not self._encoder_has_trainable_params:
            self._sentence_encoder.eval()
        outputs_by_batch = []
        role_outputs_by_batch: Dict[str, List[torch.Tensor]] = {
            "features": [],
            "w_features": [],
            "x_features": [],
        }
        token_weights_by_batch: List[torch.Tensor] = []
        role_token_weights_by_batch: Dict[str, List[torch.Tensor]] = {"w": [], "x": []}
        input_ids_by_batch: List[torch.Tensor] = []
        attention_mask_by_batch: List[torch.Tensor] = []
        for start in range(0, input_ids.shape[0], self._sentence_encoder_batch_size):
            batch_input_ids = input_ids[start : start + self._sentence_encoder_batch_size]
            batch_attention_mask = attention_mask[start : start + self._sentence_encoder_batch_size]
            with torch.set_grad_enabled(self._encoder_has_trainable_params):
                if not self._encoder_has_trainable_params:
                    with torch.no_grad():
                        outputs = self._sentence_encoder(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                        )
                else:
                    outputs = self._sentence_encoder(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                    )
            pooled, token_weights = self._pool_sentence_output(
                outputs.last_hidden_state,
                batch_attention_mask,
            )
            self._capture_token_weights(token_weights, batch_attention_mask)
            if isinstance(pooled, dict):
                for key, value in pooled.items():
                    role_outputs_by_batch[key].append(value.float())
            else:
                outputs_by_batch.append(pooled.float())
            if return_attention_tensors:
                if isinstance(token_weights, dict):
                    for role in ("w", "x"):
                        role_weights = token_weights.get(role)
                        if role_weights is not None:
                            role_token_weights_by_batch[role].append(role_weights)
                    if token_weights.get("w") is not None and token_weights.get("x") is not None:
                        token_weights_by_batch.append(
                            0.5 * (token_weights["w"] + token_weights["x"])
                        )
                elif token_weights is not None:
                    token_weights_by_batch.append(token_weights)
                input_ids_by_batch.append(batch_input_ids)
                attention_mask_by_batch.append(batch_attention_mask)
        if role_outputs_by_batch["features"]:
            embeddings = {
                key: torch.cat(value, dim=0)
                for key, value in role_outputs_by_batch.items()
                if value
            }
        else:
            embeddings = torch.cat(outputs_by_batch, dim=0)
        if not return_attention_tensors:
            return embeddings
        token_info: Dict[str, Any] = {
            "token_alpha": (
                self._pad_and_cat_token_batches(token_weights_by_batch, pad_value=0.0)
                if token_weights_by_batch
                else None
            ),
            "token_alpha_sources": token_weights_by_batch,
            "role_token_alpha": {
                role: (self._pad_and_cat_token_batches(values, pad_value=0.0) if values else None)
                for role, values in role_token_weights_by_batch.items()
            },
            "input_ids": self._pad_and_cat_token_batches(input_ids_by_batch, pad_value=0),
            "attention_mask": self._pad_and_cat_token_batches(attention_mask_by_batch, pad_value=0),
            "offset_mapping": torch.full(
                (input_ids.shape[0], input_ids.shape[1], 2),
                -1,
                dtype=torch.long,
                device=self._device,
            ),
        }
        return embeddings, token_info

    @staticmethod
    def _pad_and_cat_token_batches(
        batches: Sequence[torch.Tensor],
        *,
        pad_value: float,
    ) -> torch.Tensor:
        if not batches:
            return torch.empty(0)
        max_len = max(int(batch.shape[1]) for batch in batches)
        padded = []
        for batch in batches:
            pad = max_len - int(batch.shape[1])
            if pad > 0:
                padded.append(F.pad(batch, (0, pad), value=pad_value))
            else:
                padded.append(batch)
        return torch.cat(padded, dim=0)

    @staticmethod
    def _pad_and_cat_offset_batches(batches: Sequence[torch.Tensor]) -> torch.Tensor:
        if not batches:
            return torch.empty(0, 0, 2, dtype=torch.long)
        max_len = max(int(batch.shape[1]) for batch in batches)
        padded = []
        for batch in batches:
            pad = max_len - int(batch.shape[1])
            if pad > 0:
                padded.append(F.pad(batch, (0, 0, 0, pad), value=-1))
            else:
                padded.append(batch)
        return torch.cat(padded, dim=0)

    def _tokenize_chunks_for_transformers(
        self,
        chunks: Sequence[str],
        *,
        return_offsets_mapping: bool = False,
    ) -> Dict[str, torch.Tensor]:
        entries = [
            self._tokenize_one_chunk_for_transformers(
                chunk,
                return_offsets_mapping=return_offsets_mapping,
            )
            for chunk in chunks
        ]
        return self._collate_tokenized_chunks(entries)

    def _tokenize_one_chunk_for_transformers(
        self,
        chunk: str,
        *,
        return_offsets_mapping: bool = False,
    ):
        key = str(chunk or "")
        cached = None if return_offsets_mapping else self._tokenization_cache.get(key)
        if cached is not None:
            return cached

        kwargs = {
            "padding": False,
            "truncation": False,
        }
        if return_offsets_mapping and bool(getattr(self._tokenizer, "is_fast", False)):
            kwargs["return_offsets_mapping"] = True
        encoded = self._tokenizer(key, **kwargs)
        input_ids = tuple(int(token_id) for token_id in encoded["input_ids"])
        attention_mask = tuple(int(mask_value) for mask_value in encoded["attention_mask"])
        if len(input_ids) > self._max_chunk_length:
            raise ValueError(
                "HTR tokenizer input exceeds max_chunk_length; semantic truncation is forbidden "
                f"({len(input_ids)} > {self._max_chunk_length})"
            )
        if return_offsets_mapping:
            offsets = encoded.get("offset_mapping")
            if offsets is None:
                offsets = [(-1, -1) for _ in input_ids]
            offset_tuple = tuple((int(start), int(end)) for start, end in offsets)
            return input_ids, attention_mask, offset_tuple
        if len(self._tokenization_cache) < self._tokenization_cache_max_entries:
            self._tokenization_cache[key] = (input_ids, attention_mask)
        return input_ids, attention_mask

    def _collate_tokenized_chunks(
        self,
        entries: Sequence[Any],
    ) -> Dict[str, torch.Tensor]:
        if not entries:
            return _collate_tokenized_chunks(self._tokenizer, entries)
        has_offsets = len(entries[0]) == 3
        if not has_offsets:
            return _collate_tokenized_chunks(self._tokenizer, entries)
        basic_entries = [(input_ids, attention_mask) for input_ids, attention_mask, _ in entries]
        collated = _collate_tokenized_chunks(self._tokenizer, basic_entries)
        max_length = int(collated["input_ids"].shape[1])
        offsets_tensor = torch.full(
            (len(entries), max_length, 2),
            -1,
            dtype=torch.long,
        )
        left_pad = getattr(self._tokenizer, "padding_side", "right") == "left"
        for row, (_input_ids, _attention_mask, offsets) in enumerate(entries):
            offset = max_length - len(offsets) if left_pad else 0
            offsets_tensor[row, offset : offset + len(offsets)] = torch.as_tensor(
                offsets,
                dtype=torch.long,
            )
        collated["offset_mapping"] = offsets_tensor
        return collated

    def _chunks_for_texts(self, texts: Sequence[str]) -> List[List[str]]:
        chunks_by_text: List[List[str]] = []
        for text in texts:
            key = str(text or "")
            chunks = self._chunk_cache.get(key)
            if chunks is None:
                chunks = _htr_chunks_for_text(
                    key,
                    chunk_size_words=self._chunk_size_words,
                    chunk_overlap_words=self._chunk_overlap_words,
                    max_chunks=self._max_chunks,
                    tokenizer=(
                        self._tokenizer
                        if self._effective_sentence_encoder_backend() == "transformers"
                        else None
                    ),
                    max_chunk_length=self._max_chunk_length,
                )
                if len(self._chunk_cache) < self._chunk_cache_max_entries:
                    self._chunk_cache[key] = chunks
            chunks_by_text.append(chunks)
        return chunks_by_text

    def _populate_chunk_cache(self, texts: Sequence[str]) -> None:
        if self._chunk_cache_max_entries <= 0:
            return
        self._chunks_for_texts(texts)

    def make_batch_preprocessor(self) -> HierarchicalTransformerBatchPreprocessor:
        self._ensure_encoder_initialized()
        tokenize_for_transformers = (
            not self._hash_backend and self._effective_sentence_encoder_backend() == "transformers"
        )
        return HierarchicalTransformerBatchPreprocessor(
            tokenizer=self._tokenizer if tokenize_for_transformers else None,
            chunk_size_words=self._chunk_size_words,
            chunk_overlap_words=self._chunk_overlap_words,
            max_chunks=self._max_chunks,
            max_chunk_length=self._max_chunk_length,
            tokenize_for_transformers=tokenize_for_transformers,
            chunk_cache_max_entries=self._chunk_cache_max_entries,
            tokenization_cache_max_entries=self._tokenization_cache_max_entries,
        )

    def prepare_batch(self, texts: Sequence[str]) -> Dict[str, Any]:
        return self.make_batch_preprocessor()(texts)

    def forward_role_features(self, texts_or_batch):
        """Return shared, W-role, and X-role document features."""
        if not self._role_attention:
            features = self.forward(texts_or_batch)
            return {
                "features": features,
                "w_features": features,
                "x_features": features,
            }
        return self.forward(texts_or_batch, return_role_features=True)

    def _pack_chunk_tensor(
        self,
        flat_embeddings: torch.Tensor,
        batch_chunks: Sequence[Sequence[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(batch_chunks)
        max_chunks = max(len(chunks) for chunks in batch_chunks)
        chunk_tensor = torch.zeros(
            batch_size,
            max_chunks,
            self._transformer_dim,
            device=self._device,
        )
        chunk_mask = torch.zeros(batch_size, max_chunks, dtype=torch.bool, device=self._device)
        offset = 0
        for row, chunks in enumerate(batch_chunks):
            count = len(chunks)
            chunk_tensor[row, :count] = flat_embeddings[offset : offset + count]
            chunk_mask[row, :count] = True
            offset += count
        return chunk_tensor, chunk_mask

    def _run_chunk_transformer(
        self,
        chunk_tensor: torch.Tensor,
        chunk_mask: torch.Tensor,
        *,
        return_attention_tensors: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size = chunk_tensor.shape[0]
        pool = self._pool_token.to(self._device).expand(batch_size, 1, -1)
        sequence = torch.cat([pool, chunk_tensor], dim=1)
        sequence = sequence + self._positional_encoding[: sequence.shape[1]].to(self._device)
        sequence_input = sequence
        if return_attention_tensors and sequence_input.requires_grad:
            sequence_input.retain_grad()

        valid_mask = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=torch.bool, device=self._device),
                chunk_mask,
            ],
            dim=1,
        )
        key_padding_mask = ~valid_mask
        attn_weights = None
        for layer in self._transformer_layers:
            sequence, attn_weights = layer(
                sequence,
                key_padding_mask=key_padding_mask,
                return_attention=True,
            )
        return sequence, sequence_input, attn_weights

    def _role_document_features(
        self,
        flat_embeddings_by_role: Dict[str, torch.Tensor],
        batch_chunks: Sequence[Sequence[str]],
        *,
        return_attention_tensors: bool = False,
    ):
        if self._w_chunk_pooling is None or self._x_chunk_pooling is None:
            raise RuntimeError("role chunk attention pooling was not initialized")

        role_outputs: Dict[str, torch.Tensor] = {}
        role_chunk_alpha: Dict[str, torch.Tensor] = {}
        role_sequence_inputs: Dict[str, torch.Tensor] = {}
        chunk_mask = None
        for feature_key, role, pooling in (
            ("w_features", "w", self._w_chunk_pooling),
            ("x_features", "x", self._x_chunk_pooling),
        ):
            flat_embeddings = self._input_projection(flat_embeddings_by_role[feature_key])
            chunk_tensor, chunk_mask = self._pack_chunk_tensor(flat_embeddings, batch_chunks)
            sequence, sequence_input, _ = self._run_chunk_transformer(
                chunk_tensor,
                chunk_mask,
                return_attention_tensors=return_attention_tensors,
            )
            chunk_context = sequence[:, 1 : 1 + chunk_tensor.shape[1], :]
            pooled_by_head, weights_by_head = pooling(
                chunk_context,
                attention_mask=chunk_mask,
            )
            role_hidden = pooled_by_head.mean(dim=1)
            role_outputs[feature_key] = self._output_projection(role_hidden)
            role_chunk_alpha[role] = weights_by_head.mean(dim=1)
            role_sequence_inputs[role] = sequence_input

        role_outputs["features"] = 0.5 * (role_outputs["w_features"] + role_outputs["x_features"])
        self._last_role_chunk_weights = {
            role: weights.detach() for role, weights in role_chunk_alpha.items()
        }
        self._last_chunk_weights = 0.5 * (
            self._last_role_chunk_weights["w"] + self._last_role_chunk_weights["x"]
        )
        if not return_attention_tensors:
            return role_outputs, chunk_mask, None
        return (
            role_outputs,
            chunk_mask,
            {
                "role_chunk_alpha": role_chunk_alpha,
                "role_sequence_input": role_sequence_inputs,
            },
        )

    def forward(
        self,
        texts_or_batch,
        *,
        return_attention_tensors: bool = False,
        return_role_features: bool = False,
    ):
        prepared_batch = texts_or_batch if isinstance(texts_or_batch, dict) else None
        if prepared_batch is not None:
            texts = prepared_batch.get("texts")
            batch_chunks = prepared_batch.get("chunks")
            if texts is None and batch_chunks is None:
                raise ValueError(
                    "hierarchical_transformer batch input requires 'texts' or 'chunks'"
                )
        else:
            texts = texts_or_batch
        if prepared_batch is not None and batch_chunks is not None:
            batch_chunks = [list(chunks) for chunks in batch_chunks]
            texts = list(texts) if texts is not None else ["" for _ in batch_chunks]
        else:
            if isinstance(texts, str):
                texts = [texts]
            texts = list(texts)
            if (
                not self._hash_backend
                and self._effective_sentence_encoder_backend() == "transformers"
            ):
                self._ensure_encoder_initialized()
            batch_chunks = self._chunks_for_texts(texts)
        if not texts:
            features = torch.zeros(0, self._projection_dim, device=self._device)
            output = (
                {"features": features, "w_features": features, "x_features": features}
                if return_role_features
                else features
            )
            if return_attention_tensors:
                return output, {
                    "token_alpha": None,
                    "chunk_alpha": None,
                    "input_ids": None,
                    "attention_mask": None,
                    "offset_mapping": None,
                    "token_alpha_sources": [],
                    "batch_chunks": [],
                }
            return output

        flat_chunks = [chunk for chunks in batch_chunks for chunk in chunks]
        if self._capture_token_attention:
            self._token_weight_capture_buffer = []
            self._role_token_weight_capture_buffer = {"w": [], "x": []}
        token_info = None
        if (
            prepared_batch is not None
            and "chunk_input_ids" in prepared_batch
            and "chunk_attention_mask" in prepared_batch
            and not self._hash_backend
            and self._effective_sentence_encoder_backend() == "transformers"
        ):
            self._ensure_encoder_initialized()
            encoded_chunks = self._encode_prepared_transformer_chunks(
                prepared_batch["chunk_input_ids"],
                prepared_batch["chunk_attention_mask"],
                return_attention_tensors=return_attention_tensors,
            )
        else:
            encoded_chunks = self._encode_chunks(
                flat_chunks,
                return_attention_tensors=return_attention_tensors,
            )
        if return_attention_tensors:
            flat_embeddings, token_info = encoded_chunks
        else:
            flat_embeddings = encoded_chunks
        if self._capture_token_attention:
            self._last_token_weights_by_chunk = list(self._token_weight_capture_buffer)
            self._last_role_token_weights_by_chunk = {
                role: list(values)
                for role, values in self._role_token_weight_capture_buffer.items()
            }
        else:
            self._last_token_weights_by_chunk = []
            self._last_role_token_weights_by_chunk = {}

        if isinstance(flat_embeddings, dict):
            role_outputs, chunk_mask, role_attention_info = self._role_document_features(
                flat_embeddings,
                batch_chunks,
                return_attention_tensors=return_attention_tensors,
            )
            features = role_outputs["features"]
            chunk_alpha = self._last_chunk_weights
            sequence_input = None
            output = role_outputs if return_role_features else features
        else:
            flat_embeddings = self._input_projection(flat_embeddings)
            chunk_tensor, chunk_mask = self._pack_chunk_tensor(flat_embeddings, batch_chunks)
            sequence, sequence_input, attn_weights = self._run_chunk_transformer(
                chunk_tensor,
                chunk_mask,
                return_attention_tensors=return_attention_tensors,
            )
            pool_output = sequence[:, 0, :]
            features = self._output_projection(pool_output)

            if attn_weights is not None:
                pool_attention = attn_weights[:, 0, 1 : 1 + chunk_tensor.shape[1]]
                pool_attention = pool_attention.masked_fill(~chunk_mask, 0.0)
                denom = pool_attention.sum(dim=1, keepdim=True).clamp_min(1e-9)
                chunk_alpha = pool_attention / denom
                self._last_chunk_weights = chunk_alpha.detach()
            else:
                chunk_alpha = None
                self._last_chunk_weights = None
            self._last_role_chunk_weights = {}
            role_attention_info = None
            output = features
        self._last_chunks = batch_chunks
        if return_attention_tensors:
            if token_info is None:
                token_info = {
                    "token_alpha": None,
                    "input_ids": None,
                    "attention_mask": None,
                    "offset_mapping": None,
                    "token_alpha_sources": [],
                }
            token_alpha = token_info.get("token_alpha")
            for token_alpha_source in token_info.get("token_alpha_sources") or []:
                if token_alpha_source is not None and token_alpha_source.requires_grad:
                    token_alpha_source.retain_grad()
            if token_alpha is not None and token_alpha.requires_grad:
                token_alpha.retain_grad()
            if chunk_alpha is not None and chunk_alpha.requires_grad:
                chunk_alpha.retain_grad()
            role_token_alpha = token_info.get("role_token_alpha", {})
            for role_alpha in role_token_alpha.values():
                if role_alpha is not None and role_alpha.requires_grad:
                    role_alpha.retain_grad()
            if role_attention_info is not None:
                for role_alpha in role_attention_info.get("role_chunk_alpha", {}).values():
                    if role_alpha is not None and role_alpha.requires_grad:
                        role_alpha.retain_grad()
            attention_payload = {
                "token_alpha": token_alpha,
                "chunk_alpha": chunk_alpha,
                "input_ids": token_info.get("input_ids"),
                "attention_mask": token_info.get("attention_mask"),
                "offset_mapping": token_info.get("offset_mapping"),
                "token_alpha_sources": token_info.get("token_alpha_sources") or [],
                "role_token_alpha": role_token_alpha,
                "batch_chunks": batch_chunks,
                "sequence_input": sequence_input,
                "chunk_mask": chunk_mask,
            }
            if role_attention_info is not None:
                attention_payload.update(role_attention_info)
            return output, attention_payload
        return output

    def fit_tokenizer(self, texts: List[str]) -> None:
        self._ensure_encoder_initialized()
        self._populate_chunk_cache(texts)
        logger.info(
            "HierarchicalTransformerExtractor ready: backend=%s pooling=%s "
            "role_attention=%s W_heads=%s X_heads=%s device=%s "
            "trainable_params=%s chunk_cache=%s token_cache=%s",
            self._effective_sentence_encoder_backend(),
            self._effective_sentence_pooling(),
            self._role_attention,
            self._w_attention_heads,
            self._x_attention_heads,
            self._device,
            self.get_num_parameters(),
            len(self._chunk_cache),
            len(self._tokenization_cache),
        )

    @staticmethod
    def _attention_role_from_stage(stage: Optional[str]) -> Optional[str]:
        normalized = str(stage or "").strip().lower()
        if normalized in {"nuisance", "w", "confounder", "propensity", "outcome"}:
            return "w"
        if normalized in {"effect_modifier", "effect", "x", "tau", "r_stage"}:
            return "x"
        return None

    def interpret_attention(
        self,
        texts: List[str],
        top_k: int = 5,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.eval()
        previous_capture = self._capture_token_attention
        self._capture_token_attention = True
        with torch.no_grad():
            try:
                self.forward(texts)
            finally:
                self._capture_token_attention = previous_capture
        role = self._attention_role_from_stage(role) or role
        weights = self._last_chunk_weights
        token_weights_by_chunk = self._last_token_weights_by_chunk
        if role in {"w", "x"}:
            role_weights = self._last_role_chunk_weights.get(role)
            if role_weights is not None:
                weights = role_weights
            role_token_weights = self._last_role_token_weights_by_chunk.get(role, [])
            if role_token_weights:
                token_weights_by_chunk = role_token_weights
        results = []
        flat_offsets = []
        offset = 0
        for chunks in self._last_chunks:
            flat_offsets.append(offset)
            offset += len(chunks)
        for row, chunks in enumerate(self._last_chunks):
            row_weights = (
                weights[row, : len(chunks)].cpu().numpy().tolist()
                if weights is not None
                else [0.0 for _ in chunks]
            )
            order = sorted(range(len(chunks)), key=lambda idx: row_weights[idx], reverse=True)
            top = []
            for idx in order[: min(top_k, len(order))]:
                item = {
                    "chunk_index": int(idx),
                    "chunk": chunks[idx],
                    "attention": float(row_weights[idx]),
                }
                flat_idx = flat_offsets[row] + idx
                if flat_idx < len(token_weights_by_chunk):
                    token_spans = self._top_token_spans(
                        chunks[idx],
                        token_weights_by_chunk[flat_idx],
                    )
                    if token_spans:
                        item["top_token_spans"] = token_spans
                        item["attended_token_summary"] = "; ".join(
                            span["text"] for span in token_spans[:6]
                        )
                        item["highlighted_chunk"] = self._highlight_chunk(
                            chunks[idx],
                            token_spans,
                        )
                top.append(item)
            results.append(
                {
                    "chunks": chunks,
                    "chunk_attention_weights": row_weights,
                    "top_chunks": top,
                }
            )
        return results

    def complete_attention_inventory(
        self,
        texts: Sequence[str],
        *,
        role: Optional[str] = None,
        normalization_tolerance: float = 1e-5,
    ) -> Dict[str, Any]:
        """Return every model-native token and chunk attention occurrence.

        This is a lossless evidence boundary, not a post-hoc attribution
        method.  It exposes the learned gated token-pooler weights and the
        final document-transformer pool-token weights that were used by the
        fitted model.  Padding is never part of the returned token domain;
        special tokens remain present and are explicitly marked.
        """

        values = [str(text) for text in texts]
        if not values:
            raise ValueError("complete HTR attention requires at least one note")
        if (
            not math.isfinite(float(normalization_tolerance))
            or float(normalization_tolerance) <= 0.0
        ):
            raise ValueError("attention normalization tolerance must be positive")
        if self._effective_sentence_pooling() != "token_attention":
            raise RuntimeError(
                "complete token evidence requires effective token_attention pooling"
            )
        self.eval()
        with torch.no_grad():
            _features, raw = self.forward(
                values,
                return_attention_tensors=True,
            )
        selected_role = self._attention_role_from_stage(role) or role
        token_alpha = raw.get("token_alpha")
        chunk_alpha = raw.get("chunk_alpha")
        if selected_role in {"w", "x"}:
            selected_token = (raw.get("role_token_alpha") or {}).get(
                selected_role
            )
            selected_chunk = (raw.get("role_chunk_alpha") or {}).get(
                selected_role
            )
            if selected_token is not None:
                token_alpha = selected_token
            if selected_chunk is not None:
                chunk_alpha = selected_chunk
        input_ids = raw.get("input_ids")
        attention_mask = raw.get("attention_mask")
        offset_mapping = raw.get("offset_mapping")
        batch_chunks = raw.get("batch_chunks")
        if (
            token_alpha is None
            or chunk_alpha is None
            or input_ids is None
            or attention_mask is None
            or offset_mapping is None
            or not isinstance(batch_chunks, list)
        ):
            raise RuntimeError(
                "token_attention extractor omitted a native attention tensor"
            )
        if (
            token_alpha.ndim != 2
            or input_ids.shape != token_alpha.shape
            or attention_mask.shape != token_alpha.shape
            or offset_mapping.shape
            != (token_alpha.shape[0], token_alpha.shape[1], 2)
            or chunk_alpha.ndim != 2
        ):
            raise RuntimeError("native HTR attention tensor shapes changed")

        token_values = token_alpha.detach().cpu().to(torch.float64)
        chunk_values = chunk_alpha.detach().cpu().to(torch.float64)
        id_values = input_ids.detach().cpu().to(torch.int64)
        mask_values = attention_mask.detach().cpu().to(torch.int64)
        offset_values = offset_mapping.detach().cpu().to(torch.int64)
        flat_chunk_count = sum(len(chunks) for chunks in batch_chunks)
        if flat_chunk_count != int(token_values.shape[0]):
            raise RuntimeError("HTR token tensors changed the flat chunk order")
        if len(batch_chunks) != int(chunk_values.shape[0]):
            raise RuntimeError("HTR chunk tensor changed the note order")

        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("token_attention evidence lacks its fitted tokenizer")
        special_ids = frozenset(
            int(value)
            for value in (getattr(tokenizer, "all_special_ids", None) or ())
        )
        try:
            vocabulary = tokenizer.get_vocab()
        except Exception as exc:
            raise RuntimeError(
                "fitted HTR tokenizer does not expose its vocabulary"
            ) from exc
        vocabulary_rows = sorted(
            (str(token), int(token_id))
            for token, token_id in vocabulary.items()
        )
        tokenizer_identity = {
            "tokenizer_class": type(tokenizer).__name__,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "vocabulary_size": len(vocabulary_rows),
            "vocabulary_sha256": hashlib.sha256(
                json.dumps(
                    vocabulary_rows,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "all_special_ids": sorted(special_ids),
            "padding_side": str(getattr(tokenizer, "padding_side", "right")),
            "pad_token_id": (
                None
                if getattr(tokenizer, "pad_token_id", None) is None
                else int(tokenizer.pad_token_id)
            ),
        }

        notes: List[Dict[str, Any]] = []
        flat_index = 0
        for note_index, chunks in enumerate(batch_chunks):
            chunk_count = len(chunks)
            note_chunk_weights = chunk_values[note_index, :chunk_count]
            if (
                chunk_count < 1
                or not torch.isfinite(note_chunk_weights).all()
                or torch.any(note_chunk_weights < 0)
                or abs(float(note_chunk_weights.sum().item()) - 1.0)
                > float(normalization_tolerance)
            ):
                raise RuntimeError(
                    "document-pool-token attention is not finite and normalized"
                )
            chunk_rows: List[Dict[str, Any]] = []
            for chunk_index, chunk in enumerate(chunks):
                mask = mask_values[flat_index]
                valid_positions = torch.nonzero(mask != 0, as_tuple=False).flatten()
                padding_positions = torch.nonzero(mask == 0, as_tuple=False).flatten()
                if valid_positions.numel() < 1:
                    raise RuntimeError("HTR chunk has an empty token pooling domain")
                weights = token_values[flat_index]
                valid_weights = weights[valid_positions]
                if (
                    not torch.isfinite(valid_weights).all()
                    or torch.any(valid_weights < 0)
                    or abs(float(valid_weights.sum().item()) - 1.0)
                    > float(normalization_tolerance)
                    or (
                        padding_positions.numel()
                        and torch.any(
                            torch.abs(weights[padding_positions])
                            > float(normalization_tolerance)
                        )
                    )
                ):
                    raise RuntimeError(
                        "learned token-pooler attention is not masked and normalized"
                    )
                token_rows: List[Dict[str, Any]] = []
                token_ids = [
                    int(id_values[flat_index, int(position)].item())
                    for position in valid_positions
                ]
                decoded = tokenizer.convert_ids_to_tokens(token_ids)
                if isinstance(decoded, str):
                    decoded = [decoded]
                if not isinstance(decoded, list) or len(decoded) != len(token_ids):
                    raise RuntimeError(
                        "fitted tokenizer did not losslessly decode token IDs"
                    )
                for local_token_position, (
                    tensor_position,
                    token_id,
                    decoded_token,
                ) in enumerate(
                    zip(valid_positions.tolist(), token_ids, decoded, strict=True)
                ):
                    start = int(
                        offset_values[flat_index, int(tensor_position), 0].item()
                    )
                    end = int(
                        offset_values[flat_index, int(tensor_position), 1].item()
                    )
                    is_special = token_id in special_ids
                    if is_special:
                        if start < 0 or end < 0 or end < start:
                            raise RuntimeError(
                                "special-token offsets are not representable"
                            )
                    elif not (0 <= start < end <= len(chunk)):
                        raise RuntimeError(
                            "non-special token offsets do not align to the chunk"
                        )
                    token_rows.append(
                        {
                            "token_position": int(local_token_position),
                            "tensor_position": int(tensor_position),
                            "token_id": token_id,
                            "decoded_token_text": str(decoded_token),
                            "char_start": start,
                            "char_end": end,
                            "is_special_token": bool(is_special),
                            "is_padding": False,
                            "token_attention": float(
                                weights[int(tensor_position)].item()
                            ),
                        }
                    )
                chunk_rows.append(
                    {
                        "chunk_index": int(chunk_index),
                        "chunk_text": str(chunk),
                        "chunk_attention": float(
                            note_chunk_weights[chunk_index].item()
                        ),
                        "tokens": token_rows,
                        "padding_positions_excluded": int(
                            padding_positions.numel()
                        ),
                    }
                )
                flat_index += 1
            notes.append(
                {
                    "note_index": int(note_index),
                    "chunks": chunk_rows,
                }
            )
        if flat_index != flat_chunk_count:
            raise RuntimeError("HTR complete attention traversal omitted a chunk")
        return {
            "schema_version": "complete_htr_native_attention_inventory_v1",
            "sentence_pooling": str(self._sentence_pooling),
            "effective_sentence_pooling": self._effective_sentence_pooling(),
            "attention_role": selected_role,
            "normalization_tolerance": float(normalization_tolerance),
            "tokenizer_identity": tokenizer_identity,
            "notes": notes,
            "padding_excluded_from_token_normalization": True,
            "special_tokens_retained": True,
            "raw_bert_self_attention_used": False,
            "post_hoc_attribution_used": False,
        }

    def get_attention_evidence(
        self,
        texts: List[str],
        row_ids: Optional[Sequence[Any]] = None,
        fold: Optional[int] = None,
        stage: str = "nuisance",
        top_k: int = 5,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        role = self._attention_role_from_stage(stage)
        interpretations = self.interpret_attention(texts, top_k=top_k, role=role)
        if row_ids is None:
            row_ids = list(range(len(texts)))
        if metadata is None:
            metadata = [{} for _ in texts]
        records: List[Dict[str, Any]] = []
        for row_id, interp, meta in zip(row_ids, interpretations, metadata):
            for item in interp["top_chunks"]:
                record = {
                    "row_id": row_id,
                    "fold": fold,
                    "stage": stage,
                    "attention_role": role,
                    "chunk_index": item["chunk_index"],
                    "chunk_text": item["chunk"],
                    "attention": item["attention"],
                }
                if item.get("top_token_spans"):
                    record["top_token_spans_json"] = json.dumps(
                        item["top_token_spans"],
                        ensure_ascii=False,
                    )
                    record["attended_token_summary"] = item.get(
                        "attended_token_summary",
                        "",
                    )
                    record["highlighted_chunk_text"] = item.get(
                        "highlighted_chunk",
                        item["chunk"],
                    )
                record.update(meta)
                records.append(record)
        return records

    def _top_token_spans(
        self,
        chunk: str,
        token_weights: torch.Tensor,
        top_n: int = 8,
    ) -> List[Dict[str, Any]]:
        if self._tokenizer is None or token_weights is None:
            return []
        try:
            encoded = self._tokenizer(
                chunk,
                padding=False,
                truncation=False,
                return_offsets_mapping=True,
            )
        except Exception:
            return self._top_token_strings(token_weights, chunk, top_n=top_n)
        input_ids = encoded.get("input_ids") or []
        if len(input_ids) > self._max_chunk_length:
            raise ValueError(
                "HTR evidence tokenizer input exceeds max_chunk_length; "
                "semantic truncation is forbidden "
                f"({len(input_ids)} > {self._max_chunk_length})"
            )
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            return self._top_token_strings(token_weights, chunk, top_n=top_n)
        words = [
            (match.group(0), int(match.start()), int(match.end()))
            for match in re.finditer(r"\S+", str(chunk or ""))
        ]
        if not words:
            return []
        word_scores = [0.0 for _ in words]
        word_token_scores: List[List[float]] = [[] for _ in words]
        max_len = min(len(offsets), int(token_weights.numel()))
        weights = token_weights.detach().cpu().numpy().tolist()
        special_ids = set(getattr(self._tokenizer, "all_special_ids", []) or [])
        for token_idx in range(max_len):
            start, end = offsets[token_idx]
            start = int(start)
            end = int(end)
            if end <= start:
                continue
            if token_idx < len(input_ids) and int(input_ids[token_idx]) in special_ids:
                continue
            token_text = str(chunk[start:end])
            if not re.search(r"[A-Za-z0-9]", token_text):
                continue
            word_idx = _find_overlapping_word(words, start, end)
            if word_idx is None:
                continue
            score = float(weights[token_idx])
            word_scores[word_idx] += score
            word_token_scores[word_idx].append(score)
        order = sorted(range(len(words)), key=lambda idx: word_scores[idx], reverse=True)
        spans: List[Dict[str, Any]] = []
        seen_text = set()
        for word_idx in order:
            if word_scores[word_idx] <= 0.0:
                break
            left = max(0, word_idx - 2)
            right = min(len(words), word_idx + 3)
            phrase_start = words[left][1]
            phrase_end = words[right - 1][2]
            text = str(chunk[phrase_start:phrase_end]).strip()
            if not text or text in seen_text:
                continue
            seen_text.add(text)
            spans.append(
                {
                    "text": text,
                    "focus_token": words[word_idx][0],
                    "token_attention": float(max(word_token_scores[word_idx] or [0.0])),
                    "salience": float(word_scores[word_idx]),
                    "char_start": int(phrase_start),
                    "char_end": int(phrase_end),
                }
            )
            if len(spans) >= top_n:
                break
        return spans

    def _top_token_strings(
        self,
        token_weights: torch.Tensor,
        chunk: str,
        top_n: int = 8,
    ) -> List[Dict[str, Any]]:
        try:
            encoded = self._tokenizer(
                chunk,
                padding=False,
                truncation=False,
            )
            input_ids = encoded.get("input_ids") or []
        except Exception:
            return []
        if len(input_ids) > self._max_chunk_length:
            raise ValueError(
                "HTR evidence tokenizer input exceeds max_chunk_length; "
                "semantic truncation is forbidden "
                f"({len(input_ids)} > {self._max_chunk_length})"
            )
        try:
            tokens = self._tokenizer.convert_ids_to_tokens(input_ids)
        except Exception:
            return []
        weights = token_weights.detach().cpu().numpy().tolist()
        special_tokens = set(getattr(self._tokenizer, "all_special_tokens", []) or [])
        candidates = []
        for idx, token in enumerate(tokens[: len(weights)]):
            token_text = str(token)
            if token_text in special_tokens:
                continue
            token_text = token_text.replace("##", "").strip()
            if not re.search(r"[A-Za-z0-9]", token_text):
                continue
            candidates.append((float(weights[idx]), token_text))
        candidates.sort(reverse=True)
        spans = []
        seen = set()
        for score, token_text in candidates:
            if token_text in seen:
                continue
            seen.add(token_text)
            spans.append(
                {
                    "text": token_text,
                    "focus_token": token_text,
                    "token_attention": float(score),
                    "salience": float(score),
                }
            )
            if len(spans) >= top_n:
                break
        return spans

    @staticmethod
    def _highlight_chunk(chunk: str, spans: Sequence[Dict[str, Any]]) -> str:
        intervals = []
        for span in spans:
            if "char_start" not in span or "char_end" not in span:
                continue
            start = int(span["char_start"])
            end = int(span["char_end"])
            if end <= start:
                continue
            intervals.append((start, end))
        intervals.sort()
        selected = []
        last_end = -1
        for start, end in intervals:
            if start < last_end:
                continue
            selected.append((start, end))
            last_end = end
            if len(selected) >= 5:
                break
        if not selected:
            return chunk
        pieces = []
        cursor = 0
        for start, end in selected:
            pieces.append(chunk[cursor:start])
            pieces.append("[[")
            pieces.append(chunk[start:end])
            pieces.append("]]")
            cursor = end
        pieces.append(chunk[cursor:])
        return "".join(pieces)

    def get_attention_weights(self, texts: List[str]) -> Dict[str, Any]:
        return {
            "interpretations": self.interpret_attention(texts, top_k=self._max_chunks),
            "num_layers": self._num_layers,
            "num_heads": self._num_heads,
            "model": self._sentence_encoder_model,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "extractor_type": "hierarchical_transformer",
            "sentence_encoder_model": self._sentence_encoder_model,
            "freeze_sentence_encoder": self._freeze,
            "chunk_size_words": self._chunk_size_words,
            "chunk_overlap_words": self._chunk_overlap_words,
            "max_chunks": self._max_chunks,
            "max_chunk_length": self._max_chunk_length,
            "sentence_encoder_batch_size": self._sentence_encoder_batch_size,
            "sentence_encoder_backend": self._sentence_encoder_backend,
            "effective_sentence_encoder_backend": self._effective_sentence_encoder_backend(),
            "sentence_pooling": self._sentence_pooling,
            "effective_sentence_pooling": self._effective_sentence_pooling(),
            "normalize_sentence_embeddings": self._normalize_sentence_embeddings,
            "trainable_sentence_encoder_layers": self._trainable_sentence_encoder_layers,
            "trainable_sentence_encoder_params": (
                self._trainable_sentence_encoder_parameter_count()
            ),
            "num_transformer_layers": self._num_layers,
            "num_attention_heads": self._num_heads,
            "transformer_dim": self._transformer_dim,
            "transformer_dropout": self._dropout,
            "projection_dim": self._projection_dim,
            "hash_embedding_dim": self._hash_embedding_dim,
            "role_attention": self._role_attention,
            "w_attention_heads": self._w_attention_heads,
            "x_attention_heads": self._x_attention_heads,
            "output_dim": self._projection_dim,
        }

    def get_num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def to(self, device):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)

    def train(self, mode: bool = True):
        result = super().train(mode)
        if self._sentence_encoder is not None and not self._encoder_has_trainable_params:
            self._sentence_encoder.eval()
        return result
