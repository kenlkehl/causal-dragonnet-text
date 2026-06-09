"""Seeded and free slot extractor over cached sentence chunk embeddings."""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .concept_embedding_cache import load_sentence_transformer
from .concept_embedding_utils import chunk_text_words


logger = logging.getLogger(__name__)


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_PERCENT_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*%")
_WORD_RE = re.compile(r"[a-z]+")

VALUE_FEATURE_NAMES = [
    "bias",
    "has_number",
    "number_count",
    "first_number",
    "max_number",
    "min_number",
    "mean_number",
    "has_percent",
    "first_percent",
    "has_high_comparator",
    "has_low_comparator",
    "has_negation",
    "has_positive",
    "has_unknown",
    "has_none",
    "has_yes",
    "has_no",
    "has_male",
    "has_female",
    "has_time_unit",
    "has_lab_unit",
    "has_status_word",
]


def _dedupe_texts(texts: Sequence[str]) -> List[str]:
    result = []
    seen = set()
    for text in texts:
        value = str(text).strip()
        if not value or value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def _scaled_number(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    scaled = math.copysign(math.log1p(abs(float(value))) / 10.0, float(value))
    return float(max(-1.0, min(1.0, scaled)))


def _contains_any(words: set, options: Sequence[str]) -> float:
    return 1.0 if any(option in words for option in options) else 0.0


def value_features_for_chunk(text: str) -> np.ndarray:
    """Return generic value-bearing text features for one chunk."""
    lowered = str(text).lower()
    words = set(_WORD_RE.findall(lowered))
    numbers = [float(match.group(0)) for match in _NUMBER_RE.finditer(lowered)]
    percents = [float(match.group(1)) for match in _PERCENT_RE.finditer(lowered)]

    if numbers:
        first_number = _scaled_number(numbers[0])
        max_number = _scaled_number(max(numbers))
        min_number = _scaled_number(min(numbers))
        mean_number = _scaled_number(float(np.mean(numbers)))
    else:
        first_number = max_number = min_number = mean_number = 0.0

    first_percent = 0.0
    if percents:
        first_percent = max(-1.0, min(1.0, percents[0] / 100.0))

    high_comparator = (
        ">=" in lowered
        or ">" in lowered
        or _contains_any(words, ["above", "over", "greater", "high", "elevated", "positive"])
    )
    low_comparator = (
        "<=" in lowered
        or "<" in lowered
        or _contains_any(words, ["below", "under", "less", "low", "reduced", "negative"])
    )

    features = [
        1.0,
        1.0 if numbers else 0.0,
        min(len(numbers), 10) / 10.0,
        first_number,
        max_number,
        min_number,
        mean_number,
        1.0 if percents else 0.0,
        first_percent,
        1.0 if high_comparator else 0.0,
        1.0 if low_comparator else 0.0,
        _contains_any(words, ["no", "not", "none", "without", "negative", "denies", "absent"]),
        _contains_any(words, ["yes", "positive", "present", "detected", "elevated"]),
        _contains_any(words, ["unknown", "unclear", "indeterminate", "pending", "missing"]),
        _contains_any(words, ["none", "absent", "nil"]),
        _contains_any(words, ["yes"]),
        _contains_any(words, ["no"]),
        _contains_any(words, ["male", "man", "m"]),
        _contains_any(words, ["female", "woman", "f"]),
        _contains_any(words, ["year", "years", "month", "months", "week", "weeks", "day", "days"]),
        _contains_any(words, ["mg", "ml", "dl", "mmol", "cells", "ratio", "percent"]),
        _contains_any(words, ["status", "stage", "grade", "score", "level", "class"]),
    ]
    return np.asarray(features, dtype=np.float32)


class SlotValueDiscoveryExtractor(nn.Module):
    """Differentiable seeded/free slot extractor for concept discovery.

    Queries attend softly over sentence chunk embeddings. Seeded queries are
    initialized from candidate concept descriptions, while free queries are
    random learned slots. Each slot carries attended semantic context plus a
    small generic value summary extracted from the attended chunks.
    """

    def __init__(
        self,
        sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size_words: int = 64,
        chunk_overlap_words: int = 16,
        max_chunks: int = 128,
        confounder_concepts: Optional[List[str]] = None,
        effect_modifier_concepts: Optional[List[str]] = None,
        num_free_slots: int = 16,
        slot_dim: int = 128,
        num_value_prototypes: int = 4,
        dropout: float = 0.1,
        anchor_weight: float = 0.01,
        cached_embedding_dim: int = 0,
        normalize_embeddings: bool = True,
        attention_temperature: float = 0.1,
        attention_entropy_weight: float = 0.0,
        query_diversity_weight: float = 0.0,
        random_state: int = 42,
        device: Optional[torch.device] = None,
        sentence_encoder: Optional[Any] = None,
    ):
        super().__init__()
        if slot_dim < 1:
            raise ValueError("slot_dim must be >= 1")
        if num_free_slots < 0:
            raise ValueError("num_free_slots must be >= 0")
        if num_value_prototypes < 0:
            raise ValueError("num_value_prototypes must be >= 0")
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self._device = device or torch.device("cpu")
        self._sentence_model_name = sentence_model_name
        self._chunk_size_words = int(chunk_size_words)
        self._chunk_overlap_words = int(chunk_overlap_words)
        self._max_chunks = int(max_chunks)
        self._confounder_concepts = _dedupe_texts(confounder_concepts or [])
        self._effect_modifier_concepts = _dedupe_texts(effect_modifier_concepts or [])
        self._seed_concepts = _dedupe_texts(
            [*self._confounder_concepts, *self._effect_modifier_concepts]
        )
        self._num_free_slots = int(num_free_slots)
        self._slot_dim = int(slot_dim)
        self._num_value_prototypes = int(num_value_prototypes)
        self._dropout = float(dropout)
        self._anchor_weight = float(anchor_weight)
        self._normalize_embeddings = bool(normalize_embeddings)
        self._attention_temperature = float(attention_temperature)
        self._attention_entropy_weight = float(attention_entropy_weight)
        self._query_diversity_weight = float(query_diversity_weight)
        self._random_state = int(random_state)
        self._sentence_encoder = sentence_encoder

        if cached_embedding_dim > 0:
            embedding_dim = int(cached_embedding_dim)
        else:
            encoder = self._get_encoder()
            embedding_dim = int(encoder.get_sentence_embedding_dimension() or 0)
            if embedding_dim <= 0:
                probe = self._encode_texts([""], batch_size=1)
                embedding_dim = int(probe.shape[1])
        self._embedding_dim = embedding_dim

        seed_embeddings = self._embed_seed_concepts()
        free_embeddings = self._random_query_matrix(self._num_free_slots, embedding_dim)
        if seed_embeddings.size and free_embeddings.size:
            initial_queries = np.vstack([seed_embeddings, free_embeddings])
        elif seed_embeddings.size:
            initial_queries = seed_embeddings
        elif free_embeddings.size:
            initial_queries = free_embeddings
        else:
            raise ValueError(
                "SlotValueDiscoveryExtractor requires seeded concepts or free slots"
            )

        self._num_seed_slots = int(seed_embeddings.shape[0])
        self._num_slots = int(initial_queries.shape[0])
        self._slot_names = [
            *self._seed_concepts,
            *[f"free_slot_{idx}" for idx in range(self._num_free_slots)],
        ]

        self._queries = nn.Parameter(torch.as_tensor(initial_queries, dtype=torch.float32))

        anchor_target = np.zeros((self._num_slots, embedding_dim), dtype=np.float32)
        anchor_mask = np.zeros((self._num_slots, 1), dtype=np.float32)
        if self._num_seed_slots:
            anchor_target[: self._num_seed_slots] = seed_embeddings
            anchor_mask[: self._num_seed_slots] = 1.0
        self.register_buffer(
            "_anchor_target",
            torch.as_tensor(anchor_target, dtype=torch.float32),
        )
        self.register_buffer(
            "_anchor_mask",
            torch.as_tensor(anchor_mask, dtype=torch.float32),
        )

        value_dim = len(VALUE_FEATURE_NAMES)
        if self._num_value_prototypes > 0:
            rng = np.random.RandomState(self._random_state + 91_337)
            proto = rng.normal(scale=0.05, size=(self._num_value_prototypes, value_dim))
            self._value_prototypes = nn.Parameter(torch.as_tensor(proto, dtype=torch.float32))
        else:
            self.register_parameter("_value_prototypes", None)

        slot_raw_dim = embedding_dim + value_dim + self._num_value_prototypes + 2
        self._slot_projection = nn.Sequential(
            nn.Linear(slot_raw_dim, slot_dim),
            nn.LayerNorm(slot_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(slot_dim, slot_dim),
            nn.LayerNorm(slot_dim),
        )

        self._last_attention: Optional[torch.Tensor] = None
        self._last_scores: Optional[torch.Tensor] = None
        self._last_slot_features: Optional[torch.Tensor] = None
        self._last_value_features: Optional[torch.Tensor] = None
        self._last_chunk_mask: Optional[torch.Tensor] = None

        self.to(self._device)
        logger.info(
            "SlotValueDiscoveryExtractor initialized: model=%s, seed_slots=%d, "
            "free_slots=%d, embedding_dim=%d, slot_dim=%d",
            sentence_model_name,
            self._num_seed_slots,
            self._num_free_slots,
            embedding_dim,
            slot_dim,
        )

    @property
    def output_dim(self) -> int:
        return self._num_slots * self._slot_dim

    @property
    def slot_feature_dim(self) -> int:
        return self._slot_dim

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def hidden_size(self) -> int:
        return self._embedding_dim

    @property
    def value_feature_names(self) -> List[str]:
        return list(VALUE_FEATURE_NAMES)

    def fit_tokenizer(self, texts: List[str]) -> None:
        del texts

    def _get_encoder(self):
        if self._sentence_encoder is None:
            self._sentence_encoder = load_sentence_transformer(
                self._sentence_model_name,
                device=self._device,
            )
        return self._sentence_encoder

    def _encode_texts(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        encoder = self._get_encoder()
        embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise RuntimeError(f"Unexpected sentence embedding shape {embeddings.shape}")
        return embeddings

    def _embed_seed_concepts(self) -> np.ndarray:
        if not self._seed_concepts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)
        embeddings = self._encode_texts(
            self._seed_concepts,
            batch_size=min(128, len(self._seed_concepts)),
        )
        if embeddings.shape[1] != self._embedding_dim:
            raise ValueError(
                "Seed concept embedding dimension does not match cached embedding "
                f"dimension: {embeddings.shape[1]} != {self._embedding_dim}"
            )
        if self._normalize_embeddings:
            denom = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(denom, 1e-12)
        return embeddings.astype(np.float32)

    def _random_query_matrix(self, n_slots: int, embedding_dim: int) -> np.ndarray:
        if n_slots <= 0:
            return np.zeros((0, embedding_dim), dtype=np.float32)
        rng = np.random.RandomState(self._random_state + 71_003)
        values = rng.normal(size=(n_slots, embedding_dim)).astype(np.float32)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        return values

    def _chunks_to_tensor(
        self,
        texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        sample_chunks = [
            chunk_text_words(
                text,
                self._chunk_size_words,
                self._chunk_overlap_words,
                self._max_chunks,
            )
            for text in texts
        ]
        flat_chunks = [chunk for chunks in sample_chunks for chunk in chunks]
        counts = [len(chunks) for chunks in sample_chunks]
        embeddings = self._encode_texts(flat_chunks)
        if embeddings.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Raw chunk embedding dim {embeddings.shape[1]} does not match "
                f"extractor embedding dim {self._embedding_dim}"
            )
        max_chunks = max(counts) if counts else 1
        batch = np.zeros((len(texts), max_chunks, self._embedding_dim), dtype=np.float32)
        mask = np.zeros((len(texts), max_chunks), dtype=np.float32)
        offset = 0
        for i, count in enumerate(counts):
            batch[i, :count] = embeddings[offset:offset + count]
            mask[i, :count] = 1.0
            offset += count
        return (
            torch.as_tensor(batch, dtype=torch.float32, device=self._device),
            torch.as_tensor(mask, dtype=torch.float32, device=self._device),
            sample_chunks,
        )

    def _extract_chunk_embeddings(
        self,
        texts_or_batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        if isinstance(texts_or_batch, dict) and "cached_hidden_states" in texts_or_batch:
            embeddings = texts_or_batch["cached_hidden_states"].to(self._device).float()
            mask = texts_or_batch["cached_attention_mask"].to(self._device).float()
            return embeddings, mask, list(texts_or_batch.get("texts", []))
        if isinstance(texts_or_batch, dict):
            texts = list(texts_or_batch.get("texts", []))
        else:
            texts = list(texts_or_batch)
        embeddings, mask, _ = self._chunks_to_tensor(texts)
        return embeddings, mask, texts

    def _chunk_texts_for_values(self, texts: List[str], max_len: int) -> List[List[str]]:
        sample_chunks = [
            chunk_text_words(
                text,
                self._chunk_size_words,
                self._chunk_overlap_words,
                self._max_chunks,
            )
            for text in texts
        ]
        adjusted = []
        for chunks in sample_chunks:
            row = list(chunks[:max_len])
            if len(row) < max_len:
                row.extend([""] * (max_len - len(row)))
            adjusted.append(row)
        return adjusted

    def _value_features_to_tensor(self, texts: List[str], max_len: int) -> torch.Tensor:
        if not texts:
            return torch.zeros(
                (0, max_len, len(VALUE_FEATURE_NAMES)),
                dtype=torch.float32,
                device=self._device,
            )
        sample_chunks = self._chunk_texts_for_values(texts, max_len)
        values = np.zeros(
            (len(sample_chunks), max_len, len(VALUE_FEATURE_NAMES)),
            dtype=np.float32,
        )
        for i, chunks in enumerate(sample_chunks):
            for j, chunk in enumerate(chunks):
                values[i, j] = value_features_for_chunk(chunk)
        return torch.as_tensor(values, dtype=torch.float32, device=self._device)

    def _compute_slot_features(
        self,
        chunk_embeddings: torch.Tensor,
        chunk_mask: torch.Tensor,
        value_features: torch.Tensor,
    ) -> torch.Tensor:
        if self._normalize_embeddings:
            chunks = F.normalize(chunk_embeddings, p=2, dim=-1)
            queries = F.normalize(self._queries, p=2, dim=-1)
        else:
            chunks = chunk_embeddings
            queries = self._queries

        scores = torch.einsum("bld,sd->bsl", chunks, queries)
        scores = scores / self._attention_temperature
        mask = chunk_mask[:, None, :].clamp(0, 1)
        masked_scores = scores.masked_fill(mask <= 0, -1e9)
        attention = F.softmax(masked_scores, dim=-1)
        attention = attention * mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        attended_semantic = torch.einsum("bsl,bld->bsd", attention, chunk_embeddings)
        attended_values = torch.einsum("bsl,blv->bsv", attention, value_features)

        valid_counts = mask.sum(dim=-1).clamp_min(1.0)
        entropy = -(attention.clamp_min(1e-8).log() * attention).sum(dim=-1)
        entropy = entropy / valid_counts.log().clamp_min(1.0)

        max_scores = masked_scores.max(dim=-1).values
        max_scores = torch.where(
            torch.isfinite(max_scores),
            max_scores,
            torch.zeros_like(max_scores),
        )

        prototype_probs = []
        if self._value_prototypes is not None:
            distances = torch.cdist(
                attended_values.reshape(-1, attended_values.shape[-1]),
                self._value_prototypes,
                p=2,
            )
            prototype_probs = F.softmax(-distances, dim=-1).reshape(
                attended_values.shape[0],
                attended_values.shape[1],
                self._num_value_prototypes,
            )
        else:
            prototype_probs = attended_values.new_zeros(
                attended_values.shape[0],
                attended_values.shape[1],
                0,
            )

        slot_raw = torch.cat(
            [
                attended_semantic,
                attended_values,
                prototype_probs,
                max_scores.unsqueeze(-1),
                entropy.unsqueeze(-1),
            ],
            dim=-1,
        )
        slot_features = self._slot_projection(slot_raw)

        self._last_attention = attention
        self._last_scores = scores.detach()
        self._last_slot_features = slot_features.detach()
        self._last_value_features = value_features.detach()
        self._last_chunk_mask = chunk_mask.detach()

        return slot_features

    def forward(self, texts_or_batch) -> torch.Tensor:
        chunk_embeddings, chunk_mask, texts = self._extract_chunk_embeddings(texts_or_batch)
        if chunk_embeddings.shape[-1] != self._embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self._embedding_dim}, got "
                f"{chunk_embeddings.shape[-1]}"
            )
        if not texts:
            texts = [""] * int(chunk_embeddings.shape[0])
        value_features = self._value_features_to_tensor(
            texts,
            max_len=int(chunk_embeddings.shape[1]),
        )
        slot_features = self._compute_slot_features(
            chunk_embeddings,
            chunk_mask,
            value_features,
        )
        return slot_features.reshape(slot_features.shape[0], -1)

    def compute_anchor_loss(self) -> torch.Tensor:
        if self._anchor_weight <= 0 or self._num_seed_slots <= 0:
            return torch.tensor(0.0, device=self._queries.device)
        diff = (self._queries - self._anchor_target) * self._anchor_mask
        denom = self._anchor_mask.sum().clamp_min(1.0) * self._embedding_dim
        return self._anchor_weight * (diff.pow(2).sum() / denom)

    def compute_regularization_losses(self) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        if self._last_attention is not None and self._attention_entropy_weight > 0:
            attention = self._last_attention
            entropy = -(attention.clamp_min(1e-8).log() * attention).sum(dim=-1)
            losses["slot_attention_entropy_loss"] = (
                self._attention_entropy_weight * entropy.mean()
            )
        if self._query_diversity_weight > 0 and self._num_slots > 1:
            queries = F.normalize(self._queries, p=2, dim=-1)
            sim = queries @ queries.T
            eye = torch.eye(self._num_slots, device=sim.device, dtype=sim.dtype)
            off_diag = sim * (1.0 - eye)
            losses["slot_query_diversity_loss"] = (
                self._query_diversity_weight * off_diag.pow(2).sum()
                / (self._num_slots * (self._num_slots - 1))
            )
        return losses

    def get_attention_diagnostics(self) -> Dict[str, Any]:
        return {
            "slot_names": list(self._slot_names),
            "attention": self._last_attention,
            "scores": self._last_scores,
            "value_features": self._last_value_features,
            "chunk_mask": self._last_chunk_mask,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "extractor_type": "slot_value_discovery",
            "sentence_model_name": self._sentence_model_name,
            "chunk_size_words": self._chunk_size_words,
            "chunk_overlap_words": self._chunk_overlap_words,
            "max_chunks": self._max_chunks,
            "confounder_concepts": self._confounder_concepts,
            "effect_modifier_concepts": self._effect_modifier_concepts,
            "num_seed_slots": self._num_seed_slots,
            "num_free_slots": self._num_free_slots,
            "num_slots": self._num_slots,
            "slot_dim": self._slot_dim,
            "embedding_dim": self._embedding_dim,
            "num_value_prototypes": self._num_value_prototypes,
            "dropout": self._dropout,
            "anchor_weight": self._anchor_weight,
            "attention_temperature": self._attention_temperature,
            "attention_entropy_weight": self._attention_entropy_weight,
            "query_diversity_weight": self._query_diversity_weight,
            "output_dim": self.output_dim,
            "value_feature_names": list(VALUE_FEATURE_NAMES),
        }

    def get_num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def to(self, device):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)
