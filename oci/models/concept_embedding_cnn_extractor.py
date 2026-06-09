"""Concept-initialized CNN over sentence-transformer text chunks."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .concept_embedding_cache import load_sentence_transformer
from .concept_embedding_utils import chunk_text_words


logger = logging.getLogger(__name__)


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


class ConceptEmbeddingCNNExtractor(nn.Module):
    """CNN feature extractor initialized from explicit concept embeddings.

    The v1 design uses sentence-transformer embeddings for short text chunks.
    We considered token-level contextual embeddings with variable-width concept
    kernels for a later version; sentence chunks are the first implementation
    because they make the concept filters efficient and interpretable as soft
    semantic chunk detectors.
    """

    def __init__(
        self,
        sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size_words: int = 64,
        chunk_overlap_words: int = 16,
        max_chunks: int = 128,
        confounder_concepts: Optional[List[str]] = None,
        effect_modifier_concepts: Optional[List[str]] = None,
        random_features: int = 0,
        random_confounder_features: Optional[int] = None,
        random_modifier_features: Optional[int] = None,
        kernel_role: str = "combined",
        projection_dim: int = 128,
        dropout: float = 0.1,
        anchor_weight: float = 0.01,
        cached_embedding_dim: int = 0,
        normalize_embeddings: bool = True,
        random_state: int = 42,
        device: Optional[torch.device] = None,
        sentence_encoder: Optional[Any] = None,
    ):
        super().__init__()
        if kernel_role not in {"combined", "confounder", "effect_modifier"}:
            raise ValueError(
                "kernel_role must be one of combined, confounder, effect_modifier"
            )
        if projection_dim < 1:
            raise ValueError("projection_dim must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self._device = device or torch.device("cpu")
        self._sentence_model_name = sentence_model_name
        self._chunk_size_words = chunk_size_words
        self._chunk_overlap_words = chunk_overlap_words
        self._max_chunks = max_chunks
        self._confounder_concepts = _dedupe_texts(confounder_concepts or [])
        self._effect_modifier_concepts = _dedupe_texts(effect_modifier_concepts or [])
        self._random_features = int(random_features)
        self._random_confounder_features = random_confounder_features
        self._random_modifier_features = random_modifier_features
        self._kernel_role = kernel_role
        self._projection_dim = projection_dim
        self._dropout = dropout
        self._anchor_weight = float(anchor_weight)
        self._normalize_embeddings = normalize_embeddings
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

        concept_texts, n_random = self._select_concepts_and_random_count()
        concept_embeddings = self._embed_concepts(concept_texts)
        random_embeddings = self._random_kernel_matrix(n_random, embedding_dim)

        if concept_embeddings.size and random_embeddings.size:
            initial = np.vstack([concept_embeddings, random_embeddings])
        elif concept_embeddings.size:
            initial = concept_embeddings
        elif random_embeddings.size:
            initial = random_embeddings
        else:
            raise ValueError(
                "ConceptEmbeddingCNNExtractor requires at least one concept or "
                "one random feature."
            )

        n_features = int(initial.shape[0])
        self._num_concept_features = len(concept_texts)
        self._num_random_features = n_random
        self._num_features = n_features
        self._concept_texts = concept_texts

        self._concept_conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=n_features,
            kernel_size=1,
        )
        with torch.no_grad():
            weight = torch.as_tensor(initial[:, :, None], dtype=torch.float32)
            self._concept_conv.weight.copy_(weight)
            self._concept_conv.bias.zero_()

        anchor_target = np.zeros((n_features, embedding_dim, 1), dtype=np.float32)
        anchor_mask = np.zeros((n_features, 1, 1), dtype=np.float32)
        if self._num_concept_features:
            anchor_target[: self._num_concept_features, :, 0] = concept_embeddings
            anchor_mask[: self._num_concept_features, :, :] = 1.0
        self.register_buffer(
            "_anchor_target",
            torch.as_tensor(anchor_target, dtype=torch.float32),
        )
        self.register_buffer(
            "_anchor_mask",
            torch.as_tensor(anchor_mask, dtype=torch.float32),
        )

        pooled_dim = 2 * n_features
        self._output_dim = projection_dim
        self._projection = nn.Sequential(
            nn.Linear(pooled_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        self.to(self._device)
        logger.info(
            "ConceptEmbeddingCNNExtractor initialized: model=%s, role=%s, "
            "concepts=%d, random=%d, embedding_dim=%d, output_dim=%d",
            sentence_model_name,
            kernel_role,
            self._num_concept_features,
            self._num_random_features,
            embedding_dim,
            projection_dim,
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def hidden_size(self) -> int:
        return self._embedding_dim

    def fit_tokenizer(self, texts: List[str]) -> None:
        """No-op; sentence-transformer tokenization is fixed."""
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

    def _select_concepts_and_random_count(self) -> Tuple[List[str], int]:
        if self._kernel_role == "confounder":
            n_random = (
                self._random_features
                if self._random_confounder_features is None
                else int(self._random_confounder_features)
            )
            return list(self._confounder_concepts), n_random
        if self._kernel_role == "effect_modifier":
            n_random = (
                self._random_features
                if self._random_modifier_features is None
                else int(self._random_modifier_features)
            )
            return list(self._effect_modifier_concepts), n_random
        n_random = int(self._random_features)
        return (
            _dedupe_texts([*self._confounder_concepts, *self._effect_modifier_concepts]),
            n_random,
        )

    def _embed_concepts(self, concept_texts: List[str]) -> np.ndarray:
        if not concept_texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)
        embeddings = self._encode_texts(concept_texts, batch_size=min(128, len(concept_texts)))
        if embeddings.shape[1] != self._embedding_dim:
            raise ValueError(
                "Concept embedding dimension does not match cached embedding "
                f"dimension: {embeddings.shape[1]} != {self._embedding_dim}"
            )
        if self._normalize_embeddings:
            denom = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(denom, 1e-12)
        return embeddings.astype(np.float32)

    def _random_kernel_matrix(self, n_random: int, embedding_dim: int) -> np.ndarray:
        if n_random <= 0:
            return np.zeros((0, embedding_dim), dtype=np.float32)
        seed = self._random_state + {
            "combined": 0,
            "confounder": 10_000,
            "effect_modifier": 20_000,
        }[self._kernel_role]
        rng = np.random.RandomState(seed)
        values = rng.normal(size=(n_random, embedding_dim)).astype(np.float32)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        return values

    def _chunks_to_tensor(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
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
        )

    def _extract_chunk_embeddings(self, texts_or_batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(texts_or_batch, dict) and "cached_hidden_states" in texts_or_batch:
            embeddings = texts_or_batch["cached_hidden_states"].to(self._device).float()
            mask = texts_or_batch["cached_attention_mask"].to(self._device).float()
            return embeddings, mask
        if isinstance(texts_or_batch, dict):
            texts = texts_or_batch.get("texts", [])
        else:
            texts = texts_or_batch
        return self._chunks_to_tensor(list(texts))

    def forward(self, texts_or_batch) -> torch.Tensor:
        chunk_embeddings, chunk_mask = self._extract_chunk_embeddings(texts_or_batch)
        if chunk_embeddings.shape[-1] != self._embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self._embedding_dim}, got "
                f"{chunk_embeddings.shape[-1]}"
            )
        if self._normalize_embeddings:
            chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=-1)

        responses = self._concept_conv(chunk_embeddings.transpose(1, 2))
        mask = chunk_mask[:, None, :].clamp(0, 1)
        masked = responses.masked_fill(mask <= 0, -1e9)
        max_pooled = masked.max(dim=-1).values
        max_pooled = torch.where(
            torch.isfinite(max_pooled),
            max_pooled,
            torch.zeros_like(max_pooled),
        )
        summed = (responses * mask).sum(dim=-1)
        counts = mask.sum(dim=-1).clamp_min(1.0)
        mean_pooled = summed / counts
        pooled = torch.cat([max_pooled, mean_pooled], dim=1)
        self._last_response_maps = responses.detach()
        self._last_chunk_mask = chunk_mask.detach()
        return self._projection(pooled)

    def compute_anchor_loss(self) -> torch.Tensor:
        if self._anchor_weight <= 0 or self._num_concept_features <= 0:
            return torch.tensor(0.0, device=self._concept_conv.weight.device)
        diff = (self._concept_conv.weight - self._anchor_target) * self._anchor_mask
        denom = self._anchor_mask.sum().clamp_min(1.0) * self._embedding_dim
        return self._anchor_weight * (diff.pow(2).sum() / denom)

    def get_state(self) -> Dict[str, Any]:
        return {
            "extractor_type": "concept_embedding_cnn",
            "sentence_model_name": self._sentence_model_name,
            "chunk_size_words": self._chunk_size_words,
            "chunk_overlap_words": self._chunk_overlap_words,
            "max_chunks": self._max_chunks,
            "confounder_concepts": self._confounder_concepts,
            "effect_modifier_concepts": self._effect_modifier_concepts,
            "random_features": self._random_features,
            "random_confounder_features": self._random_confounder_features,
            "random_modifier_features": self._random_modifier_features,
            "kernel_role": self._kernel_role,
            "embedding_dim": self._embedding_dim,
            "num_concept_features": self._num_concept_features,
            "num_random_features": self._num_random_features,
            "projection_dim": self._projection_dim,
            "dropout": self._dropout,
            "anchor_weight": self._anchor_weight,
            "output_dim": self._output_dim,
        }

    def get_num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def to(self, device):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)
