"""Hierarchical short-chunk transformer extractor.

This revives the older sentence-encoder + transformer-pooling idea and adapts
it to short overlapping word chunks. A pretrained encoder maps each chunk to a
vector, a small transformer with a learnable pool token aggregates chunks, and
attention from the pool token is exported as chunk-level evidence.
"""

import hashlib
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def split_text_into_word_chunks(
    text: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
    max_chunks: int,
) -> List[str]:
    """Split text into short overlapping word chunks.

    If the text is too long for ``max_chunks``, keep the tail of the note. In
    longitudinal clinical notes, the most recent information is usually at the
    end of the concatenated history.
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    if chunk_overlap_words >= chunk_size_words:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
    words = re.findall(r"\S+", str(text or ""))
    if not words:
        return [""]

    stride = chunk_size_words - chunk_overlap_words
    max_window_words = chunk_size_words + (max_chunks - 1) * stride
    if len(words) > max_window_words:
        words = words[-max_window_words:]

    chunks = []
    start = 0
    while start < len(words) and len(chunks) < max_chunks:
        chunk_words = words[start:start + chunk_size_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        start += stride
    return chunks or [" ".join(words[-chunk_size_words:])]


class _InterpretableTransformerLayer(nn.Module):
    """Transformer encoder layer that can return self-attention weights."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_output, attn_weights = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_output))
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
        freeze_sentence_encoder: bool = True,
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
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")

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
        if sentence_pooling not in {"auto", "cls", "last", "mean"}:
            raise ValueError("sentence_pooling must be one of: auto, cls, last, mean")
        if trainable_sentence_encoder_layers < 0:
            raise ValueError("trainable_sentence_encoder_layers must be >= 0")
        self._sentence_encoder_batch_size = int(sentence_encoder_batch_size)
        self._sentence_encoder_backend = sentence_encoder_backend
        self._sentence_pooling = sentence_pooling
        self._normalize_sentence_embeddings = bool(normalize_sentence_embeddings)
        self._trainable_sentence_encoder_layers = int(trainable_sentence_encoder_layers)
        self._encoder_has_trainable_params = False
        self._hash_backend = str(sentence_encoder_model).lower() in {
            "hash",
            "hashed",
            "hashing",
            "test_hash",
        }

        self._tokenizer = None
        self._sentence_encoder = None
        self._sentence_transformer_encoder = None
        self._sentence_dim = self._hash_embedding_dim if self._hash_backend else None
        self._encoder_initialized = self._hash_backend

        self._input_projection = nn.Linear(
            self._hash_embedding_dim if self._hash_backend else transformer_dim,
            transformer_dim,
        )
        if not self._hash_backend:
            # Replaced lazily once the encoder hidden size is known.
            self._input_projection = None

        self._pool_token = nn.Parameter(torch.randn(1, transformer_dim) * 0.02)
        self.register_buffer(
            "_positional_encoding",
            self._make_positional_encoding(max_chunks + 1, transformer_dim),
        )
        self._transformer_layers = nn.ModuleList(
            [
                _InterpretableTransformerLayer(
                    d_model=transformer_dim,
                    nhead=num_attention_heads,
                    dim_feedforward=transformer_dim * 4,
                    dropout=transformer_dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self._output_projection = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(transformer_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self._last_chunks: List[List[str]] = []
        self._last_chunk_weights: Optional[torch.Tensor] = None
        self.to(self._device)

    @property
    def output_dim(self) -> int:
        return self._projection_dim

    @staticmethod
    def _make_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
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
            self._ensure_sentence_transformer_initialized()
        else:
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
            raise ImportError("transformers is required for hierarchical_transformer") from exc

        logger.info("Loading chunk encoder: %s", self._sentence_encoder_model)
        self._tokenizer = self._load_tokenizer(AutoTokenizer)
        if self._effective_sentence_pooling() == "last":
            self._tokenizer.padding_side = "left"
        self._sentence_encoder = AutoModel.from_pretrained(self._sentence_encoder_model)
        self._sentence_encoder = self._sentence_encoder.to(self._device)
        self._sentence_dim = int(self._sentence_encoder.config.hidden_size)
        self._configure_sentence_encoder_training()
        self._input_projection = nn.Linear(self._sentence_dim, self._transformer_dim).to(
            self._device
        )
        logger.info(
            "Chunk encoder initialized with backend=%s pooling=%s hidden_dim=%s "
            "trainable_encoder_params=%s/%s",
            self._effective_sentence_encoder_backend(),
            self._effective_sentence_pooling(),
            self._sentence_dim,
            self._trainable_sentence_encoder_parameter_count(),
            self._total_sentence_encoder_parameter_count(),
        )

    def _load_tokenizer(self, auto_tokenizer_cls):
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
                raise RuntimeError(
                    "Could not load tokenizer for "
                    f"{self._sentence_encoder_model!r}. Install tokenizer conversion "
                    "dependencies with `pip install sentencepiece tiktoken`, or use a "
                    "BERT/WordPiece model with tokenizer files available locally. "
                    f"Fast tokenizer error: {fast_exc}. Slow tokenizer error: {slow_exc}."
                ) from slow_exc

    def _effective_sentence_encoder_backend(self) -> str:
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
        if self._sentence_pooling != "auto":
            return self._sentence_pooling
        model_name = str(self._sentence_encoder_model).lower()
        if "qwen" in model_name and "embedding" in model_name:
            return "last"
        return "cls"

    def _configure_sentence_encoder_training(self) -> None:
        if self._sentence_encoder is None:
            return
        for param in self._sentence_encoder.parameters():
            param.requires_grad = not self._freeze
        self._encoder_has_trainable_params = not self._freeze
        if self._freeze and self._trainable_sentence_encoder_layers > 0:
            layers = self._find_encoder_layers()
            if layers:
                for layer in layers[-self._trainable_sentence_encoder_layers:]:
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
            param.numel()
            for param in self._sentence_encoder.parameters()
            if param.requires_grad
        )

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
    def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
    ) -> torch.Tensor:
        pooling = self._effective_sentence_pooling()
        if pooling == "last":
            pooled = self._last_token_pool(last_hidden_state, attention_mask)
        elif pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = last_hidden_state[:, 0, :]
        if self._normalize_sentence_embeddings:
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled

    def _encode_chunks(self, chunks: Sequence[str]) -> torch.Tensor:
        self._ensure_encoder_initialized()
        if self._hash_backend:
            return torch.stack([self._hash_chunk_embedding(chunk) for chunk in chunks])

        if self._effective_sentence_encoder_backend() == "sentence_transformers":
            return self._encode_chunks_with_sentence_transformer(chunks)
        return self._encode_chunks_with_transformers(chunks)

    def _encode_chunks_with_sentence_transformer(self, chunks: Sequence[str]) -> torch.Tensor:
        if self._sentence_transformer_encoder is None:
            raise RuntimeError("Sentence-transformer encoder was not initialized")
        chunk_list = list(chunks)
        embeddings_by_batch = []
        for start in range(0, len(chunk_list), self._sentence_encoder_batch_size):
            batch_chunks = chunk_list[start:start + self._sentence_encoder_batch_size]
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

    def _encode_chunks_with_transformers(self, chunks: Sequence[str]) -> torch.Tensor:
        chunk_list = list(chunks)
        outputs_by_batch = []
        for start in range(0, len(chunk_list), self._sentence_encoder_batch_size):
            batch_chunks = chunk_list[start:start + self._sentence_encoder_batch_size]
            encoded = self._tokenizer(
                batch_chunks,
                padding=True,
                truncation=True,
                max_length=self._max_chunk_length,
                return_tensors="pt",
            )
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
            outputs_by_batch.append(
                self._pool_sentence_output(outputs.last_hidden_state, attention_mask).float()
            )
        return torch.cat(outputs_by_batch, dim=0)

    def _chunks_for_texts(self, texts: Sequence[str]) -> List[List[str]]:
        return [
            split_text_into_word_chunks(
                text,
                self._chunk_size_words,
                self._chunk_overlap_words,
                self._max_chunks,
            )
            for text in texts
        ]

    def forward(self, texts_or_batch) -> torch.Tensor:
        if isinstance(texts_or_batch, dict):
            texts = texts_or_batch.get("texts")
            if texts is None:
                raise ValueError("hierarchical_transformer batch input requires 'texts'")
        else:
            texts = texts_or_batch
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return torch.zeros(0, self._projection_dim, device=self._device)

        batch_chunks = self._chunks_for_texts(texts)
        flat_chunks = [chunk for chunks in batch_chunks for chunk in chunks]
        flat_embeddings = self._encode_chunks(flat_chunks)
        flat_embeddings = self._input_projection(flat_embeddings)

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
            chunk_tensor[row, :count] = flat_embeddings[offset:offset + count]
            chunk_mask[row, :count] = True
            offset += count

        pool = self._pool_token.to(self._device).expand(batch_size, 1, -1)
        sequence = torch.cat([pool, chunk_tensor], dim=1)
        sequence = sequence + self._positional_encoding[: sequence.shape[1]].to(self._device)

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

        pool_output = sequence[:, 0, :]
        features = self._output_projection(pool_output)

        if attn_weights is not None:
            pool_attention = attn_weights[:, 0, 1: 1 + max_chunks]
            pool_attention = pool_attention.masked_fill(~chunk_mask, 0.0)
            denom = pool_attention.sum(dim=1, keepdim=True).clamp_min(1e-9)
            self._last_chunk_weights = (pool_attention / denom).detach()
        else:
            self._last_chunk_weights = None
        self._last_chunks = batch_chunks
        return features

    def fit_tokenizer(self, texts: List[str]) -> None:
        del texts
        self._ensure_encoder_initialized()
        logger.info(
            "HierarchicalTransformerExtractor ready: backend=%s pooling=%s "
            "trainable_params=%s",
            self._effective_sentence_encoder_backend(),
            self._effective_sentence_pooling(),
            self.get_num_parameters(),
        )

    def interpret_attention(self, texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        self.eval()
        with torch.no_grad():
            self.forward(texts)
        weights = self._last_chunk_weights
        results = []
        for row, chunks in enumerate(self._last_chunks):
            row_weights = (
                weights[row, : len(chunks)].cpu().numpy().tolist()
                if weights is not None
                else [0.0 for _ in chunks]
            )
            order = sorted(range(len(chunks)), key=lambda idx: row_weights[idx], reverse=True)
            top = [
                {
                    "chunk_index": int(idx),
                    "chunk": chunks[idx],
                    "attention": float(row_weights[idx]),
                }
                for idx in order[: min(top_k, len(order))]
            ]
            results.append(
                {
                    "chunks": chunks,
                    "chunk_attention_weights": row_weights,
                    "top_chunks": top,
                }
            )
        return results

    def get_attention_evidence(
        self,
        texts: List[str],
        row_ids: Optional[Sequence[Any]] = None,
        fold: Optional[int] = None,
        stage: str = "nuisance",
        top_k: int = 5,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        interpretations = self.interpret_attention(texts, top_k=top_k)
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
                    "chunk_index": item["chunk_index"],
                    "chunk_text": item["chunk"],
                    "attention": item["attention"],
                }
                record.update(meta)
                records.append(record)
        return records

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
