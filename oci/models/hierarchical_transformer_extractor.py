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

from .gated_attention_pooling import GatedAttentionPooling

logger = logging.getLogger(__name__)
_TRANSFORMERS_ENCODER_INIT_LOCK = threading.Lock()
_SENTENCE_TRANSFORMER_INIT_LOCK = threading.Lock()
_LEGACY_BERT_MODEL_PREFIXES = ("prajjwal1/bert-",)


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
                chunks = split_text_into_word_chunks(
                    key,
                    self._chunk_size_words,
                    self._chunk_overlap_words,
                    self._max_chunks,
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
            truncation=True,
            max_length=self._max_chunk_length,
        )
        input_ids = tuple(int(token_id) for token_id in encoded["input_ids"])
        attention_mask = tuple(int(mask_value) for mask_value in encoded["attention_mask"])
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
        input_ids_tensor[row, offset:offset + len(input_ids)] = torch.as_tensor(
            input_ids,
            dtype=torch.long,
        )
        attention_mask_tensor[row, offset:offset + len(attention_mask)] = torch.as_tensor(
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
        valid_pooling = {"auto", "cls", "last", "mean", "token_attention"}
        if sentence_pooling not in valid_pooling:
            raise ValueError(
                "sentence_pooling must be one of: auto, cls, last, mean, token_attention"
            )
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
        if self._hash_backend and sentence_pooling == "token_attention":
            raise ValueError("token_attention pooling requires a transformer token encoder")

        self._tokenizer = None
        self._sentence_encoder = None
        self._sentence_transformer_encoder = None
        self._token_pooling: Optional[GatedAttentionPooling] = None
        self._resolved_sentence_encoder_path: Optional[str] = None
        self._sentence_dim = self._hash_embedding_dim if self._hash_backend else None
        self._encoder_initialized = self._hash_backend
        self._chunk_cache: Dict[str, List[str]] = {}
        self._tokenization_cache: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        self._chunk_cache_max_entries = int(
            os.environ.get("OCI_HTR_CHUNK_CACHE_MAX_ENTRIES", "100000")
        )
        self._tokenization_cache_max_entries = int(
            os.environ.get("OCI_HTR_TOKEN_CACHE_MAX_ENTRIES", "200000")
        )

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
        self._last_token_weights_by_chunk: List[torch.Tensor] = []
        self._capture_token_attention = False
        self._token_weight_capture_buffer: List[torch.Tensor] = []
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
            key in config
            for key in ("hidden_size", "num_hidden_layers", "num_attention_heads")
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pooling = self._effective_sentence_pooling()
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
        return pooled, token_weights

    def _capture_token_weights(
        self,
        token_weights: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> None:
        if not self._capture_token_attention or token_weights is None:
            return
        weights_cpu = token_weights.detach().cpu()
        mask_cpu = attention_mask.detach().cpu()
        for row in range(weights_cpu.shape[0]):
            valid_len = int(mask_cpu[row].sum().item())
            self._token_weight_capture_buffer.append(
                weights_cpu[row, :valid_len].clone()
            )

    def _encode_chunks(
        self,
        chunks: Sequence[str],
        *,
        return_attention_tensors: bool = False,
    ):
        self._ensure_encoder_initialized()
        if self._hash_backend:
            embeddings = torch.stack([self._hash_chunk_embedding(chunk) for chunk in chunks])
            if return_attention_tensors:
                return embeddings, None
            return embeddings

        if self._effective_sentence_encoder_backend() == "sentence_transformers":
            embeddings = self._encode_chunks_with_sentence_transformer(chunks)
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

    def _encode_chunks_with_transformers(
        self,
        chunks: Sequence[str],
        *,
        return_attention_tensors: bool = False,
    ):
        chunk_list = list(chunks)
        outputs_by_batch = []
        token_weights_by_batch: List[torch.Tensor] = []
        input_ids_by_batch: List[torch.Tensor] = []
        attention_mask_by_batch: List[torch.Tensor] = []
        offset_mapping_by_batch: List[torch.Tensor] = []
        for start in range(0, len(chunk_list), self._sentence_encoder_batch_size):
            batch_chunks = chunk_list[start:start + self._sentence_encoder_batch_size]
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
            outputs_by_batch.append(pooled.float())
            if return_attention_tensors:
                if token_weights is not None:
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
        token_weights_by_batch: List[torch.Tensor] = []
        input_ids_by_batch: List[torch.Tensor] = []
        attention_mask_by_batch: List[torch.Tensor] = []
        for start in range(0, input_ids.shape[0], self._sentence_encoder_batch_size):
            batch_input_ids = input_ids[start:start + self._sentence_encoder_batch_size]
            batch_attention_mask = attention_mask[start:start + self._sentence_encoder_batch_size]
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
            outputs_by_batch.append(pooled.float())
            if return_attention_tensors:
                if token_weights is not None:
                    token_weights_by_batch.append(token_weights)
                input_ids_by_batch.append(batch_input_ids)
                attention_mask_by_batch.append(batch_attention_mask)
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
            "truncation": True,
            "max_length": self._max_chunk_length,
        }
        if return_offsets_mapping and bool(getattr(self._tokenizer, "is_fast", False)):
            kwargs["return_offsets_mapping"] = True
        encoded = self._tokenizer(key, **kwargs)
        input_ids = tuple(int(token_id) for token_id in encoded["input_ids"])
        attention_mask = tuple(int(mask_value) for mask_value in encoded["attention_mask"])
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
            offsets_tensor[row, offset:offset + len(offsets)] = torch.as_tensor(
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
                chunks = split_text_into_word_chunks(
                    key,
                    self._chunk_size_words,
                    self._chunk_overlap_words,
                    self._max_chunks,
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
            not self._hash_backend
            and self._effective_sentence_encoder_backend() == "transformers"
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

    def forward(self, texts_or_batch, *, return_attention_tensors: bool = False):
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
            batch_chunks = self._chunks_for_texts(texts)
        if not texts:
            features = torch.zeros(0, self._projection_dim, device=self._device)
            if return_attention_tensors:
                return features, {
                    "token_alpha": None,
                    "chunk_alpha": None,
                    "input_ids": None,
                    "attention_mask": None,
                    "offset_mapping": None,
                    "token_alpha_sources": [],
                    "batch_chunks": [],
                }
            return features

        flat_chunks = [chunk for chunks in batch_chunks for chunk in chunks]
        if self._capture_token_attention:
            self._token_weight_capture_buffer = []
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
        else:
            self._last_token_weights_by_chunk = []
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

        pool_output = sequence[:, 0, :]
        features = self._output_projection(pool_output)

        if attn_weights is not None:
            pool_attention = attn_weights[:, 0, 1: 1 + max_chunks]
            pool_attention = pool_attention.masked_fill(~chunk_mask, 0.0)
            denom = pool_attention.sum(dim=1, keepdim=True).clamp_min(1e-9)
            chunk_alpha = pool_attention / denom
            self._last_chunk_weights = chunk_alpha.detach()
        else:
            chunk_alpha = None
            self._last_chunk_weights = None
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
            return features, {
                "token_alpha": token_alpha,
                "chunk_alpha": chunk_alpha,
                "input_ids": token_info.get("input_ids"),
                "attention_mask": token_info.get("attention_mask"),
                "offset_mapping": token_info.get("offset_mapping"),
                "token_alpha_sources": token_info.get("token_alpha_sources") or [],
                "batch_chunks": batch_chunks,
                "sequence_input": sequence_input,
                "chunk_mask": chunk_mask,
            }
        return features

    def fit_tokenizer(self, texts: List[str]) -> None:
        self._populate_chunk_cache(texts)
        self._ensure_encoder_initialized()
        logger.info(
            "HierarchicalTransformerExtractor ready: backend=%s pooling=%s "
            "device=%s trainable_params=%s chunk_cache=%s token_cache=%s",
            self._effective_sentence_encoder_backend(),
            self._effective_sentence_pooling(),
            self._device,
            self.get_num_parameters(),
            len(self._chunk_cache),
            len(self._tokenization_cache),
        )

    def interpret_attention(self, texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        self.eval()
        previous_capture = self._capture_token_attention
        self._capture_token_attention = True
        with torch.no_grad():
            try:
                self.forward(texts)
            finally:
                self._capture_token_attention = previous_capture
        weights = self._last_chunk_weights
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
                if flat_idx < len(self._last_token_weights_by_chunk):
                    token_spans = self._top_token_spans(
                        chunks[idx],
                        self._last_token_weights_by_chunk[flat_idx],
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
                truncation=True,
                max_length=self._max_chunk_length,
                return_offsets_mapping=True,
            )
        except Exception:
            return self._top_token_strings(token_weights, chunk, top_n=top_n)
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
        input_ids = encoded.get("input_ids") or []
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
                truncation=True,
                max_length=self._max_chunk_length,
            )
            input_ids = encoded.get("input_ids") or []
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
