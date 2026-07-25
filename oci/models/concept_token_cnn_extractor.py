"""Token-level concept-initialized CNN over LLM hidden states."""

from __future__ import annotations

import gc
import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpu_hidden_state_store import _get_hidden_size, _get_model_dtype, _make_downprojection
from .hidden_state_cache import _sanitize_hidden_states
from .lossless_tokenization import tokenize_losslessly


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


def _normalize_token_rows(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denom, 1e-12)


class LLMTokenHiddenStateEncoder:
    """Lazy encoder for token-level final hidden states from a causal LM."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        downprojection_dim: Optional[int] = None,
    ):
        self._model_name = model_name
        self._device = device
        self._downprojection_dim = downprojection_dim
        self._tokenizer = None
        self._model = None
        self._downproj_layer = None
        self._hidden_size = None
        self._store_dim = None
        self._compute_dtype = None

    def get_hidden_size(self) -> int:
        if self._store_dim is not None:
            return int(self._store_dim)
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self._model_name, trust_remote_code=True)
        hidden_size = _get_hidden_size(config)
        if self._downprojection_dim is not None and self._downprojection_dim < hidden_size:
            return int(self._downprojection_dim)
        return int(hidden_size)

    def _load(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(self._model_name, trust_remote_code=True)
        self._hidden_size = _get_hidden_size(config)
        self._compute_dtype = _get_model_dtype(config)

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            padding_side="right",
            truncation_side="left",
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=self._compute_dtype,
            device_map={"": self._device},
        )
        try:
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(model, recurse=True)
        except ImportError:
            pass

        if tokenizer.pad_token == "[PAD]":
            model.resize_token_embeddings(len(tokenizer))

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self._store_dim = self._hidden_size
        self._downproj_layer = None
        if (
            self._downprojection_dim is not None
            and self._downprojection_dim < self._hidden_size
        ):
            self._downproj_layer = _make_downprojection(
                self._hidden_size,
                self._downprojection_dim,
                self._model_name,
            ).float().to(self._device)
            self._store_dim = self._downprojection_dim

        self._tokenizer = tokenizer
        self._model = model

    def encode_token_sequences(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        batch_size: int = 8,
        normalize_embeddings: bool = True,
    ) -> List[np.ndarray]:
        if not texts:
            return []

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._compute_dtype is not None

        sequences: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoding = tokenize_losslessly(
                self._tokenizer,
                batch_texts,
                add_special_tokens=add_special_tokens,
                configured_max_length=max_length,
                context="LLMTokenHiddenStateEncoder input",
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(self._device)
            attention_mask = encoding["attention_mask"].to(self._device)

            autocast_ctx = (
                torch.autocast(device_type=self._device.type, dtype=self._compute_dtype)
                if self._device.type == "cuda"
                else nullcontext()
            )
            with torch.no_grad(), autocast_ctx:
                backbone = getattr(self._model, "model", self._model)
                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.last_hidden_state
                hidden_states = _sanitize_hidden_states(
                    hidden_states,
                    context="concept_token_cnn",
                )

            if self._downproj_layer is not None:
                with torch.no_grad():
                    hidden_states = self._downproj_layer(hidden_states.float())

            hidden_states = hidden_states.float()
            if normalize_embeddings:
                hidden_states = F.normalize(hidden_states, p=2, dim=-1)

            lengths = attention_mask.sum(dim=1).detach().cpu().tolist()
            batch_cpu = hidden_states.detach().cpu().numpy().astype(np.float32)
            for row, length in zip(batch_cpu, lengths):
                sequences.append(row[: int(length)])

        return sequences

    def close(self) -> None:
        self._tokenizer = None
        self._model = None
        self._downproj_layer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ConceptTokenCNNExtractor(nn.Module):
    """CNN initialized from token-level hidden states of explicit concepts."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B-Base",
        chunk_size: int = 2048,
        chunk_overlap: int = 256,
        max_chunks: int = 16,
        confounder_concepts: Optional[List[str]] = None,
        effect_modifier_concepts: Optional[List[str]] = None,
        random_features: int = 0,
        random_confounder_features: Optional[int] = None,
        random_modifier_features: Optional[int] = None,
        kernel_role: str = "combined",
        projection_dim: int = 128,
        dropout: float = 0.1,
        anchor_weight: float = 0.01,
        cached_hidden_size: int = 0,
        downprojection_dim: Optional[int] = None,
        normalize_embeddings: bool = True,
        random_state: int = 42,
        device: Optional[torch.device] = None,
        token_encoder: Optional[Any] = None,
    ):
        super().__init__()
        if kernel_role not in {"combined", "confounder", "effect_modifier"}:
            raise ValueError(
                "kernel_role must be one of combined, confounder, effect_modifier"
            )
        if projection_dim < 1:
            raise ValueError("projection_dim must be >= 1")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self._device = device or torch.device("cpu")
        self._model_name = model_name
        self._chunk_size = int(chunk_size)
        self._chunk_overlap = int(chunk_overlap)
        self._max_chunks = int(max_chunks)
        self._confounder_concepts = _dedupe_texts(confounder_concepts or [])
        self._effect_modifier_concepts = _dedupe_texts(effect_modifier_concepts or [])
        self._random_features = int(random_features)
        self._random_confounder_features = random_confounder_features
        self._random_modifier_features = random_modifier_features
        self._kernel_role = kernel_role
        self._projection_dim = projection_dim
        self._dropout = dropout
        self._anchor_weight = float(anchor_weight)
        self._cached_hidden_size = int(cached_hidden_size)
        self._downprojection_dim = downprojection_dim
        self._normalize_embeddings = normalize_embeddings
        self._random_state = int(random_state)
        self._token_encoder = token_encoder
        self._owns_token_encoder = token_encoder is None

        concept_texts, n_random = self._select_concepts_and_random_count()
        concept_kernels = self._embed_concepts(concept_texts)

        if concept_kernels:
            embedding_dim = int(concept_kernels[0].shape[1])
        elif self._cached_hidden_size > 0:
            embedding_dim = self._cached_hidden_size
        else:
            embedding_dim = self._infer_embedding_dim()

        if self._cached_hidden_size > 0 and embedding_dim != self._cached_hidden_size:
            raise ValueError(
                "Concept token hidden size does not match cached hidden size: "
                f"{embedding_dim} != {self._cached_hidden_size}"
            )
        self._embedding_dim = int(embedding_dim)

        random_kernels = self._random_kernel_sequences(n_random, self._embedding_dim)
        kernel_arrays = concept_kernels + random_kernels
        if not kernel_arrays:
            raise ValueError(
                "ConceptTokenCNNExtractor requires at least one concept or one "
                "random feature."
            )

        self._num_concept_features = len(concept_kernels)
        self._num_random_features = len(random_kernels)
        self._num_features = len(kernel_arrays)
        self._concept_texts = concept_texts
        self._kernel_lengths = [int(arr.shape[0]) for arr in kernel_arrays]

        self._filters = nn.ParameterList()
        self._filter_is_concept: List[bool] = []
        for i, arr in enumerate(kernel_arrays):
            if arr.ndim != 2 or arr.shape[1] != self._embedding_dim:
                raise ValueError(
                    f"Kernel {i} has shape {arr.shape}; expected (*, {self._embedding_dim})"
                )
            if arr.shape[0] < 1:
                raise ValueError(f"Kernel {i} has no tokens")
            init = np.asarray(arr.T, dtype=np.float32)
            self._filters.append(nn.Parameter(torch.as_tensor(init).clone()))
            is_concept = i < self._num_concept_features
            self._filter_is_concept.append(is_concept)
            anchor = init if is_concept else np.zeros_like(init, dtype=np.float32)
            self.register_buffer(
                f"_anchor_target_{i}",
                torch.as_tensor(anchor, dtype=torch.float32).clone(),
            )

        self._bias = nn.Parameter(torch.zeros(self._num_features, dtype=torch.float32))

        pooled_dim = 2 * self._num_features
        self._output_dim = projection_dim
        self._projection = nn.Sequential(
            nn.Linear(pooled_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        if self._cached_hidden_size > 0 and self._owns_token_encoder:
            self._release_internal_encoder()

        self.to(self._device)
        logger.info(
            "ConceptTokenCNNExtractor initialized: model=%s, role=%s, "
            "concepts=%d, random=%d, hidden_dim=%d, output_dim=%d",
            model_name,
            kernel_role,
            self._num_concept_features,
            self._num_random_features,
            self._embedding_dim,
            projection_dim,
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def hidden_size(self) -> int:
        return self._embedding_dim

    def fit_tokenizer(self, texts: List[str]) -> None:
        """No-op; the causal LM tokenizer is fixed."""
        del texts

    def _get_encoder(self):
        if self._token_encoder is None:
            self._token_encoder = LLMTokenHiddenStateEncoder(
                self._model_name,
                self._device,
                downprojection_dim=self._downprojection_dim,
            )
            self._owns_token_encoder = True
        return self._token_encoder

    def _release_internal_encoder(self) -> None:
        if self._token_encoder is not None and hasattr(self._token_encoder, "close"):
            self._token_encoder.close()
        self._token_encoder = None

    def _call_encoder(
        self,
        texts: List[str],
        add_special_tokens: bool,
        max_length: Optional[int] = None,
    ) -> List[np.ndarray]:
        encoder = self._get_encoder()
        if not hasattr(encoder, "encode_token_sequences"):
            raise TypeError(
                "token_encoder must provide encode_token_sequences(texts, ...)"
            )

        try:
            sequences = encoder.encode_token_sequences(
                texts,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                normalize_embeddings=self._normalize_embeddings,
            )
        except TypeError:
            sequences = encoder.encode_token_sequences(texts)

        result = [np.asarray(seq, dtype=np.float32) for seq in sequences]
        if len(result) != len(texts):
            raise RuntimeError(
                f"Token encoder returned {len(result)} sequences for {len(texts)} texts"
            )
        return result

    def _infer_embedding_dim(self) -> int:
        encoder = self._get_encoder()
        for attr in ("get_hidden_size", "get_token_embedding_dimension"):
            if hasattr(encoder, attr):
                value = getattr(encoder, attr)()
                if int(value) > 0:
                    return int(value)
        if hasattr(encoder, "hidden_size"):
            value = int(getattr(encoder, "hidden_size"))
            if value > 0:
                return value
        probe = self._call_encoder(["probe"], add_special_tokens=False)
        if not probe or probe[0].ndim != 2 or probe[0].shape[1] <= 0:
            raise RuntimeError("Unable to infer token hidden size")
        return int(probe[0].shape[1])

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

    def _embed_concepts(self, concept_texts: List[str]) -> List[np.ndarray]:
        if not concept_texts:
            return []
        sequences = self._call_encoder(concept_texts, add_special_tokens=False)
        result = []
        for text, seq in zip(concept_texts, sequences):
            if seq.ndim != 2:
                raise RuntimeError(
                    f"Unexpected token embedding shape for concept {text!r}: {seq.shape}"
                )
            if seq.shape[0] < 1:
                raise ValueError(f"Concept {text!r} produced no tokens")
            if self._normalize_embeddings:
                seq = _normalize_token_rows(seq)
            result.append(seq.astype(np.float32))
        return result

    def _random_kernel_sequences(self, n_random: int, embedding_dim: int) -> List[np.ndarray]:
        if n_random <= 0:
            return []
        seed = self._random_state + {
            "combined": 0,
            "confounder": 10_000,
            "effect_modifier": 20_000,
        }[self._kernel_role]
        rng = np.random.RandomState(seed)
        values = rng.normal(size=(int(n_random), 1, embedding_dim)).astype(np.float32)
        if self._normalize_embeddings:
            values[:, 0, :] = _normalize_token_rows(values[:, 0, :])
        return [values[i] for i in range(values.shape[0])]

    def _texts_to_tensor(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_length = self._chunk_size * self._max_chunks
        sequences = self._call_encoder(
            list(texts),
            add_special_tokens=True,
            max_length=max_length,
        )
        lengths = [seq.shape[0] for seq in sequences]
        max_len = max(lengths) if lengths else 1
        max_len = max(max_len, 1)
        batch = np.zeros((len(texts), max_len, self._embedding_dim), dtype=np.float32)
        mask = np.zeros((len(texts), max_len), dtype=np.float32)
        for i, seq in enumerate(sequences):
            if seq.ndim != 2 or seq.shape[1] != self._embedding_dim:
                raise ValueError(
                    f"Raw token hidden dim {seq.shape} does not match extractor "
                    f"hidden dim {self._embedding_dim}"
                )
            length = seq.shape[0]
            batch[i, :length] = seq
            mask[i, :length] = 1.0
        return (
            torch.as_tensor(batch, dtype=torch.float32, device=self._device),
            torch.as_tensor(mask, dtype=torch.float32, device=self._device),
        )

    def _extract_token_embeddings(
        self,
        texts_or_batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Sequence[int]]]:
        sample_chunk_counts = None
        if isinstance(texts_or_batch, dict) and "cached_hidden_states" in texts_or_batch:
            embeddings = texts_or_batch["cached_hidden_states"].to(self._device).float()
            mask = texts_or_batch.get("cached_attention_mask")
            if mask is None:
                mask = torch.ones(
                    embeddings.shape[:2],
                    dtype=torch.float32,
                    device=embeddings.device,
                )
            else:
                mask = mask.to(self._device).float()
            sample_chunk_counts = texts_or_batch.get("sample_chunk_counts")
            return embeddings, mask, sample_chunk_counts
        if isinstance(texts_or_batch, dict):
            texts = texts_or_batch.get("texts", [])
        else:
            texts = texts_or_batch
        embeddings, mask = self._texts_to_tensor(list(texts))
        return embeddings, mask, sample_chunk_counts

    def _valid_window_mask(
        self,
        token_mask: torch.Tensor,
        kernel_length: int,
        sample_chunk_counts: Optional[Sequence[int]],
    ) -> torch.Tensor:
        batch_size, seq_len = token_mask.shape
        del batch_size
        if seq_len < kernel_length:
            return torch.zeros(
                (token_mask.shape[0], 0),
                dtype=torch.bool,
                device=token_mask.device,
            )
        ones = torch.ones(
            (1, 1, kernel_length),
            dtype=token_mask.dtype,
            device=token_mask.device,
        )
        counts = F.conv1d(token_mask[:, None, :].clamp(0, 1), ones).squeeze(1)
        valid = counts >= (float(kernel_length) - 1e-6)

        if sample_chunk_counts is not None and self._chunk_size > 0 and kernel_length > 1:
            starts = torch.arange(seq_len - kernel_length + 1, device=token_mask.device)
            same_chunk = (
                starts // self._chunk_size
                == (starts + kernel_length - 1) // self._chunk_size
            )
            valid = valid & same_chunk.unsqueeze(0)
        return valid

    def forward(self, texts_or_batch) -> torch.Tensor:
        token_embeddings, token_mask, sample_chunk_counts = self._extract_token_embeddings(
            texts_or_batch
        )
        if token_embeddings.shape[-1] != self._embedding_dim:
            raise ValueError(
                f"Expected token hidden dim {self._embedding_dim}, got "
                f"{token_embeddings.shape[-1]}"
            )
        if self._normalize_embeddings:
            token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)

        conv_input = token_embeddings.transpose(1, 2)
        max_values = []
        mean_values = []
        response_maps = []
        valid_maps = []

        for i, kernel in enumerate(self._filters):
            kernel_length = int(kernel.shape[1])
            valid = self._valid_window_mask(
                token_mask,
                kernel_length,
                sample_chunk_counts=sample_chunk_counts,
            )
            if valid.shape[1] == 0:
                zeros = token_embeddings.new_zeros(token_embeddings.shape[0])
                max_values.append(zeros)
                mean_values.append(zeros)
                response_maps.append(token_embeddings.new_zeros(token_embeddings.shape[0], 0))
                valid_maps.append(valid)
                continue

            weight = kernel.unsqueeze(0)
            response = F.conv1d(
                conv_input,
                weight,
                bias=self._bias[i:i + 1],
            ).squeeze(1)
            response = response / float(kernel_length)

            valid_f = valid.float()
            masked = response.masked_fill(~valid, -1e9)
            valid_counts = valid_f.sum(dim=1)
            max_response = masked.max(dim=1).values
            max_response = torch.where(
                valid_counts > 0,
                max_response,
                torch.zeros_like(max_response),
            )
            mean_response = (response * valid_f).sum(dim=1) / valid_counts.clamp_min(1.0)
            mean_response = torch.where(
                valid_counts > 0,
                mean_response,
                torch.zeros_like(mean_response),
            )

            max_values.append(max_response)
            mean_values.append(mean_response)
            response_maps.append(response.detach())
            valid_maps.append(valid.detach())

        max_pooled = torch.stack(max_values, dim=1)
        mean_pooled = torch.stack(mean_values, dim=1)
        pooled = torch.cat([max_pooled, mean_pooled], dim=1)
        self._last_response_maps = response_maps
        self._last_valid_window_masks = valid_maps
        self._last_token_mask = token_mask.detach()
        return self._projection(pooled)

    def compute_anchor_loss(self) -> torch.Tensor:
        if self._anchor_weight <= 0 or self._num_concept_features <= 0:
            return torch.tensor(0.0, device=self._bias.device)
        total = torch.tensor(0.0, device=self._bias.device)
        denom = 0
        for i, kernel in enumerate(self._filters):
            if not self._filter_is_concept[i]:
                continue
            target = getattr(self, f"_anchor_target_{i}")
            total = total + (kernel - target).pow(2).sum()
            denom += kernel.numel()
        return self._anchor_weight * (total / max(denom, 1))

    def get_state(self) -> Dict[str, Any]:
        return {
            "extractor_type": "concept_token_cnn",
            "model_name": self._model_name,
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
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
            "kernel_lengths": list(self._kernel_lengths),
            "projection_dim": self._projection_dim,
            "dropout": self._dropout,
            "anchor_weight": self._anchor_weight,
            "downprojection_dim": self._downprojection_dim,
            "output_dim": self._output_dim,
        }

    def get_num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def to(self, device):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)
