"""Neural causal-forest text extractor with token-level evidence.

This module is intentionally self-contained: it does not require modifying the
main OCI runner/config stack before experimentation.  It implements the core
idea behind a causal forest in a differentiable text model:

* nuisance functions are cross-fit first;
* treatment effects are learned with the orthogonal R-loss, not with noisy
  patient-level pseudo-labels;
* the treatment-effect head is an ensemble of soft trees with leaf-wise CATEs;
* after learning tree structure, leaf CATEs are re-estimated by an honest,
  closed-form orthogonal moment estimator;
* effect-modifier evidence is exported at token level via gradient x attention
  of the neural causal-forest CATE output.

The files under ``oci/inference`` and ``oracle_experiment_scripts`` wrap this
module in train/predict/oracle command line scripts.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

OutcomeType = Literal["binary", "continuous"]
StageType = Literal["nuisance", "effect_modifier", "cate"]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class NeuralCausalForestConfig:
    """Configuration for the neural causal forest extractor.

    Defaults are deliberately conservative for the structured synthetic oncology
    experiments: trainable small encoder, short chunks, many shallow soft trees,
    honest leaf refitting, and token evidence from gradient x attention.
    """

    # Text encoder
    encoder_architecture: str = "hierarchical_transformer"
    encoder_model_name: str = "prajjwal1/bert-tiny"
    encoder_backend: str = "transformers"  # "transformers" or "hash" for no-download tests
    freeze_encoder: bool = False
    trainable_encoder_layers: int = 0
    max_length: int = 128
    chunk_size_words: int = 96
    chunk_overlap_words: int = 24
    max_chunks: int = 128
    representation_dim: int = 128
    token_attention_dim: int = 128
    chunk_attention_dim: int = 128
    dropout: float = 0.10
    normalize_representations: bool = False

    # HTR encoder settings.  This is the default encoder architecture for both
    # nuisance and CATE models; the older NCF token-attention encoder remains
    # available via encoder_architecture="ncf_token_attention".
    htr_num_layers: int = 2
    htr_num_heads: int = 4
    htr_transformer_dim: int = 256
    htr_sentence_encoder_batch_size: int = 128
    htr_sentence_encoder_backend: str = "transformers"
    htr_sentence_pooling: str = "token_attention"
    htr_normalize_sentence_embeddings: bool = True
    htr_hash_embedding_dim: int = 256

    # Nuisance heads
    nuisance_hidden_dim: int = 128
    nuisance_epochs: int = 50
    nuisance_learning_rate: float = 1e-4
    nuisance_weight_decay: float = 0.01
    nuisance_folds: int = 5
    inner_fold_parallelism: str = "auto"
    alpha_propensity: float = 1.0

    # Soft neural causal forest
    n_trees: int = 32
    depth: int = 3
    forest_learning_rate: float = 1e-4
    forest_weight_decay: float = 0.01
    forest_epochs: int = 80
    batch_size: int = 8
    effect_batch_size: Optional[int] = 16
    gradient_clip_norm: float = 1.0
    temperature_start: float = 1.5
    temperature_end: float = 0.35
    feature_subsample_fraction: float = 1.0
    leaf_ridge: float = 1e-3
    leaf_min_mass: float = 5.0
    tau_clip: Optional[float] = 2.0

    # Forest regularization.  Heterogeneity reward is a differentiable analogue
    # of causal-forest split selection.  Keep it small; honest refitting is the
    # main protection against fitting noise.
    lambda_leaf_balance: float = 0.02
    lambda_leaf_min_mass: float = 0.01
    lambda_leaf_tau_l2: float = 1e-4
    lambda_heterogeneity: float = 0.05

    # Honest split/leaf refit
    honesty_fraction: float = 0.50
    refit_leaf_values_after_training: bool = True

    # Attribution / evidence
    attention_top_k: int = 8
    evidence_batch_size: int = 4
    max_evidence_tokens_per_chunk: int = 16
    snippet_context_chars: int = 80

    # Reproducibility
    seed: int = 42
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.n_trees < 1:
            raise ValueError("n_trees must be >= 1")
        if self.max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
        if not 0.0 < self.feature_subsample_fraction <= 1.0:
            raise ValueError("feature_subsample_fraction must be in (0, 1]")
        if not 0.0 < self.honesty_fraction < 1.0:
            raise ValueError("honesty_fraction must be in (0, 1)")
        if str(self.inner_fold_parallelism).strip().lower() != "auto":
            try:
                if int(self.inner_fold_parallelism) < 1:
                    raise ValueError
            except ValueError as exc:
                raise ValueError("inner_fold_parallelism must be 'auto' or a positive integer") from exc
        architecture = str(self.encoder_architecture or "").lower()
        architecture_aliases = {
            "htr": "hierarchical_transformer",
            "hierarchical_transformer": "hierarchical_transformer",
            "ncf": "ncf_token_attention",
            "ncf_token_attention": "ncf_token_attention",
            "token_attention": "ncf_token_attention",
        }
        if architecture not in architecture_aliases:
            raise ValueError(
                "encoder_architecture must be 'hierarchical_transformer' or 'ncf_token_attention'"
            )
        self.encoder_architecture = architecture_aliases[architecture]
        if self.encoder_backend not in {"transformers", "hash"}:
            raise ValueError("encoder_backend must be 'transformers' or 'hash'")
        if self.htr_num_layers < 1:
            raise ValueError("htr_num_layers must be >= 1")
        if self.htr_num_heads < 1:
            raise ValueError("htr_num_heads must be >= 1")
        if self.htr_transformer_dim < 1:
            raise ValueError("htr_transformer_dim must be >= 1")
        if self.htr_transformer_dim % self.htr_num_heads != 0:
            raise ValueError("htr_transformer_dim must be divisible by htr_num_heads")
        if self.htr_sentence_encoder_batch_size < 1:
            raise ValueError("htr_sentence_encoder_batch_size must be >= 1")
        if self.htr_sentence_encoder_backend not in {"auto", "sentence_transformers", "transformers"}:
            raise ValueError(
                "htr_sentence_encoder_backend must be one of: auto, sentence_transformers, transformers"
            )
        if self.htr_sentence_pooling not in {"auto", "cls", "last", "mean", "token_attention"}:
            raise ValueError(
                "htr_sentence_pooling must be one of: auto, cls, last, mean, token_attention"
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "NeuralCausalForestConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**payload)

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=True)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        path = resolve_dataset_file(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported dataset file extension: {path}")


def resolve_dataset_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(path)
    preferred_names = [
        "data.parquet",
        "dataset.parquet",
        "synthetic_data.parquet",
        "patients.parquet",
        "data.csv",
        "dataset.csv",
    ]
    for name in preferred_names:
        candidate = path / name
        if candidate.exists():
            return candidate
    candidates = sorted(path.glob("*.parquet")) + sorted(path.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No parquet/csv dataset file found under {path}")
    return candidates[0]


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        df.to_json(path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported output extension: {path}")


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    if int(mask.sum()) < 2 or len(np.unique(y_true[mask])) < 2:
        return None
    try:
        return float(roc_auc_score(y_true[mask], y_score[mask]))
    except ValueError:
        return None


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    if float(np.std(x[mask])) == 0.0 or float(np.std(y[mask])) == 0.0:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _linear_temperature(config: NeuralCausalForestConfig, epoch: int) -> float:
    if config.forest_epochs <= 1:
        return float(config.temperature_end)
    frac = (epoch - 1) / max(1, config.forest_epochs - 1)
    return float(config.temperature_start + frac * (config.temperature_end - config.temperature_start))


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _make_linear_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batches_per_epoch: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    total_steps = max(1, int(epochs) * max(1, int(batches_per_epoch)))

    def lr_lambda(step: int) -> float:
        return max(0.0, 1.0 - float(step) / float(total_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# -----------------------------------------------------------------------------
# Chunking and datasets
# -----------------------------------------------------------------------------


@dataclass
class ChunkInfo:
    text: str
    char_start: int
    char_end: int
    chunk_index: int


def split_text_into_word_span_chunks(
    text: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
    max_chunks: int,
) -> List[ChunkInfo]:
    """Split a note into short overlapping word chunks while keeping char spans."""
    text = str(text or "")
    words = [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]
    if not words:
        return [ChunkInfo(text="", char_start=0, char_end=0, chunk_index=0)]

    stride = chunk_size_words - chunk_overlap_words
    max_window_words = chunk_size_words + (max_chunks - 1) * stride
    first_word = max(0, len(words) - max_window_words)
    window_words = words[first_word:]

    chunks: List[ChunkInfo] = []
    start_idx = 0
    while start_idx < len(window_words) and len(chunks) < max_chunks:
        stop_idx = min(start_idx + chunk_size_words, len(window_words))
        chunk_words = window_words[start_idx:stop_idx]
        if chunk_words:
            char_start = int(chunk_words[0][1])
            char_end = int(chunk_words[-1][2])
            chunks.append(
                ChunkInfo(
                    text=text[char_start:char_end],
                    char_start=char_start,
                    char_end=char_end,
                    chunk_index=len(chunks),
                )
            )
        start_idx += stride
    return chunks or [ChunkInfo(text=text, char_start=0, char_end=len(text), chunk_index=0)]


class TextOutcomeDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        row_ids: Sequence[Any],
        treatment: Optional[Sequence[float]] = None,
        outcome: Optional[Sequence[float]] = None,
        e_hat: Optional[Sequence[float]] = None,
        m_hat: Optional[Sequence[float]] = None,
    ) -> None:
        self.texts = [str(text or "") for text in texts]
        self.row_ids = list(row_ids)
        self.treatment = None if treatment is None else np.asarray(treatment, dtype=np.float32)
        self.outcome = None if outcome is None else np.asarray(outcome, dtype=np.float32)
        self.e_hat = None if e_hat is None else np.asarray(e_hat, dtype=np.float32)
        self.m_hat = None if m_hat is None else np.asarray(m_hat, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "text": self.texts[index],
            "row_id": self.row_ids[index],
            "position": int(index),
        }
        if self.treatment is not None:
            item["t"] = float(self.treatment[index])
        if self.outcome is not None:
            item["y"] = float(self.outcome[index])
        if self.e_hat is not None:
            item["e_hat"] = float(self.e_hat[index])
        if self.m_hat is not None:
            item["m_hat"] = float(self.m_hat[index])
        return item


def _collate_text_batch(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    batch: Dict[str, Any] = {
        "texts": [item["text"] for item in items],
        "row_ids": [item["row_id"] for item in items],
        "position": torch.as_tensor([int(item["position"]) for item in items], dtype=torch.long),
    }
    for key in ("t", "y", "e_hat", "m_hat"):
        if key in items[0]:
            batch[key] = torch.as_tensor([float(item[key]) for item in items], dtype=torch.float32)
    return batch


def make_text_loader(
    texts: Sequence[str],
    row_ids: Sequence[Any],
    treatment: Optional[Sequence[float]] = None,
    outcome: Optional[Sequence[float]] = None,
    e_hat: Optional[Sequence[float]] = None,
    m_hat: Optional[Sequence[float]] = None,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TextOutcomeDataset(
        texts=texts,
        row_ids=row_ids,
        treatment=treatment,
        outcome=outcome,
        e_hat=e_hat,
        m_hat=m_hat,
    )
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        num_workers=max(0, int(num_workers)),
        collate_fn=_collate_text_batch,
        pin_memory=torch.cuda.is_available(),
    )


# -----------------------------------------------------------------------------
# Token/chunk attention encoder
# -----------------------------------------------------------------------------


@dataclass
class EncoderForwardOutput:
    representation: torch.Tensor
    token_alpha: Optional[torch.Tensor] = None
    token_alpha_sources: Optional[List[torch.Tensor]] = None
    chunk_alpha: Optional[torch.Tensor] = None
    flat_chunk_patient_index: Optional[torch.Tensor] = None
    flat_chunk_local_index: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    offset_mapping: Optional[torch.Tensor] = None
    chunks_by_patient: Optional[List[List[ChunkInfo]]] = None
    texts: Optional[List[str]] = None


class HierarchicalTokenAttentionEncoder(nn.Module):
    """Transformer/hash chunk encoder with token- and chunk-level attention.

    The production path uses a HuggingFace encoder.  ``encoder_backend='hash'`` is
    included to make syntax/shape tests and CPU dry runs possible without model
    downloads; it is not intended for final scientific runs.
    """

    def __init__(self, config: NeuralCausalForestConfig, device: torch.device | str) -> None:
        super().__init__()
        self.config = config
        self.device_obj = torch.device(device)
        self.output_dim = int(config.representation_dim)
        self.encoder_backend = config.encoder_backend
        self._tokenizer: Any = None
        self._backbone: Optional[nn.Module] = None
        self._resolved_encoder_model_path: Optional[str] = None
        self._hidden_size: int

        if config.encoder_backend == "transformers":
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for encoder_backend='transformers'. "
                    "Install the repo requirements or set encoder_backend='hash' for a dry run."
                ) from exc
            self._tokenizer = self._load_tokenizer(AutoTokenizer)
            if getattr(self._tokenizer, "pad_token", None) is None:
                eos_token = getattr(self._tokenizer, "eos_token", None)
                self._tokenizer.pad_token = eos_token or getattr(self._tokenizer, "unk_token", "[PAD]")
            self._backbone = self._load_transformers_model(AutoModel)
            self._hidden_size = int(getattr(self._backbone.config, "hidden_size"))
            self._configure_encoder_trainability()
        else:
            self._hidden_size = int(max(config.representation_dim, 64))
            self.hash_embedding = nn.Embedding(50000, self._hidden_size)
            nn.init.normal_(self.hash_embedding.weight, std=0.02)

        self.token_score = nn.Sequential(
            nn.Linear(self._hidden_size, config.token_attention_dim),
            nn.Tanh(),
            nn.Linear(config.token_attention_dim, 1),
        )
        self.chunk_score = nn.Sequential(
            nn.Linear(self._hidden_size, config.chunk_attention_dim),
            nn.Tanh(),
            nn.Linear(config.chunk_attention_dim, 1),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(self._hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(self._hidden_size, config.representation_dim),
            nn.GELU(),
            nn.LayerNorm(config.representation_dim),
        )
        self.to(self.device_obj)

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def _load_tokenizer(self, auto_tokenizer_cls):
        try:
            return auto_tokenizer_cls.from_pretrained(
                self.config.encoder_model_name,
                use_fast=True,
            )
        except Exception as fast_exc:
            logger.warning(
                "Fast tokenizer load failed for %s (%s). Retrying with use_fast=False.",
                self.config.encoder_model_name,
                fast_exc,
            )
            try:
                return auto_tokenizer_cls.from_pretrained(
                    self.config.encoder_model_name,
                    use_fast=False,
                )
            except Exception as slow_exc:
                try:
                    tokenizer = self._load_legacy_bert_tokenizer()
                except Exception as legacy_exc:
                    raise RuntimeError(
                        "Could not load tokenizer for neural causal forest encoder "
                        f"{self.config.encoder_model_name!r}. Install tokenizer conversion "
                        "dependencies with `pip install sentencepiece tiktoken`, or use a "
                        "BERT/WordPiece model with tokenizer files available locally. "
                        f"Fast tokenizer error: {fast_exc}. Slow tokenizer error: {slow_exc}. "
                        f"Legacy BERT tokenizer error: {legacy_exc}."
                    ) from legacy_exc
                if tokenizer is not None:
                    return tokenizer
                raise RuntimeError(
                    "Could not load tokenizer for neural causal forest encoder "
                    f"{self.config.encoder_model_name!r}. Install tokenizer conversion "
                    "dependencies with `pip install sentencepiece tiktoken`, or use a "
                    "BERT/WordPiece model with tokenizer files available locally. "
                    f"Fast tokenizer error: {fast_exc}. Slow tokenizer error: {slow_exc}."
                ) from slow_exc

    def _load_transformers_model(self, auto_model_cls):
        try:
            return auto_model_cls.from_pretrained(self.config.encoder_model_name)
        except Exception as auto_exc:
            try:
                model = self._load_legacy_bert_model()
            except Exception as legacy_exc:
                raise RuntimeError(
                    "Could not load neural causal forest transformer encoder "
                    f"{self.config.encoder_model_name!r}. AutoModel error: {auto_exc}. "
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

        resolved_model = self._resolve_encoder_model_path()
        vocab_file = Path(resolved_model) / "vocab.txt"
        if not vocab_file.exists():
            raise FileNotFoundError(f"legacy BERT tokenizer fallback expected {vocab_file}")
        logger.info("Loading legacy BERT tokenizer from local snapshot: %s", resolved_model)
        return BertTokenizer.from_pretrained(resolved_model, local_files_only=True)

    def _load_legacy_bert_model(self):
        if not self._should_try_legacy_bert_loader():
            return None
        try:
            from transformers import BertConfig, BertModel
        except ImportError as exc:
            raise ImportError("transformers is required for BertModel fallback") from exc

        resolved_model = self._resolve_encoder_model_path()
        logger.info("Loading legacy BERT model from local snapshot: %s", resolved_model)
        config = BertConfig.from_pretrained(resolved_model, local_files_only=True)
        return BertModel.from_pretrained(
            resolved_model,
            config=config,
            local_files_only=True,
        )

    def _should_try_legacy_bert_loader(self) -> bool:
        model_name = str(self.config.encoder_model_name).lower()
        if "bert" in model_name:
            return True
        model_path = Path(str(self.config.encoder_model_name)).expanduser()
        return model_path.exists() and (model_path / "vocab.txt").exists()

    @staticmethod
    def _huggingface_offline() -> bool:
        for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
            value = os.environ.get(name)
            if value and value.lower() in {"1", "true", "yes", "on"}:
                return True
        return False

    def _resolve_encoder_model_path(self) -> str:
        if self._resolved_encoder_model_path is not None:
            return self._resolved_encoder_model_path

        model_path = Path(str(self.config.encoder_model_name)).expanduser()
        if model_path.exists():
            self._resolved_encoder_model_path = str(model_path)
            return self._resolved_encoder_model_path

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to resolve legacy BERT checkpoints"
            ) from exc

        self._resolved_encoder_model_path = snapshot_download(
            str(self.config.encoder_model_name),
            local_files_only=self._huggingface_offline(),
        )
        return self._resolved_encoder_model_path

    def _configure_encoder_trainability(self) -> None:
        if self._backbone is None:
            return
        for parameter in self._backbone.parameters():
            parameter.requires_grad = not self.config.freeze_encoder
        if self.config.trainable_encoder_layers > 0:
            # Best effort across BERT/RoBERTa/DeBERTa-like encoders.
            layers = None
            for attr_path in (
                ("encoder", "layer"),
                ("bert", "encoder", "layer"),
                ("roberta", "encoder", "layer"),
                ("deberta", "encoder", "layer"),
            ):
                module = self._backbone
                ok = True
                for attr in attr_path:
                    if hasattr(module, attr):
                        module = getattr(module, attr)
                    else:
                        ok = False
                        break
                if ok:
                    layers = module
                    break
            if layers is not None:
                for layer in list(layers)[-int(self.config.trainable_encoder_layers):]:
                    for parameter in layer.parameters():
                        parameter.requires_grad = True

    def split_texts(self, texts: Sequence[str]) -> List[List[ChunkInfo]]:
        return [
            split_text_into_word_span_chunks(
                text,
                self.config.chunk_size_words,
                self.config.chunk_overlap_words,
                self.config.max_chunks,
            )
            for text in texts
        ]

    def forward(
        self,
        texts: Sequence[str],
        *,
        return_attention_tensors: bool = False,
    ) -> EncoderForwardOutput:
        texts = [str(text or "") for text in texts]
        chunks_by_patient = self.split_texts(texts)
        flat_chunks: List[ChunkInfo] = [chunk for chunks in chunks_by_patient for chunk in chunks]
        if not flat_chunks:
            flat_chunks = [ChunkInfo(text="", char_start=0, char_end=0, chunk_index=0)]
            chunks_by_patient = [[flat_chunks[0]] for _ in texts]

        flat_patient_index: List[int] = []
        flat_local_index: List[int] = []
        for patient_idx, chunks in enumerate(chunks_by_patient):
            for local_idx, _chunk in enumerate(chunks):
                flat_patient_index.append(patient_idx)
                flat_local_index.append(local_idx)

        chunk_hidden, input_ids, attention_mask, offset_mapping = self._encode_flat_chunks(flat_chunks)
        token_logits = self.token_score(chunk_hidden).squeeze(-1)
        token_logits = token_logits.masked_fill(attention_mask <= 0, -1e4)
        token_alpha = torch.softmax(token_logits, dim=1)
        chunk_vectors = torch.sum(token_alpha.unsqueeze(-1) * chunk_hidden, dim=1)

        batch_size = len(texts)
        max_chunks = max(len(chunks) for chunks in chunks_by_patient)
        hidden_size = chunk_vectors.shape[-1]
        chunk_tensor = torch.zeros(
            batch_size,
            max_chunks,
            hidden_size,
            dtype=chunk_vectors.dtype,
            device=chunk_vectors.device,
        )
        chunk_mask = torch.zeros(batch_size, max_chunks, dtype=torch.bool, device=chunk_vectors.device)
        for flat_idx, (patient_idx, local_idx) in enumerate(zip(flat_patient_index, flat_local_index)):
            chunk_tensor[patient_idx, local_idx] = chunk_vectors[flat_idx]
            chunk_mask[patient_idx, local_idx] = True

        chunk_logits = self.chunk_score(chunk_tensor).squeeze(-1)
        chunk_logits = chunk_logits.masked_fill(~chunk_mask, -1e4)
        chunk_alpha = torch.softmax(chunk_logits, dim=1)
        patient_vector = torch.sum(chunk_alpha.unsqueeze(-1) * chunk_tensor, dim=1)
        representation = self.projection(patient_vector)
        if self.config.normalize_representations:
            representation = F.normalize(representation, p=2, dim=-1)

        if return_attention_tensors:
            token_alpha.retain_grad()
            chunk_alpha.retain_grad()
            return EncoderForwardOutput(
                representation=representation,
                token_alpha=token_alpha,
                chunk_alpha=chunk_alpha,
                flat_chunk_patient_index=torch.as_tensor(
                    flat_patient_index, dtype=torch.long, device=representation.device
                ),
                flat_chunk_local_index=torch.as_tensor(
                    flat_local_index, dtype=torch.long, device=representation.device
                ),
                attention_mask=attention_mask,
                input_ids=input_ids,
                offset_mapping=offset_mapping,
                chunks_by_patient=chunks_by_patient,
                texts=texts,
            )
        return EncoderForwardOutput(representation=representation)

    def _encode_flat_chunks(
        self,
        flat_chunks: Sequence[ChunkInfo],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.encoder_backend == "hash":
            return self._encode_flat_chunks_hash(flat_chunks)
        assert self._tokenizer is not None and self._backbone is not None
        chunk_texts = [chunk.text for chunk in flat_chunks]
        use_offsets = bool(getattr(self._tokenizer, "is_fast", False))
        encoded = self._tokenizer(
            chunk_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            return_offsets_mapping=use_offsets,
        )
        offset_mapping = encoded.pop("offset_mapping", None)
        input_ids = encoded["input_ids"].to(self.device_obj)
        attention_mask = encoded["attention_mask"].to(self.device_obj)
        model_inputs = {key: value.to(self.device_obj) for key, value in encoded.items()}
        if self.config.freeze_encoder and self.config.trainable_encoder_layers <= 0:
            with torch.no_grad():
                outputs = self._backbone(**model_inputs)
        else:
            outputs = self._backbone(**model_inputs)
        hidden = outputs.last_hidden_state
        if offset_mapping is None:
            offset_mapping = torch.full(
                (input_ids.shape[0], input_ids.shape[1], 2),
                -1,
                dtype=torch.long,
            )
        return hidden, input_ids, attention_mask, offset_mapping.to(self.device_obj)

    def _encode_flat_chunks_hash(
        self,
        flat_chunks: Sequence[ChunkInfo],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_rows: List[List[int]] = []
        offset_rows: List[List[Tuple[int, int]]] = []
        for chunk in flat_chunks:
            tokens = list(re.finditer(r"\S+", chunk.text))[: self.config.max_length]
            if not tokens:
                token_rows.append([0])
                offset_rows.append([(0, 0)])
                continue
            ids = [abs(hash(match.group(0).lower())) % 49999 + 1 for match in tokens]
            offsets = [(match.start(), match.end()) for match in tokens]
            token_rows.append(ids)
            offset_rows.append(offsets)
        max_len = max(len(row) for row in token_rows)
        input_ids = torch.zeros(len(token_rows), max_len, dtype=torch.long, device=self.device_obj)
        attention_mask = torch.zeros(len(token_rows), max_len, dtype=torch.long, device=self.device_obj)
        offsets = torch.full(
            (len(token_rows), max_len, 2), -1, dtype=torch.long, device=self.device_obj
        )
        for row_idx, ids in enumerate(token_rows):
            n = len(ids)
            input_ids[row_idx, :n] = torch.as_tensor(ids, dtype=torch.long, device=self.device_obj)
            attention_mask[row_idx, :n] = 1
            offsets[row_idx, :n] = torch.as_tensor(
                offset_rows[row_idx], dtype=torch.long, device=self.device_obj
            )
        hidden = self.hash_embedding(input_ids)
        return hidden, input_ids, attention_mask, offsets

    def decode_token(self, token_id: int) -> str:
        if self.encoder_backend == "hash" or self._tokenizer is None:
            return ""
        try:
            return str(self._tokenizer.convert_ids_to_tokens([int(token_id)])[0])
        except Exception:
            return ""

    def attention_records_from_output(
        self,
        output: EncoderForwardOutput,
        row_ids: Sequence[Any],
        *,
        stage: str,
        top_k: int,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
        token_scores: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """Convert saved attention tensors into token/snippet evidence records."""
        if output.token_alpha is None or output.chunk_alpha is None:
            return []
        if output.attention_mask is None or output.input_ids is None:
            return []
        if output.offset_mapping is None or output.chunks_by_patient is None or output.texts is None:
            return []
        token_alpha = output.token_alpha.detach().cpu()
        chunk_alpha = output.chunk_alpha.detach().cpu()
        attention_mask = output.attention_mask.detach().cpu()
        input_ids = output.input_ids.detach().cpu()
        offsets = output.offset_mapping.detach().cpu()
        patient_index = output.flat_chunk_patient_index.detach().cpu().numpy()
        local_index = output.flat_chunk_local_index.detach().cpu().numpy()
        if token_scores is None:
            token_score_tensor = token_alpha.clone()
        else:
            token_score_tensor = token_scores.detach().cpu()

        records: List[Dict[str, Any]] = []
        top_k = max(1, int(top_k))
        metadata = list(metadata or [{} for _ in row_ids])
        per_patient_candidates: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in range(len(row_ids))}
        tokenizer = getattr(self, "tokenizer", None)
        special_token_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        special_tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])

        for flat_idx in range(token_alpha.shape[0]):
            patient_idx = int(patient_index[flat_idx])
            chunk_idx = int(local_index[flat_idx])
            chunk = output.chunks_by_patient[patient_idx][chunk_idx]
            valid_positions = torch.where(attention_mask[flat_idx] > 0)[0].tolist()
            if not valid_positions:
                continue
            local_scores = token_score_tensor[flat_idx, valid_positions]
            local_scores = torch.nan_to_num(local_scores, nan=0.0, posinf=0.0, neginf=0.0)
            if local_scores.numel() == 0:
                continue
            n_take = min(
                max(top_k * 2, self.config.max_evidence_tokens_per_chunk),
                int(local_scores.numel()),
            )
            top_local = torch.topk(local_scores, k=n_take).indices.tolist()
            for rank_position in top_local:
                pos = int(valid_positions[rank_position])
                token_id = int(input_ids[flat_idx, pos].item())
                if token_id in special_token_ids:
                    continue
                start_local = int(offsets[flat_idx, pos, 0].item())
                end_local = int(offsets[flat_idx, pos, 1].item())
                if start_local < 0 or end_local <= start_local:
                    token_text = self.decode_token(token_id)
                    char_start = chunk.char_start
                    char_end = min(chunk.char_end, chunk.char_start + len(token_text))
                else:
                    char_start = chunk.char_start + start_local
                    char_end = chunk.char_start + end_local
                    token_text = output.texts[patient_idx][char_start:char_end]
                if token_text in special_tokens:
                    continue
                if not re.search(r"[A-Za-z0-9]", str(token_text).replace("##", "")):
                    continue
                snippet_start = max(0, char_start - self.config.snippet_context_chars)
                snippet_end = min(
                    len(output.texts[patient_idx]),
                    char_end + self.config.snippet_context_chars,
                )
                score = float(token_score_tensor[flat_idx, pos].item())
                candidate = {
                    "row_id": row_ids[patient_idx],
                    "stage": stage,
                    "chunk_index": chunk_idx,
                    "token_position": pos,
                    "token_text": token_text,
                    "char_start": char_start,
                    "char_end": char_end,
                    "snippet": output.texts[patient_idx][snippet_start:snippet_end],
                    "token_attention": float(token_alpha[flat_idx, pos].item()),
                    "chunk_attention": float(chunk_alpha[patient_idx, chunk_idx].item()),
                    "evidence_score": score,
                }
                candidate.update(metadata[patient_idx])
                per_patient_candidates[patient_idx].append(candidate)

        for patient_idx, candidates in per_patient_candidates.items():
            candidates.sort(key=lambda record: abs(float(record["evidence_score"])), reverse=True)
            # Deduplicate near-identical tokens by char span.
            seen_spans: set[Tuple[int, int]] = set()
            emitted = 0
            for candidate in candidates:
                span = (int(candidate["char_start"]), int(candidate["char_end"]))
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                emitted += 1
                candidate["rank_within_patient"] = emitted
                records.append(candidate)
                if emitted >= top_k:
                    break
        return records


class HTRGradientAttentionEncoder(nn.Module):
    """HTR encoder adapter with the NCF attribution tensor contract."""

    def __init__(self, config: NeuralCausalForestConfig, device: torch.device | str) -> None:
        super().__init__()
        from .hierarchical_transformer_extractor import HierarchicalTransformerExtractor

        self.config = config
        self.device_obj = torch.device(device)
        self.output_dim = int(config.representation_dim)
        sentence_encoder_model = (
            "hash" if config.encoder_backend == "hash" else config.encoder_model_name
        )
        sentence_encoder_backend = (
            "auto" if config.encoder_backend == "hash" else config.htr_sentence_encoder_backend
        )
        sentence_pooling = (
            "auto" if config.encoder_backend == "hash" else config.htr_sentence_pooling
        )
        self.htr = HierarchicalTransformerExtractor(
            sentence_encoder_model=sentence_encoder_model,
            freeze_sentence_encoder=config.freeze_encoder,
            chunk_size_words=config.chunk_size_words,
            chunk_overlap_words=config.chunk_overlap_words,
            max_chunks=config.max_chunks,
            max_chunk_length=config.max_length,
            num_transformer_layers=config.htr_num_layers,
            num_attention_heads=config.htr_num_heads,
            transformer_dim=config.htr_transformer_dim,
            transformer_dropout=config.dropout,
            projection_dim=config.representation_dim,
            hash_embedding_dim=config.htr_hash_embedding_dim,
            sentence_encoder_batch_size=config.htr_sentence_encoder_batch_size,
            sentence_encoder_backend=sentence_encoder_backend,
            sentence_pooling=sentence_pooling,
            normalize_sentence_embeddings=config.htr_normalize_sentence_embeddings,
            trainable_sentence_encoder_layers=config.trainable_encoder_layers,
            device=self.device_obj,
        )
        # HTR initializes transformer/token-pooling parameters lazily.  The NCF
        # optimizers are created immediately after model construction, so force
        # initialization here to include the encoder parameters in the optimizer.
        self.htr.fit_tokenizer([])
        self.to(self.device_obj)

    @property
    def tokenizer(self) -> Any:
        return getattr(self.htr, "_tokenizer", None)

    def split_texts(self, texts: Sequence[str]) -> List[List[ChunkInfo]]:
        return [
            split_text_into_word_span_chunks(
                text,
                self.config.chunk_size_words,
                self.config.chunk_overlap_words,
                self.config.max_chunks,
            )
            for text in texts
        ]

    def _make_htr_batch(
        self,
        texts: Sequence[str],
        chunks_by_patient: Sequence[Sequence[ChunkInfo]],
    ) -> Dict[str, Any]:
        return {
            "texts": list(texts),
            "chunks": [[chunk.text for chunk in chunks] for chunks in chunks_by_patient],
        }

    def forward(
        self,
        texts: Sequence[str],
        *,
        return_attention_tensors: bool = False,
    ) -> EncoderForwardOutput:
        texts = [str(text or "") for text in texts]
        chunks_by_patient = self.split_texts(texts)
        batch = self._make_htr_batch(texts, chunks_by_patient)
        flat_patient_index: List[int] = []
        flat_local_index: List[int] = []
        for patient_idx, chunks in enumerate(chunks_by_patient):
            for local_idx, _chunk in enumerate(chunks):
                flat_patient_index.append(patient_idx)
                flat_local_index.append(local_idx)

        if return_attention_tensors:
            representation, attention_info = self.htr(
                batch,
                return_attention_tensors=True,
            )
        else:
            representation = self.htr(batch)
            attention_info = None
        if self.config.normalize_representations:
            representation = F.normalize(representation, p=2, dim=-1)

        if not return_attention_tensors:
            return EncoderForwardOutput(representation=representation)

        attention_info = attention_info or {}
        token_alpha = attention_info.get("token_alpha")
        token_alpha_sources = attention_info.get("token_alpha_sources") or None
        chunk_alpha = attention_info.get("chunk_alpha")
        return EncoderForwardOutput(
            representation=representation,
            token_alpha=token_alpha,
            token_alpha_sources=token_alpha_sources,
            chunk_alpha=chunk_alpha,
            flat_chunk_patient_index=torch.as_tensor(
                flat_patient_index, dtype=torch.long, device=representation.device
            ),
            flat_chunk_local_index=torch.as_tensor(
                flat_local_index, dtype=torch.long, device=representation.device
            ),
            attention_mask=attention_info.get("attention_mask"),
            input_ids=attention_info.get("input_ids"),
            offset_mapping=attention_info.get("offset_mapping"),
            chunks_by_patient=chunks_by_patient,
            texts=texts,
        )

    def decode_token(self, token_id: int) -> str:
        tokenizer = self.tokenizer
        if tokenizer is None:
            return ""
        try:
            return str(tokenizer.convert_ids_to_tokens([int(token_id)])[0])
        except Exception:
            return ""

    def attention_records_from_output(
        self,
        output: EncoderForwardOutput,
        row_ids: Sequence[Any],
        *,
        stage: str,
        top_k: int,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
        token_scores: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        return HierarchicalTokenAttentionEncoder.attention_records_from_output(
            self,
            output,
            row_ids,
            stage=stage,
            top_k=top_k,
            metadata=metadata,
            token_scores=token_scores,
        )


def make_ncf_text_encoder(
    config: NeuralCausalForestConfig,
    device: torch.device | str,
) -> nn.Module:
    if config.encoder_architecture == "hierarchical_transformer":
        return HTRGradientAttentionEncoder(config, device)
    if config.encoder_architecture == "ncf_token_attention":
        return HierarchicalTokenAttentionEncoder(config, device)
    raise ValueError(f"Unsupported encoder_architecture: {config.encoder_architecture!r}")


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class NuisanceTextModel(nn.Module):
    def __init__(
        self,
        config: NeuralCausalForestConfig,
        device: torch.device | str,
        outcome_type: OutcomeType = "binary",
    ) -> None:
        super().__init__()
        self.config = config
        self.outcome_type = outcome_type
        self.encoder = make_ncf_text_encoder(config, device)
        self.shared = nn.Sequential(
            nn.Linear(config.representation_dim, config.nuisance_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.nuisance_hidden_dim),
        )
        self.propensity = nn.Linear(config.nuisance_hidden_dim, 1)
        self.outcome = nn.Linear(config.nuisance_hidden_dim, 1)
        self.to(torch.device(device))

    def forward(
        self,
        texts: Sequence[str],
        *,
        return_attention_tensors: bool = False,
    ) -> Dict[str, Any]:
        encoder_output = self.encoder(texts, return_attention_tensors=return_attention_tensors)
        representation = encoder_output.representation
        hidden = self.shared(representation)
        return {
            "propensity_logit": self.propensity(hidden).squeeze(-1),
            "outcome_raw": self.outcome(hidden).squeeze(-1),
            "encoder_output": encoder_output,
        }


class SoftCausalForestHead(nn.Module):
    """Differentiable ensemble of shallow soft trees for CATE prediction."""

    def __init__(self, config: NeuralCausalForestConfig, input_dim: int) -> None:
        super().__init__()
        self.config = config
        self.input_dim = int(input_dim)
        self.n_trees = int(config.n_trees)
        self.depth = int(config.depth)
        self.n_internal = 2 ** self.depth - 1
        self.n_leaves = 2 ** self.depth
        self.gate_weight = nn.Parameter(torch.empty(self.n_trees, self.n_internal, self.input_dim))
        self.gate_bias = nn.Parameter(torch.zeros(self.n_trees, self.n_internal))
        self.leaf_tau = nn.Parameter(torch.zeros(self.n_trees, self.n_leaves))
        nn.init.normal_(self.gate_weight, mean=0.0, std=0.02)
        self.register_buffer("path_directions", self._build_path_directions())
        self.register_buffer("feature_mask", self._build_feature_mask())

    def _build_feature_mask(self) -> torch.Tensor:
        rng = np.random.default_rng(int(self.config.seed) + 17)
        mask = rng.binomial(
            n=1,
            p=float(self.config.feature_subsample_fraction),
            size=(self.n_trees, self.n_internal, self.input_dim),
        ).astype(np.float32)
        # Guarantee at least one active feature per node.
        flat = mask.reshape(-1, self.input_dim)
        for row in range(flat.shape[0]):
            if flat[row].sum() == 0:
                flat[row, rng.integers(0, self.input_dim)] = 1.0
        return torch.as_tensor(mask, dtype=torch.float32)

    def _build_path_directions(self) -> torch.Tensor:
        directions = torch.full((self.n_leaves, self.n_internal), -1, dtype=torch.long)
        for leaf in range(self.n_leaves):
            node = 0
            for level in range(self.depth):
                bit = (leaf >> (self.depth - level - 1)) & 1
                directions[leaf, node] = bit
                node = 2 * node + 1 + bit
        return directions

    def path_probabilities(
        self,
        z: torch.Tensor,
        *,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_weight = self.gate_weight * self.feature_mask.to(z.device)
        gate_logits = torch.einsum("bd,tnd->btn", z, masked_weight) + self.gate_bias.unsqueeze(0)
        gate = torch.sigmoid(gate_logits / max(float(temperature), 1e-3))
        gate = torch.clamp(gate, 1e-5, 1.0 - 1e-5)
        log_gate = torch.log(gate)
        log_not_gate = torch.log1p(-gate)
        log_probs: List[torch.Tensor] = []
        directions = self.path_directions.to(z.device)
        for leaf in range(self.n_leaves):
            dir_leaf = directions[leaf]
            used = dir_leaf >= 0
            right = dir_leaf == 1
            log_p = torch.zeros(z.shape[0], self.n_trees, device=z.device, dtype=z.dtype)
            if used.any():
                node_idx = torch.where(used)[0]
                right_idx = right[node_idx]
                for local, node in enumerate(node_idx.tolist()):
                    log_p = log_p + (log_gate[:, :, node] if bool(right_idx[local]) else log_not_gate[:, :, node])
            log_probs.append(log_p)
        path_log_prob = torch.stack(log_probs, dim=-1)
        path_prob = torch.exp(path_log_prob)
        # Normalize small numerical drift.
        path_prob = path_prob / torch.clamp(path_prob.sum(dim=-1, keepdim=True), min=1e-8)
        return path_prob, gate_logits

    def forward(
        self,
        z: torch.Tensor,
        *,
        temperature: float,
        return_paths: bool = False,
    ) -> Dict[str, torch.Tensor]:
        path_prob, gate_logits = self.path_probabilities(z, temperature=temperature)
        tau_by_tree = torch.sum(path_prob * self.leaf_tau.unsqueeze(0), dim=-1)
        tau = tau_by_tree.mean(dim=-1)
        if self.config.tau_clip is not None:
            tau = torch.clamp(tau, -float(self.config.tau_clip), float(self.config.tau_clip))
        out = {"tau": tau, "tau_by_tree": tau_by_tree}
        if return_paths:
            out["path_prob"] = path_prob
            out["gate_logits"] = gate_logits
        return out

    def moment_leaf_tau(
        self,
        path_prob: torch.Tensor,
        y_residual: torch.Tensor,
        t_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        numerator = torch.sum(
            path_prob * (y_residual * t_residual).view(-1, 1, 1),
            dim=0,
        )
        denominator = torch.sum(
            path_prob * torch.square(t_residual).view(-1, 1, 1),
            dim=0,
        )
        tau = numerator / torch.clamp(denominator + float(self.config.leaf_ridge), min=1e-8)
        if self.config.tau_clip is not None:
            tau = torch.clamp(tau, -float(self.config.tau_clip), float(self.config.tau_clip))
        return tau, denominator


class NeuralCausalForestModel(nn.Module):
    def __init__(self, config: NeuralCausalForestConfig, device: torch.device | str) -> None:
        super().__init__()
        self.config = config
        self.device_obj = torch.device(device)
        self.encoder = make_ncf_text_encoder(config, self.device_obj)
        self.dropout = nn.Dropout(config.dropout)
        self.forest = SoftCausalForestHead(config, input_dim=config.representation_dim)
        self.to(self.device_obj)

    def forward(
        self,
        texts: Sequence[str],
        *,
        temperature: Optional[float] = None,
        return_attention_tensors: bool = False,
        return_paths: bool = False,
    ) -> Dict[str, Any]:
        temperature = float(self.config.temperature_end if temperature is None else temperature)
        encoder_output = self.encoder(texts, return_attention_tensors=return_attention_tensors)
        z = self.dropout(encoder_output.representation)
        forest_output = self.forest(z, temperature=temperature, return_paths=return_paths)
        forest_output["encoder_output"] = encoder_output
        forest_output["representation"] = encoder_output.representation
        return forest_output


# -----------------------------------------------------------------------------
# Losses and training
# -----------------------------------------------------------------------------


def nuisance_loss(
    model_output: Dict[str, Any],
    t: torch.Tensor,
    y: torch.Tensor,
    *,
    outcome_type: OutcomeType,
    alpha_propensity: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    prop_loss = F.binary_cross_entropy_with_logits(model_output["propensity_logit"], t)
    if outcome_type == "continuous":
        out_loss = F.mse_loss(model_output["outcome_raw"], y)
    else:
        out_loss = F.binary_cross_entropy_with_logits(model_output["outcome_raw"], y)
    loss = out_loss + float(alpha_propensity) * prop_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "propensity_loss": float(prop_loss.detach().cpu()),
        "outcome_loss": float(out_loss.detach().cpu()),
    }


def forest_r_loss(
    forest_output: Dict[str, torch.Tensor],
    y: torch.Tensor,
    t: torch.Tensor,
    e_hat: torch.Tensor,
    m_hat: torch.Tensor,
    config: NeuralCausalForestConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    tau = forest_output["tau"]
    y_residual = y - m_hat
    t_residual = t - torch.clamp(e_hat, 1e-3, 1.0 - 1e-3)
    residual = y_residual - tau * t_residual
    r_loss = torch.mean(torch.square(residual))

    path_prob = forest_output.get("path_prob")
    if path_prob is None:
        raise ValueError("forest_r_loss requires return_paths=True")
    leaf_mass = torch.sum(path_prob, dim=0)  # [trees, leaves]
    leaf_fraction = leaf_mass / torch.clamp(torch.sum(leaf_mass, dim=-1, keepdim=True), min=1e-8)
    uniform = torch.full_like(leaf_fraction, 1.0 / float(config.depth and 2 ** config.depth))
    balance = torch.mean(torch.square(leaf_fraction - uniform))
    min_mass_penalty = torch.mean(F.relu(float(config.leaf_min_mass) - leaf_mass)) / max(
        float(config.leaf_min_mass), 1.0
    )
    tau_l2 = torch.mean(torch.square(forest_output["tau_by_tree"]))

    moment_tau, moment_denom = forest_output_model_moment(forest_output, y_residual, t_residual)
    moment_mass = moment_denom / torch.clamp(moment_denom.sum(dim=-1, keepdim=True), min=1e-8)
    mean_tau = torch.sum(moment_mass * moment_tau, dim=-1, keepdim=True)
    heterogeneity = torch.mean(torch.sum(moment_mass * torch.square(moment_tau - mean_tau), dim=-1))

    loss = (
        r_loss
        + float(config.lambda_leaf_balance) * balance
        + float(config.lambda_leaf_min_mass) * min_mass_penalty
        + float(config.lambda_leaf_tau_l2) * tau_l2
        - float(config.lambda_heterogeneity) * heterogeneity
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "r_loss": float(r_loss.detach().cpu()),
        "leaf_balance": float(balance.detach().cpu()),
        "leaf_min_mass_penalty": float(min_mass_penalty.detach().cpu()),
        "leaf_tau_l2": float(tau_l2.detach().cpu()),
        "leaf_tau_heterogeneity": float(heterogeneity.detach().cpu()),
        "tau_mean": float(tau.detach().mean().cpu()),
        "tau_std": float(tau.detach().std().cpu()) if tau.numel() > 1 else 0.0,
    }
    return loss, metrics


def forest_output_model_moment(
    forest_output: Dict[str, torch.Tensor],
    y_residual: torch.Tensor,
    t_residual: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    path_prob = forest_output["path_prob"]
    numerator = torch.sum(path_prob * (y_residual * t_residual).view(-1, 1, 1), dim=0)
    denominator = torch.sum(path_prob * torch.square(t_residual).view(-1, 1, 1), dim=0)
    tau = numerator / torch.clamp(denominator + 1e-3, min=1e-8)
    return tau, denominator


def train_nuisance_model(
    model: NuisanceTextModel,
    df: pd.DataFrame,
    *,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    outcome_type: OutcomeType,
    config: NeuralCausalForestConfig,
    device: torch.device,
    row_id_column: str = "_ncf_row_id",
) -> Dict[str, Any]:
    loader = make_text_loader(
        df[text_column].astype(str).tolist(),
        df[row_id_column].tolist(),
        treatment=df[treatment_column].to_numpy(dtype=np.float32),
        outcome=df[outcome_column].to_numpy(dtype=np.float32),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.nuisance_learning_rate,
        weight_decay=config.nuisance_weight_decay,
    )
    scheduler = _make_linear_scheduler(optimizer, config.nuisance_epochs, max(1, len(loader)))
    history: List[Dict[str, float]] = []
    for epoch in range(1, config.nuisance_epochs + 1):
        model.train()
        sums: Dict[str, float] = {"loss": 0.0, "propensity_loss": 0.0, "outcome_loss": 0.0}
        count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            t = batch["t"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            out = model(batch["texts"])
            loss, metrics = nuisance_loss(
                out,
                t,
                y,
                outcome_type=outcome_type,
                alpha_propensity=config.alpha_propensity,
            )
            loss.backward()
            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            count += 1
            for key in sums:
                sums[key] += metrics[key]
        row = {key: value / max(1, count) for key, value in sums.items()}
        row["epoch"] = float(epoch)
        row["lr"] = _current_lr(optimizer)
        history.append(row)
        logger.info(
            "nuisance epoch %d/%d loss=%.4f propensity=%.4f outcome=%.4f lr=%.3g",
            epoch,
            config.nuisance_epochs,
            row["loss"],
            row["propensity_loss"],
            row["outcome_loss"],
            row["lr"],
        )
    return {"history": history}


@torch.no_grad()
def predict_nuisance_model(
    model: NuisanceTextModel,
    df: pd.DataFrame,
    *,
    text_column: str,
    outcome_type: OutcomeType,
    config: NeuralCausalForestConfig,
    device: torch.device,
    row_id_column: str = "_ncf_row_id",
) -> pd.DataFrame:
    model.eval()
    loader = make_text_loader(
        df[text_column].astype(str).tolist(),
        df[row_id_column].tolist(),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    rows: List[pd.DataFrame] = []
    for batch in loader:
        out = model(batch["texts"])
        e_hat = torch.sigmoid(out["propensity_logit"]).detach().cpu().numpy()
        if outcome_type == "continuous":
            m_hat = out["outcome_raw"].detach().cpu().numpy()
        else:
            m_hat = torch.sigmoid(out["outcome_raw"]).detach().cpu().numpy()
        rows.append(
            pd.DataFrame(
                {
                    row_id_column: batch["row_ids"],
                    "e_hat": e_hat,
                    "m_hat": m_hat,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _resolve_inner_fold_parallelism(
    config: NeuralCausalForestConfig,
    folds: int,
    device: torch.device,
) -> int:
    setting = str(config.inner_fold_parallelism).strip().lower()
    if setting == "auto":
        if device.type != "cpu":
            return 1
        return max(1, min(int(config.num_workers or 1), int(folds)))
    return max(1, min(int(setting), int(folds)))


def _run_ncf_inner_fold_tasks(run_fold, split_items, n_jobs: int) -> List[Dict[str, Any]]:
    if n_jobs <= 1:
        return [
            run_fold(fold, fit_idx, heldout_idx)
            for fold, (fit_idx, heldout_idx) in split_items
        ]
    with ThreadPoolExecutor(
        max_workers=int(n_jobs),
        thread_name_prefix="ncf-nuisance-fold",
    ) as executor:
        futures = [
            executor.submit(run_fold, fold, fit_idx, heldout_idx)
            for fold, (fit_idx, heldout_idx) in split_items
        ]
        return [future.result() for future in futures]


def crossfit_nuisance_predictions(
    df: pd.DataFrame,
    *,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    outcome_type: OutcomeType,
    config: NeuralCausalForestConfig,
    device: torch.device,
    row_id_column: str = "_ncf_row_id",
    collect_attention: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """Cross-fit nuisance functions and optionally collect held-out attention."""
    set_global_seed(config.seed)
    df = df.reset_index(drop=True).copy()
    if row_id_column not in df.columns:
        df[row_id_column] = np.arange(len(df), dtype=int)
    folds = min(max(2, int(config.nuisance_folds)), len(df))
    if df[treatment_column].nunique(dropna=True) == 2 and len(df) >= folds * 2:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.seed + 101)
        split_iter = splitter.split(df, df[treatment_column].astype(int).to_numpy())
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=config.seed + 101)
        split_iter = splitter.split(df)

    split_items = list(enumerate(split_iter, start=1))
    n_jobs = _resolve_inner_fold_parallelism(config, folds, device)
    logger.info(
        "nuisance cross-fit parallelism: folds=%d n_jobs=%d setting=%s device=%s",
        folds,
        n_jobs,
        config.inner_fold_parallelism,
        device,
    )

    def run_fold(fold: int, fit_idx: np.ndarray, heldout_idx: np.ndarray) -> Dict[str, Any]:
        logger.info("nuisance fold %d/%d: train=%d heldout=%d", fold, folds, len(fit_idx), len(heldout_idx))
        fold_config = replace(config, num_workers=0) if n_jobs > 1 and config.num_workers else config
        model = NuisanceTextModel(fold_config, device=device, outcome_type=outcome_type)
        try:
            train_result = train_nuisance_model(
                model,
                df.iloc[fit_idx].reset_index(drop=True),
                text_column=text_column,
                treatment_column=treatment_column,
                outcome_column=outcome_column,
                outcome_type=outcome_type,
                config=fold_config,
                device=device,
                row_id_column=row_id_column,
            )
            pred = predict_nuisance_model(
                model,
                df.iloc[heldout_idx].reset_index(drop=True),
                text_column=text_column,
                outcome_type=outcome_type,
                config=fold_config,
                device=device,
                row_id_column=row_id_column,
            )
            pred["nuisance_fold"] = fold
            fold_attention: List[Dict[str, Any]] = []
            if collect_attention:
                heldout_df = df.iloc[heldout_idx].reset_index(drop=True)
                fold_attention = nuisance_attention_evidence(
                    model,
                    heldout_df[text_column].astype(str).tolist(),
                    row_ids=heldout_df[row_id_column].tolist(),
                    config=fold_config,
                    stage="nuisance",
                    top_k=fold_config.attention_top_k,
                    metadata=[{"nuisance_fold": fold} for _ in range(len(heldout_df))],
                )
            return {
                "fold": fold,
                "predictions": pred,
                "history": train_result["history"],
                "attention": fold_attention,
            }
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fold_results = _run_ncf_inner_fold_tasks(run_fold, split_items, n_jobs)

    prediction_rows: List[pd.DataFrame] = []
    history_rows: List[Dict[str, Any]] = []
    attention_rows: List[Dict[str, Any]] = []
    for result in fold_results:
        fold = int(result["fold"])
        prediction_rows.append(result["predictions"])
        for row in result["history"]:
            history_rows.append({"fold": fold, **row})
        attention_rows.extend(result["attention"])

    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values(row_id_column)
    predictions = predictions.merge(
        df[[row_id_column, treatment_column, outcome_column]],
        on=row_id_column,
        how="left",
    )
    predictions["y_residual"] = predictions[outcome_column].astype(float) - predictions["m_hat"].astype(float)
    predictions["t_residual"] = predictions[treatment_column].astype(float) - predictions["e_hat"].astype(float)
    predictions["r_loss_at_zero_tau"] = predictions["y_residual"] ** 2
    history = pd.DataFrame(history_rows)
    return predictions.reset_index(drop=True), history, attention_rows


def train_neural_causal_forest(
    model: NeuralCausalForestModel,
    df: pd.DataFrame,
    nuisance_predictions: pd.DataFrame,
    *,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    config: NeuralCausalForestConfig,
    device: torch.device,
    row_id_column: str = "_ncf_row_id",
) -> Dict[str, Any]:
    """Train soft tree structure by R-loss, then optionally refit honest leaves."""
    df = df.reset_index(drop=True).copy()
    if row_id_column not in df.columns:
        df[row_id_column] = np.arange(len(df), dtype=int)
    nuisance = nuisance_predictions.set_index(row_id_column).loc[df[row_id_column]].reset_index()

    # Honest structure-vs-estimation split.  The structure sample learns gates;
    # the estimation sample is held back for closed-form leaf CATE refitting.
    all_pos = np.arange(len(df), dtype=int)
    if len(df) >= 4:
        train_size = max(1, int(round(len(df) * (1.0 - config.honesty_fraction))))
        train_size = min(train_size, len(df) - 1)
        stratify = None
        if df[treatment_column].nunique() == 2:
            counts = df[treatment_column].astype(int).value_counts()
            if int(counts.min()) >= 2:
                stratify = df[treatment_column].astype(int).to_numpy()
        try:
            structure_pos, estimate_pos = train_test_split(
                all_pos,
                train_size=train_size,
                random_state=config.seed + 303,
                shuffle=True,
                stratify=stratify,
            )
        except ValueError:
            structure_pos, estimate_pos = train_test_split(
                all_pos,
                train_size=train_size,
                random_state=config.seed + 303,
                shuffle=True,
                stratify=None,
            )
    else:
        structure_pos = all_pos
        estimate_pos = all_pos

    structure_df = df.iloc[structure_pos].reset_index(drop=True)
    structure_nuisance = nuisance.iloc[structure_pos].reset_index(drop=True)
    loader = make_text_loader(
        structure_df[text_column].astype(str).tolist(),
        structure_df[row_id_column].tolist(),
        treatment=structure_df[treatment_column].to_numpy(dtype=np.float32),
        outcome=structure_df[outcome_column].to_numpy(dtype=np.float32),
        e_hat=structure_nuisance["e_hat"].to_numpy(dtype=np.float32),
        m_hat=structure_nuisance["m_hat"].to_numpy(dtype=np.float32),
        batch_size=config.effect_batch_size or config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.forest_learning_rate,
        weight_decay=config.forest_weight_decay,
    )
    scheduler = _make_linear_scheduler(optimizer, config.forest_epochs, max(1, len(loader)))
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.forest_epochs + 1):
        temperature = _linear_temperature(config, epoch)
        model.train()
        sums: Dict[str, float] = {}
        count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            y = batch["y"].to(device, non_blocking=True)
            t = batch["t"].to(device, non_blocking=True)
            e_hat = batch["e_hat"].to(device, non_blocking=True)
            m_hat = batch["m_hat"].to(device, non_blocking=True)
            out = model(batch["texts"], temperature=temperature, return_paths=True)
            loss, metrics = forest_r_loss(out, y, t, e_hat, m_hat, config)
            loss.backward()
            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            count += 1
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value
        row = {key: value / max(1, count) for key, value in sums.items()}
        row["epoch"] = float(epoch)
        row["temperature"] = temperature
        row["lr"] = _current_lr(optimizer)
        history.append(row)
        logger.info(
            "forest epoch %d/%d loss=%.4f r_loss=%.4f tau_std=%.4f het=%.4f temp=%.3f lr=%.3g",
            epoch,
            config.forest_epochs,
            row.get("loss", float("nan")),
            row.get("r_loss", float("nan")),
            row.get("tau_std", float("nan")),
            row.get("leaf_tau_heterogeneity", float("nan")),
            temperature,
            row["lr"],
        )

    leaf_refit = None
    if config.refit_leaf_values_after_training:
        estimate_df = df.iloc[estimate_pos].reset_index(drop=True)
        estimate_nuisance = nuisance.iloc[estimate_pos].reset_index(drop=True)
        leaf_refit = refit_honest_leaf_values(
            model,
            estimate_df,
            estimate_nuisance,
            text_column=text_column,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            config=config,
            device=device,
            row_id_column=row_id_column,
        )
    return {
        "history": pd.DataFrame(history),
        "structure_rows": int(len(structure_pos)),
        "honest_estimation_rows": int(len(estimate_pos)),
        "leaf_refit": leaf_refit,
    }


@torch.no_grad()
def refit_honest_leaf_values(
    model: NeuralCausalForestModel,
    df: pd.DataFrame,
    nuisance_predictions: pd.DataFrame,
    *,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    config: NeuralCausalForestConfig,
    device: torch.device,
    row_id_column: str = "_ncf_row_id",
) -> Dict[str, Any]:
    model.eval()
    loader = make_text_loader(
        df[text_column].astype(str).tolist(),
        df[row_id_column].tolist(),
        treatment=df[treatment_column].to_numpy(dtype=np.float32),
        outcome=df[outcome_column].to_numpy(dtype=np.float32),
        e_hat=nuisance_predictions["e_hat"].to_numpy(dtype=np.float32),
        m_hat=nuisance_predictions["m_hat"].to_numpy(dtype=np.float32),
        batch_size=config.effect_batch_size or config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    n_trees = model.forest.n_trees
    n_leaves = model.forest.n_leaves
    numerator = torch.zeros(n_trees, n_leaves, device=device)
    denominator = torch.zeros(n_trees, n_leaves, device=device)
    mass = torch.zeros(n_trees, n_leaves, device=device)
    global_num = torch.tensor(0.0, device=device)
    global_den = torch.tensor(0.0, device=device)
    for batch in loader:
        y = batch["y"].to(device, non_blocking=True)
        t = batch["t"].to(device, non_blocking=True)
        e_hat = torch.clamp(batch["e_hat"].to(device, non_blocking=True), 1e-3, 1.0 - 1e-3)
        m_hat = batch["m_hat"].to(device, non_blocking=True)
        y_resid = y - m_hat
        t_resid = t - e_hat
        out = model(
            batch["texts"],
            temperature=config.temperature_end,
            return_paths=True,
            return_attention_tensors=False,
        )
        path_prob = out["path_prob"]
        numerator += torch.sum(path_prob * (y_resid * t_resid).view(-1, 1, 1), dim=0)
        denominator += torch.sum(path_prob * torch.square(t_resid).view(-1, 1, 1), dim=0)
        mass += torch.sum(path_prob, dim=0)
        global_num += torch.sum(y_resid * t_resid)
        global_den += torch.sum(torch.square(t_resid))

    global_tau = global_num / torch.clamp(global_den + config.leaf_ridge, min=1e-8)
    raw_tau = numerator / torch.clamp(denominator + config.leaf_ridge, min=1e-8)
    shrinkage = denominator / torch.clamp(denominator + float(config.leaf_min_mass) * config.leaf_ridge, min=1e-8)
    refit_tau = shrinkage * raw_tau + (1.0 - shrinkage) * global_tau
    if config.tau_clip is not None:
        refit_tau = torch.clamp(refit_tau, -float(config.tau_clip), float(config.tau_clip))
    model.forest.leaf_tau.copy_(refit_tau)
    return {
        "global_tau": float(global_tau.detach().cpu()),
        "mean_leaf_mass": float(mass.detach().mean().cpu()),
        "min_leaf_mass": float(mass.detach().min().cpu()),
        "max_leaf_mass": float(mass.detach().max().cpu()),
        "leaf_tau_mean": float(refit_tau.detach().mean().cpu()),
        "leaf_tau_std": float(refit_tau.detach().std().cpu()) if refit_tau.numel() > 1 else 0.0,
    }


@torch.no_grad()
def predict_neural_causal_forest(
    model: NeuralCausalForestModel,
    df: pd.DataFrame,
    *,
    text_column: str,
    config: NeuralCausalForestConfig,
    device: torch.device,
    row_id_column: str = "_ncf_row_id",
) -> pd.DataFrame:
    model.eval()
    loader = make_text_loader(
        df[text_column].astype(str).tolist(),
        df[row_id_column].tolist(),
        batch_size=config.effect_batch_size or config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    frames: List[pd.DataFrame] = []
    for batch in loader:
        out = model(batch["texts"], temperature=config.temperature_end, return_paths=False)
        tau = out["tau"].detach().cpu().numpy()
        tau_by_tree = out["tau_by_tree"].detach().cpu().numpy()
        frames.append(
            pd.DataFrame(
                {
                    row_id_column: batch["row_ids"],
                    "tau_hat_ncf": tau,
                    "tau_hat_tree_mean": tau_by_tree.mean(axis=1),
                    "tau_hat_tree_std": tau_by_tree.std(axis=1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# -----------------------------------------------------------------------------
# Token evidence / attribution
# -----------------------------------------------------------------------------


def nuisance_attention_evidence(
    model: NuisanceTextModel,
    texts: Sequence[str],
    *,
    row_ids: Sequence[Any],
    config: NeuralCausalForestConfig,
    stage: str = "nuisance",
    top_k: Optional[int] = None,
    metadata: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Token evidence for nuisance prediction signal.

    The target is a sum of absolute treatment and outcome logits.  This tends to
    highlight confounding/prognostic variables such as age without asking the
    model to regress noisy residual labels.
    """
    model.eval()
    device = next(model.parameters()).device
    records: List[Dict[str, Any]] = []
    top_k = int(top_k or config.attention_top_k)
    metadata = list(metadata or [{} for _ in texts])
    batch_size = max(1, int(config.evidence_batch_size))
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = [str(text or "") for text in texts[start:end]]
        batch_meta = metadata[start:end]
        model.zero_grad(set_to_none=True)
        out = model(batch_texts, return_attention_tensors=True)
        target = torch.sum(torch.abs(out["propensity_logit"]) + torch.abs(out["outcome_raw"]))
        target.backward()
        enc = out["encoder_output"]
        token_scores = _grad_x_attention_token_scores(enc, device=device)
        records.extend(
            model.encoder.attention_records_from_output(
                enc,
                row_ids=row_ids[start:end],
                stage=stage,
                top_k=top_k,
                metadata=batch_meta,
                token_scores=token_scores,
            )
        )
    return records


def causal_forest_attention_evidence(
    model: NeuralCausalForestModel,
    texts: Sequence[str],
    *,
    row_ids: Sequence[Any],
    config: NeuralCausalForestConfig,
    stage: str = "effect_modifier",
    top_k: Optional[int] = None,
    metadata: Optional[Sequence[Dict[str, Any]]] = None,
    target: Literal["tau_abs", "tau_heterogeneity", "tau_signed"] = "tau_heterogeneity",
) -> List[Dict[str, Any]]:
    """Token evidence for the neural causal forest CATE/gate signal.

    ``tau_heterogeneity`` is the default because effect modifiers are variables
    that move tau around, not necessarily variables that push tau in a fixed
    sign.  It backpropagates the within-batch tau variance through the soft-tree
    gates and token attention.
    """
    model.eval()
    device = next(model.parameters()).device
    records: List[Dict[str, Any]] = []
    top_k = int(top_k or config.attention_top_k)
    metadata = list(metadata or [{} for _ in texts])
    batch_size = max(1, int(config.evidence_batch_size))
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = [str(text or "") for text in texts[start:end]]
        batch_meta = metadata[start:end]
        model.zero_grad(set_to_none=True)
        out = model(
            batch_texts,
            temperature=config.temperature_end,
            return_attention_tensors=True,
            return_paths=True,
        )
        tau = out["tau"]
        if target == "tau_signed":
            objective = torch.sum(tau)
        elif target == "tau_abs":
            objective = torch.sum(torch.abs(tau))
        else:
            objective = torch.sum(torch.square(tau - tau.detach().mean()))
            if float(objective.detach().cpu()) <= 1e-12:
                objective = torch.sum(torch.abs(tau))
        objective.backward()
        enc = out["encoder_output"]
        token_scores = _grad_x_attention_token_scores(enc, device=device)
        for idx, meta in enumerate(batch_meta):
            meta.setdefault("tau_hat_ncf", float(tau.detach().cpu().numpy()[idx]))
        records.extend(
            model.encoder.attention_records_from_output(
                enc,
                row_ids=row_ids[start:end],
                stage=stage,
                top_k=top_k,
                metadata=batch_meta,
                token_scores=token_scores,
            )
        )
    return records


def _grad_x_attention_token_scores(
    enc: EncoderForwardOutput,
    *,
    device: torch.device,
) -> torch.Tensor:
    if enc.token_alpha is None:
        raise ValueError("Encoder output does not include token attention tensors")
    source_scores = _grad_x_attention_source_scores(enc)
    if source_scores is not None:
        token_scores = source_scores
    else:
        token_grad = enc.token_alpha.grad
        if token_grad is None:
            token_scores = enc.token_alpha.detach()
        else:
            token_scores = torch.abs(token_grad.detach() * enc.token_alpha.detach())
    if enc.chunk_alpha is not None and enc.flat_chunk_patient_index is not None and enc.flat_chunk_local_index is not None:
        chunk_scores = enc.chunk_alpha.detach()[enc.flat_chunk_patient_index, enc.flat_chunk_local_index]
        token_scores = token_scores * chunk_scores.unsqueeze(-1)
    if enc.attention_mask is not None:
        token_scores = token_scores.masked_fill(enc.attention_mask <= 0, 0.0)
    return token_scores.to(device)


def _grad_x_attention_source_scores(enc: EncoderForwardOutput) -> Optional[torch.Tensor]:
    """Gradient x attention for original token-attention tensors before padding.

    HTR encodes chunks in mini-batches.  The padded token_alpha tensor is only a
    reporting view, while the per-mini-batch tensors are the ones used to pool
    chunk embeddings.  Reading gradients from those source tensors preserves
    actual gradient-based attribution.
    """
    if not enc.token_alpha_sources:
        return None
    if enc.token_alpha is None:
        return None
    max_len = int(enc.token_alpha.shape[1])
    rows: List[torch.Tensor] = []
    for source in enc.token_alpha_sources:
        if source is None:
            continue
        grad = source.grad
        if grad is None:
            score = source.detach()
        else:
            score = torch.abs(grad.detach() * source.detach())
        if int(score.shape[1]) < max_len:
            score = F.pad(score, (0, max_len - int(score.shape[1])), value=0.0)
        elif int(score.shape[1]) > max_len:
            score = score[:, :max_len]
        rows.append(score)
    if not rows:
        return None
    token_scores = torch.cat(rows, dim=0)
    if token_scores.shape != enc.token_alpha.shape:
        return None
    return token_scores


# -----------------------------------------------------------------------------
# End-to-end pipeline helpers
# -----------------------------------------------------------------------------


@dataclass
class FitPipelineResult:
    model: NeuralCausalForestModel
    nuisance_predictions: pd.DataFrame
    nuisance_history: pd.DataFrame
    forest_history: pd.DataFrame
    train_predictions: pd.DataFrame
    nuisance_attention: pd.DataFrame
    effect_attention: pd.DataFrame
    metrics: Dict[str, Any]
    leaf_refit: Optional[Dict[str, Any]] = None


def fit_neural_causal_forest_pipeline(
    df: pd.DataFrame,
    *,
    text_column: str = "clinical_text",
    treatment_column: str = "treatment_indicator",
    outcome_column: str = "outcome_indicator",
    outcome_type: OutcomeType = "binary",
    config: Optional[NeuralCausalForestConfig] = None,
    device: torch.device | str = "cpu",
    row_id_column: str = "_ncf_row_id",
    collect_attention: bool = True,
    nuisance_artifact_dir: Optional[str | Path] = None,
) -> FitPipelineResult:
    config = config or NeuralCausalForestConfig()
    config.__post_init__()
    set_global_seed(config.seed)
    device_obj = torch.device(device)
    df = df.reset_index(drop=True).copy()
    if row_id_column not in df.columns:
        df[row_id_column] = np.arange(len(df), dtype=int)
    required = {text_column, treatment_column, outcome_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    nuisance_predictions, nuisance_history, nuisance_attention_rows = crossfit_nuisance_predictions(
        df,
        text_column=text_column,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        outcome_type=outcome_type,
        config=config,
        device=device_obj,
        row_id_column=row_id_column,
        collect_attention=collect_attention,
    )
    if nuisance_artifact_dir is not None:
        nuisance_dir = Path(nuisance_artifact_dir)
        write_dataframe(nuisance_predictions, nuisance_dir / "train_nuisance_predictions.parquet")
        write_dataframe(nuisance_history, nuisance_dir / "nuisance_history.parquet")
    model = NeuralCausalForestModel(config, device=device_obj)
    train_result = train_neural_causal_forest(
        model,
        df,
        nuisance_predictions,
        text_column=text_column,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        config=config,
        device=device_obj,
        row_id_column=row_id_column,
    )
    train_predictions = predict_neural_causal_forest(
        model,
        df,
        text_column=text_column,
        config=config,
        device=device_obj,
        row_id_column=row_id_column,
    )
    merged = df[[row_id_column, treatment_column, outcome_column]].merge(
        nuisance_predictions[[row_id_column, "e_hat", "m_hat", "y_residual", "t_residual"]],
        on=row_id_column,
        how="left",
    ).merge(train_predictions, on=row_id_column, how="left")
    merged["r_loss_ncf"] = (
        merged["y_residual"].astype(float)
        - merged["tau_hat_ncf"].astype(float) * merged["t_residual"].astype(float)
    ) ** 2
    merged["r_loss_at_zero_tau"] = merged["y_residual"].astype(float) ** 2
    train_predictions = train_predictions.merge(
        merged[[row_id_column, "r_loss_ncf", "r_loss_at_zero_tau"]],
        on=row_id_column,
        how="left",
    )

    effect_attention_rows: List[Dict[str, Any]] = []
    if collect_attention:
        pred_lookup = train_predictions.set_index(row_id_column)
        metadata = [
            {
                "tau_hat_ncf": float(pred_lookup.loc[row_id, "tau_hat_ncf"]),
                "split": "train",
            }
            for row_id in df[row_id_column].tolist()
        ]
        effect_attention_rows = causal_forest_attention_evidence(
            model,
            df[text_column].astype(str).tolist(),
            row_ids=df[row_id_column].tolist(),
            config=config,
            stage="effect_modifier",
            top_k=config.attention_top_k,
            metadata=metadata,
            target="tau_heterogeneity",
        )

    metrics = summarize_pipeline_metrics(
        df=df,
        predictions=train_predictions.merge(
            nuisance_predictions[[row_id_column, "e_hat", "m_hat"]], on=row_id_column, how="left"
        ),
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        outcome_type=outcome_type,
        row_id_column=row_id_column,
    )
    metrics.update(
        {
            "n_rows": int(len(df)),
            "structure_rows": int(train_result["structure_rows"]),
            "honest_estimation_rows": int(train_result["honest_estimation_rows"]),
        }
    )
    if train_result.get("leaf_refit"):
        metrics.update({f"leaf_refit_{k}": v for k, v in train_result["leaf_refit"].items()})

    return FitPipelineResult(
        model=model,
        nuisance_predictions=nuisance_predictions,
        nuisance_history=nuisance_history,
        forest_history=train_result["history"],
        train_predictions=train_predictions,
        nuisance_attention=pd.DataFrame(nuisance_attention_rows),
        effect_attention=pd.DataFrame(effect_attention_rows),
        metrics=metrics,
        leaf_refit=train_result.get("leaf_refit"),
    )


def summarize_pipeline_metrics(
    *,
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    outcome_type: OutcomeType,
    row_id_column: str = "_ncf_row_id",
) -> Dict[str, Any]:
    source = df[[row_id_column, treatment_column, outcome_column]].merge(
        predictions,
        on=row_id_column,
        how="left",
        suffixes=("", "_pred"),
    )
    metrics: Dict[str, Any] = {
        "tau_hat_mean": _finite_or_none(source["tau_hat_ncf"].mean()),
        "tau_hat_std": _finite_or_none(source["tau_hat_ncf"].std()),
        "r_loss_mean": _finite_or_none(source.get("r_loss_ncf", pd.Series(dtype=float)).mean()),
        "r_loss_at_zero_tau_mean": _finite_or_none(
            source.get("r_loss_at_zero_tau", pd.Series(dtype=float)).mean()
        ),
    }
    if metrics["r_loss_at_zero_tau_mean"] and metrics["r_loss_at_zero_tau_mean"] > 0:
        metrics["r_loss_relative_improvement"] = float(
            1.0 - metrics["r_loss_mean"] / metrics["r_loss_at_zero_tau_mean"]
        )
    if "e_hat" in source:
        metrics["propensity_auroc"] = _safe_roc_auc(
            source[treatment_column].to_numpy(dtype=float), source["e_hat"].to_numpy(dtype=float)
        )
    if "m_hat" in source:
        if outcome_type == "continuous":
            metrics["outcome_rmse"] = float(
                math.sqrt(
                    mean_squared_error(
                        source[outcome_column].to_numpy(dtype=float),
                        source["m_hat"].to_numpy(dtype=float),
                    )
                )
            )
        else:
            metrics["outcome_auroc"] = _safe_roc_auc(
                source[outcome_column].to_numpy(dtype=float), source["m_hat"].to_numpy(dtype=float)
            )
    for true_col in ("true_ite_prob", "true_ite", "tau", "true_tau"):
        if true_col in df.columns:
            source = source.merge(df[[row_id_column, true_col]], on=row_id_column, how="left")
            metrics[f"{true_col}_corr"] = _safe_corr(
                source[true_col].to_numpy(dtype=float), source["tau_hat_ncf"].to_numpy(dtype=float)
            )
            try:
                from scipy import stats

                rho, _ = stats.spearmanr(
                    source[true_col].to_numpy(dtype=float), source["tau_hat_ncf"].to_numpy(dtype=float)
                )
                metrics[f"{true_col}_spearman"] = _finite_or_none(rho)
            except Exception:
                metrics[f"{true_col}_spearman"] = None
            metrics[f"{true_col}_rmse"] = float(
                math.sqrt(
                    mean_squared_error(
                        source[true_col].to_numpy(dtype=float),
                        source["tau_hat_ncf"].to_numpy(dtype=float),
                    )
                )
            )
            break
    return metrics


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


# -----------------------------------------------------------------------------
# Saving/loading
# -----------------------------------------------------------------------------


def save_neural_causal_forest_model(
    model: NeuralCausalForestModel,
    output_dir: str | Path,
    *,
    config: NeuralCausalForestConfig,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(output_dir / "neural_causal_forest_config.json")
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "metadata": metadata or {},
    }
    torch.save(payload, output_dir / "neural_causal_forest.pt")


def load_neural_causal_forest_model(
    model_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> Tuple[NeuralCausalForestModel, NeuralCausalForestConfig, Dict[str, Any]]:
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / "neural_causal_forest.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    config_payload = checkpoint.get("config")
    if config_payload is None:
        config = NeuralCausalForestConfig.from_json(model_dir / "neural_causal_forest_config.json")
    else:
        config = NeuralCausalForestConfig(**config_payload)
    model = NeuralCausalForestModel(config, device=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config, dict(checkpoint.get("metadata") or {})


# -----------------------------------------------------------------------------
# Agent-context helpers
# -----------------------------------------------------------------------------


EFFECT_MODIFIER_PATTERNS = {
    "pdl1": re.compile(r"\b(?:pd[-\s]?l1|programmed\s+death[-\s]?ligand\s*1)\b", re.I),
    "pdl1_high": re.compile(r"\b(?:pd[-\s]?l1|tps|tumou?r\s+proportion).{0,40}\b(?:high|>=?\s*50|50\s*%|positive)\b", re.I),
}
CONFOUNDER_PATTERNS = {
    "age": re.compile(r"\b(?:age|aged|years?\s+old|yo|y/o|\d{2}\s*[- ]?year[- ]?old)\b", re.I),
}


def add_oracle_attention_hits(evidence: pd.DataFrame) -> pd.DataFrame:
    """Annotate evidence rows with simple oracle term hits for synthetic NSCLC data."""
    if evidence.empty:
        return evidence
    evidence = evidence.copy()
    text = (
        evidence.get("token_text", pd.Series("", index=evidence.index)).fillna("").astype(str)
        + " "
        + evidence.get("snippet", pd.Series("", index=evidence.index)).fillna("").astype(str)
    )
    for name, pattern in {**EFFECT_MODIFIER_PATTERNS, **CONFOUNDER_PATTERNS}.items():
        evidence[f"hit_{name}"] = text.apply(lambda value: bool(pattern.search(value)))
    return evidence


def build_agent_context_rows(
    evidence: pd.DataFrame,
    *,
    stage: str,
    max_rows: int = 80,
    min_abs_score_quantile: float = 0.50,
) -> List[Dict[str, Any]]:
    """Build compact JSON-serializable rows that can be sent to a proposal agent."""
    if evidence.empty:
        return []
    frame = evidence.copy()
    if "stage" in frame.columns:
        frame = frame[frame["stage"].astype(str) == str(stage)].copy()
    if frame.empty:
        return []
    frame["abs_score"] = frame["evidence_score"].astype(float).abs()
    threshold = frame["abs_score"].quantile(float(min_abs_score_quantile))
    frame = frame[frame["abs_score"] >= threshold].sort_values("abs_score", ascending=False)
    rows = []
    for _, record in frame.head(int(max_rows)).iterrows():
        rows.append(
            {
                "row_id": record.get("row_id"),
                "stage": record.get("stage", stage),
                "rank_within_patient": int(record.get("rank_within_patient", 0) or 0),
                "token_text": str(record.get("token_text", "")),
                "snippet": str(record.get("snippet", "")),
                "evidence_score": _finite_or_none(record.get("evidence_score")),
                "tau_hat_ncf": _finite_or_none(record.get("tau_hat_ncf")),
            }
        )
    return rows
