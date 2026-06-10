#!/usr/bin/env python
"""Oracle runner for the R-learner representation -> causal forest X/W split.

This is a narrowed variant of run_oracle_experiments.py.  It only evaluates the
two-stage CausalTextForest path where Stage 1 trains an R-learner representation
and Stage 2 fits EconML CausalForestDML with separate X and W matrices:

- X: effect-modifier branch activations plus explicit features with the
  "effect_modifier" role.
- W: nuisance branch activations plus explicit features with the "confounder"
  role.

LLM hidden-state downprojection is intentionally disabled for the LLM-based
extractors: flp_downprojection_dim=None and hlm_downprojection_dim=None.

Output is compatible with the existing oracle result directory layout.
"""

import argparse
import gc
import hashlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import threading
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.config import ContrastiveEffectConfig, ExplicitFeatureSpec, TRAINABLE_EXTRACTOR_TYPES
from oci.data import (
    CachedHiddenStateDataset,
    ClinicalTextDataset,
    collate_batch,
    collate_cached_batch,
    prepare_cached_batch,
)
from oci.models.causal_text_forest import CausalTextForest
from oci.models.contrastive_causal_text_forest import ContrastiveCausalTextForest
from oci.models.gpu_hidden_state_store import GPUHiddenStateStore
from oci.models.hidden_state_cache import HiddenStateCache
from oci.training.contrastive_effect import (
    PropensityBinBalancedBatchSampler,
    make_propensity_bins,
)

from run_oracle_experiments import (
    _common_model_kwargs,
    _get_cache_info,
    _open_cache_for_worker,
    _resolve_parquet_file,
    compute_metrics,
    group_configs_by_cache_key,
    load_single_gpu_store,
    precompute_single_cache,
    resolve_workers_per_gpu,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class XWRLearnerForestConfig:
    """Configuration for one X/W R-learner causal forest experiment."""

    dataset_path: str
    dataset_name: str

    # Fixed by this runner. Kept in the serialized config for analyzer parity.
    model_type: str = "causal_forest"
    rlearner_mode: str = "staged_separate_nets"
    xw_feature_split: bool = True
    cf_rlearner_representation_mode: Optional[str] = None
    shared_rlearner_nuisance_source: str = "inner_oof"
    use_explicit_features: bool = False
    # Compatibility alias for older analysis scripts.
    use_explicit_confounders: bool = False

    feature_extractor_type: str = "frozen_llm_pooler"
    repeat_index: int = 0

    # Frozen LLM Pooler hyperparameters.
    flp_max_length: int = 50000
    flp_freeze_llm: bool = True
    flp_projection_dim: int = 128
    flp_gated_attention_dim: int = 128
    flp_downprojection_dim: Optional[int] = None
    flp_cache_hidden_states: bool = False
    flp_chat_template_prompt: Optional[str] = None
    flp_model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    flp_dropout: float = 0.1
    flp_gradient_checkpointing: bool = True

    # Fixed training parameters.
    epochs: int = 30
    batch_size: int = 2
    learning_rate: float = 1e-4
    n_folds: int = 5
    rlearner_nuisance_folds: int = 5
    gamma_rlearner: float = 1.0
    rlearner_effect_batch_size: Optional[int] = None
    rlearner_effect_accumulation_steps: int = 1
    rlearner_effect_e_clip: float = 0.01
    rlearner_effect_grad_clip: float = 1.0

    # Optional matched-contrastive X-stage replacement for per-patient R-loss.
    contrastive_effect_enabled: bool = False
    contrastive_bottleneck_dim: int = 8
    contrastive_hidden_dim: int = 64
    contrastive_batch_size: int = 16
    contrastive_n_propensity_bins: int = 10
    contrastive_overlap_min: float = 0.05
    contrastive_overlap_max: float = 0.95
    contrastive_min_arm_per_bin: int = 2
    contrastive_lambda_factual: float = 1.0
    contrastive_lambda_contrast: float = 2.0
    contrastive_lambda_adversary: float = 0.05
    contrastive_lambda_z_l2: float = 1e-4
    contrastive_target_clip: float = 1.0
    contrastive_forest_x_mode: str = "bottleneck_plus_tau"

    # Causal forest parameters.
    cf_n_estimators: int = 200
    cf_min_samples_leaf: int = 5

    # Hierarchical LLM hyperparameters.
    hlm_model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    hlm_chunk_size: int = 2048
    hlm_chunk_overlap: int = 256
    hlm_max_chunks: int = 16
    hlm_downprojection_dim: Optional[int] = None
    hlm_freeze_llm: bool = True
    hlm_cache_hidden_states: bool = False
    hlm_chat_template_prompt: Optional[str] = None

    # Hierarchical CNN hyperparameters.
    hcnn_embedding_dim: int = 256
    hcnn_conv_dim: int = 256
    hcnn_kernel_size: int = 5
    hcnn_num_conv_blocks: int = 4
    hcnn_chunk_size: int = 12000
    hcnn_chunk_overlap: int = 64
    hcnn_max_chunks: int = 32
    hcnn_vocab_size: int = 50000
    hcnn_projection_dim: int = 128
    hcnn_dropout: float = 0.1

    # Hierarchical GRU hyperparameters.
    hgru_embedding_dim: int = 256
    hgru_gru_hidden_dim: int = 256
    hgru_num_gru_layers: int = 2
    hgru_chunk_size: int = 12000
    hgru_chunk_overlap: int = 64
    hgru_max_chunks: int = 32
    hgru_vocab_size: int = 50000
    hgru_projection_dim: int = 128
    hgru_dropout: float = 0.1

    # Simple CNN hyperparameters.
    scnn_embedding_dim: int = 256
    scnn_conv_dim: int = 256
    scnn_kernel_size: int = 5
    scnn_num_conv_blocks: int = 4
    scnn_max_length: int = 20000
    scnn_vocab_size: int = 50000
    scnn_projection_dim: int = 128
    scnn_dropout: float = 0.1

    # Concept embedding CNN hyperparameters.
    cecnn_sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cecnn_chunk_size_words: int = 64
    cecnn_chunk_overlap_words: int = 16
    cecnn_max_chunks: int = 128
    cecnn_confounder_concepts: List[str] = field(default_factory=list)
    cecnn_effect_modifier_concepts: List[str] = field(default_factory=list)
    cecnn_random_features: int = 0
    cecnn_random_confounder_features: Optional[int] = None
    cecnn_random_modifier_features: Optional[int] = None
    cecnn_projection_dim: int = 128
    cecnn_dropout: float = 0.1
    cecnn_anchor_weight: float = 0.01
    cecnn_cache_chunk_embeddings: bool = False
    cecnn_cached_embedding_dim: int = 0
    cecnn_normalize_embeddings: bool = True
    cecnn_random_state: int = 42

    # Concept token CNN hyperparameters.
    ctcnn_model_name: str = "Qwen/Qwen3-0.6B-Base"
    ctcnn_chunk_size: int = 2048
    ctcnn_chunk_overlap: int = 256
    ctcnn_max_chunks: int = 16
    ctcnn_confounder_concepts: List[str] = field(default_factory=list)
    ctcnn_effect_modifier_concepts: List[str] = field(default_factory=list)
    ctcnn_random_features: int = 0
    ctcnn_random_confounder_features: Optional[int] = None
    ctcnn_random_modifier_features: Optional[int] = None
    ctcnn_projection_dim: int = 128
    ctcnn_dropout: float = 0.1
    ctcnn_anchor_weight: float = 0.01
    ctcnn_cache_hidden_states: bool = False
    ctcnn_cached_hidden_size: int = 0
    ctcnn_downprojection_dim: Optional[int] = None
    ctcnn_normalize_embeddings: bool = True
    ctcnn_random_state: int = 42

    # Slot-value discovery hyperparameters.
    svx_sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    svx_chunk_size_words: int = 64
    svx_chunk_overlap_words: int = 16
    svx_max_chunks: int = 128
    svx_confounder_concepts: List[str] = field(default_factory=list)
    svx_effect_modifier_concepts: List[str] = field(default_factory=list)
    svx_num_free_slots: int = 16
    svx_slot_dim: int = 128
    svx_num_value_prototypes: int = 4
    svx_dropout: float = 0.1
    svx_anchor_weight: float = 0.01
    svx_cache_chunk_embeddings: bool = False
    svx_cached_embedding_dim: int = 0
    svx_normalize_embeddings: bool = True
    svx_attention_temperature: float = 0.1
    svx_attention_entropy_weight: float = 0.0
    svx_query_diversity_weight: float = 0.0
    svx_gate_l1_weight: float = 0.0
    svx_random_state: int = 42

    _EXTRACTOR_PREFIXES = {
        "frozen_llm_pooler": {"flp_"},
        "hierarchical_llm": {"hlm_"},
        "hierarchical_cnn": {"hcnn_"},
        "hierarchical_gru": {"hgru_"},
        "simple_cnn": {"scnn_"},
        "concept_embedding_cnn": {"cecnn_"},
        "concept_token_cnn": {"ctcnn_"},
        "slot_value_discovery": {"svx_"},
    }
    _ALL_EXTRACTOR_PREFIXES = set().union(*_EXTRACTOR_PREFIXES.values())

    def __post_init__(self):
        self.model_type = "causal_forest"
        requested_mode = str(
            self.cf_rlearner_representation_mode
            or self.rlearner_mode
            or "staged_separate_nets"
        ).strip().lower()
        if self.contrastive_effect_enabled:
            self.rlearner_mode = "matched_contrastive_effect"
            self.xw_feature_split = True
            self.cf_rlearner_representation_mode = "staged_separate_nets"
        elif requested_mode in {"shared", "shared_features", "shared_rlearner"} or not self.xw_feature_split:
            self.rlearner_mode = "shared_features"
            self.xw_feature_split = False
            self.cf_rlearner_representation_mode = "shared_features"
            source = str(self.shared_rlearner_nuisance_source).strip().lower()
            if source in {"oof", "crossfit", "cross_fit", "inner_oof"}:
                self.shared_rlearner_nuisance_source = "inner_oof"
            elif source in {"in_sample", "insample", "same_model"}:
                self.shared_rlearner_nuisance_source = "in_sample"
            else:
                raise ValueError(
                    "shared_rlearner_nuisance_source must be 'inner_oof' or 'in_sample'"
                )
        else:
            self.rlearner_mode = "staged_separate_nets"
            self.xw_feature_split = True
            self.cf_rlearner_representation_mode = "staged_separate_nets"
        self.use_explicit_confounders = self.use_explicit_features
        self.flp_downprojection_dim = None
        self.hlm_downprojection_dim = None

    def config_hash(self) -> str:
        d = asdict(self)
        d.pop("cf_rlearner_representation_mode", None)
        if d.get("rlearner_mode") != "shared_features":
            d.pop("shared_rlearner_nuisance_source", None)
        keep_prefixes = self._EXTRACTOR_PREFIXES.get(
            self.feature_extractor_type, set()
        )
        remove_prefixes = self._ALL_EXTRACTOR_PREFIXES - keep_prefixes
        d = {
            k: v for k, v in d.items()
            if not any(k.startswith(p) for p in remove_prefixes)
        }
        config_str = json.dumps(d, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


def _feature_key(spec: ExplicitFeatureSpec) -> Tuple[str, str]:
    categories = ",".join(spec.categories or [])
    return (spec.name, categories)


def _metadata_entry_to_spec(
    entry: Dict[str, Any],
    default_roles: Optional[List[str]] = None,
) -> ExplicitFeatureSpec:
    roles = entry.get("roles") or default_roles or ["confounder"]
    return ExplicitFeatureSpec(
        name=entry["name"],
        type=entry["type"],
        categories=entry.get("categories"),
        description=entry.get("description"),
        roles=list(roles),
    )


def load_explicit_feature_specs_from_metadata(
    dataset_path: str,
) -> List[ExplicitFeatureSpec]:
    """Load role-tagged explicit feature specs from metadata.json.

    New datasets should provide metadata["features"] with roles.  For older
    datasets, metadata["confounders"] is treated as confounder-role features,
    and metadata["effect_modifiers"] is also honored if present.
    """
    metadata_file = Path(dataset_path) / "metadata.json"
    if not metadata_file.exists():
        logger.warning("metadata.json not found at %s", metadata_file)
        return []

    with open(metadata_file) as f:
        metadata = json.load(f)

    roles_by_name: Dict[str, List[str]] = {}
    for key, role in (("confounders", "confounder"), ("effect_modifiers", "effect_modifier")):
        for entry in metadata.get(key, []):
            name = entry["name"]
            roles_by_name.setdefault(name, [])
            if role not in roles_by_name[name]:
                roles_by_name[name].append(role)

    specs: List[ExplicitFeatureSpec] = []
    if metadata.get("features"):
        for entry in metadata["features"]:
            specs.append(
                _metadata_entry_to_spec(
                    entry,
                    default_roles=roles_by_name.get(entry["name"]),
                )
            )
    else:
        merged: Dict[str, Dict[str, Any]] = {}
        for key, role in (("confounders", "confounder"), ("effect_modifiers", "effect_modifier")):
            for entry in metadata.get(key, []):
                name = entry["name"]
                if name not in merged:
                    merged[name] = dict(entry)
                    merged[name]["roles"] = []
                if role not in merged[name]["roles"]:
                    merged[name]["roles"].append(role)
        specs = [_metadata_entry_to_spec(entry) for entry in merged.values()]

    seen = set()
    unique_specs = []
    for spec in specs:
        key = _feature_key(spec)
        if key in seen:
            continue
        seen.add(key)
        unique_specs.append(spec)
    return unique_specs


def prepare_explicit_feature_columns(
    df: pd.DataFrame,
    specs: List[ExplicitFeatureSpec],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Normalize explicit feature columns to explicit_feat_* names."""
    if not specs:
        return df, [], []

    df = df.copy()
    feature_cols = []
    missing = []

    for spec in specs:
        target = f"explicit_feat_{spec.name}"
        candidates = [
            target,
            f"explicit_conf_{spec.name}",
            f"llm_extracted_{spec.name}",
        ]
        source = next((col for col in candidates if col in df.columns), None)
        if source is None:
            missing.append(target)
            continue

        if source != target:
            df[target] = df[source]

        source_missing = f"{source}_missing"
        target_missing = f"{target}_missing"
        if source_missing in df.columns and target_missing not in df.columns:
            df[target_missing] = df[source_missing]

        feature_cols.append(target)

    return df, feature_cols, missing


def _create_datasets_and_loaders(
    train_df,
    test_df,
    train_idx,
    test_idx,
    text_column,
    explicit_feature_cols,
    batch_size,
    hidden_state_cache,
    gpu_store,
):
    """Create train/test datasets and DataLoaders with optional hidden-state cache."""
    use_cache = hidden_state_cache is not None
    if use_cache:
        chunk_counts = hidden_state_cache.chunk_counts
    elif gpu_store is not None:
        chunk_counts = gpu_store.chunk_counts
    else:
        chunk_counts = None

    if gpu_store is not None:
        train_dataset = CachedHiddenStateDataset(
            data=train_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            dataset_indices=np.array(train_idx),
            explicit_feature_columns=explicit_feature_cols,
            cache_chunk_counts=chunk_counts,
        )
        test_dataset = CachedHiddenStateDataset(
            data=test_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            dataset_indices=np.array(test_idx),
            explicit_feature_columns=explicit_feature_cols,
            cache_chunk_counts=chunk_counts,
        )
        collate_fn = collate_cached_batch
    elif use_cache:
        train_dataset = CachedHiddenStateDataset(
            data=train_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            dataset_indices=np.array(train_idx),
            explicit_feature_columns=explicit_feature_cols,
            cache_hidden_states=hidden_state_cache.hidden_states_array,
            cache_attention_masks=hidden_state_cache.attention_mask_array,
            cache_chunk_counts=chunk_counts,
        )
        test_dataset = CachedHiddenStateDataset(
            data=test_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            dataset_indices=np.array(test_idx),
            explicit_feature_columns=explicit_feature_cols,
            cache_hidden_states=hidden_state_cache.hidden_states_array,
            cache_attention_masks=hidden_state_cache.attention_mask_array,
            cache_chunk_counts=chunk_counts,
        )
        collate_fn = collate_cached_batch
    else:
        train_dataset = ClinicalTextDataset(
            data=train_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            explicit_feature_columns=explicit_feature_cols,
        )
        test_dataset = ClinicalTextDataset(
            data=test_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            explicit_feature_columns=explicit_feature_cols,
        )
        collate_fn = collate_batch

    if gpu_store is not None:
        dl_kwargs = dict(num_workers=0)
    elif use_cache:
        dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True)
    else:
        dl_kwargs = dict(
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **dl_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **dl_kwargs,
    )
    return train_dataset, test_dataset, train_loader, test_loader, collate_fn, dl_kwargs


def _make_combined_loader(
    combined_df,
    combined_indices,
    text_column,
    explicit_feature_cols,
    batch_size,
    hidden_state_cache,
    gpu_store,
    dl_kwargs,
):
    """Build the combined loader used by the causal forest stage."""
    if hidden_state_cache is not None:
        chunk_counts = hidden_state_cache.chunk_counts
    elif gpu_store is not None:
        chunk_counts = gpu_store.chunk_counts
    else:
        chunk_counts = None

    if gpu_store is not None:
        dataset = CachedHiddenStateDataset(
            data=combined_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            dataset_indices=combined_indices,
            explicit_feature_columns=explicit_feature_cols,
            cache_chunk_counts=chunk_counts,
        )
        collate_fn = collate_cached_batch
    elif hidden_state_cache is not None:
        dataset = CachedHiddenStateDataset(
            data=combined_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            dataset_indices=combined_indices,
            explicit_feature_columns=explicit_feature_cols,
            cache_hidden_states=hidden_state_cache.hidden_states_array,
            cache_attention_masks=hidden_state_cache.attention_mask_array,
            cache_chunk_counts=chunk_counts,
        )
        collate_fn = collate_cached_batch
    else:
        dataset = ClinicalTextDataset(
            data=combined_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            explicit_feature_columns=explicit_feature_cols,
        )
        collate_fn = collate_batch

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **dl_kwargs,
    )


def _oracle_contrastive_config(config: XWRLearnerForestConfig) -> ContrastiveEffectConfig:
    """Map oracle-runner flat fields to the library contrastive config."""
    return ContrastiveEffectConfig(
        enabled=config.contrastive_effect_enabled,
        bottleneck_dim=config.contrastive_bottleneck_dim,
        hidden_dim=config.contrastive_hidden_dim,
        batch_size=config.contrastive_batch_size,
        n_propensity_bins=config.contrastive_n_propensity_bins,
        overlap_min=config.contrastive_overlap_min,
        overlap_max=config.contrastive_overlap_max,
        min_arm_per_bin=config.contrastive_min_arm_per_bin,
        lambda_factual=config.contrastive_lambda_factual,
        lambda_contrast=config.contrastive_lambda_contrast,
        lambda_adversary=config.contrastive_lambda_adversary,
        lambda_z_l2=config.contrastive_lambda_z_l2,
        target_clip=config.contrastive_target_clip,
        forest_x_mode=config.contrastive_forest_x_mode,
    )


def _make_xw_model(
    config: XWRLearnerForestConfig,
    device: torch.device,
    explicit_feature_specs: List[ExplicitFeatureSpec],
    gpu_store,
    hidden_state_cache,
    tokenizer_texts: Optional[List[str]] = None,
) -> CausalTextForest:
    """Create the staged X/W model with consistent oracle-runner settings."""
    model_kwargs = _common_model_kwargs(
        config,
        gpu_store,
        hidden_state_cache,
        explicit_feature_specs,
        device,
    )
    model_kwargs.update(
        dict(
            representation_dim=128,
            hidden_dim=64,
            dropout=0.2,
            cf_n_estimators=config.cf_n_estimators,
            cf_min_samples_leaf=config.cf_min_samples_leaf,
            cf_honest=True,
            cf_inference=True,
            cf_use_rlearner_representation=(
                config.cf_rlearner_representation_mode != "none"
            ),
            cf_rlearner_representation_mode=config.cf_rlearner_representation_mode,
            cf_gamma_rlearner=config.gamma_rlearner,
            explicit_feature_specs=explicit_feature_specs,
        )
    )

    model_class = ContrastiveCausalTextForest if config.contrastive_effect_enabled else CausalTextForest
    if config.contrastive_effect_enabled:
        model_kwargs["contrastive_effect_config"] = _oracle_contrastive_config(config)

    model = model_class(**model_kwargs)
    if config.feature_extractor_type in TRAINABLE_EXTRACTOR_TYPES and tokenizer_texts is not None:
        model.fit_tokenizer(tokenizer_texts)

    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            logger.warning(
                "Parameter %s has dtype %s; casting to float32",
                name,
                param.dtype,
            )
            param.data = param.data.float()
    return model


def _fit_explicit_feature_state(model: CausalTextForest, dataset) -> None:
    """Fit MLP and raw explicit-feature normalization from a training dataset."""
    if getattr(dataset, "explicit_feature_values", None):
        model.fit_explicit_features(dataset.explicit_feature_values)
        model.fit_explicit_feature_featurizer(dataset.explicit_feature_values)


def _train_nuisance_stage(
    model: CausalTextForest,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: XWRLearnerForestConfig,
    device: torch.device,
    use_cached: bool,
    gpu_store,
) -> None:
    """Train e(W), m(W); if val_loader is provided, restore the best val state."""
    params = [p for p in model.nuisance_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_loss = float("inf")
    best_state = None

    for _epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            batch["treatment"] = batch["treatment"].to(device)
            batch["outcome"] = batch["outcome"].to(device)
            if use_cached:
                prepare_cached_batch(batch, device, gpu_store=gpu_store)

            optimizer.zero_grad()
            losses = model.train_nuisance_step(batch, alpha_propensity=1.0)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

        scheduler.step()

        if val_loader is None:
            continue

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch["treatment"] = batch["treatment"].to(device)
                batch["outcome"] = batch["outcome"].to(device)
                if use_cached:
                    prepare_cached_batch(batch, device, gpu_store=gpu_store)
                losses = model.train_nuisance_step(batch, alpha_propensity=1.0)
                val_loss += losses["loss"].item()

        val_loss /= max(len(val_loader), 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)


def _loader_worker_kwargs(source_loader: DataLoader) -> Dict[str, Any]:
    """Reuse DataLoader worker settings without copying sampler/batch settings."""
    loader_kwargs: Dict[str, Any] = {}
    if getattr(source_loader, "num_workers", 0) > 0:
        loader_kwargs["num_workers"] = source_loader.num_workers
        loader_kwargs["persistent_workers"] = getattr(
            source_loader, "persistent_workers", False
        )
        loader_kwargs["pin_memory"] = getattr(source_loader, "pin_memory", False)
        prefetch_factor = getattr(source_loader, "prefetch_factor", None)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return loader_kwargs


def _make_effect_loader(
    train_loader: DataLoader,
    config: XWRLearnerForestConfig,
) -> Tuple[DataLoader, int]:
    """Build the canonical R-loss effect DataLoader."""
    effect_batch_size = config.rlearner_effect_batch_size or config.batch_size
    if effect_batch_size < 1:
        raise ValueError("rlearner_effect_batch_size must be >= 1 when set")
    if effect_batch_size == config.batch_size:
        return train_loader, effect_batch_size
    return (
        DataLoader(
            train_loader.dataset,
            batch_size=effect_batch_size,
            shuffle=True,
            collate_fn=train_loader.collate_fn,
            **_loader_worker_kwargs(train_loader),
        ),
        effect_batch_size,
    )


def _binary_cross_entropy_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pred = np.clip(pred, 1e-6, 1.0 - 1e-6)
    return float(np.mean(-(target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred))))


def _nuisance_oof_summary(
    propensity: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    observed_outcome: np.ndarray,
) -> Dict[str, float]:
    return {
        "propensity_bce": _binary_cross_entropy_np(propensity, treatment),
        "outcome_bce": _binary_cross_entropy_np(outcome, observed_outcome),
        "propensity_mean": float(np.mean(propensity)),
        "outcome_mean": float(np.mean(outcome)),
    }


def _is_high_pdl1_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        value_float = float(value)
        return value_float >= (0.5 if value_float <= 1.0 else 50.0)
    text = str(value).strip().lower()
    text = (
        text.replace("\u2265", ">=")
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    if not text or text in {"nan", "none", "unknown"}:
        return False
    if text.startswith("<") or "1-49" in text or "low" in text:
        return False
    return ">=50" in text or ">50" in text or "50%" in text or text == "50"


def _high_pdl1_mask_from_dataset(dataset) -> Tuple[Optional[np.ndarray], Optional[str]]:
    data = getattr(dataset, "data", None)
    if data is None or not hasattr(data, "columns"):
        return None, None
    preferred = [
        "true_pdl1_expression",
        "explicit_feat_pdl1_expression",
        "llm_extracted_pdl1_expression",
    ]
    pdl1_columns = [
        col for col in preferred if col in data.columns
    ] or [
        col for col in data.columns
        if "pdl1" in col.lower() or "pd_l1" in col.lower() or "pd-l1" in col.lower()
    ]
    if not pdl1_columns:
        return None, None
    column = pdl1_columns[0]
    return data[column].map(_is_high_pdl1_value).to_numpy(dtype=bool), column


def _summarize_pdl1_cell_counts(counts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not counts:
        return {}
    min_counts = [
        min(
            item["high_treated"],
            item["high_control"],
            item["low_treated"],
            item["low_control"],
        )
        for item in counts
    ]
    return {
        "num_batches": len(counts),
        "num_batches_all_cells_present": int(
            sum(1 for item in counts if item["all_cells_present"])
        ),
        "fraction_batches_all_cells_present": float(
            np.mean([item["all_cells_present"] for item in counts])
        ),
        "mean_min_cell_count": float(np.mean(min_counts)),
    }


def _slot_extractor_summary(extractor) -> Dict[str, Any]:
    if extractor is None or not hasattr(extractor, "get_state"):
        return {}
    state = extractor.get_state()
    keys = [
        "extractor_type",
        "num_seed_slots",
        "num_free_slots",
        "num_slots",
        "slot_dim",
        "output_dim",
    ]
    return {key: state[key] for key in keys if key in state}


def _train_effect_stage(
    model: CausalTextForest,
    train_loader: DataLoader,
    nuisance_propensity: np.ndarray,
    nuisance_outcome: np.ndarray,
    config: XWRLearnerForestConfig,
    device: torch.device,
    use_cached: bool,
    gpu_store,
) -> Dict[str, Any]:
    """Train tau(X) from fixed outer-train OOF nuisance predictions."""
    accumulation_steps = max(1, int(config.rlearner_effect_accumulation_steps or 1))
    if not (0.0 < float(config.rlearner_effect_e_clip) < 0.5):
        raise ValueError("rlearner_effect_e_clip must be in (0, 0.5)")

    use_contrastive = (
        config.contrastive_effect_enabled
        and hasattr(model, "train_effect_contrastive_step")
    )
    effect_loader, physical_batch_size = _make_effect_loader(train_loader, config)
    propensity_bin_ids = None
    if use_contrastive:
        physical_batch_size = config.contrastive_batch_size
        dataset_treatment = train_loader.dataset.treatments
        if hasattr(dataset_treatment, "detach"):
            dataset_treatment = dataset_treatment.detach().cpu().numpy()
        else:
            dataset_treatment = np.asarray(dataset_treatment)
        propensity_bin_ids = make_propensity_bins(
            propensity=nuisance_propensity,
            treatment=dataset_treatment,
            n_bins=config.contrastive_n_propensity_bins,
            overlap_min=config.contrastive_overlap_min,
            overlap_max=config.contrastive_overlap_max,
            min_arm_per_bin=config.contrastive_min_arm_per_bin,
        )
        sampler = PropensityBinBalancedBatchSampler(
            treatment=dataset_treatment,
            bin_ids=propensity_bin_ids,
            batch_size=config.contrastive_batch_size,
            min_arm_per_bin=config.contrastive_min_arm_per_bin,
            seed=42 + config.repeat_index,
        )
        effect_loader = DataLoader(
            train_loader.dataset,
            batch_sampler=sampler,
            collate_fn=train_loader.collate_fn,
            **_loader_worker_kwargs(train_loader),
        )
        logger.info(
            "Contrastive effect bins: %d bins, %d/%d samples in overlap",
            len(np.unique(propensity_bin_ids[propensity_bin_ids >= 0])),
            int(np.sum(propensity_bin_ids >= 0)),
            len(propensity_bin_ids),
        )

    params = [p for p in model.effect_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    pdl1_high_mask, pdl1_column = _high_pdl1_mask_from_dataset(train_loader.dataset)
    epoch_history: List[Dict[str, Any]] = []
    pdl1_cell_counts: List[Dict[str, Any]] = []
    optimizer_steps = 0

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config.epochs):
        model.train()
        epoch_r_losses: List[float] = []
        epoch_losses: List[float] = []
        pending_accumulation = 0
        for batch_index, batch in enumerate(effect_loader, start=1):
            batch["treatment"] = batch["treatment"].to(device)
            batch["outcome"] = batch["outcome"].to(device)
            if use_cached:
                prepare_cached_batch(batch, device, gpu_store=gpu_store)

            batch_ids = np.asarray(batch["text_id"], dtype=int)
            e_hat = torch.as_tensor(
                nuisance_propensity[batch_ids],
                dtype=torch.float32,
                device=device,
            )
            m_hat = torch.as_tensor(
                nuisance_outcome[batch_ids],
                dtype=torch.float32,
                device=device,
            )

            if use_contrastive:
                bin_ids = torch.as_tensor(
                    propensity_bin_ids[batch_ids],
                    dtype=torch.long,
                    device=device,
                )
                losses = model.train_effect_contrastive_step(
                    batch,
                    e_hat=e_hat,
                    m_hat=m_hat,
                    bin_ids=bin_ids,
                )
            else:
                losses = model.train_effect_r_step(
                    batch,
                    e_hat=e_hat,
                    m_hat=m_hat,
                    gamma_rlearner=config.gamma_rlearner,
                    e_clip=config.rlearner_effect_e_clip,
                )
            (losses["loss"] / accumulation_steps).backward()
            pending_accumulation += 1

            epoch_losses.append(float(losses["loss"].detach().cpu()))
            if "r_loss" in losses:
                epoch_r_losses.append(float(losses["r_loss"].detach().cpu()))

            if pdl1_high_mask is not None:
                high = pdl1_high_mask[batch_ids]
                treatment = batch["treatment"].detach().cpu().numpy() > 0.5
                count = {
                    "epoch": epoch + 1,
                    "batch": batch_index,
                    "high_treated": int(np.sum(high & treatment)),
                    "high_control": int(np.sum(high & ~treatment)),
                    "low_treated": int(np.sum(~high & treatment)),
                    "low_control": int(np.sum(~high & ~treatment)),
                }
                count["all_cells_present"] = all(
                    count[key] > 0
                    for key in (
                        "high_treated",
                        "high_control",
                        "low_treated",
                        "low_control",
                    )
                )
                pdl1_cell_counts.append(count)

            if pending_accumulation >= accumulation_steps:
                torch.nn.utils.clip_grad_norm_(params, config.rlearner_effect_grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending_accumulation = 0

        if pending_accumulation > 0:
            torch.nn.utils.clip_grad_norm_(params, config.rlearner_effect_grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        epoch_history.append(
            {
                "epoch": epoch + 1,
                "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                "mean_r_loss": float(np.mean(epoch_r_losses)) if epoch_r_losses else float("nan"),
                "num_batches": len(epoch_losses),
            }
        )

        scheduler.step()

    effective_batch_size = physical_batch_size * accumulation_steps
    return {
        "effect_physical_batch_size": int(physical_batch_size),
        "effect_accumulation_steps": int(accumulation_steps),
        "effect_effective_batch_size": int(effective_batch_size),
        "effect_optimizer_steps": int(optimizer_steps),
        "effect_r_loss_history": epoch_history,
        "effect_pdl1_column": pdl1_column,
        "effect_pdl1_cell_counts": pdl1_cell_counts,
        "effect_pdl1_cell_summary": _summarize_pdl1_cell_counts(pdl1_cell_counts),
        "nuisance_extractor": _slot_extractor_summary(
            getattr(model, "feature_extractor", None)
        ),
        "effect_extractor": _slot_extractor_summary(
            getattr(model, "effect_feature_extractor", None)
        ),
    }


def _train_shared_representation_stage(
    model: CausalTextForest,
    train_loader: DataLoader,
    config: XWRLearnerForestConfig,
    device: torch.device,
    use_cached: bool,
    gpu_store,
    nuisance_propensity: Optional[np.ndarray] = None,
    nuisance_outcome: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Train a single extractor with propensity, outcome, and shared R-loss."""
    if (nuisance_propensity is None) != (nuisance_outcome is None):
        raise ValueError("nuisance_propensity and nuisance_outcome must be provided together")
    use_oof_nuisance = nuisance_propensity is not None
    if use_oof_nuisance:
        nuisance_propensity = np.asarray(nuisance_propensity, dtype=np.float32)
        nuisance_outcome = np.asarray(nuisance_outcome, dtype=np.float32)
        if len(nuisance_propensity) != len(train_loader.dataset) or len(nuisance_outcome) != len(train_loader.dataset):
            raise ValueError(
                "OOF nuisance arrays must have one value per training dataset sample"
            )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    history: List[Dict[str, Any]] = []

    for epoch in range(config.epochs):
        model.train()
        epoch_losses: List[float] = []
        epoch_outcome_losses: List[float] = []
        epoch_propensity_losses: List[float] = []
        epoch_r_losses: List[float] = []

        for batch in train_loader:
            batch["treatment"] = batch["treatment"].to(device)
            batch["outcome"] = batch["outcome"].to(device)
            if use_cached:
                prepare_cached_batch(batch, device, gpu_store=gpu_store)

            e_hat = None
            m_hat = None
            if use_oof_nuisance:
                batch_ids = np.asarray(batch["text_id"], dtype=int)
                e_hat = torch.as_tensor(
                    nuisance_propensity[batch_ids],
                    dtype=torch.float32,
                    device=device,
                )
                m_hat = torch.as_tensor(
                    nuisance_outcome[batch_ids],
                    dtype=torch.float32,
                    device=device,
                )

            optimizer.zero_grad(set_to_none=True)
            losses = model.train_shared_rlearner_step(
                batch,
                alpha_propensity=1.0,
                gamma_rlearner=config.gamma_rlearner,
                e_clip=config.rlearner_effect_e_clip,
                e_hat=e_hat,
                m_hat=m_hat,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, config.rlearner_effect_grad_clip)
            optimizer.step()

            epoch_losses.append(float(losses["loss"].detach().cpu()))
            epoch_outcome_losses.append(float(losses["outcome_loss"].detach().cpu()))
            epoch_propensity_losses.append(float(losses["propensity_loss"].detach().cpu()))
            epoch_r_losses.append(float(losses["r_loss"].detach().cpu()))

        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                "mean_outcome_loss": (
                    float(np.mean(epoch_outcome_losses))
                    if epoch_outcome_losses
                    else float("nan")
                ),
                "mean_propensity_loss": (
                    float(np.mean(epoch_propensity_losses))
                    if epoch_propensity_losses
                    else float("nan")
                ),
                "mean_r_loss": float(np.mean(epoch_r_losses)) if epoch_r_losses else float("nan"),
                "num_batches": len(epoch_losses),
            }
        )

    return {
        "shared_r_loss_history": history,
        "shared_r_loss_nuisance_source": (
            "inner_oof" if use_oof_nuisance else "in_sample"
        ),
        "shared_extractor": _slot_extractor_summary(
            getattr(model, "feature_extractor", None)
        ),
        "forest_x_representation": "extractor_output",
        "forest_w_representation": None,
    }


def run_shared_rlearner_forest_experiment(
    config: XWRLearnerForestConfig,
    device: torch.device,
    df: pd.DataFrame,
    explicit_feature_specs: List[ExplicitFeatureSpec],
    explicit_feature_cols: Optional[List[str]],
    gpu_store,
    hidden_state_cache,
) -> Dict[str, Any]:
    """Run K-fold CV for shared slot/text features -> CausalForestDML."""
    text_column = "clinical_text"
    batch_size = config.batch_size
    df = df.reset_index(drop=True)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42 + config.repeat_index)

    all_predictions = []
    diagnostics: Dict[str, Any] = {"folds": []}
    use_cached = gpu_store is not None or hidden_state_cache is not None

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        oof_propensity = None
        oof_outcome = None
        nuisance_summary = {}
        if config.shared_rlearner_nuisance_source == "inner_oof":
            n_inner = min(config.rlearner_nuisance_folds, len(train_df))
            if n_inner < 2:
                raise ValueError(
                    "rlearner_nuisance_folds requires at least 2 outer-train samples"
                )
            inner_kf = KFold(
                n_splits=n_inner,
                shuffle=True,
                random_state=20_000 + 42 + config.repeat_index + fold,
            )
            oof_propensity = np.full(len(train_df), np.nan, dtype=np.float32)
            oof_outcome = np.full(len(train_df), np.nan, dtype=np.float32)

            for inner_train_pos, inner_val_pos in inner_kf.split(train_df):
                inner_train_df = train_df.iloc[inner_train_pos]
                inner_val_df = train_df.iloc[inner_val_pos]
                inner_train_idx = np.asarray(train_idx)[inner_train_pos]
                inner_val_idx = np.asarray(train_idx)[inner_val_pos]

                inner_model = _make_xw_model(
                    config,
                    device,
                    explicit_feature_specs,
                    gpu_store,
                    hidden_state_cache,
                    tokenizer_texts=inner_train_df[text_column].tolist(),
                )
                (
                    inner_train_dataset,
                    _inner_val_dataset,
                    inner_train_loader,
                    inner_val_loader,
                    _inner_collate_fn,
                    _inner_dl_kwargs,
                ) = _create_datasets_and_loaders(
                    inner_train_df,
                    inner_val_df,
                    inner_train_idx,
                    inner_val_idx,
                    text_column,
                    explicit_feature_cols,
                    batch_size,
                    hidden_state_cache,
                    gpu_store,
                )
                _fit_explicit_feature_state(inner_model, inner_train_dataset)
                _train_nuisance_stage(
                    inner_model,
                    inner_train_loader,
                    inner_val_loader,
                    config,
                    device,
                    use_cached,
                    gpu_store,
                )
                cf_kwargs = dict(gpu_store=gpu_store) if use_cached else {}
                prop_hat, outcome_hat = inner_model.predict_nuisance(
                    inner_val_loader,
                    **cf_kwargs,
                )
                oof_propensity[inner_val_pos] = prop_hat
                oof_outcome[inner_val_pos] = outcome_hat

                del inner_model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if np.isnan(oof_propensity).any() or np.isnan(oof_outcome).any():
                raise RuntimeError("Incomplete out-of-fold nuisance predictions")
            nuisance_summary = _nuisance_oof_summary(
                propensity=oof_propensity,
                outcome=oof_outcome,
                treatment=train_df["treatment_indicator"].values,
                observed_outcome=train_df["outcome_indicator"].values,
            )

        model = _make_xw_model(
            config,
            device,
            explicit_feature_specs,
            gpu_store,
            hidden_state_cache,
            tokenizer_texts=train_df[text_column].tolist(),
        )
        (
            train_dataset,
            _test_dataset,
            train_loader,
            test_loader,
            _collate_fn,
            dl_kwargs,
        ) = _create_datasets_and_loaders(
            train_df,
            test_df,
            train_idx,
            test_idx,
            text_column,
            explicit_feature_cols,
            batch_size,
            hidden_state_cache,
            gpu_store,
        )

        _fit_explicit_feature_state(model, train_dataset)
        shared_diagnostics = _train_shared_representation_stage(
            model,
            train_loader,
            config,
            device,
            use_cached,
            gpu_store,
            nuisance_propensity=oof_propensity,
            nuisance_outcome=oof_outcome,
        )

        train_eval_loader = _make_combined_loader(
            train_df,
            np.asarray(train_idx),
            text_column,
            explicit_feature_cols,
            batch_size,
            hidden_state_cache,
            gpu_store,
            dl_kwargs,
        )

        train_T = train_df["treatment_indicator"].values
        train_Y = train_df["outcome_indicator"].values
        cf_kwargs = dict(gpu_store=gpu_store) if use_cached else {}
        model.train_causal_forest(train_eval_loader, train_T, train_Y, **cf_kwargs)
        preds = model.predict(test_loader, return_ci=True, **cf_kwargs)

        diagnostics["folds"].append(
            {
                "fold": fold + 1,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "nuisance_oof": nuisance_summary,
                "shared_stage": shared_diagnostics,
            }
        )

        fold_preds = test_df.copy()
        fold_preds["pred_y0_prob"] = preds["pred_y0_prob"]
        fold_preds["pred_y1_prob"] = preds["pred_y1_prob"]
        fold_preds["pred_ite_prob"] = preds["pred_ite_prob"]
        fold_preds["pred_propensity"] = preds["propensity_prob"]
        fold_preds["pred_tau"] = preds["tau_pred"]
        fold_preds["cv_fold"] = fold + 1
        if "tau_lower" in preds:
            fold_preds["pred_tau_lower"] = preds["tau_lower"]
            fold_preds["pred_tau_upper"] = preds["tau_upper"]

        all_predictions.append(fold_preds)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_df = pd.concat(all_predictions).sort_index()
    metrics = compute_metrics(
        pred_ite=results_df["pred_ite_prob"].values,
        true_ite=results_df["true_ite_prob"].values,
        pred_propensity=results_df["pred_propensity"].values,
        true_treatment=results_df["treatment_indicator"].values,
        pred_y0=results_df["pred_y0_prob"].values,
        pred_y1=results_df["pred_y1_prob"].values,
        true_y0=results_df["true_y0_prob"].values,
        true_y1=results_df["true_y1_prob"].values,
        true_outcome=results_df["outcome_indicator"].values,
        tau_lower=(
            results_df["pred_tau_lower"].values
            if "pred_tau_lower" in results_df.columns
            else None
        ),
        tau_upper=(
            results_df["pred_tau_upper"].values
            if "pred_tau_upper" in results_df.columns
            else None
        ),
    )
    return {
        "metrics": metrics,
        "n_samples": len(results_df),
        "diagnostics": diagnostics,
    }


def run_xw_rlearner_forest_experiment(
    config: XWRLearnerForestConfig,
    device: torch.device,
    df: pd.DataFrame,
    explicit_feature_specs: List[ExplicitFeatureSpec],
    explicit_feature_cols: Optional[List[str]],
    gpu_store,
    hidden_state_cache,
) -> Dict[str, Any]:
    """Run K-fold CV for the R-learner representation -> causal forest path."""
    text_column = "clinical_text"
    batch_size = config.batch_size
    df = df.reset_index(drop=True)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42 + config.repeat_index)

    all_predictions = []
    diagnostics: Dict[str, Any] = {"folds": []}
    use_cached = gpu_store is not None or hidden_state_cache is not None

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        n_inner = min(config.rlearner_nuisance_folds, len(train_df))
        if n_inner < 2:
            raise ValueError("rlearner_nuisance_folds requires at least 2 outer-train samples")
        inner_kf = KFold(
            n_splits=n_inner,
            shuffle=True,
            random_state=10_000 + 42 + config.repeat_index + fold,
        )
        oof_propensity = np.full(len(train_df), np.nan, dtype=np.float32)
        oof_outcome = np.full(len(train_df), np.nan, dtype=np.float32)

        for inner_train_pos, inner_val_pos in inner_kf.split(train_df):
            inner_train_df = train_df.iloc[inner_train_pos]
            inner_val_df = train_df.iloc[inner_val_pos]
            inner_train_idx = np.asarray(train_idx)[inner_train_pos]
            inner_val_idx = np.asarray(train_idx)[inner_val_pos]

            inner_model = _make_xw_model(
                config,
                device,
                explicit_feature_specs,
                gpu_store,
                hidden_state_cache,
                tokenizer_texts=inner_train_df[text_column].tolist(),
            )
            (
                inner_train_dataset,
                _inner_val_dataset,
                inner_train_loader,
                inner_val_loader,
                _inner_collate_fn,
                _inner_dl_kwargs,
            ) = _create_datasets_and_loaders(
                inner_train_df,
                inner_val_df,
                inner_train_idx,
                inner_val_idx,
                text_column,
                explicit_feature_cols,
                batch_size,
                hidden_state_cache,
                gpu_store,
            )
            _fit_explicit_feature_state(inner_model, inner_train_dataset)
            _train_nuisance_stage(
                inner_model,
                inner_train_loader,
                inner_val_loader,
                config,
                device,
                use_cached,
                gpu_store,
            )
            cf_kwargs = dict(gpu_store=gpu_store) if use_cached else {}
            prop_hat, outcome_hat = inner_model.predict_nuisance(
                inner_val_loader,
                **cf_kwargs,
            )
            oof_propensity[inner_val_pos] = prop_hat
            oof_outcome[inner_val_pos] = outcome_hat

            del inner_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if np.isnan(oof_propensity).any() or np.isnan(oof_outcome).any():
            raise RuntimeError("Incomplete out-of-fold nuisance predictions")
        nuisance_summary = _nuisance_oof_summary(
            propensity=oof_propensity,
            outcome=oof_outcome,
            treatment=train_df["treatment_indicator"].values,
            observed_outcome=train_df["outcome_indicator"].values,
        )

        model = _make_xw_model(
            config,
            device,
            explicit_feature_specs,
            gpu_store,
            hidden_state_cache,
            tokenizer_texts=train_df[text_column].tolist(),
        )
        (
            train_dataset,
            _test_dataset,
            train_loader,
            test_loader,
            _collate_fn,
            dl_kwargs,
        ) = _create_datasets_and_loaders(
            train_df,
            test_df,
            train_idx,
            test_idx,
            text_column,
            explicit_feature_cols,
            batch_size,
            hidden_state_cache,
            gpu_store,
        )

        _fit_explicit_feature_state(model, train_dataset)
        _train_nuisance_stage(
            model,
            train_loader,
            None,
            config,
            device,
            use_cached,
            gpu_store,
        )
        effect_diagnostics = _train_effect_stage(
            model,
            train_loader,
            oof_propensity,
            oof_outcome,
            config,
            device,
            use_cached,
            gpu_store,
        )
        diagnostics["folds"].append(
            {
                "fold": fold + 1,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "nuisance_oof": nuisance_summary,
                "effect_stage": effect_diagnostics,
            }
        )

        train_eval_loader = _make_combined_loader(
            train_df,
            np.asarray(train_idx),
            text_column,
            explicit_feature_cols,
            batch_size,
            hidden_state_cache,
            gpu_store,
            dl_kwargs,
        )

        train_T = train_df["treatment_indicator"].values
        train_Y = train_df["outcome_indicator"].values
        cf_kwargs = dict(gpu_store=gpu_store) if use_cached else {}
        model.train_causal_forest(train_eval_loader, train_T, train_Y, **cf_kwargs)
        preds = model.predict(test_loader, return_ci=True, **cf_kwargs)

        fold_preds = test_df.copy()
        fold_preds["pred_y0_prob"] = preds["pred_y0_prob"]
        fold_preds["pred_y1_prob"] = preds["pred_y1_prob"]
        fold_preds["pred_ite_prob"] = preds["pred_ite_prob"]
        fold_preds["pred_propensity"] = preds["propensity_prob"]
        fold_preds["pred_tau"] = preds["tau_pred"]
        fold_preds["cv_fold"] = fold + 1
        if "tau_lower" in preds:
            fold_preds["pred_tau_lower"] = preds["tau_lower"]
            fold_preds["pred_tau_upper"] = preds["tau_upper"]

        all_predictions.append(fold_preds)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_df = pd.concat(all_predictions).sort_index()
    metrics = compute_metrics(
        pred_ite=results_df["pred_ite_prob"].values,
        true_ite=results_df["true_ite_prob"].values,
        pred_propensity=results_df["pred_propensity"].values,
        true_treatment=results_df["treatment_indicator"].values,
        pred_y0=results_df["pred_y0_prob"].values,
        pred_y1=results_df["pred_y1_prob"].values,
        true_y0=results_df["true_y0_prob"].values,
        true_y1=results_df["true_y1_prob"].values,
        true_outcome=results_df["outcome_indicator"].values,
        tau_lower=(
            results_df["pred_tau_lower"].values
            if "pred_tau_lower" in results_df.columns
            else None
        ),
        tau_upper=(
            results_df["pred_tau_upper"].values
            if "pred_tau_upper" in results_df.columns
            else None
        ),
    )
    return {
        "metrics": metrics,
        "n_samples": len(results_df),
        "diagnostics": diagnostics,
    }


def run_single_experiment(
    config: XWRLearnerForestConfig,
    device: str,
    output_dir: Path,
    cache_registry: Optional[Dict[str, HiddenStateCache]] = None,
    gpu_store_registry: Optional[Dict[str, GPUHiddenStateStore]] = None,
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    del output_dir
    device_obj = torch.device(device)

    parquet_file = _resolve_parquet_file(config.dataset_path)
    if parquet_file is None:
        return {"error": f"Dataset not found in {config.dataset_path}", "skipped": True}

    df = pd.read_parquet(parquet_file)
    if "clinical_text" not in df.columns:
        return {"error": "Text column 'clinical_text' not found", "skipped": True}

    explicit_feature_specs: List[ExplicitFeatureSpec] = []
    explicit_feature_cols = None
    if config.use_explicit_features:
        explicit_feature_specs = load_explicit_feature_specs_from_metadata(config.dataset_path)
        if not explicit_feature_specs:
            return {
                "error": f"No explicit feature specs found in {config.dataset_path}",
                "skipped": True,
            }

        df, explicit_feature_cols, missing_cols = prepare_explicit_feature_columns(
            df,
            explicit_feature_specs,
        )
        if missing_cols:
            return {
                "error": (
                    "Explicit feature columns missing from dataset: "
                    f"{missing_cols}. Expected explicit_feat_*, explicit_conf_*, "
                    "or llm_extracted_* columns."
                ),
                "skipped": True,
            }

        role_counts = {"confounder": 0, "effect_modifier": 0, "both": 0}
        for spec in explicit_feature_specs:
            role_set = set(spec.roles)
            if role_set == {"confounder", "effect_modifier"}:
                role_counts["both"] += 1
            else:
                for role in role_set:
                    role_counts[role] += 1
        logger.info(
            "Using %d role-tagged explicit features: %s",
            len(explicit_feature_specs),
            role_counts,
        )

    gpu_store, hidden_state_cache = _get_cache_info(
        config,
        parquet_file,
        cache_registry,
        gpu_store_registry,
    )

    if config.rlearner_mode == "shared_features":
        result = run_shared_rlearner_forest_experiment(
            config,
            device_obj,
            df,
            explicit_feature_specs,
            explicit_feature_cols,
            gpu_store,
            hidden_state_cache,
        )
    else:
        result = run_xw_rlearner_forest_experiment(
            config,
            device_obj,
            df,
            explicit_feature_specs,
            explicit_feature_cols,
            gpu_store,
            hidden_state_cache,
        )

    return {
        "config": asdict(config),
        "metrics": result["metrics"],
        "n_samples": result["n_samples"],
        "diagnostics": result.get("diagnostics", {}),
        "skipped": False,
        "error": None,
    }


def generate_experiment_grid(
    dataset_paths: List[str],
    filter_max_lengths: Optional[List[int]] = None,
    model_names: Optional[List[str]] = None,
    chat_template_prompt: Optional[str] = None,
    filter_extractor_types: Optional[List[str]] = None,
    learning_rates: Optional[List[float]] = None,
    epoch_counts: Optional[List[int]] = None,
    include_explicit_feature_options: Optional[List[bool]] = None,
    rlearner_effect_batch_size: Optional[int] = None,
    rlearner_effect_accumulation_steps: int = 1,
    rlearner_effect_e_clip: float = 0.01,
    rlearner_effect_grad_clip: float = 1.0,
    contrastive_effect_enabled: bool = False,
    contrastive_bottleneck_dim: int = 8,
    contrastive_hidden_dim: int = 64,
    contrastive_batch_size: int = 16,
    contrastive_n_propensity_bins: int = 10,
    contrastive_overlap_min: float = 0.05,
    contrastive_overlap_max: float = 0.95,
    contrastive_min_arm_per_bin: int = 2,
    contrastive_lambda_factual: float = 1.0,
    contrastive_lambda_contrast: float = 2.0,
    contrastive_lambda_adversary: float = 0.05,
    contrastive_lambda_z_l2: float = 1e-4,
    contrastive_target_clip: float = 1.0,
    contrastive_forest_x_mode: str = "bottleneck_plus_tau",
) -> List[XWRLearnerForestConfig]:
    """Generate the narrowed experiment grid."""
    if model_names is None:
        model_names = [
            "Qwen/Qwen3.5-0.8B-Base",
            "Qwen/Qwen3.5-0.8B",
            "google/medgemma-1.5-4b-it",
        ]
    if learning_rates is None:
        learning_rates = [1e-5, 1e-4]
    if epoch_counts is None:
        epoch_counts = [50]
    if include_explicit_feature_options is None:
        include_explicit_feature_options = [False, True]

    datasets = [(p, Path(p).name) for p in dataset_paths]
    all_extractor_types = [
        "frozen_llm_pooler",
        "hierarchical_llm",
        "hierarchical_cnn",
        "hierarchical_gru",
        "simple_cnn",
    ]
    extractor_types = all_extractor_types
    if filter_extractor_types:
        extractor_types = [e for e in all_extractor_types if e in filter_extractor_types]

    configs: List[XWRLearnerForestConfig] = []

    for ext_type in extractor_types:
        if ext_type == "frozen_llm_pooler":
            max_lengths = [50000]
            if filter_max_lengths:
                max_lengths = [m for m in max_lengths if m in filter_max_lengths]
            chat_template_options = [None]
            if chat_template_prompt is not None:
                chat_template_options = [None, chat_template_prompt]

            for (
                dataset_path,
                dataset_name,
            ), max_len, use_feats, ctp, mn, lr, ep in itertools.product(
                datasets,
                max_lengths,
                include_explicit_feature_options,
                chat_template_options,
                model_names,
                learning_rates,
                epoch_counts,
            ):
                configs.append(
                    XWRLearnerForestConfig(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        use_explicit_features=use_feats,
                        feature_extractor_type="frozen_llm_pooler",
                        flp_max_length=max_len,
                        flp_downprojection_dim=None,
                        flp_model_name=mn,
                        flp_chat_template_prompt=ctp,
                        learning_rate=lr,
                        epochs=ep,
                    )
                )

        elif ext_type == "hierarchical_llm":
            chunk_size = 2048
            chunk_overlap = 256
            max_chunks_options = [16]

            for (
                dataset_path,
                dataset_name,
            ), n_chunks, use_feats, mn, lr, ep in itertools.product(
                datasets,
                max_chunks_options,
                include_explicit_feature_options,
                model_names,
                learning_rates,
                epoch_counts,
            ):
                configs.append(
                    XWRLearnerForestConfig(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        use_explicit_features=use_feats,
                        feature_extractor_type="hierarchical_llm",
                        hlm_model_name=mn,
                        hlm_chunk_size=chunk_size,
                        hlm_chunk_overlap=chunk_overlap,
                        hlm_max_chunks=n_chunks,
                        hlm_downprojection_dim=None,
                        learning_rate=lr,
                        epochs=ep,
                    )
                )

        elif ext_type == "hierarchical_cnn":
            chunk_sizes = [256, 512]
            for (
                dataset_path,
                dataset_name,
            ), use_feats, cs, lr, ep in itertools.product(
                datasets,
                include_explicit_feature_options,
                chunk_sizes,
                learning_rates,
                epoch_counts,
            ):
                configs.append(
                    XWRLearnerForestConfig(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        use_explicit_features=use_feats,
                        feature_extractor_type="hierarchical_cnn",
                        hcnn_chunk_size=cs,
                        learning_rate=lr,
                        epochs=ep,
                    )
                )

        elif ext_type == "hierarchical_gru":
            chunk_sizes = [256, 512]
            for (
                dataset_path,
                dataset_name,
            ), use_feats, cs, lr, ep in itertools.product(
                datasets,
                include_explicit_feature_options,
                chunk_sizes,
                learning_rates,
                epoch_counts,
            ):
                configs.append(
                    XWRLearnerForestConfig(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        use_explicit_features=use_feats,
                        feature_extractor_type="hierarchical_gru",
                        hgru_chunk_size=cs,
                        learning_rate=lr,
                        epochs=ep,
                    )
                )

        elif ext_type == "simple_cnn":
            scnn_max_lengths = [5000, 10000, 25000]
            if filter_max_lengths:
                scnn_max_lengths = [m for m in scnn_max_lengths if m in filter_max_lengths]

            for (
                dataset_path,
                dataset_name,
            ), use_feats, max_len, lr, ep in itertools.product(
                datasets,
                include_explicit_feature_options,
                scnn_max_lengths,
                learning_rates,
                epoch_counts,
            ):
                configs.append(
                    XWRLearnerForestConfig(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        use_explicit_features=use_feats,
                        feature_extractor_type="simple_cnn",
                        scnn_max_length=max_len,
                        learning_rate=lr,
                        epochs=ep,
                    )
                )

    for cfg in configs:
        cfg.contrastive_effect_enabled = contrastive_effect_enabled
        cfg.rlearner_effect_batch_size = rlearner_effect_batch_size
        cfg.rlearner_effect_accumulation_steps = rlearner_effect_accumulation_steps
        cfg.rlearner_effect_e_clip = rlearner_effect_e_clip
        cfg.rlearner_effect_grad_clip = rlearner_effect_grad_clip
        cfg.contrastive_bottleneck_dim = contrastive_bottleneck_dim
        cfg.contrastive_hidden_dim = contrastive_hidden_dim
        cfg.contrastive_batch_size = contrastive_batch_size
        cfg.contrastive_n_propensity_bins = contrastive_n_propensity_bins
        cfg.contrastive_overlap_min = contrastive_overlap_min
        cfg.contrastive_overlap_max = contrastive_overlap_max
        cfg.contrastive_min_arm_per_bin = contrastive_min_arm_per_bin
        cfg.contrastive_lambda_factual = contrastive_lambda_factual
        cfg.contrastive_lambda_contrast = contrastive_lambda_contrast
        cfg.contrastive_lambda_adversary = contrastive_lambda_adversary
        cfg.contrastive_lambda_z_l2 = contrastive_lambda_z_l2
        cfg.contrastive_target_clip = contrastive_target_clip
        cfg.contrastive_forest_x_mode = contrastive_forest_x_mode
        cfg.__post_init__()

    random.Random(42).shuffle(configs)
    return configs


def _is_real_cache_group(cache_hash: str, cache_info: Optional[dict]) -> bool:
    return bool(cache_info) and not cache_hash.startswith("__no_cache__")


def randomize_execution_groups(
    cache_groups: List[Tuple[str, dict, List[XWRLearnerForestConfig]]],
    seed: int = 42,
) -> List[Tuple[str, dict, List[XWRLearnerForestConfig]]]:
    """Randomize cache-group order and job order within each group.

    We still run grouped by cache key so a hidden-state cache is created once,
    but the groups returned by group_configs_by_cache_key are sorted. Shuffle
    them here so cached runs do not execute in dataset/extractor order.
    """
    rng = random.Random(seed)
    randomized_groups = []
    for cache_hash, cache_info, group_configs in cache_groups:
        shuffled_configs = list(group_configs)
        rng.shuffle(shuffled_configs)
        randomized_groups.append((cache_hash, cache_info, shuffled_configs))
    rng.shuffle(randomized_groups)
    return randomized_groups


def worker_process_fn(
    device: str,
    job_queue: mp.Queue,
    progress_queue: mp.Queue,
    output_dir: str,
    cache_hash: str,
    cache_info: Optional[dict],
    use_gpu_cache: bool,
    cache_base_dir: Optional[str] = None,
):
    """Worker process for cached LLM experiments."""
    output_dir_path = Path(output_dir)
    torch.set_default_dtype(torch.float32)

    cache_registry = {}
    gpu_store_registry = {}

    if _is_real_cache_group(cache_hash, cache_info):
        cache = _open_cache_for_worker(cache_hash, cache_info, cache_base_dir=cache_base_dir)
        cache_registry[cache_hash] = cache
        if use_gpu_cache:
            store = load_single_gpu_store(cache, cache_info, device)
            if store is not None:
                gpu_store_registry = {cache_hash: store}

    logger.info("Worker process started on %s (pid=%s)", device, os.getpid())

    while True:
        try:
            config = job_queue.get(timeout=2)
        except Exception:
            break

        config_hash = config.config_hash()
        try:
            result = run_single_experiment(
                config,
                device,
                output_dir_path,
                cache_registry,
                gpu_store_registry,
            )
            result_file = output_dir_path / "results" / f"{config_hash}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            progress_queue.put(("done", config_hash, result))
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Experiment %s FAILED: %s\n%s", config_hash, e, tb)
            error_result = {
                "config": asdict(config),
                "error": str(e),
                "skipped": True,
            }
            result_file = output_dir_path / "results" / f"{config_hash}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, "w") as f:
                json.dump(error_result, f, indent=2, default=str)
            progress_queue.put(("error", config_hash, error_result))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for store in gpu_store_registry.values():
        store.free()
    for cache in cache_registry.values():
        cache.close()
    logger.info("Worker process on %s (pid=%s) finished", device, os.getpid())


def worker_thread(
    device: str,
    job_queue: queue.Queue,
    results_dict: Dict[str, Any],
    output_dir: Path,
    lock: threading.Lock,
    progress_bar: tqdm,
):
    """Thread worker for live LLM or trainable non-cache groups."""
    while True:
        try:
            config = job_queue.get(timeout=1)
        except queue.Empty:
            break

        config_hash = config.config_hash()
        try:
            result = run_single_experiment(config, device, output_dir, {}, {})
            with lock:
                results_dict[config_hash] = result
                result_file = output_dir / "results" / f"{config_hash}.json"
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                progress_bar.update(1)
                if result.get("skipped"):
                    progress_bar.set_postfix_str(
                        f"Skipped: {result.get('error', 'unknown')[:30]}"
                    )
                else:
                    metrics = result.get("metrics", {})
                    progress_bar.set_postfix_str(
                        f"X/W RF-CF ITE corr: {metrics.get('ite_corr', float('nan')):.3f}"
                    )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Experiment %s FAILED: %s\n%s", config_hash, e, tb)
            with lock:
                error_result = {
                    "config": asdict(config),
                    "error": str(e),
                    "skipped": True,
                }
                results_dict[config_hash] = error_result
                result_file = output_dir / "results" / f"{config_hash}.json"
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, "w") as f:
                    json.dump(error_result, f, indent=2, default=str)
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"Error: {str(e)[:50]}")
        finally:
            job_queue.task_done()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_bool_grid(values: Optional[List[str]]) -> Optional[List[bool]]:
    if values is None:
        return None
    parsed = []
    for value in values:
        lowered = value.lower()
        if lowered in {"true", "1", "yes", "y"}:
            parsed.append(True)
        elif lowered in {"false", "0", "no", "n"}:
            parsed.append(False)
        else:
            raise argparse.ArgumentTypeError(
                f"Boolean grid values must be true/false, got {value}"
            )
    return parsed


def print_grid_summary(
    pending_configs: List[XWRLearnerForestConfig],
    completed_count: int,
    n_repeats: int,
    base_config_count: int,
):
    model_type_summary = {}
    extractor_summary = {}
    llm_model_summary = {}
    for config in pending_configs:
        model_type_summary[config.model_type] = model_type_summary.get(config.model_type, 0) + 1
        extractor_summary[config.feature_extractor_type] = (
            extractor_summary.get(config.feature_extractor_type, 0) + 1
        )
        if config.feature_extractor_type == "frozen_llm_pooler":
            llm_model_summary[config.flp_model_name] = (
                llm_model_summary.get(config.flp_model_name, 0) + 1
            )
        elif config.feature_extractor_type == "hierarchical_llm":
            llm_model_summary[config.hlm_model_name] = (
                llm_model_summary.get(config.hlm_model_name, 0) + 1
            )
        elif config.feature_extractor_type == "concept_token_cnn":
            llm_model_summary[config.ctcnn_model_name] = (
                llm_model_summary.get(config.ctcnn_model_name, 0) + 1
            )

    dataset_names = sorted(set(config.dataset_name for config in pending_configs))
    lr_values = sorted(set(config.learning_rate for config in pending_configs))
    epoch_values = sorted(set(config.epochs for config in pending_configs))
    explicit_values = sorted(set(config.use_explicit_features for config in pending_configs))
    contrastive_values = sorted(set(config.contrastive_effect_enabled for config in pending_configs))
    effect_batch_values = sorted(
        set(config.rlearner_effect_batch_size or config.batch_size for config in pending_configs)
    )
    accumulation_values = sorted(
        set(config.rlearner_effect_accumulation_steps for config in pending_configs)
    )

    print(f"\n{'=' * 60}")
    print("X/W R-Learner -> Causal Forest Grid Summary")
    print(f"{'=' * 60}")
    print(f"Base configs before repeats: {base_config_count}")
    print(f"Repeats: {n_repeats}")
    print(f"Total experiments to run: {len(pending_configs)}")
    if completed_count:
        print(f"Already completed (skipped): {completed_count}")
    print("Model path: causal_forest with cf_use_rlearner_representation=True")
    print("X/W split: enabled")
    print(f"Contrastive X stage: {', '.join(str(v) for v in contrastive_values)}")
    print(f"Effect physical batch sizes: {', '.join(str(v) for v in effect_batch_values)}")
    print(f"Effect accumulation steps: {', '.join(str(v) for v in accumulation_values)}")
    print("LLM hidden-state downprojection: disabled")
    print(f"Model types: {', '.join(f'{k}({v})' for k, v in sorted(model_type_summary.items()))}")
    print(f"Extractors:  {', '.join(f'{k}({v})' for k, v in sorted(extractor_summary.items()))}")
    if llm_model_summary:
        print(f"LLMs:       {', '.join(f'{k}({v})' for k, v in sorted(llm_model_summary.items()))}")
    print(f"Datasets:   {', '.join(dataset_names)}")
    print(f"Explicit features: {', '.join(str(v) for v in explicit_values)}")
    print(f"LR values:  {', '.join(str(v) for v in lr_values)}")
    print(f"Epochs:     {', '.join(str(v) for v in epoch_values)}")
    print(f"{'=' * 60}")


def aggregate_results(output_dir: Path, results_dict: Dict[str, Any]):
    all_results = []
    for result in results_dict.values():
        if not result.get("skipped"):
            row = {**result.get("config", {}), **result.get("metrics", {})}
            all_results.append(row)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / "all_results.csv", index=False)
        results_df.to_parquet(output_dir / "all_results.parquet", index=False)

        group_cols = [
            "dataset_name",
            "feature_extractor_type",
            "model_type",
            "rlearner_mode",
            "contrastive_effect_enabled",
            "contrastive_forest_x_mode",
            "flp_model_name",
            "hlm_model_name",
            "flp_max_length",
            "use_explicit_features",
            "learning_rate",
            "epochs",
        ]
        group_cols = [col for col in group_cols if col in results_df.columns]
        metric_agg = {}
        for metric in [
            "ite_corr",
            "ite_spearman_corr",
            "ate_bias",
            "propensity_auroc",
            "ite_mse",
            "ite_mae",
            "ci_coverage",
            "mean_ci_width",
        ]:
            if metric in results_df.columns:
                metric_agg[metric] = ["mean", "std"]

        summary = results_df.groupby(group_cols).agg(metric_agg)
        summary.to_csv(output_dir / "summary.csv")
        logger.info("\nSummary (mean +/- std across repeats):\n%s", summary)

    logger.info(
        "Total experiments: %d, Successful: %d, Skipped/Failed: %d",
        len(results_dict),
        len(all_results),
        len(results_dict) - len(all_results),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Oracle runner for R-learner X/W activations into CausalForestDML"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="../pcori_experiments/oracle_xw_rlearner_forest",
        help="Output directory for results",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="GPU devices to use",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of pending experiments to run",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved grid and exit without running experiments",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset directories containing dataset.parquet or dataset_with_extraction.parquet",
    )
    parser.add_argument(
        "--max-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Filter max lengths for frozen_llm_pooler/simple_cnn grids",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Opt in to pre-caching hidden states to disk",
    )
    parser.add_argument(
        "--gpu-cache",
        action="store_true",
        help="Keep pre-computed hidden states in GPU VRAM instead of disk cache",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of repeats per base config",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        default=[
            "Qwen/Qwen3.5-0.8B-Base",
            "Qwen/Qwen3.5-0.8B",
            "google/medgemma-1.5-4b-it",
        ],
        help="HuggingFace model names for LLM-based extractors",
    )
    parser.add_argument(
        "--chat-template-prompt",
        type=str,
        default=None,
        help="Optional chat template prompt for frozen_llm_pooler runs",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=str,
        default="auto",
        help="Concurrent workers per GPU for cached LLM experiments: 'auto' or integer",
    )
    parser.add_argument(
        "--filter-extractor-types",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Feature extractors to include: frozen_llm_pooler, hierarchical_llm, "
            "hierarchical_cnn, hierarchical_gru, simple_cnn"
        ),
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-5, 1e-4],
        help="Learning-rate grid",
    )
    parser.add_argument(
        "--epoch-counts",
        type=int,
        nargs="+",
        default=[50],
        help="Epoch-count grid",
    )
    parser.add_argument(
        "--rlearner-effect-batch-size",
        type=int,
        default=None,
        help="Physical batch size for the fixed-nuisance R-loss effect stage",
    )
    parser.add_argument(
        "--rlearner-effect-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for the fixed-nuisance R-loss effect stage",
    )
    parser.add_argument(
        "--rlearner-effect-e-clip",
        type=float,
        default=0.01,
        help="Propensity clipping value for fixed-nuisance R-loss",
    )
    parser.add_argument(
        "--rlearner-effect-grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm for the fixed-nuisance R-loss effect stage",
    )
    parser.add_argument(
        "--explicit-feature-options",
        type=str,
        nargs="+",
        default=None,
        help="Boolean grid for using role-tagged explicit features; default false true",
    )
    parser.add_argument(
        "--contrastive-effect",
        action="store_true",
        help="Use matched contrastive X-stage training instead of per-patient R-loss",
    )
    parser.add_argument("--contrastive-bottleneck-dim", type=int, default=8)
    parser.add_argument("--contrastive-hidden-dim", type=int, default=64)
    parser.add_argument("--contrastive-batch-size", type=int, default=16)
    parser.add_argument("--contrastive-n-propensity-bins", type=int, default=10)
    parser.add_argument("--contrastive-overlap-min", type=float, default=0.05)
    parser.add_argument("--contrastive-overlap-max", type=float, default=0.95)
    parser.add_argument("--contrastive-min-arm-per-bin", type=int, default=2)
    parser.add_argument("--contrastive-lambda-factual", type=float, default=1.0)
    parser.add_argument("--contrastive-lambda-contrast", type=float, default=2.0)
    parser.add_argument("--contrastive-lambda-adversary", type=float, default=0.05)
    parser.add_argument("--contrastive-lambda-z-l2", type=float, default=1e-4)
    parser.add_argument("--contrastive-target-clip", type=float, default=1.0)
    parser.add_argument(
        "--contrastive-forest-x-mode",
        type=str,
        default="bottleneck_plus_tau",
        choices=["bottleneck", "tau", "bottleneck_plus_tau"],
    )

    args = parser.parse_args()

    if args.workers_per_gpu != "auto":
        try:
            workers_per_gpu = int(args.workers_per_gpu)
            if workers_per_gpu < 1:
                parser.error("--workers-per-gpu must be >= 1")
        except ValueError:
            parser.error("--workers-per-gpu must be 'auto' or an integer")

    explicit_feature_options = _parse_bool_grid(args.explicit_feature_options)

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "command_line.txt").write_text(" ".join(sys.argv) + "\n")

    base_configs = generate_experiment_grid(
        dataset_paths=args.datasets,
        filter_max_lengths=args.max_lengths,
        model_names=args.model_names,
        chat_template_prompt=args.chat_template_prompt,
        filter_extractor_types=args.filter_extractor_types,
        learning_rates=args.learning_rates,
        epoch_counts=args.epoch_counts,
        include_explicit_feature_options=explicit_feature_options,
        rlearner_effect_batch_size=args.rlearner_effect_batch_size,
        rlearner_effect_accumulation_steps=args.rlearner_effect_accumulation_steps,
        rlearner_effect_e_clip=args.rlearner_effect_e_clip,
        rlearner_effect_grad_clip=args.rlearner_effect_grad_clip,
        contrastive_effect_enabled=args.contrastive_effect,
        contrastive_bottleneck_dim=args.contrastive_bottleneck_dim,
        contrastive_hidden_dim=args.contrastive_hidden_dim,
        contrastive_batch_size=args.contrastive_batch_size,
        contrastive_n_propensity_bins=args.contrastive_n_propensity_bins,
        contrastive_overlap_min=args.contrastive_overlap_min,
        contrastive_overlap_max=args.contrastive_overlap_max,
        contrastive_min_arm_per_bin=args.contrastive_min_arm_per_bin,
        contrastive_lambda_factual=args.contrastive_lambda_factual,
        contrastive_lambda_contrast=args.contrastive_lambda_contrast,
        contrastive_lambda_adversary=args.contrastive_lambda_adversary,
        contrastive_lambda_z_l2=args.contrastive_lambda_z_l2,
        contrastive_target_clip=args.contrastive_target_clip,
        contrastive_forest_x_mode=args.contrastive_forest_x_mode,
    )

    use_cache = args.cache or args.gpu_cache
    configs = []
    for base_config in base_configs:
        for repeat_idx in range(args.n_repeats):
            config = deepcopy(base_config)
            config.repeat_index = repeat_idx
            config.n_folds = args.n_folds
            config.flp_cache_hidden_states = use_cache
            config.hlm_cache_hidden_states = use_cache
            config.ctcnn_cache_hidden_states = use_cache
            config.flp_downprojection_dim = None
            config.hlm_downprojection_dim = None
            config.use_explicit_confounders = config.use_explicit_features
            configs.append(config)
    random.Random(42).shuffle(configs)

    logger.info(
        "Generated %d base configs x %d repeats = %d experiments",
        len(base_configs),
        args.n_repeats,
        len(configs),
    )
    logger.info(
        "Mode: %s",
        "cached hidden states" if use_cache else "live LLM forward per batch",
    )

    completed_hashes = set()
    results_dict: Dict[str, Any] = {}
    if args.resume:
        results_dir = output_dir / "results"
        if results_dir.exists():
            for result_file in results_dir.glob("*.json"):
                completed_hashes.add(result_file.stem)
                with open(result_file) as f:
                    results_dict[result_file.stem] = json.load(f)
            logger.info("Resuming: found %d completed experiments", len(completed_hashes))

    pending_configs = [config for config in configs if config.config_hash() not in completed_hashes]
    if args.max_experiments:
        pending_configs = pending_configs[: args.max_experiments]

    print_grid_summary(
        pending_configs=pending_configs,
        completed_count=len(completed_hashes),
        n_repeats=args.n_repeats,
        base_config_count=len(base_configs),
    )

    if args.dry_run or not pending_configs:
        if not pending_configs:
            logger.info("No experiments to run")
        return

    cache_base_dir = str(output_dir / ".oci_cache")
    cache_groups = randomize_execution_groups(
        group_configs_by_cache_key(pending_configs, use_cache)
    )
    logger.info("Randomized execution order across %d cache group(s)", len(cache_groups))

    if use_cache:
        wpg_per_device = {
            device: resolve_workers_per_gpu(args.workers_per_gpu, device, use_cache)
            for device in args.devices
        }
    else:
        wpg_per_device = {device: 1 for device in args.devices}

    progress_bar = tqdm(total=len(pending_configs), desc="X/W RF-CF experiments")

    for group_idx, (cache_hash, cache_info, group_configs) in enumerate(cache_groups):
        if not group_configs:
            continue

        real_cache_group = _is_real_cache_group(cache_hash, cache_info)
        if real_cache_group:
            logger.info("\n%s", "=" * 60)
            logger.info("Cache group %d/%d: %s", group_idx + 1, len(cache_groups), cache_hash)
            logger.info(
                "  max_length=%s, downprojection_dim=%s, dataset=%s",
                cache_info.get("max_length"),
                cache_info.get("downprojection_dim"),
                cache_info.get("dataset_name"),
            )
            logger.info("  %d experiment(s) in this group", len(group_configs))
            logger.info("%s", "=" * 60)

            cache = precompute_single_cache(
                cache_info,
                args.devices,
                cache_base_dir=cache_base_dir,
            )
            torch.set_default_dtype(torch.float32)
            cache.close()
            del cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            serializable_cache_info = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in cache_info.items()
            }

            ctx = mp.get_context("spawn")
            job_queue_mp = ctx.Queue()
            progress_queue = ctx.Queue()
            for config in group_configs:
                job_queue_mp.put(config)

            processes = []
            for device in args.devices:
                for worker_idx in range(wpg_per_device[device]):
                    process = ctx.Process(
                        target=worker_process_fn,
                        args=(
                            device,
                            job_queue_mp,
                            progress_queue,
                            str(output_dir),
                            cache_hash,
                            serializable_cache_info,
                            args.gpu_cache,
                            cache_base_dir,
                        ),
                        name=f"worker-{device}-{worker_idx}",
                    )
                    process.start()
                    processes.append(process)

            completed_in_group = 0
            expected = len(group_configs)
            while completed_in_group < expected:
                alive = [process for process in processes if process.is_alive()]
                if not alive and completed_in_group < expected:
                    logger.error(
                        "All workers died with %d experiments remaining",
                        expected - completed_in_group,
                    )
                    break

                try:
                    _status, config_hash, result = progress_queue.get(timeout=5)
                    results_dict[config_hash] = result
                    completed_in_group += 1
                    progress_bar.update(1)
                    if result.get("skipped"):
                        progress_bar.set_postfix_str(
                            f"Skipped: {result.get('error', 'unknown')[:30]}"
                        )
                    else:
                        metrics = result.get("metrics", {})
                        progress_bar.set_postfix_str(
                            f"X/W RF-CF ITE corr: {metrics.get('ite_corr', float('nan')):.3f}"
                        )
                except Exception:
                    pass

            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    logger.warning("Worker %s did not exit cleanly; terminating", process.name)
                    process.terminate()

        else:
            lock = threading.Lock()
            job_queue_t = queue.Queue()
            for config in group_configs:
                job_queue_t.put(config)

            threads = []
            for device in args.devices:
                n_threads = 1 if not use_cache else wpg_per_device[device]
                for worker_idx in range(n_threads):
                    thread = threading.Thread(
                        target=worker_thread,
                        args=(
                            device,
                            job_queue_t,
                            results_dict,
                            output_dir,
                            lock,
                            progress_bar,
                        ),
                        name=f"worker-{device}-{worker_idx}",
                    )
                    thread.start()
                    threads.append(thread)

            for thread in threads:
                thread.join()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    progress_bar.close()
    logger.info("Aggregating results...")
    aggregate_results(output_dir, results_dict)
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
