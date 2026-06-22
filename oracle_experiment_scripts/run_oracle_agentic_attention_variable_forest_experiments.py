#!/usr/bin/env python
"""Oracle runner for agentic attention-variable causal forests.

This script evaluates the ``agentic_attention_variable_forest`` pipeline on
synthetic oracle datasets.  It trains cross-fitted hierarchical-transformer
nuisance/R-loss models, asks an OpenAI-compatible agent to propose variables
from high-attention chunks, extracts those variables with vLLM, and fits the
final explicit-feature causal forest.

The script expects an OpenAI-compatible LLM server to already be running.  For
thinking models, start vLLM with the reasoning parser matching your model, e.g.:

    vllm serve Qwen/Qwen3.6-27B --reasoning-parser qwen3 ...
    vllm serve google/gemma-4-27b-it --reasoning-parser gemma4 ...

The defaults serve a Qwen model, but the pipeline is model-agnostic: pass
``--agent-model-name`` / ``--extraction-model-name`` to switch models.  The
extraction reasoning parser defaults to ``auto``, which infers the parser from
the model name (qwen->qwen3, gemma->gemma4, gpt-oss->openai_gptoss).

Outputs follow the oracle-script convention:

    <output-dir>/results/<config-hash>.json
    <output-dir>/all_results.csv
    <output-dir>/all_results.parquet
    <output-dir>/agentic_attention_predictions/<config-hash>/predictions.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import random
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.config import (
    AgenticAttentionVariableForestConfig,
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    ModelArchitectureConfig,
    TrainingConfig,
)
from oci.inference.agentic_attention_variable_forest import (
    run_agentic_attention_variable_forest,
)
from run_oracle_experiments import (
    _resolve_parquet_file,
    compute_metrics,
    load_feature_specs_from_metadata,
    select_agentic_initial_feature_specs,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_DATASETS = [
    "synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured",
    "synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured",
]


@dataclass
class AgenticAttentionOracleConfig:
    """Configuration for one agentic attention-variable oracle experiment."""

    dataset_path: str
    dataset_name: str
    repeat_index: int = 0
    model_type: str = "agentic_attention_variable_forest"
    feature_extractor_type: str = "hierarchical_transformer"

    n_folds: int = 5
    seed: int = 42
    sample_size: Optional[int] = None
    text_max_chars: Optional[int] = None

    htr_sentence_model: str = "prajjwal1/bert-tiny"
    htr_freeze_sentence_encoder: bool = True
    htr_chunk_size_words: int = 96
    htr_chunk_overlap_words: int = 24
    htr_max_chunks: int = 128
    htr_max_chunk_length: int = 128
    htr_num_layers: int = 2
    htr_num_heads: int = 4
    htr_transformer_dim: int = 256
    htr_projection_dim: int = 128
    htr_hash_embedding_dim: int = 256
    htr_sentence_encoder_batch_size: int = 128
    htr_sentence_encoder_backend: str = "auto"
    htr_sentence_pooling: str = "token_attention"
    htr_normalize_sentence_embeddings: bool = True
    htr_trainable_sentence_encoder_layers: int = 0
    htr_dropout: float = 0.1

    non_nuisance_epochs: int = 3
    batch_size: int = 8
    effect_batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    alpha_propensity: float = 1.0

    nuisance_folds: int = 5
    nuisance_epochs: int = 20
    nuisance_weight_decay: float = 0.05
    nuisance_label_smoothing: float = 0.02
    nuisance_calibration: str = "temperature_isotonic"
    effect_folds: int = 5
    fold_parallelism: str = "auto"
    outer_parallelism: str = "1"
    attention_top_k_chunks: int = 5
    candidate_proposals_per_fold: int = 3
    candidate_proposal_parallelism: str = "1"
    coverage_retry_attempts: int = 1
    signal_retry_attempts: int = 1
    association_alpha: float = 0.05
    association_min_n: int = 20
    association_min_non_missing: int = 10
    signal_cv_folds: int = 3
    min_signal_treatment_auroc: float = 0.55
    min_signal_outcome_auroc: float = 0.55
    consensus_min_folds: Optional[int] = 2
    consensus_min_fold_fraction: float = 2.0 / 3.0
    consensus_recovery_enabled: bool = True
    consensus_recovery_max_candidates: int = 12
    min_extraction_coverage: float = 0.10
    e_clip: float = 0.01
    r_stage_min_propensity: float = 0.0
    r_stage_max_propensity: float = 1.0
    effect_objective: str = "squared_r_loss"
    neural_stage_mode: str = "staged"
    joint_rlearner_gamma: float = 1.0
    interaction_l2_weight: float = 1e-3
    tarnet_offset_batch_size: Optional[int] = 128
    tarnet_offset_heterogeneity_weight: float = 0.1
    tarnet_offset_min_logit_std: float = 0.5
    residual_contrastive_enabled: bool = False
    residual_contrastive_use_for_effect_discovery: bool = True
    residual_contrastive_score: str = "r_score"
    residual_contrastive_high_quantile: float = 0.80
    residual_contrastive_low_quantile: float = 0.20
    residual_contrastive_neutral_abs_quantile: float = 0.40
    residual_contrastive_min_class_count: int = 10
    neural_only: bool = False

    cf_n_estimators: int = 200
    cf_min_samples_leaf: int = 10
    cf_max_depth: Optional[int] = None
    cf_max_features: str = "sqrt"
    cf_honest: bool = True
    cf_inference: bool = True

    initial_feature_count: int = 0
    initial_feature_strategy: str = "none"
    initial_feature_names: List[str] = field(default_factory=list)

    agent_server_url: str = "http://localhost:8000/v1"
    agent_model_name: str = "Qwen/Qwen3.6-27B"
    agent_api_key: str = "EMPTY"
    agent_temperature: float = 0.0
    agent_max_tokens: int = 25000
    agent_schema_repair_attempts: int = 1
    agent_request_max_retries: int = 3
    agent_retry_initial_delay: float = 1.0
    agent_retry_max_delay: float = 30.0
    agent_retry_backoff_factor: float = 2.0
    agent_save_context: bool = False
    agent_save_raw_output: bool = False

    extraction_server_url: str = "http://localhost:8000/v1"
    extraction_model_name: str = "Qwen/Qwen3.6-27B"
    extraction_mode: str = "server"
    extraction_reasoning_parser: Optional[str] = "auto"
    extraction_batch_size: int = 16
    extraction_max_retries: int = 3
    extraction_retry_initial_delay: float = 1.0
    extraction_retry_max_delay: float = 30.0
    extraction_retry_backoff_factor: float = 2.0
    extraction_temperature: float = 0.0
    extraction_max_tokens: int = 25000
    extraction_max_text_length: int = 400000
    extraction_cache_enabled: bool = True
    extraction_cache_dir: Optional[str] = None

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:12]


def _dataset_dir(dataset_path: str, parquet_file: Path) -> str:
    path = Path(dataset_path)
    if path.is_dir():
        return str(path)
    return str(parquet_file.parent)


def _load_initial_specs(config: AgenticAttentionOracleConfig, parquet_file: Path) -> List[ExplicitFeatureSpec]:
    dataset_dir = _dataset_dir(config.dataset_path, parquet_file)
    if config.initial_feature_names:
        specs = load_feature_specs_from_metadata(dataset_dir, section="features")
        by_name = {spec.name: spec for spec in specs}
        missing = [name for name in config.initial_feature_names if name not in by_name]
        if missing:
            raise ValueError(f"Initial feature names not found in metadata: {missing}")
        return [by_name[name] for name in config.initial_feature_names]

    if config.initial_feature_count <= 0:
        return []

    return select_agentic_initial_feature_specs(
        dataset_dir,
        count=config.initial_feature_count,
        strategy=config.initial_feature_strategy,
    )


def _make_applied_config(
    config: AgenticAttentionOracleConfig,
    parquet_file: Path,
    initial_specs: Sequence[ExplicitFeatureSpec],
) -> AppliedInferenceConfig:
    return AppliedInferenceConfig(
        clinical_question=(
            "Estimate heterogeneous treatment effects from clinical text and "
            "identify text-derived confounders and effect modifiers."
        ),
        outcome_type="binary",
        dataset_path=str(parquet_file),
        text_column="clinical_text",
        outcome_column="outcome_indicator",
        treatment_column="treatment_indicator",
        cv_folds=config.n_folds,
        architecture=ModelArchitectureConfig(
            model_type="agentic_attention_variable_forest",
            feature_extractor_type="hierarchical_transformer",
            htr_sentence_model=config.htr_sentence_model,
            htr_freeze_sentence_encoder=config.htr_freeze_sentence_encoder,
            htr_chunk_size_words=config.htr_chunk_size_words,
            htr_chunk_overlap_words=config.htr_chunk_overlap_words,
            htr_max_chunks=config.htr_max_chunks,
            htr_max_chunk_length=config.htr_max_chunk_length,
            htr_num_layers=config.htr_num_layers,
            htr_num_heads=config.htr_num_heads,
            htr_transformer_dim=config.htr_transformer_dim,
            htr_projection_dim=config.htr_projection_dim,
            htr_hash_embedding_dim=config.htr_hash_embedding_dim,
            htr_sentence_encoder_batch_size=config.htr_sentence_encoder_batch_size,
            htr_sentence_encoder_backend=config.htr_sentence_encoder_backend,
            htr_sentence_pooling=config.htr_sentence_pooling,
            htr_normalize_sentence_embeddings=config.htr_normalize_sentence_embeddings,
            htr_trainable_sentence_encoder_layers=config.htr_trainable_sentence_encoder_layers,
            htr_dropout=config.htr_dropout,
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=config.cf_n_estimators,
                max_depth=config.cf_max_depth,
                min_samples_leaf=config.cf_min_samples_leaf,
                max_features=config.cf_max_features,
                honest=config.cf_honest,
                inference=config.cf_inference,
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=max(2, config.n_folds),
                inner_folds=max(2, config.nuisance_folds),
                max_iterations=1,
                max_additions_per_iter=config.candidate_proposals_per_fold,
                max_removals_per_iter=0,
                agent_server_url=config.agent_server_url,
                agent_model_name=config.agent_model_name,
                agent_api_key=config.agent_api_key,
                agent_temperature=config.agent_temperature,
                agent_max_tokens=config.agent_max_tokens,
                agent_schema_repair_attempts=config.agent_schema_repair_attempts,
                agent_request_max_retries=config.agent_request_max_retries,
                agent_retry_initial_delay=config.agent_retry_initial_delay,
                agent_retry_max_delay=config.agent_retry_max_delay,
                agent_retry_backoff_factor=config.agent_retry_backoff_factor,
                save_agent_context=config.agent_save_context,
                save_agent_raw_output=config.agent_save_raw_output,
                random_state=config.seed + config.repeat_index,
            ),
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=config.nuisance_folds,
                nuisance_epochs=config.nuisance_epochs,
                nuisance_weight_decay=config.nuisance_weight_decay,
                nuisance_label_smoothing=config.nuisance_label_smoothing,
                nuisance_calibration=config.nuisance_calibration,
                effect_folds=config.effect_folds,
                fold_parallelism=config.fold_parallelism,
                outer_parallelism=config.outer_parallelism,
                attention_top_k_chunks=config.attention_top_k_chunks,
                candidate_proposals_per_fold=config.candidate_proposals_per_fold,
                candidate_proposal_parallelism=config.candidate_proposal_parallelism,
                coverage_retry_attempts=config.coverage_retry_attempts,
                signal_retry_attempts=config.signal_retry_attempts,
                association_alpha=config.association_alpha,
                association_min_n=config.association_min_n,
                association_min_non_missing=config.association_min_non_missing,
                signal_cv_folds=config.signal_cv_folds,
                min_signal_treatment_auroc=config.min_signal_treatment_auroc,
                min_signal_outcome_auroc=config.min_signal_outcome_auroc,
                consensus_min_folds=config.consensus_min_folds,
                consensus_min_fold_fraction=config.consensus_min_fold_fraction,
                consensus_recovery_enabled=config.consensus_recovery_enabled,
                consensus_recovery_max_candidates=(
                    config.consensus_recovery_max_candidates
                ),
                min_extraction_coverage=config.min_extraction_coverage,
                e_clip=config.e_clip,
                r_stage_min_propensity=config.r_stage_min_propensity,
                r_stage_max_propensity=config.r_stage_max_propensity,
                effect_objective=config.effect_objective,
                neural_stage_mode=config.neural_stage_mode,
                joint_rlearner_gamma=config.joint_rlearner_gamma,
                interaction_l2_weight=config.interaction_l2_weight,
                tarnet_offset_batch_size=config.tarnet_offset_batch_size,
                tarnet_offset_heterogeneity_weight=(
                    config.tarnet_offset_heterogeneity_weight
                ),
                tarnet_offset_min_logit_std=config.tarnet_offset_min_logit_std,
                residual_contrastive_enabled=config.residual_contrastive_enabled,
                residual_contrastive_use_for_effect_discovery=(
                    config.residual_contrastive_use_for_effect_discovery
                ),
                residual_contrastive_score=config.residual_contrastive_score,
                residual_contrastive_high_quantile=(
                    config.residual_contrastive_high_quantile
                ),
                residual_contrastive_low_quantile=(
                    config.residual_contrastive_low_quantile
                ),
                residual_contrastive_neutral_abs_quantile=(
                    config.residual_contrastive_neutral_abs_quantile
                ),
                residual_contrastive_min_class_count=(
                    config.residual_contrastive_min_class_count
                ),
                neural_only=config.neural_only,
            ),
        ),
        training=TrainingConfig(
            epochs=config.non_nuisance_epochs,
            batch_size=config.batch_size,
            effect_batch_size=config.effect_batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_clip_norm=config.gradient_clip_norm,
            alpha_propensity=config.alpha_propensity,
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=bool(initial_specs),
            features=list(initial_specs),
            vllm_mode=config.extraction_mode,
            vllm_server_url=config.extraction_server_url,
            vllm_model_name=config.extraction_model_name,
            vllm_reasoning_parser=config.extraction_reasoning_parser,
            extraction_batch_size=config.extraction_batch_size,
            extraction_max_retries=config.extraction_max_retries,
            extraction_retry_initial_delay=config.extraction_retry_initial_delay,
            extraction_retry_max_delay=config.extraction_retry_max_delay,
            extraction_retry_backoff_factor=config.extraction_retry_backoff_factor,
            extraction_temperature=config.extraction_temperature,
            extraction_max_tokens=config.extraction_max_tokens,
            extraction_max_text_length=config.extraction_max_text_length,
            cache_enabled=config.extraction_cache_enabled,
            cache_dir=config.extraction_cache_dir,
        ),
    )


def _prepare_dataset(config: AgenticAttentionOracleConfig, parquet_file: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_file)
    if config.sample_size is not None and config.sample_size < len(df):
        df = df.sample(n=config.sample_size, random_state=config.seed + config.repeat_index)
        df = df.reset_index(drop=True)
    if config.text_max_chars is not None:
        df = df.copy()
        df["clinical_text_full_chars"] = df["clinical_text"].astype(str).str.len()
        df["clinical_text"] = df["clinical_text"].astype(str).str.slice(0, config.text_max_chars)
    return df


def _safe_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    required = {
        "pred_ite_prob",
        "true_ite_prob",
        "pred_propensity_prob",
        "treatment_indicator",
        "pred_y0_prob",
        "pred_y1_prob",
        "true_y0_prob",
        "true_y1_prob",
        "outcome_indicator",
    }
    if not required.issubset(results_df.columns):
        return {}
    return compute_metrics(
        pred_ite=results_df["pred_ite_prob"].values,
        true_ite=results_df["true_ite_prob"].values,
        pred_propensity=results_df["pred_propensity_prob"].values,
        true_treatment=results_df["treatment_indicator"].values,
        pred_y0=results_df["pred_y0_prob"].values,
        pred_y1=results_df["pred_y1_prob"].values,
        true_y0=results_df["true_y0_prob"].values,
        true_y1=results_df["true_y1_prob"].values,
        true_outcome=results_df["outcome_indicator"].values,
        tau_lower=(
            results_df["pred_ite_lower"].values
            if "pred_ite_lower" in results_df.columns
            else None
        ),
        tau_upper=(
            results_df["pred_ite_upper"].values
            if "pred_ite_upper" in results_df.columns
            else None
        ),
    )


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _safe_roc_auc(y_true: pd.Series, y_score: pd.Series) -> Optional[float]:
    y = np.asarray(y_true)
    score = np.asarray(y_score)
    mask = pd.notna(y) & pd.notna(score)
    if int(mask.sum()) < 2 or len(np.unique(y[mask])) < 2:
        return None
    try:
        return float(roc_auc_score(y[mask], score[mask]))
    except ValueError:
        return None


def _safe_neural_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    required = {"tau_hat_r_stage", "r_loss", "r_loss_at_zero_tau", "e_hat", "m_hat"}
    if not required.issubset(results_df.columns):
        return {}

    metrics: Dict[str, Any] = {
        "neural_effect_objective": (
            str(results_df["effect_objective"].iloc[0])
            if "effect_objective" in results_df.columns and len(results_df) > 0
            else "squared_r_loss"
        ),
        "neural_r_loss_mean": _finite_or_none(results_df["r_loss"].mean()),
        "neural_r_loss_at_zero_tau_mean": _finite_or_none(
            results_df["r_loss_at_zero_tau"].mean()
        ),
        "neural_tau_hat_mean": _finite_or_none(results_df["tau_hat_r_stage"].mean()),
        "neural_tau_hat_std": _finite_or_none(results_df["tau_hat_r_stage"].std()),
    }
    base_loss = metrics["neural_r_loss_at_zero_tau_mean"]
    r_loss = metrics["neural_r_loss_mean"]
    if base_loss is not None and base_loss > 0 and r_loss is not None:
        # Interprets the neural R-stage tau model against the no-effect baseline:
        # >0 means tau_hat reduces residual R-loss vs tau=0, ~0 means no gain,
        # and <0 means the learned tau worsens the R-loss. A positive value is
        # useful but not sufficient evidence of oracle CATE recovery.
        metrics["neural_r_loss_relative_improvement"] = float(1.0 - r_loss / base_loss)
    if {
        "effect_loss",
        "effect_loss_at_zero_tau",
    }.issubset(results_df.columns):
        metrics["neural_effect_loss_mean"] = _finite_or_none(
            results_df["effect_loss"].mean()
        )
        metrics["neural_effect_loss_at_zero_tau_mean"] = _finite_or_none(
            results_df["effect_loss_at_zero_tau"].mean()
        )
        effect_zero = metrics.get("neural_effect_loss_at_zero_tau_mean")
        effect_loss = metrics.get("neural_effect_loss_mean")
        if effect_zero is not None and effect_zero > 0 and effect_loss is not None:
            metrics["neural_effect_loss_relative_improvement"] = float(
                1.0 - effect_loss / effect_zero
            )
    if "tau_logit_modifier" in results_df.columns:
        modifier = results_df["tau_logit_modifier"].to_numpy(dtype=float)
        finite = modifier[np.isfinite(modifier)]
        metrics["neural_tau_logit_modifier_mean"] = (
            _finite_or_none(np.mean(finite)) if finite.size > 0 else None
        )
        metrics["neural_tau_logit_modifier_std"] = (
            _finite_or_none(np.std(finite)) if finite.size > 0 else None
        )
    if "treatment_indicator" in results_df.columns:
        metrics["neural_propensity_auroc"] = _safe_roc_auc(
            results_df["treatment_indicator"],
            results_df["e_hat"],
        )
    if "outcome_indicator" in results_df.columns:
        metrics["neural_outcome_auroc"] = _safe_roc_auc(
            results_df["outcome_indicator"],
            results_df["m_hat"],
        )
    if {"true_ite_prob", "tau_hat_r_stage"}.issubset(results_df.columns):
        metrics["neural_r_stage_ite_corr"] = _finite_or_none(
            results_df["true_ite_prob"].corr(results_df["tau_hat_r_stage"])
        )
        metrics["neural_r_stage_ite_spearman_corr"] = _finite_or_none(
            results_df["true_ite_prob"].corr(
                results_df["tau_hat_r_stage"],
                method="spearman",
            )
        )
    if {"true_treatment_prob", "e_hat"}.issubset(results_df.columns):
        metrics["neural_true_propensity_corr"] = _finite_or_none(
            results_df["true_treatment_prob"].corr(results_df["e_hat"])
        )
    if {"true_outcome_prob", "m_hat"}.issubset(results_df.columns):
        metrics["neural_true_outcome_rmse"] = _finite_or_none(
            np.sqrt(
                mean_squared_error(
                    results_df["true_outcome_prob"],
                    results_df["m_hat"],
                )
            )
        )
    return metrics


def _selected_feature_summary(results_df: pd.DataFrame) -> Dict[str, float]:
    if "selected_feature_names" not in results_df.columns:
        return {}
    selected = [str(v) for v in results_df["selected_feature_names"].fillna("").tolist()]
    counts = [0 if not value else len([name for name in value.split(",") if name]) for value in selected]
    return {
        "agentic_attention_n_selected_features_mean": float(np.mean(counts)),
        "agentic_attention_n_selected_feature_sets": float(len(set(selected))),
    }


def _load_candidate_names(path: Path) -> List[str]:
    names: List[str] = []
    if not path.exists():
        return names
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for proposal in row.get("proposals", []):
                name = proposal.get("name")
                if name and name not in names:
                    names.append(name)
    return names


def _append_unique(names: List[str], value: Any) -> None:
    name = str(value or "").strip()
    if name and name not in names:
        names.append(name)


def _load_selected_variable_summary(path: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "confounders": [],
        "effect_modifiers": [],
        "selected_features": [],
        "by_outer_fold": [],
    }
    if not path.exists():
        return summary

    with open(path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        return summary

    for row in rows:
        if not isinstance(row, dict):
            continue
        confounders = [str(name) for name in row.get("confounders", []) if name]
        effect_modifiers = [
            str(name) for name in row.get("effect_modifiers", []) if name
        ]
        selected_features = []
        for feature in row.get("selected_features", []):
            if not isinstance(feature, dict):
                continue
            name = feature.get("name")
            if name:
                selected_features.append(str(name))
            roles = set(feature.get("roles", []))
            if "confounder" in roles:
                _append_unique(confounders, name)
            if "effect_modifier" in roles:
                _append_unique(effect_modifiers, name)

        for name in confounders:
            _append_unique(summary["confounders"], name)
            _append_unique(summary["selected_features"], name)
        for name in effect_modifiers:
            _append_unique(summary["effect_modifiers"], name)
            _append_unique(summary["selected_features"], name)
        for name in selected_features:
            _append_unique(summary["selected_features"], name)

        summary["by_outer_fold"].append(
            {
                "outer_fold": row.get("outer_fold"),
                "confounders": confounders,
                "effect_modifiers": effect_modifiers,
                "selected_features": selected_features,
            }
        )
    return summary


def run_experiment(
    config: AgenticAttentionOracleConfig,
    output_dir: Path,
    device: torch.device,
    devices: Sequence[torch.device],
    num_workers: int,
) -> Dict[str, Any]:
    parquet_file = _resolve_parquet_file(config.dataset_path)
    if parquet_file is None:
        return {
            "config": asdict(config),
            "skipped": True,
            "error": f"Dataset not found: {config.dataset_path}",
            "metrics": {},
            "n_samples": 0,
        }

    random.seed(config.seed + config.repeat_index)
    np.random.seed(config.seed + config.repeat_index)
    torch.manual_seed(config.seed + config.repeat_index)

    initial_specs = _load_initial_specs(config, parquet_file)
    applied_config = _make_applied_config(config, parquet_file, initial_specs)
    df = _prepare_dataset(config, parquet_file)

    config_hash = config.config_hash()
    prediction_path = (
        output_dir
        / "agentic_attention_predictions"
        / config_hash
        / "predictions.parquet"
    )
    run_agentic_attention_variable_forest(
        dataset=df,
        config=applied_config,
        output_path=prediction_path,
        device=device,
        devices=devices,
        num_workers=num_workers,
    )

    results_df = pd.read_parquet(prediction_path)
    artifact_dir = prediction_path.parent / "agentic_attention_variable_forest"
    metrics = _safe_metrics(results_df)
    metrics.update(_safe_neural_metrics(results_df))
    metrics.update(_selected_feature_summary(results_df))
    metrics["agentic_attention_n_initial_features"] = float(len(initial_specs))

    return {
        "config": asdict(config),
        "metrics": metrics,
        "n_samples": len(results_df),
        "skipped": False,
        "error": None,
        "artifacts": {
            "predictions_path": str(prediction_path),
            "artifact_dir": str(artifact_dir),
        },
        "agentic_attention_variables_tried": {
            "confounders": _load_candidate_names(
                artifact_dir / "confounder_candidates_by_fold.jsonl"
            ),
            "effect_modifiers": _load_candidate_names(
                artifact_dir / "effect_modifier_candidates_by_fold.jsonl"
            ),
        },
        "agentic_attention_variables_selected": _load_selected_variable_summary(
            artifact_dir / "consensus.json"
        ),
    }


def _result_row(config_hash: str, result: Dict[str, Any]) -> Dict[str, Any]:
    config = result.get("config", {})
    row: Dict[str, Any] = {
        "config_hash": config_hash,
        "skipped": result.get("skipped", False),
        "error": result.get("error"),
        "n_samples": result.get("n_samples", 0),
    }
    for key in [
        "dataset_name",
        "dataset_path",
        "repeat_index",
        "model_type",
        "feature_extractor_type",
        "htr_sentence_model",
        "htr_freeze_sentence_encoder",
        "htr_sentence_encoder_batch_size",
        "htr_sentence_encoder_backend",
        "htr_sentence_pooling",
        "htr_trainable_sentence_encoder_layers",
        "n_folds",
        "nuisance_folds",
        "effect_folds",
        "fold_parallelism",
        "outer_parallelism",
        "candidate_proposal_parallelism",
        "candidate_proposals_per_fold",
        "batch_size",
        "effect_batch_size",
        "alpha_propensity",
        "r_stage_min_propensity",
        "r_stage_max_propensity",
        "consensus_recovery_enabled",
        "consensus_recovery_max_candidates",
        "effect_objective",
        "neural_stage_mode",
        "joint_rlearner_gamma",
        "interaction_l2_weight",
        "tarnet_offset_batch_size",
        "tarnet_offset_heterogeneity_weight",
        "tarnet_offset_min_logit_std",
        "residual_contrastive_enabled",
        "residual_contrastive_use_for_effect_discovery",
        "residual_contrastive_score",
        "residual_contrastive_high_quantile",
        "residual_contrastive_low_quantile",
        "residual_contrastive_neutral_abs_quantile",
        "residual_contrastive_min_class_count",
        "neural_only",
        "initial_feature_count",
        "initial_feature_strategy",
    ]:
        row[key] = config.get(key)
    for key, value in result.get("metrics", {}).items():
        row[key] = value
    artifacts = result.get("artifacts", {})
    row["predictions_path"] = artifacts.get("predictions_path")
    row["artifact_dir"] = artifacts.get("artifact_dir")
    selected = result.get("agentic_attention_variables_selected", {})
    row["agentic_attention_selected_confounders"] = ",".join(
        selected.get("confounders", [])
    )
    row["agentic_attention_selected_effect_modifiers"] = ",".join(
        selected.get("effect_modifiers", [])
    )
    row["agentic_attention_selected_features"] = ",".join(
        selected.get("selected_features", [])
    )
    return row


def _write_aggregate_outputs(output_dir: Path) -> None:
    rows = []
    results_dir = output_dir / "results"
    if not results_dir.exists():
        return
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            result = json.load(f)
        rows.append(_result_row(path.stem, result))
    if not rows:
        return
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    results_df.to_parquet(output_dir / "all_results.parquet", index=False)
    with open(output_dir / "all_results.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def _parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _parse_optional_positive_int(value: str) -> Optional[int]:
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected positive integer or 'none', got {value!r}"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"Expected positive integer or 'none', got {value!r}"
        )
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--output-dir",
        default="../pcori_experiments/oracle_agentic_attention_variable_forest",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Worker count for CPU-side batching/tokenization; also used by fold "
            "parallelism on CPU when --fold-parallelism=auto."
        ),
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["auto"],
        help=(
            "Training devices to use for outer/inner neural fold scheduling, "
            "e.g. --devices cuda:0 cuda:1 cuda:2 cuda:3. Defaults to auto, "
            "which uses cuda:0 when available and otherwise cpu."
        ),
    )
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--text-max-chars", type=int, default=None)

    parser.add_argument("--htr-sentence-model", default="prajjwal1/bert-tiny")
    parser.add_argument("--htr-freeze-sentence-encoder", type=_parse_bool, default=True)
    parser.add_argument("--htr-chunk-size-words", type=int, default=96)
    parser.add_argument("--htr-chunk-overlap-words", type=int, default=24)
    parser.add_argument("--htr-max-chunks", type=int, default=128)
    parser.add_argument("--htr-max-chunk-length", type=int, default=128)
    parser.add_argument("--htr-num-layers", type=int, default=2)
    parser.add_argument("--htr-num-heads", type=int, default=4)
    parser.add_argument("--htr-transformer-dim", type=int, default=256)
    parser.add_argument("--htr-projection-dim", type=int, default=128)
    parser.add_argument("--htr-hash-embedding-dim", type=int, default=256)
    parser.add_argument("--htr-sentence-encoder-batch-size", type=int, default=128)
    parser.add_argument(
        "--htr-sentence-encoder-backend",
        default="auto",
        choices=["auto", "sentence_transformers", "transformers"],
    )
    parser.add_argument(
        "--htr-sentence-pooling",
        default="token_attention",
        choices=["auto", "cls", "last", "mean", "token_attention"],
    )
    parser.add_argument("--htr-normalize-sentence-embeddings", type=_parse_bool, default=True)
    parser.add_argument("--htr-trainable-sentence-encoder-layers", type=int, default=0)
    parser.add_argument("--htr-dropout", type=float, default=0.1)

    parser.add_argument(
        "--non-nuisance-epochs",
        dest="non_nuisance_epochs",
        type=int,
        default=3,
        help="Epochs for non-nuisance neural stages, including the R/effect stage.",
    )
    parser.add_argument(
        "--epochs",
        dest="non_nuisance_epochs",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--effect-batch-size", type=int, default=32)
    parser.add_argument(
        "--tarnet-offset-batch-size",
        type=int,
        default=128,
        help=(
            "Batch size for neural-stage-mode=tarnet_offset training, "
            "prediction, and attribution. Overrides --effect-batch-size for "
            "that stage."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--alpha-propensity",
        type=float,
        default=1.0,
        help="Weight on the treatment/propensity prediction loss in neural stages.",
    )

    parser.add_argument("--nuisance-folds", type=int, default=5)
    parser.add_argument("--nuisance-epochs", type=int, default=20)
    parser.add_argument("--nuisance-weight-decay", type=float, default=0.05)
    parser.add_argument("--nuisance-label-smoothing", type=float, default=0.02)
    parser.add_argument(
        "--nuisance-calibration",
        choices=["none", "temperature", "isotonic", "temperature_isotonic"],
        default="temperature_isotonic",
    )
    parser.add_argument("--effect-folds", type=int, default=5)
    parser.add_argument(
        "--fold-parallelism",
        "--inner-fold-parallelism",
        dest="fold_parallelism",
        default="auto",
        help=(
            "Number of cross-fit nuisance/effect folds to train concurrently. "
            "'auto' uses num_workers on CPU, stays serial on single-device CUDA, "
            "and uses the configured device count when multiple CUDA devices are "
            "provided."
        ),
    )
    parser.add_argument(
        "--outer-parallelism",
        default="1",
        help=(
            "Number of outer analysis folds to run concurrently. Use 'auto' or "
            "a positive integer."
        ),
    )
    parser.add_argument("--attention-top-k-chunks", type=int, default=5)
    parser.add_argument("--candidate-proposals-per-fold", type=int, default=3)
    parser.add_argument(
        "--candidate-proposal-parallelism",
        default="1",
        help=(
            "Number of per-inner-fold agent candidate proposal calls to run "
            "concurrently. Use 'auto' or a positive integer."
        ),
    )
    parser.add_argument("--coverage-retry-attempts", type=int, default=1)
    parser.add_argument("--signal-retry-attempts", type=int, default=1)
    parser.add_argument("--association-alpha", type=float, default=0.05)
    parser.add_argument("--association-min-n", type=int, default=20)
    parser.add_argument("--association-min-non-missing", type=int, default=10)
    parser.add_argument("--signal-cv-folds", type=int, default=3)
    parser.add_argument("--min-signal-treatment-auroc", type=float, default=0.55)
    parser.add_argument("--min-signal-outcome-auroc", type=float, default=0.55)
    parser.add_argument(
        "--consensus-min-folds",
        type=_parse_optional_positive_int,
        default=2,
    )
    parser.add_argument("--consensus-min-fold-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--consensus-recovery-enabled", type=_parse_bool, default=True)
    parser.add_argument("--consensus-recovery-max-candidates", type=int, default=12)
    parser.add_argument("--min-extraction-coverage", type=float, default=0.10)
    parser.add_argument("--e-clip", type=float, default=0.01)
    parser.add_argument("--r-stage-min-propensity", type=float, default=0.0)
    parser.add_argument("--r-stage-max-propensity", type=float, default=1.0)
    parser.add_argument(
        "--effect-objective",
        choices=["squared_r_loss", "logistic_r_loss", "pseudo_outcome_mse"],
        default="squared_r_loss",
        help=(
            "Neural effect-stage objective. logistic_r_loss trains a Bernoulli "
            "R-learner logit modifier and reports probability-scale CATE; "
            "pseudo_outcome_mse regresses the R pseudo-outcome directly."
        ),
    )
    parser.add_argument(
        "--neural-stage-mode",
        choices=["staged", "joint_rlearner", "interaction_outcome", "tarnet_offset"],
        default="staged",
        help=(
            "Neural learning mode. staged trains nuisance and R/effect models "
            "sequentially; joint_rlearner trains nuisance and tau heads in one "
            "HTR model with detached nuisance predictions inside the R-loss; "
            "interaction_outcome trains a supervised outcome model with an "
            "explicit treatment-interaction branch; tarnet_offset trains "
            "nuisance first, then treatment-specific outcome-logit offset heads."
        ),
    )
    parser.add_argument(
        "--joint-rlearner-gamma",
        type=float,
        default=1.0,
        help="Weight on the detached-nuisance R-loss in neural-stage-mode=joint_rlearner.",
    )
    parser.add_argument(
        "--interaction-l2-weight",
        type=float,
        default=1e-3,
        help=(
            "L2 penalty on the interaction/offset outcome component in "
            "neural-stage-mode=interaction_outcome or tarnet_offset."
        ),
    )
    parser.add_argument(
        "--tarnet-offset-heterogeneity-weight",
        type=float,
        default=0.1,
        help=(
            "Weight for the TarNet offset within-batch heterogeneity floor. "
            "The penalty is max(0, min_std^2 - var(offset1 - offset0))."
        ),
    )
    parser.add_argument(
        "--tarnet-offset-min-logit-std",
        type=float,
        default=0.5,
        help=(
            "Target minimum within-batch standard deviation for the TarNet "
            "offset logit contrast offset1 - offset0."
        ),
    )
    parser.add_argument(
        "--residual-contrastive-enabled",
        action="store_true",
        help=(
            "Train tail-vs-neutral residual-score text classifiers and save "
            "their attention evidence."
        ),
    )
    parser.add_argument(
        "--residual-contrastive-use-for-effect-discovery",
        type=_parse_bool,
        default=True,
        help=(
            "When residual contrastive training is enabled, use its tail-vs-neutral "
            "attention evidence for effect-modifier proposals."
        ),
    )
    parser.add_argument(
        "--residual-contrastive-score",
        default="r_score",
        choices=["r_score", "r_score_normalized"],
        help="Residual score used to define high/low tails.",
    )
    parser.add_argument("--residual-contrastive-high-quantile", type=float, default=0.80)
    parser.add_argument("--residual-contrastive-low-quantile", type=float, default=0.20)
    parser.add_argument(
        "--residual-contrastive-neutral-abs-quantile",
        type=float,
        default=0.40,
    )
    parser.add_argument("--residual-contrastive-min-class-count", type=int, default=10)
    parser.add_argument(
        "--neural-only",
        action="store_true",
        help="Run nuisance/R-stage neural cross-fitting and attention artifacts only.",
    )

    parser.add_argument("--cf-n-estimators", type=int, default=200)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=10)
    parser.add_argument("--cf-max-depth", type=int, default=None)
    parser.add_argument("--cf-max-features", default="sqrt")
    parser.add_argument("--cf-honest", type=_parse_bool, default=True)
    parser.add_argument("--cf-inference", type=_parse_bool, default=True)

    parser.add_argument("--initial-feature-counts", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--initial-feature-strategies",
        nargs="+",
        default=["true_first"],
        choices=["none", "true_first", "modifiers_first", "mixed", "distractors"],
    )
    parser.add_argument("--initial-feature-names", nargs="*", default=[])

    parser.add_argument(
        "--agent-server-url",
        "--agent-server-urls",
        dest="agent_server_url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible agent endpoint, or comma-separated endpoints.",
    )
    parser.add_argument(
        "--agent-model-name",
        default="Qwen/Qwen3.6-27B",
        help=(
            "Agent model id. Use 'auto' to read the first model id from the "
            "OpenAI-compatible server /v1/models endpoint. The legacy default "
            "Qwen/Qwen3.6-27B is also autodiscovered at runtime."
        ),
    )
    parser.add_argument("--agent-api-key", default="EMPTY")
    parser.add_argument("--agent-temperature", type=float, default=0.0)
    parser.add_argument("--agent-max-tokens", type=int, default=25000)
    parser.add_argument("--agent-schema-repair-attempts", type=int, default=1)
    parser.add_argument("--agent-request-max-retries", type=int, default=3)
    parser.add_argument("--agent-retry-initial-delay", type=float, default=1.0)
    parser.add_argument("--agent-retry-max-delay", type=float, default=30.0)
    parser.add_argument("--agent-retry-backoff-factor", type=float, default=2.0)
    parser.add_argument("--save-agent-context", action="store_true")
    parser.add_argument("--save-agent-raw-output", action="store_true")

    parser.add_argument(
        "--extraction-server-url",
        "--extraction-server-urls",
        dest="extraction_server_url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible extraction endpoint, or comma-separated endpoints.",
    )
    parser.add_argument(
        "--extraction-model-name",
        default="Qwen/Qwen3.6-27B",
        help=(
            "Extraction model id. Use 'auto' to read the first model id from the "
            "OpenAI-compatible server /v1/models endpoint. The legacy default "
            "Qwen/Qwen3.6-27B is also autodiscovered at runtime."
        ),
    )
    parser.add_argument("--extraction-mode", default="server")
    parser.add_argument("--extraction-reasoning-parser", default="auto")
    parser.add_argument("--extraction-batch-size", type=int, default=16)
    parser.add_argument("--extraction-max-retries", type=int, default=3)
    parser.add_argument("--extraction-retry-initial-delay", type=float, default=1.0)
    parser.add_argument("--extraction-retry-max-delay", type=float, default=30.0)
    parser.add_argument("--extraction-retry-backoff-factor", type=float, default=2.0)
    parser.add_argument("--extraction-temperature", type=float, default=0.0)
    parser.add_argument("--extraction-max-tokens", type=int, default=25000)
    parser.add_argument("--extraction-max-text-length", type=int, default=400000)
    parser.add_argument("--extraction-cache-dir", default=None)
    parser.add_argument("--no-extraction-cache", action="store_true")
    return parser


def _make_configs(args: argparse.Namespace) -> List[AgenticAttentionOracleConfig]:
    configs: List[AgenticAttentionOracleConfig] = []
    datasets = [(path, Path(path).name) for path in args.datasets]
    for (dataset_path, dataset_name), repeat_idx, initial_count in itertools.product(
        datasets,
        range(args.n_repeats),
        args.initial_feature_counts,
    ):
        strategies = ["none"] if initial_count <= 0 or args.initial_feature_names else args.initial_feature_strategies
        for strategy in strategies:
            configs.append(
                AgenticAttentionOracleConfig(
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    repeat_index=repeat_idx,
                    n_folds=args.n_folds,
                    sample_size=args.sample_size,
                    text_max_chars=args.text_max_chars,
                    htr_sentence_model=args.htr_sentence_model,
                    htr_freeze_sentence_encoder=args.htr_freeze_sentence_encoder,
                    htr_chunk_size_words=args.htr_chunk_size_words,
                    htr_chunk_overlap_words=args.htr_chunk_overlap_words,
                    htr_max_chunks=args.htr_max_chunks,
                    htr_max_chunk_length=args.htr_max_chunk_length,
                    htr_num_layers=args.htr_num_layers,
                    htr_num_heads=args.htr_num_heads,
                    htr_transformer_dim=args.htr_transformer_dim,
                    htr_projection_dim=args.htr_projection_dim,
                    htr_hash_embedding_dim=args.htr_hash_embedding_dim,
                    htr_sentence_encoder_batch_size=args.htr_sentence_encoder_batch_size,
                    htr_sentence_encoder_backend=args.htr_sentence_encoder_backend,
                    htr_sentence_pooling=args.htr_sentence_pooling,
                    htr_normalize_sentence_embeddings=args.htr_normalize_sentence_embeddings,
                    htr_trainable_sentence_encoder_layers=(
                        args.htr_trainable_sentence_encoder_layers
                    ),
                    htr_dropout=args.htr_dropout,
                    non_nuisance_epochs=args.non_nuisance_epochs,
                    batch_size=args.batch_size,
                    effect_batch_size=args.effect_batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    gradient_clip_norm=args.gradient_clip_norm,
                    alpha_propensity=args.alpha_propensity,
                    nuisance_folds=args.nuisance_folds,
                    nuisance_epochs=args.nuisance_epochs,
                    nuisance_weight_decay=args.nuisance_weight_decay,
                    nuisance_label_smoothing=args.nuisance_label_smoothing,
                    nuisance_calibration=args.nuisance_calibration,
                    effect_folds=args.effect_folds,
                    fold_parallelism=args.fold_parallelism,
                    outer_parallelism=args.outer_parallelism,
                    attention_top_k_chunks=args.attention_top_k_chunks,
                    candidate_proposals_per_fold=args.candidate_proposals_per_fold,
                    candidate_proposal_parallelism=args.candidate_proposal_parallelism,
                    coverage_retry_attempts=args.coverage_retry_attempts,
                    signal_retry_attempts=args.signal_retry_attempts,
                    association_alpha=args.association_alpha,
                    association_min_n=args.association_min_n,
                    association_min_non_missing=args.association_min_non_missing,
                    signal_cv_folds=args.signal_cv_folds,
                    min_signal_treatment_auroc=args.min_signal_treatment_auroc,
                    min_signal_outcome_auroc=args.min_signal_outcome_auroc,
                    consensus_min_folds=args.consensus_min_folds,
                    consensus_min_fold_fraction=args.consensus_min_fold_fraction,
                    consensus_recovery_enabled=args.consensus_recovery_enabled,
                    consensus_recovery_max_candidates=(
                        args.consensus_recovery_max_candidates
                    ),
                    min_extraction_coverage=args.min_extraction_coverage,
                    e_clip=args.e_clip,
                    r_stage_min_propensity=args.r_stage_min_propensity,
                    r_stage_max_propensity=args.r_stage_max_propensity,
                    effect_objective=args.effect_objective,
                    neural_stage_mode=args.neural_stage_mode,
                    joint_rlearner_gamma=args.joint_rlearner_gamma,
                    interaction_l2_weight=args.interaction_l2_weight,
                    tarnet_offset_batch_size=args.tarnet_offset_batch_size,
                    tarnet_offset_heterogeneity_weight=(
                        args.tarnet_offset_heterogeneity_weight
                    ),
                    tarnet_offset_min_logit_std=args.tarnet_offset_min_logit_std,
                    residual_contrastive_enabled=args.residual_contrastive_enabled,
                    residual_contrastive_use_for_effect_discovery=(
                        args.residual_contrastive_use_for_effect_discovery
                    ),
                    residual_contrastive_score=args.residual_contrastive_score,
                    residual_contrastive_high_quantile=(
                        args.residual_contrastive_high_quantile
                    ),
                    residual_contrastive_low_quantile=(
                        args.residual_contrastive_low_quantile
                    ),
                    residual_contrastive_neutral_abs_quantile=(
                        args.residual_contrastive_neutral_abs_quantile
                    ),
                    residual_contrastive_min_class_count=(
                        args.residual_contrastive_min_class_count
                    ),
                    neural_only=args.neural_only,
                    cf_n_estimators=args.cf_n_estimators,
                    cf_min_samples_leaf=args.cf_min_samples_leaf,
                    cf_max_depth=args.cf_max_depth,
                    cf_max_features=args.cf_max_features,
                    cf_honest=args.cf_honest,
                    cf_inference=args.cf_inference,
                    initial_feature_count=initial_count,
                    initial_feature_strategy=strategy,
                    initial_feature_names=list(args.initial_feature_names),
                    agent_server_url=args.agent_server_url,
                    agent_model_name=args.agent_model_name,
                    agent_api_key=args.agent_api_key,
                    agent_temperature=args.agent_temperature,
                    agent_max_tokens=args.agent_max_tokens,
                    agent_schema_repair_attempts=args.agent_schema_repair_attempts,
                    agent_request_max_retries=args.agent_request_max_retries,
                    agent_retry_initial_delay=args.agent_retry_initial_delay,
                    agent_retry_max_delay=args.agent_retry_max_delay,
                    agent_retry_backoff_factor=args.agent_retry_backoff_factor,
                    agent_save_context=args.save_agent_context,
                    agent_save_raw_output=args.save_agent_raw_output,
                    extraction_server_url=args.extraction_server_url,
                    extraction_model_name=args.extraction_model_name,
                    extraction_mode=args.extraction_mode,
                    extraction_reasoning_parser=args.extraction_reasoning_parser,
                    extraction_batch_size=args.extraction_batch_size,
                    extraction_max_retries=args.extraction_max_retries,
                    extraction_retry_initial_delay=args.extraction_retry_initial_delay,
                    extraction_retry_max_delay=args.extraction_retry_max_delay,
                    extraction_retry_backoff_factor=args.extraction_retry_backoff_factor,
                    extraction_temperature=args.extraction_temperature,
                    extraction_max_tokens=args.extraction_max_tokens,
                    extraction_max_text_length=args.extraction_max_text_length,
                    extraction_cache_enabled=not args.no_extraction_cache,
                    extraction_cache_dir=args.extraction_cache_dir,
                )
            )
    random.Random(42).shuffle(configs)
    if args.max_experiments:
        configs = configs[: args.max_experiments]
    return configs


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.n_repeats < 1:
        parser.error("--n-repeats must be >= 1")
    if args.n_folds < 2:
        parser.error("--n-folds must be >= 2")
    if args.non_nuisance_epochs < 1:
        parser.error("--non-nuisance-epochs must be >= 1")
    if args.nuisance_folds < 2 or args.effect_folds < 2:
        parser.error("--nuisance-folds and --effect-folds must be >= 2")
    if args.nuisance_epochs < 1:
        parser.error("--nuisance-epochs must be >= 1")
    if args.nuisance_weight_decay < 0:
        parser.error("--nuisance-weight-decay must be >= 0")
    if not 0.0 <= args.nuisance_label_smoothing < 1.0:
        parser.error("--nuisance-label-smoothing must be in [0, 1)")
    if args.candidate_proposals_per_fold < 1:
        parser.error("--candidate-proposals-per-fold must be >= 1")
    if str(args.fold_parallelism).strip().lower() != "auto":
        try:
            if int(args.fold_parallelism) < 1:
                raise ValueError
        except ValueError:
            parser.error("--fold-parallelism must be 'auto' or a positive integer")
    if str(args.outer_parallelism).strip().lower() != "auto":
        try:
            if int(args.outer_parallelism) < 1:
                raise ValueError
        except ValueError:
            parser.error("--outer-parallelism must be 'auto' or a positive integer")
    if str(args.candidate_proposal_parallelism).strip().lower() != "auto":
        try:
            if int(args.candidate_proposal_parallelism) < 1:
                raise ValueError
        except ValueError:
            parser.error(
                "--candidate-proposal-parallelism must be 'auto' or a positive integer"
            )
    if args.coverage_retry_attempts < 0:
        parser.error("--coverage-retry-attempts must be >= 0")
    if args.signal_retry_attempts < 0:
        parser.error("--signal-retry-attempts must be >= 0")
    if not 0.0 < args.association_alpha < 1.0:
        parser.error("--association-alpha must be in (0, 1)")
    if args.association_min_n < 1:
        parser.error("--association-min-n must be >= 1")
    if args.association_min_non_missing < 1:
        parser.error("--association-min-non-missing must be >= 1")
    if args.signal_cv_folds < 2:
        parser.error("--signal-cv-folds must be >= 2")
    if not 0.5 <= args.min_signal_treatment_auroc <= 1.0:
        parser.error("--min-signal-treatment-auroc must be in [0.5, 1]")
    if not 0.5 <= args.min_signal_outcome_auroc <= 1.0:
        parser.error("--min-signal-outcome-auroc must be in [0.5, 1]")
    if args.consensus_min_folds is not None and args.consensus_min_folds < 1:
        parser.error("--consensus-min-folds must be >= 1 or 'none'")
    if not 0.0 < args.consensus_min_fold_fraction <= 1.0:
        parser.error("--consensus-min-fold-fraction must be in (0, 1]")
    if args.consensus_recovery_max_candidates < 0:
        parser.error("--consensus-recovery-max-candidates must be >= 0")
    if args.htr_sentence_encoder_batch_size < 1:
        parser.error("--htr-sentence-encoder-batch-size must be >= 1")
    if args.htr_trainable_sentence_encoder_layers < 0:
        parser.error("--htr-trainable-sentence-encoder-layers must be >= 0")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.effect_batch_size < 1:
        parser.error("--effect-batch-size must be >= 1")
    if args.tarnet_offset_batch_size < 1:
        parser.error("--tarnet-offset-batch-size must be >= 1")
    if args.tarnet_offset_heterogeneity_weight < 0:
        parser.error("--tarnet-offset-heterogeneity-weight must be >= 0")
    if args.tarnet_offset_min_logit_std < 0:
        parser.error("--tarnet-offset-min-logit-std must be >= 0")
    if not 0.0 <= args.r_stage_min_propensity < args.r_stage_max_propensity <= 1.0:
        parser.error(
            "--r-stage-min-propensity and --r-stage-max-propensity must satisfy "
            "0 <= min < max <= 1"
        )
    if not (
        0.0
        < args.residual_contrastive_low_quantile
        < args.residual_contrastive_high_quantile
        < 1.0
    ):
        parser.error(
            "--residual-contrastive-low-quantile and --residual-contrastive-high-quantile "
            "must satisfy 0 < low < high < 1"
        )
    if not 0.0 < args.residual_contrastive_neutral_abs_quantile < 1.0:
        parser.error("--residual-contrastive-neutral-abs-quantile must be in (0, 1)")
    if args.residual_contrastive_min_class_count < 1:
        parser.error("--residual-contrastive-min-class-count must be >= 1")
    if any(count < 0 for count in args.initial_feature_counts):
        parser.error("--initial-feature-counts must be >= 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command_line.txt").write_text(" ".join(sys.argv) + "\n")

    device_names = list(args.devices or ["auto"])
    if "auto" in {str(name).strip().lower() for name in device_names}:
        if len(device_names) > 1:
            parser.error("--devices auto cannot be combined with explicit device names")
        device_names = ["cuda:0" if torch.cuda.is_available() else "cpu"]
    devices = [torch.device(name) for name in device_names]
    device = devices[0]

    configs = _make_configs(args)
    pending = []
    for config in configs:
        result_path = output_dir / "results" / f"{config.config_hash()}.json"
        if args.resume and result_path.exists():
            continue
        pending.append(config)

    print(f"Agentic attention oracle experiments: {len(pending)} pending / {len(configs)} total")
    print(f"Datasets: {', '.join(sorted({c.dataset_name for c in configs}))}")
    print(f"Devices: {', '.join(str(item) for item in devices)}")
    print(f"Output: {output_dir}")

    for idx, config in enumerate(pending, start=1):
        config_hash = config.config_hash()
        result_path = output_dir / "results" / f"{config_hash}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[%s/%s] Running %s repeat=%s hash=%s",
            idx,
            len(pending),
            config.dataset_name,
            config.repeat_index,
            config_hash,
        )
        try:
            result = run_experiment(config, output_dir, device, devices, args.num_workers)
        except Exception as exc:
            logger.error("Experiment %s failed: %s\n%s", config_hash, exc, traceback.format_exc())
            result = {
                "config": asdict(config),
                "metrics": {},
                "n_samples": 0,
                "skipped": True,
                "error": str(exc),
            }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        _write_aggregate_outputs(output_dir)

    _write_aggregate_outputs(output_dir)
    logger.info("Done. Aggregate results written under %s", output_dir)


if __name__ == "__main__":
    main()
