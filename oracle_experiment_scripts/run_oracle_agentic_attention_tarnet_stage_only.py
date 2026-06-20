#!/usr/bin/env python
"""TarNet-offset-stage-only oracle runner for agentic attention experiments.

This script skips nuisance training, agentic variable proposal, extraction, and
the final forest. It loads saved cross-fitted nuisance predictions, retrains
only the TarNet-offset text stage, and writes neural diagnostics plus gradient
token-attribution artifacts.

The nuisance source may be any of:

* a ``nuisance_oof_predictions.parquet`` file;
* an ``agentic_attention_variable_forest`` artifact directory;
* a ``crossfit_fold_checkpoints/nuisance`` checkpoint directory;
* a full oracle experiment output directory containing
  ``agentic_attention_predictions/<hash>/agentic_attention_variable_forest``.

Typical use after a full oracle run that completed nuisance checkpoints:

    python oracle_experiment_scripts/run_oracle_agentic_attention_tarnet_stage_only.py \
        --dataset synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured \
        --nuisance-source ../pcori_experiments/agentic_htr_tarnet_offset \
        --output-dir ../pcori_experiments/agentic_htr_tarnet_offset_tarnet_only \
        --device cuda:2 \
        --n-folds 5 \
        --effect-folds 5 \
        --fold-parallelism 1 \
        --htr-freeze-sentence-encoder false \
        --non-nuisance-epochs 20

Outputs:

    <output-dir>/results/<config-hash>.json
    <output-dir>/all_results.csv
    <output-dir>/all_results.parquet
    <output-dir>/tarnet_stage_only_predictions/<config-hash>/predictions.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.config import (  # noqa: E402
    AgenticAttentionVariableForestConfig,
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ModelArchitectureConfig,
    TrainingConfig,
)
from oci.inference.agentic_attention_variable_forest import (  # noqa: E402
    AgenticAttentionVariableForestRunner,
)
from oci.utils.calibration import binary_calibration_metrics  # noqa: E402
from run_oracle_experiments import _resolve_parquet_file  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_DATASET = (
    "synthetic_data/example_synthetic_datasets/"
    "one_confounder_one_effect_modifier_nsclc_with_structured"
)

NUISANCE_REQUIRED_COLUMNS = {"_oci_row_id", "outer_fold", "e_hat", "m_hat"}


@dataclass
class TarNetStageOnlyOracleConfig:
    """Configuration for one TarNet-offset-stage-only oracle experiment."""

    dataset_path: str
    dataset_name: str
    nuisance_source: str
    nuisance_config_hash: Optional[str] = None
    outer_folds: Optional[List[int]] = None

    n_folds: int = 5
    seed: int = 42
    repeat_index: int = 0
    sample_size: Optional[int] = None
    text_max_chars: Optional[int] = None

    htr_sentence_model: str = "prajjwal1/bert-tiny"
    htr_freeze_sentence_encoder: bool = False
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

    non_nuisance_epochs: int = 20
    batch_size: int = 8
    effect_batch_size: int = 32
    tarnet_offset_batch_size: Optional[int] = 128
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
    attention_top_k_chunks: int = 5
    e_clip: float = 0.01
    r_stage_min_propensity: float = 0.0
    r_stage_max_propensity: float = 1.0
    interaction_l2_weight: float = 0.0
    tarnet_offset_heterogeneity_weight: float = 0.1
    tarnet_offset_min_logit_std: float = 0.5

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


class _UnusedAgent:
    def propose(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("Proposal agent is not available in TarNet-stage-only mode.")


class _UnusedExtractionProvider:
    def ensure_features(self, df: pd.DataFrame, specs: List[Any]) -> pd.DataFrame:
        raise RuntimeError("Feature extraction is not available in TarNet-stage-only mode.")


def _parse_bool(value: str) -> bool:
    lowered = str(value).lower().strip()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _parse_outer_folds(value: Optional[str]) -> Optional[List[int]]:
    if value is None or str(value).strip().lower() in {"", "all"}:
        return None
    folds = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            fold = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Expected comma-separated integer fold IDs, got {value!r}"
            ) from exc
        if fold < 1:
            raise argparse.ArgumentTypeError("Outer fold IDs must be >= 1")
        folds.append(fold)
    if not folds:
        return None
    return sorted(set(folds))


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _safe_roc_auc(y_true: Sequence[Any], y_score: Sequence[Any]) -> Optional[float]:
    y = np.asarray(y_true)
    score = np.asarray(y_score)
    mask = pd.notna(y) & pd.notna(score)
    if int(mask.sum()) < 2 or len(np.unique(y[mask])) < 2:
        return None
    try:
        return float(roc_auc_score(y[mask], score[mask]))
    except ValueError:
        return None


def _safe_corr(left: pd.Series, right: pd.Series, *, method: str = "pearson") -> Optional[float]:
    try:
        value = left.corr(right, method=method)
    except Exception:
        return None
    return _finite_or_none(value)


def _prepare_dataset(config: TarNetStageOnlyOracleConfig, parquet_file: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_file)
    if config.sample_size is not None and config.sample_size < len(df):
        df = df.sample(
            n=config.sample_size,
            random_state=config.seed + config.repeat_index,
        ).reset_index(drop=True)
    if config.text_max_chars is not None:
        df = df.copy()
        df["clinical_text_full_chars"] = df["clinical_text"].astype(str).str.len()
        df["clinical_text"] = (
            df["clinical_text"].astype(str).str.slice(0, config.text_max_chars)
        )
    return df


def _make_applied_config(
    config: TarNetStageOnlyOracleConfig,
    parquet_file: Path,
) -> AppliedInferenceConfig:
    return AppliedInferenceConfig(
        clinical_question=(
            "Estimate text-derived treatment-specific outcome offsets from "
            "saved nuisance predictions."
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
            htr_normalize_sentence_embeddings=(
                config.htr_normalize_sentence_embeddings
            ),
            htr_trainable_sentence_encoder_layers=(
                config.htr_trainable_sentence_encoder_layers
            ),
            htr_dropout=config.htr_dropout,
            explicit_feature_forest=ExplicitFeatureForestConfig(),
            agentic_feature_search=AgenticFeatureSearchConfig(),
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=config.nuisance_folds,
                nuisance_epochs=config.nuisance_epochs,
                nuisance_weight_decay=config.nuisance_weight_decay,
                nuisance_label_smoothing=config.nuisance_label_smoothing,
                nuisance_calibration=config.nuisance_calibration,
                effect_folds=config.effect_folds,
                fold_parallelism=config.fold_parallelism,
                attention_top_k_chunks=config.attention_top_k_chunks,
                e_clip=config.e_clip,
                r_stage_min_propensity=config.r_stage_min_propensity,
                r_stage_max_propensity=config.r_stage_max_propensity,
                neural_stage_mode="tarnet_offset",
                interaction_l2_weight=config.interaction_l2_weight,
                tarnet_offset_batch_size=config.tarnet_offset_batch_size,
                tarnet_offset_heterogeneity_weight=(
                    config.tarnet_offset_heterogeneity_weight
                ),
                tarnet_offset_min_logit_std=config.tarnet_offset_min_logit_std,
                neural_only=True,
            ),
        ),
        training=TrainingConfig(
            epochs=config.non_nuisance_epochs,
            batch_size=config.batch_size,
            effect_batch_size=config.effect_batch_size,
            learning_rate=config.learning_rate,
            lr_schedule="linear",
            weight_decay=config.weight_decay,
            gradient_clip_norm=config.gradient_clip_norm,
            alpha_propensity=config.alpha_propensity,
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=False, features=[]),
    )


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    unique = []
    for path in paths:
        resolved = path.resolve() if path.exists() else path
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _candidate_artifact_dirs(source: Path, config_hash: Optional[str]) -> List[Path]:
    candidates: List[Path] = [
        source,
        source / "agentic_attention_variable_forest",
    ]
    if config_hash:
        candidates.append(
            source
            / "agentic_attention_predictions"
            / config_hash
            / "agentic_attention_variable_forest"
        )
    predictions_root = source / "agentic_attention_predictions"
    if predictions_root.exists():
        dirs = [
            path / "agentic_attention_variable_forest"
            for path in predictions_root.iterdir()
            if path.is_dir()
        ]
        dirs = [path for path in dirs if path.exists()]
        dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        if config_hash is None and len(dirs) > 1:
            logger.warning(
                "Multiple nuisance artifact hashes found under %s; using newest: %s. "
                "Pass --nuisance-config-hash to pin one.",
                predictions_root,
                dirs[0],
            )
        candidates.extend(dirs)
    return _unique_paths(candidates)


def _load_checkpoint_predictions(checkpoint_dir: Path) -> pd.DataFrame:
    paths = sorted(checkpoint_dir.glob("*_predictions.parquet"))
    usable_paths = []
    for path in paths:
        stem = path.name.removesuffix("_predictions.parquet")
        done = checkpoint_dir / f"{stem}.done.json"
        if done.exists():
            usable_paths.append(path)
        else:
            logger.warning("Skipping checkpoint without done marker: %s", path)
    if not usable_paths:
        raise FileNotFoundError(f"No completed nuisance checkpoint predictions in {checkpoint_dir}")
    frames = [pd.read_parquet(path) for path in usable_paths]
    nuisance = pd.concat(frames, ignore_index=True)
    return nuisance.sort_values(["outer_fold", "_oci_row_id"]).reset_index(drop=True)


def _load_nuisance_predictions(
    source: Path,
    config_hash: Optional[str],
) -> Tuple[pd.DataFrame, Path]:
    if source.is_file():
        nuisance = pd.read_parquet(source)
        return nuisance.copy(), source

    if not source.exists():
        raise FileNotFoundError(f"Nuisance source does not exist: {source}")

    candidates: List[Tuple[str, Path]] = []
    for artifact_dir in _candidate_artifact_dirs(source, config_hash):
        candidates.append(("parquet", artifact_dir / "nuisance_oof_predictions.parquet"))
        candidates.append(
            (
                "checkpoint_dir",
                artifact_dir / "crossfit_fold_checkpoints" / "nuisance",
            )
        )
    if source.name == "nuisance":
        candidates.append(("checkpoint_dir", source))

    for kind, path in candidates:
        if kind == "parquet" and path.exists():
            nuisance = pd.read_parquet(path)
            return nuisance.copy(), path
        if kind == "checkpoint_dir" and path.exists():
            nuisance = _load_checkpoint_predictions(path)
            return nuisance, path

    searched = "\n  ".join(str(path) for _, path in candidates)
    raise FileNotFoundError(
        "Could not find nuisance predictions under source. Searched:\n  " + searched
    )


def _validate_nuisance_predictions(nuisance: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(NUISANCE_REQUIRED_COLUMNS - set(nuisance.columns))
    if missing:
        raise ValueError(
            "Nuisance predictions are missing required columns: "
            + ", ".join(missing)
        )
    nuisance = nuisance.copy()
    nuisance["outer_fold"] = pd.to_numeric(
        nuisance["outer_fold"],
        errors="raise",
    ).astype(int)
    for column in ["e_hat", "m_hat", "e_hat_raw", "m_hat_raw"]:
        if column in nuisance.columns:
            nuisance[column] = pd.to_numeric(nuisance[column], errors="coerce")
    if nuisance[["e_hat", "m_hat"]].isna().any().any():
        raise ValueError("Nuisance predictions contain NaN e_hat or m_hat values.")
    duplicate_mask = nuisance.duplicated(["outer_fold", "_oci_row_id"])
    if duplicate_mask.any():
        examples = (
            nuisance.loc[duplicate_mask, ["outer_fold", "_oci_row_id"]]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "Nuisance predictions contain duplicate outer_fold/_oci_row_id rows: "
            f"{examples}"
        )
    return nuisance


def _nuisance_for_outer_fold(
    nuisance_df: pd.DataFrame,
    discovery_df: pd.DataFrame,
    outer_fold: int,
    applied_config: AppliedInferenceConfig,
) -> pd.DataFrame:
    fold_df = nuisance_df[nuisance_df["outer_fold"] == int(outer_fold)]
    if fold_df.empty:
        available = sorted(nuisance_df["outer_fold"].dropna().astype(int).unique().tolist())
        raise ValueError(
            f"No nuisance predictions found for outer_fold={outer_fold}. "
            f"Available outer folds: {available}"
        )

    expected_ids = discovery_df["_oci_row_id"].to_numpy()
    by_id = fold_df.set_index("_oci_row_id", drop=False)
    missing_ids = sorted(set(expected_ids) - set(by_id.index))
    if missing_ids:
        preview = ", ".join(str(value) for value in missing_ids[:10])
        raise ValueError(
            f"Nuisance predictions for outer_fold={outer_fold} are missing "
            f"{len(missing_ids)} discovery row id(s): {preview}. Make sure "
            "the stage-only run uses the same dataset order/sample settings as "
            "the nuisance run, or restrict --outer-folds to completed folds."
        )

    predictions = by_id.loc[expected_ids].reset_index(drop=True).copy()
    y = discovery_df[applied_config.outcome_column].to_numpy(dtype=float)
    t = discovery_df[applied_config.treatment_column].to_numpy(dtype=float)
    e_hat = predictions["e_hat"].to_numpy(dtype=float)
    m_hat = predictions["m_hat"].to_numpy(dtype=float)

    predictions["outer_fold"] = int(outer_fold)
    if "e_hat_raw" not in predictions.columns:
        predictions["e_hat_raw"] = e_hat
    if "m_hat_raw" not in predictions.columns:
        predictions["m_hat_raw"] = m_hat
    predictions["y_residual"] = y - m_hat
    predictions["t_residual"] = t - e_hat
    if "r_loss_at_zero_tau" not in predictions.columns:
        predictions["r_loss_at_zero_tau"] = predictions["y_residual"] ** 2
    if "nuisance_fold" not in predictions.columns:
        predictions["nuisance_fold"] = np.nan
    return predictions


def _aggregate_neural_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "mode": "tarnet_stage_only",
        "neural_stage_mode": "tarnet_offset",
        "n_train_rows": int(len(results_df)),
        "outer_folds_completed": (
            ",".join(str(int(v)) for v in sorted(results_df["outer_fold"].unique()))
            if "outer_fold" in results_df.columns and len(results_df) > 0
            else ""
        ),
    }
    if results_df.empty:
        return metrics

    for column, key in [
        ("tau_hat_r_stage", "tau_hat_r_stage"),
        ("tau_logit_modifier", "tau_logit_modifier"),
        ("offset0", "offset0"),
        ("offset1", "offset1"),
        ("offset_contrast", "offset_contrast"),
    ]:
        if column in results_df.columns:
            series = pd.to_numeric(results_df[column], errors="coerce")
            metrics[f"{key}_mean"] = _finite_or_none(series.mean())
            metrics[f"{key}_std"] = _finite_or_none(series.std())

    if {"r_loss", "r_loss_at_zero_tau"}.issubset(results_df.columns):
        r_loss = _finite_or_none(results_df["r_loss"].mean())
        zero_loss = _finite_or_none(results_df["r_loss_at_zero_tau"].mean())
        metrics["r_loss_mean"] = r_loss
        metrics["r_loss_at_zero_tau_mean"] = zero_loss
        if zero_loss is not None and zero_loss > 0 and r_loss is not None:
            metrics["r_loss_relative_improvement"] = float(1.0 - r_loss / zero_loss)

    if {"effect_loss", "effect_loss_at_zero_tau"}.issubset(results_df.columns):
        effect_loss = _finite_or_none(results_df["effect_loss"].mean())
        effect_zero = _finite_or_none(results_df["effect_loss_at_zero_tau"].mean())
        metrics["effect_loss_mean"] = effect_loss
        metrics["effect_loss_at_zero_tau_mean"] = effect_zero
        if effect_zero is not None and effect_zero > 0 and effect_loss is not None:
            metrics["effect_loss_relative_improvement"] = float(
                1.0 - effect_loss / effect_zero
            )

    if "r_stage_train_eligible" in results_df.columns:
        eligible = results_df["r_stage_train_eligible"].astype(bool)
        metrics["r_stage_train_eligible_rows"] = int(eligible.sum())
        metrics["r_stage_train_eligible_fraction"] = _finite_or_none(eligible.mean())

    if {"treatment_indicator", "e_hat"}.issubset(results_df.columns):
        treatment = results_df["treatment_indicator"].to_numpy()
        e_hat = results_df["e_hat"].to_numpy()
        metrics["nuisance_treatment_auroc"] = _safe_roc_auc(treatment, e_hat)
        metrics.update(binary_calibration_metrics(treatment, e_hat, prefix="nuisance_treatment"))
        if "e_hat_raw" in results_df.columns:
            e_raw = results_df["e_hat_raw"].to_numpy()
            metrics["nuisance_treatment_raw_auroc"] = _safe_roc_auc(treatment, e_raw)
            metrics.update(
                binary_calibration_metrics(treatment, e_raw, prefix="nuisance_treatment_raw")
            )

    if {"outcome_indicator", "m_hat"}.issubset(results_df.columns):
        outcome = results_df["outcome_indicator"].to_numpy()
        m_hat = results_df["m_hat"].to_numpy()
        metrics["nuisance_outcome_auroc"] = _safe_roc_auc(outcome, m_hat)
        metrics.update(binary_calibration_metrics(outcome, m_hat, prefix="nuisance_outcome"))
        if "m_hat_raw" in results_df.columns:
            m_raw = results_df["m_hat_raw"].to_numpy()
            metrics["nuisance_outcome_raw_auroc"] = _safe_roc_auc(outcome, m_raw)
            metrics.update(
                binary_calibration_metrics(outcome, m_raw, prefix="nuisance_outcome_raw")
            )

    if {"true_ite_prob", "tau_hat_r_stage"}.issubset(results_df.columns):
        metrics["r_stage_ite_corr"] = _safe_corr(
            results_df["true_ite_prob"],
            results_df["tau_hat_r_stage"],
        )
        metrics["r_stage_ite_spearman_corr"] = _safe_corr(
            results_df["true_ite_prob"],
            results_df["tau_hat_r_stage"],
            method="spearman",
        )
    if {"true_treatment_prob", "e_hat"}.issubset(results_df.columns):
        metrics["nuisance_true_propensity_corr"] = _safe_corr(
            results_df["true_treatment_prob"],
            results_df["e_hat"],
        )
    if {"true_outcome_prob", "m_hat"}.issubset(results_df.columns):
        metrics["nuisance_true_outcome_rmse"] = _finite_or_none(
            np.sqrt(mean_squared_error(results_df["true_outcome_prob"], results_df["m_hat"]))
        )
    if {"true_y0_prob", "y0_hat"}.issubset(results_df.columns):
        metrics["y0_hat_true_corr"] = _safe_corr(results_df["true_y0_prob"], results_df["y0_hat"])
        metrics["y0_hat_true_rmse"] = _finite_or_none(
            np.sqrt(mean_squared_error(results_df["true_y0_prob"], results_df["y0_hat"]))
        )
    if {"true_y1_prob", "y1_hat"}.issubset(results_df.columns):
        metrics["y1_hat_true_corr"] = _safe_corr(results_df["true_y1_prob"], results_df["y1_hat"])
        metrics["y1_hat_true_rmse"] = _finite_or_none(
            np.sqrt(mean_squared_error(results_df["true_y1_prob"], results_df["y1_hat"]))
        )
    return metrics


def run_tarnet_stage_only(
    config: TarNetStageOnlyOracleConfig,
    output_dir: Path,
    device: torch.device,
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

    df = _prepare_dataset(config, parquet_file)
    nuisance_df, resolved_nuisance_source = _load_nuisance_predictions(
        Path(config.nuisance_source),
        config.nuisance_config_hash,
    )
    nuisance_df = _validate_nuisance_predictions(nuisance_df)
    applied_config = _make_applied_config(config, parquet_file)

    config_hash = config.config_hash()
    prediction_path = (
        output_dir
        / "tarnet_stage_only_predictions"
        / config_hash
        / "predictions.parquet"
    )
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=applied_config,
        output_path=prediction_path,
        device=device,
        num_workers=num_workers,
        proposal_agent=_UnusedAgent(),
        extraction_provider=_UnusedExtractionProvider(),
    )

    selected_outer_folds = set(config.outer_folds or [])
    analysis_splits = runner._analysis_splits()
    folds_to_run = [
        outer_fold
        for outer_fold, _train_idx, _test_idx in analysis_splits
        if not selected_outer_folds or outer_fold in selected_outer_folds
    ]
    available_outer_folds = set(nuisance_df["outer_fold"].astype(int).unique().tolist())
    missing_outer_folds = [
        outer_fold
        for outer_fold in folds_to_run
        if outer_fold not in available_outer_folds
    ]
    if missing_outer_folds:
        raise ValueError(
            "Nuisance source is missing requested outer fold(s) "
            f"{missing_outer_folds}. Available outer folds: "
            f"{sorted(available_outer_folds)}. Use --outer-folds to run only "
            "completed folds, or finish the nuisance run first."
        )

    prediction_frames: List[pd.DataFrame] = []
    for outer_fold, train_idx, _test_idx in analysis_splits:
        if selected_outer_folds and outer_fold not in selected_outer_folds:
            continue
        discovery_df = runner.dataset.iloc[train_idx].reset_index(drop=True)
        nuisance_predictions = _nuisance_for_outer_fold(
            nuisance_df=nuisance_df,
            discovery_df=discovery_df,
            outer_fold=outer_fold,
            applied_config=applied_config,
        )
        runner.nuisance_rows.append(nuisance_predictions)
        tarnet_stage = runner._crossfit_tarnet_offset(
            discovery_df,
            nuisance_predictions,
            outer_fold,
        )
        predictions = runner._neural_only_prediction_frame(
            discovery_df=discovery_df,
            r_stage_predictions=tarnet_stage["predictions"],
            outer_fold=outer_fold,
        )
        fold_metrics = runner._neural_only_metrics(predictions)
        fold_metrics["mode"] = "tarnet_stage_only"
        runner.metric_rows.append({"outer_fold": outer_fold, **fold_metrics})
        prediction_frames.append(predictions)

    if not prediction_frames:
        raise ValueError(
            "No outer folds were run. Check --outer-folds and --n-folds."
        )

    results_df = (
        pd.concat(prediction_frames)
        .sort_values(["_oci_row_id", "outer_fold"])
        .reset_index(drop=True)
    )
    runner._save_predictions(results_df)
    runner._save_artifacts(results_df)

    artifact_dir = prediction_path.parent / "agentic_attention_variable_forest"
    metrics = _aggregate_neural_metrics(results_df)
    return {
        "config": asdict(config),
        "metrics": metrics,
        "n_samples": len(results_df),
        "skipped": False,
        "error": None,
        "artifacts": {
            "predictions_path": str(prediction_path),
            "artifact_dir": str(artifact_dir),
            "nuisance_source": str(resolved_nuisance_source),
        },
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
        "nuisance_source",
        "nuisance_config_hash",
        "outer_folds",
        "repeat_index",
        "n_folds",
        "effect_folds",
        "non_nuisance_epochs",
        "batch_size",
        "effect_batch_size",
        "tarnet_offset_batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "fold_parallelism",
        "r_stage_min_propensity",
        "r_stage_max_propensity",
        "interaction_l2_weight",
        "tarnet_offset_heterogeneity_weight",
        "tarnet_offset_min_logit_std",
        "htr_sentence_model",
        "htr_freeze_sentence_encoder",
        "htr_sentence_encoder_backend",
        "htr_sentence_pooling",
        "htr_trainable_sentence_encoder_layers",
        "e_clip",
    ]:
        row[key] = config.get(key)
    for key, value in result.get("metrics", {}).items():
        row[key] = value
    artifacts = result.get("artifacts", {})
    row["predictions_path"] = artifacts.get("predictions_path")
    row["artifact_dir"] = artifacts.get("artifact_dir")
    row["resolved_nuisance_source"] = artifacts.get("nuisance_source")
    return row


def _write_aggregate_outputs(output_dir: Path) -> None:
    results_dir = output_dir / "results"
    if not results_dir.exists():
        return
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            rows.append(_result_row(path.stem, json.load(f)))
    if not rows:
        return
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    results_df.to_parquet(output_dir / "all_results.parquet", index=False)
    with open(output_dir / "all_results.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--nuisance-source",
        required=True,
        help=(
            "Saved nuisance parquet, artifact directory, nuisance checkpoint "
            "directory, or full oracle output directory."
        ),
    )
    parser.add_argument(
        "--nuisance-config-hash",
        default=None,
        help="Hash under <run-dir>/agentic_attention_predictions to reuse.",
    )
    parser.add_argument(
        "--outer-folds",
        type=_parse_outer_folds,
        default=None,
        help="Comma-separated outer folds to run, or 'all' for every fold.",
    )
    parser.add_argument(
        "--output-dir",
        default="../pcori_experiments/oracle_agentic_attention_tarnet_stage_only",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--effect-folds", type=int, default=5)
    parser.add_argument(
        "--fold-parallelism",
        "--inner-fold-parallelism",
        dest="fold_parallelism",
        default="auto",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--text-max-chars", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-index", type=int, default=0)

    parser.add_argument("--htr-sentence-model", default="prajjwal1/bert-tiny")
    parser.add_argument("--htr-freeze-sentence-encoder", type=_parse_bool, default=False)
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
    parser.add_argument(
        "--htr-normalize-sentence-embeddings",
        type=_parse_bool,
        default=True,
    )
    parser.add_argument("--htr-trainable-sentence-encoder-layers", type=int, default=0)
    parser.add_argument("--htr-dropout", type=float, default=0.1)

    parser.add_argument(
        "--non-nuisance-epochs",
        dest="non_nuisance_epochs",
        type=int,
        default=20,
        help="Epochs for the TarNet-offset neural stage.",
    )
    parser.add_argument(
        "--epochs",
        dest="non_nuisance_epochs",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--effect-batch-size", type=int, default=32)
    parser.add_argument("--tarnet-offset-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--alpha-propensity", type=float, default=1.0)

    parser.add_argument("--nuisance-folds", type=int, default=5)
    parser.add_argument("--nuisance-epochs", type=int, default=20)
    parser.add_argument("--nuisance-weight-decay", type=float, default=0.05)
    parser.add_argument("--nuisance-label-smoothing", type=float, default=0.02)
    parser.add_argument(
        "--nuisance-calibration",
        choices=["none", "temperature", "isotonic", "temperature_isotonic"],
        default="temperature_isotonic",
    )
    parser.add_argument("--attention-top-k-chunks", type=int, default=5)
    parser.add_argument("--e-clip", type=float, default=0.01)
    parser.add_argument("--r-stage-min-propensity", type=float, default=0.0)
    parser.add_argument("--r-stage-max-propensity", type=float, default=1.0)
    parser.add_argument("--interaction-l2-weight", type=float, default=0.0)
    parser.add_argument("--tarnet-offset-heterogeneity-weight", type=float, default=0.1)
    parser.add_argument("--tarnet-offset-min-logit-std", type=float, default=0.5)
    return parser


def _config_from_args(args: argparse.Namespace) -> TarNetStageOnlyOracleConfig:
    dataset_path = str(args.dataset)
    return TarNetStageOnlyOracleConfig(
        dataset_path=dataset_path,
        dataset_name=Path(dataset_path).name,
        nuisance_source=str(args.nuisance_source),
        nuisance_config_hash=args.nuisance_config_hash,
        outer_folds=args.outer_folds,
        n_folds=args.n_folds,
        seed=args.seed,
        repeat_index=args.repeat_index,
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
        htr_trainable_sentence_encoder_layers=args.htr_trainable_sentence_encoder_layers,
        htr_dropout=args.htr_dropout,
        non_nuisance_epochs=args.non_nuisance_epochs,
        batch_size=args.batch_size,
        effect_batch_size=args.effect_batch_size,
        tarnet_offset_batch_size=args.tarnet_offset_batch_size,
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
        attention_top_k_chunks=args.attention_top_k_chunks,
        e_clip=args.e_clip,
        r_stage_min_propensity=args.r_stage_min_propensity,
        r_stage_max_propensity=args.r_stage_max_propensity,
        interaction_l2_weight=args.interaction_l2_weight,
        tarnet_offset_heterogeneity_weight=args.tarnet_offset_heterogeneity_weight,
        tarnet_offset_min_logit_std=args.tarnet_offset_min_logit_std,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.n_folds < 2:
        parser.error("--n-folds must be >= 2")
    if args.effect_folds < 2:
        parser.error("--effect-folds must be >= 2")
    if args.nuisance_folds < 2:
        parser.error("--nuisance-folds must be >= 2")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be >= 1")
    if args.text_max_chars is not None and args.text_max_chars < 1:
        parser.error("--text-max-chars must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.effect_batch_size < 1:
        parser.error("--effect-batch-size must be >= 1")
    if args.tarnet_offset_batch_size < 1:
        parser.error("--tarnet-offset-batch-size must be >= 1")
    if args.non_nuisance_epochs < 1:
        parser.error("--non-nuisance-epochs must be >= 1")
    if args.nuisance_epochs < 1:
        parser.error("--nuisance-epochs must be >= 1")
    if args.nuisance_weight_decay < 0:
        parser.error("--nuisance-weight-decay must be >= 0")
    if not 0.0 <= args.nuisance_label_smoothing < 1.0:
        parser.error("--nuisance-label-smoothing must be in [0, 1)")
    if args.htr_sentence_encoder_batch_size < 1:
        parser.error("--htr-sentence-encoder-batch-size must be >= 1")
    if args.htr_trainable_sentence_encoder_layers < 0:
        parser.error("--htr-trainable-sentence-encoder-layers must be >= 0")
    if not 0.0 < args.e_clip < 0.5:
        parser.error("--e-clip must be in (0, 0.5)")
    if not 0.0 <= args.r_stage_min_propensity < args.r_stage_max_propensity <= 1.0:
        parser.error(
            "--r-stage-min-propensity and --r-stage-max-propensity must satisfy "
            "0 <= min < max <= 1"
        )
    if args.interaction_l2_weight < 0:
        parser.error("--interaction-l2-weight must be >= 0")
    if args.tarnet_offset_heterogeneity_weight < 0:
        parser.error("--tarnet-offset-heterogeneity-weight must be >= 0")
    if args.tarnet_offset_min_logit_std < 0:
        parser.error("--tarnet-offset-min-logit-std must be >= 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command_line.txt").write_text(" ".join(sys.argv) + "\n")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    config = _config_from_args(args)
    config_hash = config.config_hash()
    result_path = output_dir / "results" / f"{config_hash}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    print("Agentic attention TarNet-stage-only experiment")
    print(f"Dataset: {config.dataset_name}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Hash: {config_hash}")

    if args.resume and result_path.exists():
        logger.info("Result exists and --resume was set; skipping %s", config_hash)
        _write_aggregate_outputs(output_dir)
        return

    try:
        result = run_tarnet_stage_only(
            config=config,
            output_dir=output_dir,
            device=device,
            num_workers=args.num_workers,
        )
    except Exception as exc:
        logger.error(
            "TarNet-stage-only experiment %s failed: %s\n%s",
            config_hash,
            exc,
            traceback.format_exc(),
        )
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
    logger.info("Done. Results written under %s", output_dir)


if __name__ == "__main__":
    main()
