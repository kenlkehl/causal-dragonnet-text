#!/usr/bin/env python
"""R-stage-only oracle runner for agentic attention experiments.

This script is for quick iteration on neural R-learner hyperparameters when
cross-fitted nuisance predictions already exist. It skips nuisance training,
agentic variable proposal, feature extraction, and the final forest. It loads a
saved ``nuisance_oof_predictions.parquet`` file, retrains only the text-based
R-stage tau model, and writes the usual neural diagnostics and attention
artifacts. Optionally, it can also train residual-score tail-vs-neutral
contrastive text classifiers from the same saved nuisance residuals.

Typical use, starting from a neural-only run:

    python oracle_experiment_scripts/run_oracle_agentic_attention_r_stage_only.py \
        --dataset synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured \
        --nuisance-predictions-path ../pcori_experiments/oracle_agentic_attention_variable_forest_6-16-26_neural_only/agentic_attention_predictions/81733620a4cc/agentic_attention_variable_forest/nuisance_oof_predictions.parquet \
        --output-dir ../pcori_experiments/oracle_agentic_attention_r_stage_only_128 \
        --effect-batch-size 128

Outputs:

    <output-dir>/results/<config-hash>.json
    <output-dir>/all_results.csv
    <output-dir>/all_results.parquet
    <output-dir>/r_stage_only_predictions/<config-hash>/predictions.parquet
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
from typing import Any, Dict, List, Optional

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
class RStageOnlyOracleConfig:
    """Configuration for one R-stage-only oracle experiment."""

    dataset_path: str
    dataset_name: str
    nuisance_predictions_path: str

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
    htr_sentence_encoder_backend: str = "transformers"
    htr_sentence_pooling: str = "token_attention"
    htr_normalize_sentence_embeddings: bool = True
    htr_trainable_sentence_encoder_layers: int = 0
    htr_dropout: float = 0.1

    epochs: int = 50
    batch_size: int = 8
    effect_batch_size: int = 128
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0

    effect_folds: int = 5
    fold_parallelism: str = "auto"
    attention_top_k_chunks: int = 5
    e_clip: float = 0.01
    r_stage_min_propensity: float = 0.0
    r_stage_max_propensity: float = 1.0
    effect_objective: str = "squared_r_loss"
    residual_contrastive_enabled: bool = False
    residual_contrastive_score: str = "r_score"
    residual_contrastive_high_quantile: float = 0.80
    residual_contrastive_low_quantile: float = 0.20
    residual_contrastive_neutral_abs_quantile: float = 0.40
    residual_contrastive_min_class_count: int = 10

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


class _UnusedAgent:
    def propose(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("Proposal agent is not available in R-stage-only mode.")


class _UnusedExtractionProvider:
    def ensure_features(self, df: pd.DataFrame, specs: List[Any]) -> pd.DataFrame:
        raise RuntimeError("Feature extraction is not available in R-stage-only mode.")


def _parse_bool(value: str) -> bool:
    lowered = str(value).lower().strip()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


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


def _prepare_dataset(config: RStageOnlyOracleConfig, parquet_file: Path) -> pd.DataFrame:
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
    config: RStageOnlyOracleConfig,
    parquet_file: Path,
) -> AppliedInferenceConfig:
    return AppliedInferenceConfig(
        clinical_question=(
            "Estimate text-derived heterogeneous treatment effects from saved "
            "nuisance predictions."
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
                nuisance_folds=2,
                effect_folds=config.effect_folds,
                fold_parallelism=config.fold_parallelism,
                attention_top_k_chunks=config.attention_top_k_chunks,
                e_clip=config.e_clip,
                r_stage_min_propensity=config.r_stage_min_propensity,
                r_stage_max_propensity=config.r_stage_max_propensity,
                effect_objective=config.effect_objective,
                residual_contrastive_enabled=config.residual_contrastive_enabled,
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
                neural_only=True,
            ),
        ),
        training=TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            effect_batch_size=config.effect_batch_size,
            learning_rate=config.learning_rate,
            lr_schedule="linear",
            weight_decay=config.weight_decay,
            gradient_clip_norm=config.gradient_clip_norm,
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=False, features=[]),
    )


def _load_nuisance_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Nuisance predictions not found: {path}")
    nuisance_df = pd.read_parquet(path)
    missing = sorted(NUISANCE_REQUIRED_COLUMNS - set(nuisance_df.columns))
    if missing:
        raise ValueError(
            "Nuisance predictions are missing required columns: "
            + ", ".join(missing)
        )
    return nuisance_df.copy()


def _nuisance_for_outer_fold(
    nuisance_df: pd.DataFrame,
    discovery_df: pd.DataFrame,
    outer_fold: int,
    config: AppliedInferenceConfig,
) -> pd.DataFrame:
    fold_df = nuisance_df[nuisance_df["outer_fold"].astype(int) == int(outer_fold)]
    if fold_df.empty:
        raise ValueError(f"No nuisance predictions found for outer_fold={outer_fold}")
    if fold_df["_oci_row_id"].duplicated().any():
        raise ValueError(
            f"Nuisance predictions have duplicate _oci_row_id values for "
            f"outer_fold={outer_fold}"
        )

    expected_ids = discovery_df["_oci_row_id"].to_numpy()
    by_id = fold_df.set_index("_oci_row_id", drop=False)
    missing_ids = sorted(set(expected_ids) - set(by_id.index))
    if missing_ids:
        preview = ", ".join(str(value) for value in missing_ids[:10])
        raise ValueError(
            f"Nuisance predictions for outer_fold={outer_fold} are missing "
            f"{len(missing_ids)} discovery row id(s): {preview}"
        )

    predictions = by_id.loc[expected_ids].reset_index(drop=True).copy()
    y = discovery_df[config.outcome_column].to_numpy(dtype=float)
    t = discovery_df[config.treatment_column].to_numpy(dtype=float)
    e_hat = predictions["e_hat"].to_numpy(dtype=float)
    m_hat = predictions["m_hat"].to_numpy(dtype=float)

    predictions["outer_fold"] = int(outer_fold)
    if "y_residual" not in predictions.columns:
        predictions["y_residual"] = y - m_hat
    if "t_residual" not in predictions.columns:
        predictions["t_residual"] = t - e_hat
    if "r_loss_at_zero_tau" not in predictions.columns:
        predictions["r_loss_at_zero_tau"] = predictions["y_residual"] ** 2
    if "nuisance_fold" not in predictions.columns:
        predictions["nuisance_fold"] = np.nan

    required_numeric = ["e_hat", "m_hat", "y_residual", "t_residual"]
    for column in required_numeric:
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    if predictions[required_numeric].isna().any().any():
        raise ValueError(
            f"Nuisance predictions for outer_fold={outer_fold} contain NaNs "
            "after numeric coercion."
        )
    return predictions


def _aggregate_neural_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    required = {"tau_hat_r_stage", "r_loss", "r_loss_at_zero_tau", "e_hat", "m_hat"}
    if not required.issubset(results_df.columns):
        return {}

    metrics: Dict[str, Any] = {
        "mode": "r_stage_only",
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
    zero_loss = metrics.get("neural_r_loss_at_zero_tau_mean")
    r_loss = metrics.get("neural_r_loss_mean")
    if zero_loss is not None and zero_loss > 0 and r_loss is not None:
        # >0 means tau_hat reduces residual R-loss vs tau=0, ~0 means no gain,
        # and <0 means the learned tau worsens the R-loss. This is useful but
        # not sufficient evidence of oracle CATE recovery.
        metrics["neural_r_loss_relative_improvement"] = float(1.0 - r_loss / zero_loss)
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
    if "r_stage_train_eligible" in results_df.columns:
        eligible = results_df["r_stage_train_eligible"].astype(bool)
        metrics["neural_r_stage_train_eligible_rows"] = int(eligible.sum())
        metrics["neural_r_stage_train_eligible_fraction"] = _finite_or_none(
            eligible.mean()
        )
    if "treatment_indicator" in results_df.columns:
        treatment = results_df["treatment_indicator"].to_numpy()
        e_hat = results_df["e_hat"].to_numpy()
        metrics["neural_propensity_auroc"] = _safe_roc_auc(treatment, e_hat)
        metrics.update(binary_calibration_metrics(treatment, e_hat, prefix="neural_propensity"))
        if "e_hat_raw" in results_df.columns:
            e_raw = results_df["e_hat_raw"].to_numpy()
            metrics["neural_propensity_raw_auroc"] = _safe_roc_auc(treatment, e_raw)
            metrics.update(
                binary_calibration_metrics(treatment, e_raw, prefix="neural_propensity_raw")
            )
    if "outcome_indicator" in results_df.columns:
        outcome = results_df["outcome_indicator"].to_numpy()
        m_hat = results_df["m_hat"].to_numpy()
        metrics["neural_outcome_auroc"] = _safe_roc_auc(outcome, m_hat)
        metrics.update(binary_calibration_metrics(outcome, m_hat, prefix="neural_outcome"))
        if "m_hat_raw" in results_df.columns:
            m_raw = results_df["m_hat_raw"].to_numpy()
            metrics["neural_outcome_raw_auroc"] = _safe_roc_auc(outcome, m_raw)
            metrics.update(binary_calibration_metrics(outcome, m_raw, prefix="neural_outcome_raw"))
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


def run_r_stage_only(
    config: RStageOnlyOracleConfig,
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
    nuisance_df = _load_nuisance_predictions(Path(config.nuisance_predictions_path))
    applied_config = _make_applied_config(config, parquet_file)

    config_hash = config.config_hash()
    prediction_path = (
        output_dir
        / "r_stage_only_predictions"
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

    prediction_frames: List[pd.DataFrame] = []
    for outer_fold, train_idx, _test_idx in runner._analysis_splits():
        discovery_df = runner.dataset.iloc[train_idx].reset_index(drop=True)
        nuisance_predictions = _nuisance_for_outer_fold(
            nuisance_df=nuisance_df,
            discovery_df=discovery_df,
            outer_fold=outer_fold,
            config=applied_config,
        )
        runner.nuisance_rows.append(nuisance_predictions)
        residual_contrastive = None
        if config.residual_contrastive_enabled:
            residual_contrastive = runner._crossfit_residual_contrastive(
                discovery_df,
                nuisance_predictions,
                outer_fold,
            )
        r_stage = runner._crossfit_effect(
            discovery_df,
            nuisance_predictions,
            outer_fold,
        )
        predictions = runner._neural_only_prediction_frame(
            discovery_df=discovery_df,
            r_stage_predictions=r_stage["predictions"],
            outer_fold=outer_fold,
        )
        if residual_contrastive is not None:
            predictions = runner._merge_residual_contrastive_predictions(
                predictions,
                residual_contrastive["predictions"],
            )
        fold_metrics = runner._neural_only_metrics(predictions)
        fold_metrics["mode"] = "r_stage_only"
        if residual_contrastive is not None:
            fold_metrics.update(residual_contrastive["metrics"])
        runner.metric_rows.append({"outer_fold": outer_fold, **fold_metrics})
        prediction_frames.append(predictions)

    results_df = (
        pd.concat(prediction_frames)
        .sort_values(["_oci_row_id", "outer_fold"])
        .reset_index(drop=True)
    )
    runner._save_predictions(results_df)
    runner._save_artifacts(results_df)

    artifact_dir = prediction_path.parent / "agentic_attention_variable_forest"
    metrics = _aggregate_neural_metrics(results_df)
    if config.residual_contrastive_enabled:
        metrics.update(runner._residual_contrastive_metrics(results_df))
    return {
        "config": asdict(config),
        "metrics": metrics,
        "n_samples": len(results_df),
        "skipped": False,
        "error": None,
        "artifacts": {
            "predictions_path": str(prediction_path),
            "artifact_dir": str(artifact_dir),
            "nuisance_predictions_path": config.nuisance_predictions_path,
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
        "nuisance_predictions_path",
        "repeat_index",
        "n_folds",
        "effect_folds",
        "epochs",
        "batch_size",
        "effect_batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "fold_parallelism",
        "r_stage_min_propensity",
        "r_stage_max_propensity",
        "effect_objective",
        "residual_contrastive_enabled",
        "residual_contrastive_score",
        "residual_contrastive_high_quantile",
        "residual_contrastive_low_quantile",
        "residual_contrastive_neutral_abs_quantile",
        "residual_contrastive_min_class_count",
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
    parser.add_argument("--nuisance-predictions-path", required=True)
    parser.add_argument(
        "--output-dir",
        default="../pcori_experiments/oracle_agentic_attention_r_stage_only",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--effect-folds", type=int, default=5)
    parser.add_argument("--fold-parallelism", "--inner-fold-parallelism", dest="fold_parallelism", default="auto")
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
        default="transformers",
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

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--effect-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--attention-top-k-chunks", type=int, default=5)
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
        "--residual-contrastive-enabled",
        action="store_true",
        help=(
            "Train residual-score high-vs-neutral and low-vs-neutral text "
            "classifiers from the saved nuisance residuals."
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
    return parser


def _config_from_args(args: argparse.Namespace) -> RStageOnlyOracleConfig:
    dataset_path = str(args.dataset)
    return RStageOnlyOracleConfig(
        dataset_path=dataset_path,
        dataset_name=Path(dataset_path).name,
        nuisance_predictions_path=str(args.nuisance_predictions_path),
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        effect_batch_size=args.effect_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        effect_folds=args.effect_folds,
        fold_parallelism=args.fold_parallelism,
        attention_top_k_chunks=args.attention_top_k_chunks,
        e_clip=args.e_clip,
        r_stage_min_propensity=args.r_stage_min_propensity,
        r_stage_max_propensity=args.r_stage_max_propensity,
        effect_objective=args.effect_objective,
        residual_contrastive_enabled=args.residual_contrastive_enabled,
        residual_contrastive_score=args.residual_contrastive_score,
        residual_contrastive_high_quantile=args.residual_contrastive_high_quantile,
        residual_contrastive_low_quantile=args.residual_contrastive_low_quantile,
        residual_contrastive_neutral_abs_quantile=(
            args.residual_contrastive_neutral_abs_quantile
        ),
        residual_contrastive_min_class_count=(
            args.residual_contrastive_min_class_count
        ),
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.n_folds < 2:
        parser.error("--n-folds must be >= 2")
    if args.effect_folds < 2:
        parser.error("--effect-folds must be >= 2")
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
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
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
    if not (
        0.0
        < args.residual_contrastive_low_quantile
        < args.residual_contrastive_high_quantile
        < 1.0
    ):
        parser.error(
            "--residual-contrastive-low-quantile and "
            "--residual-contrastive-high-quantile must satisfy 0 < low < high < 1"
        )
    if not 0.0 < args.residual_contrastive_neutral_abs_quantile < 1.0:
        parser.error("--residual-contrastive-neutral-abs-quantile must be in (0, 1)")
    if args.residual_contrastive_min_class_count < 1:
        parser.error("--residual-contrastive-min-class-count must be >= 1")

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

    print("Agentic attention R-stage-only experiment")
    print(f"Dataset: {config.dataset_name}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Hash: {config_hash}")

    if args.resume and result_path.exists():
        logger.info("Result exists and --resume was set; skipping %s", config_hash)
        _write_aggregate_outputs(output_dir)
        return

    try:
        result = run_r_stage_only(
            config=config,
            output_dir=output_dir,
            device=device,
            num_workers=args.num_workers,
        )
    except Exception as exc:
        logger.error(
            "R-stage-only experiment %s failed: %s\n%s",
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
