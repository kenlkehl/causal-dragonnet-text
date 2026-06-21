#!/usr/bin/env python
"""CATE-stage-only oracle runner for the neural causal-forest text extractor.

This script skips cross-fitted nuisance training and retrains only the neural
causal-forest CATE stage from saved fold-level nuisance predictions.  It expects
each nuisance/full-oracle fold directory to contain:

    repeat_XX/fold_YY/train_nuisance_predictions.parquet

The full NCF oracle runner writes that artifact for new runs. Older runs that
only saved nuisance attention need their nuisance folds rerun or exported first.

Typical use:

    python oracle_experiment_scripts/run_neural_causal_forest_cate_stage_only.py \
        --dataset synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured \
        --nuisance-run-dir ../ncf_oracle_runs/one_confounder_one_modifier \
        --output-dir ../ncf_oracle_runs/one_confounder_one_modifier_cate_only \
        --device cuda:2 --n-folds 5 --repeats 1 \
        --config example_configs/neural_causal_forest_config.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import multiprocessing as mp
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

SCRIPT_PATH = Path(__file__).resolve()
for candidate in (SCRIPT_PATH.parents[1], SCRIPT_PATH.parents[2]):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from oci.models.neural_causal_forest_extractor import (  # noqa: E402
    NeuralCausalForestConfig,
    NeuralCausalForestModel,
    add_oracle_attention_hits,
    build_agent_context_rows,
    causal_forest_attention_evidence,
    predict_neural_causal_forest,
    read_dataframe,
    save_neural_causal_forest_model,
    summarize_pipeline_metrics,
    train_neural_causal_forest,
    write_dataframe,
)

logger = logging.getLogger(__name__)

TRUE_TAU_CANDIDATES = ["true_ite_prob", "true_ite", "true_tau", "tau"]


def _bool_arg(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="CSV/parquet file or dataset directory")
    parser.add_argument("--nuisance-run-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--row-id-column", default="_ncf_row_id")
    parser.add_argument("--outcome-type", choices=["binary", "continuous"], default="binary")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--text-max-chars", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Optional device list for outer-fold parallelism.",
    )
    parser.add_argument(
        "--fold-parallelism",
        "--outer-fold-parallelism",
        dest="fold_parallelism",
        default="auto",
        help=(
            "Number of outer folds to run concurrently. 'auto' uses one worker "
            "per listed device when --devices has multiple entries, stays serial "
            "on a single CUDA device, and uses --num-workers on CPU."
        ),
    )
    parser.add_argument("--config", default=None, help="Optional base NeuralCausalForestConfig JSON")
    parser.add_argument(
        "--reuse-fold-config",
        action="store_true",
        help=(
            "Use each nuisance run fold's resolved_config.json as the base config "
            "when --config is not provided. CLI overrides still apply."
        ),
    )
    parser.add_argument(
        "--nuisance-predictions-name",
        default="train_nuisance_predictions.parquet",
        help="File name under repeat_XX/fold_YY in --nuisance-run-dir.",
    )
    parser.add_argument(
        "--nuisance-predictions-template",
        default=None,
        help=(
            "Optional explicit path template with {repeat}, {fold}, {repeat02}, "
            "and {fold02} placeholders. Overrides --nuisance-run-dir lookup."
        ),
    )

    # Common NeuralCausalForestConfig overrides.
    parser.add_argument(
        "--encoder-architecture",
        choices=["hierarchical_transformer", "htr", "ncf_token_attention", "ncf"],
        default=None,
    )
    parser.add_argument("--encoder-model", default=None)
    parser.add_argument("--encoder-backend", choices=["transformers", "hash"], default=None)
    parser.add_argument("--freeze-encoder", type=_bool_arg, default=None)
    parser.add_argument("--trainable-encoder-layers", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--chunk-size-words", type=int, default=None)
    parser.add_argument("--chunk-overlap-words", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--representation-dim", type=int, default=None)
    parser.add_argument("--token-attention-dim", type=int, default=None)
    parser.add_argument("--chunk-attention-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--normalize-representations", type=_bool_arg, default=None)
    parser.add_argument("--htr-num-layers", type=int, default=None)
    parser.add_argument("--htr-num-heads", type=int, default=None)
    parser.add_argument("--htr-transformer-dim", type=int, default=None)
    parser.add_argument("--htr-sentence-encoder-batch-size", type=int, default=None)
    parser.add_argument(
        "--htr-sentence-encoder-backend",
        choices=["auto", "sentence_transformers", "transformers"],
        default=None,
    )
    parser.add_argument(
        "--htr-sentence-pooling",
        choices=["auto", "cls", "last", "mean", "token_attention"],
        default=None,
    )
    parser.add_argument("--htr-normalize-sentence-embeddings", type=_bool_arg, default=None)
    parser.add_argument("--htr-hash-embedding-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--effect-batch-size", type=int, default=None)
    parser.add_argument(
        "--inner-fold-parallelism",
        default=None,
        help=(
            "Accepted for config compatibility with full NCF training. This "
            "CATE-only runner does not train nuisance inner folds."
        ),
    )
    parser.add_argument("--n-trees", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--forest-epochs", type=int, default=None)
    parser.add_argument("--forest-learning-rate", type=float, default=None)
    parser.add_argument("--forest-weight-decay", type=float, default=None)
    parser.add_argument("--lambda-heterogeneity", type=float, default=None)
    parser.add_argument("--lambda-leaf-balance", type=float, default=None)
    parser.add_argument("--lambda-leaf-min-mass", type=float, default=None)
    parser.add_argument("--lambda-leaf-tau-l2", type=float, default=None)
    parser.add_argument("--feature-subsample-fraction", type=float, default=None)
    parser.add_argument("--temperature-start", type=float, default=None)
    parser.add_argument("--temperature-end", type=float, default=None)
    parser.add_argument("--honesty-fraction", type=float, default=None)
    parser.add_argument("--refit-leaf-values", type=_bool_arg, default=None)
    parser.add_argument("--leaf-ridge", type=float, default=None)
    parser.add_argument("--leaf-min-mass", type=float, default=None)
    parser.add_argument("--tau-clip", type=float, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--attention-top-k", type=int, default=None)
    parser.add_argument("--evidence-batch-size", type=int, default=None)
    parser.add_argument(
        "--effect-attribution-target",
        choices=["tau_heterogeneity", "tau_abs", "tau_signed"],
        default="tau_heterogeneity",
    )
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--no-attention", action="store_true", help="Skip effect attention artifacts")
    parser.add_argument(
        "--add-oracle-hits",
        action="store_true",
        help="Annotate attention evidence artifacts with synthetic oracle regex hits for debugging.",
    )
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.nuisance_predictions_template is None and args.nuisance_run_dir is None:
        parser.error("Provide --nuisance-run-dir or --nuisance-predictions-template")
    return args


def _fold_dir(root: str | Path, repeat_index: int, fold_index: int) -> Path:
    return Path(root) / f"repeat_{repeat_index:02d}" / f"fold_{fold_index:02d}"


def _nuisance_predictions_path(args: argparse.Namespace, repeat_index: int, fold_index: int) -> Path:
    if args.nuisance_predictions_template:
        return Path(
            str(args.nuisance_predictions_template).format(
                repeat=repeat_index,
                fold=fold_index,
                repeat_index=repeat_index,
                fold_index=fold_index,
                repeat02=f"{repeat_index:02d}",
                fold02=f"{fold_index:02d}",
            )
        )
    return _fold_dir(args.nuisance_run_dir, repeat_index, fold_index) / args.nuisance_predictions_name


def _base_config_path(args: argparse.Namespace, repeat_index: int, fold_index: int) -> Optional[Path]:
    if args.config:
        return Path(args.config)
    if args.reuse_fold_config and args.nuisance_run_dir:
        candidate = _fold_dir(args.nuisance_run_dir, repeat_index, fold_index) / "resolved_config.json"
        if candidate.exists():
            return candidate
    return None


def _make_config(args: argparse.Namespace, repeat_index: int, fold_index: int) -> NeuralCausalForestConfig:
    base_path = _base_config_path(args, repeat_index, fold_index)
    config = NeuralCausalForestConfig.from_json(base_path) if base_path else NeuralCausalForestConfig()
    override_map = {
        "encoder_architecture": args.encoder_architecture,
        "encoder_model_name": args.encoder_model,
        "encoder_backend": args.encoder_backend,
        "freeze_encoder": args.freeze_encoder,
        "trainable_encoder_layers": args.trainable_encoder_layers,
        "max_length": args.max_length,
        "chunk_size_words": args.chunk_size_words,
        "chunk_overlap_words": args.chunk_overlap_words,
        "max_chunks": args.max_chunks,
        "representation_dim": args.representation_dim,
        "token_attention_dim": args.token_attention_dim,
        "chunk_attention_dim": args.chunk_attention_dim,
        "dropout": args.dropout,
        "normalize_representations": args.normalize_representations,
        "htr_num_layers": args.htr_num_layers,
        "htr_num_heads": args.htr_num_heads,
        "htr_transformer_dim": args.htr_transformer_dim,
        "htr_sentence_encoder_batch_size": args.htr_sentence_encoder_batch_size,
        "htr_sentence_encoder_backend": args.htr_sentence_encoder_backend,
        "htr_sentence_pooling": args.htr_sentence_pooling,
        "htr_normalize_sentence_embeddings": args.htr_normalize_sentence_embeddings,
        "htr_hash_embedding_dim": args.htr_hash_embedding_dim,
        "batch_size": args.batch_size,
        "effect_batch_size": args.effect_batch_size,
        "inner_fold_parallelism": args.inner_fold_parallelism,
        "n_trees": args.n_trees,
        "depth": args.depth,
        "forest_epochs": args.forest_epochs,
        "forest_learning_rate": args.forest_learning_rate,
        "forest_weight_decay": args.forest_weight_decay,
        "lambda_heterogeneity": args.lambda_heterogeneity,
        "lambda_leaf_balance": args.lambda_leaf_balance,
        "lambda_leaf_min_mass": args.lambda_leaf_min_mass,
        "lambda_leaf_tau_l2": args.lambda_leaf_tau_l2,
        "feature_subsample_fraction": args.feature_subsample_fraction,
        "temperature_start": args.temperature_start,
        "temperature_end": args.temperature_end,
        "honesty_fraction": args.honesty_fraction,
        "refit_leaf_values_after_training": args.refit_leaf_values,
        "leaf_ridge": args.leaf_ridge,
        "leaf_min_mass": args.leaf_min_mass,
        "tau_clip": args.tau_clip,
        "gradient_clip_norm": args.gradient_clip_norm,
        "attention_top_k": args.attention_top_k,
        "evidence_batch_size": args.evidence_batch_size,
        "num_workers": args.num_workers,
    }
    for key, value in override_map.items():
        if value is not None:
            setattr(config, key, value)
    config.seed = int(args.seed) + 1000 * int(repeat_index) + int(fold_index)
    config.__post_init__()
    return config


def _prepare_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    df = read_dataframe(args.dataset).reset_index(drop=True).copy()
    if args.sample_size is not None and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
    if args.text_max_chars is not None:
        df[args.text_column] = df[args.text_column].astype(str).str.slice(0, int(args.text_max_chars))
    if args.row_id_column not in df.columns:
        df[args.row_id_column] = np.arange(len(df), dtype=int)
    required = {args.text_column, args.treatment_column, args.outcome_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def _outer_splits(
    df: pd.DataFrame,
    args: argparse.Namespace,
    repeat_index: int,
) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    n_splits = max(2, min(int(args.n_folds), len(df)))
    y = df[args.treatment_column].astype(int).to_numpy()
    counts = pd.Series(y).value_counts()
    if len(counts) == 2 and int(counts.min()) >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed + repeat_index)
        split_iter = splitter.split(df, y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed + repeat_index)
        split_iter = splitter.split(df)
    for fold_index, (train_idx, test_idx) in enumerate(split_iter, start=1):
        yield fold_index, np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def _resolve_devices(args: argparse.Namespace) -> List[str]:
    devices = list(args.devices) if args.devices else [str(args.device)]
    if not devices:
        devices = ["cuda:0" if torch.cuda.is_available() else "cpu"]
    return [str(device) for device in devices]


def _resolve_fold_parallelism(args: argparse.Namespace, devices: List[str]) -> int:
    setting = str(args.fold_parallelism).strip().lower()
    if setting == "auto":
        if len(devices) > 1:
            return len(devices)
        device_name = devices[0] if devices else str(args.device)
        if str(device_name).startswith("cuda"):
            return 1
        return max(1, int(args.num_workers or 1))
    try:
        parsed = int(setting)
    except ValueError as exc:
        raise ValueError("--fold-parallelism must be 'auto' or a positive integer") from exc
    if parsed < 1:
        raise ValueError("--fold-parallelism must be >= 1")
    return parsed


def _finite(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(numeric):
        return numeric
    return None


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    if int(mask.sum()) < 2 or len(np.unique(y_true[mask])) < 2:
        return None
    try:
        return float(roc_auc_score(y_true[mask], y_score[mask]))
    except ValueError:
        return None


def _true_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _oracle_prediction_metrics(predictions: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"heldout_rows": int(len(predictions))}
    tau_col = _true_column(predictions, TRUE_TAU_CANDIDATES)
    if tau_col is not None:
        frame = predictions[[tau_col, "tau_hat_ncf"]].dropna()
        if len(frame) >= 2:
            metrics["true_tau_column"] = tau_col
            metrics["true_tau_corr"] = _finite(frame[tau_col].corr(frame["tau_hat_ncf"]))
            metrics["true_tau_spearman"] = _finite(frame[tau_col].corr(frame["tau_hat_ncf"], method="spearman"))
            metrics["true_tau_rmse"] = _finite(
                math.sqrt(mean_squared_error(frame[tau_col], frame["tau_hat_ncf"]))
            )
    return metrics


def _attention_hit_metrics(evidence: pd.DataFrame, *, prefix: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {f"{prefix}_rows": int(len(evidence))}
    if evidence.empty:
        return metrics
    evidence = add_oracle_attention_hits(evidence)
    row_col = "row_id" if "row_id" in evidence.columns else None
    for hit_col in [col for col in evidence.columns if col.startswith("hit_")]:
        metrics[f"{prefix}_{hit_col}_row_fraction"] = _finite(evidence[hit_col].astype(bool).mean())
        if row_col is not None:
            per_patient = evidence.groupby(row_col)[hit_col].max().astype(bool)
            metrics[f"{prefix}_{hit_col}_patient_fraction"] = _finite(per_patient.mean())
    return metrics


def _write_jsonl(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


def _write_summary(metric_rows: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"n_folds_completed": int(len(metric_rows))}
    numeric_cols = [col for col in metric_rows.columns if pd.api.types.is_numeric_dtype(metric_rows[col])]
    for col in numeric_cols:
        values = metric_rows[col].dropna()
        if values.empty:
            continue
        summary[f"{col}_mean"] = _finite(values.mean())
        summary[f"{col}_std"] = _finite(values.std())
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def _load_nuisance_predictions(
    path: Path,
    train_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing CATE-stage nuisance predictions: {path}. The NCF oracle "
            "runner writes train_nuisance_predictions.parquet after this change. "
            "Older or in-flight runs that only saved nuisance attention need the "
            "nuisance/full-oracle fold rerun or the pipeline's nuisance_predictions exported."
        )
    nuisance = read_dataframe(path).copy()
    required = {args.row_id_column, "e_hat", "m_hat"}
    missing = sorted(required - set(nuisance.columns))
    if missing:
        raise ValueError(f"Nuisance predictions are missing required columns: {missing}")
    if nuisance[args.row_id_column].duplicated().any():
        duplicated = nuisance.loc[nuisance[args.row_id_column].duplicated(), args.row_id_column].head(5).tolist()
        raise ValueError(f"Nuisance predictions contain duplicate row ids, e.g. {duplicated}")

    expected_ids = train_df[args.row_id_column].tolist()
    by_id = nuisance.set_index(args.row_id_column, drop=False)
    missing_ids = [row_id for row_id in expected_ids if row_id not in by_id.index]
    if missing_ids:
        preview = ", ".join(str(value) for value in missing_ids[:10])
        raise ValueError(
            f"Nuisance predictions at {path} are missing {len(missing_ids)} "
            f"training row id(s), e.g. {preview}"
        )
    nuisance = by_id.loc[expected_ids].reset_index(drop=True).copy()

    t = train_df[args.treatment_column].to_numpy(dtype=float)
    y = train_df[args.outcome_column].to_numpy(dtype=float)
    e_hat = pd.to_numeric(nuisance["e_hat"], errors="coerce").to_numpy(dtype=float)
    m_hat = pd.to_numeric(nuisance["m_hat"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(e_hat).all() or not np.isfinite(m_hat).all():
        raise ValueError(f"Nuisance predictions at {path} contain non-finite e_hat or m_hat values")

    nuisance["e_hat"] = e_hat
    nuisance["m_hat"] = m_hat
    if args.treatment_column not in nuisance.columns:
        nuisance[args.treatment_column] = t
    if args.outcome_column not in nuisance.columns:
        nuisance[args.outcome_column] = y
    nuisance["y_residual"] = y - m_hat
    nuisance["t_residual"] = t - e_hat
    if "r_loss_at_zero_tau" not in nuisance.columns:
        nuisance["r_loss_at_zero_tau"] = nuisance["y_residual"] ** 2
    return nuisance


def _add_r_loss(
    predictions: pd.DataFrame,
    *,
    row_id_column: str,
) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["r_loss_ncf"] = (
        predictions["y_residual"].astype(float)
        - predictions["tau_hat_ncf"].astype(float) * predictions["t_residual"].astype(float)
    ) ** 2
    predictions["r_loss_at_zero_tau"] = predictions["y_residual"].astype(float) ** 2
    return predictions


def _forest_final_metrics(history: pd.DataFrame) -> Dict[str, Any]:
    if history.empty:
        return {}
    last = history.iloc[-1]
    metrics: Dict[str, Any] = {}
    for key, value in last.items():
        if key == "epoch":
            continue
        numeric = _finite(value)
        if numeric is not None:
            metrics[f"forest_final_{key}"] = numeric
    return metrics


def _run_one_fold(
    df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    repeat_index: int,
    fold_index: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    fold_dir = output_dir / f"repeat_{repeat_index:02d}" / f"fold_{fold_index:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_df = df.iloc[train_idx].reset_index(drop=True).copy()
    test_df = df.iloc[test_idx].reset_index(drop=True).copy()
    config = _make_config(args, repeat_index=repeat_index, fold_index=fold_index)
    config.to_json(fold_dir / "resolved_config.json")

    nuisance_path = _nuisance_predictions_path(args, repeat_index, fold_index)
    nuisance_predictions = _load_nuisance_predictions(nuisance_path, train_df, args)
    write_dataframe(nuisance_predictions, fold_dir / "train_nuisance_predictions.parquet")

    model = NeuralCausalForestModel(config, device=device)
    train_result = train_neural_causal_forest(
        model,
        train_df,
        nuisance_predictions,
        text_column=args.text_column,
        treatment_column=args.treatment_column,
        outcome_column=args.outcome_column,
        config=config,
        device=device,
        row_id_column=args.row_id_column,
    )
    forest_history = train_result["history"]
    write_dataframe(forest_history, fold_dir / "forest_history.parquet")

    if args.save_fold_models:
        save_neural_causal_forest_model(
            model,
            fold_dir / "model",
            config=config,
            metadata={
                "mode": "cate_stage_only",
                "nuisance_predictions_path": str(nuisance_path),
                "text_column": args.text_column,
                "treatment_column": args.treatment_column,
                "outcome_column": args.outcome_column,
                "outcome_type": args.outcome_type,
                "row_id_column": args.row_id_column,
                "repeat_index": repeat_index,
                "fold_index": fold_index,
            },
        )

    train_tau = predict_neural_causal_forest(
        model,
        train_df,
        text_column=args.text_column,
        config=config,
        device=device,
        row_id_column=args.row_id_column,
    )
    keep_train_cols = [col for col in train_df.columns if col != args.text_column]
    nuisance_keep = [
        col
        for col in [
            args.row_id_column,
            "e_hat",
            "m_hat",
            "y_residual",
            "t_residual",
            "nuisance_fold",
        ]
        if col in nuisance_predictions.columns
    ]
    train_predictions = (
        train_df[keep_train_cols]
        .merge(nuisance_predictions[nuisance_keep], on=args.row_id_column, how="left")
        .merge(train_tau, on=args.row_id_column, how="left")
    )
    train_predictions = _add_r_loss(train_predictions, row_id_column=args.row_id_column)
    train_predictions["repeat_index"] = int(repeat_index)
    train_predictions["outer_fold"] = int(fold_index)
    train_predictions["split"] = "train"
    write_dataframe(train_predictions, fold_dir / "train_predictions.parquet")

    test_tau = predict_neural_causal_forest(
        model,
        test_df,
        text_column=args.text_column,
        config=config,
        device=device,
        row_id_column=args.row_id_column,
    )
    keep_test_cols = [col for col in test_df.columns if col != args.text_column]
    test_predictions = test_df[keep_test_cols].merge(test_tau, on=args.row_id_column, how="left")
    test_predictions["repeat_index"] = int(repeat_index)
    test_predictions["outer_fold"] = int(fold_index)
    test_predictions["split"] = "heldout"
    write_dataframe(test_predictions, fold_dir / "heldout_predictions.parquet")

    effect_attention = pd.DataFrame()
    if not args.no_attention:
        lookup = test_predictions.set_index(args.row_id_column)
        metadata_rows = [
            {
                "tau_hat_ncf": float(lookup.loc[row_id, "tau_hat_ncf"]),
                "split": "heldout",
                "repeat_index": int(repeat_index),
                "outer_fold": int(fold_index),
            }
            for row_id in test_df[args.row_id_column].tolist()
        ]
        effect_attention = pd.DataFrame(
            causal_forest_attention_evidence(
                model,
                test_df[args.text_column].astype(str).tolist(),
                row_ids=test_df[args.row_id_column].tolist(),
                config=config,
                stage="effect_modifier",
                top_k=config.attention_top_k,
                metadata=metadata_rows,
                target=args.effect_attribution_target,
            )
        )
        if not effect_attention.empty:
            effect_attention_artifact = (
                add_oracle_attention_hits(effect_attention)
                if args.add_oracle_hits
                else effect_attention
            )
            write_dataframe(effect_attention_artifact, fold_dir / "heldout_effect_attention.parquet")
            _write_jsonl(
                build_agent_context_rows(effect_attention_artifact, stage="effect_modifier", max_rows=80),
                fold_dir / "agent_context_effect_modifier.jsonl",
            )

    metric_predictions = train_predictions[
        [
            args.row_id_column,
            "e_hat",
            "m_hat",
            "tau_hat_ncf",
            "r_loss_ncf",
            "r_loss_at_zero_tau",
        ]
    ]
    train_metrics = summarize_pipeline_metrics(
        df=train_df,
        predictions=metric_predictions,
        treatment_column=args.treatment_column,
        outcome_column=args.outcome_column,
        outcome_type=args.outcome_type,
        row_id_column=args.row_id_column,
    )
    metrics = {
        "mode": "cate_stage_only",
        "repeat_index": int(repeat_index),
        "outer_fold": int(fold_index),
        "train_rows": int(len(train_df)),
        "nuisance_predictions_path": str(nuisance_path),
        "structure_rows": int(train_result["structure_rows"]),
        "honest_estimation_rows": int(train_result["honest_estimation_rows"]),
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **_forest_final_metrics(forest_history),
        **_oracle_prediction_metrics(test_predictions, args),
        **_attention_hit_metrics(effect_attention, prefix="effect_attention"),
    }
    if train_result.get("leaf_refit"):
        metrics.update({f"leaf_refit_{k}": v for k, v in train_result["leaf_refit"].items()})
    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return train_predictions, test_predictions, effect_attention, metrics


def _run_one_fold_job(
    job: Tuple[
        pd.DataFrame,
        argparse.Namespace,
        str,
        str,
        int,
        int,
        np.ndarray,
        np.ndarray,
    ],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    (
        df,
        args,
        output_dir,
        device_name,
        repeat_index,
        fold_index,
        train_idx,
        test_idx,
    ) = job
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info(
        "Worker starting NCF CATE stage repeat=%s fold=%s on %s",
        repeat_index,
        fold_index,
        device_name,
    )
    return _run_one_fold(
        df,
        args,
        Path(output_dir),
        torch.device(device_name),
        repeat_index,
        fold_index,
        train_idx,
        test_idx,
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    devices = _resolve_devices(args)
    fold_parallelism = _resolve_fold_parallelism(args, devices)
    args.resolved_devices = devices
    args.resolved_fold_parallelism = int(fold_parallelism)
    df = _prepare_dataframe(args)
    logger.info("Loaded %s rows from %s", len(df), args.dataset)
    logger.info(
        "CATE-stage-only fold execution: devices=%s fold_parallelism=%s",
        ", ".join(devices),
        fold_parallelism,
    )

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)

    train_prediction_frames: List[pd.DataFrame] = []
    heldout_prediction_frames: List[pd.DataFrame] = []
    effect_attention_frames: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    jobs: List[
        Tuple[pd.DataFrame, argparse.Namespace, str, str, int, int, np.ndarray, np.ndarray]
    ] = []
    job_labels: List[Tuple[int, int, str]] = []
    for repeat_index in range(int(args.repeats)):
        for fold_index, train_idx, test_idx in _outer_splits(df, args, repeat_index):
            device_name = devices[len(jobs) % len(devices)]
            jobs.append(
                (
                    df,
                    args,
                    str(output_dir),
                    device_name,
                    int(repeat_index),
                    int(fold_index),
                    train_idx,
                    test_idx,
                )
            )
            job_labels.append((int(repeat_index), int(fold_index), device_name))

    def _record_result(
        train_predictions: pd.DataFrame,
        heldout_predictions: pd.DataFrame,
        effect_attention: pd.DataFrame,
        metrics: Dict[str, Any],
    ) -> None:
        train_prediction_frames.append(train_predictions)
        heldout_prediction_frames.append(heldout_predictions)
        if not effect_attention.empty:
            effect_attention_frames.append(effect_attention)
        metric_rows.append(metrics)

    if fold_parallelism <= 1 or len(jobs) <= 1:
        for job, (repeat_index, fold_index, device_name) in zip(jobs, job_labels):
            _df, _args, _output_dir, _device_name, _repeat_index, _fold_index, train_idx, test_idx = job
            logger.info(
                "NCF CATE stage repeat=%s fold=%s train=%s heldout=%s device=%s",
                repeat_index,
                fold_index,
                len(train_idx),
                len(test_idx),
                device_name,
            )
            try:
                train_predictions, heldout_predictions, effect_attention, metrics = _run_one_fold_job(job)
            except Exception as exc:
                error_row = {
                    "repeat_index": int(repeat_index),
                    "outer_fold": int(fold_index),
                    "device": device_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(error_row)
                logger.exception("Fold failed")
                if args.fail_fast:
                    raise
                continue
            _record_result(train_predictions, heldout_predictions, effect_attention, metrics)
    else:
        max_workers = min(int(fold_parallelism), len(jobs))
        logger.info("Submitting %d CATE fold job(s) with %d worker process(es)", len(jobs), max_workers)
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
        ) as executor:
            future_to_label = {
                executor.submit(_run_one_fold_job, job): label
                for job, label in zip(jobs, job_labels)
            }
            for future in concurrent.futures.as_completed(future_to_label):
                repeat_index, fold_index, device_name = future_to_label[future]
                try:
                    train_predictions, heldout_predictions, effect_attention, metrics = future.result()
                except Exception as exc:
                    error_row = {
                        "repeat_index": int(repeat_index),
                        "outer_fold": int(fold_index),
                        "device": device_name,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    errors.append(error_row)
                    logger.exception(
                        "Fold failed repeat=%s fold=%s device=%s",
                        repeat_index,
                        fold_index,
                        device_name,
                    )
                    if args.fail_fast:
                        raise
                    continue
                logger.info(
                    "Completed CATE fold repeat=%s fold=%s device=%s",
                    repeat_index,
                    fold_index,
                    device_name,
                )
                _record_result(train_predictions, heldout_predictions, effect_attention, metrics)

    if train_prediction_frames:
        write_dataframe(
            pd.concat(train_prediction_frames, ignore_index=True),
            output_dir / "all_train_predictions.parquet",
        )
    if heldout_prediction_frames:
        write_dataframe(
            pd.concat(heldout_prediction_frames, ignore_index=True),
            output_dir / "all_heldout_predictions.parquet",
        )
    if effect_attention_frames:
        write_dataframe(
            pd.concat(effect_attention_frames, ignore_index=True),
            output_dir / "all_effect_attention.parquet",
        )

    metrics_df = pd.DataFrame(metric_rows)
    write_dataframe(metrics_df, output_dir / "fold_metrics.parquet")
    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary = _write_summary(metrics_df, output_dir)

    if errors:
        with open(output_dir / "errors.json", "w", encoding="utf-8") as handle:
            json.dump(errors, handle, indent=2, sort_keys=True)
    logger.info("Completed %s CATE fold(s). Summary: %s", len(metric_rows), summary)


if __name__ == "__main__":
    main()
