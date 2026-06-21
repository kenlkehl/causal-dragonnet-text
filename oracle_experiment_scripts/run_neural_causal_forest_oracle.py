#!/usr/bin/env python
"""Oracle experiment runner for the neural causal-forest text extractor.

The runner is tailored to synthetic datasets where true treatment-effect columns
may be present.  It trains the neural causal forest inside outer folds, evaluates
held-out CATE recovery, and can optionally add synthetic oracle-hit annotations
to exported attention evidence for debugging.

Typical quick run:

    python oracle_experiment_scripts/run_neural_causal_forest_oracle.py \
        --dataset synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured \
        --output-dir ../ncf_oracle_runs/one_confounder_one_modifier \
        --device cuda:0 --n-folds 5 --repeats 1 \
        --encoder-model prajjwal1/bert-tiny --nuisance-epochs 50 --forest-epochs 80

To run outer folds concurrently across GPUs, pass a device list:

    python oracle_experiment_scripts/run_neural_causal_forest_oracle.py \
        --dataset synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured \
        --output-dir ../ncf_oracle_runs/one_confounder_one_modifier \
        --devices cuda:0 cuda:1 cuda:2 cuda:3 --fold-parallelism auto \
        --inner-fold-parallelism 2 \
        --n-folds 5 --repeats 1 --config example_configs/neural_causal_forest_config.json

For no-download smoke tests, use ``--encoder-backend hash --no-attention
--nuisance-epochs 1 --forest-epochs 1 --sample-size 64``.
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
from dataclasses import asdict
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
    add_oracle_attention_hits,
    build_agent_context_rows,
    causal_forest_attention_evidence,
    fit_neural_causal_forest_pipeline,
    predict_neural_causal_forest,
    read_dataframe,
    save_neural_causal_forest_model,
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
        help=(
            "Optional device list for outer-fold parallelism, e.g. "
            "--devices cuda:0 cuda:1 cuda:2 cuda:3. Defaults to --device."
        ),
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
    parser.add_argument("--config", default=None, help="Optional NeuralCausalForestConfig JSON")

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
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--effect-batch-size", type=int, default=None)
    parser.add_argument("--nuisance-folds", type=int, default=None)
    parser.add_argument(
        "--inner-fold-parallelism",
        default=None,
        help=(
            "Number of nuisance cross-fit folds inside each outer fold to train "
            "concurrently. 'auto' uses num_workers on CPU and stays serial on "
            "CUDA; an explicit integer opts into that many concurrent folds, "
            "including on CUDA."
        ),
    )
    parser.add_argument("--nuisance-epochs", type=int, default=None)
    parser.add_argument("--forest-epochs", type=int, default=None)
    parser.add_argument("--n-trees", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--forest-learning-rate", type=float, default=None)
    parser.add_argument("--nuisance-learning-rate", type=float, default=None)
    parser.add_argument("--nuisance-weight-decay", type=float, default=None)
    parser.add_argument("--nuisance-label-smoothing", type=float, default=None)
    parser.add_argument(
        "--nuisance-calibration",
        choices=["none", "temperature", "isotonic", "temperature_isotonic"],
        default=None,
    )
    parser.add_argument("--lambda-heterogeneity", type=float, default=None)
    parser.add_argument("--attention-top-k", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--no-attention", action="store_true", help="Skip attention evidence artifacts")
    parser.add_argument(
        "--add-oracle-hits",
        action="store_true",
        help="Annotate attention evidence artifacts with synthetic oracle regex hits for debugging.",
    )
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _make_config(args: argparse.Namespace, repeat_index: int, fold_index: int) -> NeuralCausalForestConfig:
    config = NeuralCausalForestConfig.from_json(args.config) if args.config else NeuralCausalForestConfig()
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
        "htr_num_layers": args.htr_num_layers,
        "htr_num_heads": args.htr_num_heads,
        "htr_transformer_dim": args.htr_transformer_dim,
        "htr_sentence_encoder_batch_size": args.htr_sentence_encoder_batch_size,
        "htr_sentence_encoder_backend": args.htr_sentence_encoder_backend,
        "htr_sentence_pooling": args.htr_sentence_pooling,
        "batch_size": args.batch_size,
        "effect_batch_size": args.effect_batch_size,
        "nuisance_folds": args.nuisance_folds,
        "inner_fold_parallelism": args.inner_fold_parallelism,
        "nuisance_epochs": args.nuisance_epochs,
        "forest_epochs": args.forest_epochs,
        "n_trees": args.n_trees,
        "depth": args.depth,
        "forest_learning_rate": args.forest_learning_rate,
        "nuisance_learning_rate": args.nuisance_learning_rate,
        "nuisance_weight_decay": args.nuisance_weight_decay,
        "nuisance_label_smoothing": args.nuisance_label_smoothing,
        "nuisance_calibration": args.nuisance_calibration,
        "lambda_heterogeneity": args.lambda_heterogeneity,
        "attention_top_k": args.attention_top_k,
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
    if args.treatment_column in predictions and "e_hat" in predictions:
        metrics["heldout_propensity_auroc_if_available"] = _safe_roc_auc(
            predictions[args.treatment_column].to_numpy(dtype=float),
            predictions["e_hat"].to_numpy(dtype=float),
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

    result = fit_neural_causal_forest_pipeline(
        train_df,
        text_column=args.text_column,
        treatment_column=args.treatment_column,
        outcome_column=args.outcome_column,
        outcome_type=args.outcome_type,
        config=config,
        device=device,
        row_id_column=args.row_id_column,
        collect_attention=not args.no_attention,
        nuisance_artifact_dir=fold_dir,
    )
    write_dataframe(result.nuisance_predictions, fold_dir / "train_nuisance_predictions.parquet")
    write_dataframe(result.nuisance_history, fold_dir / "nuisance_history.parquet")
    write_dataframe(result.forest_history, fold_dir / "forest_history.parquet")
    write_dataframe(result.train_predictions, fold_dir / "train_predictions.parquet")
    if args.save_fold_models:
        save_neural_causal_forest_model(
            result.model,
            fold_dir / "model",
            config=config,
            metadata={
                "text_column": args.text_column,
                "treatment_column": args.treatment_column,
                "outcome_column": args.outcome_column,
                "outcome_type": args.outcome_type,
                "row_id_column": args.row_id_column,
                "repeat_index": repeat_index,
                "fold_index": fold_index,
            },
        )

    test_predictions = predict_neural_causal_forest(
        result.model,
        test_df,
        text_column=args.text_column,
        config=config,
        device=device,
        row_id_column=args.row_id_column,
    )
    keep_cols = [col for col in test_df.columns if col != args.text_column]
    test_predictions = test_df[keep_cols].merge(test_predictions, on=args.row_id_column, how="left")
    test_predictions["repeat_index"] = int(repeat_index)
    test_predictions["outer_fold"] = int(fold_index)
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
                result.model,
                test_df[args.text_column].astype(str).tolist(),
                row_ids=test_df[args.row_id_column].tolist(),
                config=config,
                stage="effect_modifier",
                top_k=config.attention_top_k,
                metadata=metadata_rows,
                target="tau_heterogeneity",
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
        if not result.nuisance_attention.empty:
            nuisance_attention = result.nuisance_attention
            nuisance_attention_artifact = (
                add_oracle_attention_hits(nuisance_attention)
                if args.add_oracle_hits
                else nuisance_attention
            )
            write_dataframe(nuisance_attention_artifact, fold_dir / "train_nuisance_attention.parquet")
            _write_jsonl(
                build_agent_context_rows(nuisance_attention_artifact, stage="nuisance", max_rows=80),
                fold_dir / "agent_context_nuisance.jsonl",
            )
        else:
            nuisance_attention = pd.DataFrame()
    else:
        nuisance_attention = pd.DataFrame()

    metrics = {
        "repeat_index": int(repeat_index),
        "outer_fold": int(fold_index),
        "train_rows": int(len(train_df)),
        **{f"train_{k}": v for k, v in result.metrics.items()},
        **_oracle_prediction_metrics(test_predictions, args),
        **_attention_hit_metrics(nuisance_attention, prefix="nuisance_attention"),
        **_attention_hit_metrics(effect_attention, prefix="effect_attention"),
    }
    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return test_predictions, effect_attention, nuisance_attention, metrics


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
        "Worker starting neural causal forest repeat=%s fold=%s on %s",
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
        "Outer fold execution: devices=%s fold_parallelism=%s",
        ", ".join(devices),
        fold_parallelism,
    )

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)

    prediction_frames: List[pd.DataFrame] = []
    effect_attention_frames: List[pd.DataFrame] = []
    nuisance_attention_frames: List[pd.DataFrame] = []
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
        predictions: pd.DataFrame,
        effect_attention: pd.DataFrame,
        nuisance_attention: pd.DataFrame,
        metrics: Dict[str, Any],
    ) -> None:
        prediction_frames.append(predictions)
        if not effect_attention.empty:
            effect_attention_frames.append(effect_attention)
        if not nuisance_attention.empty:
            nuisance_attention_frames.append(nuisance_attention)
        metric_rows.append(metrics)

    if fold_parallelism <= 1 or len(jobs) <= 1:
        for job, (repeat_index, fold_index, device_name) in zip(jobs, job_labels):
            _df, _args, _output_dir, _device_name, _repeat_index, _fold_index, train_idx, test_idx = job
            logger.info(
                "Neural causal forest oracle repeat=%s fold=%s train=%s heldout=%s device=%s",
                repeat_index,
                fold_index,
                len(train_idx),
                len(test_idx),
                device_name,
            )
            try:
                predictions, effect_attention, nuisance_attention, metrics = _run_one_fold_job(job)
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
            _record_result(predictions, effect_attention, nuisance_attention, metrics)
    else:
        max_workers = min(int(fold_parallelism), len(jobs))
        logger.info("Submitting %d outer fold job(s) with %d worker process(es)", len(jobs), max_workers)
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
                    predictions, effect_attention, nuisance_attention, metrics = future.result()
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
                    "Completed fold repeat=%s fold=%s device=%s",
                    repeat_index,
                    fold_index,
                    device_name,
                )
                _record_result(predictions, effect_attention, nuisance_attention, metrics)

    if prediction_frames:
        write_dataframe(
            pd.concat(prediction_frames, ignore_index=True),
            output_dir / "all_heldout_predictions.parquet",
        )
    if effect_attention_frames:
        write_dataframe(
            pd.concat(effect_attention_frames, ignore_index=True),
            output_dir / "all_effect_attention.parquet",
        )
    if nuisance_attention_frames:
        write_dataframe(
            pd.concat(nuisance_attention_frames, ignore_index=True),
            output_dir / "all_nuisance_attention.parquet",
        )

    metrics_df = pd.DataFrame(metric_rows)
    write_dataframe(metrics_df, output_dir / "fold_metrics.parquet")
    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary = _write_summary(metrics_df, output_dir)

    if errors:
        with open(output_dir / "errors.json", "w", encoding="utf-8") as handle:
            json.dump(errors, handle, indent=2, sort_keys=True)
    logger.info("Completed %s fold(s). Summary: %s", len(metric_rows), summary)


if __name__ == "__main__":
    main()
