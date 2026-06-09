#!/usr/bin/env python
"""Run one neural R-learner experiment with a Simple CNN raw-text extractor.

This is a thin, deterministic wrapper around run_oracle_experiments.py for the
common "just run one analysis" case. It reuses the oracle runner's training,
cross-validation, metrics, result JSON, and aggregate output schema without
constructing the full oracle grid.

Cross-validation folds are independent, so they are distributed across the
requested devices round-robin and trained concurrently. With 5 folds and 4 GPUs,
for example, three GPUs each train one fold while the remaining folds double up so
that all folds run at once. Pooled predictions and metrics are identical to a
sequential single-device run (same KFold seed).

Usage:
    # Single device (folds run concurrently on that device)
    python oracle_experiment_scripts/run_one_off_simple_cnn_rlearner.py \
        --dataset synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured \
        --output-dir ../pcori_experiments/simple_cnn_rlearner_oneoff \
        --device cuda:0 \
        --epochs 25 \
        --n-folds 5

    # Multiple devices (folds parallelized across GPUs)
    python oracle_experiment_scripts/run_one_off_simple_cnn_rlearner.py \
        --dataset synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured \
        --output-dir ../pcori_experiments/simple_cnn_rlearner_oneoff \
        --devices cuda:0 cuda:1 cuda:2 cuda:3 \
        --epochs 25 \
        --n-folds 5
"""

import argparse
import concurrent.futures
import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_oracle_experiments import (
    ExperimentConfig,
    _rename_confounder_columns,
    _resolve_parquet_file,
    _run_neural_fold,
    compute_metrics,
    load_confounder_specs_from_metadata,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_device(value: str) -> str:
    if value == "cpu" or value.startswith("cuda"):
        return value
    raise argparse.ArgumentTypeError("device must be 'cpu' or start with 'cuda'")


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    dataset_path = str(Path(args.dataset))
    return ExperimentConfig(
        dataset_path=dataset_path,
        dataset_name=args.dataset_name or Path(dataset_path).name,
        model_type="rlearner",
        use_explicit_confounders=args.use_explicit_confounders,
        feature_extractor_type="simple_cnn",
        repeat_index=args.repeat_index,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_folds=args.n_folds,
        gamma_rlearner=args.gamma_rlearner,
        scnn_embedding_dim=args.scnn_embedding_dim,
        scnn_conv_dim=args.scnn_conv_dim,
        scnn_kernel_size=args.scnn_kernel_size,
        scnn_num_conv_blocks=args.scnn_num_conv_blocks,
        scnn_max_length=args.scnn_max_length,
        scnn_vocab_size=args.scnn_vocab_size,
        scnn_projection_dim=args.scnn_projection_dim,
        scnn_dropout=args.scnn_dropout,
    )


def run_parallel_fold_experiment(
    config: ExperimentConfig,
    devices: List[str],
    max_parallel_folds: int = None,
) -> Dict[str, Any]:
    """Run the rlearner CV experiment with folds parallelized across devices.

    Folds are independent, so each fold trains its own model on an assigned device.
    Folds are mapped to devices round-robin (fold i -> devices[i % n_devices]) and
    trained concurrently: with 5 folds and 4 devices, three devices each train one
    fold while the remaining folds double up on the others, so every fold runs at
    once. Predictions are pooled across folds and metrics computed exactly as in
    the sequential runner (identical KFold seed), so results match a single-device
    run. ``max_parallel_folds`` caps how many folds train at the same time (default:
    all folds).

    Returns a result dict matching ``run_single_experiment``'s schema.
    """
    parquet_file = _resolve_parquet_file(config.dataset_path)
    if parquet_file is None:
        return {
            "config": asdict(config),
            "error": f"Dataset not found in {config.dataset_path}",
            "skipped": True,
        }

    df = pd.read_parquet(parquet_file)
    text_column = "clinical_text"
    if text_column not in df.columns:
        return {
            "config": asdict(config),
            "error": f"Text column '{text_column}' not found",
            "skipped": True,
        }

    # Optional explicit confounders (mirrors run_single_experiment preprocessing).
    confounder_specs = None
    confounder_cols = None
    if config.use_explicit_confounders:
        confounder_specs = load_confounder_specs_from_metadata(config.dataset_path)
        if not confounder_specs:
            return {
                "config": asdict(config),
                "error": f"No confounder specs found in {config.dataset_path}",
                "skipped": True,
            }
        logger.info(
            "Using %d explicit confounders: %s",
            len(confounder_specs),
            [s.name for s in confounder_specs],
        )
        df = _rename_confounder_columns(df, confounder_specs)
        confounder_cols = [f"explicit_conf_{s.name}" for s in confounder_specs]
        missing_cols = [c for c in confounder_cols if c not in df.columns]
        if missing_cols:
            return {
                "config": asdict(config),
                "error": (
                    f"Confounder columns missing from dataset: {missing_cols}. "
                    f"Run LLM extraction first to create llm_extracted_* columns."
                ),
                "skipped": True,
            }

    # simple_cnn is a trainable extractor: no cached/frozen hidden states.
    gpu_store = None
    hidden_state_cache = None

    df = df.reset_index(drop=True)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42 + config.repeat_index)
    folds = list(enumerate(kf.split(df)))

    devices_t = [torch.device(d) for d in devices]
    n_workers = len(folds) if max_parallel_folds is None else min(max_parallel_folds, len(folds))

    def _run_fold(fold_item):
        fold, (train_idx, test_idx) = fold_item
        device = devices_t[fold % len(devices_t)]
        logger.info("Fold %d/%d training on %s", fold + 1, config.n_folds, device)
        fold_preds, _history = _run_neural_fold(
            config, device, df, confounder_specs, confounder_cols,
            gpu_store, hidden_state_cache, fold, train_idx, test_idx,
        )
        logger.info("Fold %d/%d finished on %s", fold + 1, config.n_folds, device)
        return fold_preds

    all_predictions = [None] * len(folds)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_fold = {executor.submit(_run_fold, item): item[0] for item in folds}
        for future in concurrent.futures.as_completed(future_to_fold):
            fold = future_to_fold[future]
            all_predictions[fold] = future.result()

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
    )

    return {
        "config": asdict(config),
        "metrics": metrics,
        "n_samples": len(results_df),
        "skipped": False,
        "error": None,
        "artifacts": {},
    }


def write_aggregate_outputs(output_dir: Path, result: Dict[str, Any]) -> None:
    all_results = []
    if not result.get("skipped"):
        all_results.append({**result.get("config", {}), **result.get("metrics", {})})

    if not all_results:
        logger.info("No successful results to aggregate")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    results_df.to_parquet(output_dir / "all_results.parquet", index=False)

    group_cols = [
        "dataset_name",
        "feature_extractor_type",
        "model_type",
        "use_explicit_confounders",
        "learning_rate",
        "epochs",
        "scnn_max_length",
        "scnn_projection_dim",
    ]
    group_cols = [col for col in group_cols if col in results_df.columns]

    metric_agg = {}
    for metric in [
        "ite_corr",
        "ite_spearman_corr",
        "ate_bias",
        "ate_pred",
        "ate_true",
        "propensity_auroc",
        "ite_mse",
        "ite_mae",
        "y0_mse",
        "y1_mse",
    ]:
        if metric in results_df.columns:
            metric_agg[metric] = ["mean", "std"]

    if group_cols and metric_agg:
        summary = results_df.groupby(group_cols, dropna=False).agg(metric_agg)
        summary.to_csv(output_dir / "summary.csv")
        logger.info("\nSummary:\n%s", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-off neural R-learner run with a Simple CNN raw-text extractor"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset directory containing dataset.parquet or dataset_with_extraction.parquet",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset label stored in result configs; defaults to directory name",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="../pcori_experiments/simple_cnn_rlearner_oneoff",
        help="Output directory for result JSON and aggregate files",
    )
    parser.add_argument(
        "--devices",
        "--device",
        dest="devices",
        type=_parse_device,
        nargs="+",
        default=["cuda:0"],
        help=(
            "One or more training devices, e.g. --devices cuda:0 cuda:1 cuda:2 cuda:3 "
            "(or cpu). CV folds are distributed across devices round-robin and trained "
            "concurrently. '--device' is accepted as an alias."
        ),
    )
    parser.add_argument(
        "--max-parallel-folds",
        type=int,
        default=None,
        help=(
            "Maximum number of folds to train concurrently (default: all folds at once). "
            "Lower this if a single GPU runs out of memory when folds double up on it."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Skip if result JSON already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument(
        "--use-explicit-confounders",
        action="store_true",
        help="Include metadata confounders if extracted columns are present",
    )

    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma-rlearner", type=float, default=1.0)
    parser.add_argument("--repeat-index", type=int, default=0)

    parser.add_argument("--scnn-embedding-dim", type=int, default=256)
    parser.add_argument("--scnn-conv-dim", type=int, default=256)
    parser.add_argument("--scnn-kernel-size", type=int, default=5)
    parser.add_argument("--scnn-num-conv-blocks", type=int, default=4)
    parser.add_argument("--scnn-max-length", type=int, default=50000)
    parser.add_argument("--scnn-vocab-size", type=int, default=50000)
    parser.add_argument("--scnn-projection-dim", type=int, default=128)
    parser.add_argument("--scnn-dropout", type=float, default=0.1)

    args = parser.parse_args()

    if args.n_folds < 2:
        parser.error("--n-folds must be >= 2")
    if args.max_parallel_folds is not None and args.max_parallel_folds < 1:
        parser.error("--max-parallel-folds must be >= 1")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be > 0")
    if args.scnn_max_length < 1:
        parser.error("--scnn-max-length must be >= 1")
    if args.scnn_vocab_size < 2:
        parser.error("--scnn-vocab-size must be >= 2")
    if args.scnn_projection_dim < 1:
        parser.error("--scnn-projection-dim must be >= 1")
    if args.scnn_dropout < 0 or args.scnn_dropout >= 1:
        parser.error("--scnn-dropout must be in [0, 1)")

    return args


def main() -> None:
    args = parse_args()
    config = build_config(args)
    config_hash = config.config_hash()
    output_dir = Path(args.output_dir)

    if args.dry_run:
        print(json.dumps(asdict(config), indent=2, default=str))
        print(f"config_hash: {config_hash}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command_line.txt").write_text(" ".join(sys.argv) + "\n")

    result_file = output_dir / "results" / f"{config_hash}.json"
    if args.resume and result_file.exists():
        logger.info("Result already exists, loading %s", result_file)
        with open(result_file) as f:
            result = json.load(f)
        write_aggregate_outputs(output_dir, result)
        return

    try:
        logger.info(
            "Running one-off simple_cnn rlearner: dataset=%s, folds=%d, epochs=%d, lr=%s, devices=%s",
            config.dataset_name,
            config.n_folds,
            config.epochs,
            config.learning_rate,
            ", ".join(args.devices),
        )
        result = run_parallel_fold_experiment(
            config,
            args.devices,
            max_parallel_folds=args.max_parallel_folds,
        )
    except Exception as exc:
        logger.error("Experiment %s failed: %s\n%s", config_hash, exc, traceback.format_exc())
        result = {
            "config": asdict(config),
            "error": str(exc),
            "skipped": True,
        }

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    write_aggregate_outputs(output_dir, result)

    if result.get("skipped"):
        raise SystemExit(f"Experiment skipped/failed: {result.get('error', 'unknown error')}")

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
