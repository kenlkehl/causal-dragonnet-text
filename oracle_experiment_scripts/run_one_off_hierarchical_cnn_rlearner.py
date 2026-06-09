#!/usr/bin/env python
"""Run one neural R-learner experiment with a Hierarchical CNN raw-text extractor.

This is the hierarchical-CNN counterpart to
run_one_off_simple_cnn_rlearner.py. It reuses the oracle runner's training,
cross-validation, metrics, result JSON, and aggregate output schema without
constructing the full oracle grid.

Defaults target roughly 100k tokens of text capacity:
12,000-token chunks, 64-token overlap, and 9 chunks.

Cross-validation folds are independent, so they are distributed across the
requested devices round-robin and trained concurrently. With 5 folds and 4 GPUs,
for example, three GPUs each train one fold while the remaining folds double up so
that all folds run at once. Pooled predictions and metrics are identical to a
sequential single-device run (same KFold seed).

Usage:
    # Single device (folds run concurrently on that device)
    python oracle_experiment_scripts/run_one_off_hierarchical_cnn_rlearner.py \
        --dataset synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured \
        --output-dir ../pcori_experiments/hierarchical_cnn_rlearner_oneoff \
        --device cuda:0 \
        --epochs 25 \
        --n-folds 5

    # Multiple devices (folds parallelized across GPUs)
    python oracle_experiment_scripts/run_one_off_hierarchical_cnn_rlearner.py \
        --dataset synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured \
        --output-dir ../pcori_experiments/hierarchical_cnn_rlearner_oneoff \
        --devices cuda:0 cuda:1 cuda:2 cuda:3 \
        --epochs 25 \
        --n-folds 5
"""

import argparse
import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_one_off_simple_cnn_rlearner import _parse_device, run_parallel_fold_experiment
from run_oracle_experiments import ExperimentConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _effective_token_capacity(chunk_size: int, chunk_overlap: int, max_chunks: int) -> int:
    return chunk_size + (max_chunks - 1) * (chunk_size - chunk_overlap)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    dataset_path = str(Path(args.dataset))
    return ExperimentConfig(
        dataset_path=dataset_path,
        dataset_name=args.dataset_name or Path(dataset_path).name,
        model_type="rlearner",
        use_explicit_confounders=args.use_explicit_confounders,
        feature_extractor_type="hierarchical_cnn",
        repeat_index=args.repeat_index,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_folds=args.n_folds,
        gamma_rlearner=args.gamma_rlearner,
        hcnn_embedding_dim=args.hcnn_embedding_dim,
        hcnn_conv_dim=args.hcnn_conv_dim,
        hcnn_kernel_size=args.hcnn_kernel_size,
        hcnn_num_conv_blocks=args.hcnn_num_conv_blocks,
        hcnn_chunk_size=args.hcnn_chunk_size,
        hcnn_chunk_overlap=args.hcnn_chunk_overlap,
        hcnn_max_chunks=args.hcnn_max_chunks,
        hcnn_vocab_size=args.hcnn_vocab_size,
        hcnn_projection_dim=args.hcnn_projection_dim,
        hcnn_dropout=args.hcnn_dropout,
    )


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
        "hcnn_chunk_size",
        "hcnn_chunk_overlap",
        "hcnn_max_chunks",
        "hcnn_projection_dim",
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
        description="One-off neural R-learner run with a Hierarchical CNN raw-text extractor"
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
        default="../pcori_experiments/hierarchical_cnn_rlearner_oneoff",
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
            "Lower this if host RAM or a single GPU runs out of memory."
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

    parser.add_argument("--hcnn-embedding-dim", type=int, default=256)
    parser.add_argument("--hcnn-conv-dim", type=int, default=256)
    parser.add_argument("--hcnn-kernel-size", type=int, default=5)
    parser.add_argument("--hcnn-num-conv-blocks", type=int, default=4)
    parser.add_argument("--hcnn-chunk-size", type=int, default=12000)
    parser.add_argument("--hcnn-chunk-overlap", type=int, default=64)
    parser.add_argument("--hcnn-max-chunks", type=int, default=9)
    parser.add_argument("--hcnn-vocab-size", type=int, default=50000)
    parser.add_argument("--hcnn-projection-dim", type=int, default=128)
    parser.add_argument("--hcnn-dropout", type=float, default=0.1)

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
    if args.hcnn_chunk_size < 1:
        parser.error("--hcnn-chunk-size must be >= 1")
    if args.hcnn_chunk_overlap < 0:
        parser.error("--hcnn-chunk-overlap must be >= 0")
    if args.hcnn_chunk_overlap >= args.hcnn_chunk_size:
        parser.error("--hcnn-chunk-overlap must be < --hcnn-chunk-size")
    if args.hcnn_max_chunks < 1:
        parser.error("--hcnn-max-chunks must be >= 1")
    if args.hcnn_vocab_size < 2:
        parser.error("--hcnn-vocab-size must be >= 2")
    if args.hcnn_projection_dim < 1:
        parser.error("--hcnn-projection-dim must be >= 1")
    if args.hcnn_dropout < 0 or args.hcnn_dropout >= 1:
        parser.error("--hcnn-dropout must be in [0, 1)")

    return args


def main() -> None:
    args = parse_args()
    config = build_config(args)
    config_hash = config.config_hash()
    output_dir = Path(args.output_dir)
    token_capacity = _effective_token_capacity(
        args.hcnn_chunk_size,
        args.hcnn_chunk_overlap,
        args.hcnn_max_chunks,
    )

    if args.dry_run:
        print(json.dumps(asdict(config), indent=2, default=str))
        print(f"config_hash: {config_hash}")
        print(f"hcnn_effective_token_capacity: {token_capacity}")
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
            "Running one-off hierarchical_cnn rlearner: dataset=%s, folds=%d, epochs=%d, "
            "lr=%s, devices=%s, chunk_size=%d, overlap=%d, max_chunks=%d, "
            "effective_token_capacity=%d",
            config.dataset_name,
            config.n_folds,
            config.epochs,
            config.learning_rate,
            ", ".join(args.devices),
            config.hcnn_chunk_size,
            config.hcnn_chunk_overlap,
            config.hcnn_max_chunks,
            token_capacity,
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
