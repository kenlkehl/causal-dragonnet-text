#!/usr/bin/env python
"""Run one direct R-learner oracle experiment with HTR W/X role attention.

This is the one-off runner for the direct ``RLearnerNet`` path with the
hierarchical transformer text encoder configured for distinct W and X
attention. It reuses the oracle runner's cross-validation, metrics, result JSON,
and aggregate output schema without constructing the full oracle grid.

Defaults intentionally keep the text window large rather than using smoke-test
settings: 256 chunks, 128 words per chunk, 32-word overlap, and 192 subword
tokens per chunk. That is roughly 24k words of effective note capacity before
tail truncation.

Usage:
    python oracle_experiment_scripts/run_one_off_htr_role_attention_rlearner.py \
        --dataset synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured \
        --output-dir ../pcori_experiments/htr_role_attention_rlearner_oneoff \
        --device cuda:0 \
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


def _parse_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _effective_word_capacity(chunk_size_words: int, overlap_words: int, max_chunks: int) -> int:
    return chunk_size_words + max(0, max_chunks - 1) * (
        chunk_size_words - overlap_words
    )


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    dataset_path = str(Path(args.dataset))
    return ExperimentConfig(
        dataset_path=dataset_path,
        dataset_name=args.dataset_name or Path(dataset_path).name,
        model_type="rlearner",
        use_explicit_confounders=args.use_explicit_confounders,
        feature_extractor_type="hierarchical_transformer",
        repeat_index=args.repeat_index,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_folds=args.n_folds,
        gamma_rlearner=args.gamma_rlearner,
        gamma_rlearner_start=args.gamma_rlearner_start,
        gamma_rlearner_warmup_epochs=args.gamma_rlearner_warmup_epochs,
        gamma_rlearner_ramp_epochs=args.gamma_rlearner_ramp_epochs,
        gamma_rlearner_schedule=args.gamma_rlearner_schedule,
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
        htr_role_attention=args.htr_role_attention,
        htr_w_attention_heads=args.htr_w_attention_heads,
        htr_x_attention_heads=args.htr_x_attention_heads,
    )


def write_aggregate_outputs(output_dir: Path, result: Dict[str, Any]) -> None:
    rows = []
    if not result.get("skipped"):
        rows.append({**result.get("config", {}), **result.get("metrics", {})})
    if not rows:
        logger.info("No successful results to aggregate")
        return

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    results_df.to_parquet(output_dir / "all_results.parquet", index=False)

    group_cols = [
        "dataset_name",
        "feature_extractor_type",
        "model_type",
        "use_explicit_confounders",
        "learning_rate",
        "epochs",
        "htr_sentence_model",
        "htr_sentence_pooling",
        "htr_chunk_size_words",
        "htr_chunk_overlap_words",
        "htr_max_chunks",
        "htr_max_chunk_length",
        "htr_role_attention",
        "htr_w_attention_heads",
        "htr_x_attention_heads",
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
    parser = argparse.ArgumentParser(description=__doc__)
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
        default="../pcori_experiments/htr_role_attention_rlearner_oneoff",
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
            "One or more training devices, e.g. --devices cuda:0 cuda:1. "
            "CV folds are distributed across devices round-robin and trained "
            "concurrently. '--device' is accepted as an alias."
        ),
    )
    parser.add_argument(
        "--max-parallel-folds",
        type=int,
        default=None,
        help=(
            "Maximum number of folds to train concurrently (default: all folds at once). "
            "Lower this if a GPU runs out of memory."
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma-rlearner", type=float, default=1.0)
    parser.add_argument("--gamma-rlearner-start", type=float, default=None)
    parser.add_argument("--gamma-rlearner-warmup-epochs", type=int, default=0)
    parser.add_argument("--gamma-rlearner-ramp-epochs", type=int, default=0)
    parser.add_argument(
        "--gamma-rlearner-schedule",
        choices=["constant", "linear", "cosine"],
        default="constant",
    )
    parser.add_argument("--repeat-index", type=int, default=0)

    parser.add_argument("--htr-sentence-model", default="prajjwal1/bert-tiny")
    parser.add_argument("--htr-freeze-sentence-encoder", type=_parse_bool, default=False)
    parser.add_argument("--htr-chunk-size-words", type=int, default=128)
    parser.add_argument("--htr-chunk-overlap-words", type=int, default=32)
    parser.add_argument("--htr-max-chunks", type=int, default=256)
    parser.add_argument("--htr-max-chunk-length", type=int, default=192)
    parser.add_argument("--htr-num-layers", type=int, default=2)
    parser.add_argument("--htr-num-heads", type=int, default=4)
    parser.add_argument("--htr-transformer-dim", type=int, default=256)
    parser.add_argument("--htr-projection-dim", type=int, default=128)
    parser.add_argument("--htr-hash-embedding-dim", type=int, default=256)
    parser.add_argument("--htr-sentence-encoder-batch-size", type=int, default=64)
    parser.add_argument(
        "--htr-sentence-encoder-backend",
        choices=["auto", "sentence_transformers", "transformers"],
        default="transformers",
    )
    parser.add_argument(
        "--htr-sentence-pooling",
        choices=["auto", "cls", "last", "mean", "token_attention"],
        default="token_attention",
    )
    parser.add_argument("--htr-normalize-sentence-embeddings", type=_parse_bool, default=True)
    parser.add_argument("--htr-trainable-sentence-encoder-layers", type=int, default=0)
    parser.add_argument("--htr-dropout", type=float, default=0.1)
    parser.add_argument("--htr-role-attention", type=_parse_bool, default=True)
    parser.add_argument("--htr-w-attention-heads", type=int, default=4)
    parser.add_argument("--htr-x-attention-heads", type=int, default=4)

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
    if args.gamma_rlearner_warmup_epochs < 0:
        parser.error("--gamma-rlearner-warmup-epochs must be >= 0")
    if args.gamma_rlearner_ramp_epochs < 0:
        parser.error("--gamma-rlearner-ramp-epochs must be >= 0")
    if args.htr_chunk_size_words < 1:
        parser.error("--htr-chunk-size-words must be >= 1")
    if args.htr_chunk_overlap_words < 0:
        parser.error("--htr-chunk-overlap-words must be >= 0")
    if args.htr_chunk_overlap_words >= args.htr_chunk_size_words:
        parser.error("--htr-chunk-overlap-words must be < --htr-chunk-size-words")
    if args.htr_max_chunks < 1:
        parser.error("--htr-max-chunks must be >= 1")
    if args.htr_max_chunk_length < 1:
        parser.error("--htr-max-chunk-length must be >= 1")
    if args.htr_num_layers < 1:
        parser.error("--htr-num-layers must be >= 1")
    if args.htr_num_heads < 1:
        parser.error("--htr-num-heads must be >= 1")
    if args.htr_transformer_dim < 1:
        parser.error("--htr-transformer-dim must be >= 1")
    if args.htr_projection_dim < 1:
        parser.error("--htr-projection-dim must be >= 1")
    if args.htr_hash_embedding_dim < 1:
        parser.error("--htr-hash-embedding-dim must be >= 1")
    if args.htr_sentence_encoder_batch_size < 1:
        parser.error("--htr-sentence-encoder-batch-size must be >= 1")
    if args.htr_trainable_sentence_encoder_layers < 0:
        parser.error("--htr-trainable-sentence-encoder-layers must be >= 0")
    if args.htr_dropout < 0 or args.htr_dropout >= 1:
        parser.error("--htr-dropout must be in [0, 1)")
    if args.htr_w_attention_heads < 1:
        parser.error("--htr-w-attention-heads must be >= 1")
    if args.htr_x_attention_heads < 1:
        parser.error("--htr-x-attention-heads must be >= 1")

    return args


def main() -> None:
    args = parse_args()
    config = build_config(args)
    config_hash = config.config_hash()
    output_dir = Path(args.output_dir)
    word_capacity = _effective_word_capacity(
        args.htr_chunk_size_words,
        args.htr_chunk_overlap_words,
        args.htr_max_chunks,
    )

    if args.dry_run:
        print(json.dumps(asdict(config), indent=2, default=str))
        print(f"config_hash: {config_hash}")
        print(f"htr_effective_word_capacity: {word_capacity}")
        print(f"htr_max_chunk_length: {args.htr_max_chunk_length}")
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
            "Running one-off HTR role-attention rlearner: dataset=%s, folds=%d, "
            "epochs=%d, lr=%s, devices=%s, chunk_size_words=%d, overlap_words=%d, "
            "max_chunks=%d, effective_word_capacity=%d, max_chunk_length=%d, "
            "W_heads=%d, X_heads=%d",
            config.dataset_name,
            config.n_folds,
            config.epochs,
            config.learning_rate,
            ", ".join(args.devices),
            config.htr_chunk_size_words,
            config.htr_chunk_overlap_words,
            config.htr_max_chunks,
            word_capacity,
            config.htr_max_chunk_length,
            config.htr_w_attention_heads,
            config.htr_x_attention_heads,
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
