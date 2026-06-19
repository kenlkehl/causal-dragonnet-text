#!/usr/bin/env python
"""Train a neural causal-forest text extractor and export token evidence.

Example:

    python oci/inference/train_neural_causal_forest.py \
        --data synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured \
        --output-dir ../ncf_runs/one_confounder_one_modifier \
        --device cuda:0 \
        --encoder-model prajjwal1/bert-tiny \
        --nuisance-epochs 50 \
        --forest-epochs 80 \
        --inner-fold-parallelism 2 \
        --n-trees 32 --depth 3

The script writes:

    <output-dir>/model/neural_causal_forest.pt
    <output-dir>/model/neural_causal_forest_config.json
    <output-dir>/nuisance_oof_predictions.parquet
    <output-dir>/train_predictions.parquet
    <output-dir>/nuisance_attention_evidence.parquet
    <output-dir>/effect_attention_evidence.parquet
    <output-dir>/agent_context_nuisance.jsonl
    <output-dir>/agent_context_effect_modifier.jsonl
    <output-dir>/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

# Allow running the file directly from the repo root or from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from oci.models.neural_causal_forest_extractor import (  # noqa: E402
    NeuralCausalForestConfig,
    add_oracle_attention_hits,
    build_agent_context_rows,
    fit_neural_causal_forest_pipeline,
    read_dataframe,
    save_neural_causal_forest_model,
    write_dataframe,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV/parquet file or dataset directory")
    parser.add_argument("--output-dir", required=True, help="Directory for model and artifacts")
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--row-id-column", default="_ncf_row_id")
    parser.add_argument("--outcome-type", choices=["binary", "continuous"], default="binary")
    parser.add_argument("--split-column", default="split")
    parser.add_argument(
        "--train-split-values",
        nargs="*",
        default=["train", "val"],
        help="When split-column exists, use these split labels for training. Empty means all rows.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--text-max-chars", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None, help="Optional JSON config file")

    # Common overrides.
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
            "Number of nuisance cross-fit folds to train concurrently. 'auto' "
            "uses num_workers on CPU and stays serial on CUDA; an explicit "
            "integer opts into that many concurrent folds, including on CUDA."
        ),
    )
    parser.add_argument("--nuisance-epochs", type=int, default=None)
    parser.add_argument("--forest-epochs", type=int, default=None)
    parser.add_argument("--n-trees", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--forest-learning-rate", type=float, default=None)
    parser.add_argument("--nuisance-learning-rate", type=float, default=None)
    parser.add_argument("--lambda-heterogeneity", type=float, default=None)
    parser.add_argument("--attention-top-k", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-attention", action="store_true", help="Skip attention evidence export")
    return parser.parse_args()


def _bool_arg(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def _make_config(args: argparse.Namespace) -> NeuralCausalForestConfig:
    if args.config:
        config = NeuralCausalForestConfig.from_json(args.config)
    else:
        config = NeuralCausalForestConfig()
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
        "lambda_heterogeneity": args.lambda_heterogeneity,
        "attention_top_k": args.attention_top_k,
        "num_workers": args.num_workers,
    }
    for key, value in override_map.items():
        if value is not None:
            setattr(config, key, value)
    config.seed = int(args.seed)
    config.__post_init__()
    return config


def _prepare_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    df = read_dataframe(args.data)
    if args.max_rows is not None and args.max_rows < len(df):
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    if args.text_max_chars is not None:
        df = df.copy()
        df[args.text_column] = df[args.text_column].astype(str).str.slice(0, int(args.text_max_chars))
    if args.split_column in df.columns and args.train_split_values:
        df = df[df[args.split_column].astype(str).isin(set(args.train_split_values))].reset_index(drop=True)
    if args.row_id_column not in df.columns:
        df = df.reset_index(drop=True)
        df[args.row_id_column] = np.arange(len(df), dtype=int)
    return df


def _write_jsonl(rows: list[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _make_config(args)
    config.to_json(output_dir / "resolved_config.json")
    df = _prepare_dataframe(args)
    logger.info("Training neural causal forest on %d rows", len(df))
    device = torch.device(args.device)

    result = fit_neural_causal_forest_pipeline(
        df,
        text_column=args.text_column,
        treatment_column=args.treatment_column,
        outcome_column=args.outcome_column,
        outcome_type=args.outcome_type,
        config=config,
        device=device,
        row_id_column=args.row_id_column,
        collect_attention=not args.no_attention,
        nuisance_artifact_dir=output_dir,
    )

    model_dir = output_dir / "model"
    save_neural_causal_forest_model(
        result.model,
        model_dir,
        config=config,
        metadata={
            "text_column": args.text_column,
            "treatment_column": args.treatment_column,
            "outcome_column": args.outcome_column,
            "outcome_type": args.outcome_type,
            "row_id_column": args.row_id_column,
        },
    )

    write_dataframe(result.nuisance_predictions, output_dir / "nuisance_oof_predictions.parquet")
    write_dataframe(result.nuisance_history, output_dir / "nuisance_history.parquet")
    write_dataframe(result.forest_history, output_dir / "forest_history.parquet")
    write_dataframe(result.train_predictions, output_dir / "train_predictions.parquet")

    if not result.nuisance_attention.empty:
        nuisance_attention = add_oracle_attention_hits(result.nuisance_attention)
        write_dataframe(nuisance_attention, output_dir / "nuisance_attention_evidence.parquet")
        _write_jsonl(
            build_agent_context_rows(nuisance_attention, stage="nuisance", max_rows=100),
            output_dir / "agent_context_nuisance.jsonl",
        )
    if not result.effect_attention.empty:
        effect_attention = add_oracle_attention_hits(result.effect_attention)
        write_dataframe(effect_attention, output_dir / "effect_attention_evidence.parquet")
        _write_jsonl(
            build_agent_context_rows(effect_attention, stage="effect_modifier", max_rows=100),
            output_dir / "agent_context_effect_modifier.jsonl",
        )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, indent=2, sort_keys=True)
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump({"config": asdict(config), "n_rows": len(df)}, handle, indent=2, sort_keys=True)
    logger.info("Done. Wrote artifacts to %s", output_dir)


if __name__ == "__main__":
    main()
