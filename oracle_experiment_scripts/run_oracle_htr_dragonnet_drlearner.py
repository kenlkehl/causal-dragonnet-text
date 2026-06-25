#!/usr/bin/env python
"""Run an HTR DragonNet DR-learner oracle experiment.

This script trains nested cross-fitted DragonNet nuisance models, constructs
DR/AIPW pseudo-outcomes from out-of-fold nuisance predictions, and trains an
independent hierarchical-transformer tau model. It reports treatment/outcome
predictiveness and oracle ITE metrics when the dataset includes true effects.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.config import (  # noqa: E402
    AppliedInferenceConfig,
    DragonNetDRLearnerConfig,
    ModelArchitectureConfig,
    TrainingConfig,
)
from oci.inference.dragonnet_drlearner import run_dragonnet_drlearner  # noqa: E402
from run_oracle_experiments import _resolve_parquet_file, compute_metrics  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DATASET = (
    "synthetic_data/example_synthetic_datasets/"
    "one_confounder_one_effect_modifier_nsclc_with_structured"
)


def _parse_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="oracle_results/htr_dragonnet_drlearner")
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--cv-folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of outer CV folds to train concurrently.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="*",
        default=None,
        help="CUDA device ids to assign across parallel outer folds, e.g. --gpu-ids 0 1.",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=None,
        help="Per-training DataLoader workers. Default is 0 on CPU, 2 on GPU.",
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--nuisance-epochs", type=int, default=None)
    parser.add_argument("--effect-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--effect-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--alpha-propensity", type=float, default=1.0)
    parser.add_argument("--beta-targreg", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--nuisance-folds", type=int, default=2)
    parser.add_argument("--effect-folds", type=int, default=2)
    parser.add_argument("--nuisance-calibration", default="temperature_isotonic")
    parser.add_argument("--e-clip", type=float, default=0.01)
    parser.add_argument("--effect-loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--attention-top-k-chunks", type=int, default=5)

    parser.add_argument("--htr-sentence-model", default="prajjwal1/bert-tiny")
    parser.add_argument("--htr-freeze-sentence-encoder", type=_parse_bool, default=False)
    parser.add_argument("--htr-chunk-size-words", type=int, default=96)
    parser.add_argument("--htr-chunk-overlap-words", type=int, default=24)
    parser.add_argument("--htr-max-chunks", type=int, default=128)
    parser.add_argument("--htr-max-chunk-length", type=int, default=128)
    parser.add_argument("--htr-num-layers", type=int, default=2)
    parser.add_argument("--htr-num-heads", type=int, default=4)
    parser.add_argument("--htr-transformer-dim", type=int, default=256)
    parser.add_argument("--htr-dropout", type=float, default=0.1)
    parser.add_argument("--htr-projection-dim", type=int, default=128)
    parser.add_argument("--htr-hash-embedding-dim", type=int, default=256)
    parser.add_argument("--htr-sentence-encoder-batch-size", type=int, default=128)
    parser.add_argument("--htr-sentence-encoder-backend", default="auto")
    parser.add_argument("--htr-sentence-pooling", default="auto")
    parser.add_argument("--htr-normalize-sentence-embeddings", type=_parse_bool, default=True)
    parser.add_argument("--htr-trainable-sentence-encoder-layers", type=int, default=0)

    parser.add_argument("--causal-head-representation-dim", type=int, default=128)
    parser.add_argument("--causal-head-hidden-outcome-dim", type=int, default=64)
    parser.add_argument("--causal-head-dropout", type=float, default=0.2)
    return parser.parse_args()


def _load_dataset(dataset_arg: str) -> pd.DataFrame:
    path = Path(dataset_arg)
    if path.is_dir():
        resolved = _resolve_parquet_file(str(path))
        if resolved is None:
            raise FileNotFoundError(f"No dataset parquet found under {path}")
        path = resolved
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset file type: {path}")


def _build_config(args: argparse.Namespace, dataset_path: str) -> AppliedInferenceConfig:
    dr_config = DragonNetDRLearnerConfig(
        nuisance_folds=args.nuisance_folds,
        effect_folds=args.effect_folds,
        nuisance_epochs=args.nuisance_epochs,
        effect_epochs=args.effect_epochs,
        nuisance_calibration=args.nuisance_calibration,
        e_clip=args.e_clip,
        effect_loss=args.effect_loss,
        huber_beta=args.huber_beta,
        attention_top_k_chunks=args.attention_top_k_chunks,
    )
    arch = ModelArchitectureConfig(
        model_type="dragonnet_drlearner",
        feature_extractor_type="hierarchical_transformer",
        htr_sentence_model=args.htr_sentence_model,
        htr_freeze_sentence_encoder=args.htr_freeze_sentence_encoder,
        htr_chunk_size_words=args.htr_chunk_size_words,
        htr_chunk_overlap_words=args.htr_chunk_overlap_words,
        htr_max_chunks=args.htr_max_chunks,
        htr_max_chunk_length=args.htr_max_chunk_length,
        htr_num_layers=args.htr_num_layers,
        htr_num_heads=args.htr_num_heads,
        htr_transformer_dim=args.htr_transformer_dim,
        htr_dropout=args.htr_dropout,
        htr_projection_dim=args.htr_projection_dim,
        htr_hash_embedding_dim=args.htr_hash_embedding_dim,
        htr_sentence_encoder_batch_size=args.htr_sentence_encoder_batch_size,
        htr_sentence_encoder_backend=args.htr_sentence_encoder_backend,
        htr_sentence_pooling=args.htr_sentence_pooling,
        htr_normalize_sentence_embeddings=args.htr_normalize_sentence_embeddings,
        htr_trainable_sentence_encoder_layers=args.htr_trainable_sentence_encoder_layers,
        causal_head_representation_dim=args.causal_head_representation_dim,
        causal_head_hidden_outcome_dim=args.causal_head_hidden_outcome_dim,
        causal_head_dropout=args.causal_head_dropout,
        dragonnet_drlearner=dr_config,
    )
    training = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        effect_batch_size=args.effect_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        alpha_propensity=args.alpha_propensity,
        beta_targreg=args.beta_targreg,
        label_smoothing=args.label_smoothing,
        dataloader_workers=args.dataloader_workers,
    )
    return AppliedInferenceConfig(
        dataset_path=dataset_path,
        text_column=args.text_column,
        outcome_column=args.outcome_column,
        treatment_column=args.treatment_column,
        split_column=args.split_column,
        cv_folds=args.cv_folds,
        architecture=arch,
        training=training,
    )


def _oracle_metrics(
    results: pd.DataFrame,
    *,
    outcome_column: str,
    treatment_column: str,
) -> Dict[str, Any]:
    required = {
        "true_ite_prob",
        "true_y0_prob",
        "true_y1_prob",
        outcome_column,
        treatment_column,
    }
    metrics: Dict[str, Any] = {}
    if required.issubset(results.columns):
        metrics.update(
            compute_metrics(
                pred_ite=results["pred_ite_prob"].to_numpy(dtype=float),
                true_ite=results["true_ite_prob"].to_numpy(dtype=float),
                pred_propensity=results["pred_propensity_prob"].to_numpy(dtype=float),
                true_treatment=results[treatment_column].to_numpy(dtype=float),
                pred_y0=results["pred_y0_prob"].to_numpy(dtype=float),
                pred_y1=results["pred_y1_prob"].to_numpy(dtype=float),
                true_y0=results["true_y0_prob"].to_numpy(dtype=float),
                true_y1=results["true_y1_prob"].to_numpy(dtype=float),
                true_outcome=results[outcome_column].to_numpy(dtype=float),
            )
        )
        metrics["dragonnet_plugin_ite_mse"] = float(
            np.mean(
                np.square(
                    results["dragonnet_plugin_ite_prob"].to_numpy(dtype=float)
                    - results["true_ite_prob"].to_numpy(dtype=float)
                )
            )
        )
    metrics["pred_ite_mean"] = _finite_or_none(results["pred_ite_prob"].mean())
    metrics["pred_ite_std"] = _finite_or_none(results["pred_ite_prob"].std())
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_path = str(Path(args.dataset))
    dataset = _load_dataset(dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.parquet"
    config = _build_config(args, dataset_path)
    device = torch.device(args.device)
    try:
        run_dragonnet_drlearner(
            dataset=dataset,
            config=config,
            output_path=output_path,
            device=device,
            num_workers=args.num_workers,
            gpu_ids=args.gpu_ids,
        )
        results = pd.read_parquet(output_path)
        metrics = _oracle_metrics(
            results,
            outcome_column=args.outcome_column,
            treatment_column=args.treatment_column,
        )
        with open(output_dir / "oracle_metrics.json", "w") as handle:
            json.dump(metrics, handle, indent=2)
        logger.info("Oracle metrics saved to %s", output_dir / "oracle_metrics.json")
    except Exception as exc:
        logger.error("DragonNet DR-learner oracle run failed: %s", exc)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
