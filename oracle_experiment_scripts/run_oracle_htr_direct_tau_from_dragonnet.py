#!/usr/bin/env python
"""Train a hierarchical-transformer direct tau predictor from DragonNet labels.

This script consumes the OOF parquet produced by
``run_oracle_htr_dragonnet_tau_labels.py`` and trains a text-only tau regressor
to predict the DragonNet-derived label y1 - y0. It reports performance against
that label and, for oracle datasets, against true_ite_prob.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.models.extractor_factory import create_feature_extractor  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class _TauDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, label_col: str):
        self.df = df.reset_index(drop=True)
        self.texts = self.df[text_col].astype(str).tolist()
        self.labels = self.df[label_col].to_numpy(dtype=np.float32)
        self.row_ids = self.df["_oci_row_id"].to_numpy(dtype=int)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "text": self.texts[idx],
            "label": float(self.labels[idx]),
            "row_id": int(self.row_ids[idx]),
        }


class _TauBatchCollator:
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [item["text"] for item in batch]
        model_input = self.preprocessor(texts) if self.preprocessor is not None else {"texts": texts}
        model_input["texts"] = texts
        return {
            "model_input": model_input,
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
            "row_id": np.asarray([item["row_id"] for item in batch], dtype=int),
        }


class DirectTauNet(nn.Module):
    def __init__(self, extractor: nn.Module, hidden_dim: int, dropout: float):
        super().__init__()
        self.extractor = extractor
        self.head = nn.Sequential(
            nn.Linear(int(extractor.output_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, model_input):
        features = self.extractor(model_input)
        return self.head(features).squeeze(-1)


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


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> Optional[float]:
    mask = pd.notna(x) & pd.notna(y)
    if int(mask.sum()) < 2:
        return None
    if method == "spearman":
        return _finite_or_none(stats.spearmanr(x[mask], y[mask]).correlation)
    return _finite_or_none(pd.Series(x[mask]).corr(pd.Series(y[mask])))


def _create_extractor(args: argparse.Namespace, device: torch.device) -> nn.Module:
    return create_feature_extractor(
        extractor_type="hierarchical_transformer",
        device=device,
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
    )


def _make_loader(
    df: pd.DataFrame,
    model: DirectTauNet,
    args: argparse.Namespace,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    preprocessor = None
    if hasattr(model.extractor, "make_batch_preprocessor"):
        preprocessor = model.extractor.make_batch_preprocessor()
    return DataLoader(
        _TauDataset(df, args.text_column, args.label_column),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_TauBatchCollator(preprocessor),
    )


def _train_one_fold(
    args: argparse.Namespace,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    device: torch.device,
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    extractor = _create_extractor(args, device)
    if hasattr(extractor, "fit_tokenizer"):
        extractor.fit_tokenizer(train_df[args.text_column].astype(str).tolist())
    model = DirectTauNet(extractor, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    train_loader = _make_loader(train_df, model, args, shuffle=True, device=device)
    test_loader = _make_loader(test_df, model, args, shuffle=False, device=device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta) if args.loss == "huber" else nn.MSELoss()
    best_state = None
    best_val = float("inf")
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["model_input"])
            loss = loss_fn(pred, label)
            loss.backward()
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            train_batches += 1
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                label = batch["label"].to(device, non_blocking=True)
                pred = model(batch["model_input"])
                loss = loss_fn(pred, label)
                val_loss += float(loss.detach().cpu())
                val_batches += 1
        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_batches),
            "val_loss": val_loss / max(1, val_batches),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        logger.info(
            "fold=%s epoch=%s train_loss=%.5f val_loss=%.5f",
            fold,
            epoch,
            row["train_loss"],
            row["val_loss"],
        )
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    rows: List[pd.DataFrame] = []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch["model_input"]).detach().cpu().numpy()
            rows.append(pd.DataFrame({"_oci_row_id": batch["row_id"], "direct_tau_pred": pred}))
    pred_df = pd.concat(rows, ignore_index=True)
    pred_df["cv_fold"] = fold
    pred_df = test_df.merge(pred_df, on="_oci_row_id", how="left")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred_df, history


def _metrics(results: pd.DataFrame, label_col: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "n_samples": int(len(results)),
        "direct_tau_mean": _finite_or_none(results["direct_tau_pred"].mean()),
        "direct_tau_std": _finite_or_none(results["direct_tau_pred"].std()),
        "dragonnet_label_mean": _finite_or_none(results[label_col].mean()),
        "dragonnet_label_std": _finite_or_none(results[label_col].std()),
        "label_mse": _finite_or_none(mean_squared_error(results[label_col], results["direct_tau_pred"])),
        "label_mae": _finite_or_none(mean_absolute_error(results[label_col], results["direct_tau_pred"])),
        "label_corr": _safe_corr(results[label_col], results["direct_tau_pred"]),
        "label_spearman_corr": _safe_corr(results[label_col], results["direct_tau_pred"], method="spearman"),
    }
    if "true_ite_prob" in results.columns:
        metrics["true_ite_mse"] = _finite_or_none(
            mean_squared_error(results["true_ite_prob"], results["direct_tau_pred"])
        )
        metrics["true_ite_mae"] = _finite_or_none(
            mean_absolute_error(results["true_ite_prob"], results["direct_tau_pred"])
        )
        metrics["true_ite_corr"] = _safe_corr(results["true_ite_prob"], results["direct_tau_pred"])
        metrics["true_ite_spearman_corr"] = _safe_corr(
            results["true_ite_prob"],
            results["direct_tau_pred"],
            method="spearman",
        )
        metrics["dragonnet_label_true_ite_corr"] = _safe_corr(results["true_ite_prob"], results[label_col])
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        metrics[f"direct_tau_q{int(q * 100):02d}"] = _finite_or_none(results["direct_tau_pred"].quantile(q))
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dragonnet-predictions-path", required=True)
    parser.add_argument("--output-dir", default="../pcori_experiments/oracle_htr_direct_tau_from_dragonnet")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--label-column", default="dragonnet_tau_prob")

    parser.add_argument("--htr-sentence-model", default="prajjwal1/bert-tiny")
    parser.add_argument("--htr-freeze-sentence-encoder", type=_parse_bool, default=False)
    parser.add_argument("--htr-chunk-size-words", type=int, default=96)
    parser.add_argument("--htr-chunk-overlap-words", type=int, default=24)
    parser.add_argument("--htr-max-chunks", type=int, default=512)
    parser.add_argument("--htr-max-chunk-length", type=int, default=128)
    parser.add_argument("--htr-num-layers", type=int, default=2)
    parser.add_argument("--htr-num-heads", type=int, default=4)
    parser.add_argument("--htr-transformer-dim", type=int, default=256)
    parser.add_argument("--htr-projection-dim", type=int, default=128)
    parser.add_argument("--htr-hash-embedding-dim", type=int, default=256)
    parser.add_argument("--htr-sentence-encoder-batch-size", type=int, default=128)
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
    parser.add_argument("--htr-dropout", type=float, default=0.05)

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--huber-beta", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.n_folds < 2:
        raise SystemExit("--n-folds must be >= 2")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command_line.txt").write_text(" ".join(sys.argv) + "\n")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        df = pd.read_parquet(args.dragonnet_predictions_path).reset_index(drop=True)
        if "_oci_row_id" not in df.columns:
            df["_oci_row_id"] = np.arange(len(df), dtype=int)
        required = {args.text_column, args.label_column}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"DragonNet predictions file missing columns: {missing}")
        df = df[pd.notna(df[args.label_column])].reset_index(drop=True)
        splits = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed).split(df)
        predictions = []
        histories = []
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            logger.info("Direct tau fold %s/%s train=%s test=%s", fold, args.n_folds, len(train_idx), len(test_idx))
            fold_preds, history = _train_one_fold(args, df, train_idx, test_idx, fold, device)
            predictions.append(fold_preds)
            histories.extend(history)
        results = pd.concat(predictions, ignore_index=True).sort_values("_oci_row_id")
        predictions_path = output_dir / "direct_tau_predictions.parquet"
        results.to_parquet(predictions_path, index=False)
        pd.DataFrame(histories).to_csv(output_dir / "training_history.csv", index=False)
        metrics = _metrics(results, args.label_column)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        if "true_pdl1_expression" in results.columns:
            agg_spec = {
                "n": ("_oci_row_id", "size"),
                "dragonnet_tau": (args.label_column, "mean"),
                "direct_tau": ("direct_tau_pred", "mean"),
                "direct_tau_std": ("direct_tau_pred", "std"),
            }
            if "true_ite_prob" in results.columns:
                agg_spec["true_ite"] = ("true_ite_prob", "mean")
            group = results.groupby("true_pdl1_expression").agg(**agg_spec)
            group.to_csv(output_dir / "direct_tau_by_true_pdl1_expression.csv")
        logger.info("Saved predictions to %s", predictions_path)
        logger.info("Metrics: %s", json.dumps(metrics, indent=2, default=str))
    except Exception as exc:
        logger.error("Direct tau run failed: %s\n%s", exc, traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
