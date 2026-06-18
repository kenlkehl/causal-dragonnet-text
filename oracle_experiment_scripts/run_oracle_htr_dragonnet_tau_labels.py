#!/usr/bin/env python
"""Train hierarchical-transformer DragonNet and save tau labels.

This is a focused oracle script for probing whether a DragonNet-style outcome
head can expose a useful treatment-effect target from unstructured text. It
trains cross-fitted hierarchical-transformer + DragonNet models, saves OOF
y0/y1/propensity predictions, and reports nuisance performance plus the
distribution of tau = y1 - y0.
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
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.models.dragonnet import DragonNet  # noqa: E402
from oci.models.extractor_factory import create_feature_extractor  # noqa: E402
from run_oracle_experiments import _resolve_parquet_file  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DATASET = (
    "synthetic_data/example_synthetic_datasets/"
    "one_confounder_one_effect_modifier_nsclc_with_structured"
)


class _TextOutcomeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, outcome_col: str, treatment_col: str):
        self.df = df.reset_index(drop=True)
        self.texts = self.df[text_col].astype(str).tolist()
        self.outcomes = self.df[outcome_col].to_numpy(dtype=np.float32)
        self.treatments = self.df[treatment_col].to_numpy(dtype=np.float32)
        self.row_ids = self.df["_oci_row_id"].to_numpy(dtype=int)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "text": self.texts[idx],
            "outcome": float(self.outcomes[idx]),
            "treatment": float(self.treatments[idx]),
            "row_id": int(self.row_ids[idx]),
        }


class _TextBatchCollator:
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [item["text"] for item in batch]
        result = self.preprocessor(texts) if self.preprocessor is not None else {"texts": texts}
        result["texts"] = texts
        result["outcome"] = torch.tensor([item["outcome"] for item in batch], dtype=torch.float32)
        result["treatment"] = torch.tensor([item["treatment"] for item in batch], dtype=torch.float32)
        result["row_id"] = np.asarray([item["row_id"] for item in batch], dtype=int)
        return result


class _HierarchicalDragonNet(nn.Module):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__()
        self._device = device
        self.feature_extractor = create_feature_extractor(
            extractor_type="hierarchical_transformer",
            device=device,
            model_type="dragonnet",
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
        self.net = DragonNet(
            input_dim=int(self.feature_extractor.output_dim),
            representation_dim=args.causal_head_representation_dim,
            hidden_outcome_dim=args.causal_head_hidden_outcome_dim,
            dropout=args.causal_head_dropout,
        )
        self.to(device)

    @staticmethod
    def _get_extractor_input(batch: Dict[str, Any], texts: List[str]) -> Any:
        if (
            "input_ids" in batch
            or "chunk_input_ids" in batch
            or "chunk_token_ids" in batch
        ):
            return batch
        return texts

    @staticmethod
    def _outcome_activation(logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logit)

    @staticmethod
    def _outcome_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logit, target)

    def fit_tokenizer(self, texts: Sequence[str]) -> None:
        if hasattr(self.feature_extractor, "fit_tokenizer"):
            self.feature_extractor.fit_tokenizer(texts)

    def _feature_extractor_anchor_loss(self) -> torch.Tensor:
        if hasattr(self.feature_extractor, "compute_anchor_loss"):
            return self.feature_extractor.compute_anchor_loss()
        return torch.tensor(0.0, device=self._device)

    def _regularization_losses(self) -> Dict[str, torch.Tensor]:
        if hasattr(self.feature_extractor, "compute_regularization_losses"):
            return self.feature_extractor.compute_regularization_losses()
        return {}

    def train_step(
        self,
        batch: Dict[str, Any],
        *,
        alpha_propensity: float,
        beta_targreg: float,
        label_smoothing: float,
        stop_grad_propensity: bool,
    ) -> Dict[str, torch.Tensor]:
        texts = batch["texts"]
        treatments = batch["treatment"]
        outcomes = batch["outcome"]
        extractor_input = self._get_extractor_input(batch, texts)

        if label_smoothing > 0:
            treatments_smooth = treatments * (1 - label_smoothing) + 0.5 * label_smoothing
            outcomes_smooth = outcomes * (1 - label_smoothing) + 0.5 * label_smoothing
        else:
            treatments_smooth = treatments
            outcomes_smooth = outcomes

        features = self.feature_extractor(extractor_input)
        if stop_grad_propensity:
            phi_detached = self.net.get_representation(features.detach())
            t_logit_for_loss = self.net.propensity_from_representation(phi_detached)
            y0_logit, y1_logit, t_logit, _phi = self.net(features)
        else:
            y0_logit, y1_logit, t_logit, _phi = self.net(features)
            t_logit_for_loss = t_logit

        propensity_loss = F.binary_cross_entropy_with_logits(
            t_logit_for_loss.squeeze(-1),
            treatments_smooth,
        )
        factual_logit = torch.where(treatments.unsqueeze(1) > 0.5, y1_logit, y0_logit)
        outcome_loss = self._outcome_loss(factual_logit.squeeze(-1), outcomes_smooth)

        if beta_targreg > 0:
            propensity = torch.sigmoid(t_logit).clamp(1e-3, 1 - 1e-3)
            H = (treatments.unsqueeze(1) / propensity) - (
                (1 - treatments.unsqueeze(1)) / (1 - propensity)
            )
            factual_prob = self._outcome_activation(factual_logit)
            moment = torch.mean((outcomes.unsqueeze(1) - factual_prob) * H)
            targreg_loss = moment ** 2
        else:
            targreg_loss = torch.tensor(0.0, device=self._device)

        anchor_loss = self._feature_extractor_anchor_loss()
        regularization_losses = self._regularization_losses()
        regularization_loss = (
            sum(regularization_losses.values())
            if regularization_losses
            else torch.tensor(0.0, device=self._device)
        )
        total_loss = (
            outcome_loss
            + alpha_propensity * propensity_loss
            + beta_targreg * targreg_loss
            + anchor_loss
            + regularization_loss
        )
        result = {
            "loss": total_loss,
            "outcome_loss": outcome_loss.detach(),
            "propensity_loss": propensity_loss.detach(),
            "targreg_loss": targreg_loss.detach(),
            "anchor_loss": anchor_loss.detach(),
            "regularization_loss": regularization_loss.detach(),
            "y0_logit": y0_logit.detach(),
            "y1_logit": y1_logit.detach(),
            "t_logit": t_logit.detach(),
        }
        for name, value in regularization_losses.items():
            result[name] = value.detach()
        return result

    def predict(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        texts = batch["texts"]
        extractor_input = self._get_extractor_input(batch, texts)
        features = self.feature_extractor(extractor_input)
        y0_logit, y1_logit, t_logit, final_common_layer = self.net(features)
        y0_prob = self._outcome_activation(y0_logit).squeeze(-1)
        y1_prob = self._outcome_activation(y1_logit).squeeze(-1)
        propensity = torch.sigmoid(t_logit).squeeze(-1)
        return {
            "y0_prob": y0_prob,
            "y1_prob": y1_prob,
            "propensity": propensity,
            "y0_logit": y0_logit.squeeze(-1),
            "y1_logit": y1_logit.squeeze(-1),
            "t_logit": t_logit.squeeze(-1),
            "final_common_layer": final_common_layer,
            "tau_pred": (y1_logit - y0_logit).squeeze(-1),
        }


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


def _safe_auc(y_true: pd.Series, y_score: pd.Series) -> Optional[float]:
    mask = pd.notna(y_true) & pd.notna(y_score)
    y = np.asarray(y_true[mask])
    score = np.asarray(y_score[mask])
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return None


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> Optional[float]:
    mask = pd.notna(x) & pd.notna(y)
    if int(mask.sum()) < 2:
        return None
    if method == "spearman":
        return _finite_or_none(stats.spearmanr(x[mask], y[mask]).correlation)
    return _finite_or_none(pd.Series(x[mask]).corr(pd.Series(y[mask])))


def _resolve_dataset(dataset: str, sample_size: Optional[int], seed: int, text_max_chars: Optional[int]) -> pd.DataFrame:
    parquet = _resolve_parquet_file(dataset)
    if parquet is None:
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    df = pd.read_parquet(parquet).reset_index(drop=True)
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    if text_max_chars is not None:
        df = df.copy()
        df["clinical_text_full_chars"] = df["clinical_text"].astype(str).str.len()
        df["clinical_text"] = df["clinical_text"].astype(str).str.slice(0, text_max_chars)
    df["_oci_row_id"] = np.arange(len(df), dtype=int)
    return df


def _make_loader(
    df: pd.DataFrame,
    model: _HierarchicalDragonNet,
    args: argparse.Namespace,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    preprocessor = None
    extractor = getattr(model, "feature_extractor", None)
    if extractor is not None and hasattr(extractor, "make_batch_preprocessor"):
        preprocessor = extractor.make_batch_preprocessor()
    return DataLoader(
        _TextOutcomeDataset(
            df,
            text_col=args.text_column,
            outcome_col=args.outcome_column,
            treatment_col=args.treatment_column,
        ),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_TextBatchCollator(preprocessor),
    )


def _add_scalar_tensors(totals: Dict[str, float], values: Dict[str, Any]) -> None:
    for key, value in values.items():
        if torch.is_tensor(value) and value.numel() == 1:
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())


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
    model = _HierarchicalDragonNet(args, device)
    model.fit_tokenizer(train_df[args.text_column].astype(str).tolist())

    train_loader = _make_loader(train_df, model, args, shuffle=True, device=device)
    test_loader = _make_loader(test_df, model, args, shuffle=False, device=device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_state = None
    best_val = float("inf")
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_totals: Dict[str, float] = {}
        train_batches = 0
        for batch in train_loader:
            batch["outcome"] = batch["outcome"].to(device, non_blocking=True)
            batch["treatment"] = batch["treatment"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = model.train_step(
                batch,
                alpha_propensity=args.alpha_propensity,
                beta_targreg=args.beta_targreg,
                label_smoothing=args.label_smoothing,
                stop_grad_propensity=args.stop_grad_propensity,
            )
            losses["loss"].backward()
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            train_batches += 1
            _add_scalar_tensors(train_totals, losses)
        scheduler.step()

        model.eval()
        val_totals: Dict[str, float] = {}
        val_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                batch["outcome"] = batch["outcome"].to(device, non_blocking=True)
                batch["treatment"] = batch["treatment"].to(device, non_blocking=True)
                losses = model.train_step(
                    batch,
                    alpha_propensity=args.alpha_propensity,
                    beta_targreg=args.beta_targreg,
                    label_smoothing=args.label_smoothing,
                    stop_grad_propensity=args.stop_grad_propensity,
                )
                val_batches += 1
                _add_scalar_tensors(val_totals, losses)

        row = {"fold": fold, "epoch": epoch, "lr": float(scheduler.get_last_lr()[0])}
        for key, value in train_totals.items():
            row[f"train_{key}"] = value / max(1, train_batches)
        for key, value in val_totals.items():
            row[f"val_{key}"] = value / max(1, val_batches)
        history.append(row)
        val_loss = row.get("val_loss", float("inf"))
        logger.info(
            "fold=%s epoch=%s train_loss=%.4f val_loss=%.4f targreg=%.6f",
            fold,
            epoch,
            row.get("train_loss", float("nan")),
            val_loss,
            row.get("val_targreg_loss", float("nan")),
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    pred_rows: List[pd.DataFrame] = []
    with torch.no_grad():
        for batch in test_loader:
            batch["outcome"] = batch["outcome"].to(device, non_blocking=True)
            batch["treatment"] = batch["treatment"].to(device, non_blocking=True)
            preds = model.predict(batch)
            pred_rows.append(
                pd.DataFrame(
                    {
                        "_oci_row_id": batch["row_id"],
                        "pred_y0_prob": preds["y0_prob"].detach().cpu().numpy(),
                        "pred_y1_prob": preds["y1_prob"].detach().cpu().numpy(),
                        "pred_propensity_prob": preds["propensity"].detach().cpu().numpy(),
                        "pred_y0_logit": preds["y0_logit"].detach().cpu().numpy(),
                        "pred_y1_logit": preds["y1_logit"].detach().cpu().numpy(),
                        "pred_t_logit": preds["t_logit"].detach().cpu().numpy(),
                    }
                )
            )
    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_df["cv_fold"] = fold
    pred_df["dragonnet_tau_prob"] = pred_df["pred_y1_prob"] - pred_df["pred_y0_prob"]
    pred_df["dragonnet_tau_logit"] = pred_df["pred_y1_logit"] - pred_df["pred_y0_logit"]
    pred_df = test_df.merge(pred_df, on="_oci_row_id", how="left")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred_df, history


def _metrics(results: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "n_samples": int(len(results)),
        "tau_mean": _finite_or_none(results["dragonnet_tau_prob"].mean()),
        "tau_std": _finite_or_none(results["dragonnet_tau_prob"].std()),
        "tau_min": _finite_or_none(results["dragonnet_tau_prob"].min()),
        "tau_max": _finite_or_none(results["dragonnet_tau_prob"].max()),
    }
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        metrics[f"tau_q{int(q * 100):02d}"] = _finite_or_none(results["dragonnet_tau_prob"].quantile(q))
    metrics["propensity_auroc"] = _safe_auc(results["treatment_indicator"], results["pred_propensity_prob"])
    factual = np.where(
        results["treatment_indicator"].to_numpy(dtype=float) > 0.5,
        results["pred_y1_prob"].to_numpy(dtype=float),
        results["pred_y0_prob"].to_numpy(dtype=float),
    )
    metrics["outcome_auroc"] = _safe_auc(results["outcome_indicator"], pd.Series(factual))
    if "true_ite_prob" in results.columns:
        metrics["true_ite_corr"] = _safe_corr(results["true_ite_prob"], results["dragonnet_tau_prob"])
        metrics["true_ite_spearman_corr"] = _safe_corr(
            results["true_ite_prob"],
            results["dragonnet_tau_prob"],
            method="spearman",
        )
        metrics["ite_mse"] = _finite_or_none(
            mean_squared_error(results["true_ite_prob"], results["dragonnet_tau_prob"])
        )
    if {"true_y0_prob", "true_y1_prob"}.issubset(results.columns):
        metrics["y0_mse"] = _finite_or_none(
            mean_squared_error(results["true_y0_prob"], results["pred_y0_prob"])
        )
        metrics["y1_mse"] = _finite_or_none(
            mean_squared_error(results["true_y1_prob"], results["pred_y1_prob"])
            )
    return metrics


def _write_pdl1_summary(results: pd.DataFrame, path: Path) -> None:
    if "true_pdl1_expression" not in results.columns:
        return
    agg_spec = {
        "n": ("_oci_row_id", "size"),
        "dragonnet_tau": ("dragonnet_tau_prob", "mean"),
        "dragonnet_tau_std": ("dragonnet_tau_prob", "std"),
    }
    if "true_ite_prob" in results.columns:
        agg_spec["true_ite"] = ("true_ite_prob", "mean")
    group = results.groupby("true_pdl1_expression").agg(**agg_spec)
    group.to_csv(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="../pcori_experiments/oracle_htr_dragonnet_tau_labels")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--text-max-chars", type=int, default=None)
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--treatment-column", default="treatment_indicator")

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

    parser.add_argument("--causal-head-representation-dim", type=int, default=128)
    parser.add_argument("--causal-head-hidden-outcome-dim", type=int, default=64)
    parser.add_argument("--causal-head-dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--alpha-propensity", type=float, default=1.0)
    parser.add_argument("--beta-targreg", type=float, default=10.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--stop-grad-propensity", type=_parse_bool, default=False)
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
        df = _resolve_dataset(args.dataset, args.sample_size, args.seed, args.text_max_chars)
        splits = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed).split(df)
        predictions = []
        histories = []
        fold_output_dir = output_dir / "fold_predictions"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            logger.info("DragonNet fold %s/%s train=%s test=%s", fold, args.n_folds, len(train_idx), len(test_idx))
            fold_preds, history = _train_one_fold(args, df, train_idx, test_idx, fold, device)
            predictions.append(fold_preds)
            histories.extend(history)
            fold_preds.to_parquet(
                fold_output_dir / f"fold_{fold:03d}_predictions.parquet",
                index=False,
            )
            partial_results = pd.concat(predictions, ignore_index=True).sort_values("_oci_row_id")
            partial_results.to_parquet(output_dir / "partial_dragonnet_oof_predictions.parquet", index=False)
            pd.DataFrame(histories).to_csv(output_dir / "training_history.csv", index=False)
            partial_metrics = _metrics(partial_results)
            partial_metrics["completed_folds"] = fold
            partial_metrics["requested_folds"] = args.n_folds
            partial_metrics["is_partial"] = fold < args.n_folds
            with open(output_dir / "partial_metrics.json", "w") as f:
                json.dump(partial_metrics, f, indent=2, default=str)
            _write_pdl1_summary(
                partial_results,
                output_dir / "partial_tau_by_true_pdl1_expression.csv",
            )
            logger.info(
                "Partial metrics after fold %s: tau_mean=%s tau_std=%s outcome_auroc=%s propensity_auroc=%s true_ite_corr=%s",
                fold,
                partial_metrics.get("tau_mean"),
                partial_metrics.get("tau_std"),
                partial_metrics.get("outcome_auroc"),
                partial_metrics.get("propensity_auroc"),
                partial_metrics.get("true_ite_corr"),
            )
        results = pd.concat(predictions, ignore_index=True).sort_values("_oci_row_id")
        results_path = output_dir / "dragonnet_oof_predictions.parquet"
        results.to_parquet(results_path, index=False)
        pd.DataFrame(histories).to_csv(output_dir / "training_history.csv", index=False)
        metrics = _metrics(results)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        _write_pdl1_summary(results, output_dir / "tau_by_true_pdl1_expression.csv")
        logger.info("Saved predictions to %s", results_path)
        logger.info("Metrics: %s", json.dumps(metrics, indent=2, default=str))
    except Exception as exc:
        logger.error("DragonNet tau-label run failed: %s\n%s", exc, traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
