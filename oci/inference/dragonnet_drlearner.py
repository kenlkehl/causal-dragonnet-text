"""DragonNet nuisance model plus doubly-robust pseudo-outcome learner."""

from __future__ import annotations

import copy
import gc
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from ..config import (
    AppliedInferenceConfig,
    normalize_feature_extractor_type,
)
from ..models.causal_text import CausalText
from ..models.extractor_factory import create_feature_extractor_from_config
from ..utils import cuda_cleanup, get_memory_info
from ..utils.calibration import (
    BinaryProbabilityCalibrator,
    binary_calibration_metrics,
)
from .applied import _predict_dataset, _train_single_model


logger = logging.getLogger(__name__)


def dr_pseudo_outcome(
    outcome: Any,
    treatment: Any,
    mu0: Any,
    mu1: Any,
    propensity: Any,
    *,
    e_clip: float = 0.01,
) -> np.ndarray:
    """AIPW/DR pseudo-outcome for CATE on the outcome scale."""
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment, dtype=float)
    m0 = np.asarray(mu0, dtype=float)
    m1 = np.asarray(mu1, dtype=float)
    e = np.clip(np.asarray(propensity, dtype=float), float(e_clip), 1.0 - float(e_clip))
    pseudo = (m1 - m0) + t * (y - m1) / e - (1.0 - t) * (y - m0) / (1.0 - e)
    return np.where(np.isfinite(pseudo), pseudo, np.nan)


def run_dragonnet_drlearner(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    num_workers: int = 1,
    gpu_ids: Optional[Sequence[int]] = None,
) -> None:
    """Run DragonNet DR-learner inference and save predictions/artifacts."""
    runner = DragonNetDRLearnerRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        device=device,
        num_workers=num_workers,
        gpu_ids=gpu_ids,
    )
    runner.run()


class _TextPositionDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        positions: Sequence[int],
        fields: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.texts = [str(text or "") for text in texts]
        self.positions = np.asarray(positions, dtype=int)
        self.fields = {
            name: np.asarray(values)
            for name, values in (fields or {}).items()
        }

    def __len__(self) -> int:
        return int(len(self.positions))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        position = int(self.positions[index])
        item: Dict[str, Any] = {
            "position": position,
            "text": self.texts[position],
        }
        for name, values in self.fields.items():
            item[name] = float(values[position])
        return item


class _TextPositionCollator:
    def __init__(self, text_preprocessor: Optional[Any] = None):
        self.text_preprocessor = text_preprocessor

    def __call__(self, items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [str(item["text"]) for item in items]
        batch: Dict[str, Any] = {
            "model_input": (
                self.text_preprocessor(texts)
                if self.text_preprocessor is not None
                else texts
            ),
            "position": torch.as_tensor(
                [int(item["position"]) for item in items],
                dtype=torch.long,
            ),
        }
        if not items:
            return batch
        for name in items[0]:
            if name in {"position", "text"}:
                continue
            batch[name] = torch.as_tensor(
                [float(item[name]) for item in items],
                dtype=torch.float32,
            )
        return batch


class _DirectTauNet(nn.Module):
    def __init__(self, extractor: nn.Module, hidden_dim: int, dropout: float):
        super().__init__()
        self.extractor = extractor
        self.head = nn.Sequential(
            nn.Linear(extractor.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, texts_or_batch):
        features = self.extractor(
            texts_or_batch if isinstance(texts_or_batch, dict) else list(texts_or_batch)
        )
        return self.head(features).squeeze(-1)


class DragonNetDRLearnerRunner:
    """Cross-fitted DragonNet nuisance extraction and DR effect learning."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        config: AppliedInferenceConfig,
        output_path: Path,
        device: torch.device,
        num_workers: int = 1,
        gpu_ids: Optional[Sequence[int]] = None,
    ):
        self.dataset = _ensure_row_ids(dataset).reset_index(drop=True)
        self.config = config
        self.dr_config = config.architecture.dragonnet_drlearner
        self.output_path = Path(output_path)
        self.device = device
        self.num_workers = int(num_workers)
        self.gpu_ids = list(gpu_ids or [])
        self.artifact_dir = self.output_path.parent / "dragonnet_drlearner"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.nuisance_rows: List[pd.DataFrame] = []
        self.effect_rows: List[pd.DataFrame] = []
        self.nuisance_attention_rows: List[Dict[str, Any]] = []
        self.effect_attention_rows: List[Dict[str, Any]] = []
        self.training_rows: List[Dict[str, Any]] = []

    def run(self) -> None:
        logger.info("=" * 80)
        logger.info("DRAGONNET DR-LEARNER")
        logger.info("=" * 80)
        splits = self._analysis_splits()
        n_jobs = self._outer_n_jobs(splits)
        if n_jobs > 1:
            devices = self._parallel_devices()
            worker_config = self._parallel_worker_config()
            logger.info(
                "Parallelizing DragonNet DR-learner outer folds: folds=%s n_jobs=%s devices=%s",
                len(splits),
                n_jobs,
                [str(device) for device in devices],
            )
            fold_results = Parallel(n_jobs=n_jobs)(
                delayed(_run_dragonnet_drlearner_outer_fold)(
                    dataset=self.dataset,
                    config=worker_config,
                    output_path=self.output_path,
                    device=str(devices[(idx % len(devices))]),
                    num_workers=1,
                    gpu_ids=None,
                    outer_fold=outer_fold,
                    train_idx=train_idx,
                    test_idx=test_idx,
                )
                for idx, (outer_fold, train_idx, test_idx) in enumerate(splits)
            )
            result_frames = []
            for result in fold_results:
                result_frames.append(result["predictions"])
                self.nuisance_rows.extend(result["nuisance_rows"])
                self.effect_rows.extend(result["effect_rows"])
                self.nuisance_attention_rows.extend(result["nuisance_attention_rows"])
                self.effect_attention_rows.extend(result["effect_attention_rows"])
                self.training_rows.extend(result["training_rows"])
            results_df = pd.concat(result_frames, ignore_index=True)
            self._save_predictions(results_df)
            self._save_artifacts(results_df)
            return

        result_frames: List[pd.DataFrame] = []
        for outer_fold, train_idx, test_idx in splits:
            result_frames.append(self._run_one_analysis_split(outer_fold, train_idx, test_idx))

        results_df = pd.concat(result_frames, ignore_index=True)
        self._save_predictions(results_df)
        self._save_artifacts(results_df)

    def _run_one_analysis_split(
        self,
        outer_fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> pd.DataFrame:
        logger.info(
            "Outer fold %s: discovery=%s heldout=%s device=%s%s",
            outer_fold,
            len(train_idx),
            len(test_idx),
            self.device,
            self._memory_summary(),
        )
        discovery_df = self.dataset.iloc[train_idx].reset_index(drop=True)
        test_df = self.dataset.iloc[test_idx].reset_index(drop=True)
        nuisance = self.crossfit_nuisance(discovery_df, outer_fold)
        self.nuisance_rows.append(nuisance["predictions"])
        self.nuisance_attention_rows.extend(nuisance["attention"])
        self.training_rows.extend(nuisance.get("history", []))

        effect = self.crossfit_effect(discovery_df, nuisance["predictions"], outer_fold)
        self.effect_rows.append(effect["predictions"])
        self.effect_attention_rows.extend(effect["attention"])
        self.training_rows.extend(effect.get("history", []))

        fold_predictions = self._fit_final_models_and_predict(
            discovery_df=discovery_df,
            test_df=test_df,
            nuisance_predictions=nuisance["predictions"],
            outer_fold=outer_fold,
        )
        cuda_cleanup()
        logger.info("Outer fold %s complete%s", outer_fold, self._memory_summary())
        return fold_predictions

    def crossfit_nuisance(self, df: pd.DataFrame, outer_fold: int) -> Dict[str, Any]:
        """Train DragonNet nuisances and produce OOF nuisance predictions."""
        df = _ensure_row_ids(df).reset_index(drop=True)
        folds = _bounded_fold_count(self.dr_config.nuisance_folds, len(df))
        predictions = pd.DataFrame(
            {
                "_oci_row_id": df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "pred_y0_prob": np.nan,
                "pred_y1_prob": np.nan,
                "pred_propensity_prob": np.nan,
                "pred_y0_raw": np.nan,
                "pred_y1_raw": np.nan,
                "pred_propensity_raw": np.nan,
                "dragonnet_plugin_ite_prob": np.nan,
                "e_hat": np.nan,
                "m_hat": np.nan,
                "y_residual": np.nan,
                "t_residual": np.nan,
                "dr_pseudo_outcome": np.nan,
                "r_pseudo_outcome": np.nan,
                "nuisance_fold": np.nan,
                "neural_stage_mode": "dragonnet_dr",
            }
        )
        attention_rows: List[Dict[str, Any]] = []
        history_rows: List[Dict[str, Any]] = []

        split_iter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=31_000 + int(outer_fold),
        ).split(df)
        for fold, (fit_pos, heldout_pos) in enumerate(split_iter, start=1):
            logger.info(
                "Outer fold %s DragonNet nuisance fold %s/%s: train=%s heldout=%s%s",
                outer_fold,
                fold,
                folds,
                len(fit_pos),
                len(heldout_pos),
                self._memory_summary(),
            )
            nuisance_config = self._nuisance_config()
            model, history = self._train_dragonnet_nuisance(
                train_df=df.iloc[fit_pos].reset_index(drop=True),
                val_df=df.iloc[heldout_pos].reset_index(drop=True),
                nuisance_config=nuisance_config,
            )
            for entry in history:
                entry = dict(entry)
                entry.update(
                    {
                        "outer_fold": int(outer_fold),
                        "fold": int(fold),
                        "stage": "nuisance",
                    }
                )
                history_rows.append(entry)

            fit_raw = self._predict_dragonnet_nuisance(
                model,
                df.iloc[fit_pos].reset_index(drop=True),
                nuisance_config,
            )
            heldout_df = df.iloc[heldout_pos].reset_index(drop=True)
            heldout_raw = self._predict_dragonnet_nuisance(model, heldout_df, nuisance_config)
            calibrators = self._fit_nuisance_calibrators(
                df.iloc[fit_pos].reset_index(drop=True),
                fit_raw,
            )
            heldout_cal = self._apply_nuisance_calibrators(heldout_raw, calibrators)
            y = heldout_df[self.config.outcome_column].to_numpy(dtype=float)
            t = heldout_df[self.config.treatment_column].to_numpy(dtype=float)
            m_hat = heldout_cal["propensity"] * heldout_cal["y1"] + (
                1.0 - heldout_cal["propensity"]
            ) * heldout_cal["y0"]
            y_resid = y - m_hat
            t_resid = t - heldout_cal["propensity"]
            pseudo = dr_pseudo_outcome(
                y,
                t,
                heldout_cal["y0"],
                heldout_cal["y1"],
                heldout_cal["propensity"],
                e_clip=self.dr_config.e_clip,
            )
            predictions.loc[heldout_pos, "pred_y0_prob"] = heldout_cal["y0"]
            predictions.loc[heldout_pos, "pred_y1_prob"] = heldout_cal["y1"]
            predictions.loc[heldout_pos, "pred_propensity_prob"] = heldout_cal["propensity"]
            predictions.loc[heldout_pos, "pred_y0_raw"] = heldout_raw["y0_prob"]
            predictions.loc[heldout_pos, "pred_y1_raw"] = heldout_raw["y1_prob"]
            predictions.loc[heldout_pos, "pred_propensity_raw"] = heldout_raw["propensity_prob"]
            predictions.loc[heldout_pos, "dragonnet_plugin_ite_prob"] = (
                heldout_cal["y1"] - heldout_cal["y0"]
            )
            predictions.loc[heldout_pos, "e_hat"] = heldout_cal["propensity"]
            predictions.loc[heldout_pos, "m_hat"] = m_hat
            predictions.loc[heldout_pos, "y_residual"] = y_resid
            predictions.loc[heldout_pos, "t_residual"] = t_resid
            predictions.loc[heldout_pos, "dr_pseudo_outcome"] = pseudo
            predictions.loc[heldout_pos, "r_pseudo_outcome"] = pseudo
            predictions.loc[heldout_pos, "nuisance_fold"] = int(fold)

            attention = self._attention_evidence(
                extractor=model.feature_extractor,
                df=heldout_df,
                fold=fold,
                outer_fold=outer_fold,
                stage="nuisance",
                extra={
                    "pred_y0_prob": heldout_cal["y0"],
                    "pred_y1_prob": heldout_cal["y1"],
                    "pred_propensity_prob": heldout_cal["propensity"],
                    "dragonnet_plugin_ite_prob": heldout_cal["y1"] - heldout_cal["y0"],
                    "dr_pseudo_outcome": pseudo,
                    "y_residual": y_resid,
                    "t_residual": t_resid,
                    "neural_stage_mode": np.asarray(["dragonnet_dr"] * len(heldout_df), dtype=object),
                },
            )
            attention_rows.extend(attention)
            self._cleanup_model(model)
            del model
            gc.collect()
            logger.info(
                "Outer fold %s DragonNet nuisance fold %s/%s complete: "
                "pseudo_mean=%.4f propensity_auroc=%s outcome_auroc=%s%s",
                outer_fold,
                fold,
                folds,
                float(np.nanmean(pseudo)),
                _format_optional_metric(_safe_roc_auc(t, heldout_cal["propensity"])),
                _format_optional_metric(
                    _safe_roc_auc(
                        y,
                        np.where(t >= 0.5, heldout_cal["y1"], heldout_cal["y0"]),
                    )
                    if self.config.outcome_type != "continuous"
                    else None
                ),
                self._memory_summary(),
            )

        return {
            "predictions": predictions,
            "attention": attention_rows,
            "history": history_rows,
        }

    def crossfit_effect(
        self,
        df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        """Train an independent direct tau network on OOF DR pseudo-outcomes."""
        df = _ensure_row_ids(df).reset_index(drop=True)
        folds = _bounded_fold_count(self.dr_config.effect_folds, len(df))
        r_df = nuisance_predictions.copy()
        pseudo = r_df["dr_pseudo_outcome"].to_numpy(dtype=float)
        r_df["tau_hat_r_stage"] = np.nan
        r_df["pred_ite_prob"] = np.nan
        r_df["r_loss"] = np.nan
        r_df["effect_loss"] = np.nan
        r_df["effect_loss_at_zero_tau"] = pseudo**2
        r_df["effect_fold"] = np.nan
        r_df["effect_objective"] = f"dr_pseudo_outcome_{self.dr_config.effect_loss}"
        r_df["neural_stage_mode"] = "dragonnet_dr"

        attention_rows: List[Dict[str, Any]] = []
        history_rows: List[Dict[str, Any]] = []
        split_iter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=32_000 + int(outer_fold),
        ).split(df)
        for fold, (fit_pos, heldout_pos) in enumerate(split_iter, start=1):
            eligible_fit_pos = np.asarray(
                [pos for pos in fit_pos if np.isfinite(pseudo[int(pos)])],
                dtype=int,
            )
            if len(eligible_fit_pos) < 1:
                raise ValueError(
                    "No finite DR pseudo-outcomes for effect-stage training in "
                    f"outer fold {outer_fold}, effect fold {fold}"
                )
            logger.info(
                "Outer fold %s DR effect fold %s/%s: train=%s heldout=%s%s",
                outer_fold,
                fold,
                folds,
                len(eligible_fit_pos),
                len(heldout_pos),
                self._memory_summary(),
            )
            model = self._create_tau_model()
            history = self._train_tau_model(
                model=model,
                df=df,
                positions=eligible_fit_pos,
                targets=pseudo,
                outer_fold=outer_fold,
                fold=fold,
                total_folds=folds,
            )
            history_rows.extend(history)
            heldout_df = df.iloc[heldout_pos].reset_index(drop=True)
            tau_hat = self._predict_tau_model(model, heldout_df)
            heldout_pseudo = pseudo[heldout_pos]
            effect_loss = (tau_hat - heldout_pseudo) ** 2
            r_df.loc[heldout_pos, "tau_hat_r_stage"] = tau_hat
            r_df.loc[heldout_pos, "pred_ite_prob"] = tau_hat
            r_df.loc[heldout_pos, "r_loss"] = effect_loss
            r_df.loc[heldout_pos, "effect_loss"] = effect_loss
            r_df.loc[heldout_pos, "effect_fold"] = int(fold)
            attention = self._attention_evidence(
                extractor=model.extractor,
                df=heldout_df,
                fold=fold,
                outer_fold=outer_fold,
                stage="effect_modifier",
                extra={
                    "tau_hat_r_stage": tau_hat,
                    "pred_ite_prob": tau_hat,
                    "dr_pseudo_outcome": heldout_pseudo,
                    "r_pseudo_outcome": heldout_pseudo,
                    "r_loss": effect_loss,
                    "effect_loss": effect_loss,
                    "effect_objective": np.asarray(
                        [f"dr_pseudo_outcome_{self.dr_config.effect_loss}"] * len(heldout_df),
                        dtype=object,
                    ),
                    "neural_stage_mode": np.asarray(["dragonnet_dr"] * len(heldout_df), dtype=object),
                },
            )
            attention_rows.extend(attention)
            self._cleanup_model(model)
            del model
            gc.collect()
            logger.info(
                "Outer fold %s DR effect fold %s/%s complete: tau_mean=%.4f loss=%.4f%s",
                outer_fold,
                fold,
                folds,
                float(np.nanmean(tau_hat)),
                float(np.nanmean(effect_loss)),
                self._memory_summary(),
            )
        return {
            "predictions": r_df,
            "attention": attention_rows,
            "history": history_rows,
        }

    def _fit_final_models_and_predict(
        self,
        discovery_df: pd.DataFrame,
        test_df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
    ) -> pd.DataFrame:
        nuisance_config = self._nuisance_config()
        fit_df, val_df = _train_validation_split(discovery_df)
        nuisance_model, history = self._train_dragonnet_nuisance(
            train_df=fit_df,
            val_df=val_df,
            nuisance_config=nuisance_config,
        )
        for entry in history:
            entry = dict(entry)
            entry.update({"outer_fold": int(outer_fold), "fold": 0, "stage": "final_nuisance"})
            self.training_rows.append(entry)
        discovery_raw = self._predict_dragonnet_nuisance(
            nuisance_model,
            discovery_df.reset_index(drop=True),
            nuisance_config,
        )
        test_raw = self._predict_dragonnet_nuisance(
            nuisance_model,
            test_df.reset_index(drop=True),
            nuisance_config,
        )
        calibrators = self._fit_nuisance_calibrators(discovery_df, discovery_raw)
        test_cal = self._apply_nuisance_calibrators(test_raw, calibrators)
        self._cleanup_model(nuisance_model)
        del nuisance_model

        pseudo = nuisance_predictions["dr_pseudo_outcome"].to_numpy(dtype=float)
        eligible = np.flatnonzero(np.isfinite(pseudo))
        if len(eligible) < 1:
            raise ValueError("No finite DR pseudo-outcomes for final effect model")
        tau_model = self._create_tau_model()
        history = self._train_tau_model(
            model=tau_model,
            df=discovery_df.reset_index(drop=True),
            positions=eligible,
            targets=pseudo,
            outer_fold=outer_fold,
            fold=0,
            total_folds=1,
            stage="final_effect",
        )
        self.training_rows.extend(history)
        tau_hat = self._predict_tau_model(tau_model, test_df.reset_index(drop=True))
        self._cleanup_model(tau_model)
        del tau_model

        result = test_df.copy()
        result["pred_y0_prob"] = test_cal["y0"]
        result["pred_y1_prob"] = test_cal["y1"]
        result["pred_propensity_prob"] = test_cal["propensity"]
        result["pred_y0_raw"] = test_raw["y0_prob"]
        result["pred_y1_raw"] = test_raw["y1_prob"]
        result["pred_propensity_raw"] = test_raw["propensity_prob"]
        result["dragonnet_plugin_ite_prob"] = test_cal["y1"] - test_cal["y0"]
        result["pred_ite_prob"] = tau_hat
        result["tau_hat_drlearner"] = tau_hat
        result["cv_fold"] = int(outer_fold)
        result["model_type"] = "dragonnet_drlearner"
        return result

    def _train_dragonnet_nuisance(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        nuisance_config: AppliedInferenceConfig,
    ) -> Tuple[CausalText, List[Dict[str, Any]]]:
        return _train_single_model(
            train_df=train_df,
            val_df=val_df,
            config=nuisance_config,
            device=self.device,
        )

    def _predict_dragonnet_nuisance(
        self,
        model: CausalText,
        df: pd.DataFrame,
        nuisance_config: AppliedInferenceConfig,
    ) -> Dict[str, np.ndarray]:
        return _predict_dataset(
            model=model,
            df=df,
            config=nuisance_config,
            device=self.device,
        )

    def _fit_nuisance_calibrators(
        self,
        fit_df: pd.DataFrame,
        raw: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        t = fit_df[self.config.treatment_column].to_numpy(dtype=float)
        y = fit_df[self.config.outcome_column].to_numpy(dtype=float)
        prop = BinaryProbabilityCalibrator.fit(
            raw["propensity_prob"],
            t,
            method=self.dr_config.nuisance_calibration,
        )
        calibrators: Dict[str, Any] = {"propensity": prop}
        if self.config.outcome_type == "continuous":
            calibrators["y0"] = None
            calibrators["y1"] = None
            return calibrators
        control = t < 0.5
        treated = t >= 0.5
        calibrators["y0"] = BinaryProbabilityCalibrator.fit(
            np.asarray(raw["y0_prob"])[control],
            y[control],
            method=self.dr_config.nuisance_calibration,
        )
        calibrators["y1"] = BinaryProbabilityCalibrator.fit(
            np.asarray(raw["y1_prob"])[treated],
            y[treated],
            method=self.dr_config.nuisance_calibration,
        )
        return calibrators

    def _apply_nuisance_calibrators(
        self,
        raw: Dict[str, np.ndarray],
        calibrators: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        propensity = calibrators["propensity"].transform(raw["propensity_prob"])
        if self.config.outcome_type == "continuous":
            y0 = np.asarray(raw["y0_prob"], dtype=float)
            y1 = np.asarray(raw["y1_prob"], dtype=float)
        else:
            y0 = calibrators["y0"].transform(raw["y0_prob"])
            y1 = calibrators["y1"].transform(raw["y1_prob"])
        return {"y0": y0, "y1": y1, "propensity": propensity}

    def _create_tau_model(self) -> _DirectTauNet:
        arch = self.config.architecture
        extractor_config = asdict(arch)
        extractor_config["feature_extractor_type"] = self._effective_feature_extractor_type()
        extractor = create_feature_extractor_from_config(
            extractor_config,
            self.device,
            model_type="dragonnet_drlearner",
        )
        return _DirectTauNet(
            extractor=extractor,
            hidden_dim=getattr(arch, "causal_head_hidden_outcome_dim", 64),
            dropout=getattr(arch, "causal_head_dropout", 0.2),
        ).to(self.device)

    def _train_tau_model(
        self,
        model: _DirectTauNet,
        df: pd.DataFrame,
        positions: Sequence[int],
        targets: np.ndarray,
        outer_fold: int,
        fold: int,
        total_folds: int,
        stage: str = "effect",
    ) -> List[Dict[str, Any]]:
        positions = np.asarray(positions, dtype=int)
        if hasattr(model.extractor, "fit_tokenizer"):
            model.extractor.fit_tokenizer(df.iloc[positions][self.config.text_column].tolist())
        train_config = self.config.training
        epochs = int(
            self.dr_config.effect_epochs
            if self.dr_config.effect_epochs is not None
            else train_config.epochs
        )
        loader = self._make_text_loader(
            model=model,
            df=df,
            positions=positions,
            fields={"target": np.asarray(targets, dtype=np.float32)},
            shuffle=True,
            batch_size=getattr(train_config, "effect_batch_size", None),
        )
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=train_config.learning_rate,
            weight_decay=getattr(train_config, "weight_decay", 0.01),
        )
        scheduler = _make_linear_lr_scheduler(optimizer, train_config, len(loader), epochs)
        history: List[Dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            batch_count = 0
            for batch in loader:
                target = batch["target"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                pred = model(batch["model_input"])
                if self.dr_config.effect_loss == "mse":
                    loss = F.mse_loss(pred, target)
                else:
                    loss = F.smooth_l1_loss(
                        pred,
                        target,
                        beta=float(self.dr_config.huber_beta),
                    )
                loss.backward()
                grad_clip = getattr(train_config, "gradient_clip_norm", 0.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                loss_sum += float(loss.detach().cpu())
                batch_count += 1
            mean_loss = loss_sum / max(1, batch_count)
            history.append(
                {
                    "outer_fold": int(outer_fold),
                    "fold": int(fold),
                    "stage": stage,
                    "epoch": int(epoch),
                    "train_loss": mean_loss,
                    "loss": mean_loss,
                    "total_folds": int(total_folds),
                    "effect_loss": self.dr_config.effect_loss,
                }
            )
            logger.info(
                "Outer fold %s DR %s fold %s/%s epoch %s/%s loss=%.4f%s",
                outer_fold,
                stage,
                fold,
                total_folds,
                epoch,
                epochs,
                mean_loss,
                self._memory_summary(),
            )
        return history

    def _predict_tau_model(self, model: _DirectTauNet, df: pd.DataFrame) -> np.ndarray:
        model.eval()
        values: List[np.ndarray] = []
        loader = self._make_text_loader(
            model=model,
            df=df,
            positions=np.arange(len(df), dtype=int),
            shuffle=False,
            batch_size=getattr(self.config.training, "effect_batch_size", None),
        )
        with torch.no_grad():
            for batch in loader:
                values.append(model(batch["model_input"]).cpu().numpy())
        if not values:
            return np.zeros(0, dtype=float)
        return np.concatenate(values)

    def _make_text_loader(
        self,
        model: nn.Module,
        df: pd.DataFrame,
        positions: Sequence[int],
        *,
        fields: Optional[Dict[str, np.ndarray]] = None,
        shuffle: bool = False,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        extractor = getattr(model, "extractor", None)
        text_preprocessor = None
        if extractor is not None and hasattr(extractor, "make_batch_preprocessor"):
            text_preprocessor = extractor.make_batch_preprocessor()
        effective_batch_size = batch_size or self.config.training.batch_size
        return DataLoader(
            _TextPositionDataset(
                texts=df[self.config.text_column].astype(str).tolist(),
                positions=positions,
                fields=fields,
            ),
            batch_size=max(1, int(effective_batch_size)),
            shuffle=bool(shuffle),
            collate_fn=_TextPositionCollator(text_preprocessor),
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

    def _attention_evidence(
        self,
        extractor: nn.Module,
        df: pd.DataFrame,
        fold: int,
        outer_fold: int,
        stage: str,
        extra: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        if not hasattr(extractor, "get_attention_evidence"):
            return []
        texts = df[self.config.text_column].astype(str).tolist()
        row_ids = df["_oci_row_id"].tolist()
        metadata: List[Dict[str, Any]] = []
        for offset in range(len(df)):
            item: Dict[str, Any] = {"outer_fold": int(outer_fold)}
            for key, values in extra.items():
                item[key] = _metadata_value(values, offset)
            metadata.append(item)
        records: List[Dict[str, Any]] = []
        batch_size = max(1, int(self.config.training.batch_size))
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            records.extend(
                extractor.get_attention_evidence(
                    texts[start:end],
                    row_ids=row_ids[start:end],
                    fold=fold,
                    stage=stage,
                    top_k=self.dr_config.attention_top_k_chunks,
                    metadata=metadata[start:end],
                )
            )
        return records

    def _nuisance_config(self) -> AppliedInferenceConfig:
        cfg = copy.deepcopy(self.config)
        cfg.architecture.model_type = "dragonnet"
        cfg.architecture.feature_extractor_type = self._effective_feature_extractor_type()
        if self.dr_config.nuisance_epochs is not None:
            cfg.training.epochs = int(self.dr_config.nuisance_epochs)
        return cfg

    def _effective_feature_extractor_type(self) -> str:
        extractor_type = normalize_feature_extractor_type(
            getattr(self.config.architecture, "feature_extractor_type", "hierarchical_transformer")
        )
        if extractor_type == "frozen_llm_pooler":
            return "hierarchical_transformer"
        return extractor_type

    def _analysis_splits(self) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        dataset = self.dataset.reset_index(drop=True)
        if self.config.cv_folds and int(self.config.cv_folds) > 1:
            folds = _bounded_fold_count(int(self.config.cv_folds), len(dataset))
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)
            return [
                (fold, train_idx, test_idx)
                for fold, (train_idx, test_idx) in enumerate(kf.split(dataset), start=1)
            ]
        split_column = getattr(self.config, "split_column", "split")
        if split_column in dataset.columns and "test" in set(dataset[split_column].astype(str)):
            train_mask = dataset[split_column].astype(str).isin(["train", "val"])
            test_mask = dataset[split_column].astype(str) == "test"
            return [
                (
                    1,
                    np.flatnonzero(train_mask.to_numpy()),
                    np.flatnonzero(test_mask.to_numpy()),
                )
            ]
        logger.warning(
            "No CV or fixed test split configured for dragonnet_drlearner; "
            "using all rows for both discovery and prediction."
        )
        all_idx = np.arange(len(dataset), dtype=int)
        return [(1, all_idx, all_idx)]

    def _parallel_devices(self) -> List[torch.device]:
        if self.gpu_ids and self.device.type == "cuda":
            return [torch.device(f"cuda:{int(gpu_id)}") for gpu_id in self.gpu_ids]
        return [self.device]

    def _outer_n_jobs(self, splits: Sequence[Tuple[int, np.ndarray, np.ndarray]]) -> int:
        requested = max(1, int(self.num_workers))
        if requested <= 1 or len(splits) <= 1:
            return 1
        devices = self._parallel_devices()
        if self.device.type == "cuda":
            requested = min(requested, max(1, len(devices)))
        return max(1, min(requested, len(splits)))

    def _parallel_worker_config(self) -> AppliedInferenceConfig:
        cfg = copy.deepcopy(self.config)
        if getattr(cfg.training, "dataloader_workers", None) not in (None, 0):
            logger.warning(
                "Disabling DataLoader worker multiprocessing inside parallel "
                "outer-fold jobs. Nested joblib + PyTorch multiprocessing can "
                "fail under forkserver/spawn start methods."
            )
        cfg.training.dataloader_workers = 0
        return cfg

    def _cleanup_model(self, model: nn.Module) -> None:
        try:
            model.cpu()
        except Exception:
            pass
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
        cuda_cleanup()

    def _memory_summary(self) -> str:
        try:
            return f" | {get_memory_info()}"
        except Exception:
            return ""

    def _save_predictions(self, results_df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(self.output_path, index=False)
        logger.info("DragonNet DR-learner predictions saved to: %s", self.output_path)

    def _save_artifacts(self, results_df: pd.DataFrame) -> None:
        if self.nuisance_rows:
            pd.concat(self.nuisance_rows, ignore_index=True).to_parquet(
                self.artifact_dir / "nuisance_oof_predictions.parquet",
                index=False,
            )
        if self.effect_rows:
            pd.concat(self.effect_rows, ignore_index=True).to_parquet(
                self.artifact_dir / "dr_effect_oof_predictions.parquet",
                index=False,
            )
        pd.DataFrame(self.nuisance_attention_rows).to_parquet(
            self.artifact_dir / "nuisance_attention_evidence.parquet",
            index=False,
        )
        pd.DataFrame(self.effect_attention_rows).to_parquet(
            self.artifact_dir / "dr_effect_attention_evidence.parquet",
            index=False,
        )
        pd.DataFrame(self.training_rows).to_csv(
            self.artifact_dir / "training_log.csv",
            index=False,
        )
        metrics = _prediction_metrics(
            results_df,
            outcome_column=self.config.outcome_column,
            treatment_column=self.config.treatment_column,
        )
        with open(self.artifact_dir / "metrics.json", "w") as handle:
            json.dump(metrics, handle, indent=2)
        manifest = {
            "model_type": "dragonnet_drlearner",
            "n_rows": int(len(results_df)),
            "output_path": str(self.output_path),
            "config": asdict(self.dr_config),
            "feature_extractor_type": self._effective_feature_extractor_type(),
        }
        with open(self.artifact_dir / "run_manifest.json", "w") as handle:
            json.dump(manifest, handle, indent=2)


def _run_dragonnet_drlearner_outer_fold(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: str,
    num_workers: int,
    gpu_ids: Optional[Sequence[int]],
    outer_fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    runner = DragonNetDRLearnerRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        device=torch.device(device),
        num_workers=num_workers,
        gpu_ids=gpu_ids,
    )
    predictions = runner._run_one_analysis_split(outer_fold, train_idx, test_idx)
    return {
        "predictions": predictions,
        "nuisance_rows": runner.nuisance_rows,
        "effect_rows": runner.effect_rows,
        "nuisance_attention_rows": runner.nuisance_attention_rows,
        "effect_attention_rows": runner.effect_attention_rows,
        "training_rows": runner.training_rows,
    }


def _ensure_row_ids(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "_oci_row_id" not in result.columns:
        result["_oci_row_id"] = np.arange(len(result), dtype=int)
    return result


def _train_validation_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 4:
        return df.reset_index(drop=True), df.reset_index(drop=True)
    rng = np.random.RandomState(2026)
    order = rng.permutation(len(df))
    val_size = max(1, int(round(0.2 * len(df))))
    val_pos = order[:val_size]
    train_pos = order[val_size:]
    if len(train_pos) == 0:
        train_pos = order
    return (
        df.iloc[train_pos].reset_index(drop=True),
        df.iloc[val_pos].reset_index(drop=True),
    )


def _bounded_fold_count(requested: int, n: int) -> int:
    if n < 2:
        raise ValueError("At least two rows are required for cross-fitting")
    return max(2, min(int(requested), int(n)))


def _make_linear_lr_scheduler(optimizer, train_config, num_batches: int, epochs: int):
    if getattr(train_config, "lr_schedule", "linear") != "linear":
        return None
    total_steps = max(1, int(num_batches) * int(epochs))
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_steps,
    )


def _metadata_value(values: Any, offset: int) -> Any:
    try:
        value = values[offset]
    except Exception:
        value = values
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    return value


def _safe_roc_auc(y_true: Any, y_score: Any) -> Optional[float]:
    y = np.asarray(y_true, dtype=float)
    score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(score)
    y = y[mask]
    score = score[mask]
    if len(y) < 2 or np.unique(y).size < 2:
        return None
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return None


def _safe_corr(a: Any, b: Any) -> Optional[float]:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _format_optional_metric(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.4f}"


def _prediction_metrics(
    df: pd.DataFrame,
    *,
    outcome_column: str,
    treatment_column: str,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "pred_ite_mean": _finite_or_none(df["pred_ite_prob"].mean()),
        "pred_ite_std": _finite_or_none(df["pred_ite_prob"].std()),
        "dragonnet_plugin_ite_mean": _finite_or_none(df["dragonnet_plugin_ite_prob"].mean()),
    }
    if treatment_column in df.columns:
        t = df[treatment_column].to_numpy(dtype=float)
        metrics["propensity_treatment_auroc"] = _safe_roc_auc(
            t,
            df["pred_propensity_prob"].to_numpy(dtype=float),
        )
        metrics["dr_ite_treatment_auroc"] = _safe_roc_auc(
            t,
            df["pred_ite_prob"].to_numpy(dtype=float),
        )
        metrics.update(
            binary_calibration_metrics(
                t,
                df["pred_propensity_prob"].to_numpy(dtype=float),
                prefix="propensity",
            )
        )
    if outcome_column in df.columns:
        y = df[outcome_column].to_numpy(dtype=float)
        t = (
            df[treatment_column].to_numpy(dtype=float)
            if treatment_column in df.columns
            else np.zeros(len(df), dtype=float)
        )
        factual = np.where(
            t >= 0.5,
            df["pred_y1_prob"].to_numpy(dtype=float),
            df["pred_y0_prob"].to_numpy(dtype=float),
        )
        metrics["factual_outcome_auroc"] = _safe_roc_auc(y, factual)
        metrics["dr_ite_outcome_auroc"] = _safe_roc_auc(
            y,
            df["pred_ite_prob"].to_numpy(dtype=float),
        )
    true_ite_col = None
    for candidate in ("true_ite_prob", "true_ite", "ite"):
        if candidate in df.columns:
            true_ite_col = candidate
            break
    if true_ite_col is not None:
        true_ite = df[true_ite_col].to_numpy(dtype=float)
        pred = df["pred_ite_prob"].to_numpy(dtype=float)
        plugin = df["dragonnet_plugin_ite_prob"].to_numpy(dtype=float)
        mask = np.isfinite(true_ite) & np.isfinite(pred)
        if np.any(mask):
            metrics["ite_mse"] = float(mean_squared_error(true_ite[mask], pred[mask]))
            metrics["ite_mae"] = float(mean_absolute_error(true_ite[mask], pred[mask]))
            metrics["ite_corr"] = _safe_corr(true_ite[mask], pred[mask])
            metrics["ate_bias"] = float(abs(np.mean(pred[mask]) - np.mean(true_ite[mask])))
        plugin_mask = np.isfinite(true_ite) & np.isfinite(plugin)
        if np.any(plugin_mask):
            metrics["dragonnet_plugin_ite_mse"] = float(
                mean_squared_error(true_ite[plugin_mask], plugin[plugin_mask])
            )
            metrics["dragonnet_plugin_ite_corr"] = _safe_corr(
                true_ite[plugin_mask],
                plugin[plugin_mask],
            )
    for true_col, pred_col in (
        ("true_y0_prob", "pred_y0_prob"),
        ("true_y1_prob", "pred_y1_prob"),
        ("true_treatment_prob", "pred_propensity_prob"),
    ):
        if true_col in df.columns and pred_col in df.columns:
            true = df[true_col].to_numpy(dtype=float)
            pred = df[pred_col].to_numpy(dtype=float)
            mask = np.isfinite(true) & np.isfinite(pred)
            if np.any(mask):
                metrics[f"{pred_col}_oracle_mse"] = float(mean_squared_error(true[mask], pred[mask]))
                metrics[f"{pred_col}_oracle_corr"] = _safe_corr(true[mask], pred[mask])
    return metrics
