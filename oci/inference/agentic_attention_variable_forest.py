"""Agentic attention-evidence variable discovery plus explicit-feature forest."""

from __future__ import annotations

import gc
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold

from ..config import (
    AgenticAttentionVariableForestConfig,
    AppliedInferenceConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
)
from ..models.causal_forest_head import CausalForestHead
from ..models.extractor_factory import create_feature_extractor
from .agentic_explicit_feature_forest import (
    OpenAICompatibleFeatureSearchAgent,
    VLLMExplicitFeatureExtractionProvider,
)
from .applied_explicit_feature_forest import _build_features, _hstack_present

logger = logging.getLogger(__name__)

VALID_ROLES = {"confounder", "effect_modifier"}
VALID_TYPES = {"categorical", "continuous"}


def run_agentic_attention_variable_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: Optional[torch.device] = None,
    num_workers: int = 1,
    proposal_agent: Optional[Any] = None,
    extraction_provider: Optional[Any] = None,
) -> None:
    """Run the attention-evidence variable discovery forest pipeline."""
    runner = AgenticAttentionVariableForestRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        device=device or torch.device("cpu"),
        num_workers=num_workers,
        proposal_agent=proposal_agent,
        extraction_provider=extraction_provider,
    )
    runner.run()


class _NuisanceNet(nn.Module):
    def __init__(self, extractor: nn.Module, hidden_dim: int, outcome_type: str):
        super().__init__()
        self.extractor = extractor
        self.outcome_type = outcome_type
        self.shared = nn.Sequential(
            nn.Linear(extractor.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.propensity = nn.Linear(hidden_dim, 1)
        self.outcome = nn.Linear(hidden_dim, 1)

    def forward(self, texts: Sequence[str]):
        features = self.extractor(list(texts))
        hidden = self.shared(features)
        return self.propensity(hidden).squeeze(-1), self.outcome(hidden).squeeze(-1)


class _EffectNet(nn.Module):
    def __init__(self, extractor: nn.Module, hidden_dim: int):
        super().__init__()
        self.extractor = extractor
        self.head = nn.Sequential(
            nn.Linear(extractor.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, texts: Sequence[str]):
        return self.head(self.extractor(list(texts))).squeeze(-1)


class AgenticAttentionVariableForestRunner:
    """End-to-end implementation of the proposed attention-variable strategy."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        config: AppliedInferenceConfig,
        output_path: Path,
        device: torch.device,
        num_workers: int = 1,
        proposal_agent: Optional[Any] = None,
        extraction_provider: Optional[Any] = None,
    ):
        self.dataset = dataset.reset_index(drop=True).copy()
        self.dataset["_oci_row_id"] = np.arange(len(self.dataset), dtype=int)
        self.config = config
        self.output_path = Path(output_path)
        self.artifact_dir = self.output_path.parent / "agentic_attention_variable_forest"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.num_workers = int(num_workers or 1)
        self.avf_config: AgenticAttentionVariableForestConfig = getattr(
            config.architecture,
            "agentic_attention_variable_forest",
            AgenticAttentionVariableForestConfig(),
        )
        self.agent_search_config = getattr(config.architecture, "agentic_feature_search")
        self.cf_config: ExplicitFeatureForestConfig = getattr(
            config.architecture,
            "explicit_feature_forest",
            ExplicitFeatureForestConfig(),
        )
        self.proposal_agent = proposal_agent or OpenAICompatibleFeatureSearchAgent(
            self.agent_search_config
        )
        self.extraction_provider = extraction_provider or VLLMExplicitFeatureExtractionProvider(
            config=config,
            output_dir=self.artifact_dir,
        )

        self.nuisance_rows: List[pd.DataFrame] = []
        self.r_stage_rows: List[pd.DataFrame] = []
        self.nuisance_attention_rows: List[Dict[str, Any]] = []
        self.effect_attention_rows: List[Dict[str, Any]] = []
        self.confounder_candidate_rows: List[Dict[str, Any]] = []
        self.modifier_candidate_rows: List[Dict[str, Any]] = []
        self.consensus_rows: List[Dict[str, Any]] = []
        self.metric_rows: List[Dict[str, Any]] = []

    def run(self) -> None:
        logger.info("=" * 80)
        logger.info("AGENTIC ATTENTION VARIABLE FOREST")
        logger.info("=" * 80)
        prediction_frames = []

        for outer_fold, train_idx, test_idx in self._analysis_splits():
            logger.info(
                "Attention-variable fold %s: train=%s test=%s",
                outer_fold,
                len(train_idx),
                len(test_idx),
            )
            fold_predictions = self._run_one_analysis_split(
                outer_fold=outer_fold,
                train_idx=train_idx,
                test_idx=test_idx,
            )
            prediction_frames.append(fold_predictions)

        results_df = pd.concat(prediction_frames).sort_values("_oci_row_id")
        self._save_predictions(results_df)
        self._save_artifacts(results_df)

    def _analysis_splits(self) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        if self.config.cv_folds > 1:
            splits = KFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=42,
            ).split(self.dataset)
            return [
                (fold, np.asarray(train_idx), np.asarray(test_idx))
                for fold, (train_idx, test_idx) in enumerate(splits, start=1)
            ]

        split_col = self.config.split_column
        if split_col in self.dataset.columns and "test" in set(self.dataset[split_col]):
            train_mask = self.dataset[split_col].isin(["train", "val"])
            test_mask = self.dataset[split_col] == "test"
            return [
                (
                    1,
                    np.where(train_mask.to_numpy())[0],
                    np.where(test_mask.to_numpy())[0],
                )
            ]

        all_idx = np.arange(len(self.dataset))
        logger.warning(
            "No held-out split configured for agentic_attention_variable_forest; "
            "variable discovery and final estimates will use the full dataset."
        )
        return [(1, all_idx, all_idx)]

    def _run_one_analysis_split(
        self,
        outer_fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> pd.DataFrame:
        discovery_df = self.dataset.iloc[train_idx].reset_index(drop=True)
        nuisance = self._crossfit_nuisance(discovery_df, outer_fold)
        confounders = self._discover_variables_from_attention(
            stage="confounder",
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            attention_rows=nuisance["attention"],
            existing_specs=self._initial_specs(),
        )
        r_stage = self._crossfit_effect(discovery_df, nuisance["predictions"], outer_fold)
        modifiers = self._discover_variables_from_attention(
            stage="effect_modifier",
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            attention_rows=r_stage["attention"],
            existing_specs=self._merge_specs(self._initial_specs(), confounders),
        )
        selected_specs = self._merge_specs(self._initial_specs(), confounders, modifiers)

        self.dataset = self.extraction_provider.ensure_features(self.dataset, selected_specs)
        train_df = self.dataset.iloc[train_idx].copy()
        test_df = self.dataset.iloc[test_idx].copy()
        selected_specs = self._filter_specs_by_extraction_coverage(
            train_df,
            selected_specs,
            manual_specs=self._initial_specs(),
        )
        self.consensus_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
                "confounders": [spec.name for spec in selected_specs if "confounder" in spec.roles],
                "effect_modifiers": [
                    spec.name for spec in selected_specs if "effect_modifier" in spec.roles
                ],
            }
        )
        predictions, metrics = self._fit_final_forest(
            train_df=train_df,
            test_df=test_df,
            selected_specs=selected_specs,
            fold_id=outer_fold,
        )
        predictions["outer_fold"] = outer_fold
        predictions["selected_feature_names"] = ",".join(spec.name for spec in selected_specs)
        self.metric_rows.append({"outer_fold": outer_fold, **metrics})
        return predictions

    def _initial_specs(self) -> List[ExplicitFeatureSpec]:
        if getattr(self.config.explicit_features, "features", None):
            return list(self.config.explicit_features.features)
        return []

    def _create_extractor(self) -> nn.Module:
        arch = self.config.architecture
        extractor_type = getattr(arch, "feature_extractor_type", "hierarchical_transformer")
        if extractor_type == "frozen_llm_pooler":
            extractor_type = "hierarchical_transformer"
        return create_feature_extractor(
            extractor_type=extractor_type,
            device=self.device,
            htr_sentence_model=getattr(arch, "htr_sentence_model", "prajjwal1/bert-tiny"),
            htr_freeze_sentence_encoder=getattr(arch, "htr_freeze_sentence_encoder", True),
            htr_chunk_size_words=getattr(arch, "htr_chunk_size_words", 96),
            htr_chunk_overlap_words=getattr(arch, "htr_chunk_overlap_words", 24),
            htr_max_chunks=getattr(arch, "htr_max_chunks", 128),
            htr_max_chunk_length=getattr(arch, "htr_max_chunk_length", 128),
            htr_num_layers=getattr(arch, "htr_num_layers", 2),
            htr_num_heads=getattr(arch, "htr_num_heads", 4),
            htr_transformer_dim=getattr(arch, "htr_transformer_dim", 256),
            htr_dropout=getattr(arch, "htr_dropout", 0.1),
            htr_projection_dim=getattr(arch, "htr_projection_dim", 128),
            htr_hash_embedding_dim=getattr(arch, "htr_hash_embedding_dim", 256),
            htr_sentence_encoder_batch_size=getattr(
                arch,
                "htr_sentence_encoder_batch_size",
                128,
            ),
            htr_sentence_encoder_backend=getattr(
                arch,
                "htr_sentence_encoder_backend",
                "auto",
            ),
            htr_sentence_pooling=getattr(arch, "htr_sentence_pooling", "auto"),
            htr_normalize_sentence_embeddings=getattr(
                arch,
                "htr_normalize_sentence_embeddings",
                True,
            ),
            htr_trainable_sentence_encoder_layers=getattr(
                arch,
                "htr_trainable_sentence_encoder_layers",
                0,
            ),
        )

    def _crossfit_nuisance(self, df: pd.DataFrame, outer_fold: int) -> Dict[str, Any]:
        folds = _bounded_fold_count(self.avf_config.nuisance_folds, len(df))
        predictions = pd.DataFrame(
            {
                "_oci_row_id": df["_oci_row_id"].to_numpy(),
                "outer_fold": outer_fold,
                "e_hat": np.nan,
                "m_hat": np.nan,
                "y_residual": np.nan,
                "t_residual": np.nan,
                "r_loss_at_zero_tau": np.nan,
                "nuisance_fold": np.nan,
            }
        )
        attention_rows: List[Dict[str, Any]] = []

        split_items = list(
            enumerate(
                KFold(n_splits=folds, shuffle=True, random_state=10_000 + outer_fold).split(df),
                start=1,
            )
        )

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            model = None
            logger.info(
                "Outer fold %s nuisance fold %s/%s: train=%s heldout=%s%s",
                outer_fold,
                fold,
                folds,
                len(fit_pos),
                len(heldout_pos),
                self._cuda_memory_summary(),
            )
            try:
                model = _NuisanceNet(
                    extractor=self._create_extractor(),
                    hidden_dim=getattr(
                        self.config.architecture,
                        "causal_head_hidden_outcome_dim",
                        64,
                    ),
                    outcome_type=self.config.outcome_type,
                ).to(self.device)
                self._train_nuisance_model(
                    model,
                    df,
                    fit_pos,
                    outer_fold=outer_fold,
                    fold=fold,
                    total_folds=folds,
                )
                heldout = df.iloc[heldout_pos]
                logger.info(
                    "Outer fold %s nuisance fold %s/%s: predicting heldout%s",
                    outer_fold,
                    fold,
                    folds,
                    self._cuda_memory_summary(),
                )
                e_hat, m_hat = self._predict_nuisance_model(model, heldout)
                y = heldout[self.config.outcome_column].to_numpy(dtype=float)
                t = heldout[self.config.treatment_column].to_numpy(dtype=float)
                y_resid = y - m_hat
                t_resid = t - e_hat
                logger.info(
                    "Outer fold %s nuisance fold %s/%s: collecting attention evidence",
                    outer_fold,
                    fold,
                    folds,
                )
                fold_attention = self._attention_evidence(
                    model.extractor,
                    heldout,
                    fold=fold,
                    outer_fold=outer_fold,
                    stage="nuisance",
                    extra={
                        "e_hat": e_hat,
                        "m_hat": m_hat,
                        "y_residual": y_resid,
                        "t_residual": t_resid,
                    },
                )
                logger.info(
                    "Outer fold %s nuisance fold %s/%s complete: attention_rows=%s "
                    "e_hat_mean=%.4f m_hat_mean=%.4f%s",
                    outer_fold,
                    fold,
                    folds,
                    len(fold_attention),
                    float(np.mean(e_hat)),
                    float(np.mean(m_hat)),
                    self._cuda_memory_summary(),
                )
                return {
                    "fold": fold,
                    "heldout_pos": heldout_pos,
                    "e_hat": e_hat,
                    "m_hat": m_hat,
                    "y_resid": y_resid,
                    "t_resid": t_resid,
                    "attention": fold_attention,
                }
            except RuntimeError as exc:
                if _is_cuda_oom(exc):
                    logger.error(
                        "CUDA OOM in outer fold %s nuisance fold %s/%s%s",
                        outer_fold,
                        fold,
                        folds,
                        self._cuda_memory_summary(),
                    )
                raise
            finally:
                if model is not None:
                    self._cleanup_model(model)
                    model = None
                    logger.info(
                        "Outer fold %s nuisance fold %s/%s: model cleanup complete%s",
                        outer_fold,
                        fold,
                        folds,
                        self._cuda_memory_summary(),
                    )

        n_jobs = self._fold_n_jobs(folds)
        if n_jobs > 1:
            fold_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(run_fold)(fold, fit_pos, heldout_pos)
                for fold, (fit_pos, heldout_pos) in split_items
            )
        else:
            fold_results = [
                run_fold(fold, fit_pos, heldout_pos)
                for fold, (fit_pos, heldout_pos) in split_items
            ]

        for result in fold_results:
            heldout_pos = result["heldout_pos"]
            predictions.loc[heldout_pos, "e_hat"] = result["e_hat"]
            predictions.loc[heldout_pos, "m_hat"] = result["m_hat"]
            predictions.loc[heldout_pos, "y_residual"] = result["y_resid"]
            predictions.loc[heldout_pos, "t_residual"] = result["t_resid"]
            predictions.loc[heldout_pos, "r_loss_at_zero_tau"] = result["y_resid"] ** 2
            predictions.loc[heldout_pos, "nuisance_fold"] = result["fold"]
            attention_rows.extend(result["attention"])

        self.nuisance_rows.append(predictions)
        self.nuisance_attention_rows.extend(attention_rows)
        return {"predictions": predictions, "attention": attention_rows}

    def _crossfit_effect(
        self,
        df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        folds = _bounded_fold_count(self.avf_config.effect_folds, len(df))
        r_df = nuisance_predictions.copy()
        r_df["tau_hat_r_stage"] = np.nan
        r_df["r_loss"] = np.nan
        r_df["effect_fold"] = np.nan
        attention_rows: List[Dict[str, Any]] = []

        e = r_df["e_hat"].to_numpy(dtype=float)
        m = r_df["m_hat"].to_numpy(dtype=float)
        y = df[self.config.outcome_column].to_numpy(dtype=float)
        t = df[self.config.treatment_column].to_numpy(dtype=float)
        t_resid = t - np.clip(e, self.avf_config.e_clip, 1.0 - self.avf_config.e_clip)
        y_resid = y - m
        weights = np.square(t_resid).astype(np.float32)
        weights = np.maximum(weights, 1e-6)
        denom = t_resid.copy()
        near_zero = np.abs(denom) < 1e-6
        denom[near_zero] = np.where(denom[near_zero] < 0, -1e-6, 1e-6)
        targets = (y_resid / denom).astype(np.float32)

        split_items = list(
            enumerate(
                KFold(n_splits=folds, shuffle=True, random_state=20_000 + outer_fold).split(df),
                start=1,
            )
        )

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            model = None
            logger.info(
                "Outer fold %s effect fold %s/%s: train=%s heldout=%s%s",
                outer_fold,
                fold,
                folds,
                len(fit_pos),
                len(heldout_pos),
                self._cuda_memory_summary(),
            )
            try:
                model = _EffectNet(
                    extractor=self._create_extractor(),
                    hidden_dim=getattr(
                        self.config.architecture,
                        "causal_head_hidden_outcome_dim",
                        64,
                    ),
                ).to(self.device)
                self._train_effect_model(
                    model,
                    df,
                    fit_pos,
                    targets,
                    weights,
                    outer_fold=outer_fold,
                    fold=fold,
                    total_folds=folds,
                )
                heldout = df.iloc[heldout_pos]
                logger.info(
                    "Outer fold %s effect fold %s/%s: predicting heldout%s",
                    outer_fold,
                    fold,
                    folds,
                    self._cuda_memory_summary(),
                )
                tau_hat = self._predict_effect_model(model, heldout)
                heldout_r_loss = (y_resid[heldout_pos] - tau_hat * t_resid[heldout_pos]) ** 2
                logger.info(
                    "Outer fold %s effect fold %s/%s: collecting attention evidence",
                    outer_fold,
                    fold,
                    folds,
                )
                fold_attention = self._attention_evidence(
                    model.extractor,
                    heldout,
                    fold=fold,
                    outer_fold=outer_fold,
                    stage="effect_modifier",
                    extra={"tau_hat_r_stage": tau_hat, "r_loss": heldout_r_loss},
                )
                logger.info(
                    "Outer fold %s effect fold %s/%s complete: attention_rows=%s "
                    "tau_mean=%.4f r_loss_mean=%.4f%s",
                    outer_fold,
                    fold,
                    folds,
                    len(fold_attention),
                    float(np.mean(tau_hat)),
                    float(np.mean(heldout_r_loss)),
                    self._cuda_memory_summary(),
                )
                return {
                    "fold": fold,
                    "heldout_pos": heldout_pos,
                    "tau_hat": tau_hat,
                    "r_loss": heldout_r_loss,
                    "attention": fold_attention,
                }
            except RuntimeError as exc:
                if _is_cuda_oom(exc):
                    logger.error(
                        "CUDA OOM in outer fold %s effect fold %s/%s%s",
                        outer_fold,
                        fold,
                        folds,
                        self._cuda_memory_summary(),
                    )
                raise
            finally:
                if model is not None:
                    self._cleanup_model(model)
                    model = None
                    logger.info(
                        "Outer fold %s effect fold %s/%s: model cleanup complete%s",
                        outer_fold,
                        fold,
                        folds,
                        self._cuda_memory_summary(),
                    )

        n_jobs = self._fold_n_jobs(folds)
        if n_jobs > 1:
            fold_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(run_fold)(fold, fit_pos, heldout_pos)
                for fold, (fit_pos, heldout_pos) in split_items
            )
        else:
            fold_results = [
                run_fold(fold, fit_pos, heldout_pos)
                for fold, (fit_pos, heldout_pos) in split_items
            ]

        for result in fold_results:
            heldout_pos = result["heldout_pos"]
            r_df.loc[heldout_pos, "tau_hat_r_stage"] = result["tau_hat"]
            r_df.loc[heldout_pos, "r_loss"] = result["r_loss"]
            r_df.loc[heldout_pos, "effect_fold"] = result["fold"]
            attention_rows.extend(result["attention"])

        self.r_stage_rows.append(r_df)
        self.effect_attention_rows.extend(attention_rows)
        return {"predictions": r_df, "attention": attention_rows}

    def _train_nuisance_model(
        self,
        model: _NuisanceNet,
        df: pd.DataFrame,
        positions,
        outer_fold: int,
        fold: int,
        total_folds: int,
    ):
        train_config = self.config.training
        model.extractor.fit_tokenizer(
            df.iloc[positions][self.config.text_column].astype(str).tolist()
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=train_config.learning_rate,
            weight_decay=getattr(train_config, "weight_decay", 0.01),
        )
        num_batches = max(1, int(np.ceil(len(positions) / train_config.batch_size)))
        progress_every = max(1, num_batches // 5)
        logger.info(
            "Outer fold %s nuisance fold %s/%s: training for %s epoch(s), "
            "batch_size=%s, batches/epoch=%s%s",
            outer_fold,
            fold,
            total_folds,
            train_config.epochs,
            train_config.batch_size,
            num_batches,
            self._cuda_memory_summary(),
        )
        for epoch in range(1, train_config.epochs + 1):
            model.train()
            loss_sum = 0.0
            prop_sum = 0.0
            outcome_sum = 0.0
            batch_count = 0
            for batch_idx, batch_pos in enumerate(
                _batch_positions(positions, train_config.batch_size, shuffle=True),
                start=1,
            ):
                batch = df.iloc[batch_pos]
                texts = batch[self.config.text_column].tolist()
                t = torch.as_tensor(
                    batch[self.config.treatment_column].to_numpy(dtype=np.float32),
                    device=self.device,
                )
                y = torch.as_tensor(
                    batch[self.config.outcome_column].to_numpy(dtype=np.float32),
                    device=self.device,
                )
                optimizer.zero_grad()
                t_logit, y_pred = model(texts)
                prop_loss = F.binary_cross_entropy_with_logits(t_logit, t)
                if self.config.outcome_type == "continuous":
                    outcome_loss = F.mse_loss(y_pred, y)
                else:
                    outcome_loss = F.binary_cross_entropy_with_logits(y_pred, y)
                loss = outcome_loss + self.config.training.alpha_propensity * prop_loss
                loss.backward()
                self._clip_and_step(model, optimizer)
                batch_count += 1
                loss_value = float(loss.detach().cpu())
                prop_value = float(prop_loss.detach().cpu())
                outcome_value = float(outcome_loss.detach().cpu())
                loss_sum += loss_value
                prop_sum += prop_value
                outcome_sum += outcome_value
                if (
                    batch_idx == 1
                    or batch_idx == num_batches
                    or batch_idx % progress_every == 0
                ):
                    logger.info(
                        "Outer fold %s nuisance fold %s/%s epoch %s/%s "
                        "batch %s/%s loss=%.4f outcome=%.4f propensity=%.4f%s",
                        outer_fold,
                        fold,
                        total_folds,
                        epoch,
                        train_config.epochs,
                        batch_idx,
                        num_batches,
                        loss_value,
                        outcome_value,
                        prop_value,
                        self._cuda_memory_summary(),
                    )
            denom = max(1, batch_count)
            logger.info(
                "Outer fold %s nuisance fold %s/%s epoch %s/%s complete: "
                "loss=%.4f outcome=%.4f propensity=%.4f%s",
                outer_fold,
                fold,
                total_folds,
                epoch,
                train_config.epochs,
                loss_sum / denom,
                outcome_sum / denom,
                prop_sum / denom,
                self._cuda_memory_summary(),
            )

    def _train_effect_model(
        self,
        model: _EffectNet,
        df: pd.DataFrame,
        positions,
        targets: np.ndarray,
        weights: np.ndarray,
        outer_fold: int,
        fold: int,
        total_folds: int,
    ):
        train_config = self.config.training
        model.extractor.fit_tokenizer(
            df.iloc[positions][self.config.text_column].astype(str).tolist()
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=train_config.learning_rate,
            weight_decay=getattr(train_config, "weight_decay", 0.01),
        )
        num_batches = max(1, int(np.ceil(len(positions) / train_config.batch_size)))
        progress_every = max(1, num_batches // 5)
        logger.info(
            "Outer fold %s effect fold %s/%s: training for %s epoch(s), "
            "batch_size=%s, batches/epoch=%s%s",
            outer_fold,
            fold,
            total_folds,
            train_config.epochs,
            train_config.batch_size,
            num_batches,
            self._cuda_memory_summary(),
        )
        for epoch in range(1, train_config.epochs + 1):
            model.train()
            loss_sum = 0.0
            batch_count = 0
            for batch_idx, batch_pos in enumerate(
                _batch_positions(positions, train_config.batch_size, shuffle=True),
                start=1,
            ):
                batch = df.iloc[batch_pos]
                texts = batch[self.config.text_column].tolist()
                target = torch.as_tensor(targets[batch_pos], device=self.device)
                weight = torch.as_tensor(weights[batch_pos], device=self.device)
                optimizer.zero_grad()
                tau = model(texts)
                loss = torch.mean(weight * torch.square(target - tau))
                loss.backward()
                self._clip_and_step(model, optimizer)
                batch_count += 1
                loss_value = float(loss.detach().cpu())
                loss_sum += loss_value
                if (
                    batch_idx == 1
                    or batch_idx == num_batches
                    or batch_idx % progress_every == 0
                ):
                    logger.info(
                        "Outer fold %s effect fold %s/%s epoch %s/%s "
                        "batch %s/%s r_loss=%.4f%s",
                        outer_fold,
                        fold,
                        total_folds,
                        epoch,
                        train_config.epochs,
                        batch_idx,
                        num_batches,
                        loss_value,
                        self._cuda_memory_summary(),
                    )
            logger.info(
                "Outer fold %s effect fold %s/%s epoch %s/%s complete: "
                "r_loss=%.4f%s",
                outer_fold,
                fold,
                total_folds,
                epoch,
                train_config.epochs,
                loss_sum / max(1, batch_count),
                self._cuda_memory_summary(),
            )

    def _clip_and_step(self, model: nn.Module, optimizer) -> None:
        clip_norm = getattr(self.config.training, "gradient_clip_norm", 0.0)
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

    def _predict_nuisance_model(self, model: _NuisanceNet, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        prop = []
        outcome = []
        with torch.no_grad():
            for start in range(0, len(df), self.config.training.batch_size):
                batch = df.iloc[start:start + self.config.training.batch_size]
                t_logit, y_pred = model(batch[self.config.text_column].tolist())
                prop.append(torch.sigmoid(t_logit).cpu().numpy())
                if self.config.outcome_type == "continuous":
                    outcome.append(y_pred.cpu().numpy())
                else:
                    outcome.append(torch.sigmoid(y_pred).cpu().numpy())
        return np.concatenate(prop), np.concatenate(outcome)

    def _predict_effect_model(self, model: _EffectNet, df: pd.DataFrame) -> np.ndarray:
        model.eval()
        tau = []
        with torch.no_grad():
            for start in range(0, len(df), self.config.training.batch_size):
                batch = df.iloc[start:start + self.config.training.batch_size]
                tau.append(model(batch[self.config.text_column].tolist()).cpu().numpy())
        return np.concatenate(tau)

    def _attention_evidence(
        self,
        extractor: nn.Module,
        df: pd.DataFrame,
        fold: int,
        outer_fold: int,
        stage: str,
        extra: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        texts = df[self.config.text_column].tolist()
        row_ids = df["_oci_row_id"].tolist()
        metadata = []
        for offset in range(len(df)):
            item = {"outer_fold": outer_fold}
            for key, values in extra.items():
                item[key] = float(np.asarray(values)[offset])
            metadata.append(item)
        batch_size = max(1, int(self.config.training.batch_size))
        total_batches = max(1, int(np.ceil(len(texts) / batch_size)))
        progress_every = max(1, total_batches // 5)
        records: List[Dict[str, Any]] = []
        for batch_idx, start in enumerate(range(0, len(texts), batch_size), start=1):
            end = min(start + batch_size, len(texts))
            records.extend(
                extractor.get_attention_evidence(
                    texts[start:end],
                    row_ids=row_ids[start:end],
                    fold=fold,
                    stage=stage,
                    top_k=self.avf_config.attention_top_k_chunks,
                    metadata=metadata[start:end],
                )
            )
            if (
                batch_idx == 1
                or batch_idx == total_batches
                or batch_idx % progress_every == 0
            ):
                logger.info(
                    "Outer fold %s %s fold %s: attention batch %s/%s rows=%s/%s%s",
                    outer_fold,
                    stage,
                    fold,
                    batch_idx,
                    total_batches,
                    end,
                    len(texts),
                    self._cuda_memory_summary(),
                )
        return records

    def _discover_variables_from_attention(
        self,
        stage: str,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        attention_rows: Sequence[Dict[str, Any]],
        existing_specs: Sequence[ExplicitFeatureSpec],
    ) -> List[ExplicitFeatureSpec]:
        proposals_by_fold: Dict[int, List[ExplicitFeatureSpec]] = {}
        for fold in sorted({int(row["fold"]) for row in attention_rows if row.get("fold") is not None}):
            fold_rows = [row for row in attention_rows if int(row.get("fold")) == fold]
            context = self._build_agent_context(
                stage=stage,
                outer_fold=outer_fold,
                inner_fold=fold,
                discovery_df=discovery_df,
                attention_rows=fold_rows,
                existing_specs=existing_specs,
            )
            raw = self.proposal_agent.propose(context)
            specs = _proposal_dicts_to_specs(raw, required_role=stage)
            proposals_by_fold[fold] = specs
            row = {
                "outer_fold": outer_fold,
                "fold": fold,
                "stage": stage,
                "context": _scrub_context(context),
                "proposals": [_spec_to_dict(spec) for spec in specs],
            }
            if stage == "confounder":
                self.confounder_candidate_rows.append(row)
            else:
                self.modifier_candidate_rows.append(row)

        selected = consensus_feature_specs(
            proposals_by_fold,
            min_fold_fraction=self.avf_config.consensus_min_fold_fraction,
            required_role=stage,
        )
        logger.info(
            "Outer fold %s %s consensus selected %s variable(s): %s",
            outer_fold,
            stage,
            len(selected),
            [spec.name for spec in selected],
        )
        return selected

    def _build_agent_context(
        self,
        stage: str,
        outer_fold: int,
        inner_fold: int,
        discovery_df: pd.DataFrame,
        attention_rows: Sequence[Dict[str, Any]],
        existing_specs: Sequence[ExplicitFeatureSpec],
    ) -> Dict[str, Any]:
        evidence = sorted(
            attention_rows,
            key=lambda row: abs(float(row.get("attention", 0.0))),
            reverse=True,
        )[: max(1, self.avf_config.attention_top_k_chunks * 20)]
        instruction = (
            "Propose pre-treatment confounder variables explaining treatment and outcome prediction evidence."
            if stage == "confounder"
            else "Propose pre-treatment effect modifier variables explaining high R-stage attention evidence."
        )
        return {
            "prompt_version": "agentic_attention_variable_forest_v1",
            "stage": stage,
            "outer_fold": outer_fold,
            "fold": inner_fold,
            "instruction": instruction,
            "clinical_question": self.config.clinical_question,
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "current_features": [_spec_to_dict(spec) for spec in existing_specs],
            "attention_evidence": [
                {
                    "row_id": int(row["row_id"]),
                    "chunk_text": row["chunk_text"],
                    "attention": float(row["attention"]),
                    **{
                        key: row[key]
                        for key in [
                            "e_hat",
                            "m_hat",
                            "y_residual",
                            "t_residual",
                            "tau_hat_r_stage",
                            "r_loss",
                        ]
                        if key in row
                    },
                }
                for row in evidence
            ],
            "fold_label_summary": {
                "n": int(len(discovery_df)),
                "treatment_rate": float(discovery_df[self.config.treatment_column].mean()),
                "outcome_mean": float(discovery_df[self.config.outcome_column].mean()),
            },
            "response_contract": {
                "proposals": [
                    {
                        "action": "add",
                        "name": "snake_case_variable_name",
                        "type": "categorical|continuous",
                        "categories": ["category_a", "category_b"],
                        "roles": [stage],
                        "description": "exact pre-treatment extraction target",
                        "rationale": "why the attended chunks support this variable",
                    }
                ]
            },
        }

    def _fit_final_forest(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        selected_specs: List[ExplicitFeatureSpec],
        fold_id: int,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        train_T = train_df[self.config.treatment_column].to_numpy()
        train_Y = train_df[self.config.outcome_column].to_numpy()
        test_T = test_df[self.config.treatment_column].to_numpy()

        X_train, W_train, x_names, w_names, means, stds = _build_features(
            train_df,
            selected_specs,
        )
        X_test, W_test, _, _, _, _ = _build_features(test_df, selected_specs, means, stds)
        actual_x_dim = 0 if X_train is None else X_train.shape[1]
        if X_train is None:
            X_train = np.zeros((len(train_df), 1), dtype=np.float32)
            X_test = np.zeros((len(test_df), 1), dtype=np.float32)
            x_names = ["intercept_effect"]

        forest = CausalForestHead(
            n_estimators=self.cf_config.n_estimators,
            max_depth=self.cf_config.max_depth,
            min_samples_leaf=self.cf_config.min_samples_leaf,
            max_features=self.cf_config.max_features,
            honest=self.cf_config.honest,
            inference=self.cf_config.inference,
            random_state=42 + fold_id,
        )
        forest.fit(X=X_train, W=W_train, T=train_T, Y=train_Y)
        cf_preds = forest.predict(X_test, return_ci=True)
        tau = cf_preds["tau_pred"]

        nuisance_train = _hstack_present(X_train, W_train)
        nuisance_test = _hstack_present(X_test, W_test)
        if nuisance_train is None:
            nuisance_train = np.zeros((len(train_df), 1), dtype=np.float32)
            nuisance_test = np.zeros((len(test_df), 1), dtype=np.float32)

        propensity = _fit_predict_propensity(
            nuisance_train,
            train_T,
            nuisance_test,
            self.cf_config,
            random_state=142 + fold_id,
        )
        outcome_pred = _fit_predict_outcome(
            nuisance_train,
            train_Y,
            nuisance_test,
            self.config.outcome_type,
            self.cf_config,
            random_state=242 + fold_id,
        )

        y0 = outcome_pred - propensity * tau
        y1 = outcome_pred + (1.0 - propensity) * tau
        if self.config.outcome_type == "binary":
            y0 = np.clip(y0, 0.0, 1.0)
            y1 = np.clip(y1, 0.0, 1.0)

        predictions = test_df.copy()
        predictions["pred_ite_prob"] = tau
        predictions["pred_y0_prob"] = y0
        predictions["pred_y1_prob"] = y1
        predictions["pred_propensity_prob"] = propensity
        predictions["pred_outcome_prob"] = outcome_pred
        predictions["cv_fold"] = fold_id
        if "tau_lower" in cf_preds:
            predictions["pred_ite_lower"] = cf_preds["tau_lower"]
            predictions["pred_ite_upper"] = cf_preds["tau_upper"]

        metrics = {
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_selected_features": int(len(selected_specs)),
            "n_x_features": int(actual_x_dim),
            "n_w_features": int(0 if W_train is None else W_train.shape[1]),
            "ate_estimate": float(np.mean(tau)),
            "treatment_auroc": _safe_roc_auc(test_T, propensity),
            "x_feature_names": x_names,
            "w_feature_names": w_names,
        }
        if "true_ite_prob" in test_df.columns:
            true_ite = test_df["true_ite_prob"].to_numpy()
            metrics["ite_mse"] = float(mean_squared_error(true_ite, tau))
            metrics["ite_mae"] = float(mean_absolute_error(true_ite, tau))
            metrics["ite_corr"] = _safe_corr(true_ite, tau)
        return predictions, metrics

    @staticmethod
    def _merge_specs(*spec_groups: Sequence[ExplicitFeatureSpec]) -> List[ExplicitFeatureSpec]:
        merged: Dict[str, ExplicitFeatureSpec] = {}
        for group in spec_groups:
            for spec in group:
                name = _normalize_feature_name(spec.name)
                if name in merged:
                    roles = list(dict.fromkeys([*merged[name].roles, *spec.roles]))
                    merged[name] = ExplicitFeatureSpec(
                        name=merged[name].name,
                        type=merged[name].type,
                        categories=merged[name].categories,
                        description=merged[name].description or spec.description,
                        roles=roles,
                    )
                else:
                    merged[name] = spec
        return list(merged.values())

    def _cleanup_model(self, model: nn.Module) -> None:
        model.cpu()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _cuda_memory_summary(self) -> str:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return ""
        try:
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_index) / 1e9
            reserved = torch.cuda.memory_reserved(device_index) / 1e9
            peak = torch.cuda.max_memory_allocated(device_index) / 1e9
            return (
                f" cuda_alloc={allocated:.2f}GB"
                f" cuda_reserved={reserved:.2f}GB"
                f" cuda_peak={peak:.2f}GB"
            )
        except Exception:
            return ""

    def _fold_n_jobs(self, folds: int) -> int:
        if self.device.type != "cpu":
            return 1
        setting = str(self.avf_config.fold_parallelism)
        if setting == "auto":
            return max(1, min(int(self.num_workers), int(folds)))
        return max(1, min(int(setting), int(folds)))

    def _filter_specs_by_extraction_coverage(
        self,
        df: pd.DataFrame,
        specs: Sequence[ExplicitFeatureSpec],
        manual_specs: Sequence[ExplicitFeatureSpec],
    ) -> List[ExplicitFeatureSpec]:
        manual_names = {_normalize_feature_name(spec.name) for spec in manual_specs}
        kept = []
        for spec in specs:
            name = _normalize_feature_name(spec.name)
            coverage = _feature_coverage(df, name)
            if (
                coverage < self.avf_config.min_extraction_coverage
                and not (self.avf_config.manual_features_locked and name in manual_names)
            ):
                logger.info(
                    "Dropping discovered feature %s for low extraction coverage %.3f < %.3f",
                    name,
                    coverage,
                    self.avf_config.min_extraction_coverage,
                )
                continue
            kept.append(spec)
        return kept

    def _save_predictions(self, results_df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(self.output_path, index=False)
        logger.info("Predictions saved to: %s", self.output_path)

    def _save_artifacts(self, results_df: pd.DataFrame) -> None:
        if self.nuisance_rows:
            pd.concat(self.nuisance_rows).to_parquet(
                self.artifact_dir / "nuisance_oof_predictions.parquet",
                index=False,
            )
        if self.r_stage_rows:
            pd.concat(self.r_stage_rows).to_parquet(
                self.artifact_dir / "r_stage_oof_predictions.parquet",
                index=False,
            )
        pd.DataFrame(self.nuisance_attention_rows).to_parquet(
            self.artifact_dir / "nuisance_attention_evidence.parquet",
            index=False,
        )
        pd.DataFrame(self.effect_attention_rows).to_parquet(
            self.artifact_dir / "r_stage_attention_evidence.parquet",
            index=False,
        )
        _write_jsonl(
            self.artifact_dir / "confounder_candidates_by_fold.jsonl",
            self.confounder_candidate_rows,
        )
        _write_jsonl(
            self.artifact_dir / "effect_modifier_candidates_by_fold.jsonl",
            self.modifier_candidate_rows,
        )
        with open(self.artifact_dir / "consensus.json", "w") as f:
            json.dump(self.consensus_rows, f, indent=2)
        metrics_for_csv = [
            {key: value for key, value in row.items() if not isinstance(value, list)}
            for row in self.metric_rows
        ]
        pd.DataFrame(metrics_for_csv).to_csv(self.artifact_dir / "metrics.csv", index=False)
        if "true_ite_prob" in results_df.columns:
            metrics = _oracle_metrics(results_df)
            with open(self.artifact_dir / "oracle_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
        manifest = {
            "model_type": "agentic_attention_variable_forest",
            "config": asdict(self.avf_config),
            "n_rows": int(len(results_df)),
            "output_path": str(self.output_path),
        }
        with open(self.artifact_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)


def consensus_feature_specs(
    proposals_by_fold: Dict[int, Sequence[ExplicitFeatureSpec]],
    min_fold_fraction: float,
    required_role: str,
) -> List[ExplicitFeatureSpec]:
    """Select specs whose normalized concept recurs across enough folds."""
    fold_count = max(1, len(proposals_by_fold))
    threshold = int(np.ceil(min_fold_fraction * fold_count))
    grouped: Dict[str, List[ExplicitFeatureSpec]] = {}
    for specs in proposals_by_fold.values():
        seen_in_fold = set()
        for spec in specs:
            if required_role not in spec.roles:
                continue
            key = _normalize_feature_name(spec.name)
            if key in seen_in_fold:
                continue
            grouped.setdefault(key, []).append(spec)
            seen_in_fold.add(key)

    selected = []
    for key, specs in sorted(grouped.items()):
        if len(specs) < threshold:
            continue
        prototype = specs[0]
        roles = list(dict.fromkeys([role for spec in specs for role in spec.roles]))
        if required_role not in roles:
            roles.append(required_role)
        selected.append(
            ExplicitFeatureSpec(
                name=key,
                type=prototype.type,
                categories=prototype.categories,
                description=prototype.description,
                roles=roles,
            )
        )
    return selected


def _proposal_dicts_to_specs(
    raw_proposals: Any,
    required_role: str,
) -> List[ExplicitFeatureSpec]:
    if isinstance(raw_proposals, dict):
        raw_proposals = raw_proposals.get("proposals", [])
    specs = []
    for proposal in raw_proposals or []:
        if not isinstance(proposal, dict):
            continue
        action = str(proposal.get("action", "add")).lower()
        if action not in {"add", "update_role"}:
            continue
        name = _normalize_feature_name(proposal.get("name", ""))
        if not name:
            continue
        typ = str(proposal.get("type") or "categorical").lower()
        if typ not in VALID_TYPES:
            typ = "categorical"
        roles = [role for role in proposal.get("roles", []) if role in VALID_ROLES]
        if required_role not in roles:
            roles.append(required_role)
        categories = proposal.get("categories")
        if typ == "categorical":
            if not categories:
                categories = ["absent", "present"]
            categories = [str(cat) for cat in categories[:8]]
        else:
            categories = None
        try:
            specs.append(
                ExplicitFeatureSpec(
                    name=name,
                    type=typ,
                    categories=categories,
                    description=proposal.get("description") or name.replace("_", " "),
                    roles=roles,
                )
            )
        except ValueError:
            continue
    return specs


def _fit_predict_propensity(
    train_x: np.ndarray,
    train_t: np.ndarray,
    test_x: np.ndarray,
    cf_config: ExplicitFeatureForestConfig,
    random_state: int,
) -> np.ndarray:
    if len(np.unique(train_t)) < 2:
        return np.full(len(test_x), float(np.mean(train_t)))
    model = RandomForestClassifier(
        n_estimators=max(50, cf_config.n_estimators // 2),
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_x, train_t)
    return model.predict_proba(test_x)[:, 1]


def _fit_predict_outcome(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    outcome_type: str,
    cf_config: ExplicitFeatureForestConfig,
    random_state: int,
) -> np.ndarray:
    if outcome_type == "continuous":
        model = RandomForestRegressor(
            n_estimators=max(50, cf_config.n_estimators // 2),
            max_depth=cf_config.max_depth,
            min_samples_leaf=cf_config.min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(train_x, train_y)
        return model.predict(test_x)
    if len(np.unique(train_y)) < 2:
        return np.full(len(test_x), float(np.mean(train_y)))
    model = RandomForestClassifier(
        n_estimators=max(50, cf_config.n_estimators // 2),
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def _batch_positions(positions, batch_size: int, shuffle: bool) -> List[np.ndarray]:
    positions = np.asarray(positions)
    if shuffle:
        positions = positions.copy()
        np.random.shuffle(positions)
    return [positions[start:start + batch_size] for start in range(0, len(positions), batch_size)]


def _bounded_fold_count(requested: int, n: int) -> int:
    if n < 2:
        raise ValueError("At least two rows are required for cross-fitting")
    return max(2, min(int(requested), int(n)))


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _normalize_feature_name(name: Any) -> str:
    value = str(name or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _feature_coverage(df: pd.DataFrame, name: str) -> float:
    col = f"explicit_feat_{name}"
    miss_col = f"{col}_missing"
    if col not in df.columns:
        return 0.0
    if miss_col in df.columns:
        missing = df[miss_col].astype(bool)
    else:
        missing = df[col].isna()
    return float(1.0 - missing.mean())


def _spec_to_dict(spec: ExplicitFeatureSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "type": spec.type,
        "categories": spec.categories,
        "description": spec.description,
        "roles": list(spec.roles),
    }


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _oracle_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    metrics = {
        "ite_mse": float(mean_squared_error(results_df["true_ite_prob"], results_df["pred_ite_prob"])),
        "ite_mae": float(mean_absolute_error(results_df["true_ite_prob"], results_df["pred_ite_prob"])),
        "ite_corr": _safe_corr(results_df["true_ite_prob"], results_df["pred_ite_prob"]),
        "ate_bias": float(
            abs(results_df["pred_ite_prob"].mean() - results_df["true_ite_prob"].mean())
        ),
    }
    if "true_y0_prob" in results_df.columns:
        metrics["y0_mse"] = float(
            mean_squared_error(results_df["true_y0_prob"], results_df["pred_y0_prob"])
        )
    if "true_y1_prob" in results_df.columns:
        metrics["y1_mse"] = float(
            mean_squared_error(results_df["true_y1_prob"], results_df["pred_y1_prob"])
        )
    return metrics


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def _scrub_context(context: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(context)
    copied["attention_evidence"] = [
        {key: value for key, value in row.items() if key != "chunk_text"}
        for row in copied.get("attention_evidence", [])
    ]
    return copied
