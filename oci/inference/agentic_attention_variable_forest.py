"""Agentic attention-evidence variable discovery plus explicit-feature forest."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

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
    _get_agent_response_trace,
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

    def forward(self, texts_or_batch):
        features = self.extractor(
            texts_or_batch if isinstance(texts_or_batch, dict) else list(texts_or_batch)
        )
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

    def forward(self, texts_or_batch):
        features = self.extractor(
            texts_or_batch if isinstance(texts_or_batch, dict) else list(texts_or_batch)
        )
        return self.head(features).squeeze(-1)


class _FoldTextDataset(Dataset):
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


class _FoldTextBatchCollator:
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
        field_names = [
            key
            for key in items[0].keys()
            if key not in {"position", "text"}
        ] if items else []
        for name in field_names:
            batch[name] = torch.as_tensor(
                [float(item[name]) for item in items],
                dtype=torch.float32,
            )
        return batch


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
        self.coverage_filter_rows: List[Dict[str, Any]] = []
        self.association_filter_rows: List[Dict[str, Any]] = []
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
        confounders = self._discover_extract_filter_with_retries(
            stage="confounder",
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            train_idx=train_idx,
            attention_rows=nuisance["attention"],
            existing_specs=self._initial_specs(),
        )
        r_stage = self._crossfit_effect(discovery_df, nuisance["predictions"], outer_fold)
        modifiers = self._discover_extract_filter_with_retries(
            stage="effect_modifier",
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            train_idx=train_idx,
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

    def _make_text_loader(
        self,
        model: nn.Module,
        df: pd.DataFrame,
        positions: Sequence[int],
        *,
        fields: Optional[Dict[str, np.ndarray]] = None,
        shuffle: bool = False,
        total_folds: Optional[int] = None,
    ) -> DataLoader:
        extractor = getattr(model, "extractor", None)
        text_preprocessor = None
        if extractor is not None and hasattr(extractor, "make_batch_preprocessor"):
            text_preprocessor = extractor.make_batch_preprocessor()
        workers = self._data_loader_workers(total_folds=total_folds)
        loader_kwargs: Dict[str, Any] = {
            "batch_size": max(1, int(self.config.training.batch_size)),
            "shuffle": bool(shuffle),
            "collate_fn": _FoldTextBatchCollator(text_preprocessor),
            "num_workers": workers,
            "pin_memory": self.device.type == "cuda",
        }
        if workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(
            _FoldTextDataset(
                texts=df[self.config.text_column].astype(str).tolist(),
                positions=positions,
                fields=fields,
            ),
            **loader_kwargs,
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
        checkpoint_fingerprint = self._crossfit_checkpoint_fingerprint("nuisance", folds)

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            cached = self._load_nuisance_fold_checkpoint(
                df=df,
                outer_fold=outer_fold,
                fold=fold,
                heldout_pos=heldout_pos,
                fingerprint=checkpoint_fingerprint,
            )
            if cached is not None:
                return cached

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
                propensity_auroc = _safe_roc_auc(t, e_hat)
                outcome_auroc = (
                    _safe_roc_auc(y, m_hat)
                    if self.config.outcome_type != "continuous"
                    else None
                )
                logger.info(
                    "Outer fold %s nuisance fold %s/%s heldout metrics: "
                    "propensity_auroc=%s outcome_auroc=%s",
                    outer_fold,
                    fold,
                    folds,
                    _format_optional_metric(propensity_auroc),
                    _format_optional_metric(outcome_auroc),
                )
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
                result = {
                    "fold": fold,
                    "heldout_pos": heldout_pos,
                    "e_hat": e_hat,
                    "m_hat": m_hat,
                    "y_resid": y_resid,
                    "t_resid": t_resid,
                    "attention": fold_attention,
                }
                self._save_nuisance_fold_checkpoint(
                    df=df,
                    result=result,
                    outer_fold=outer_fold,
                    fingerprint=checkpoint_fingerprint,
                )
                return result
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
        logger.info(
            "Outer fold %s nuisance cross-fit parallelism: folds=%s n_jobs=%s "
            "setting=%s device=%s",
            outer_fold,
            folds,
            n_jobs,
            self.avf_config.fold_parallelism,
            self.device,
        )
        fold_results = _run_crossfit_fold_tasks(run_fold, split_items, n_jobs)

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
        checkpoint_fingerprint = self._crossfit_checkpoint_fingerprint(
            "r_stage",
            folds,
            extra_payload={
                "e_hat_hash": _hash_numeric_array(e),
                "m_hat_hash": _hash_numeric_array(m),
            },
        )

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            cached = self._load_effect_fold_checkpoint(
                df=df,
                outer_fold=outer_fold,
                fold=fold,
                heldout_pos=heldout_pos,
                fingerprint=checkpoint_fingerprint,
            )
            if cached is not None:
                return cached

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
                result = {
                    "fold": fold,
                    "heldout_pos": heldout_pos,
                    "tau_hat": tau_hat,
                    "r_loss": heldout_r_loss,
                    "attention": fold_attention,
                }
                self._save_effect_fold_checkpoint(
                    df=df,
                    result=result,
                    outer_fold=outer_fold,
                    fingerprint=checkpoint_fingerprint,
                )
                return result
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
        logger.info(
            "Outer fold %s effect cross-fit parallelism: folds=%s n_jobs=%s "
            "setting=%s device=%s",
            outer_fold,
            folds,
            n_jobs,
            self.avf_config.fold_parallelism,
            self.device,
        )
        fold_results = _run_crossfit_fold_tasks(run_fold, split_items, n_jobs)

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
        train_loader = self._make_text_loader(
            model,
            df,
            positions,
            fields={
                "t": df[self.config.treatment_column].to_numpy(dtype=np.float32),
                "y": df[self.config.outcome_column].to_numpy(dtype=np.float32),
            },
            shuffle=True,
            total_folds=total_folds,
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=train_config.learning_rate,
            weight_decay=getattr(train_config, "weight_decay", 0.01),
        )
        num_batches = max(1, len(train_loader))
        scheduler = _make_linear_lr_scheduler(optimizer, train_config, num_batches)
        progress_every = max(1, num_batches // 5)
        logger.info(
            "Outer fold %s nuisance fold %s/%s: training for %s epoch(s), "
            "batch_size=%s, batches/epoch=%s, dataloader_workers=%s, "
            "lr=%.3g, lr_schedule=%s%s",
            outer_fold,
            fold,
            total_folds,
            train_config.epochs,
            train_config.batch_size,
            num_batches,
            train_loader.num_workers,
            _current_lr(optimizer),
            "linear" if scheduler is not None else "none",
            self._cuda_memory_summary(),
        )
        for epoch in range(1, train_config.epochs + 1):
            model.train()
            loss_sum = 0.0
            prop_sum = 0.0
            outcome_sum = 0.0
            batch_count = 0
            for batch_idx, batch in enumerate(train_loader, start=1):
                t = batch["t"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                t_logit, y_pred = model(batch["model_input"])
                prop_loss = F.binary_cross_entropy_with_logits(t_logit, t)
                if self.config.outcome_type == "continuous":
                    outcome_loss = F.mse_loss(y_pred, y)
                else:
                    outcome_loss = F.binary_cross_entropy_with_logits(y_pred, y)
                loss = outcome_loss + self.config.training.alpha_propensity * prop_loss
                loss.backward()
                self._clip_and_step(model, optimizer, scheduler)
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
                        "batch %s/%s loss=%.4f outcome=%.4f propensity=%.4f lr=%.3g%s",
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
                        _current_lr(optimizer),
                        self._cuda_memory_summary(),
                    )
            denom = max(1, batch_count)
            logger.info(
                "Outer fold %s nuisance fold %s/%s epoch %s/%s complete: "
                "loss=%.4f outcome=%.4f propensity=%.4f lr=%.3g%s",
                outer_fold,
                fold,
                total_folds,
                epoch,
                train_config.epochs,
                loss_sum / denom,
                outcome_sum / denom,
                prop_sum / denom,
                _current_lr(optimizer),
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
        train_loader = self._make_text_loader(
            model,
            df,
            positions,
            fields={
                "target": np.asarray(targets, dtype=np.float32),
                "weight": np.asarray(weights, dtype=np.float32),
            },
            shuffle=True,
            total_folds=total_folds,
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=train_config.learning_rate,
            weight_decay=getattr(train_config, "weight_decay", 0.01),
        )
        num_batches = max(1, len(train_loader))
        scheduler = _make_linear_lr_scheduler(optimizer, train_config, num_batches)
        progress_every = max(1, num_batches // 5)
        logger.info(
            "Outer fold %s effect fold %s/%s: training for %s epoch(s), "
            "batch_size=%s, batches/epoch=%s, dataloader_workers=%s, "
            "lr=%.3g, lr_schedule=%s%s",
            outer_fold,
            fold,
            total_folds,
            train_config.epochs,
            train_config.batch_size,
            num_batches,
            train_loader.num_workers,
            _current_lr(optimizer),
            "linear" if scheduler is not None else "none",
            self._cuda_memory_summary(),
        )
        for epoch in range(1, train_config.epochs + 1):
            model.train()
            loss_sum = 0.0
            batch_count = 0
            for batch_idx, batch in enumerate(train_loader, start=1):
                target = batch["target"].to(self.device, non_blocking=True)
                weight = batch["weight"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                tau = model(batch["model_input"])
                loss = torch.mean(weight * torch.square(target - tau))
                loss.backward()
                self._clip_and_step(model, optimizer, scheduler)
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
                        "batch %s/%s r_loss=%.4f lr=%.3g%s",
                        outer_fold,
                        fold,
                        total_folds,
                        epoch,
                        train_config.epochs,
                        batch_idx,
                        num_batches,
                        loss_value,
                        _current_lr(optimizer),
                        self._cuda_memory_summary(),
                    )
            logger.info(
                "Outer fold %s effect fold %s/%s epoch %s/%s complete: "
                "r_loss=%.4f lr=%.3g%s",
                outer_fold,
                fold,
                total_folds,
                epoch,
                train_config.epochs,
                loss_sum / max(1, batch_count),
                _current_lr(optimizer),
                self._cuda_memory_summary(),
            )

    def _clip_and_step(self, model: nn.Module, optimizer, scheduler=None) -> None:
        clip_norm = getattr(self.config.training, "gradient_clip_norm", 0.0)
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    def _predict_nuisance_model(self, model: _NuisanceNet, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        prop = []
        outcome = []
        loader = self._make_text_loader(
            model,
            df,
            np.arange(len(df), dtype=int),
            shuffle=False,
        )
        with torch.no_grad():
            for batch in loader:
                t_logit, y_pred = model(batch["model_input"])
                prop.append(torch.sigmoid(t_logit).cpu().numpy())
                if self.config.outcome_type == "continuous":
                    outcome.append(y_pred.cpu().numpy())
                else:
                    outcome.append(torch.sigmoid(y_pred).cpu().numpy())
        return np.concatenate(prop), np.concatenate(outcome)

    def _predict_effect_model(self, model: _EffectNet, df: pd.DataFrame) -> np.ndarray:
        model.eval()
        tau = []
        loader = self._make_text_loader(
            model,
            df,
            np.arange(len(df), dtype=int),
            shuffle=False,
        )
        with torch.no_grad():
            for batch in loader:
                tau.append(model(batch["model_input"]).cpu().numpy())
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
        proposal_attempt: int = 1,
        rejected_low_coverage: Optional[Sequence[Dict[str, Any]]] = None,
        rejected_low_signal: Optional[Sequence[Dict[str, Any]]] = None,
        multivariable_signal_feedback: Optional[Dict[str, Any]] = None,
        excluded_feature_names: Optional[Sequence[str]] = None,
    ) -> List[ExplicitFeatureSpec]:
        proposals_by_fold: Dict[int, List[ExplicitFeatureSpec]] = {}
        proposal_limit = self._candidate_proposal_limit()
        excluded_names = {
            _normalize_feature_name(name)
            for name in (excluded_feature_names or [])
            if _normalize_feature_name(name)
        }
        for fold in sorted({int(row["fold"]) for row in attention_rows if row.get("fold") is not None}):
            fold_rows = [row for row in attention_rows if int(row.get("fold")) == fold]
            context = self._build_agent_context(
                stage=stage,
                outer_fold=outer_fold,
                inner_fold=fold,
                discovery_df=discovery_df,
                attention_rows=fold_rows,
                existing_specs=existing_specs,
                proposal_attempt=proposal_attempt,
                max_proposals=proposal_limit,
                rejected_low_coverage=rejected_low_coverage or [],
                rejected_low_signal=rejected_low_signal or [],
                multivariable_signal_feedback=multivariable_signal_feedback or {},
                excluded_feature_names=sorted(excluded_names),
            )
            self._save_agent_candidate_checkpoint(
                {
                    "outer_fold": outer_fold,
                    "fold": fold,
                    "stage": stage,
                    "proposal_attempt": int(proposal_attempt),
                    "status": "started",
                    "context": self._stored_agent_context(context),
                },
                stage=stage,
                outer_fold=outer_fold,
                fold=fold,
            )
            try:
                raw = self.proposal_agent.propose(context)
            except Exception as exc:
                error_row = {
                    "outer_fold": outer_fold,
                    "fold": fold,
                    "stage": stage,
                    "proposal_attempt": int(proposal_attempt),
                    "status": "error",
                    "context": self._stored_agent_context(context),
                    "error": str(exc),
                }
                if getattr(self.agent_search_config, "save_agent_raw_output", False):
                    error_row["agent_raw_output"] = _get_agent_response_trace(
                        self.proposal_agent
                    )
                self._save_agent_candidate_checkpoint(
                    error_row,
                    stage=stage,
                    outer_fold=outer_fold,
                    fold=fold,
                )
                raise
            raw_proposals = _proposal_list(raw)
            specs = _proposal_dicts_to_specs(
                raw_proposals,
                required_role=stage,
                max_specs=proposal_limit,
                excluded_feature_names=excluded_names,
            )
            proposals_by_fold[fold] = specs
            row = {
                "outer_fold": outer_fold,
                "fold": fold,
                "stage": stage,
                "proposal_attempt": int(proposal_attempt),
                "status": "complete",
                "context": self._stored_agent_context(context),
                "proposals": _proposal_artifact_dicts(raw_proposals, specs),
            }
            if getattr(self.agent_search_config, "save_agent_raw_output", False):
                row["agent_raw_output"] = _get_agent_response_trace(self.proposal_agent)
            if stage == "confounder":
                self.confounder_candidate_rows.append(row)
            else:
                self.modifier_candidate_rows.append(row)
            self._save_agent_candidate_checkpoint(
                row,
                stage=stage,
                outer_fold=outer_fold,
                fold=fold,
            )
            self._flush_agent_candidate_rows()

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

    def _discover_extract_filter_with_retries(
        self,
        stage: str,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        train_idx: np.ndarray,
        attention_rows: Sequence[Dict[str, Any]],
        existing_specs: Sequence[ExplicitFeatureSpec],
    ) -> List[ExplicitFeatureSpec]:
        kept_specs: List[ExplicitFeatureSpec] = []
        rejected_low_coverage: List[Dict[str, Any]] = []
        rejected_low_signal: List[Dict[str, Any]] = []
        multivariable_signal_feedback: Dict[str, Any] = {}
        excluded_names: set[str] = set()
        max_attempts = 1 + max(
            0,
            int(self.avf_config.coverage_retry_attempts),
            int(getattr(self.avf_config, "signal_retry_attempts", 0)),
        )

        for attempt in range(1, max_attempts + 1):
            current_specs = self._merge_specs(existing_specs, kept_specs)
            candidates = self._discover_variables_from_attention(
                stage=stage,
                outer_fold=outer_fold,
                discovery_df=discovery_df,
                attention_rows=attention_rows,
                existing_specs=current_specs,
                proposal_attempt=attempt,
                rejected_low_coverage=rejected_low_coverage,
                rejected_low_signal=rejected_low_signal,
                multivariable_signal_feedback=multivariable_signal_feedback,
                excluded_feature_names=sorted(excluded_names),
            )
            current_names = {_normalize_feature_name(spec.name) for spec in current_specs}
            candidates = [
                spec
                for spec in candidates
                if _normalize_feature_name(spec.name) not in current_names
                and _normalize_feature_name(spec.name) not in excluded_names
            ]
            if not candidates:
                break

            self.dataset = self.extraction_provider.ensure_features(self.dataset, candidates)
            train_df = self.dataset.iloc[train_idx].copy()
            coverage_kept, coverage_dropped = self._partition_specs_by_extraction_coverage(
                train_df,
                candidates,
                manual_specs=[],
            )
            self.coverage_filter_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "stage": stage,
                    "proposal_attempt": int(attempt),
                    "candidate_features": [spec.name for spec in candidates],
                    "kept_features": [spec.name for spec in coverage_kept],
                    "dropped_features": coverage_dropped,
                }
            )
            self._flush_coverage_filter_rows()

            signal_kept, signal_dropped = self._partition_specs_by_association_signal(
                train_df=train_df,
                stage=stage,
                specs=coverage_kept,
                existing_specs=current_specs,
            )
            kept_specs = self._merge_specs(kept_specs, signal_kept)
            multivariable_signal_feedback = self._multivariable_signal_summary(
                train_df=train_df,
                stage=stage,
                specs=self._merge_specs(existing_specs, kept_specs),
            )
            self.association_filter_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "stage": stage,
                    "proposal_attempt": int(attempt),
                    "candidate_features": [spec.name for spec in coverage_kept],
                    "kept_features": [spec.name for spec in signal_kept],
                    "dropped_features": signal_dropped,
                    "multivariable_signal": multivariable_signal_feedback,
                }
            )
            self._flush_association_filter_rows()

            dropped = [*coverage_dropped, *signal_dropped]
            signal_inadequate = not bool(
                multivariable_signal_feedback.get("adequate", True)
            )

            if (not dropped and not signal_inadequate) or attempt >= max_attempts:
                break

            for row in coverage_dropped:
                name = _normalize_feature_name(row.get("name", ""))
                if name:
                    excluded_names.add(name)
                    rejected_low_coverage.append(row)
            for row in signal_dropped:
                name = _normalize_feature_name(row.get("name", ""))
                if name:
                    excluded_names.add(name)
                    rejected_low_signal.append(row)

        return kept_specs

    def _stored_agent_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if getattr(self.agent_search_config, "save_agent_context", False):
            return context
        return _scrub_context(context)

    def _build_agent_context(
        self,
        stage: str,
        outer_fold: int,
        inner_fold: int,
        discovery_df: pd.DataFrame,
        attention_rows: Sequence[Dict[str, Any]],
        existing_specs: Sequence[ExplicitFeatureSpec],
        proposal_attempt: int = 1,
        max_proposals: Optional[int] = None,
        rejected_low_coverage: Optional[Sequence[Dict[str, Any]]] = None,
        rejected_low_signal: Optional[Sequence[Dict[str, Any]]] = None,
        multivariable_signal_feedback: Optional[Dict[str, Any]] = None,
        excluded_feature_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        evidence = sorted(
            attention_rows,
            key=lambda row: abs(float(row.get("attention", 0.0))),
            reverse=True,
        )[: max(1, self.avf_config.attention_top_k_chunks * 20)]
        instruction = (
            "Propose only pre-treatment confounder variables directly supported by repeated high-attention token spans inside high-attention chunks."
            if stage == "confounder"
            else "Propose only pre-treatment effect modifier variables directly supported by repeated high-attention token spans inside high R-stage attention chunks."
        )
        return {
            "prompt_version": "agentic_attention_variable_forest_v1",
            "stage": stage,
            "outer_fold": outer_fold,
            "fold": inner_fold,
            "proposal_attempt": int(proposal_attempt),
            "max_proposals": int(max_proposals or self._candidate_proposal_limit()),
            "instruction": instruction,
            "clinical_question": self.config.clinical_question,
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "current_features": [_spec_to_dict(spec) for spec in existing_specs],
            "excluded_feature_names": list(excluded_feature_names or []),
            "rejected_low_coverage_features": list(rejected_low_coverage or []),
            "rejected_low_signal_features": list(rejected_low_signal or []),
            "multivariable_signal_feedback": multivariable_signal_feedback or {},
            "attention_evidence": [
                self._attention_evidence_context_row(row) for row in evidence
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

    def _attention_evidence_context_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        context_row: Dict[str, Any] = {
            "row_id": int(row["row_id"]),
            "chunk_text": row["chunk_text"],
            "attention": float(row["attention"]),
        }
        for key in [
            "e_hat",
            "m_hat",
            "y_residual",
            "t_residual",
            "tau_hat_r_stage",
            "r_loss",
        ]:
            if key in row:
                context_row[key] = row[key]
        spans = _parse_top_token_spans(row.get("top_token_spans_json"))
        if spans:
            context_row["top_token_spans"] = spans
            summary = row.get("attended_token_summary")
            if isinstance(summary, str) and summary:
                context_row["attended_token_summary"] = summary
            highlighted = row.get("highlighted_chunk_text")
            if isinstance(highlighted, str) and highlighted:
                context_row["highlighted_chunk_text"] = highlighted
        return context_row

    def _candidate_proposal_limit(self) -> int:
        configured = int(getattr(self.avf_config, "candidate_proposals_per_fold", 3))
        agent_limit = int(getattr(self.agent_search_config, "max_additions_per_iter", configured))
        return max(1, min(configured, max(1, agent_limit)))

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
        setting = str(self.avf_config.fold_parallelism).strip().lower()
        if setting == "auto":
            if self.device.type != "cpu":
                return 1
            return max(1, min(int(self.num_workers), int(folds)))
        return max(1, min(int(setting), int(folds)))

    def _data_loader_workers(self, total_folds: Optional[int] = None) -> int:
        env_workers = os.environ.get("OCI_AVF_DATALOADER_WORKERS")
        if env_workers is not None:
            return max(0, int(env_workers))
        if total_folds is not None and self._fold_n_jobs(total_folds) > 1:
            return 0
        return max(0, int(self.num_workers or 0))

    def _crossfit_checkpoint_fingerprint(
        self,
        stage: str,
        folds: int,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        arch = self.config.architecture
        train = self.config.training
        arch_keys = [
            "feature_extractor_type",
            "htr_sentence_model",
            "htr_freeze_sentence_encoder",
            "htr_chunk_size_words",
            "htr_chunk_overlap_words",
            "htr_max_chunks",
            "htr_max_chunk_length",
            "htr_num_layers",
            "htr_num_heads",
            "htr_transformer_dim",
            "htr_dropout",
            "htr_projection_dim",
            "htr_hash_embedding_dim",
            "htr_sentence_encoder_batch_size",
            "htr_sentence_encoder_backend",
            "htr_sentence_pooling",
            "htr_normalize_sentence_embeddings",
            "htr_trainable_sentence_encoder_layers",
            "causal_head_hidden_outcome_dim",
        ]
        train_keys = [
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "alpha_propensity",
            "lr_schedule",
        ]
        payload = {
            "stage": stage,
            "folds": int(folds),
            "outcome_type": self.config.outcome_type,
            "text_column": self.config.text_column,
            "outcome_column": self.config.outcome_column,
            "treatment_column": self.config.treatment_column,
            "attention_top_k_chunks": self.avf_config.attention_top_k_chunks,
            "e_clip": self.avf_config.e_clip,
            "architecture": {key: getattr(arch, key, None) for key in arch_keys},
            "training": {key: getattr(train, key, None) for key in train_keys},
        }
        if extra_payload:
            payload["extra"] = extra_payload
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _fold_checkpoint_paths(
        self,
        stage: str,
        outer_fold: int,
        fold: int,
    ) -> Dict[str, Path]:
        stage_dir = self.artifact_dir / "crossfit_fold_checkpoints" / stage
        stem = f"outer_{int(outer_fold):03d}_fold_{int(fold):03d}"
        return {
            "predictions": stage_dir / f"{stem}_predictions.parquet",
            "attention": stage_dir / f"{stem}_attention.parquet",
            "done": stage_dir / f"{stem}.done.json",
        }

    def _load_fold_checkpoint(
        self,
        stage: str,
        df: pd.DataFrame,
        outer_fold: int,
        fold: int,
        heldout_pos: np.ndarray,
        fingerprint: str,
    ) -> Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
        paths = self._fold_checkpoint_paths(stage, outer_fold, fold)
        if not (
            paths["done"].exists()
            and paths["predictions"].exists()
            and paths["attention"].exists()
        ):
            return None
        try:
            with open(paths["done"]) as f:
                marker = json.load(f)
            if marker.get("fingerprint") != fingerprint:
                logger.info(
                    "Ignoring stale %s checkpoint for outer fold %s fold %s: "
                    "fingerprint mismatch",
                    stage,
                    outer_fold,
                    fold,
                )
                return None

            pred_df = pd.read_parquet(paths["predictions"])
            attention_df = pd.read_parquet(paths["attention"])
            expected_ids = df.iloc[heldout_pos]["_oci_row_id"].to_numpy()
            if "_oci_row_id" not in pred_df.columns or pred_df["_oci_row_id"].duplicated().any():
                logger.warning(
                    "Ignoring invalid %s checkpoint for outer fold %s fold %s: "
                    "missing or duplicate _oci_row_id",
                    stage,
                    outer_fold,
                    fold,
                )
                return None
            pred_by_id = pred_df.set_index("_oci_row_id", drop=False)
            if not set(expected_ids).issubset(set(pred_by_id.index)):
                logger.info(
                    "Ignoring stale %s checkpoint for outer fold %s fold %s: "
                    "heldout row IDs changed",
                    stage,
                    outer_fold,
                    fold,
                )
                return None
            pred_df = pred_by_id.loc[expected_ids].reset_index(drop=True)
            if len(pred_df) != len(expected_ids):
                logger.info(
                    "Ignoring stale %s checkpoint for outer fold %s fold %s: "
                    "heldout row count changed",
                    stage,
                    outer_fold,
                    fold,
                )
                return None
            attention_rows = attention_df.to_dict("records")
            logger.info(
                "Outer fold %s %s fold %s: loaded cached checkpoint "
                "predictions=%s attention_rows=%s",
                outer_fold,
                stage,
                fold,
                len(pred_df),
                len(attention_rows),
            )
            return pred_df, attention_rows
        except Exception as exc:
            logger.warning(
                "Ignoring unreadable %s checkpoint for outer fold %s fold %s: %s",
                stage,
                outer_fold,
                fold,
                exc,
            )
            return None

    def _save_fold_checkpoint(
        self,
        stage: str,
        outer_fold: int,
        fold: int,
        predictions: pd.DataFrame,
        attention_rows: Sequence[Dict[str, Any]],
        fingerprint: str,
    ) -> None:
        paths = self._fold_checkpoint_paths(stage, outer_fold, fold)
        paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
        attention_df = pd.DataFrame(attention_rows)
        if attention_df.empty:
            attention_df = pd.DataFrame(columns=["row_id", "fold", "stage", "outer_fold"])
        _write_parquet_atomic(predictions, paths["predictions"])
        _write_parquet_atomic(attention_df, paths["attention"])
        _write_json_atomic(
            {
                "stage": stage,
                "outer_fold": int(outer_fold),
                "fold": int(fold),
                "n_predictions": int(len(predictions)),
                "n_attention_rows": int(len(attention_df)),
                "fingerprint": fingerprint,
            },
            paths["done"],
        )
        logger.info(
            "Outer fold %s %s fold %s: saved checkpoint predictions=%s "
            "attention_rows=%s",
            outer_fold,
            stage,
            fold,
            len(predictions),
            len(attention_df),
        )

    def _agent_candidate_checkpoint_path(
        self,
        stage: str,
        outer_fold: int,
        fold: int,
        proposal_attempt: int = 1,
    ) -> Path:
        stage_dir = self.artifact_dir / "agent_candidate_checkpoints" / stage
        stem = f"outer_{int(outer_fold):03d}_fold_{int(fold):03d}"
        if int(proposal_attempt) > 1:
            stem = f"{stem}_attempt_{int(proposal_attempt):03d}"
        return stage_dir / f"{stem}.json"

    def _save_agent_candidate_checkpoint(
        self,
        row: Dict[str, Any],
        stage: str,
        outer_fold: int,
        fold: int,
    ) -> None:
        path = self._agent_candidate_checkpoint_path(
            stage,
            outer_fold,
            fold,
            proposal_attempt=int(row.get("proposal_attempt", 1)),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(row, path)
        logger.info(
            "Outer fold %s %s fold %s: saved agent checkpoint status=%s path=%s",
            outer_fold,
            stage,
            fold,
            row.get("status", "unknown"),
            path,
        )

    def _flush_agent_candidate_rows(self) -> None:
        _write_jsonl(
            self.artifact_dir / "confounder_candidates_by_fold.jsonl",
            self.confounder_candidate_rows,
        )
        _write_jsonl(
            self.artifact_dir / "effect_modifier_candidates_by_fold.jsonl",
            self.modifier_candidate_rows,
        )

    def _flush_coverage_filter_rows(self) -> None:
        _write_jsonl(
            self.artifact_dir / "coverage_filter_by_attempt.jsonl",
            self.coverage_filter_rows,
        )

    def _flush_association_filter_rows(self) -> None:
        _write_jsonl(
            self.artifact_dir / "association_filter_by_attempt.jsonl",
            self.association_filter_rows,
        )

    def _load_nuisance_fold_checkpoint(
        self,
        df: pd.DataFrame,
        outer_fold: int,
        fold: int,
        heldout_pos: np.ndarray,
        fingerprint: str,
    ) -> Optional[Dict[str, Any]]:
        loaded = self._load_fold_checkpoint(
            "nuisance",
            df,
            outer_fold,
            fold,
            heldout_pos,
            fingerprint,
        )
        if loaded is None:
            return None
        pred_df, attention_rows = loaded
        return {
            "fold": fold,
            "heldout_pos": np.asarray(heldout_pos),
            "e_hat": pred_df["e_hat"].to_numpy(dtype=float),
            "m_hat": pred_df["m_hat"].to_numpy(dtype=float),
            "y_resid": pred_df["y_residual"].to_numpy(dtype=float),
            "t_resid": pred_df["t_residual"].to_numpy(dtype=float),
            "attention": attention_rows,
        }

    def _save_nuisance_fold_checkpoint(
        self,
        df: pd.DataFrame,
        result: Dict[str, Any],
        outer_fold: int,
        fingerprint: str,
    ) -> None:
        heldout_pos = np.asarray(result["heldout_pos"], dtype=int)
        predictions = pd.DataFrame(
            {
                "heldout_pos": heldout_pos,
                "_oci_row_id": df.iloc[heldout_pos]["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "nuisance_fold": int(result["fold"]),
                "e_hat": np.asarray(result["e_hat"], dtype=float),
                "m_hat": np.asarray(result["m_hat"], dtype=float),
                "y_residual": np.asarray(result["y_resid"], dtype=float),
                "t_residual": np.asarray(result["t_resid"], dtype=float),
            }
        )
        predictions["r_loss_at_zero_tau"] = predictions["y_residual"] ** 2
        self._save_fold_checkpoint(
            "nuisance",
            outer_fold,
            int(result["fold"]),
            predictions,
            result["attention"],
            fingerprint,
        )

    def _load_effect_fold_checkpoint(
        self,
        df: pd.DataFrame,
        outer_fold: int,
        fold: int,
        heldout_pos: np.ndarray,
        fingerprint: str,
    ) -> Optional[Dict[str, Any]]:
        loaded = self._load_fold_checkpoint(
            "r_stage",
            df,
            outer_fold,
            fold,
            heldout_pos,
            fingerprint,
        )
        if loaded is None:
            return None
        pred_df, attention_rows = loaded
        return {
            "fold": fold,
            "heldout_pos": np.asarray(heldout_pos),
            "tau_hat": pred_df["tau_hat_r_stage"].to_numpy(dtype=float),
            "r_loss": pred_df["r_loss"].to_numpy(dtype=float),
            "attention": attention_rows,
        }

    def _save_effect_fold_checkpoint(
        self,
        df: pd.DataFrame,
        result: Dict[str, Any],
        outer_fold: int,
        fingerprint: str,
    ) -> None:
        heldout_pos = np.asarray(result["heldout_pos"], dtype=int)
        predictions = pd.DataFrame(
            {
                "heldout_pos": heldout_pos,
                "_oci_row_id": df.iloc[heldout_pos]["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "effect_fold": int(result["fold"]),
                "tau_hat_r_stage": np.asarray(result["tau_hat"], dtype=float),
                "r_loss": np.asarray(result["r_loss"], dtype=float),
            }
        )
        self._save_fold_checkpoint(
            "r_stage",
            outer_fold,
            int(result["fold"]),
            predictions,
            result["attention"],
            fingerprint,
        )

    def _filter_specs_by_extraction_coverage(
        self,
        df: pd.DataFrame,
        specs: Sequence[ExplicitFeatureSpec],
        manual_specs: Sequence[ExplicitFeatureSpec],
    ) -> List[ExplicitFeatureSpec]:
        kept, _ = self._partition_specs_by_extraction_coverage(
            df,
            specs,
            manual_specs,
        )
        return kept

    def _partition_specs_by_extraction_coverage(
        self,
        df: pd.DataFrame,
        specs: Sequence[ExplicitFeatureSpec],
        manual_specs: Sequence[ExplicitFeatureSpec],
    ) -> Tuple[List[ExplicitFeatureSpec], List[Dict[str, Any]]]:
        manual_names = {_normalize_feature_name(spec.name) for spec in manual_specs}
        kept: List[ExplicitFeatureSpec] = []
        dropped: List[Dict[str, Any]] = []
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
                dropped.append(
                    {
                        "name": name,
                        "type": spec.type,
                        "roles": list(spec.roles),
                        "description": spec.description,
                        "coverage": float(coverage),
                        "min_extraction_coverage": float(
                            self.avf_config.min_extraction_coverage
                        ),
                    }
                )
                continue
            kept.append(spec)
        return kept, dropped

    def _partition_specs_by_association_signal(
        self,
        train_df: pd.DataFrame,
        stage: str,
        specs: Sequence[ExplicitFeatureSpec],
        existing_specs: Sequence[ExplicitFeatureSpec],
    ) -> Tuple[List[ExplicitFeatureSpec], List[Dict[str, Any]]]:
        kept: List[ExplicitFeatureSpec] = []
        dropped: List[Dict[str, Any]] = []
        alpha = float(getattr(self.avf_config, "association_alpha", 0.05))
        for spec in specs:
            diagnostic = _feature_association_diagnostic(
                df=train_df,
                spec=spec,
                config=self.config,
                existing_specs=existing_specs,
                alpha=alpha,
                min_n=int(getattr(self.avf_config, "association_min_n", 20)),
                min_non_missing=int(
                    getattr(self.avf_config, "association_min_non_missing", 10)
                ),
            )
            if diagnostic.get("status") == "skipped_insufficient_sample":
                kept.append(spec)
                continue

            if stage == "confounder":
                keep = bool(
                    diagnostic.get("treatment_associated")
                    and diagnostic.get("outcome_associated")
                )
                rejection_reason = "no_joint_treatment_outcome_association"
            else:
                keep = bool(
                    diagnostic.get("outcome_associated")
                    or diagnostic.get("interaction_associated")
                )
                rejection_reason = "no_outcome_or_interaction_association"

            if keep:
                kept.append(spec)
                continue

            dropped.append(
                {
                    "name": spec.name,
                    "type": spec.type,
                    "roles": list(spec.roles),
                    "description": spec.description,
                    "rejection_reason": rejection_reason,
                    "diagnostic": diagnostic,
                }
            )
        return kept, dropped

    def _multivariable_signal_summary(
        self,
        train_df: pd.DataFrame,
        stage: str,
        specs: Sequence[ExplicitFeatureSpec],
    ) -> Dict[str, Any]:
        specs = list(specs)
        if not specs:
            return {
                "status": "no_features",
                "adequate": False,
                "reason": "no_features_survived_association_screen",
            }
        min_n = int(getattr(self.avf_config, "association_min_n", 20))
        if len(train_df) < min_n:
            return {
                "status": "skipped_insufficient_sample",
                "adequate": True,
                "n": int(len(train_df)),
                "min_n": min_n,
            }

        matrix, feature_names = _signal_feature_matrix(train_df, specs)
        if matrix is None or matrix.shape[1] == 0 or not _has_any_variation(matrix):
            return {
                "status": "no_varying_features",
                "adequate": False,
                "feature_names": feature_names,
            }

        folds = int(getattr(self.avf_config, "signal_cv_folds", 3))
        treatment_score = _cross_validated_boosted_signal_score(
            matrix,
            train_df[self.config.treatment_column].to_numpy(),
            target_kind="binary",
            folds=folds,
            random_state=71,
        )
        outcome_score = _cross_validated_boosted_signal_score(
            matrix,
            train_df[self.config.outcome_column].to_numpy(),
            target_kind=self.config.outcome_type,
            folds=folds,
            random_state=173,
        )
        min_treatment = float(getattr(self.avf_config, "min_signal_treatment_auroc", 0.55))
        min_outcome = float(getattr(self.avf_config, "min_signal_outcome_auroc", 0.55))
        treatment_ok = _score_meets_signal_threshold(treatment_score, min_treatment)
        outcome_ok = _score_meets_signal_threshold(outcome_score, min_outcome)
        adequate = bool(
            (treatment_ok and outcome_ok)
            if stage == "confounder"
            else outcome_ok
        )
        return {
            "status": "ok",
            "adequate": adequate,
            "stage": stage,
            "required": (
                "treatment_and_outcome_auroc"
                if stage == "confounder"
                else "outcome_auroc"
            ),
            "min_treatment_auroc": min_treatment,
            "min_outcome_auroc": min_outcome,
            "treatment_ok": bool(treatment_ok),
            "outcome_ok": bool(outcome_ok),
            "treatment_model": treatment_score,
            "outcome_model": outcome_score,
            "feature_names": feature_names,
            "features": [spec.name for spec in specs],
        }

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
        _write_jsonl(
            self.artifact_dir / "coverage_filter_by_attempt.jsonl",
            self.coverage_filter_rows,
        )
        _write_jsonl(
            self.artifact_dir / "association_filter_by_attempt.jsonl",
            self.association_filter_rows,
        )
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
    max_specs: Optional[int] = None,
    excluded_feature_names: Optional[Sequence[str]] = None,
) -> List[ExplicitFeatureSpec]:
    raw_proposals = _proposal_list(raw_proposals)
    excluded = {
        _normalize_feature_name(name)
        for name in (excluded_feature_names or [])
        if _normalize_feature_name(name)
    }
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
        if name in excluded:
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
            if max_specs is not None and len(specs) >= int(max_specs):
                break
        except ValueError:
            continue
    return specs


def _proposal_list(raw_proposals: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_proposals, dict):
        raw_proposals = raw_proposals.get("proposals", [])
    if not isinstance(raw_proposals, list):
        return []
    return [proposal for proposal in raw_proposals if isinstance(proposal, dict)]


def _proposal_artifact_dicts(
    raw_proposals: Sequence[Dict[str, Any]],
    specs: Sequence[ExplicitFeatureSpec],
) -> List[Dict[str, Any]]:
    raw_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for proposal in raw_proposals:
        name = _normalize_feature_name(proposal.get("name", ""))
        if name:
            raw_by_name.setdefault(name, []).append(proposal)

    artifacts: List[Dict[str, Any]] = []
    for spec in specs:
        row = _spec_to_dict(spec)
        raw = (raw_by_name.get(_normalize_feature_name(spec.name)) or [{}]).pop(0)
        for key in ["action", "rationale", "expected_signal"]:
            value = raw.get(key)
            if value is not None:
                row[key] = value
        artifacts.append(row)
    return artifacts


def _feature_association_diagnostic(
    df: pd.DataFrame,
    spec: ExplicitFeatureSpec,
    config: AppliedInferenceConfig,
    existing_specs: Sequence[ExplicitFeatureSpec],
    alpha: float,
    min_n: int,
    min_non_missing: int,
) -> Dict[str, Any]:
    col = f"explicit_feat_{spec.name}"
    miss_col = f"{col}_missing"
    coverage = _feature_coverage(df, _normalize_feature_name(spec.name))
    if len(df) < min_n:
        return {
            "status": "skipped_insufficient_sample",
            "n": int(len(df)),
            "min_n": int(min_n),
            "coverage": float(coverage),
            "treatment_associated": True,
            "outcome_associated": True,
            "interaction_associated": False,
        }
    if col not in df.columns:
        return {
            "status": "missing_extracted_column",
            "coverage": 0.0,
            "treatment_associated": False,
            "outcome_associated": False,
            "interaction_associated": False,
        }
    missing = df[miss_col].astype(bool).to_numpy() if miss_col in df.columns else df[col].isna().to_numpy()
    non_missing_n = int((~missing).sum())
    if non_missing_n < min_non_missing:
        return {
            "status": "insufficient_non_missing",
            "n": int(len(df)),
            "coverage": float(coverage),
            "non_missing_n": non_missing_n,
            "min_non_missing": int(min_non_missing),
            "treatment_associated": False,
            "outcome_associated": False,
            "interaction_associated": False,
        }

    treatment_diag = _univariate_feature_target_association(
        df,
        spec,
        config.treatment_column,
        target_kind="binary",
    )
    outcome_diag = _univariate_feature_target_association(
        df,
        spec,
        config.outcome_column,
        target_kind=config.outcome_type,
    )
    interaction_diag = _treatment_interaction_association(
        df=df,
        spec=spec,
        config=config,
        existing_specs=existing_specs,
    )
    treatment_p = treatment_diag.get("p_value")
    outcome_p = outcome_diag.get("p_value")
    interaction_p = interaction_diag.get("p_value")
    return {
        "status": "ok",
        "n": int(len(df)),
        "coverage": float(coverage),
        "non_missing_n": non_missing_n,
        "alpha": float(alpha),
        "treatment_association": treatment_diag,
        "outcome_association": outcome_diag,
        "treatment_interaction": interaction_diag,
        "treatment_associated": _p_value_below(treatment_p, alpha),
        "outcome_associated": _p_value_below(outcome_p, alpha),
        "interaction_associated": _p_value_below(interaction_p, alpha),
    }


def _univariate_feature_target_association(
    df: pd.DataFrame,
    spec: ExplicitFeatureSpec,
    target_col: str,
    target_kind: str,
) -> Dict[str, Any]:
    col = f"explicit_feat_{spec.name}"
    miss_col = f"{col}_missing"
    if col not in df.columns or target_col not in df.columns:
        return {"status": "missing_column"}
    missing = df[miss_col].astype(bool) if miss_col in df.columns else df[col].isna()
    target = df[target_col]
    mask = (~missing) & df[col].notna() & target.notna()
    if int(mask.sum()) < 3:
        return {"status": "insufficient_rows", "n": int(mask.sum())}

    y_raw = target.loc[mask]
    x_raw = df.loc[mask, col]
    target_kind = "continuous" if target_kind == "continuous" else "binary"
    if target_kind == "binary":
        y_codes, uniques = pd.factorize(y_raw)
        if len(uniques) != 2:
            return {"status": "constant_target", "n": int(mask.sum())}
        y = y_codes.astype(float)
        if spec.type == "continuous":
            x = pd.to_numeric(x_raw, errors="coerce")
            finite = x.notna().to_numpy()
            if finite.sum() < 3 or len(np.unique(x[finite])) < 2:
                return {"status": "constant_feature", "n": int(finite.sum())}
            try:
                stat, p_value = stats.pointbiserialr(y[finite], x[finite].to_numpy(dtype=float))
            except Exception as exc:
                return {"status": "test_failed", "error": str(exc)}
            return {
                "status": "ok",
                "test": "point_biserial",
                "statistic": _finite_or_none(stat),
                "p_value": _finite_or_none(p_value),
                "n": int(finite.sum()),
            }
        table = pd.crosstab(x_raw.astype(str), y_raw.astype(str))
        if table.shape[0] < 2 or table.shape[1] != 2:
            return {"status": "constant_feature", "n": int(mask.sum())}
        try:
            chi2, p_value, dof, _ = stats.chi2_contingency(table.to_numpy())
        except Exception as exc:
            return {"status": "test_failed", "error": str(exc)}
        return {
            "status": "ok",
            "test": "chi_square",
            "statistic": _finite_or_none(chi2),
            "p_value": _finite_or_none(p_value),
            "dof": int(dof),
            "n": int(mask.sum()),
        }

    y = pd.to_numeric(y_raw, errors="coerce")
    if spec.type == "continuous":
        x = pd.to_numeric(x_raw, errors="coerce")
        finite = x.notna() & y.notna()
        if int(finite.sum()) < 3 or x[finite].nunique() < 2 or y[finite].nunique() < 2:
            return {"status": "constant_feature_or_target", "n": int(finite.sum())}
        try:
            stat, p_value = stats.pearsonr(x[finite].to_numpy(dtype=float), y[finite].to_numpy(dtype=float))
        except Exception as exc:
            return {"status": "test_failed", "error": str(exc)}
        return {
            "status": "ok",
            "test": "pearson",
            "statistic": _finite_or_none(stat),
            "p_value": _finite_or_none(p_value),
            "n": int(finite.sum()),
        }

    groups = []
    for _, values in y.groupby(x_raw.astype(str)):
        values = values.dropna().to_numpy(dtype=float)
        if len(values) >= 2:
            groups.append(values)
    if len(groups) < 2:
        return {"status": "insufficient_groups", "n": int(mask.sum())}
    try:
        stat, p_value = stats.f_oneway(*groups)
    except Exception as exc:
        return {"status": "test_failed", "error": str(exc)}
    return {
        "status": "ok",
        "test": "anova",
        "statistic": _finite_or_none(stat),
        "p_value": _finite_or_none(p_value),
        "n": int(mask.sum()),
        "n_groups": int(len(groups)),
    }


def _treatment_interaction_association(
    df: pd.DataFrame,
    spec: ExplicitFeatureSpec,
    config: AppliedInferenceConfig,
    existing_specs: Sequence[ExplicitFeatureSpec],
) -> Dict[str, Any]:
    if config.outcome_type == "continuous":
        return _continuous_interaction_association(df, spec, config, existing_specs)
    if config.treatment_column not in df.columns or config.outcome_column not in df.columns:
        return {"status": "missing_target_column"}
    outcome = np.asarray(df[config.outcome_column].to_numpy(), dtype=float)
    treatment = np.asarray(df[config.treatment_column].to_numpy(), dtype=float)
    if len(np.unique(outcome[~np.isnan(outcome)])) < 2 or len(np.unique(treatment[~np.isnan(treatment)])) < 2:
        return {"status": "constant_target_or_treatment"}

    current_confounders = [
        item
        for item in existing_specs
        if item.name != spec.name and "confounder" in item.roles
    ]
    _, w_matrix, _, _, _, _ = _build_features(df, current_confounders)
    candidate = ExplicitFeatureSpec(
        name=spec.name,
        type=spec.type,
        categories=spec.categories,
        description=spec.description,
        roles=["confounder"],
    )
    _, z_matrix, _, z_names, _, _ = _build_features(df, [candidate])
    w_matrix = _feature_matrix_or_empty(w_matrix, len(df))
    z_matrix = _feature_matrix_or_empty(z_matrix, len(df))
    if z_matrix.shape[1] == 0 or not _has_any_variation(z_matrix):
        return {"status": "constant_candidate", "candidate_feature_names": z_names}

    treatment_col = treatment.reshape(-1, 1)
    base_x = np.hstack([w_matrix, treatment_col, z_matrix])
    full_x = np.hstack([base_x, z_matrix * treatment_col])
    finite = (
        np.isfinite(outcome)
        & np.isfinite(treatment)
        & np.all(np.isfinite(base_x), axis=1)
        & np.all(np.isfinite(full_x), axis=1)
    )
    if int(finite.sum()) < 10:
        return {"status": "insufficient_finite_rows", "n": int(finite.sum())}
    try:
        p_value, lr_stat, dof = _binary_likelihood_ratio_p(
            base_x[finite],
            full_x[finite],
            outcome[finite],
            added_df=z_matrix.shape[1],
        )
    except Exception as exc:
        return {"status": "test_failed", "error": str(exc)}
    return {
        "status": "ok",
        "test": "logistic_likelihood_ratio",
        "p_value": _finite_or_none(p_value),
        "lr_statistic": _finite_or_none(lr_stat),
        "dof": int(dof),
        "candidate_feature_names": z_names,
        "n": int(finite.sum()),
    }


def _continuous_interaction_association(
    df: pd.DataFrame,
    spec: ExplicitFeatureSpec,
    config: AppliedInferenceConfig,
    existing_specs: Sequence[ExplicitFeatureSpec],
) -> Dict[str, Any]:
    outcome = np.asarray(df[config.outcome_column].to_numpy(), dtype=float)
    treatment = np.asarray(df[config.treatment_column].to_numpy(), dtype=float)
    current_confounders = [
        item
        for item in existing_specs
        if item.name != spec.name and "confounder" in item.roles
    ]
    _, w_matrix, _, _, _, _ = _build_features(df, current_confounders)
    candidate = ExplicitFeatureSpec(
        name=spec.name,
        type=spec.type,
        categories=spec.categories,
        description=spec.description,
        roles=["confounder"],
    )
    _, z_matrix, _, z_names, _, _ = _build_features(df, [candidate])
    w_matrix = _feature_matrix_or_empty(w_matrix, len(df))
    z_matrix = _feature_matrix_or_empty(z_matrix, len(df))
    if z_matrix.shape[1] == 0 or not _has_any_variation(z_matrix):
        return {"status": "constant_candidate", "candidate_feature_names": z_names}
    treatment_col = treatment.reshape(-1, 1)
    base_x = np.hstack([w_matrix, treatment_col, z_matrix])
    full_x = np.hstack([base_x, z_matrix * treatment_col])
    finite = (
        np.isfinite(outcome)
        & np.isfinite(treatment)
        & np.all(np.isfinite(base_x), axis=1)
        & np.all(np.isfinite(full_x), axis=1)
    )
    if int(finite.sum()) < 10:
        return {"status": "insufficient_finite_rows", "n": int(finite.sum())}
    try:
        p_value, f_stat, dof_num, dof_den = _linear_nested_f_test(
            base_x[finite],
            full_x[finite],
            outcome[finite],
            added_df=z_matrix.shape[1],
        )
    except Exception as exc:
        return {"status": "test_failed", "error": str(exc)}
    return {
        "status": "ok",
        "test": "linear_nested_f",
        "p_value": _finite_or_none(p_value),
        "f_statistic": _finite_or_none(f_stat),
        "dof_num": int(dof_num),
        "dof_den": int(dof_den),
        "candidate_feature_names": z_names,
        "n": int(finite.sum()),
    }


def _signal_feature_matrix(
    df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
) -> Tuple[Optional[np.ndarray], List[str]]:
    x_matrix, w_matrix, x_names, w_names, _, _ = _build_features(df, list(specs))
    matrix = _hstack_present(x_matrix, w_matrix)
    names = [*(x_names or []), *(w_names or [])]
    if matrix is None:
        return None, names
    return np.asarray(matrix, dtype=np.float32), names


def _cross_validated_boosted_signal_score(
    matrix: np.ndarray,
    target: np.ndarray,
    target_kind: str,
    folds: int,
    random_state: int,
) -> Dict[str, Any]:
    x = np.asarray(matrix, dtype=np.float32)
    y = np.asarray(target)
    finite = np.all(np.isfinite(x), axis=1) & pd.Series(y).notna().to_numpy()
    x = x[finite]
    y = y[finite]
    if len(y) < 10:
        return {"status": "insufficient_rows", "n": int(len(y))}
    if target_kind != "continuous":
        y_codes, uniques = pd.factorize(y)
        if len(uniques) != 2:
            return {"status": "constant_target", "n": int(len(y))}
        y_binary = y_codes.astype(int)
        class_counts = np.bincount(y_binary)
        n_splits = min(int(folds), int(class_counts.min()))
        if n_splits < 2:
            return {
                "status": "insufficient_class_counts",
                "n": int(len(y)),
                "class_counts": class_counts.tolist(),
            }
        preds = np.full(len(y_binary), np.nan, dtype=float)
        model_name = None
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y_binary)):
            pred, model_name = _fit_predict_boosted_classifier(
                x[train_idx],
                y_binary[train_idx],
                x[test_idx],
                random_state=random_state + fold_idx,
            )
            preds[test_idx] = pred
        mask = np.isfinite(preds)
        return {
            "status": "ok",
            "target_kind": "binary",
            "metric": "auroc",
            "score": _safe_roc_auc(y_binary[mask], preds[mask]),
            "model": model_name,
            "n": int(mask.sum()),
            "folds": int(n_splits),
        }

    y_cont = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(y_cont)
    x = x[finite]
    y_cont = y_cont[finite]
    if len(y_cont) < 10 or np.std(y_cont) == 0:
        return {"status": "insufficient_or_constant_target", "n": int(len(y_cont))}
    n_splits = min(int(folds), len(y_cont))
    if n_splits < 2:
        return {"status": "insufficient_rows", "n": int(len(y_cont))}
    preds = np.full(len(y_cont), np.nan, dtype=float)
    model_name = None
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x)):
        pred, model_name = _fit_predict_boosted_regressor(
            x[train_idx],
            y_cont[train_idx],
            x[test_idx],
            random_state=random_state + fold_idx,
        )
        preds[test_idx] = pred
    mask = np.isfinite(preds)
    return {
        "status": "ok",
        "target_kind": "continuous",
        "metric": "r2",
        "score": float(r2_score(y_cont[mask], preds[mask])),
        "model": model_name,
        "n": int(mask.sum()),
        "folds": int(n_splits),
    }


def _fit_predict_boosted_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, str]:
    try:
        from xgboost import XGBClassifier  # type: ignore

        model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(train_x, train_y)
        return model.predict_proba(test_x)[:, 1], "xgboost.XGBClassifier"
    except Exception:
        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=random_state,
        )
        model.fit(train_x, train_y)
        return model.predict_proba(test_x)[:, 1], "sklearn.HistGradientBoostingClassifier"


def _fit_predict_boosted_regressor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, str]:
    try:
        from xgboost import XGBRegressor  # type: ignore

        model = XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(train_x, train_y)
        return model.predict(test_x), "xgboost.XGBRegressor"
    except Exception:
        model = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=random_state,
        )
        model.fit(train_x, train_y)
        return model.predict(test_x), "sklearn.HistGradientBoostingRegressor"


def _score_meets_signal_threshold(score: Dict[str, Any], threshold: float) -> bool:
    if score.get("status") == "skipped_insufficient_sample":
        return True
    value = score.get("score")
    if value is None or not np.isfinite(float(value)):
        return False
    if score.get("metric") == "auroc":
        return float(value) >= float(threshold)
    return float(value) > 0.0


def _binary_likelihood_ratio_p(
    base_x: np.ndarray,
    full_x: np.ndarray,
    y: np.ndarray,
    added_df: int,
) -> Tuple[float, float, int]:
    y_codes, uniques = pd.factorize(y)
    if len(uniques) != 2:
        raise ValueError("binary likelihood ratio requires two outcome classes")
    y_binary = y_codes.astype(int)
    base_model = LogisticRegression(max_iter=1000, solver="lbfgs")
    full_model = LogisticRegression(max_iter=1000, solver="lbfgs")
    base_model.fit(_ensure_model_columns(base_x), y_binary)
    full_model.fit(_ensure_model_columns(full_x), y_binary)
    base_pred = np.clip(base_model.predict_proba(_ensure_model_columns(base_x))[:, 1], 1e-6, 1 - 1e-6)
    full_pred = np.clip(full_model.predict_proba(_ensure_model_columns(full_x))[:, 1], 1e-6, 1 - 1e-6)
    base_ll = -log_loss(y_binary, base_pred, labels=[0, 1], normalize=False)
    full_ll = -log_loss(y_binary, full_pred, labels=[0, 1], normalize=False)
    lr_stat = max(0.0, 2.0 * (float(full_ll) - float(base_ll)))
    dof = max(1, int(added_df))
    return float(stats.chi2.sf(lr_stat, dof)), float(lr_stat), dof


def _linear_nested_f_test(
    base_x: np.ndarray,
    full_x: np.ndarray,
    y: np.ndarray,
    added_df: int,
) -> Tuple[float, float, int, int]:
    base_x = _ensure_model_columns(base_x)
    full_x = _ensure_model_columns(full_x)
    base_model = LinearRegression()
    full_model = LinearRegression()
    base_model.fit(base_x, y)
    full_model.fit(full_x, y)
    base_resid = y - base_model.predict(base_x)
    full_resid = y - full_model.predict(full_x)
    rss_base = float(np.sum(base_resid ** 2))
    rss_full = float(np.sum(full_resid ** 2))
    dof_num = max(1, int(added_df))
    dof_den = max(1, int(len(y) - full_x.shape[1] - 1))
    f_stat = max(0.0, ((rss_base - rss_full) / dof_num) / max(rss_full / dof_den, 1e-12))
    return float(stats.f.sf(f_stat, dof_num, dof_den)), float(f_stat), dof_num, dof_den


def _feature_matrix_or_empty(matrix: Optional[np.ndarray], n_rows: int) -> np.ndarray:
    if matrix is None:
        return np.zeros((n_rows, 0), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _ensure_model_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], 1), dtype=np.float64)
    return x


def _has_any_variation(matrix: np.ndarray) -> bool:
    matrix = np.asarray(matrix, dtype=float)
    return bool(matrix.size and np.any(np.nanstd(matrix, axis=0) > 1e-12))


def _p_value_below(value: Any, alpha: float) -> bool:
    try:
        return bool(value is not None and np.isfinite(float(value)) and float(value) < alpha)
    except (TypeError, ValueError):
        return False


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


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


def _run_crossfit_fold_tasks(run_fold, split_items, n_jobs: int) -> List[Dict[str, Any]]:
    if n_jobs <= 1:
        return [
            run_fold(fold, fit_pos, heldout_pos)
            for fold, (fit_pos, heldout_pos) in split_items
        ]
    with ThreadPoolExecutor(
        max_workers=int(n_jobs),
        thread_name_prefix="avf-fold",
    ) as executor:
        futures = [
            executor.submit(run_fold, fold, fit_pos, heldout_pos)
            for fold, (fit_pos, heldout_pos) in split_items
        ]
        return [future.result() for future in futures]


def _make_linear_lr_scheduler(optimizer, train_config, steps_per_epoch: int):
    lr_schedule = str(getattr(train_config, "lr_schedule", "linear") or "").lower()
    if lr_schedule != "linear":
        return None
    total_steps = max(1, int(steps_per_epoch) * int(getattr(train_config, "epochs", 1)))
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_steps,
    )


def _current_lr(optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


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


def _format_optional_metric(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.4f}"


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


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_json_atomic(data: Dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _hash_numeric_array(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _parse_top_token_spans(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


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
