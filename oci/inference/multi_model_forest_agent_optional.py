"""Multi-model text-feature causal forest with optional final agent branch."""

from __future__ import annotations

import copy
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error
from sklearn.model_selection import KFold

from ..config import (
    AgenticAttentionVariableForestConfig,
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    BoWViewConfig,
    ExplicitFeatureForestConfig,
    MultiModelForestAgentOptionalConfig,
)
from ..models.causal_forest_head import CausalForestHead
from .agentic_attention_variable_forest import (
    AgenticAttentionVariableForestRunner,
    _EffectNet,
    _NuisanceNet,
    clip_probability,
)
from .agentic_explicit_feature_forest import (
    _fit_predict_outcome,
    _fit_predict_propensity,
    _r_loss,
    _safe_corr,
    _safe_roc_auc,
)
from .applied_explicit_feature_forest import _hstack_present
from .embedding_contrast_discovery import (
    EmbeddingContrastEvidenceGenerator,
    _binary_labels,
    _binary_mean_difference_direction,
    _normalize_rows,
    _normalize_vector,
    _residualize_embeddings,
    _residualize_vector_from_basis,
    _tail_labels,
)
from .multi_model_agentic_forest import (
    MultiModelHTREvidenceProvider,
    _agent_visible_metrics,
    _agentic_discovery_handoff_row,
    _align_htr_prediction_frame,
    _binary_split_items,
    _bounded_fold_count,
    _clinical_text_examples,
    _compact_multi_model_agent_context,
    _finite_or_none,
    build_multi_model_agentic_discovery_handoff,
    _fit_binary_bow_fold,
    _fit_regression_bow_fold,
    _fit_regressor,
    _htr_effect_metrics,
    _htr_nuisance_metrics,
    _make_bow_classifier,
    _make_bow_regressor,
    _make_bow_vectorizer,
    _model_feature_scores,
    _multi_view_importance,
    _normalize_texts,
    _split_is_honest,
    _top_feature_rows,
    _top_phrase_feature_rows,
    run_multi_model_agentic_forest_from_handoff,
    _write_json,
    _write_jsonl,
)

logger = logging.getLogger(__name__)


@dataclass
class _FeatureBundle:
    x_train: np.ndarray
    x_test: np.ndarray
    w_train: np.ndarray
    w_test: np.ndarray
    x_names: List[str]
    w_names: List[str]
    feature_rows: List[Dict[str, Any]]
    prediction_frames: List[pd.DataFrame]
    embedding_rows: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    handoff_evidence: Optional[Dict[str, Any]]


def _run_optional_outer_fold_job(
    *,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    outer_fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str,
    gpu_ids: Optional[Sequence[int]],
    num_workers: int,
    htr_dataloader_workers: Optional[int] = None,
) -> Dict[str, Any]:
    previous_dataloader_workers = os.environ.get("OCI_AVF_DATALOADER_WORKERS")
    if htr_dataloader_workers is not None:
        os.environ["OCI_AVF_DATALOADER_WORKERS"] = str(max(0, int(htr_dataloader_workers)))
    fold_runner = MultiModelForestAgentOptionalRunner(
        dataset=dataset.drop(columns=["_oci_row_id"], errors="ignore"),
        config=copy.deepcopy(config),
        output_path=output_path,
        device=torch.device(device),
        gpu_ids=gpu_ids,
        num_workers=num_workers,
    )
    try:
        predictions = fold_runner._run_one_analysis_split(
            outer_fold=outer_fold,
            train_idx=train_idx,
            test_idx=test_idx,
        )
        return {
            "outer_fold": int(outer_fold),
            "predictions": predictions,
            "feature_manifest_rows": fold_runner.feature_manifest_rows,
            "source_prediction_frames": fold_runner.source_prediction_frames,
            "embedding_feature_rows": fold_runner.embedding_feature_rows,
            "outer_metric_rows": fold_runner.outer_metric_rows,
            "agentic_handoff_rows": fold_runner.agentic_handoff_rows,
        }
    finally:
        if htr_dataloader_workers is not None:
            if previous_dataloader_workers is None:
                os.environ.pop("OCI_AVF_DATALOADER_WORKERS", None)
            else:
                os.environ["OCI_AVF_DATALOADER_WORKERS"] = previous_dataloader_workers


def run_multi_model_forest_agent_optional(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device=None,
    gpu_ids: Optional[Sequence[int]] = None,
    num_workers: int = 1,
    embedding_provider: Optional[Any] = None,
    htr_evidence_provider: Optional[Any] = None,
) -> None:
    """Run the non-agentic multi-model W/X causal forest path."""
    runner = MultiModelForestAgentOptionalRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        device=device,
        gpu_ids=gpu_ids,
        num_workers=num_workers,
        embedding_provider=embedding_provider,
        htr_evidence_provider=htr_evidence_provider,
    )
    runner.run()


class MultiModelOptionalHTRProvider(MultiModelHTREvidenceProvider):
    """HTR adapter with full outer-train -> outer-test prediction helpers."""

    @contextmanager
    def _temporary_effect_objective(self, objective: str):
        runner = self._runner
        if runner is None:
            raise RuntimeError("HTR runner has not been initialized")
        previous = getattr(runner.avf_config, "effect_objective", "pseudo_outcome_mse")
        runner.avf_config.effect_objective = str(objective)
        try:
            yield
        finally:
            runner.avf_config.effect_objective = previous

    def fit_effect_variant(
        self,
        discovery_df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
        *,
        effect_objective: str,
    ) -> Dict[str, Any]:
        runner = self._ensure_runner(discovery_df)
        with self._temporary_effect_objective(effect_objective):
            return runner._crossfit_effect(discovery_df, nuisance_predictions, outer_fold)

    def fit_nuisance_full_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        outer_fold: int,
    ) -> pd.DataFrame:
        runner = self._ensure_runner(train_df)
        model = None
        try:
            model = _NuisanceNet(
                extractor=runner._create_extractor(),
                hidden_dim=getattr(
                    runner.config.architecture,
                    "causal_head_hidden_outcome_dim",
                    64,
                ),
                outcome_type=runner.config.outcome_type,
            ).to(runner.device)
            positions = np.arange(len(train_df), dtype=int)
            runner._train_nuisance_model(
                model,
                train_df,
                positions,
                outer_fold=outer_fold,
                fold=0,
                total_folds=1,
            )
            e_hat, m_hat = runner._predict_nuisance_model(model, test_df)
        finally:
            if model is not None:
                runner._cleanup_model(model)
        return pd.DataFrame(
            {
                "_oci_row_id": test_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "e_hat": e_hat,
                "m_hat": m_hat,
                "model_family": "htr",
                "view_name": "htr_nuisance",
                "target_source": "htr_nuisance_outer_train_fit",
            }
        )

    def fit_effect_full_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
        *,
        effect_objective: str,
    ) -> pd.DataFrame:
        runner = self._ensure_runner(train_df)
        model = None
        r_df = train_df[["_oci_row_id"]].merge(
            nuisance_predictions.copy(),
            on="_oci_row_id",
            how="left",
            sort=False,
        )
        e = r_df["e_hat"].to_numpy(dtype=float)
        m = r_df["m_hat"].to_numpy(dtype=float)
        y = train_df[runner.config.outcome_column].to_numpy(dtype=float)
        t = train_df[runner.config.treatment_column].to_numpy(dtype=float)
        e_clipped = np.clip(e, runner.avf_config.e_clip, 1.0 - runner.avf_config.e_clip)
        m_clipped = clip_probability(m)
        t_resid = t - e_clipped
        y_resid = y - m
        try:
            model = _EffectNet(
                extractor=runner._create_extractor(),
                hidden_dim=getattr(
                    runner.config.architecture,
                    "causal_head_hidden_outcome_dim",
                    64,
                ),
            ).to(runner.device)
            positions = np.arange(len(train_df), dtype=int)
            with self._temporary_effect_objective(effect_objective):
                runner._train_effect_model(
                    model,
                    train_df,
                    positions,
                    y,
                    t,
                    e_clipped,
                    m_clipped,
                    y_resid,
                    t_resid,
                    outer_fold=outer_fold,
                    fold=0,
                    total_folds=1,
                )
                tau_hat = runner._predict_effect_model(model, test_df)
        finally:
            if model is not None:
                runner._cleanup_model(model)
        return pd.DataFrame(
            {
                "_oci_row_id": test_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "tau_hat_r_stage": tau_hat,
                "model_family": "htr",
                "view_name": f"htr_effect_{effect_objective}",
                "target_source": "ensemble_mean_nuisance_outer_train_fit",
                "effect_objective": effect_objective,
            }
        )


class MultiModelForestAgentOptionalRunner:
    """Primary non-agentic text-model W/X causal-forest runner."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        config: AppliedInferenceConfig,
        output_path: Path,
        device: Optional[Any] = None,
        gpu_ids: Optional[Sequence[int]] = None,
        num_workers: int = 1,
        embedding_provider: Optional[Any] = None,
        htr_evidence_provider: Optional[Any] = None,
    ) -> None:
        self.dataset = dataset.reset_index(drop=True).copy()
        self.dataset["_oci_row_id"] = np.arange(len(self.dataset), dtype=int)
        self.config = config
        self.output_path = Path(output_path)
        self.artifact_dir = self.output_path.parent / "multi_model_forest_agent_optional"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or "cpu")
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        self.num_workers = 1 if num_workers is None else int(num_workers)
        self.embedding_provider = embedding_provider
        self.htr_evidence_provider = htr_evidence_provider
        self.nn_config: MultiModelForestAgentOptionalConfig = getattr(
            config.architecture,
            "multi_model_forest_agent_optional",
            MultiModelForestAgentOptionalConfig(),
        )
        self.search_config: AgenticFeatureSearchConfig = getattr(
            config.architecture,
            "agentic_feature_search",
            AgenticFeatureSearchConfig(),
        )
        # Existing embedding and optional agentic code reads the old config slot.
        # Mirror the new config there so shared components use the same settings.
        config.architecture.multi_model_agentic_forest = self.nn_config
        self._sync_htr_fold_parallelism()
        self.cf_config: ExplicitFeatureForestConfig = getattr(
            config.architecture,
            "explicit_feature_forest",
            ExplicitFeatureForestConfig(),
        )
        self.embedding_evidence_generator: Optional[EmbeddingContrastEvidenceGenerator] = None
        self._default_htr_provider: Optional[MultiModelOptionalHTRProvider] = None

        self.prediction_results: Optional[pd.DataFrame] = None
        self.outer_metric_rows: List[Dict[str, Any]] = []
        self.split_provenance_rows: List[Dict[str, Any]] = []
        self.feature_manifest_rows: List[Dict[str, Any]] = []
        self.source_prediction_frames: List[pd.DataFrame] = []
        self.embedding_feature_rows: List[Dict[str, Any]] = []
        self.agentic_handoff_rows: List[Dict[str, Any]] = []

    def run(self) -> None:
        logger.info("=" * 80)
        logger.info("MULTI-MODEL FOREST WITH OPTIONAL AGENT BRANCH")
        logger.info("=" * 80)
        splits = self._analysis_splits()
        self.split_provenance_rows = self._split_provenance_rows(splits)
        if self._embedding_contrast_enabled() and self.embedding_provider is None:
            self._embedding_generator().prepare(self.dataset)

        outer_n_jobs = self._outer_n_jobs(len(splits))
        if outer_n_jobs > 1 and (
            self.embedding_provider is not None or self.htr_evidence_provider is not None
        ):
            logger.warning(
                "Outer fold parallelism disabled because custom embedding_provider "
                "or htr_evidence_provider objects were supplied."
            )
            outer_n_jobs = 1

        if outer_n_jobs > 1:
            outer_devices = self._outer_devices(outer_n_jobs)
            outer_backend = self._outer_backend_name()
            inner_workers = self._inner_workers_for_outer_job(outer_n_jobs)
            logger.info(
                "Running %s multi-model optional-agent outer folds with "
                "outer_parallelism=%s outer_backend=%s inner_workers_per_outer=%s "
                "devices=%s bow_fold_parallelism=%s htr_fold_parallelism=%s",
                len(splits),
                outer_n_jobs,
                outer_backend,
                inner_workers,
                [str(device) for device in outer_devices],
                self._bow_fold_parallelism_setting(),
                self._htr_fold_parallelism_setting(),
            )
            if outer_backend == "threads":
                with ThreadPoolExecutor(
                    max_workers=outer_n_jobs,
                    thread_name_prefix="mm-optional-outer",
                ) as executor:
                    futures = [
                        executor.submit(
                            self._run_one_analysis_split_isolated,
                            outer_fold=int(outer_fold),
                            train_idx=np.asarray(train_idx, dtype=int),
                            test_idx=np.asarray(test_idx, dtype=int),
                            device=outer_devices[(task_index - 1) % len(outer_devices)],
                            outer_n_jobs=outer_n_jobs,
                        )
                        for task_index, (outer_fold, train_idx, test_idx) in enumerate(
                            splits,
                            start=1,
                        )
                    ]
                    fold_results = [future.result() for future in futures]
            else:
                fold_results = Parallel(
                    n_jobs=outer_n_jobs,
                    backend="loky",
                    batch_size=1,
                    pre_dispatch="all",
                )(
                    delayed(_run_optional_outer_fold_job)(
                        dataset=self.dataset,
                        config=self.config,
                        output_path=self.output_path,
                        outer_fold=int(outer_fold),
                        train_idx=np.asarray(train_idx, dtype=int),
                        test_idx=np.asarray(test_idx, dtype=int),
                        device=str(outer_devices[(task_index - 1) % len(outer_devices)]),
                        gpu_ids=(
                            [int(outer_devices[(task_index - 1) % len(outer_devices)].index)]
                            if outer_devices[(task_index - 1) % len(outer_devices)].type == "cuda"
                            and outer_devices[(task_index - 1) % len(outer_devices)].index
                            is not None
                            else None
                        ),
                        num_workers=inner_workers,
                        htr_dataloader_workers=0,
                    )
                    for task_index, (outer_fold, train_idx, test_idx) in enumerate(
                        splits,
                        start=1,
                    )
                )
            fold_results = sorted(fold_results, key=lambda item: item["outer_fold"])
            prediction_frames = [item["predictions"] for item in fold_results]
            for item in fold_results:
                self.feature_manifest_rows.extend(item["feature_manifest_rows"])
                self.source_prediction_frames.extend(item["source_prediction_frames"])
                self.embedding_feature_rows.extend(item["embedding_feature_rows"])
                self.outer_metric_rows.extend(item["outer_metric_rows"])
                self.agentic_handoff_rows.extend(item.get("agentic_handoff_rows", []))
        else:
            prediction_frames = []
            for outer_fold, train_idx, test_idx in splits:
                logger.info(
                    "Multi-model optional-agent fold %s: train=%s test=%s device=%s",
                    outer_fold,
                    len(train_idx),
                    len(test_idx),
                    self.device,
                )
                prediction_frames.append(
                    self._run_one_analysis_split(
                        outer_fold=int(outer_fold),
                        train_idx=np.asarray(train_idx, dtype=int),
                        test_idx=np.asarray(test_idx, dtype=int),
                    )
                )

        results_df = pd.concat(prediction_frames).sort_values("_oci_row_id")
        self.prediction_results = results_df
        self._save_outputs(results_df)

        if bool(
            getattr(self.nn_config, "agentic_handoff_enabled", False)
            or getattr(self.nn_config, "agentic_explicit_branch_enabled", False)
        ):
            self._prepare_agentic_handoff()

        if bool(getattr(self.nn_config, "agentic_explicit_branch_enabled", False)):
            self._run_optional_agentic_branch()

    def _run_one_analysis_split_isolated(
        self,
        *,
        outer_fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        device: torch.device,
        outer_n_jobs: int,
    ) -> Dict[str, Any]:
        logger.info(
            "Multi-model optional-agent isolated fold %s: train=%s test=%s device=%s",
            outer_fold,
            len(train_idx),
            len(test_idx),
            device,
        )
        gpu_ids = None
        if device.type == "cuda" and device.index is not None:
            gpu_ids = [int(device.index)]
        return _run_optional_outer_fold_job(
            dataset=self.dataset,
            config=self.config,
            output_path=self.output_path,
            outer_fold=outer_fold,
            train_idx=train_idx,
            test_idx=test_idx,
            device=str(device),
            gpu_ids=gpu_ids,
            num_workers=self._inner_workers_for_outer_job(outer_n_jobs),
            htr_dataloader_workers=None,
        )

    def _analysis_splits(self) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        if self.config.cv_folds > 1:
            splits = KFold(
                n_splits=int(self.config.cv_folds),
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

        if bool(getattr(self.nn_config, "require_honest_outer_split", False)):
            raise ValueError(
                "multi_model_forest_agent_optional.require_honest_outer_split=True "
                "requires cv_folds > 1 or split_column with a 'test' split"
            )
        all_idx = np.arange(len(self.dataset), dtype=int)
        logger.warning(
            "No held-out split configured for multi_model_forest_agent_optional; "
            "predictions will be labeled full_data_refit_non_honest."
        )
        return [(1, all_idx, all_idx)]

    def _split_provenance_rows(
        self,
        splits: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        rows = []
        for outer_fold, train_idx, test_idx in splits:
            honest = _split_is_honest(train_idx, test_idx)
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "honest_outer_holdout": bool(honest),
                    "estimation_provenance": (
                        "honest_outer_fold" if honest else "full_data_refit_non_honest"
                    ),
                }
            )
        return rows

    def _run_one_analysis_split(
        self,
        *,
        outer_fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> pd.DataFrame:
        train_df = self.dataset.iloc[train_idx].reset_index(drop=True)
        test_df = self.dataset.iloc[test_idx].reset_index(drop=True)
        bundle = self._build_feature_bundle(
            train_df=train_df,
            test_df=test_df,
            outer_fold=outer_fold,
        )
        self.feature_manifest_rows.extend(bundle.feature_rows)
        self.embedding_feature_rows.extend(bundle.embedding_rows)
        self.source_prediction_frames.extend(bundle.prediction_frames)

        x_train, x_test = _clean_train_test_matrices(bundle.x_train, bundle.x_test)
        w_train, w_test = _clean_train_test_matrices(bundle.w_train, bundle.w_test)
        if x_train.shape[1] == 0:
            x_train = np.zeros((len(train_df), 1), dtype=np.float32)
            x_test = np.zeros((len(test_df), 1), dtype=np.float32)
            bundle.x_names.append("intercept_effect")
            bundle.feature_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "feature_name": "intercept_effect",
                    "feature_role": "X",
                    "source_family": "intercept",
                    "provenance": "fallback_no_effect_features",
                }
            )
        if w_train.shape[1] == 0:
            w_train = None
            w_test = None

        t_train = train_df[self.config.treatment_column].to_numpy(dtype=float)
        y_train = train_df[self.config.outcome_column].to_numpy(dtype=float)
        t_test = test_df[self.config.treatment_column].to_numpy(dtype=float)
        y_test = test_df[self.config.outcome_column].to_numpy(dtype=float)

        forest = CausalForestHead(
            n_estimators=self.cf_config.n_estimators,
            max_depth=self.cf_config.max_depth,
            min_samples_leaf=self.cf_config.min_samples_leaf,
            max_features=self.cf_config.max_features,
            honest=self.cf_config.honest,
            inference=self.cf_config.inference,
            random_state=42 + int(outer_fold),
        )
        forest.fit(X=x_train, T=t_train, Y=y_train, W=w_train)
        cf_preds = forest.predict(x_test, return_ci=True)
        tau = cf_preds["tau_pred"]

        nuisance_train = _hstack_present(x_train, w_train)
        nuisance_test = _hstack_present(x_test, w_test)
        if nuisance_train is None or nuisance_test is None:
            raise ValueError("Unable to build nuisance matrices for final predictions")
        propensity = _fit_predict_propensity(
            nuisance_train,
            t_train,
            nuisance_test,
            self.cf_config,
            random_state=142 + int(outer_fold),
        )
        outcome_pred = _fit_predict_outcome(
            nuisance_train,
            y_train,
            nuisance_test,
            self.config.outcome_type,
            self.cf_config,
            random_state=242 + int(outer_fold),
        )
        y0_prob = outcome_pred - propensity * tau
        y1_prob = outcome_pred + (1.0 - propensity) * tau
        if str(self.config.outcome_type).lower() == "binary":
            y0_prob = np.clip(y0_prob, 0.0, 1.0)
            y1_prob = np.clip(y1_prob, 0.0, 1.0)

        predictions = test_df.copy()
        honest = _split_is_honest(train_idx, test_idx)
        predictions["pred_ite_prob"] = tau
        predictions["pred_y0_prob"] = y0_prob
        predictions["pred_y1_prob"] = y1_prob
        predictions["pred_propensity_prob"] = propensity
        predictions["pred_outcome_prob"] = outcome_pred
        predictions["cv_fold"] = int(outer_fold)
        predictions["outer_fold"] = int(outer_fold)
        predictions["honest_outer_holdout"] = bool(honest)
        predictions["estimation_provenance"] = (
            "honest_outer_fold" if honest else "full_data_refit_non_honest"
        )
        predictions["selected_feature_names"] = ",".join(bundle.x_names + bundle.w_names)
        predictions["selected_feature_roles"] = json.dumps(
            {"X": bundle.x_names, "W": bundle.w_names}
        )
        predictions["selected_confounder_names"] = ",".join(bundle.w_names)
        predictions["selected_effect_modifier_names"] = ",".join(bundle.x_names)
        if "tau_lower" in cf_preds:
            predictions["pred_ite_lower"] = cf_preds["tau_lower"]
            predictions["pred_ite_upper"] = cf_preds["tau_upper"]

        metrics = {
            "outer_fold": int(outer_fold),
            "honest_outer_holdout": bool(honest),
            "estimation_provenance": (
                "honest_outer_fold" if honest else "full_data_refit_non_honest"
            ),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_x_features": int(x_train.shape[1]),
            "n_w_features": 0 if w_train is None else int(w_train.shape[1]),
            "x_feature_names": bundle.x_names,
            "w_feature_names": bundle.w_names,
            "ate_estimate": float(np.mean(tau)),
            "r_loss": _r_loss(y_test, t_test, outcome_pred, propensity, tau),
            "treatment_auroc": _safe_roc_auc(t_test, propensity),
            "feature_discovery_methods": self._enabled_feature_discovery_methods(),
            **bundle.metrics,
        }
        if str(self.config.outcome_type).lower() == "continuous":
            metrics["outcome_rmse"] = float(np.sqrt(mean_squared_error(y_test, outcome_pred)))
        else:
            metrics["outcome_auroc"] = _safe_roc_auc(y_test, outcome_pred)
        if "true_ite_prob" in test_df.columns:
            true_ite = test_df["true_ite_prob"].to_numpy(dtype=float)
            metrics["oracle_true_ite_corr"] = _safe_corr(true_ite, tau)
            metrics["oracle_true_ite_mae"] = float(np.mean(np.abs(true_ite - tau)))
        self.outer_metric_rows.append(metrics)

        if bundle.handoff_evidence is not None:
            handoff_result = copy.deepcopy(bundle.handoff_evidence)
            handoff_metrics = dict(handoff_result.get("metrics") or {})
            handoff_metrics.update(metrics)
            handoff_result["metrics"] = handoff_metrics
            handoff_result["context"] = self._build_primary_agent_context(
                outer_fold=outer_fold,
                discovery_df=train_df,
                metrics=handoff_metrics,
                importance=handoff_result.get("importance") or {},
                embedding_evidence=handoff_result.get("embedding_contrast_evidence") or {},
                htr_evidence=handoff_result.get("htr_evidence") or {},
            )
            self.agentic_handoff_rows.append(
                _agentic_discovery_handoff_row(
                    handoff_result,
                    fold_key=int(outer_fold),
                    outer_fold=int(outer_fold),
                    scope="full_outer_train",
                    n_rows=len(train_df),
                )
            )

        fold_dir = self.artifact_dir / f"outer_fold_{int(outer_fold):03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fold_dir / "feature_matrices.npz",
            x_train=x_train,
            x_test=x_test,
            w_train=np.zeros((len(train_df), 0), dtype=np.float32) if w_train is None else w_train,
            w_test=np.zeros((len(test_df), 0), dtype=np.float32) if w_test is None else w_test,
            x_feature_names=np.asarray(bundle.x_names, dtype=object),
            w_feature_names=np.asarray(bundle.w_names, dtype=object),
        )
        predictions.to_parquet(fold_dir / "predictions.parquet", index=False)
        return predictions

    def _build_feature_bundle(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        outer_fold: int,
    ) -> _FeatureBundle:
        texts_train = _normalize_texts(train_df[self.config.text_column].fillna(""))
        texts_test = _normalize_texts(test_df[self.config.text_column].fillna(""))
        y = train_df[self.config.outcome_column].to_numpy(dtype=float)
        t = train_df[self.config.treatment_column].to_numpy(dtype=float)
        x_train_cols: List[np.ndarray] = []
        x_test_cols: List[np.ndarray] = []
        w_train_cols: List[np.ndarray] = []
        w_test_cols: List[np.ndarray] = []
        x_names: List[str] = []
        w_names: List[str] = []
        feature_rows: List[Dict[str, Any]] = []
        prediction_frames: List[pd.DataFrame] = []
        embedding_rows: List[Dict[str, Any]] = []
        metrics: Dict[str, Any] = {}
        nuisance_train: List[Tuple[str, np.ndarray, np.ndarray]] = []
        nuisance_test: List[Tuple[str, np.ndarray, np.ndarray]] = []
        bow_nuisance_by_view: List[Dict[str, Any]] = []
        bow_view_results: List[Dict[str, Any]] = []
        ensemble_view_results: List[Dict[str, Any]] = []
        htr_evidence: Dict[str, Any] = {}

        if self._bow_enabled():
            for view_index, view in enumerate(self.nn_config.bow_views):
                e_train, e_test = self._fit_bow_binary_train_test(
                    texts_train,
                    texts_test,
                    t,
                    outer_fold=outer_fold,
                    view=view,
                    view_index=view_index,
                    label_name="treatment",
                )
                if str(self.config.outcome_type).lower() == "continuous":
                    m_train, m_test = self._fit_bow_regression_train_test(
                        texts_train,
                        texts_test,
                        y,
                        None,
                        outer_fold=outer_fold,
                        view=view,
                        view_index=view_index,
                        target_name="outcome",
                    )
                else:
                    m_train, m_test = self._fit_bow_binary_train_test(
                        texts_train,
                        texts_test,
                        y,
                        outer_fold=outer_fold,
                        view=view,
                        view_index=view_index,
                        label_name="outcome",
                    )
                nuisance_train.append((view.name, e_train, m_train))
                nuisance_test.append((view.name, e_test, m_test))
                bow_nuisance_by_view.append(
                    {
                        "view": view,
                        "view_index": int(view_index),
                        "e_hat": e_train,
                        "m_hat": m_train,
                    }
                )
                _append_feature(
                    w_train_cols,
                    w_test_cols,
                    w_names,
                    feature_rows,
                    train=e_train,
                    test=e_test,
                    name=f"bow__{view.name}__treatment_pred",
                    role="W",
                    source_family="bow",
                    outer_fold=outer_fold,
                    objective="treatment_nuisance",
                    provenance="inner_oof_train_outer_train_fit_test",
                    view_config=_bow_view_to_dict(view),
                )
                _append_feature(
                    w_train_cols,
                    w_test_cols,
                    w_names,
                    feature_rows,
                    train=m_train,
                    test=m_test,
                    name=f"bow__{view.name}__outcome_pred",
                    role="W",
                    source_family="bow",
                    outer_fold=outer_fold,
                    objective="outcome_nuisance",
                    provenance="inner_oof_train_outer_train_fit_test",
                    view_config=_bow_view_to_dict(view),
                )
                prediction_frames.append(
                    _source_prediction_frame(
                        train_df,
                        test_df,
                        outer_fold=outer_fold,
                        source_name=f"bow__{view.name}__nuisance",
                        values={
                            "e_hat": (e_train, e_test),
                            "m_hat": (m_train, m_test),
                        },
                    )
                )

        htr_train_result = None
        htr_test_predictions = None
        if self._htr_enabled():
            htr_train_result = self._htr_provider().fit_nuisance(train_df, outer_fold)
            htr_train_predictions = _align_htr_prediction_frame(
                htr_train_result.get("predictions"),
                train_df,
                required_columns=["e_hat", "m_hat"],
                source="htr_nuisance",
            )
            htr_test_predictions = self._htr_provider().fit_nuisance_full_predict(
                train_df,
                test_df,
                outer_fold,
            )
            htr_test_predictions = _align_htr_prediction_frame(
                htr_test_predictions,
                test_df,
                required_columns=["e_hat", "m_hat"],
                source="htr_nuisance_outer_train_fit",
            )
            htr_e_train = htr_train_predictions["e_hat"].to_numpy(dtype=float)
            htr_m_train = htr_train_predictions["m_hat"].to_numpy(dtype=float)
            htr_e_test = htr_test_predictions["e_hat"].to_numpy(dtype=float)
            htr_m_test = htr_test_predictions["m_hat"].to_numpy(dtype=float)
            nuisance_train.append(("htr_nuisance", htr_e_train, htr_m_train))
            nuisance_test.append(("htr_nuisance", htr_e_test, htr_m_test))
            _append_feature(
                w_train_cols,
                w_test_cols,
                w_names,
                feature_rows,
                train=htr_e_train,
                test=htr_e_test,
                name="htr__nuisance__treatment_pred",
                role="W",
                source_family="htr",
                outer_fold=outer_fold,
                objective="treatment_nuisance",
                provenance="inner_oof_train_outer_train_fit_test",
            )
            _append_feature(
                w_train_cols,
                w_test_cols,
                w_names,
                feature_rows,
                train=htr_m_train,
                test=htr_m_test,
                name="htr__nuisance__outcome_pred",
                role="W",
                source_family="htr",
                outer_fold=outer_fold,
                objective="outcome_nuisance",
                provenance="inner_oof_train_outer_train_fit_test",
            )
            metrics["htr_treatment_auroc"] = _safe_roc_auc(t, htr_e_train)
            if str(self.config.outcome_type).lower() == "continuous":
                metrics["htr_outcome_rmse"] = float(np.sqrt(mean_squared_error(y, htr_m_train)))
            else:
                metrics["htr_outcome_auroc"] = _safe_roc_auc(y, htr_m_train)
            htr_attention = [dict(row) for row in htr_train_result.get("attention", []) or []]
            for row in htr_attention:
                row.setdefault("model_family", "htr")
                row.setdefault("target_source", "htr_nuisance")
            htr_evidence["nuisance"] = {
                "metrics": _htr_nuisance_metrics(
                    discovery_df=train_df,
                    predictions=htr_train_predictions,
                    treatment_column=self.config.treatment_column,
                    outcome_column=self.config.outcome_column,
                    outcome_type=self.config.outcome_type,
                ),
                "attention": htr_attention,
            }
            prediction_frames.append(
                _source_prediction_frame(
                    train_df,
                    test_df,
                    outer_fold=outer_fold,
                    source_name="htr__nuisance",
                    values={
                        "e_hat": (htr_e_train, htr_e_test),
                        "m_hat": (htr_m_train, htr_m_test),
                    },
                )
            )

        if not nuisance_train:
            raise ValueError(
                "multi_model_forest_agent_optional requires at least one nuisance source"
            )
        e_train = np.nanmean(np.vstack([item[1] for item in nuisance_train]), axis=0)
        m_train = np.nanmean(np.vstack([item[2] for item in nuisance_train]), axis=0)
        e_test = np.nanmean(np.vstack([item[1] for item in nuisance_test]), axis=0)
        m_test = np.nanmean(np.vstack([item[2] for item in nuisance_test]), axis=0)
        e_train_clip = np.clip(e_train, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip)
        t_resid = t - e_train_clip
        y_resid = y - m_train
        pseudo_target = y_resid / t_resid
        r_weight = np.square(t_resid)
        metrics.update(
            {
                "n_nuisance_sources": int(len(nuisance_train)),
                "nuisance_sources": [item[0] for item in nuisance_train],
                "ensemble_treatment_auroc": _safe_roc_auc(t, e_train),
                "ensemble_pseudo_target_mean": _finite_or_none(np.mean(pseudo_target)),
                "ensemble_pseudo_target_std": _finite_or_none(np.std(pseudo_target)),
            }
        )
        if str(self.config.outcome_type).lower() == "continuous":
            metrics["ensemble_outcome_rmse"] = float(np.sqrt(mean_squared_error(y, m_train)))
        else:
            metrics["ensemble_outcome_auroc"] = _safe_roc_auc(y, m_train)
        ensemble_nuisance_train = pd.DataFrame(
            {
                "_oci_row_id": train_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "e_hat": e_train,
                "m_hat": m_train,
                "y_residual": y_resid,
                "t_residual": t_resid,
                "r_pseudo_outcome": pseudo_target,
                "pseudo_target": pseudo_target,
                "r_loss_at_zero_tau": np.square(y_resid),
                "target_source": "ensemble_mean_nuisance",
            }
        )
        prediction_frames.append(
            _source_prediction_frame(
                train_df,
                test_df,
                outer_fold=outer_fold,
                source_name="ensemble_mean_nuisance",
                values={
                    "e_hat": (e_train, e_test),
                    "m_hat": (m_train, m_test),
                },
            )
        )

        if self._bow_enabled():
            for view_index, view in enumerate(self.nn_config.bow_views):
                pseudo_train, pseudo_test = self._fit_bow_regression_train_test(
                    texts_train,
                    texts_test,
                    pseudo_target,
                    None,
                    outer_fold=outer_fold,
                    view=view,
                    view_index=view_index,
                    target_name="effect_pseudo_target",
                    seed_offset=50_000,
                )
                r_train, r_test = self._fit_bow_regression_train_test(
                    texts_train,
                    texts_test,
                    pseudo_target,
                    r_weight,
                    outer_fold=outer_fold,
                    view=view,
                    view_index=view_index,
                    target_name="effect_weighted_r",
                    seed_offset=70_000,
                )
                _append_feature(
                    x_train_cols,
                    x_test_cols,
                    x_names,
                    feature_rows,
                    train=pseudo_train,
                    test=pseudo_test,
                    name=f"bow__{view.name}__effect_pseudo_target_pred",
                    role="X",
                    source_family="bow",
                    outer_fold=outer_fold,
                    objective="r_pseudo_outcome",
                    provenance="inner_oof_train_outer_train_fit_test",
                    view_config=_bow_view_to_dict(view),
                )
                _append_feature(
                    x_train_cols,
                    x_test_cols,
                    x_names,
                    feature_rows,
                    train=r_train,
                    test=r_test,
                    name=f"bow__{view.name}__effect_weighted_r_tau_pred",
                    role="X",
                    source_family="bow",
                    outer_fold=outer_fold,
                    objective="direct_weighted_r",
                    provenance="inner_oof_train_outer_train_fit_test",
                    view_config=_bow_view_to_dict(view),
                )
                prediction_frames.append(
                    _source_prediction_frame(
                        train_df,
                        test_df,
                        outer_fold=outer_fold,
                        source_name=f"bow__{view.name}__effect",
                        values={
                            "tau_hat_pseudo_target": (pseudo_train, pseudo_test),
                            "tau_hat_weighted_r": (r_train, r_test),
                        },
                    )
                )
                nuisance_view = next(
                    (
                        item
                        for item in bow_nuisance_by_view
                        if int(item["view_index"]) == int(view_index)
                    ),
                    None,
                )
                if nuisance_view is not None:
                    importance = self._fit_primary_feature_importance_models(
                        texts=texts_train,
                        y=y,
                        t=t,
                        pseudo_target=pseudo_target,
                        pseudo_target_sample_weight=r_weight,
                        view=view,
                    )
                    view_metrics = self._primary_bow_metrics(
                        discovery_df=train_df,
                        y=y,
                        t=t,
                        e_hat=np.asarray(nuisance_view["e_hat"], dtype=float),
                        m_hat=np.asarray(nuisance_view["m_hat"], dtype=float),
                        pseudo_target=pseudo_target,
                        tau_hat=r_train,
                        y_resid=y_resid,
                        t_resid=t_resid,
                    )
                    bow_view_results.append(
                        {
                            "metrics": view_metrics,
                            "importance": importance,
                            "pseudo_target": pseudo_target,
                            "t_resid": t_resid,
                            "view": view,
                            "view_name": view.name,
                            "view_index": int(view_index),
                        }
                    )
                    ensemble_view_results.append(
                        {
                            "metrics": {
                                **view_metrics,
                                "target_source": "ensemble_mean_nuisance",
                            },
                            "importance": copy.deepcopy(importance),
                            "pseudo_target": pseudo_target,
                            "t_resid": t_resid,
                            "view": view,
                            "view_name": f"ensemble_r__{view.name}",
                            "view_index": int(view_index),
                        }
                    )

        if self._htr_enabled():
            htr_effect_variants: Dict[str, Any] = {}
            for effect_objective, feature_suffix in [
                ("pseudo_outcome_mse", "effect_pseudo_target_pred"),
                ("squared_r_loss", "effect_weighted_r_tau_pred"),
            ]:
                htr_effect_train = self._htr_provider().fit_effect_variant(
                    train_df,
                    ensemble_nuisance_train,
                    outer_fold,
                    effect_objective=effect_objective,
                )
                train_predictions = _align_htr_prediction_frame(
                    htr_effect_train.get("predictions"),
                    train_df,
                    required_columns=["tau_hat_r_stage"],
                    source=f"htr_effect_{effect_objective}",
                )
                test_predictions = self._htr_provider().fit_effect_full_predict(
                    train_df,
                    test_df,
                    ensemble_nuisance_train,
                    outer_fold,
                    effect_objective=effect_objective,
                )
                test_predictions = _align_htr_prediction_frame(
                    test_predictions,
                    test_df,
                    required_columns=["tau_hat_r_stage"],
                    source=f"htr_effect_{effect_objective}_outer_train_fit",
                )
                train_tau = train_predictions["tau_hat_r_stage"].to_numpy(dtype=float)
                test_tau = test_predictions["tau_hat_r_stage"].to_numpy(dtype=float)
                effect_attention = [
                    dict(row) for row in htr_effect_train.get("attention", []) or []
                ]
                for row in effect_attention:
                    row.setdefault("model_family", "htr")
                    row.setdefault("target_source", "ensemble_mean_nuisance_with_htr")
                    row.setdefault("effect_objective", effect_objective)
                effect_evidence = {
                    "metrics": _htr_effect_metrics(train_predictions),
                    "attention": effect_attention,
                    "effect_objective": effect_objective,
                }
                htr_effect_variants[effect_objective] = effect_evidence
                if effect_objective == "pseudo_outcome_mse":
                    htr_evidence["effect"] = effect_evidence
                _append_feature(
                    x_train_cols,
                    x_test_cols,
                    x_names,
                    feature_rows,
                    train=train_tau,
                    test=test_tau,
                    name=f"htr__{feature_suffix}",
                    role="X",
                    source_family="htr",
                    outer_fold=outer_fold,
                    objective=(
                        "r_pseudo_outcome"
                        if effect_objective == "pseudo_outcome_mse"
                        else "direct_weighted_r"
                    ),
                    provenance="inner_oof_train_outer_train_fit_test",
                )
                prediction_frames.append(
                    _source_prediction_frame(
                        train_df,
                        test_df,
                        outer_fold=outer_fold,
                        source_name=f"htr__{feature_suffix}",
                        values={"tau_hat": (train_tau, test_tau)},
                    )
                )
            if "effect" not in htr_evidence and htr_effect_variants:
                htr_evidence["effect"] = next(iter(htr_effect_variants.values()))
            if htr_effect_variants:
                htr_evidence["effect_variants"] = htr_effect_variants

        importance: Dict[str, Any] = _multi_view_importance(
            bow_view_results,
            top_n=int(self.nn_config.top_n_features),
        )
        importance["feature_discovery_methods"] = self._enabled_feature_discovery_methods()
        if ensemble_view_results:
            ensemble_importance = _multi_view_importance(
                ensemble_view_results,
                top_n=int(self.nn_config.top_n_features),
            )
            nuisance_source_names = [item[0] for item in nuisance_train]
            ensemble_importance["target_source"] = (
                "ensemble_mean_nuisance_with_htr"
                if any(str(name).startswith("htr") for name in nuisance_source_names)
                else "ensemble_mean_nuisance"
            )
            ensemble_importance["nuisance_sources"] = nuisance_source_names
            ensemble_importance["pseudo_target_construction"] = (
                "mean nuisance predictions across Stage 1 text models, then "
                "(Y - mean_m_hat) / (T - mean_e_hat)"
            )
            importance["ensemble_r"] = ensemble_importance

        embedding_evidence: Dict[str, Any] = {}
        if self._embedding_contrast_enabled():
            emb = self._embedding_feature_bundle(
                train_df=train_df,
                test_df=test_df,
                y=y,
                t=t,
                pseudo_target=pseudo_target,
                t_resid=t_resid,
                outer_fold=outer_fold,
            )
            for item in emb["w_features"]:
                _append_feature(
                    w_train_cols,
                    w_test_cols,
                    w_names,
                    feature_rows,
                    train=item["train"],
                    test=item["test"],
                    name=item["name"],
                    role="W",
                    source_family="embedding_contrast",
                    outer_fold=outer_fold,
                    objective=item["objective"],
                    provenance="outer_train_contrast_vector",
                    contrast_family=item.get("contrast_family"),
                )
            for item in emb["x_features"]:
                _append_feature(
                    x_train_cols,
                    x_test_cols,
                    x_names,
                    feature_rows,
                    train=item["train"],
                    test=item["test"],
                    name=item["name"],
                    role="X",
                    source_family="embedding_contrast",
                    outer_fold=outer_fold,
                    objective=item["objective"],
                    provenance="outer_train_contrast_vector",
                    contrast_family=item.get("contrast_family"),
                )
            embedding_rows.extend(emb["metadata"])
            embedding_evidence = self._build_primary_embedding_contrast_evidence(
                discovery_df=train_df,
                y=y,
                t=t,
                pseudo_target=pseudo_target,
                t_resid=t_resid,
                importance=importance,
            )

        handoff_evidence = {
            "metrics": copy.deepcopy(metrics),
            "importance": importance,
            "embedding_contrast_evidence": embedding_evidence,
            "htr_evidence": htr_evidence,
        }

        return _FeatureBundle(
            x_train=_column_matrix(x_train_cols, len(train_df)),
            x_test=_column_matrix(x_test_cols, len(test_df)),
            w_train=_column_matrix(w_train_cols, len(train_df)),
            w_test=_column_matrix(w_test_cols, len(test_df)),
            x_names=x_names,
            w_names=w_names,
            feature_rows=feature_rows,
            prediction_frames=prediction_frames,
            embedding_rows=embedding_rows,
            metrics=metrics,
            handoff_evidence=handoff_evidence,
        )

    def _primary_bow_metrics(
        self,
        *,
        discovery_df: pd.DataFrame,
        y: np.ndarray,
        t: np.ndarray,
        e_hat: np.ndarray,
        m_hat: np.ndarray,
        pseudo_target: np.ndarray,
        tau_hat: np.ndarray,
        y_resid: np.ndarray,
        t_resid: np.ndarray,
    ) -> Dict[str, Any]:
        r_loss = (np.asarray(y_resid, dtype=float) - np.asarray(tau_hat, dtype=float) * t_resid) ** 2
        r_loss_at_zero = np.asarray(y_resid, dtype=float) ** 2
        metrics: Dict[str, Any] = {
            "treatment_auroc": _safe_roc_auc(t, e_hat),
            "pseudo_target_mean": _finite_or_none(np.mean(pseudo_target)),
            "pseudo_target_std": _finite_or_none(np.std(pseudo_target)),
            "tau_hat_mean": _finite_or_none(np.mean(tau_hat)),
            "tau_hat_std": _finite_or_none(np.std(tau_hat)),
            "r_loss_mean": _finite_or_none(np.mean(r_loss)),
            "r_loss_at_zero_mean": _finite_or_none(np.mean(r_loss_at_zero)),
            "r_loss_improvement": _finite_or_none(np.mean(r_loss_at_zero) - np.mean(r_loss)),
            "pseudo_target_construction": (
                "Stage 1 ensemble nuisance predictions, then "
                "(Y - mean_m_hat) / (T - mean_e_hat)"
            ),
        }
        try:
            metrics["treatment_brier"] = _finite_or_none(brier_score_loss(t, e_hat))
        except Exception:
            pass
        try:
            metrics["treatment_log_loss"] = _finite_or_none(log_loss(t, e_hat))
        except Exception:
            pass
        if str(self.config.outcome_type).lower() == "continuous":
            metrics["outcome_rmse"] = _finite_or_none(np.sqrt(mean_squared_error(y, m_hat)))
        else:
            metrics["outcome_auroc"] = _safe_roc_auc(y, m_hat)
            try:
                metrics["outcome_brier"] = _finite_or_none(brier_score_loss(y, m_hat))
            except Exception:
                pass
        if "true_ite_prob" in discovery_df.columns:
            metrics["tau_hat_true_ite_corr"] = _safe_corr(
                tau_hat,
                discovery_df["true_ite_prob"].to_numpy(dtype=float),
            )
            metrics["pseudo_target_true_ite_corr"] = _safe_corr(
                pseudo_target,
                discovery_df["true_ite_prob"].to_numpy(dtype=float),
            )
        return metrics

    def _fit_primary_feature_importance_models(
        self,
        *,
        texts: Sequence[str],
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: np.ndarray,
        pseudo_target_sample_weight: Optional[np.ndarray],
        view: BoWViewConfig,
    ) -> Dict[str, Any]:
        vectorizer = _make_bow_vectorizer(_vectorizer_params(view))
        x_model = vectorizer.fit_transform(texts)
        features = np.asarray(vectorizer.get_feature_names_out())

        if len(np.unique(np.asarray(t, dtype=int))) < 2:
            treatment_coef = np.zeros(len(features), dtype=float)
        else:
            treatment_model = _make_bow_classifier(_model_params(view), random_state=101)
            treatment_model.fit(x_model, np.asarray(t, dtype=int))
            treatment_coef = _model_feature_scores(treatment_model, len(features))

        if str(self.config.outcome_type).lower() == "continuous":
            outcome_model = _make_bow_regressor(_model_params(view), random_state=202)
            outcome_model.fit(x_model, y)
            outcome_coef = _model_feature_scores(outcome_model, len(features))
        elif len(np.unique(np.asarray(y, dtype=int))) < 2:
            outcome_coef = np.zeros(len(features), dtype=float)
        else:
            outcome_model = _make_bow_classifier(_model_params(view), random_state=202)
            outcome_model.fit(x_model, np.asarray(y, dtype=int))
            outcome_coef = _model_feature_scores(outcome_model, len(features))

        effect_model = _make_bow_regressor(_model_params(view), random_state=303)
        _fit_regressor(
            effect_model,
            x_model,
            pseudo_target,
            sample_weight=pseudo_target_sample_weight,
        )
        effect_coef = _model_feature_scores(effect_model, len(features))

        top_n = int(self.nn_config.top_n_features)
        confounder_score = np.abs(treatment_coef) * np.abs(outcome_coef)
        return {
            "view_name": str(view.name),
            "view_config": _bow_view_to_dict(view),
            "n_features": int(len(features)),
            "n_bow_features": int(len(features)),
            "n_prespecified_features": 0,
            "n_prespecified_raw_features": 0,
            "prespecified_raw_feature_names": [],
            "phrase_features": _top_phrase_feature_rows(
                features,
                top_n=top_n,
                treatment_coef=treatment_coef,
                outcome_coef=outcome_coef,
                pseudo_target_coef=effect_coef,
                confounder_score=confounder_score,
            ),
            "confounder_overlap": _top_feature_rows(
                features,
                confounder_score,
                top_n,
                treatment_coef=treatment_coef,
                outcome_coef=outcome_coef,
            ),
            "treatment_positive": _top_feature_rows(features, treatment_coef, top_n),
            "treatment_negative": _top_feature_rows(
                features,
                treatment_coef,
                top_n,
                descending=False,
            ),
            "outcome_positive": _top_feature_rows(features, outcome_coef, top_n),
            "outcome_negative": _top_feature_rows(
                features,
                outcome_coef,
                top_n,
                descending=False,
            ),
            "pseudo_target_positive": _top_feature_rows(features, effect_coef, top_n),
            "pseudo_target_negative": _top_feature_rows(
                features,
                effect_coef,
                top_n,
                descending=False,
            ),
        }

    def _build_primary_embedding_contrast_evidence(
        self,
        *,
        discovery_df: pd.DataFrame,
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: np.ndarray,
        t_resid: np.ndarray,
        importance: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self._embedding_contrast_enabled():
            return {}
        generator = self._embedding_generator()
        generator.prepare(self.dataset)
        return generator.build_evidence(
            discovery_df=discovery_df,
            y=y,
            t=t,
            pseudo_target=[pseudo_target],
            t_resid=[t_resid],
            pseudo_target_names=["stage1_ensemble_mean_nuisance"],
            importance=importance,
        )

    def _build_primary_agent_context(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        metrics: Dict[str, Any],
        importance: Dict[str, Any],
        embedding_evidence: Optional[Dict[str, Any]] = None,
        htr_evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        instructions = [
            "You are generating candidate variables from empirical text evidence.",
            "The evidence was produced during Stage 1 primary text-model forest training.",
            "Suggest explicit pre-treatment patient-level variables, not raw text tokens.",
            "Use variables predictive of both treatment and outcome as confounders.",
            "Use variables predictive of the pseudo-target or R-stage signal as effect modifiers.",
            "Do not invent broad clinical inventory variables unsupported by the enabled evidence.",
            "Avoid near-duplicate aliases for the same extraction target.",
        ]
        if self._bow_enabled():
            instructions.append(
                "Review sparse bag-of-words feature importance across the Stage 1 views."
            )
        if self._embedding_contrast_enabled():
            instructions.append(
                "Use embedding_contrast_evidence as retrieved chunk evidence, not as a direct vector interpretation."
            )
        if self._htr_enabled():
            instructions.append(
                "Use htr_attention_evidence from the Stage 1 HTR nuisance and R-stage models as neural text evidence."
            )
        context: Dict[str, Any] = {
            "prompt_version": "multi_model_agentic_forest_v1",
            "outer_fold": int(outer_fold),
            "feature_discovery_methods": self._enabled_feature_discovery_methods(),
            "max_proposals": int(self.nn_config.candidate_proposals_per_fold),
            "clinical_question": self.config.clinical_question,
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "instructions": instructions,
            "current_features": [],
            "model_diagnostics": _agent_visible_metrics(metrics),
            "feature_importance": importance,
            "clinical_text_examples": _clinical_text_examples(
                discovery_df,
                self.config.text_column,
                n_examples=self.search_config.clinical_text_examples_per_prompt,
                max_chars=self.search_config.clinical_text_example_chars,
            ),
            "response_contract": {
                "proposals": [
                    {
                        "action": "add",
                        "name": "snake_case_variable_name",
                        "type": "categorical|continuous",
                        "categories": ["category_a", "category_b"],
                        "roles": ["confounder", "effect_modifier"],
                        "description": "exact pre-treatment extraction target",
                        "rationale": "which enabled evidence supports this variable",
                        "expected_signal": "treatment, outcome, or pseudo-target signal expected",
                    }
                ]
            },
            "handoff_provenance": {
                "source": "multi_model_forest_stage1_primary_text_models",
                "raw_text_modeling_reused_for_agentic_stage": True,
            },
        }
        if embedding_evidence:
            context["embedding_contrast_evidence"] = embedding_evidence
        if htr_evidence:
            context["htr_attention_evidence"] = htr_evidence
        compact_context = _compact_multi_model_agent_context(context)
        prompt_chars = len(json.dumps(compact_context, separators=(",", ":"), default=str))
        logger.info(
            "Multi-model forest primary handoff context outer_fold=%s: %.1fK JSON chars",
            outer_fold,
            prompt_chars / 1000.0,
        )
        return compact_context

    def _fit_bow_binary_train_test(
        self,
        texts_train: Sequence[str],
        texts_test: Sequence[str],
        labels: np.ndarray,
        *,
        outer_fold: int,
        view: BoWViewConfig,
        view_index: int,
        label_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.asarray(labels, dtype=int)
        oof = np.full(len(labels), np.nan, dtype=float)
        split_items = list(
            enumerate(
                _binary_split_items(
                    labels,
                    requested_folds=int(self.nn_config.nuisance_folds),
                    random_state=11_000
                    + 100 * int(outer_fold)
                    + 1_000 * int(view_index)
                    + (1 if label_name == "outcome" else 2),
                ),
                start=1,
            )
        )
        vectorizer_params = _vectorizer_params(view)
        model_params = _model_params(view)
        n_jobs = self._fold_n_jobs(len(split_items))
        logger.info(
            "Outer fold %s BoW binary %s view=%s model=%s folds=%s n_jobs=%s " "backend=%s",
            outer_fold,
            label_name,
            view.name,
            view.bow_model,
            len(split_items),
            n_jobs,
            self._parallel_backend_name(),
        )

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            return _fit_binary_bow_fold(
                texts_train,
                labels,
                fit_pos,
                heldout_pos,
                vectorizer_params,
                model_params,
                random_state=17 + int(fold),
            )

        for heldout_pos, values in self._run_fold_tasks(run_fold, split_items):
            oof[heldout_pos] = values
        if len(np.unique(labels)) < 2:
            test_pred = np.full(len(texts_test), float(np.mean(labels)), dtype=float)
        else:
            vectorizer = _make_bow_vectorizer(vectorizer_params)
            x_train = vectorizer.fit_transform(texts_train)
            x_test = vectorizer.transform(texts_test)
            model = _make_bow_classifier(model_params, random_state=117 + int(view_index))
            model.fit(x_train, labels)
            test_pred = model.predict_proba(x_test)[:, 1]
        return (
            np.clip(oof, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip),
            np.clip(test_pred, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip),
        )

    def _fit_bow_regression_train_test(
        self,
        texts_train: Sequence[str],
        texts_test: Sequence[str],
        values: np.ndarray,
        sample_weight: Optional[np.ndarray],
        *,
        outer_fold: int,
        view: BoWViewConfig,
        view_index: int,
        target_name: str,
        seed_offset: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float)
        oof = np.full(len(values), np.nan, dtype=float)
        folds = _bounded_fold_count(
            int(
                self.nn_config.effect_folds
                if "effect" in target_name
                else self.nn_config.nuisance_folds
            ),
            len(values),
        )
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=13_000
            + int(seed_offset)
            + 100 * int(outer_fold)
            + 1_000 * int(view_index),
        )
        split_items = list(enumerate(splitter.split(texts_train), start=1))
        vectorizer_params = _vectorizer_params(view)
        model_params = _model_params(view)
        n_jobs = self._fold_n_jobs(len(split_items))
        logger.info(
            "Outer fold %s BoW regression %s view=%s model=%s folds=%s n_jobs=%s " "backend=%s",
            outer_fold,
            target_name,
            view.name,
            view.bow_model,
            len(split_items),
            n_jobs,
            self._parallel_backend_name(),
        )

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            return _fit_regression_bow_fold(
                texts_train,
                values,
                fit_pos,
                heldout_pos,
                vectorizer_params,
                model_params,
                sample_weight=sample_weight,
                random_state=17 + int(seed_offset) + int(fold),
            )

        for heldout_pos, pred in self._run_fold_tasks(run_fold, split_items):
            oof[heldout_pos] = pred
        vectorizer = _make_bow_vectorizer(vectorizer_params)
        x_train = vectorizer.fit_transform(texts_train)
        x_test = vectorizer.transform(texts_test)
        model = _make_bow_regressor(
            model_params, random_state=217 + int(seed_offset) + int(view_index)
        )
        _fit_regressor(model, x_train, values, sample_weight=sample_weight)
        return oof, model.predict(x_test)

    def _embedding_feature_bundle(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: np.ndarray,
        t_resid: np.ndarray,
        outer_fold: int,
    ) -> Dict[str, Any]:
        generator = self._embedding_generator()
        generator.prepare(self.dataset)
        train_positions = generator._positions_for_frame(train_df)
        test_positions = generator._positions_for_frame(test_df)
        train_patient = generator._patient_embeddings(train_positions)
        train_patient = _residualize_embeddings(
            train_patient,
            train_df,
            self.nn_config.embedding_contrast.residualize_columns,
        )
        train_patient = _normalize_rows(train_patient)
        directions, metadata = self._embedding_directions(
            patient_embeddings=train_patient,
            y=y,
            t=t,
            pseudo_target=pseudo_target,
            t_resid=t_resid,
            outer_fold=outer_fold,
        )
        w_features = []
        x_features = []
        for direction in directions:
            train_mean, train_max = self._chunk_similarity_features(
                generator,
                train_positions,
                direction["direction"],
            )
            test_mean, test_max = self._chunk_similarity_features(
                generator,
                test_positions,
                direction["direction"],
            )
            target = w_features if direction["role"] == "W" else x_features
            base_name = f"embedding__{direction['name']}"
            for stat, train_values, test_values in [
                ("mean_cosine", train_mean, test_mean),
                ("max_cosine", train_max, test_max),
            ]:
                target.append(
                    {
                        "name": f"{base_name}__{stat}",
                        "train": train_values,
                        "test": test_values,
                        "objective": direction["objective"],
                        "contrast_family": direction["contrast_family"],
                    }
                )
        return {
            "w_features": w_features,
            "x_features": x_features,
            "metadata": metadata,
        }

    def _embedding_directions(
        self,
        *,
        patient_embeddings: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: np.ndarray,
        t_resid: np.ndarray,
        outer_fold: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        directions: List[Dict[str, Any]] = []
        metadata: List[Dict[str, Any]] = []
        finite = np.all(np.isfinite(patient_embeddings), axis=1)
        treatment_labels, treatment_mask = _binary_labels(t)
        t_direction, t_counts = _binary_mean_difference_direction(
            patient_embeddings,
            treatment_labels,
            treatment_mask & finite,
        )
        outcome_labels, outcome_mask = (
            _tail_labels(y, float(self.nn_config.embedding_contrast.pseudo_target_quantile))
            if str(self.config.outcome_type).lower() == "continuous"
            else _binary_labels(y)
        )
        y_direction, y_counts = _binary_mean_difference_direction(
            patient_embeddings,
            outcome_labels,
            outcome_mask & finite,
        )
        if t_direction is not None:
            self._add_embedding_direction(
                directions,
                metadata,
                outer_fold,
                name="global_treatment_contrast",
                direction=t_direction,
                role="W",
                objective="treatment_confounder",
                contrast_family="global_marginal_treatment",
                counts=t_counts,
            )
        if y_direction is not None:
            self._add_embedding_direction(
                directions,
                metadata,
                outer_fold,
                name="global_outcome_contrast",
                direction=y_direction,
                role="W",
                objective="outcome_confounder",
                contrast_family="global_marginal_outcome",
                counts=y_counts,
            )
        if t_direction is not None and y_direction is not None:
            confounder = 0.5 * _normalize_vector(t_direction) + 0.5 * _normalize_vector(y_direction)
            self._add_embedding_direction(
                directions,
                metadata,
                outer_fold,
                name="global_confounder_average",
                direction=confounder,
                role="W",
                objective="treatment_outcome_confounder_average",
                contrast_family="global_marginal_confounder_average",
                counts={"treatment": t_counts, "outcome": y_counts},
            )

        pseudo_labels, pseudo_mask = _tail_labels(
            pseudo_target,
            float(self.nn_config.embedding_contrast.pseudo_target_quantile),
        )
        pseudo_weights = np.square(np.asarray(t_resid, dtype=float))
        pseudo_direction, pseudo_counts = _weighted_binary_direction(
            patient_embeddings,
            pseudo_labels,
            pseudo_mask & finite,
            (
                pseudo_weights
                if bool(self.nn_config.embedding_contrast.pseudo_target_weighted)
                else None
            ),
        )
        if pseudo_direction is not None:
            self._add_embedding_direction(
                directions,
                metadata,
                outer_fold,
                name="global_r_pseudo_target_contrast",
                direction=pseudo_direction,
                role="X",
                objective="r_pseudo_outcome",
                contrast_family="global_r_pseudo_target",
                counts=pseudo_counts,
            )
        orthogonal_score = np.asarray(pseudo_target, dtype=float) * np.square(
            np.asarray(t_resid, dtype=float)
        )
        score_labels, score_mask = _tail_labels(
            orthogonal_score,
            float(self.nn_config.embedding_contrast.pseudo_target_quantile),
        )
        score_direction, score_counts = _binary_mean_difference_direction(
            patient_embeddings,
            score_labels,
            score_mask & finite,
        )
        if score_direction is not None:
            self._add_embedding_direction(
                directions,
                metadata,
                outer_fold,
                name="global_orthogonal_r_score_contrast",
                direction=score_direction,
                role="X",
                objective="orthogonal_r_score",
                contrast_family="global_orthogonal_r_score",
                counts=score_counts,
            )

        residual_interaction = self._residualized_interaction_direction(
            patient_embeddings,
            y,
            t,
            treatment_labels,
            treatment_mask,
            outcome_labels,
            outcome_mask,
            t_direction,
            y_direction,
            finite,
        )
        if residual_interaction is not None:
            self._add_embedding_direction(
                directions,
                metadata,
                outer_fold,
                name="global_residualized_treatment_outcome_interaction",
                direction=residual_interaction,
                role="X",
                objective="residualized_treatment_outcome_interaction",
                contrast_family="global_residualized_interaction",
                counts={},
            )

        if bool(self.nn_config.embedding_contrast.include_cluster_contrast_vectors):
            directions.extend(
                self._cluster_embedding_directions(
                    patient_embeddings=patient_embeddings,
                    y=y,
                    t=t,
                    outcome_labels=outcome_labels,
                    outcome_mask=outcome_mask,
                    treatment_labels=treatment_labels,
                    treatment_mask=treatment_mask,
                    finite=finite,
                    metadata=metadata,
                    outer_fold=outer_fold,
                )
            )
        return directions, metadata

    def _add_embedding_direction(
        self,
        directions: List[Dict[str, Any]],
        metadata: List[Dict[str, Any]],
        outer_fold: int,
        *,
        name: str,
        direction: np.ndarray,
        role: str,
        objective: str,
        contrast_family: str,
        counts: Any,
    ) -> None:
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0.0:
            return
        direction = _normalize_vector(direction)
        directions.append(
            {
                "name": name,
                "direction": direction,
                "role": role,
                "objective": objective,
                "contrast_family": contrast_family,
            }
        )
        metadata.append(
            {
                "outer_fold": int(outer_fold),
                "name": name,
                "role": role,
                "objective": objective,
                "contrast_family": contrast_family,
                "direction_norm": float(np.linalg.norm(direction)),
                "counts": counts,
            }
        )

    def _cluster_embedding_directions(
        self,
        *,
        patient_embeddings: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        outcome_labels: np.ndarray,
        outcome_mask: np.ndarray,
        treatment_labels: np.ndarray,
        treatment_mask: np.ndarray,
        finite: np.ndarray,
        metadata: List[Dict[str, Any]],
        outer_fold: int,
    ) -> List[Dict[str, Any]]:
        cfg = self.nn_config.embedding_contrast
        n_usable = int(np.sum(finite))
        n_clusters = min(
            int(cfg.cluster_contrast_n_clusters),
            n_usable // int(cfg.cluster_contrast_min_cluster_size),
        )
        if n_clusters < 2:
            metadata.append(
                {
                    "outer_fold": int(outer_fold),
                    "name": "cluster_contrast_vectors",
                    "skipped": "too_few_patients_for_cluster_contrasts",
                    "n_usable_patients": n_usable,
                }
            )
            return []
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=int(cfg.cluster_contrast_random_state),
            batch_size=max(128, min(1024, n_usable)),
            n_init=int(cfg.cluster_contrast_kmeans_n_init),
            max_iter=300,
        )
        labels = np.full(len(patient_embeddings), -1, dtype=int)
        labels[finite] = kmeans.fit_predict(patient_embeddings[finite])
        counts = np.bincount(labels[finite], minlength=n_clusters)
        treatment_items = []
        interaction_items = []
        for cluster_id in range(n_clusters):
            if int(counts[cluster_id]) < int(cfg.cluster_contrast_min_cluster_size):
                continue
            cluster_mask = labels == cluster_id
            local_mask = cluster_mask & treatment_mask & finite
            pos = local_mask & (treatment_labels == 1)
            neg = local_mask & (treatment_labels == 0)
            if int(np.sum(pos)) >= int(cfg.cluster_contrast_min_group_size) and int(
                np.sum(neg)
            ) >= int(cfg.cluster_contrast_min_group_size):
                direction = np.mean(patient_embeddings[pos], axis=0) - np.mean(
                    patient_embeddings[neg],
                    axis=0,
                )
                if float(np.linalg.norm(direction)) > 0:
                    treatment_items.append(
                        {
                            "cluster_id": int(cluster_id),
                            "n_cluster": int(counts[cluster_id]),
                            "direction": direction,
                        }
                    )
            interaction = self._cluster_local_interaction_direction(
                patient_embeddings,
                treatment_labels,
                treatment_mask,
                outcome_labels,
                outcome_mask,
                cluster_mask & finite,
            )
            if interaction is not None:
                interaction_items.append(
                    {
                        "cluster_id": int(cluster_id),
                        "n_cluster": int(counts[cluster_id]),
                        "direction": interaction,
                    }
                )
        result: List[Dict[str, Any]] = []
        result.extend(
            self._svd_cluster_components(
                items=treatment_items,
                role="W",
                objective="cluster_treatment_confounder",
                contrast_family="cluster_local_treatment_contrast_basis",
                prefix="cluster_confounder_treatment",
                metadata=metadata,
                outer_fold=outer_fold,
            )
        )
        result.extend(
            self._svd_cluster_components(
                items=interaction_items,
                role="X",
                objective="cluster_residualized_treatment_outcome_interaction",
                contrast_family="cluster_local_residualized_interaction_contrast_basis",
                prefix="cluster_effect_residualized_interaction",
                metadata=metadata,
                outer_fold=outer_fold,
            )
        )
        return result

    def _cluster_local_interaction_direction(
        self,
        patient_embeddings: np.ndarray,
        treatment_labels: np.ndarray,
        treatment_mask: np.ndarray,
        outcome_labels: np.ndarray,
        outcome_mask: np.ndarray,
        cluster_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        base = cluster_mask & treatment_mask & outcome_mask
        treated_positive = base & (treatment_labels == 1) & (outcome_labels == 1)
        treated_negative = base & (treatment_labels == 1) & (outcome_labels == 0)
        untreated_positive = base & (treatment_labels == 0) & (outcome_labels == 1)
        untreated_negative = base & (treatment_labels == 0) & (outcome_labels == 0)
        min_cell = int(self.nn_config.embedding_contrast.cluster_contrast_min_cell_size)
        if (
            min(
                int(np.sum(treated_positive)),
                int(np.sum(treated_negative)),
                int(np.sum(untreated_positive)),
                int(np.sum(untreated_negative)),
            )
            < min_cell
        ):
            return None
        raw = (
            np.mean(patient_embeddings[treated_positive], axis=0)
            - np.mean(patient_embeddings[treated_negative], axis=0)
            - np.mean(patient_embeddings[untreated_positive], axis=0)
            + np.mean(patient_embeddings[untreated_negative], axis=0)
        )
        t_dir, _ = _binary_mean_difference_direction(
            patient_embeddings,
            treatment_labels,
            cluster_mask & treatment_mask,
        )
        y_dir, _ = _binary_mean_difference_direction(
            patient_embeddings,
            outcome_labels,
            cluster_mask & outcome_mask,
        )
        if t_dir is None or y_dir is None:
            return None
        residual = _residualize_vector_from_basis(raw, [t_dir, y_dir])
        if float(np.linalg.norm(residual)) <= 0.0:
            return None
        return residual

    def _svd_cluster_components(
        self,
        *,
        items: Sequence[Dict[str, Any]],
        role: str,
        objective: str,
        contrast_family: str,
        prefix: str,
        metadata: List[Dict[str, Any]],
        outer_fold: int,
    ) -> List[Dict[str, Any]]:
        if len(items) < 2:
            return []
        matrix = np.vstack(
            [
                _normalize_vector(np.asarray(item["direction"], dtype=float))
                * np.sqrt(float(item["n_cluster"]))
                for item in items
            ]
        )
        try:
            _left, singular_values, components = np.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            return []
        result = []
        max_components = min(
            int(self.nn_config.embedding_contrast.cluster_contrast_max_components),
            len(singular_values),
        )
        total_energy = float(np.sum(np.square(singular_values)))
        for idx in range(max_components):
            sv = float(singular_values[idx])
            if not np.isfinite(sv) or sv <= 0.0:
                continue
            direction = _normalize_vector(components[idx])
            name = f"{prefix}_pc{idx + 1}"
            result.append(
                {
                    "name": name,
                    "direction": direction,
                    "role": role,
                    "objective": objective,
                    "contrast_family": contrast_family,
                }
            )
            metadata.append(
                {
                    "outer_fold": int(outer_fold),
                    "name": name,
                    "role": role,
                    "objective": objective,
                    "contrast_family": contrast_family,
                    "cluster_component_index": int(idx + 1),
                    "singular_value": sv,
                    "explained_energy": (
                        float(sv**2 / total_energy) if total_energy > 0.0 else None
                    ),
                    "local_contrast_count": int(len(items)),
                }
            )
        return result

    def _residualized_interaction_direction(
        self,
        patient_embeddings: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        treatment_labels: np.ndarray,
        treatment_mask: np.ndarray,
        outcome_labels: np.ndarray,
        outcome_mask: np.ndarray,
        treatment_direction: Optional[np.ndarray],
        outcome_direction: Optional[np.ndarray],
        finite: np.ndarray,
    ) -> Optional[np.ndarray]:
        del y, t
        if treatment_direction is None or outcome_direction is None:
            return None
        base = finite & treatment_mask & outcome_mask
        treated_positive = base & (treatment_labels == 1) & (outcome_labels == 1)
        treated_negative = base & (treatment_labels == 1) & (outcome_labels == 0)
        untreated_positive = base & (treatment_labels == 0) & (outcome_labels == 1)
        untreated_negative = base & (treatment_labels == 0) & (outcome_labels == 0)
        if (
            min(
                int(np.sum(treated_positive)),
                int(np.sum(treated_negative)),
                int(np.sum(untreated_positive)),
                int(np.sum(untreated_negative)),
            )
            < 2
        ):
            return None
        raw = (
            np.mean(patient_embeddings[treated_positive], axis=0)
            - np.mean(patient_embeddings[treated_negative], axis=0)
            - np.mean(patient_embeddings[untreated_positive], axis=0)
            + np.mean(patient_embeddings[untreated_negative], axis=0)
        )
        residual = _residualize_vector_from_basis(raw, [treatment_direction, outcome_direction])
        if float(np.linalg.norm(residual)) <= 0.0:
            return None
        return residual

    def _chunk_similarity_features(
        self,
        generator: EmbeddingContrastEvidenceGenerator,
        positions: Sequence[int],
        direction: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        means = []
        maxes = []
        direction = _normalize_vector(np.asarray(direction, dtype=np.float32))
        for position in positions:
            chunks = generator._chunk_matrix(int(position))
            if chunks.size == 0:
                means.append(0.0)
                maxes.append(0.0)
                continue
            scores = np.asarray(chunks @ direction, dtype=float)
            finite = scores[np.isfinite(scores)]
            if len(finite) == 0:
                means.append(0.0)
                maxes.append(0.0)
            else:
                means.append(float(np.mean(finite)))
                maxes.append(float(np.max(finite)))
        return np.asarray(means, dtype=np.float32), np.asarray(maxes, dtype=np.float32)

    def _enabled_feature_discovery_methods(self) -> List[str]:
        methods = []
        if self._bow_enabled():
            methods.append("bow")
        if self._htr_enabled():
            methods.append("htr")
        if self._embedding_contrast_enabled():
            methods.append("embedding_contrast")
        return methods

    def _bow_enabled(self) -> bool:
        return bool(getattr(self.nn_config, "bow_discovery_enabled", True))

    def _htr_enabled(self) -> bool:
        return bool(getattr(self.nn_config, "htr_evidence_enabled", True))

    def _embedding_contrast_enabled(self) -> bool:
        return bool(getattr(self.nn_config.embedding_contrast, "enabled", False))

    def _embedding_generator(self) -> EmbeddingContrastEvidenceGenerator:
        if self.embedding_evidence_generator is None:
            self.embedding_evidence_generator = EmbeddingContrastEvidenceGenerator(
                config=self.config,
                output_dir=self.artifact_dir,
                embedding_provider=self.embedding_provider,
            )
        return self.embedding_evidence_generator

    def _htr_provider(self) -> Any:
        if self.htr_evidence_provider is not None:
            return self.htr_evidence_provider
        if self._default_htr_provider is None:
            htr_num_workers = 0 if self._outer_backend_name() == "processes" else self.num_workers
            self._default_htr_provider = MultiModelOptionalHTRProvider(
                config=self.config,
                output_dir=self.artifact_dir,
                device=self.device,
                gpu_ids=self.gpu_ids,
                num_workers=htr_num_workers,
            )
        return self._default_htr_provider

    def _sync_htr_fold_parallelism(self) -> None:
        htr_setting = self._htr_fold_parallelism_setting()
        avf_config = getattr(self.config.architecture, "agentic_attention_variable_forest", None)
        if avf_config is None:
            avf_config = AgenticAttentionVariableForestConfig()
            self.config.architecture.agentic_attention_variable_forest = avf_config
        avf_config.fold_parallelism = str(htr_setting)

    def _outer_backend_name(self) -> str:
        backend = str(getattr(self.nn_config, "outer_parallel_backend", "threads")).strip().lower()
        if backend == "loky":
            backend = "processes"
        if backend not in {"threads", "processes"}:
            raise ValueError(
                "multi_model_forest_agent_optional.outer_parallel_backend must be "
                "'threads', 'processes', or 'loky'"
            )
        return backend

    def _bow_fold_parallelism_setting(self) -> str:
        setting = getattr(self.nn_config, "bow_fold_parallelism", None)
        if setting is None:
            setting = self.nn_config.fold_parallelism
        return str(setting).strip().lower()

    def _htr_fold_parallelism_setting(self) -> str:
        setting = getattr(self.nn_config, "htr_fold_parallelism", None)
        if setting is None:
            setting = self.nn_config.fold_parallelism
        return str(setting).strip().lower()

    def _parallel_n_jobs(self, setting: Any, tasks: int, *, auto_workers: int) -> int:
        if tasks <= 0:
            return 1
        setting_text = str(setting).strip().lower()
        if setting_text == "auto":
            return max(1, min(int(auto_workers), int(tasks)))
        return max(1, min(int(setting_text), int(tasks)))

    def _outer_n_jobs(self, folds: int) -> int:
        return self._parallel_n_jobs(
            self.nn_config.outer_parallelism,
            folds,
            auto_workers=self.num_workers,
        )

    def _inner_workers_for_outer_job(self, outer_n_jobs: int) -> int:
        if str(self.nn_config.fold_parallelism).strip().lower() != "auto":
            return self.num_workers
        return max(1, int(self.num_workers) // max(1, int(outer_n_jobs)))

    def _outer_devices(self, outer_n_jobs: int) -> List[torch.device]:
        if self.gpu_ids and self.device.type == "cuda":
            devices = [torch.device(f"cuda:{int(gpu_id)}") for gpu_id in self.gpu_ids]
            return devices[: max(1, min(len(devices), int(outer_n_jobs)))]
        return [self.device]

    def _fold_n_jobs(self, folds: int) -> int:
        return self._parallel_n_jobs(
            self._bow_fold_parallelism_setting(),
            folds,
            auto_workers=self.num_workers,
        )

    def _parallel_backend_name(self) -> str:
        return "loky" if self.nn_config.bow_parallel_backend == "processes" else "threading"

    def _run_fold_tasks(self, run_fold: Any, split_items: Sequence[Any]) -> List[Any]:
        n_jobs = self._fold_n_jobs(len(split_items))
        if n_jobs <= 1:
            return [
                run_fold(int(fold), np.asarray(fit_pos), np.asarray(heldout_pos))
                for fold, (fit_pos, heldout_pos) in split_items
            ]
        return Parallel(
            n_jobs=n_jobs,
            backend=self._parallel_backend_name(),
            batch_size=1,
            pre_dispatch="all",
        )(
            delayed(run_fold)(int(fold), np.asarray(fit_pos), np.asarray(heldout_pos))
            for fold, (fit_pos, heldout_pos) in split_items
        )

    def _save_outputs(self, results_df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(self.output_path, index=False)
        results_df.to_parquet(self.artifact_dir / "ite_estimates.parquet", index=False)
        metric_frame = pd.DataFrame(self.outer_metric_rows)
        metric_frame.to_csv(self.artifact_dir / "outer_cv_metrics.csv", index=False)
        _write_jsonl(self.artifact_dir / "feature_manifest.jsonl", self.feature_manifest_rows)
        _write_jsonl(
            self.artifact_dir / "embedding_contrast_feature_vectors.jsonl",
            self.embedding_feature_rows,
        )
        _write_jsonl(self.artifact_dir / "split_provenance.jsonl", self.split_provenance_rows)
        if self.agentic_handoff_rows:
            handoff_path = self._agentic_handoff_path()
            _write_jsonl(handoff_path, self.agentic_handoff_rows)
            _write_json(
                handoff_path.with_suffix(".manifest.json"),
                {
                    "schema_version": "multi_model_agentic_discovery_handoff_v1",
                    "path": str(handoff_path),
                    "n_rows": int(len(self.agentic_handoff_rows)),
                    "scopes": sorted(
                        {str(row.get("scope")) for row in self.agentic_handoff_rows}
                    ),
                    "source": "stage1_primary_text_model_forest",
                },
            )
        if self.source_prediction_frames:
            pd.concat(self.source_prediction_frames, ignore_index=True).to_parquet(
                self.artifact_dir / "text_model_feature_predictions.parquet",
                index=False,
            )
        report = [
            "# Multi-Model Forest With Optional Agent Branch",
            "",
            f"- Rows: {len(self.dataset)}",
            f"- Outer folds: {len(self.outer_metric_rows)}",
            f"- Feature discovery methods: {', '.join(self._enabled_feature_discovery_methods())}",
            f"- Primary predictions: {self.output_path}",
            "- Agents used in primary forest: no",
        ]
        if bool(getattr(self.nn_config, "agentic_explicit_branch_enabled", False)):
            report.append("- Optional agentic explicit branch: enabled")
        else:
            report.append("- Optional agentic explicit branch: disabled")
        (self.artifact_dir / "report.txt").write_text("\n".join(report) + "\n")
        logger.info("Multi-model optional-agent forest predictions saved to: %s", self.output_path)

    def _agentic_handoff_path(self) -> Path:
        return self.artifact_dir / "agentic_handoff.jsonl"

    def _prepare_agentic_handoff(self) -> Path:
        handoff_path = self._agentic_handoff_path()
        logger.info(
            "Preparing multi-model agentic discovery handoff at: %s",
            handoff_path,
        )
        build_multi_model_agentic_discovery_handoff(
            self.dataset.drop(columns=["_oci_row_id"], errors="ignore"),
            self.config,
            handoff_path,
            device=self.device,
            gpu_ids=self.gpu_ids,
            num_workers=self.num_workers,
            include_candidate_consistency=True,
        )
        return handoff_path

    def _run_optional_agentic_branch(self) -> None:
        logger.info("Running optional final agentic explicit-feature branch")

        handoff_path = self._agentic_handoff_path()
        if not handoff_path.exists():
            raise RuntimeError(
                "Cannot run optional final agentic explicit-feature branch because "
                f"the discovery handoff is missing: {handoff_path}. Rerun the "
                "primary optional forest with --prepare-agentic-handoff."
            )
        branch_dir = self.artifact_dir / "agentic_explicit_branch"
        branch_dir.mkdir(parents=True, exist_ok=True)
        branch_prediction_path = branch_dir / "agentic_explicit_feature_predictions.parquet"
        run_multi_model_agentic_forest_from_handoff(
            self.dataset.drop(columns=["_oci_row_id"], errors="ignore"),
            self.config,
            branch_prediction_path,
            handoff_path,
            device=self.device,
            gpu_ids=self.gpu_ids,
            num_workers=self.num_workers,
        )
        summary = {
            "prediction_path": str(branch_prediction_path),
            "agentic_handoff_path": str(handoff_path),
        }
        artifact_metrics = (
            branch_prediction_path.parent / "multi_model_agentic_forest" / "outer_cv_metrics.csv"
        )
        if artifact_metrics.exists():
            summary["outer_cv_metrics_path"] = str(artifact_metrics)
        _write_json(self.artifact_dir / "agentic_explicit_branch_summary.json", summary)


def _vectorizer_params(view: BoWViewConfig) -> Dict[str, Any]:
    return {
        "ngram_range_min": int(view.ngram_range_min),
        "ngram_range_max": int(view.ngram_range_max),
        "min_df": int(view.min_df),
        "max_df": float(view.max_df),
        "sublinear_tf": bool(view.sublinear_tf),
        "max_features": int(view.max_features),
    }


def _model_params(view: BoWViewConfig) -> Dict[str, Any]:
    return {
        "bow_model": str(view.bow_model).strip().lower(),
        "logistic_c": float(view.logistic_c),
        "logistic_max_iter": int(view.logistic_max_iter),
        "ridge_alpha": float(view.ridge_alpha),
    }


def _bow_view_to_dict(view: BoWViewConfig) -> Dict[str, Any]:
    return {
        "name": view.name,
        "max_features": int(view.max_features),
        "min_df": int(view.min_df),
        "max_df": float(view.max_df),
        "ngram_range_min": int(view.ngram_range_min),
        "ngram_range_max": int(view.ngram_range_max),
        "sublinear_tf": bool(view.sublinear_tf),
        "bow_model": str(view.bow_model),
        "logistic_c": float(view.logistic_c),
        "ridge_alpha": float(view.ridge_alpha),
    }


def _append_feature(
    train_cols: List[np.ndarray],
    test_cols: List[np.ndarray],
    names: List[str],
    feature_rows: List[Dict[str, Any]],
    *,
    train: np.ndarray,
    test: np.ndarray,
    name: str,
    role: str,
    source_family: str,
    outer_fold: int,
    objective: str,
    provenance: str,
    **metadata: Any,
) -> None:
    train_cols.append(np.asarray(train, dtype=np.float32))
    test_cols.append(np.asarray(test, dtype=np.float32))
    names.append(str(name))
    row = {
        "outer_fold": int(outer_fold),
        "feature_name": str(name),
        "feature_role": str(role),
        "source_family": str(source_family),
        "objective": str(objective),
        "provenance": str(provenance),
    }
    row.update({key: value for key, value in metadata.items() if value is not None})
    feature_rows.append(row)


def _column_matrix(cols: Sequence[np.ndarray], n_rows: int) -> np.ndarray:
    if not cols:
        return np.zeros((n_rows, 0), dtype=np.float32)
    return np.column_stack([np.asarray(col, dtype=np.float32).reshape(n_rows) for col in cols])


def _clean_train_test_matrices(
    train: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    if train.shape[1] == 0:
        return train, test
    means = np.nanmean(np.where(np.isfinite(train), train, np.nan), axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    train = np.where(np.isfinite(train), train, means)
    test = np.where(np.isfinite(test), test, means)
    return train.astype(np.float32, copy=False), test.astype(np.float32, copy=False)


def _source_prediction_frame(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    outer_fold: int,
    source_name: str,
    values: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for split_role, frame, index in [
        ("train_inner_oof", train_df, 0),
        ("test_outer_train_fit", test_df, 1),
    ]:
        payload: Dict[str, Any] = {
            "_oci_row_id": frame["_oci_row_id"].to_numpy(),
            "outer_fold": int(outer_fold),
            "split_role": split_role,
            "source_name": str(source_name),
        }
        for column, pair in values.items():
            payload[column] = np.asarray(pair[index], dtype=float)
        rows.append(pd.DataFrame(payload))
    return pd.concat(rows, ignore_index=True)


def _weighted_binary_direction(
    embeddings: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    weights: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Dict[int, int]]:
    labels = np.asarray(labels, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    pos = mask & (labels == 1)
    neg = mask & (labels == 0)
    counts = {1: int(np.sum(pos)), 0: int(np.sum(neg))}
    if counts[1] < 2 or counts[0] < 2:
        return None, counts
    if weights is None:
        return np.mean(embeddings[pos], axis=0) - np.mean(embeddings[neg], axis=0), counts
    weights = np.asarray(weights, dtype=float)
    pos_w = np.maximum(weights[pos], 0.0)
    neg_w = np.maximum(weights[neg], 0.0)
    pos_mean = (
        np.average(embeddings[pos], axis=0, weights=pos_w)
        if float(np.sum(pos_w)) > 0.0
        else np.mean(embeddings[pos], axis=0)
    )
    neg_mean = (
        np.average(embeddings[neg], axis=0, weights=neg_w)
        if float(np.sum(neg_w)) > 0.0
        else np.mean(embeddings[neg], axis=0)
    )
    return pos_mean - neg_mean, counts


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value
