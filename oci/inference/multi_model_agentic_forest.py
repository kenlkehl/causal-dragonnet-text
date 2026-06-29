"""Multi-model BoW-guided agentic variable discovery plus causal forest."""

from __future__ import annotations

import json
import logging
import unicodedata
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from ..config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    BoWViewConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    MultiModelAgenticForestConfig,
    load_explicit_feature_specs_json,
)
from ..models.explicit_feature_featurizer import get_raw_explicit_features
from .agentic_explicit_feature_forest import (
    AgenticFeatureProposal,
    CausalForestExplicitEvaluator,
    OpenAICompatibleFeatureSearchAgent,
    SplitEvaluation,
    VLLMExplicitFeatureExtractionProvider,
    _clinical_text_examples,
    _get_agent_response_trace,
    _json_default,
    _normalize_feature_name,
    _safe_corr,
    _safe_roc_auc,
    _spec_to_dict,
    apply_agentic_alias_resolution,
    apply_agentic_value_harmonization,
    apply_proposals,
    validate_agentic_proposals,
)
from .embedding_contrast_discovery import (
    EmbeddingContrastEvidenceGenerator,
    redact_embedding_contrast_evidence,
)
from .agentic_attention_variable_forest import (
    AgenticAttentionVariableForestRunner,
    _attention_evidence_snippet,
    _attention_row_has_usable_text,
    _compact_token_spans,
    _parse_top_token_spans,
)


logger = logging.getLogger(__name__)


_DASH_TRANSLATION = dict.fromkeys(
    map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"),
    "-",
)

_AGENT_PROMPT_CONSENSUS_TOP_N = 40
_AGENT_PROMPT_VIEW_TOP_N = 12
_AGENT_PROMPT_EMBEDDING_CHUNKS_PER_TAIL = 3
_AGENT_PROMPT_EMBEDDING_CHUNK_CHARS = 600
_AGENT_PROMPT_CONCEPT_TOP_N = 8
_AGENT_PROMPT_HTR_ROWS_PER_STAGE = 36
_AGENT_PROMPT_HTR_SNIPPET_CHARS = 500
_AGENT_PROMPT_HTR_SUMMARY_CHARS = 320


def run_multi_model_agentic_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device=None,
    gpu_ids: Optional[Sequence[int]] = None,
    num_workers: int = 1,
    proposal_agent: Optional[Any] = None,
    extraction_provider: Optional[Any] = None,
    evaluator: Optional[Any] = None,
    embedding_provider: Optional[Any] = None,
    htr_evidence_provider: Optional[Any] = None,
) -> None:
    """Run BoW-guided agentic variable discovery and final explicit-feature forest."""
    runner = MultiModelAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        device=device,
        gpu_ids=gpu_ids,
        num_workers=num_workers,
        proposal_agent=proposal_agent,
        extraction_provider=extraction_provider,
        evaluator=evaluator,
        embedding_provider=embedding_provider,
        htr_evidence_provider=htr_evidence_provider,
    )
    runner.run()


class MultiModelHTREvidenceProvider:
    """Adapter that reuses the attention runner's HTR cross-fit stages."""

    def __init__(
        self,
        *,
        config: AppliedInferenceConfig,
        output_dir: Path,
        device: Optional[Any] = None,
        gpu_ids: Optional[Sequence[int]] = None,
        num_workers: int = 1,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = torch.device(device or "cpu")
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        self.num_workers = 1 if num_workers is None else int(num_workers)
        self._runner: Optional[AgenticAttentionVariableForestRunner] = None

    def _ensure_runner(self, discovery_df: pd.DataFrame) -> AgenticAttentionVariableForestRunner:
        if self._runner is None:
            self._runner = AgenticAttentionVariableForestRunner(
                dataset=discovery_df,
                config=self.config,
                output_path=self.output_dir / "htr_evidence" / "predictions.parquet",
                device=self.device,
                gpu_ids=self.gpu_ids,
                num_workers=self.num_workers,
            )
        return self._runner

    def fit_nuisance(
        self,
        discovery_df: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        runner = self._ensure_runner(discovery_df)
        return runner._crossfit_nuisance(discovery_df, outer_fold)

    def fit_effect(
        self,
        discovery_df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        runner = self._ensure_runner(discovery_df)
        return runner._crossfit_effect(discovery_df, nuisance_predictions, outer_fold)


class MultiModelAgenticForestRunner:
    """Sparse-text discovery path for explicit-variable causal forests."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        config: AppliedInferenceConfig,
        output_path: Path,
        device: Optional[Any] = None,
        gpu_ids: Optional[Sequence[int]] = None,
        num_workers: int = 1,
        proposal_agent: Optional[Any] = None,
        extraction_provider: Optional[Any] = None,
        evaluator: Optional[Any] = None,
        embedding_provider: Optional[Any] = None,
        htr_evidence_provider: Optional[Any] = None,
    ) -> None:
        self.dataset = dataset.reset_index(drop=True).copy()
        self.dataset["_oci_row_id"] = np.arange(len(self.dataset), dtype=int)
        self.config = config
        self.output_path = Path(output_path)
        self.artifact_dir = self.output_path.parent / "multi_model_agentic_forest"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or "cpu")
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        self.num_workers = 1 if num_workers is None else int(num_workers)
        self._has_external_components = (
            proposal_agent is not None
            or extraction_provider is not None
            or evaluator is not None
            or embedding_provider is not None
            or htr_evidence_provider is not None
        )

        self.nn_config: MultiModelAgenticForestConfig = getattr(
            config.architecture,
            "multi_model_agentic_forest",
            MultiModelAgenticForestConfig(),
        )
        self.search_config: AgenticFeatureSearchConfig = getattr(
            config.architecture,
            "agentic_feature_search",
            AgenticFeatureSearchConfig(),
        )
        self.cf_config: ExplicitFeatureForestConfig = getattr(
            config.architecture,
            "explicit_feature_forest",
            ExplicitFeatureForestConfig(),
        )
        self.proposal_agent = proposal_agent or OpenAICompatibleFeatureSearchAgent(
            self.search_config
        )
        self.extraction_provider = extraction_provider or VLLMExplicitFeatureExtractionProvider(
            config=config,
            output_dir=self.artifact_dir,
        )
        self.evaluator = evaluator or CausalForestExplicitEvaluator(
            config=config,
            cf_config=self.cf_config,
        )
        self.embedding_provider = embedding_provider
        self.htr_evidence_provider = htr_evidence_provider
        self.embedding_evidence_generator: Optional[
            EmbeddingContrastEvidenceGenerator
        ] = None
        self._default_htr_evidence_provider: Optional[MultiModelHTREvidenceProvider] = None

        self.bow_prediction_frames: List[pd.DataFrame] = []
        self.htr_nuisance_prediction_frames: List[pd.DataFrame] = []
        self.htr_effect_prediction_frames: List[pd.DataFrame] = []
        self.htr_attention_rows: List[Dict[str, Any]] = []
        self.importance_rows: List[Dict[str, Any]] = []
        self.embedding_evidence_rows: List[Dict[str, Any]] = []
        self.agent_rows: List[Dict[str, Any]] = []
        self.extracted_feature_diagnostic_rows: List[Dict[str, Any]] = []
        self.parsimony_review_rows: List[Dict[str, Any]] = []
        self.feature_set_rows: List[Dict[str, Any]] = []
        self.outer_metric_rows: List[Dict[str, Any]] = []
        self.alias_reference_specs: List[ExplicitFeatureSpec] = self._initial_specs()

    def run(self) -> None:
        logger.info("=" * 80)
        logger.info("MULTI-MODEL AGENTIC FEATURE CAUSAL FOREST")
        logger.info("=" * 80)
        self._validate_required_evidence_sources()
        self._ensure_prespecified_features()

        splits = self._analysis_splits()
        if self._embedding_contrast_enabled() and self.embedding_provider is None:
            self._embedding_contrast_generator().prepare(self.dataset)
        outer_n_jobs = self._outer_n_jobs(len(splits))
        if outer_n_jobs > 1 and self._htr_evidence_enabled():
            logger.warning(
                "Outer fold parallelism disabled because integrated HTR evidence "
                "loads neural models; use BoW fold parallelism or disable HTR with "
                "a documented reason for lightweight runs."
            )
            outer_n_jobs = 1
        if outer_n_jobs > 1 and self._has_external_components:
            logger.warning(
                "Outer fold parallelism disabled because custom agent/extractor/"
                "evaluator objects were supplied and may not be thread-safe."
            )
            outer_n_jobs = 1

        if outer_n_jobs > 1:
            backend = self._parallel_backend_name()
            logger.info(
                "Running %s multi-model outer fold(s) with outer_parallelism=%s "
                "backend=%s joblib_backend=%s",
                len(splits),
                outer_n_jobs,
                self.nn_config.bow_parallel_backend,
                backend,
            )
            fold_results = Parallel(
                n_jobs=outer_n_jobs,
                backend=backend,
                batch_size=1,
                pre_dispatch="all",
            )(
                delayed(_run_multi_model_outer_fold_worker)(
                    self.dataset,
                    self.config,
                    self.artifact_dir,
                    int(outer_fold),
                    np.asarray(train_idx),
                    np.asarray(test_idx),
                    self._inner_workers_for_outer_job(outer_n_jobs),
                )
                for outer_fold, train_idx, test_idx in splits
            )
            fold_results = sorted(fold_results, key=lambda item: item["outer_fold"])
            prediction_frames = [item["predictions"] for item in fold_results]
            for item in fold_results:
                self.bow_prediction_frames.extend(item["bow_prediction_frames"])
                self.htr_nuisance_prediction_frames.extend(
                    item.get("htr_nuisance_prediction_frames", [])
                )
                self.htr_effect_prediction_frames.extend(
                    item.get("htr_effect_prediction_frames", [])
                )
                self.htr_attention_rows.extend(item.get("htr_attention_rows", []))
                self.importance_rows.extend(item["importance_rows"])
                self.embedding_evidence_rows.extend(item["embedding_evidence_rows"])
                self.agent_rows.extend(item["agent_rows"])
                self.extracted_feature_diagnostic_rows.extend(
                    item["extracted_feature_diagnostic_rows"]
                )
                self.parsimony_review_rows.extend(
                    item.get("parsimony_review_rows", [])
                )
                self.feature_set_rows.extend(item["feature_set_rows"])
                self.outer_metric_rows.extend(item["outer_metric_rows"])
        else:
            prediction_frames: List[pd.DataFrame] = []
            for outer_fold, train_idx, test_idx in splits:
                logger.info(
                    "Multi-model agentic fold %s: train=%s test=%s",
                    outer_fold,
                    len(train_idx),
                    len(test_idx),
                )
                prediction_frames.append(
                    self._run_one_analysis_split(
                        outer_fold=outer_fold,
                        train_idx=train_idx,
                        test_idx=test_idx,
                    )
                )

        results_df = pd.concat(prediction_frames).sort_values("_oci_row_id")
        self._save_predictions(results_df)
        self._save_artifacts()

    def _run_one_analysis_split_isolated(
        self,
        outer_fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        outer_n_jobs: int,
    ) -> Dict[str, Any]:
        logger.info(
            "Multi-model agentic isolated fold %s: train=%s test=%s",
            outer_fold,
            len(train_idx),
            len(test_idx),
        )
        fold_runner = MultiModelAgenticForestRunner(
            dataset=self.dataset,
            config=self.config,
            output_path=(
                self.artifact_dir
                / f"outer_fold_{int(outer_fold):03d}"
                / "predictions.parquet"
            ),
            num_workers=self._inner_workers_for_outer_job(outer_n_jobs),
        )
        predictions = fold_runner._run_one_analysis_split(
            outer_fold=outer_fold,
            train_idx=train_idx,
            test_idx=test_idx,
        )
        return {
            "outer_fold": int(outer_fold),
            "predictions": predictions,
            "bow_prediction_frames": fold_runner.bow_prediction_frames,
            "htr_nuisance_prediction_frames": fold_runner.htr_nuisance_prediction_frames,
            "htr_effect_prediction_frames": fold_runner.htr_effect_prediction_frames,
            "htr_attention_rows": fold_runner.htr_attention_rows,
            "importance_rows": fold_runner.importance_rows,
            "embedding_evidence_rows": fold_runner.embedding_evidence_rows,
            "agent_rows": fold_runner.agent_rows,
            "extracted_feature_diagnostic_rows": (
                fold_runner.extracted_feature_diagnostic_rows
            ),
            "parsimony_review_rows": fold_runner.parsimony_review_rows,
            "feature_set_rows": fold_runner.feature_set_rows,
            "outer_metric_rows": fold_runner.outer_metric_rows,
        }

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
            "No held-out split configured for multi_model_agentic_forest; "
            "variable discovery and final estimates will use the full dataset."
        )
        return [(1, all_idx, all_idx)]

    def _run_one_analysis_split(
        self,
        outer_fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> pd.DataFrame:
        self._ensure_prespecified_features()
        discovery_df = self.dataset.iloc[train_idx].reset_index(drop=True)
        bow_result = self._fit_bow_discovery(discovery_df, outer_fold)
        self.bow_prediction_frames.append(bow_result["predictions"])
        artifact_context = self._artifact_agent_context(bow_result["context"])
        for view in bow_result["importance"].get("views", []) or []:
            feature_importance = {
                key: value
                for key, value in view.items()
                if key not in {"view_name", "view_index", "view_config", "metrics"}
            }
            self.importance_rows.append(
                {
                    "record_type": "view",
                    "outer_fold": int(outer_fold),
                    "view_index": int(view.get("view_index", -1)),
                    "view_name": view.get("view_name"),
                    "view_config": view.get("view_config"),
                    "metrics": view.get("metrics"),
                    "feature_importance": feature_importance,
                }
            )
        ensemble_importance = bow_result["importance"].get("ensemble_r")
        if isinstance(ensemble_importance, dict):
            for view in ensemble_importance.get("views", []) or []:
                feature_importance = {
                    key: value
                    for key, value in view.items()
                    if key not in {"view_name", "view_index", "view_config", "metrics"}
                }
                self.importance_rows.append(
                    {
                        "record_type": "ensemble_r_view",
                        "outer_fold": int(outer_fold),
                        "view_index": int(view.get("view_index", -1)),
                        "view_name": view.get("view_name"),
                        "view_config": view.get("view_config"),
                        "metrics": view.get("metrics"),
                        "feature_importance": feature_importance,
                    }
                )
            self.importance_rows.append(
                {
                    "record_type": "ensemble_r_consensus",
                    "outer_fold": int(outer_fold),
                    "phrase_consensus": ensemble_importance.get("phrase_consensus", []),
                }
            )
        self.importance_rows.append(
            {
                "record_type": "consensus",
                "outer_fold": int(outer_fold),
                "phrase_consensus": bow_result["importance"].get("phrase_consensus", []),
                "context": artifact_context,
            }
        )
        embedding_evidence = bow_result.get("embedding_contrast_evidence") or {}
        if embedding_evidence:
            self.embedding_evidence_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "embedding_contrast_evidence": (
                        embedding_evidence
                        if self.search_config.save_agent_context
                        else redact_embedding_contrast_evidence(embedding_evidence)
                    ),
                }
            )

        selected_specs = self._propose_selected_specs(
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            bow_context=bow_result["context"],
        )

        self.dataset = self.extraction_provider.ensure_features(
            self.dataset,
            selected_specs,
        )
        train_df = self.dataset.iloc[train_idx].copy()
        test_df = self.dataset.iloc[test_idx].copy()
        selected_specs = self._filter_specs_by_extraction_coverage(
            train_df,
            selected_specs,
        )
        review_result = self._review_extracted_features_if_needed(
            outer_fold=outer_fold,
            train_idx=train_idx,
            selected_specs=selected_specs,
            bow_result=bow_result,
            embedding_evidence=embedding_evidence,
        )
        selected_specs = review_result["selected_specs"]
        train_df = self.dataset.iloc[train_idx].copy()
        test_df = self.dataset.iloc[test_idx].copy()
        parsimony_result = self._run_mandatory_parsimony_review(
            outer_fold=outer_fold,
            train_idx=train_idx,
            selected_specs=selected_specs,
            bow_result=bow_result,
            embedding_evidence=embedding_evidence,
        )
        selected_specs = parsimony_result["selected_specs"]
        train_df = self.dataset.iloc[train_idx].copy()
        test_df = self.dataset.iloc[test_idx].copy()
        self.feature_set_rows.append(
            {
                "outer_fold": int(outer_fold),
                "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
                "confounders": [
                    spec.name for spec in selected_specs if "confounder" in spec.roles
                ],
                "effect_modifiers": [
                    spec.name
                    for spec in selected_specs
                    if "effect_modifier" in spec.roles
                ],
                "extracted_feature_review": review_result["summary"],
                "parsimony_review": parsimony_result["summary"],
            }
        )

        final_eval: SplitEvaluation = self.evaluator.evaluate_split(
            train_df=train_df,
            test_df=test_df,
            specs=selected_specs,
            fold_id=outer_fold,
        )
        predictions = final_eval.predictions.copy()
        predictions["outer_fold"] = int(outer_fold)
        predictions["selected_feature_names"] = ",".join(
            spec.name for spec in selected_specs
        )
        predictions["selected_feature_roles"] = _format_selected_feature_roles(
            selected_specs
        )
        predictions["selected_confounder_names"] = ",".join(
            spec.name for spec in selected_specs if "confounder" in spec.roles
        )
        predictions["selected_effect_modifier_names"] = ",".join(
            spec.name for spec in selected_specs if "effect_modifier" in spec.roles
        )

        self.outer_metric_rows.append(
            {
                "outer_fold": int(outer_fold),
                "n_selected_features": int(len(selected_specs)),
                **_prefix_metrics(
                    "extracted_feature_review_",
                    _scalar_metrics(review_result["summary"]),
                ),
                **_prefix_metrics(
                    "parsimony_review_",
                    _scalar_metrics(parsimony_result["summary"]),
                ),
                **_scalar_metrics(final_eval.metrics),
                **_prefix_metrics("bow_", bow_result["metrics"]),
            }
        )
        return predictions

    def _fit_bow_discovery(
        self,
        discovery_df: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        texts = _normalize_texts(discovery_df[self.config.text_column].fillna(""))
        y = discovery_df[self.config.outcome_column].to_numpy(dtype=float)
        t = discovery_df[self.config.treatment_column].to_numpy(dtype=float)
        prespecified_specs = self._initial_specs()
        explicit_feature_dicts = _columns_to_feature_dicts(
            discovery_df,
            prespecified_specs,
        )

        view_results: List[Dict[str, Any]] = []
        for view_index, view in enumerate(self.nn_config.bow_views):
            view_results.append(
                self._fit_one_bow_view(
                    discovery_df=discovery_df,
                    texts=texts,
                    y=y,
                    t=t,
                    outer_fold=outer_fold,
                    view=view,
                    view_index=view_index,
                    explicit_feature_dicts=explicit_feature_dicts,
                    explicit_specs=prespecified_specs,
                )
            )

        htr_nuisance_result = self._fit_htr_nuisance_discovery(
            discovery_df,
            outer_fold,
        )
        htr_evidence = None
        if htr_nuisance_result is not None:
            self.htr_nuisance_prediction_frames.append(
                htr_nuisance_result["predictions"]
            )
            self.htr_attention_rows.extend(htr_nuisance_result.get("attention", []))
            htr_evidence = {
                "nuisance": {
                    "metrics": htr_nuisance_result.get("metrics", {}),
                    "attention": htr_nuisance_result.get("attention", []),
                }
            }

        ensemble_result = self._fit_ensemble_r_discovery(
            discovery_df=discovery_df,
            texts=texts,
            y=y,
            t=t,
            outer_fold=outer_fold,
            view_results=view_results,
            nuisance_results=(
                [*view_results, htr_nuisance_result]
                if htr_nuisance_result is not None
                else view_results
            ),
            explicit_feature_dicts=explicit_feature_dicts,
            explicit_specs=prespecified_specs,
        )

        if ensemble_result is not None and htr_nuisance_result is not None:
            htr_effect_result = self._fit_htr_effect_discovery(
                discovery_df=discovery_df,
                outer_fold=outer_fold,
                nuisance_predictions=ensemble_result["nuisance_predictions"],
            )
            if htr_effect_result is not None:
                ensemble_result["htr_effect_result"] = htr_effect_result
                self.htr_effect_prediction_frames.append(
                    htr_effect_result["predictions"]
                )
                self.htr_attention_rows.extend(htr_effect_result.get("attention", []))
                assert htr_evidence is not None
                htr_evidence["effect"] = {
                    "metrics": htr_effect_result.get("metrics", {}),
                    "attention": htr_effect_result.get("attention", []),
                }

        prediction_frames = [result["predictions"] for result in view_results]
        if htr_nuisance_result is not None:
            prediction_frames.append(htr_nuisance_result["predictions"])
        if ensemble_result is not None:
            prediction_frames.extend(
                result["predictions"]
                for result in ensemble_result.get("view_results", [])
            )
            htr_effect_result = ensemble_result.get("htr_effect_result")
            if htr_effect_result is not None:
                prediction_frames.append(htr_effect_result["predictions"])
        predictions = pd.concat(prediction_frames, ignore_index=True)
        metrics = _multi_view_metrics(view_results)
        if htr_nuisance_result is not None:
            metrics["htr_nuisance"] = htr_nuisance_result.get("metrics", {})
        if ensemble_result is not None:
            metrics["ensemble_r"] = ensemble_result["metrics"]
            if ensemble_result.get("htr_effect_result") is not None:
                metrics["htr_effect"] = ensemble_result["htr_effect_result"].get(
                    "metrics",
                    {},
                )
            for key, value in _scalar_metrics(ensemble_result["metrics"]).items():
                metrics[f"ensemble_{key}"] = value
        importance = _multi_view_importance(
            view_results,
            top_n=int(self.nn_config.top_n_features),
        )
        if ensemble_result is not None:
            importance["ensemble_r"] = ensemble_result["importance"]

        pseudo_targets = [result["pseudo_target"] for result in view_results]
        t_resids = [result["t_resid"] for result in view_results]
        pseudo_target_names = [view.name for view in self.nn_config.bow_views]
        if ensemble_result is not None:
            pseudo_targets.append(ensemble_result["pseudo_target"])
            t_resids.append(ensemble_result["t_resid"])
            pseudo_target_names.append(
                str(
                    ensemble_result.get("target_source")
                    or "ensemble_mean_nuisance"
                )
            )
        embedding_evidence = self._build_embedding_contrast_evidence(
            discovery_df=discovery_df,
            y=y,
            t=t,
            pseudo_target=pseudo_targets,
            t_resid=t_resids,
            pseudo_target_names=pseudo_target_names,
            importance=importance,
        )
        context = self._build_agent_context(
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            metrics=metrics,
            importance=importance,
            embedding_evidence=embedding_evidence,
            htr_evidence=htr_evidence,
        )
        return {
            "predictions": predictions,
            "metrics": metrics,
            "importance": importance,
            "embedding_contrast_evidence": embedding_evidence,
            "htr_evidence": htr_evidence or {},
            "context": context,
        }

    def _fit_one_bow_view(
        self,
        *,
        discovery_df: pd.DataFrame,
        texts: Sequence[str],
        y: np.ndarray,
        t: np.ndarray,
        outer_fold: int,
        view: BoWViewConfig,
        view_index: int,
        explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
        explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    ) -> Dict[str, Any]:
        e_hat = self._crossfit_binary(
            texts,
            t,
            "treatment",
            outer_fold,
            view=view,
            view_index=view_index,
            explicit_feature_dicts=explicit_feature_dicts,
            explicit_specs=explicit_specs,
        )
        if self.config.outcome_type == "continuous":
            m_hat = self._crossfit_continuous(
                texts,
                y,
                "outcome",
                outer_fold,
                view=view,
                view_index=view_index,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
            )
        else:
            m_hat = self._crossfit_binary(
                texts,
                y,
                "outcome",
                outer_fold,
                view=view,
                view_index=view_index,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
            )

        e_clipped = np.clip(e_hat, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip)
        t_resid = t - e_clipped
        y_resid = y - m_hat
        pseudo_target = y_resid / t_resid

        tau_hat = self._crossfit_pseudo_target(
            texts,
            pseudo_target,
            t_resid**2,
            outer_fold,
            view=view,
            view_index=view_index,
            explicit_feature_dicts=explicit_feature_dicts,
            explicit_specs=explicit_specs,
        )
        r_loss = (y_resid - tau_hat * t_resid) ** 2
        r_loss_at_zero = y_resid**2

        predictions = pd.DataFrame(
            {
                "_oci_row_id": discovery_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "view_index": int(view_index),
                "view_name": str(view.name),
                "e_hat": e_hat,
                "m_hat": m_hat,
                "y_residual": y_resid,
                "t_residual": t_resid,
                "pseudo_target": pseudo_target,
                "tau_hat_multi_model": tau_hat,
                "r_loss": r_loss,
                "r_loss_at_zero_tau": r_loss_at_zero,
            }
        )

        metrics = self._bow_metrics(
            y=y,
            t=t,
            e_hat=e_hat,
            m_hat=m_hat,
            pseudo_target=pseudo_target,
            tau_hat=tau_hat,
            y_resid=y_resid,
            t_resid=t_resid,
            r_loss=r_loss,
            r_loss_at_zero=r_loss_at_zero,
            discovery_df=discovery_df,
        )
        importance = self._fit_feature_importance_models(
            texts=texts,
            y=y,
            t=t,
            pseudo_target=pseudo_target,
            pseudo_target_sample_weight=t_resid**2,
            view=view,
            explicit_feature_dicts=explicit_feature_dicts,
            explicit_specs=explicit_specs,
        )
        return {
            "predictions": predictions,
            "metrics": metrics,
            "importance": importance,
            "e_hat": e_hat,
            "m_hat": m_hat,
            "pseudo_target": pseudo_target,
            "t_resid": t_resid,
            "view": view,
            "view_index": int(view_index),
        }

    def _fit_ensemble_r_discovery(
        self,
        *,
        discovery_df: pd.DataFrame,
        texts: Sequence[str],
        y: np.ndarray,
        t: np.ndarray,
        outer_fold: int,
        view_results: Sequence[Dict[str, Any]],
        nuisance_results: Optional[Sequence[Dict[str, Any]]] = None,
        explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
        explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    ) -> Optional[Dict[str, Any]]:
        if len(view_results) < 1:
            return None
        nuisance_results = [
            result for result in (nuisance_results or view_results) if result is not None
        ]
        if len(nuisance_results) < 2:
            return None

        e_hat = np.nanmean(
            np.vstack(
                [np.asarray(result["e_hat"], dtype=float) for result in nuisance_results]
            ),
            axis=0,
        )
        m_hat = np.nanmean(
            np.vstack(
                [np.asarray(result["m_hat"], dtype=float) for result in nuisance_results]
            ),
            axis=0,
        )
        e_clipped = np.clip(e_hat, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip)
        t_resid = t - e_clipped
        y_resid = y - m_hat
        pseudo_target = y_resid / t_resid
        sample_weight = t_resid**2
        r_loss_at_zero = y_resid**2
        nuisance_source_names = [
            str(result.get("view_name") or getattr(result.get("view"), "name", "model"))
            for result in nuisance_results
        ]
        target_source = (
            "ensemble_mean_nuisance_with_htr"
            if any(str(name).startswith("htr") for name in nuisance_source_names)
            else "ensemble_mean_nuisance"
        )
        nuisance_predictions = pd.DataFrame(
            {
                "_oci_row_id": discovery_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "e_hat": e_hat,
                "m_hat": m_hat,
                "y_residual": y_resid,
                "t_residual": t_resid,
                "r_pseudo_outcome": pseudo_target,
                "pseudo_target": pseudo_target,
                "r_loss_at_zero_tau": r_loss_at_zero,
                "nuisance_fold": -1,
                "target_source": target_source,
            }
        )

        ensemble_view_results: List[Dict[str, Any]] = []
        for result in view_results:
            view = result["view"]
            view_index = int(result["view_index"])
            tau_hat = self._crossfit_pseudo_target(
                texts,
                pseudo_target,
                sample_weight,
                outer_fold,
                view=view,
                view_index=view_index,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
                random_seed_offset=50_000,
            )
            r_loss = (y_resid - tau_hat * t_resid) ** 2
            view_name = f"ensemble_r__{view.name}"
            predictions = pd.DataFrame(
                {
                    "_oci_row_id": discovery_df["_oci_row_id"].to_numpy(),
                    "outer_fold": int(outer_fold),
                    "view_index": view_index,
                    "view_name": view_name,
                    "e_hat": e_hat,
                    "m_hat": m_hat,
                    "y_residual": y_resid,
                    "t_residual": t_resid,
                    "pseudo_target": pseudo_target,
                    "tau_hat_multi_model": tau_hat,
                    "r_loss": r_loss,
                    "r_loss_at_zero_tau": r_loss_at_zero,
                    "target_source": target_source,
                }
            )
            metrics = self._bow_metrics(
                y=y,
                t=t,
                e_hat=e_hat,
                m_hat=m_hat,
                pseudo_target=pseudo_target,
                tau_hat=tau_hat,
                y_resid=y_resid,
                t_resid=t_resid,
                r_loss=r_loss,
                r_loss_at_zero=r_loss_at_zero,
                discovery_df=discovery_df,
            )
            importance = self._fit_feature_importance_models(
                texts=texts,
                y=y,
                t=t,
                pseudo_target=pseudo_target,
                pseudo_target_sample_weight=sample_weight,
                view=view,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
            )
            ensemble_view_results.append(
                {
                    "predictions": predictions,
                    "metrics": metrics,
                    "importance": importance,
                    "pseudo_target": pseudo_target,
                    "t_resid": t_resid,
                    "view": view,
                    "view_name": view_name,
                    "view_index": view_index,
                }
            )

        metrics = _multi_view_metrics(ensemble_view_results)
        metrics["target_source"] = target_source
        metrics["n_nuisance_sources"] = int(len(nuisance_source_names))
        metrics["nuisance_sources"] = nuisance_source_names
        metrics["pseudo_target_construction"] = (
            "mean nuisance predictions across BoW and HTR models, then "
            "(Y - mean_m_hat) / (T - mean_e_hat)"
            if target_source == "ensemble_mean_nuisance_with_htr"
            else "mean nuisance predictions across BoW views, then "
            "(Y - mean_m_hat) / (T - mean_e_hat)"
        )
        importance = _multi_view_importance(
            ensemble_view_results,
            top_n=int(self.nn_config.top_n_features),
        )
        importance["target_source"] = target_source
        importance["nuisance_sources"] = nuisance_source_names
        importance["pseudo_target_construction"] = metrics[
            "pseudo_target_construction"
        ]
        return {
            "view_results": ensemble_view_results,
            "metrics": metrics,
            "importance": importance,
            "pseudo_target": pseudo_target,
            "t_resid": t_resid,
            "nuisance_predictions": nuisance_predictions,
            "target_source": target_source,
        }

    def _crossfit_binary(
        self,
        texts: Sequence[str],
        labels: np.ndarray,
        label_name: str,
        outer_fold: int,
        *,
        view: BoWViewConfig,
        view_index: int,
        explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
        explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    ) -> np.ndarray:
        labels = labels.astype(int)
        oof = np.full(len(labels), np.nan, dtype=float)
        random_state = (
            11_000
            + 100 * outer_fold
            + 1_000 * int(view_index)
            + (1 if label_name == "outcome" else 2)
        )
        split_items = list(
            enumerate(
                _binary_split_items(
                    labels,
                    requested_folds=self.nn_config.nuisance_folds,
                    random_state=random_state,
                ),
                start=1,
            )
        )
        folds = len(split_items)
        vectorizer_params = self._vectorizer_params(view)
        model_params = self._model_params(view)

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            logger.info(
                "Outer fold %s BoW view=%s %s nuisance fold %s/%s: train=%s heldout=%s",
                outer_fold,
                view.name,
                label_name,
                fold,
                folds,
                len(fit_pos),
                len(heldout_pos),
            )
            return _fit_binary_bow_fold(
                texts,
                labels,
                fit_pos,
                heldout_pos,
                vectorizer_params,
                model_params,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
                random_state=17 + fold,
            )

        results = self._run_fold_tasks(run_fold, split_items)
        for heldout_pos, values in results:
            oof[heldout_pos] = values
        return np.clip(oof, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip)

    def _crossfit_continuous(
        self,
        texts: Sequence[str],
        values: np.ndarray,
        label_name: str,
        outer_fold: int,
        *,
        view: BoWViewConfig,
        view_index: int,
        explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
        explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    ) -> np.ndarray:
        oof = np.full(len(values), np.nan, dtype=float)
        folds = _bounded_fold_count(self.nn_config.nuisance_folds, len(values))
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=12_000 + 100 * outer_fold + 1_000 * int(view_index),
        )
        split_items = list(enumerate(splitter.split(texts), start=1))
        vectorizer_params = self._vectorizer_params(view)
        model_params = self._model_params(view)

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            logger.info(
                "Outer fold %s BoW view=%s %s nuisance fold %s/%s: train=%s heldout=%s",
                outer_fold,
                view.name,
                label_name,
                fold,
                folds,
                len(fit_pos),
                len(heldout_pos),
            )
            return _fit_regression_bow_fold(
                texts,
                values,
                fit_pos,
                heldout_pos,
                vectorizer_params,
                model_params,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
                random_state=17 + fold,
            )

        results = self._run_fold_tasks(run_fold, split_items)
        for heldout_pos, fold_values in results:
            oof[heldout_pos] = fold_values
        return oof

    def _crossfit_pseudo_target(
        self,
        texts: Sequence[str],
        pseudo_target: np.ndarray,
        sample_weight: Optional[np.ndarray],
        outer_fold: int,
        *,
        view: BoWViewConfig,
        view_index: int,
        explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
        explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
        random_seed_offset: int = 0,
    ) -> np.ndarray:
        oof = np.full(len(pseudo_target), np.nan, dtype=float)
        folds = _bounded_fold_count(self.nn_config.effect_folds, len(pseudo_target))
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=(
                13_000
                + int(random_seed_offset)
                + outer_fold
                + 1_000 * int(view_index)
            ),
        )
        split_items = list(enumerate(splitter.split(texts), start=1))
        vectorizer_params = self._vectorizer_params(view)
        model_params = self._model_params(view)

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            logger.info(
                "Outer fold %s BoW view=%s pseudo-target fold %s/%s: train=%s heldout=%s",
                outer_fold,
                view.name,
                fold,
                folds,
                len(fit_pos),
                len(heldout_pos),
            )
            return _fit_regression_bow_fold(
                texts,
                pseudo_target,
                fit_pos,
                heldout_pos,
                vectorizer_params,
                model_params,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
                sample_weight=sample_weight,
                random_state=17 + int(random_seed_offset) + fold,
            )

        results = self._run_fold_tasks(run_fold, split_items)
        for heldout_pos, values in results:
            oof[heldout_pos] = values
        return oof

    def _fit_feature_importance_models(
        self,
        texts: Sequence[str],
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: np.ndarray,
        pseudo_target_sample_weight: Optional[np.ndarray],
        *,
        view: BoWViewConfig,
        explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
        explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    ) -> Dict[str, Any]:
        vectorizer = self._make_vectorizer(view)
        x_text = vectorizer.fit_transform(texts)
        x_model, features, explicit_feature_names = _append_explicit_features_full(
            x_text,
            np.asarray(vectorizer.get_feature_names_out()),
            explicit_feature_dicts=explicit_feature_dicts,
            explicit_specs=explicit_specs,
        )

        def fit_treatment() -> np.ndarray:
            if len(np.unique(t.astype(int))) < 2:
                return np.zeros(len(features), dtype=float)
            treatment_model = self._make_classifier(view, random_state=101)
            treatment_model.fit(x_model, t.astype(int))
            return _model_feature_scores(treatment_model, len(features))

        def fit_outcome() -> np.ndarray:
            if self.config.outcome_type == "continuous":
                outcome_model = self._make_regressor(view, random_state=202)
                outcome_model.fit(x_model, y)
                return _model_feature_scores(outcome_model, len(features))
            if len(np.unique(y.astype(int))) < 2:
                return np.zeros(len(features), dtype=float)
            outcome_model = self._make_classifier(view, random_state=202)
            outcome_model.fit(x_model, y.astype(int))
            return _model_feature_scores(outcome_model, len(features))

        def fit_effect() -> np.ndarray:
            effect_model = self._make_regressor(view, random_state=303)
            _fit_regressor(
                effect_model,
                x_model,
                pseudo_target,
                sample_weight=pseudo_target_sample_weight,
            )
            return _model_feature_scores(effect_model, len(features))

        n_jobs = self._feature_importance_n_jobs()
        if n_jobs > 1:
            logger.info(
                "Multi-model BoW feature-importance parallelism: tasks=3 n_jobs=%s",
                n_jobs,
            )
            treatment_coef, outcome_coef, effect_coef = Parallel(
                n_jobs=n_jobs,
                backend="threading",
                batch_size=1,
            )(
                delayed(task)()
                for task in (fit_treatment, fit_outcome, fit_effect)
            )
        else:
            treatment_coef = fit_treatment()
            outcome_coef = fit_outcome()
            effect_coef = fit_effect()

        top_n = int(self.nn_config.top_n_features)
        confounder_score = np.abs(treatment_coef) * np.abs(outcome_coef)
        return {
            "view_name": str(view.name),
            "view_config": _bow_view_to_dict(view),
            "n_features": int(len(features)),
            "n_bow_features": int(len(vectorizer.get_feature_names_out())),
            "n_prespecified_features": int(len(explicit_specs or [])),
            "n_prespecified_raw_features": int(len(explicit_feature_names)),
            "prespecified_raw_feature_names": explicit_feature_names,
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
            "treatment_positive": _top_feature_rows(
                features,
                treatment_coef,
                top_n,
                descending=True,
            ),
            "treatment_negative": _top_feature_rows(
                features,
                treatment_coef,
                top_n,
                descending=False,
            ),
            "outcome_positive": _top_feature_rows(
                features,
                outcome_coef,
                top_n,
                descending=True,
            ),
            "outcome_negative": _top_feature_rows(
                features,
                outcome_coef,
                top_n,
                descending=False,
            ),
            "pseudo_target_positive": _top_feature_rows(
                features,
                effect_coef,
                top_n,
                descending=True,
            ),
            "pseudo_target_negative": _top_feature_rows(
                features,
                effect_coef,
                top_n,
                descending=False,
            ),
        }

    def _bow_metrics(
        self,
        *,
        y: np.ndarray,
        t: np.ndarray,
        e_hat: np.ndarray,
        m_hat: np.ndarray,
        pseudo_target: np.ndarray,
        tau_hat: np.ndarray,
        y_resid: np.ndarray,
        t_resid: np.ndarray,
        r_loss: np.ndarray,
        r_loss_at_zero: np.ndarray,
        discovery_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "treatment_auroc": _safe_roc_auc(t, e_hat),
            "treatment_brier": _finite_or_none(brier_score_loss(t, e_hat)),
            "treatment_log_loss": _finite_or_none(log_loss(t, e_hat)),
            "pseudo_target_mean": _finite_or_none(np.mean(pseudo_target)),
            "pseudo_target_std": _finite_or_none(np.std(pseudo_target)),
            "tau_hat_mean": _finite_or_none(np.mean(tau_hat)),
            "tau_hat_std": _finite_or_none(np.std(tau_hat)),
            "r_loss_mean": _finite_or_none(np.mean(r_loss)),
            "r_loss_at_zero_tau_mean": _finite_or_none(np.mean(r_loss_at_zero)),
            "tau_hat_pseudo_target_corr": _safe_corr(tau_hat, pseudo_target),
        }
        if self.config.outcome_type == "continuous":
            metrics["outcome_rmse"] = _finite_or_none(
                np.sqrt(mean_squared_error(y, m_hat))
            )
        else:
            metrics["outcome_auroc"] = _safe_roc_auc(y, m_hat)
            metrics["outcome_brier"] = _finite_or_none(brier_score_loss(y, m_hat))
            metrics["outcome_log_loss"] = _finite_or_none(log_loss(y, m_hat))
        zero = metrics["r_loss_at_zero_tau_mean"]
        loss = metrics["r_loss_mean"]
        if zero is not None and zero > 0 and loss is not None:
            metrics["r_loss_relative_improvement"] = float(1.0 - loss / zero)
        if "true_ite_prob" in discovery_df.columns:
            true_ite = discovery_df["true_ite_prob"].to_numpy(dtype=float)
            metrics["tau_hat_true_ite_corr"] = _safe_corr(tau_hat, true_ite)
            metrics["pseudo_target_true_ite_corr"] = _safe_corr(pseudo_target, true_ite)
        if "true_treatment_prob" in discovery_df.columns:
            metrics["treatment_true_prob_corr"] = _safe_corr(
                e_hat,
                discovery_df["true_treatment_prob"].to_numpy(dtype=float),
            )
        if "true_outcome_prob" in discovery_df.columns:
            metrics["outcome_true_prob_corr"] = _safe_corr(
                m_hat,
                discovery_df["true_outcome_prob"].to_numpy(dtype=float),
            )
        return metrics

    def _build_agent_context(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        metrics: Dict[str, Any],
        importance: Dict[str, Any],
        embedding_evidence: Optional[Dict[str, Any]] = None,
        htr_evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = {
            "prompt_version": "multi_model_agentic_forest_v1",
            "outer_fold": int(outer_fold),
            "max_proposals": int(self.nn_config.candidate_proposals_per_fold),
            "clinical_question": self.config.clinical_question,
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "instructions": [
                "Review every sparse bag-of-words model view. Each view has its "
                "own honest nuisance predictions, R pseudo-target, metrics, and "
                "feature-importance summaries.",
                "Use feature_importance.phrase_consensus as a cross-view summary, "
                "but also inspect feature_importance.views for useful signals that "
                "appear in only one model or n-gram setting.",
                "When embedding_contrast_evidence is present, use aligned real-text "
                "chunks and concept scores as retrieval evidence, not as direct "
                "vector interpretations.",
                "Treat within-arm outcome, treatment-outcome cell interaction, "
                "and orthogonal R-score embedding contrasts as effect-modifier "
                "hypothesis evidence when their retrieved chunks recur coherently.",
                "When htr_attention_evidence is present, use the highly attended "
                "tokens/spans from HTR nuisance and R-stage models as neural text "
                "evidence for variables that may explain treatment assignment, "
                "baseline outcome risk, or heterogeneous treatment effect.",
                "Treat ensemble_mean_nuisance_with_htr diagnostics as the R-loss "
                "signal built from BoW nuisance predictions plus HTR nuisance "
                "predictions, not as a replacement for the BoW views.",
                "Suggest explicit pre-treatment patient-level variables, not raw text tokens.",
                "Use variables predictive of both treatment and outcome as confounders.",
                "Use variables predictive of the pseudo-target as effect modifiers.",
                "Avoid near-duplicate aliases for the same extraction target; a "
                "separate alias-resolution pass may merge proposal names.",
            ],
            "current_features": [_spec_to_dict(spec) for spec in self._initial_specs()],
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
                        "rationale": "which BoW features support this variable",
                        "expected_signal": "treatment, outcome, or pseudo-target signal expected",
                    }
                ]
            },
        }
        if embedding_evidence:
            context["embedding_contrast_evidence"] = embedding_evidence
        if htr_evidence:
            context["htr_attention_evidence"] = htr_evidence
        compact_context = _compact_multi_model_agent_context(context)
        prompt_chars = len(
            json.dumps(compact_context, separators=(",", ":"), default=_json_default)
        )
        logger.info(
            "Multi-model agent prompt context outer_fold=%s: %.1fK JSON chars",
            outer_fold,
            prompt_chars / 1000.0,
        )
        return compact_context

    def _embedding_contrast_enabled(self) -> bool:
        embedding_config = getattr(self.nn_config, "embedding_contrast", None)
        return bool(getattr(embedding_config, "enabled", False))

    def _htr_evidence_enabled(self) -> bool:
        return bool(getattr(self.nn_config, "htr_evidence_enabled", True))

    def _validate_required_evidence_sources(self) -> None:
        embedding_config = getattr(self.nn_config, "embedding_contrast", None)
        if not bool(getattr(embedding_config, "enabled", False)):
            reason = str(getattr(embedding_config, "disable_reason", "") or "").strip()
            if not reason:
                raise ValueError(
                    "multi_model_agentic_forest requires embedding contrast evidence; "
                    "set embedding_contrast.disable_reason when intentionally disabling it"
                )
            logger.warning("Embedding contrast evidence disabled: %s", reason)
        if not self._htr_evidence_enabled():
            reason = str(
                getattr(self.nn_config, "htr_evidence_disable_reason", "") or ""
            ).strip()
            if not reason:
                raise ValueError(
                    "multi_model_agentic_forest requires HTR attention/span evidence; "
                    "set htr_evidence_disable_reason when intentionally disabling it"
                )
            logger.warning("HTR attention/span evidence disabled: %s", reason)

    def _embedding_contrast_generator(self) -> EmbeddingContrastEvidenceGenerator:
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
        if self._default_htr_evidence_provider is None:
            self._default_htr_evidence_provider = MultiModelHTREvidenceProvider(
                config=self.config,
                output_dir=self.artifact_dir,
                device=self.device,
                gpu_ids=self.gpu_ids,
                num_workers=self.num_workers,
            )
        return self._default_htr_evidence_provider

    def _fit_htr_nuisance_discovery(
        self,
        discovery_df: pd.DataFrame,
        outer_fold: int,
    ) -> Optional[Dict[str, Any]]:
        if not self._htr_evidence_enabled():
            return None
        try:
            result = self._htr_provider().fit_nuisance(discovery_df, outer_fold)
        except Exception as exc:
            raise RuntimeError(
                "Required HTR nuisance evidence generation failed"
            ) from exc
        predictions = _align_htr_prediction_frame(
            result.get("predictions"),
            discovery_df,
            required_columns=["e_hat", "m_hat"],
            source="htr_nuisance",
        )
        attention = [dict(row) for row in result.get("attention", []) or []]
        for row in attention:
            row.setdefault("model_family", "htr")
        predictions["model_family"] = "htr"
        predictions["view_name"] = "htr_nuisance"
        predictions["target_source"] = "htr_nuisance"
        metrics = _htr_nuisance_metrics(
            discovery_df=discovery_df,
            predictions=predictions,
            treatment_column=self.config.treatment_column,
            outcome_column=self.config.outcome_column,
            outcome_type=self.config.outcome_type,
        )
        return {
            "model_family": "htr",
            "view_name": "htr_nuisance",
            "predictions": predictions,
            "attention": attention,
            "metrics": metrics,
            "e_hat": predictions["e_hat"].to_numpy(dtype=float),
            "m_hat": predictions["m_hat"].to_numpy(dtype=float),
        }

    def _fit_htr_effect_discovery(
        self,
        discovery_df: pd.DataFrame,
        outer_fold: int,
        nuisance_predictions: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        if not self._htr_evidence_enabled():
            return None
        try:
            result = self._htr_provider().fit_effect(
                discovery_df,
                nuisance_predictions,
                outer_fold,
            )
        except Exception as exc:
            raise RuntimeError("Required HTR effect evidence generation failed") from exc
        predictions = _align_htr_prediction_frame(
            result.get("predictions"),
            discovery_df,
            required_columns=["tau_hat_r_stage"],
            source="htr_effect",
        )
        attention = [dict(row) for row in result.get("attention", []) or []]
        for row in attention:
            row.setdefault("model_family", "htr")
        predictions["model_family"] = "htr"
        predictions["view_name"] = "htr_effect"
        predictions["target_source"] = "ensemble_mean_nuisance_with_htr"
        metrics = _htr_effect_metrics(predictions)
        return {
            "model_family": "htr",
            "view_name": "htr_effect",
            "predictions": predictions,
            "attention": attention,
            "metrics": metrics,
            "tau_hat": predictions["tau_hat_r_stage"].to_numpy(dtype=float),
        }

    def _build_embedding_contrast_evidence(
        self,
        *,
        discovery_df: pd.DataFrame,
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: Any,
        t_resid: Any,
        pseudo_target_names: Optional[Sequence[str]] = None,
        importance: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self._embedding_contrast_enabled():
            reason = str(
                getattr(self.nn_config.embedding_contrast, "disable_reason", "") or ""
            ).strip()
            return {"enabled": False, "disabled_reason": reason}
        try:
            generator = self._embedding_contrast_generator()
            generator.prepare(self.dataset)
            return generator.build_evidence(
                discovery_df=discovery_df,
                y=y,
                t=t,
                pseudo_target=pseudo_target,
                t_resid=t_resid,
                pseudo_target_names=(
                    pseudo_target_names
                    if pseudo_target_names is not None
                    else [view.name for view in self.nn_config.bow_views]
                ),
                importance=importance,
            )
        except Exception as exc:
            raise RuntimeError(
                "Required embedding contrast evidence generation failed"
            ) from exc

    def _artifact_agent_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.search_config.save_agent_context:
            return context
        if (
            "embedding_contrast_evidence" not in context
            and "htr_attention_evidence" not in context
        ):
            return context
        artifact_context = dict(context)
        if "embedding_contrast_evidence" in context:
            artifact_context["embedding_contrast_evidence"] = (
                redact_embedding_contrast_evidence(context["embedding_contrast_evidence"])
            )
        if "htr_attention_evidence" in context:
            artifact_context["htr_attention_evidence"] = _redact_htr_attention_evidence(
                context["htr_attention_evidence"]
            )
        return artifact_context

    def _propose_selected_specs(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        bow_context: Dict[str, Any],
    ) -> List[ExplicitFeatureSpec]:
        if bool(getattr(self.nn_config, "candidate_consistency_enabled", True)):
            return self._propose_selected_specs_with_consistency(
                outer_fold=outer_fold,
                discovery_df=discovery_df,
                bow_context=bow_context,
            )
        return self._propose_selected_specs_without_consistency(
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            bow_context=bow_context,
        )

    def _propose_selected_specs_without_consistency(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        bow_context: Dict[str, Any],
    ) -> List[ExplicitFeatureSpec]:
        del discovery_df
        raw_proposals = self.proposal_agent.propose(bow_context)
        proposal_agent_trace = _get_agent_response_trace(self.proposal_agent)
        proposals, rejected = validate_agentic_proposals(
            raw_proposals,
            current_specs=self._initial_specs(),
            search_config=self.search_config,
            allow_removals=False,
            max_additions=self.nn_config.candidate_proposals_per_fold,
        )
        proposals, alias_resolution = self._resolve_proposal_aliases(
            outer_fold=outer_fold,
            proposals=proposals,
        )
        selected_specs = _dedupe_specs(
            [
                *self._initial_specs(),
                *[
                    ExplicitFeatureSpec(
                        name=proposal.name,
                        type=proposal.type or "continuous",
                        categories=proposal.categories,
                        roles=proposal.roles,
                        description=proposal.description,
                    )
                    for proposal in proposals
                    if proposal.action == "add"
                ],
            ]
        )
        selected_specs, value_harmonization = self._harmonize_value_contracts(
            outer_fold=outer_fold,
            selected_specs=selected_specs,
        )
        self._remember_alias_reference_specs(selected_specs)
        row: Dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "raw_proposals": raw_proposals,
            "alias_resolution": alias_resolution,
            "value_harmonization": value_harmonization,
            "valid_proposals": [
                {
                    "action": proposal.action,
                    "name": proposal.name,
                    "type": proposal.type,
                    "categories": proposal.categories,
                    "roles": proposal.roles,
                    "description": proposal.description,
                    "rationale": proposal.rationale,
                    "expected_signal": proposal.expected_signal,
                }
                for proposal in proposals
            ],
            "rejected_proposals": rejected,
            "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
        }
        if self.search_config.save_agent_context:
            row["context"] = bow_context
        if self.search_config.save_agent_raw_output:
            row["agent_raw_output"] = proposal_agent_trace
        self.agent_rows.append(row)
        return selected_specs

    def _propose_selected_specs_with_consistency(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        bow_context: Dict[str, Any],
    ) -> List[ExplicitFeatureSpec]:
        full_bundle = self._propose_candidate_bundle(
            outer_fold=outer_fold,
            scope="full_outer_train",
            bow_context={
                **bow_context,
                "consistency_scope": "full_outer_train",
            },
            n_rows=len(discovery_df),
        )
        bundles = [
            full_bundle,
            *self._inner_consistency_candidate_bundles(
                outer_fold=outer_fold,
                discovery_df=discovery_df,
            ),
        ]
        all_proposals = [
            proposal
            for bundle in bundles
            for proposal in bundle.get("valid_proposals", [])
            if proposal.action == "add"
        ]
        if not all_proposals:
            selected_specs = self._initial_specs()
            selected_specs, value_harmonization = self._harmonize_value_contracts(
                outer_fold=outer_fold,
                selected_specs=selected_specs,
            )
            self.agent_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "consistency_enabled": True,
                    "proposal_bundles": [
                        _proposal_bundle_artifact(bundle) for bundle in bundles
                    ],
                    "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
                    "value_harmonization": value_harmonization,
                    "skipped": "no_valid_consistency_candidates",
                }
            )
            return selected_specs

        alias_input = _merge_duplicate_proposals(all_proposals)
        alias_resolved, alias_resolution = self._resolve_proposal_aliases(
            outer_fold=outer_fold,
            proposals=alias_input,
        )
        alias_map = {
            item["from"]: item["to"]
            for item in alias_resolution.get("applied_aliases", [])
            if item.get("from") and item.get("to")
        }
        canonical_proposals = {
            proposal.name: proposal
            for proposal in alias_resolved
            if proposal.action == "add"
        }
        candidate_summaries, threshold, inner_fold_count = (
            self._build_consistency_candidate_summaries(
                bundles=bundles,
                alias_map=alias_map,
                canonical_proposals=canonical_proposals,
            )
        )
        consistency_context = self._build_consistency_context(
            outer_fold=outer_fold,
            candidate_summaries=candidate_summaries,
            threshold=threshold,
            inner_fold_count=inner_fold_count,
        )
        consistency_proposals, consistency_selection = self._select_consistent_proposals(
            context=consistency_context,
            candidate_summaries=candidate_summaries,
            canonical_proposals=canonical_proposals,
        )
        selected_specs = self._selected_specs_from_proposals(consistency_proposals)
        selected_specs, value_harmonization = self._harmonize_value_contracts(
            outer_fold=outer_fold,
            selected_specs=selected_specs,
        )
        self._remember_alias_reference_specs(selected_specs)

        row: Dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "consistency_enabled": True,
            "proposal_bundles": [_proposal_bundle_artifact(bundle) for bundle in bundles],
            "alias_resolution": alias_resolution,
            "consistency": {
                "inner_fold_count": int(inner_fold_count),
                "min_support_folds": int(threshold),
                "candidate_summaries": candidate_summaries,
                "selection": consistency_selection,
            },
            "value_harmonization": value_harmonization,
            "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
        }
        if self.search_config.save_agent_context:
            row["consistency_context"] = consistency_context
        self.agent_rows.append(row)
        return selected_specs

    def _propose_candidate_bundle(
        self,
        *,
        outer_fold: int,
        scope: str,
        bow_context: Dict[str, Any],
        n_rows: int,
        inner_fold: Optional[int] = None,
        heldout_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        raw_proposals = self.proposal_agent.propose(bow_context)
        proposal_agent_trace = _get_agent_response_trace(self.proposal_agent)
        proposals, rejected = validate_agentic_proposals(
            raw_proposals,
            current_specs=self._initial_specs(),
            search_config=self.search_config,
            allow_removals=False,
            max_additions=self.nn_config.candidate_proposals_per_fold,
        )
        bundle: Dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "scope": scope,
            "inner_fold": inner_fold,
            "n_rows": int(n_rows),
            "heldout_rows": None if heldout_rows is None else int(heldout_rows),
            "raw_proposals": raw_proposals,
            "valid_proposals": proposals,
            "rejected_proposals": rejected,
        }
        if self.search_config.save_agent_context:
            bundle["context"] = bow_context
        if self.search_config.save_agent_raw_output:
            bundle["agent_raw_output"] = proposal_agent_trace
        return bundle

    def _inner_consistency_candidate_bundles(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        try:
            fold_count = _bounded_fold_count(
                int(self.nn_config.candidate_consistency_inner_folds),
                len(discovery_df),
            )
        except ValueError:
            return []

        splitter = KFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=51_000 + int(outer_fold),
        )
        split_items = [
            (inner_fold, np.asarray(fit_pos), np.asarray(heldout_pos))
            for inner_fold, (fit_pos, heldout_pos) in enumerate(
                splitter.split(discovery_df),
                start=1,
            )
        ]
        n_jobs = self._candidate_consistency_n_jobs(len(split_items))
        if n_jobs > 1 and self._has_external_components:
            logger.warning(
                "Candidate consistency parallelism disabled because custom "
                "agent/extractor/evaluator objects were supplied and may not be "
                "thread-safe."
            )
            n_jobs = 1

        if n_jobs <= 1:
            return [
                self._build_inner_consistency_candidate_bundle(
                    outer_fold=outer_fold,
                    discovery_df=discovery_df,
                    inner_fold=int(inner_fold),
                    fit_pos=fit_pos,
                    heldout_pos=heldout_pos,
                    total_inner_folds=fold_count,
                )
                for inner_fold, fit_pos, heldout_pos in split_items
            ]

        logger.info(
            "Multi-model candidate consistency parallelism: outer_fold=%s "
            "inner_folds=%s n_jobs=%s setting=%s backend=%s joblib_backend=%s",
            outer_fold,
            len(split_items),
            n_jobs,
            self.nn_config.candidate_consistency_parallelism,
            self.nn_config.bow_parallel_backend,
            self._parallel_backend_name(),
        )
        return Parallel(
            n_jobs=n_jobs,
            backend=self._parallel_backend_name(),
            batch_size=1,
            pre_dispatch="all",
        )(
            delayed(_build_multi_model_inner_candidate_bundle_worker)(
                self.dataset,
                self.config,
                self.artifact_dir,
                int(outer_fold),
                discovery_df,
                int(inner_fold),
                fit_pos,
                heldout_pos,
                int(fold_count),
                self._inner_workers_for_nested_job(n_jobs),
            )
            for inner_fold, fit_pos, heldout_pos in split_items
        )

    def _build_inner_consistency_candidate_bundle_isolated(
        self,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        inner_fold: int,
        fit_pos: np.ndarray,
        heldout_pos: np.ndarray,
        total_inner_folds: int,
        candidate_n_jobs: int,
    ) -> Dict[str, Any]:
        worker = MultiModelAgenticForestRunner(
            dataset=self.dataset,
            config=self.config,
            output_path=(
                self.artifact_dir
                / f"outer_{int(outer_fold):03d}_candidate_inner_{int(inner_fold):03d}"
                / "predictions.parquet"
            ),
            num_workers=self._inner_workers_for_nested_job(candidate_n_jobs),
        )
        return worker._build_inner_consistency_candidate_bundle(
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            inner_fold=inner_fold,
            fit_pos=fit_pos,
            heldout_pos=heldout_pos,
            total_inner_folds=total_inner_folds,
        )

    def _build_inner_consistency_candidate_bundle(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        inner_fold: int,
        fit_pos: np.ndarray,
        heldout_pos: np.ndarray,
        total_inner_folds: int,
    ) -> Dict[str, Any]:
        inner_df = discovery_df.iloc[np.asarray(fit_pos)].reset_index(drop=True)
        try:
            bow_result = self._fit_bow_discovery(
                inner_df,
                outer_fold=1000 * int(outer_fold) + int(inner_fold),
            )
            context = {
                **bow_result["context"],
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "consistency_scope": "inner_train",
                "inner_train_rows": int(len(fit_pos)),
                "inner_heldout_rows": int(len(heldout_pos)),
            }
            return self._propose_candidate_bundle(
                outer_fold=outer_fold,
                scope="inner_train",
                inner_fold=int(inner_fold),
                bow_context=context,
                n_rows=len(fit_pos),
                heldout_rows=len(heldout_pos),
            )
        except Exception as exc:
            logger.warning(
                "Skipping multi-model candidate consistency inner fold %s/%s "
                "for outer fold %s: %s",
                inner_fold,
                total_inner_folds,
                outer_fold,
                exc,
                exc_info=True,
            )
            return {
                "outer_fold": int(outer_fold),
                "scope": "inner_train",
                "inner_fold": int(inner_fold),
                "n_rows": int(len(fit_pos)),
                "heldout_rows": int(len(heldout_pos)),
                "error": str(exc),
                "valid_proposals": [],
                "rejected_proposals": [],
            }

    def _build_consistency_candidate_summaries(
        self,
        *,
        bundles: Sequence[Dict[str, Any]],
        alias_map: Dict[str, str],
        canonical_proposals: Dict[str, AgenticFeatureProposal],
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        inner_folds = sorted(
            {
                int(bundle["inner_fold"])
                for bundle in bundles
                if bundle.get("scope") == "inner_train"
                and bundle.get("inner_fold") is not None
                and not bundle.get("error")
            }
        )
        inner_fold_count = len(inner_folds)
        threshold = _candidate_consistency_threshold(
            inner_fold_count,
            min_folds=int(self.nn_config.candidate_consistency_min_folds),
            min_fold_fraction=float(
                self.nn_config.candidate_consistency_min_fold_fraction
            ),
        )

        summary_by_name: Dict[str, Dict[str, Any]] = {}
        for bundle in bundles:
            scope = str(bundle.get("scope") or "")
            inner_fold = bundle.get("inner_fold")
            for proposal in bundle.get("valid_proposals", []):
                if proposal.action != "add":
                    continue
                name = _resolve_alias_name(proposal.name, alias_map)
                canonical = canonical_proposals.get(name, proposal)
                summary = summary_by_name.setdefault(
                    name,
                    {
                        "name": name,
                        "type": canonical.type,
                        "categories": canonical.categories,
                        "roles": canonical.roles,
                        "description": canonical.description,
                        "expected_signal": canonical.expected_signal,
                        "inner_folds": [],
                        "proposed_on_full_outer_train": False,
                        "rationales": [],
                        "expected_signals": [],
                    },
                )
                summary["roles"] = _merge_ordered_values(
                    summary.get("roles"),
                    proposal.roles,
                )
                summary["categories"] = (
                    _merge_ordered_values(summary.get("categories"), proposal.categories)
                    or None
                )
                summary["description"] = _merge_text_values(
                    summary.get("description"),
                    proposal.description,
                )
                summary["expected_signal"] = _merge_text_values(
                    summary.get("expected_signal"),
                    proposal.expected_signal,
                )
                if proposal.rationale:
                    summary["rationales"].append(
                        {
                            "scope": scope,
                            "inner_fold": inner_fold,
                            "text": proposal.rationale,
                        }
                    )
                if proposal.expected_signal:
                    summary["expected_signals"].append(str(proposal.expected_signal))
                if scope == "inner_train" and inner_fold is not None:
                    if int(inner_fold) not in summary["inner_folds"]:
                        summary["inner_folds"].append(int(inner_fold))
                elif scope == "full_outer_train":
                    summary["proposed_on_full_outer_train"] = True

        summaries = []
        for name in sorted(summary_by_name):
            summary = summary_by_name[name]
            support_count = len(summary["inner_folds"])
            support_fraction = (
                float(support_count / inner_fold_count)
                if inner_fold_count > 0
                else None
            )
            summary["inner_folds"] = sorted(summary["inner_folds"])
            summary["inner_support_count"] = int(support_count)
            summary["inner_support_fraction"] = support_fraction
            summary["passes_consistency_gate"] = bool(
                support_count >= threshold
                or (inner_fold_count == 0 and summary["proposed_on_full_outer_train"])
            )
            summary["rationales"] = summary["rationales"][:5]
            summary["expected_signals"] = list(
                dict.fromkeys(summary["expected_signals"])
            )[:5]
            summaries.append(summary)
        return summaries, threshold, inner_fold_count

    def _build_consistency_context(
        self,
        *,
        outer_fold: int,
        candidate_summaries: List[Dict[str, Any]],
        threshold: int,
        inner_fold_count: int,
    ) -> Dict[str, Any]:
        recovery_limit = int(
            self.nn_config.candidate_consistency_recovery_max_candidates
        )
        below_threshold = [
            item
            for item in _rank_consistency_summaries(candidate_summaries)
            if not item.get("passes_consistency_gate")
        ][:recovery_limit]
        passed = [
            item
            for item in _rank_consistency_summaries(candidate_summaries)
            if item.get("passes_consistency_gate")
        ]
        return {
            "prompt_version": "multi_model_agentic_consistency_v1",
            "outer_fold": int(outer_fold),
            "max_selected_candidates": int(self.nn_config.candidate_proposals_per_fold),
            "inner_fold_count": int(inner_fold_count),
            "min_support_folds": int(threshold),
            "min_support_fraction": float(
                self.nn_config.candidate_consistency_min_fold_fraction
            ),
            "selection_policy": [
                "Keep candidates that pass the inner-fold support gate unless they are redundant or likely leakage.",
                "Recover below-threshold candidates only when full outer-train evidence is strong or fold absence appears unstable rather than absent.",
                "Do not invent variables outside candidate_summaries.",
            ],
            "candidate_summaries": passed + below_threshold,
        }

    def _select_consistent_proposals(
        self,
        *,
        context: Dict[str, Any],
        candidate_summaries: Sequence[Dict[str, Any]],
        canonical_proposals: Dict[str, AgenticFeatureProposal],
    ) -> Tuple[List[AgenticFeatureProposal], Dict[str, Any]]:
        del context
        selected = _fallback_consistency_proposals(
            candidate_summaries,
            canonical_proposals,
        )
        max_selected = int(self.nn_config.candidate_proposals_per_fold)
        capped = selected[:max_selected]
        selection_method = (
            "deterministic_consistency_gate"
            if any(
                item.get("passes_consistency_gate")
                for item in candidate_summaries
                if item.get("name") in {proposal.name for proposal in capped}
            )
            else "deterministic_full_outer_train_fallback"
        )
        return capped, {
            "selection_method": selection_method,
            "agent_selection_used": False,
            "max_selected_candidates": max_selected,
            "valid_proposals": [_proposal_to_dict(p) for p in capped],
            "rejected_proposals": [],
            "used_fallback": False,
        }

    def _selected_specs_from_proposals(
        self,
        proposals: Sequence[AgenticFeatureProposal],
    ) -> List[ExplicitFeatureSpec]:
        return _dedupe_specs(
            [
                *self._initial_specs(),
                *[
                    ExplicitFeatureSpec(
                        name=proposal.name,
                        type=proposal.type or "continuous",
                        categories=proposal.categories,
                        roles=proposal.roles,
                        description=proposal.description,
                    )
                    for proposal in proposals
                    if proposal.action == "add"
                ],
            ]
        )

    def _resolve_proposal_aliases(
        self,
        *,
        outer_fold: int,
        proposals: List[AgenticFeatureProposal],
    ) -> Tuple[List[AgenticFeatureProposal], Dict[str, Any]]:
        add_proposals = [proposal for proposal in proposals if proposal.action == "add"]
        known_specs = _dedupe_specs(self.alias_reference_specs)
        if not add_proposals:
            return proposals, {"skipped": "no_valid_additions"}
        if len(add_proposals) < 2 and not known_specs:
            return proposals, {"skipped": "fewer_than_two_additions_and_no_known_features"}

        context = {
            "prompt_version": "multi_model_agentic_alias_resolution_v1",
            "outer_fold": int(outer_fold),
            "known_canonical_features": [_spec_to_dict(spec) for spec in known_specs],
            "proposed_features": [
                {
                    "name": proposal.name,
                    "type": proposal.type,
                    "categories": proposal.categories,
                    "roles": proposal.roles,
                    "description": proposal.description,
                    "rationale": proposal.rationale,
                    "expected_signal": proposal.expected_signal,
                }
                for proposal in add_proposals
            ],
        }

        try:
            response = self.proposal_agent.propose(context)
            alias_trace = _get_agent_response_trace(self.proposal_agent)
        except Exception as exc:
            logger.warning(
                "Multi-model alias resolution failed; using unmerged proposal names",
                exc_info=True,
            )
            return proposals, {"error": str(exc), "applied_aliases": []}

        resolved, applied_aliases = apply_agentic_alias_resolution(
            proposals=proposals,
            known_specs=known_specs,
            response=response,
        )
        result: Dict[str, Any] = {
            "response": response,
            "applied_aliases": applied_aliases,
        }
        if self.search_config.save_agent_raw_output:
            result["agent_raw_output"] = alias_trace
        return resolved, result

    def _harmonize_value_contracts(
        self,
        *,
        outer_fold: int,
        selected_specs: List[ExplicitFeatureSpec],
    ) -> Tuple[List[ExplicitFeatureSpec], Dict[str, Any]]:
        if not selected_specs:
            return selected_specs, {"skipped": "no_selected_features"}

        context = {
            "prompt_version": "multi_model_agentic_value_harmonization_v1",
            "outer_fold": int(outer_fold),
            "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
            "missing_value_policy": (
                "Use null for unknown, not reported, not assessed, not tested, "
                "unavailable, and qualitative-only values that are incompatible "
                "with a numeric extraction target."
            ),
        }
        try:
            response = self.proposal_agent.propose(context)
            harmonization_trace = _get_agent_response_trace(self.proposal_agent)
        except Exception as exc:
            logger.warning(
                "Multi-model value harmonization failed; using unharmonized specs",
                exc_info=True,
            )
            return selected_specs, {"error": str(exc), "applied": []}

        harmonized, applied = apply_agentic_value_harmonization(
            specs=selected_specs,
            response=response,
        )
        result: Dict[str, Any] = {
            "response": response,
            "applied": applied,
        }
        if self.search_config.save_agent_raw_output:
            result["agent_raw_output"] = harmonization_trace
        return harmonized, result

    def _remember_alias_reference_specs(
        self,
        selected_specs: Sequence[ExplicitFeatureSpec],
    ) -> None:
        initial_names = {initial.name for initial in self._initial_specs()}
        self.alias_reference_specs = _dedupe_specs(
            [
                *self.alias_reference_specs,
                *[
                    spec
                    for spec in selected_specs
                    if spec.name not in initial_names
                ],
            ]
        )

    def _filter_specs_by_extraction_coverage(
        self,
        train_df: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
    ) -> List[ExplicitFeatureSpec]:
        initial_names = {spec.name for spec in self._initial_specs()}
        kept: List[ExplicitFeatureSpec] = []
        dropped: List[Dict[str, Any]] = []
        min_coverage = float(getattr(self.search_config, "min_feature_coverage", 0.0))
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if value_col not in train_df.columns:
                coverage = 0.0
            elif missing_col in train_df.columns:
                coverage = float(1.0 - train_df[missing_col].astype(bool).mean())
            else:
                coverage = float(train_df[value_col].notna().mean())
            if spec.name in initial_names or coverage >= min_coverage:
                kept.append(spec)
            else:
                dropped.append({"name": spec.name, "coverage": coverage})
        if dropped:
            logger.info("Dropped low-coverage multi-model agentic features: %s", dropped)
            self.agent_rows.append({"event": "coverage_filter", "dropped": dropped})
        return kept

    def _review_extracted_features_if_needed(
        self,
        *,
        outer_fold: int,
        train_idx: np.ndarray,
        selected_specs: List[ExplicitFeatureSpec],
        bow_result: Dict[str, Any],
        embedding_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not bool(getattr(self.nn_config, "extracted_feature_review_enabled", True)):
            return {
                "selected_specs": selected_specs,
                "summary": {
                    "enabled": False,
                    "review_passed": None,
                    "review_rounds": 0,
                },
            }

        max_rounds = int(getattr(self.nn_config, "extracted_feature_review_max_rounds", 3))
        if max_rounds <= 0:
            return {
                "selected_specs": selected_specs,
                "summary": {
                    "enabled": True,
                    "review_passed": None,
                    "review_rounds": 0,
                    "skipped": "max_rounds_zero",
                },
            }

        required_names = {spec.name for spec in self._initial_specs()}
        current_specs = list(selected_specs)
        best_specs = list(current_specs)
        best_diagnostic: Optional[Dict[str, Any]] = None
        best_score: Optional[Tuple[int, float, float, float]] = None
        final_status = "max_rounds_reached"
        final_passed = False

        for round_index in range(max_rounds + 1):
            train_df = self.dataset.iloc[train_idx].copy()
            current_specs = self._filter_specs_by_extraction_coverage(
                train_df,
                current_specs,
            )
            diagnostic = _evaluate_extracted_feature_set_diagnostic(
                train_df=train_df,
                specs=current_specs,
                config=self.config,
                nn_config=self.nn_config,
                bow_metrics=bow_result.get("metrics", {}),
                embedding_evidence=embedding_evidence,
                random_state=71_000 + 100 * int(outer_fold) + int(round_index),
            )
            benchmark = diagnostic.get("benchmark", {})
            gate = _extracted_feature_review_gate(
                diagnostic=diagnostic,
                nn_config=self.nn_config,
            )
            diagnostic["outer_fold"] = int(outer_fold)
            diagnostic["round"] = int(round_index)
            diagnostic["selected_features"] = [
                _spec_to_dict(spec) for spec in current_specs
            ]
            diagnostic["gate"] = gate
            self.extracted_feature_diagnostic_rows.append(
                _redact_review_artifact(diagnostic, self.search_config)
            )

            score = _extracted_review_selection_score(diagnostic, gate)
            if best_score is None or score < best_score:
                best_score = score
                best_specs = list(current_specs)
                best_diagnostic = diagnostic

            if gate.get("passed"):
                final_status = "passed"
                final_passed = True
                best_specs = list(current_specs)
                best_diagnostic = diagnostic
                break

            if round_index >= max_rounds:
                break

            context = self._build_extracted_feature_review_context(
                outer_fold=outer_fold,
                round_index=round_index,
                current_specs=current_specs,
                diagnostic=diagnostic,
                gate=gate,
                benchmark=benchmark,
                bow_context=bow_result["context"],
                embedding_evidence=embedding_evidence,
                required_names=required_names,
            )
            try:
                raw_proposals = self.proposal_agent.propose(context)
                review_agent_trace = _get_agent_response_trace(self.proposal_agent)
                proposals, rejected = validate_agentic_proposals(
                    raw_proposals,
                    current_specs=current_specs,
                    search_config=self.search_config,
                    allow_removals=True,
                    max_additions=self.nn_config.candidate_proposals_per_fold,
                )
            except Exception as exc:
                logger.warning(
                    "Multi-model extracted-feature review agent failed; "
                    "using best available feature set",
                    exc_info=True,
                )
                self.agent_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "event": "extracted_feature_review",
                        "round": int(round_index),
                        "error": str(exc),
                    }
                )
                final_status = "agent_error"
                break

            proposals, protected_rejections = _protect_required_feature_proposals(
                proposals,
                required_names,
            )
            rejected.extend(protected_rejections)
            proposals, alias_resolution = self._resolve_proposal_aliases(
                outer_fold=outer_fold,
                proposals=proposals,
            )
            revised_specs = _dedupe_specs(apply_proposals(current_specs, proposals))
            revised_specs, value_harmonization = self._harmonize_value_contracts(
                outer_fold=outer_fold,
                selected_specs=revised_specs,
            )
            self._remember_alias_reference_specs(revised_specs)

            review_row: Dict[str, Any] = {
                "outer_fold": int(outer_fold),
                "event": "extracted_feature_review",
                "round": int(round_index),
                "raw_proposals": raw_proposals,
                "valid_proposals": [_proposal_to_dict(proposal) for proposal in proposals],
                "rejected_proposals": rejected,
                "alias_resolution": alias_resolution,
                "value_harmonization": value_harmonization,
                "selected_features_before": [
                    _spec_to_dict(spec) for spec in current_specs
                ],
                "selected_features_after": [
                    _spec_to_dict(spec) for spec in revised_specs
                ],
                "gate": gate,
            }
            if self.search_config.save_agent_context:
                review_row["context"] = context
            if self.search_config.save_agent_raw_output:
                review_row["agent_raw_output"] = review_agent_trace
            self.agent_rows.append(review_row)

            if not _spec_sets_differ(current_specs, revised_specs):
                final_status = "no_review_changes"
                break

            current_specs = revised_specs
            self.dataset = self.extraction_provider.ensure_features(
                self.dataset,
                current_specs,
            )

        selected = best_specs
        if best_diagnostic is not None and not final_passed:
            selected = best_specs
        summary = _extracted_review_summary(
            diagnostic=best_diagnostic,
            status=final_status,
            passed=final_passed,
            rounds=len(
                [
                    row
                    for row in self.extracted_feature_diagnostic_rows
                    if row.get("outer_fold") == int(outer_fold)
                ]
            ),
        )
        return {"selected_specs": selected, "summary": summary}

    def _run_mandatory_parsimony_review(
        self,
        *,
        outer_fold: int,
        train_idx: np.ndarray,
        selected_specs: List[ExplicitFeatureSpec],
        bow_result: Dict[str, Any],
        embedding_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the mandatory parsimony gate before final forest fitting.

        The gate must always execute and write artifacts. It may validly retain
        all features when redundancy/ablation evidence does not justify pruning.
        """
        before_specs = list(selected_specs)
        current_specs = list(selected_specs)
        train_df = self.dataset.iloc[train_idx].copy()
        required_names = {spec.name for spec in self._initial_specs()}
        max_ablations = int(
            getattr(self.nn_config, "parsimony_review_max_single_feature_ablations", 30)
        )
        redundancy = _feature_redundancy_review(
            train_df=train_df,
            specs=current_specs,
            corr_threshold=float(
                getattr(self.nn_config, "parsimony_review_corr_threshold", 0.75)
            ),
        )
        base_diagnostic = _evaluate_extracted_feature_set_diagnostic(
            train_df=train_df,
            specs=current_specs,
            config=self.config,
            nn_config=self.nn_config,
            bow_metrics=bow_result.get("metrics", {}),
            embedding_evidence=embedding_evidence,
            random_state=91_000 + 100 * int(outer_fold),
        )
        base_gate = _extracted_feature_review_gate(
            diagnostic=base_diagnostic,
            nn_config=self.nn_config,
        )
        base_diagnostic["gate"] = base_gate

        ablations: List[Dict[str, Any]] = []
        removed: List[str] = []
        stop_reason = "no_prunable_feature_improved_or_preserved_metrics"
        n_ablations = 0
        if not current_specs:
            stop_reason = "no_selected_features"
        elif max_ablations <= 0:
            stop_reason = "max_single_feature_ablations_zero"
        else:
            while n_ablations < max_ablations and len(current_specs) > 1:
                removable = [
                    spec for spec in current_specs if spec.name not in required_names
                ]
                if not removable:
                    stop_reason = "all_remaining_features_are_required"
                    break

                best_removal: Optional[Dict[str, Any]] = None
                for spec in removable:
                    if n_ablations >= max_ablations:
                        stop_reason = "max_single_feature_ablations_reached"
                        break
                    trial_specs = [s for s in current_specs if s.name != spec.name]
                    role_guard = _parsimony_role_guard(current_specs, trial_specs)
                    if role_guard is not None:
                        ablations.append(
                            {
                                "feature": spec.name,
                                "allowed": False,
                                "reasons": [role_guard],
                                "n_features_after": int(len(trial_specs)),
                            }
                        )
                        n_ablations += 1
                        continue
                    trial_diagnostic = _evaluate_extracted_feature_set_diagnostic(
                        train_df=train_df,
                        specs=trial_specs,
                        config=self.config,
                        nn_config=self.nn_config,
                        bow_metrics=bow_result.get("metrics", {}),
                        embedding_evidence=embedding_evidence,
                        random_state=91_000
                        + 100 * int(outer_fold)
                        + 17 * (n_ablations + 1),
                    )
                    trial_gate = _extracted_feature_review_gate(
                        diagnostic=trial_diagnostic,
                        nn_config=self.nn_config,
                    )
                    trial_diagnostic["gate"] = trial_gate
                    allowed, reasons, deltas = _parsimony_removal_decision(
                        base_diagnostic=base_diagnostic,
                        trial_diagnostic=trial_diagnostic,
                        base_gate=base_gate,
                        trial_gate=trial_gate,
                        nn_config=self.nn_config,
                    )
                    ablation_row = {
                        "feature": spec.name,
                        "allowed": bool(allowed),
                        "reasons": reasons,
                        "n_features_after": int(len(trial_specs)),
                        "metric_deltas": deltas,
                        "metrics_after": _parsimony_metric_snapshot(trial_diagnostic),
                        "gate_after": trial_gate,
                    }
                    ablations.append(ablation_row)
                    n_ablations += 1
                    if allowed:
                        score = _extracted_review_selection_score(
                            trial_diagnostic,
                            trial_gate,
                        )
                        candidate = {
                            "feature": spec.name,
                            "trial_specs": trial_specs,
                            "diagnostic": trial_diagnostic,
                            "gate": trial_gate,
                            "score": score,
                        }
                        if best_removal is None or score < best_removal["score"]:
                            best_removal = candidate

                if best_removal is None:
                    break
                removed.append(str(best_removal["feature"]))
                current_specs = list(best_removal["trial_specs"])
                base_diagnostic = best_removal["diagnostic"]
                base_gate = best_removal["gate"]
                stop_reason = "greedy_removal_completed"

        decision = "prune" if removed else "retain_all"
        summary = {
            "enabled": True,
            "mandatory": True,
            "decision": decision,
            "stop_reason": stop_reason,
            "n_features_before": int(len(before_specs)),
            "n_features_after": int(len(current_specs)),
            "n_removed": int(len(removed)),
            "removed_features": removed,
            "n_single_feature_ablations": int(n_ablations),
            **_prefix_metrics("final_", _parsimony_metric_snapshot(base_diagnostic)),
        }
        review_row = {
            "outer_fold": int(outer_fold),
            "event": "mandatory_parsimony_review",
            "decision": decision,
            "stop_reason": stop_reason,
            "required_features": sorted(required_names),
            "selected_features_before": [_spec_to_dict(spec) for spec in before_specs],
            "selected_features_after": [_spec_to_dict(spec) for spec in current_specs],
            "base_metrics": _parsimony_metric_snapshot(base_diagnostic),
            "base_gate": base_gate,
            "redundancy_review": redundancy,
            "ablations": ablations,
            "summary": summary,
        }
        self.parsimony_review_rows.append(review_row)
        self.agent_rows.append(
            {
                "outer_fold": int(outer_fold),
                "event": "mandatory_parsimony_review",
                "decision": decision,
                "stop_reason": stop_reason,
                "n_features_before": int(len(before_specs)),
                "n_features_after": int(len(current_specs)),
                "removed_features": removed,
                "artifact": "parsimony_review_by_fold.jsonl",
            }
        )
        return {"selected_specs": current_specs, "summary": summary}

    def _build_extracted_feature_review_context(
        self,
        *,
        outer_fold: int,
        round_index: int,
        current_specs: Sequence[ExplicitFeatureSpec],
        diagnostic: Dict[str, Any],
        gate: Dict[str, Any],
        benchmark: Dict[str, Any],
        bow_context: Dict[str, Any],
        embedding_evidence: Dict[str, Any],
        required_names: set,
    ) -> Dict[str, Any]:
        context = {
            "prompt_version": "multi_model_agentic_extracted_feature_review_v1",
            "outer_fold": int(outer_fold),
            "review_round": int(round_index),
            "max_proposals": int(self.nn_config.candidate_proposals_per_fold),
            "clinical_question": self.config.clinical_question,
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "required_features": [
                _spec_to_dict(spec)
                for spec in current_specs
                if spec.name in required_names
            ],
            "current_features": [_spec_to_dict(spec) for spec in current_specs],
            "extraction_summary": diagnostic.get("extraction_summary", []),
            "extracted_feature_diagnostics": _agent_visible_metrics(
                diagnostic.get("metrics", {})
            ),
            "benchmarks": benchmark,
            "failed_criteria": gate.get("failed_criteria", []),
            "review_policy": {
                "auc_margin": float(
                    getattr(self.nn_config, "extracted_feature_review_auc_margin", 0.02)
                ),
                "loss_relative_margin": float(
                    getattr(
                        self.nn_config,
                        "extracted_feature_review_loss_relative_margin",
                        0.05,
                    )
                ),
                "min_benchmark_auc": float(
                    getattr(
                        self.nn_config,
                        "extracted_feature_review_min_benchmark_auc",
                        0.55,
                    )
                ),
            },
            "original_bow_context": {
                "model_diagnostics": bow_context.get("model_diagnostics"),
                "feature_importance": bow_context.get("feature_importance"),
            },
            "response_contract": {
                "proposals": [
                    {
                        "action": "add|remove|update_role|none",
                        "name": "snake_case_variable_name",
                        "type": "categorical|continuous",
                        "categories": ["category_a", "category_b"],
                        "roles": ["confounder", "effect_modifier"],
                        "description": "exact pre-treatment extraction target",
                        "rationale": "why this change addresses the diagnostic failure",
                        "expected_signal": "treatment, outcome, or pseudo-target signal expected",
                    }
                ]
            },
        }
        if embedding_evidence:
            context["embedding_contrast_evidence"] = (
                embedding_evidence
                if self.search_config.save_agent_context
                else redact_embedding_contrast_evidence(embedding_evidence)
            )
        return _compact_extracted_feature_review_context(context)

    def _ensure_prespecified_features(self) -> None:
        specs = self._initial_specs()
        if not specs:
            return
        self.dataset = self.extraction_provider.ensure_features(self.dataset, specs)

    def _initial_specs(self) -> List[ExplicitFeatureSpec]:
        specs: List[ExplicitFeatureSpec] = []
        if getattr(self.config.explicit_features, "features", None):
            specs.extend(list(self.config.explicit_features.features))
        specs.extend(list(getattr(self.nn_config, "prespecified_features", []) or []))
        specs.extend(list(getattr(self.nn_config, "prespecified_confounders", []) or []))
        specs.extend(
            list(getattr(self.nn_config, "prespecified_effect_modifiers", []) or [])
        )
        json_path = getattr(self.nn_config, "prespecified_features_json", None)
        if json_path:
            specs.extend(load_explicit_feature_specs_json(str(json_path)))
        return _dedupe_specs(specs)

    def _vectorizer_params(self, view: BoWViewConfig) -> Dict[str, Any]:
        return {
            "ngram_range_min": int(view.ngram_range_min),
            "ngram_range_max": int(view.ngram_range_max),
            "min_df": int(view.min_df),
            "max_df": float(view.max_df),
            "sublinear_tf": bool(view.sublinear_tf),
            "max_features": int(view.max_features),
        }

    def _model_params(self, view: BoWViewConfig) -> Dict[str, Any]:
        return {
            "bow_model": str(view.bow_model).strip().lower(),
            "logistic_c": float(view.logistic_c),
            "logistic_max_iter": int(view.logistic_max_iter),
            "ridge_alpha": float(view.ridge_alpha),
        }

    def _make_vectorizer(self, view: BoWViewConfig) -> TfidfVectorizer:
        return _make_bow_vectorizer(self._vectorizer_params(view))

    def _make_classifier(self, view: BoWViewConfig, random_state: int = 17):
        return _make_bow_classifier(self._model_params(view), random_state=random_state)

    def _make_regressor(self, view: BoWViewConfig, random_state: int = 17):
        return _make_bow_regressor(self._model_params(view), random_state=random_state)

    def _make_logistic_regression(
        self,
        view: BoWViewConfig,
        random_state: int = 17,
    ) -> LogisticRegression:
        return LogisticRegression(
            C=float(view.logistic_c),
            solver="liblinear",
            max_iter=int(view.logistic_max_iter),
            random_state=random_state,
        )

    def _make_ridge(self, view: BoWViewConfig) -> Ridge:
        return Ridge(alpha=float(view.ridge_alpha), random_state=17)

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

    def _candidate_consistency_n_jobs(self, folds: int) -> int:
        return self._parallel_n_jobs(
            self.nn_config.candidate_consistency_parallelism,
            folds,
            auto_workers=self.num_workers,
        )

    def _inner_workers_for_outer_job(self, outer_n_jobs: int) -> int:
        if str(self.nn_config.fold_parallelism).strip().lower() != "auto":
            return self.num_workers
        return max(1, int(self.num_workers) // max(1, int(outer_n_jobs)))

    def _inner_workers_for_nested_job(self, n_jobs: int) -> int:
        if str(self.nn_config.fold_parallelism).strip().lower() != "auto":
            return self.num_workers
        return max(1, int(self.num_workers) // max(1, int(n_jobs)))

    def _fold_n_jobs(self, folds: int) -> int:
        return self._parallel_n_jobs(
            self.nn_config.fold_parallelism,
            folds,
            auto_workers=self.num_workers,
        )

    def _feature_importance_n_jobs(self) -> int:
        return self._parallel_n_jobs(
            self.nn_config.fold_parallelism,
            3,
            auto_workers=self.num_workers,
        )

    def _parallel_backend_name(self) -> str:
        return (
            "loky"
            if self.nn_config.bow_parallel_backend == "processes"
            else "threading"
        )

    def _run_fold_tasks(self, run_fold: Any, split_items: Sequence[Any]) -> List[Any]:
        n_jobs = self._fold_n_jobs(len(split_items))
        if n_jobs <= 1:
            return [
                run_fold(int(fold), np.asarray(fit_pos), np.asarray(heldout_pos))
                for fold, (fit_pos, heldout_pos) in split_items
            ]
        backend = self._parallel_backend_name()
        logger.info(
            "Multi-model BoW cross-fit parallelism: folds=%s n_jobs=%s "
            "setting=%s backend=%s joblib_backend=%s",
            len(split_items),
            n_jobs,
            self.nn_config.fold_parallelism,
            self.nn_config.bow_parallel_backend,
            backend,
        )
        return Parallel(
            n_jobs=n_jobs,
            backend=backend,
            batch_size=1,
            pre_dispatch="all",
        )(
            delayed(run_fold)(int(fold), np.asarray(fit_pos), np.asarray(heldout_pos))
            for fold, (fit_pos, heldout_pos) in split_items
        )

    def _save_predictions(self, results_df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(self.output_path, index=False)
        logger.info("Multi-model agentic forest predictions saved to: %s", self.output_path)

    def _save_artifacts(self) -> None:
        if self.bow_prediction_frames:
            pd.concat(self.bow_prediction_frames).to_parquet(
                self.artifact_dir / "bow_view_oof_predictions.parquet",
                index=False,
            )
        text_prediction_frames = list(self.bow_prediction_frames)
        if self.htr_nuisance_prediction_frames:
            htr_nuisance = pd.concat(self.htr_nuisance_prediction_frames)
            htr_nuisance.to_parquet(
                self.artifact_dir / "htr_nuisance_oof_predictions.parquet",
                index=False,
            )
            text_prediction_frames.append(htr_nuisance)
        if self.htr_effect_prediction_frames:
            htr_effect = pd.concat(self.htr_effect_prediction_frames)
            htr_effect.to_parquet(
                self.artifact_dir / "htr_effect_oof_predictions.parquet",
                index=False,
            )
            text_prediction_frames.append(htr_effect)
        if text_prediction_frames:
            pd.concat(text_prediction_frames, ignore_index=True).to_parquet(
                self.artifact_dir / "text_model_oof_predictions.parquet",
                index=False,
            )
        if self.htr_attention_rows:
            pd.DataFrame(self.htr_attention_rows).to_parquet(
                self.artifact_dir / "htr_attention_evidence.parquet",
                index=False,
            )
        pd.DataFrame(self.outer_metric_rows).to_csv(
            self.artifact_dir / "outer_cv_metrics.csv",
            index=False,
        )
        _write_jsonl(
            self.artifact_dir / "bow_view_feature_importance_by_fold.jsonl",
            self.importance_rows,
        )
        if self.embedding_evidence_rows:
            _write_jsonl(
                self.artifact_dir / "embedding_contrast_evidence_by_fold.jsonl",
                self.embedding_evidence_rows,
            )
        if self.extracted_feature_diagnostic_rows:
            _write_jsonl(
                self.artifact_dir / "extracted_feature_diagnostics_by_fold.jsonl",
                self.extracted_feature_diagnostic_rows,
            )
        if self.parsimony_review_rows:
            _write_jsonl(
                self.artifact_dir / "parsimony_review_by_fold.jsonl",
                self.parsimony_review_rows,
            )
        _write_jsonl(self.artifact_dir / "agent_candidate_proposals.jsonl", self.agent_rows)
        with open(self.artifact_dir / "selected_feature_sets.json", "w") as f:
            json.dump(self.feature_set_rows, f, indent=2, default=_json_default)
        logger.info("Multi-model agentic forest artifacts saved to: %s", self.artifact_dir)


def _run_multi_model_outer_fold_worker(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    artifact_dir: Path,
    outer_fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    num_workers: int,
) -> Dict[str, Any]:
    logger.info(
        "Multi-model agentic isolated fold %s: train=%s test=%s workers=%s",
        outer_fold,
        len(train_idx),
        len(test_idx),
        num_workers,
    )
    fold_runner = MultiModelAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=(
            Path(artifact_dir)
            / f"outer_fold_{int(outer_fold):03d}"
            / "predictions.parquet"
        ),
        num_workers=num_workers,
    )
    predictions = fold_runner._run_one_analysis_split(
        outer_fold=outer_fold,
        train_idx=train_idx,
        test_idx=test_idx,
    )
    return {
        "outer_fold": int(outer_fold),
        "predictions": predictions,
        "bow_prediction_frames": fold_runner.bow_prediction_frames,
        "htr_nuisance_prediction_frames": fold_runner.htr_nuisance_prediction_frames,
        "htr_effect_prediction_frames": fold_runner.htr_effect_prediction_frames,
        "htr_attention_rows": fold_runner.htr_attention_rows,
        "importance_rows": fold_runner.importance_rows,
        "embedding_evidence_rows": fold_runner.embedding_evidence_rows,
        "agent_rows": fold_runner.agent_rows,
        "extracted_feature_diagnostic_rows": (
            fold_runner.extracted_feature_diagnostic_rows
        ),
        "parsimony_review_rows": fold_runner.parsimony_review_rows,
        "feature_set_rows": fold_runner.feature_set_rows,
        "outer_metric_rows": fold_runner.outer_metric_rows,
    }


def _build_multi_model_inner_candidate_bundle_worker(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    artifact_dir: Path,
    outer_fold: int,
    discovery_df: pd.DataFrame,
    inner_fold: int,
    fit_pos: np.ndarray,
    heldout_pos: np.ndarray,
    total_inner_folds: int,
    num_workers: int,
) -> Dict[str, Any]:
    worker = MultiModelAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=(
            Path(artifact_dir)
            / f"outer_{int(outer_fold):03d}_candidate_inner_{int(inner_fold):03d}"
            / "predictions.parquet"
        ),
        num_workers=num_workers,
    )
    return worker._build_inner_consistency_candidate_bundle(
        outer_fold=outer_fold,
        discovery_df=discovery_df,
        inner_fold=inner_fold,
        fit_pos=fit_pos,
        heldout_pos=heldout_pos,
        total_inner_folds=total_inner_folds,
    )


def _normalize_texts(values: Sequence[Any]) -> List[str]:
    return [_normalize_text(value) for value in values]


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).translate(_DASH_TRANSLATION)
    text = text.replace("\u2265", ">=").replace("\u2264", "<=")
    return text.lower()


def _format_selected_feature_roles(specs: Sequence[ExplicitFeatureSpec]) -> str:
    return ",".join(
        f"{spec.name}[{'+'.join(_ordered_roles(spec.roles))}]"
        for spec in specs
    )


def _ordered_roles(roles: Sequence[str]) -> List[str]:
    role_set = {str(role) for role in roles}
    ordered = [
        role for role in ("confounder", "effect_modifier") if role in role_set
    ]
    ordered.extend(sorted(role_set.difference(ordered)))
    return ordered or ["unspecified"]


def _candidate_consistency_threshold(
    fold_count: int,
    *,
    min_folds: int,
    min_fold_fraction: float,
) -> int:
    if fold_count <= 0:
        return 1
    return min(
        int(fold_count),
        max(
            1,
            int(min_folds),
            int(np.ceil(float(min_fold_fraction) * int(fold_count))),
        ),
    )


def _resolve_alias_name(name: str, alias_map: Dict[str, str]) -> str:
    current = str(name)
    seen = set()
    while current in alias_map and current not in seen:
        seen.add(current)
        current = alias_map[current]
    return current


def _merge_duplicate_proposals(
    proposals: Sequence[AgenticFeatureProposal],
) -> List[AgenticFeatureProposal]:
    merged: Dict[str, AgenticFeatureProposal] = {}
    for proposal in proposals:
        if proposal.action != "add":
            continue
        if proposal.name in merged:
            merged[proposal.name] = _merge_proposals(merged[proposal.name], proposal)
        else:
            merged[proposal.name] = proposal
    return list(merged.values())


def _fallback_consistency_proposals(
    candidate_summaries: Sequence[Dict[str, Any]],
    canonical_proposals: Dict[str, AgenticFeatureProposal],
) -> List[AgenticFeatureProposal]:
    selected = [
        canonical_proposals[item["name"]]
        for item in _rank_consistency_summaries(candidate_summaries)
        if item.get("passes_consistency_gate") and item.get("name") in canonical_proposals
    ]
    if selected:
        return selected
    full_supported = [
        canonical_proposals[item["name"]]
        for item in _rank_consistency_summaries(candidate_summaries)
        if item.get("proposed_on_full_outer_train")
        and item.get("name") in canonical_proposals
    ]
    return full_supported[:1]


def _rank_consistency_summaries(
    candidate_summaries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return sorted(
        candidate_summaries,
        key=lambda item: (
            -int(bool(item.get("passes_consistency_gate"))),
            -int(item.get("inner_support_count") or 0),
            -int(bool(item.get("proposed_on_full_outer_train"))),
            str(item.get("name") or ""),
        ),
    )


def _proposal_bundle_artifact(bundle: Dict[str, Any]) -> Dict[str, Any]:
    artifact = {
        key: value
        for key, value in bundle.items()
        if key not in {"valid_proposals"}
    }
    artifact["valid_proposals"] = [
        _proposal_to_dict(proposal)
        for proposal in bundle.get("valid_proposals", [])
    ]
    return artifact


def _proposal_to_dict(proposal: AgenticFeatureProposal) -> Dict[str, Any]:
    return {
        "action": proposal.action,
        "name": proposal.name,
        "type": proposal.type,
        "categories": proposal.categories,
        "roles": proposal.roles,
        "description": proposal.description,
        "rationale": proposal.rationale,
        "expected_signal": proposal.expected_signal,
    }


def _columns_to_feature_dicts(
    df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
) -> Optional[List[Dict[str, Any]]]:
    if not specs:
        return None
    values: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        item: Dict[str, Any] = {}
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            legacy_col = f"explicit_conf_{spec.name}"
            source_col = value_col if value_col in df.columns else legacy_col
            value = row.get(source_col)
            missing_col = f"{source_col}_missing"
            item[spec.name] = value
            item[f"{spec.name}_missing"] = bool(row.get(missing_col, pd.isna(value)))
        values.append(item)
    return values


def _evaluate_extracted_feature_set_diagnostic(
    *,
    train_df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
    config: AppliedInferenceConfig,
    nn_config: MultiModelAgenticForestConfig,
    bow_metrics: Dict[str, Any],
    embedding_evidence: Dict[str, Any],
    random_state: int,
) -> Dict[str, Any]:
    y = train_df[config.outcome_column].to_numpy(dtype=float)
    t = train_df[config.treatment_column].to_numpy(dtype=float)
    specs = list(specs)
    extraction_summary = _summarize_multi_model_extractions(train_df, specs)
    x_full, x_names = _explicit_matrix_full(train_df, specs, role="effect_modifier")
    w_full, w_names = _explicit_matrix_full(train_df, specs, role="confounder")

    status = "ok"
    if not specs:
        status = "no_selected_features"
    elif x_full.shape[1] == 0 and w_full.shape[1] == 0:
        status = "no_usable_feature_columns"

    e_hat = _crossfit_explicit_binary(
        train_df=train_df,
        labels=t,
        specs=specs,
        role="confounder",
        requested_folds=int(nn_config.nuisance_folds),
        random_state=random_state + 11,
    )
    if str(config.outcome_type).lower() == "continuous":
        m_hat = _crossfit_explicit_regression(
            train_df=train_df,
            values=y,
            specs=specs,
            role="confounder",
            requested_folds=int(nn_config.nuisance_folds),
            sample_weight=None,
            random_state=random_state + 23,
        )
    else:
        m_hat = _crossfit_explicit_binary(
            train_df=train_df,
            labels=y,
            specs=specs,
            role="confounder",
            requested_folds=int(nn_config.nuisance_folds),
            random_state=random_state + 23,
        )

    e_clipped = np.clip(e_hat, float(nn_config.e_clip), 1.0 - float(nn_config.e_clip))
    t_resid = t - e_clipped
    y_resid = y - m_hat
    pseudo_target = y_resid / t_resid
    pseudo_weight = np.square(t_resid)
    tau_hat = _crossfit_explicit_regression(
        train_df=train_df,
        values=pseudo_target,
        specs=specs,
        role="effect_modifier",
        requested_folds=int(nn_config.effect_folds),
        sample_weight=pseudo_weight,
        random_state=random_state + 37,
    )
    r_loss = np.square(y_resid - tau_hat * t_resid)
    r_loss_at_zero = np.square(y_resid)

    metrics: Dict[str, Any] = {
        "status": status,
        "n_rows": int(len(train_df)),
        "n_selected_features": int(len(specs)),
        "n_w_features": int(w_full.shape[1]),
        "n_x_features": int(x_full.shape[1]),
        "w_feature_names": w_names,
        "x_feature_names": x_names,
        "treatment_auroc": _safe_roc_auc(t, e_hat),
        "treatment_brier": _safe_brier_score(t, e_hat),
        "treatment_log_loss": _safe_log_loss(t, e_hat),
        "pseudo_target_mean": _finite_or_none(np.mean(pseudo_target)),
        "pseudo_target_std": _finite_or_none(np.std(pseudo_target)),
        "tau_hat_mean": _finite_or_none(np.mean(tau_hat)),
        "tau_hat_std": _finite_or_none(np.std(tau_hat)),
        "tau_hat_pseudo_target_corr": _safe_corr(tau_hat, pseudo_target),
        "r_loss_mean": _finite_or_none(np.mean(r_loss)),
        "r_loss_at_zero_tau_mean": _finite_or_none(np.mean(r_loss_at_zero)),
    }
    zero = metrics["r_loss_at_zero_tau_mean"]
    loss = metrics["r_loss_mean"]
    if zero is not None and zero > 0.0 and loss is not None:
        metrics["r_loss_relative_improvement"] = float(1.0 - loss / zero)
    if str(config.outcome_type).lower() == "continuous":
        metrics["outcome_rmse"] = _finite_or_none(np.sqrt(mean_squared_error(y, m_hat)))
    else:
        metrics["outcome_auroc"] = _safe_roc_auc(y, m_hat)
        metrics["outcome_brier"] = _safe_brier_score(y, m_hat)
        metrics["outcome_log_loss"] = _safe_log_loss(y, m_hat)

    return {
        "metrics": metrics,
        "benchmark": _extracted_feature_review_benchmarks(
            bow_metrics,
            embedding_evidence,
        ),
        "extraction_summary": extraction_summary,
    }


def _crossfit_explicit_binary(
    *,
    train_df: pd.DataFrame,
    labels: np.ndarray,
    specs: Sequence[ExplicitFeatureSpec],
    role: Optional[str],
    requested_folds: int,
    random_state: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=float)
    oof = np.full(len(labels), np.nan, dtype=float)
    if len(labels) == 0:
        return oof
    if len(np.unique(labels.astype(int))) < 2:
        return np.full(len(labels), float(np.nanmean(labels)), dtype=float)
    try:
        split_items = _binary_split_items(
            labels.astype(int),
            requested_folds=requested_folds,
            random_state=random_state,
        )
    except ValueError:
        return np.full(len(labels), float(np.nanmean(labels)), dtype=float)

    for fold, (fit_pos, heldout_pos) in enumerate(split_items, start=1):
        del fold
        fit_pos = np.asarray(fit_pos)
        heldout_pos = np.asarray(heldout_pos)
        fit_y = labels[fit_pos].astype(int)
        if len(np.unique(fit_y)) < 2:
            oof[heldout_pos] = float(np.mean(fit_y))
            continue
        x_fit, x_heldout = _explicit_matrix_split(
            train_df=train_df,
            fit_pos=fit_pos,
            heldout_pos=heldout_pos,
            specs=specs,
            role=role,
        )
        x_fit = _ensure_model_matrix(x_fit)
        x_heldout = _ensure_model_matrix(x_heldout)
        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1000,
            random_state=random_state,
        )
        try:
            model.fit(x_fit, fit_y)
            oof[heldout_pos] = model.predict_proba(x_heldout)[:, 1]
        except ValueError:
            oof[heldout_pos] = float(np.mean(fit_y))
    return _fill_nonfinite_predictions(oof, labels)


def _crossfit_explicit_regression(
    *,
    train_df: pd.DataFrame,
    values: np.ndarray,
    specs: Sequence[ExplicitFeatureSpec],
    role: Optional[str],
    requested_folds: int,
    sample_weight: Optional[np.ndarray],
    random_state: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    oof = np.full(len(values), np.nan, dtype=float)
    if len(values) == 0:
        return oof
    try:
        folds = _bounded_fold_count(requested_folds, len(values))
    except ValueError:
        return np.full(len(values), float(np.nanmean(values)), dtype=float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    for fit_pos, heldout_pos in splitter.split(train_df):
        fit_pos = np.asarray(fit_pos)
        heldout_pos = np.asarray(heldout_pos)
        x_fit, x_heldout = _explicit_matrix_split(
            train_df=train_df,
            fit_pos=fit_pos,
            heldout_pos=heldout_pos,
            specs=specs,
            role=role,
        )
        x_fit = _ensure_model_matrix(x_fit)
        x_heldout = _ensure_model_matrix(x_heldout)
        model = Ridge(alpha=1.0, random_state=random_state)
        fit_weight = None
        if weights is not None and len(weights) == len(values):
            fit_weight = weights[fit_pos]
            fit_weight = np.where(
                np.isfinite(fit_weight) & (fit_weight > 0.0),
                fit_weight,
                0.0,
            )
            if float(np.sum(fit_weight)) <= 0.0:
                fit_weight = None
        finite = np.isfinite(values[fit_pos])
        if np.sum(finite) < 1:
            oof[heldout_pos] = float(np.nanmean(values))
            continue
        _fit_regressor(
            model,
            x_fit[finite],
            values[fit_pos][finite],
            sample_weight=None if fit_weight is None else fit_weight[finite],
        )
        oof[heldout_pos] = model.predict(x_heldout)
    return _fill_nonfinite_predictions(oof, values)


def _explicit_matrix_split(
    *,
    train_df: pd.DataFrame,
    fit_pos: np.ndarray,
    heldout_pos: np.ndarray,
    specs: Sequence[ExplicitFeatureSpec],
    role: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    fit_df = train_df.iloc[np.asarray(fit_pos)]
    heldout_df = train_df.iloc[np.asarray(heldout_pos)]
    fit_dicts = _columns_to_feature_dicts(fit_df, specs) or []
    heldout_dicts = _columns_to_feature_dicts(heldout_df, specs) or []
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    fit_features, _ = get_raw_explicit_features(
        fit_dicts,
        list(specs),
        continuous_means=means,
        continuous_stds=stds,
        role=role,
    )
    heldout_features, _ = get_raw_explicit_features(
        heldout_dicts,
        list(specs),
        continuous_means=means,
        continuous_stds=stds,
        role=role,
    )
    return (
        _as_2d_feature_matrix(fit_features, len(fit_df)),
        _as_2d_feature_matrix(heldout_features, len(heldout_df)),
    )


def _explicit_matrix_full(
    df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
    *,
    role: Optional[str],
) -> Tuple[np.ndarray, List[str]]:
    feature_dicts = _columns_to_feature_dicts(df, specs) or []
    features, names = get_raw_explicit_features(
        feature_dicts,
        list(specs),
        continuous_means={},
        continuous_stds={},
        role=role,
    )
    return _as_2d_feature_matrix(features, len(df)), list(names)


def _as_2d_feature_matrix(values: Sequence[Sequence[float]], n_rows: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        return np.zeros((n_rows, 0), dtype=np.float32)
    if matrix.shape[0] != n_rows:
        return np.zeros((n_rows, 0), dtype=np.float32)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _ensure_model_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 1), dtype=np.float32)
    return matrix


def _fill_nonfinite_predictions(pred: np.ndarray, fallback_values: np.ndarray) -> np.ndarray:
    filled = np.asarray(pred, dtype=float).copy()
    finite = np.isfinite(filled)
    if np.all(finite):
        return filled
    fallback = float(np.nanmean(fallback_values)) if len(fallback_values) else 0.0
    if not np.isfinite(fallback):
        fallback = 0.0
    filled[~finite] = fallback
    return filled


def _safe_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    try:
        return _finite_or_none(
            log_loss(
                np.asarray(y_true, dtype=int),
                np.clip(np.asarray(y_pred, dtype=float), 1e-6, 1.0 - 1e-6),
                labels=[0, 1],
            )
        )
    except ValueError:
        return None


def _safe_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    try:
        return _finite_or_none(
            brier_score_loss(
                np.asarray(y_true, dtype=int),
                np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0),
            )
        )
    except ValueError:
        return None


def _align_htr_prediction_frame(
    frame: Any,
    discovery_df: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    source: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{source} did not return a predictions DataFrame")
    if "_oci_row_id" not in frame.columns:
        raise ValueError(f"{source} predictions must include _oci_row_id")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} predictions missing required columns: {missing}")
    if frame["_oci_row_id"].duplicated().any():
        raise ValueError(f"{source} predictions contain duplicate _oci_row_id values")

    aligned = discovery_df[["_oci_row_id"]].merge(
        frame.copy(),
        on="_oci_row_id",
        how="left",
        sort=False,
    )
    if len(aligned) != len(discovery_df):
        raise ValueError(f"{source} predictions could not be aligned to discovery rows")
    for column in required_columns:
        values = pd.to_numeric(aligned[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{source} predictions contain non-finite {column} values")
        aligned[column] = values
    return aligned


def _htr_nuisance_metrics(
    *,
    discovery_df: pd.DataFrame,
    predictions: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"n_rows": int(len(predictions))}
    e_hat = predictions["e_hat"].to_numpy(dtype=float)
    m_hat = predictions["m_hat"].to_numpy(dtype=float)
    metrics.update(
        {
            "e_hat_mean": _finite_or_none(np.mean(e_hat)),
            "e_hat_std": _finite_or_none(np.std(e_hat)),
            "m_hat_mean": _finite_or_none(np.mean(m_hat)),
            "m_hat_std": _finite_or_none(np.std(m_hat)),
        }
    )
    if treatment_column in discovery_df.columns:
        t = discovery_df[treatment_column].to_numpy(dtype=float)
        metrics.update(
            {
                "treatment_auroc": _safe_roc_auc(t, e_hat),
                "treatment_brier": _safe_brier_score(t, e_hat),
                "treatment_log_loss": _safe_log_loss(t, e_hat),
            }
        )
    if outcome_column in discovery_df.columns:
        y = discovery_df[outcome_column].to_numpy(dtype=float)
        if str(outcome_type).lower() == "continuous":
            metrics["outcome_rmse"] = _finite_or_none(np.sqrt(mean_squared_error(y, m_hat)))
        else:
            metrics.update(
                {
                    "outcome_auroc": _safe_roc_auc(y, m_hat),
                    "outcome_brier": _safe_brier_score(y, m_hat),
                    "outcome_log_loss": _safe_log_loss(y, m_hat),
                }
            )
    for column in ["y_residual", "t_residual", "r_pseudo_outcome"]:
        if column in predictions.columns:
            values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            metrics[f"{column}_mean"] = (
                _finite_or_none(np.mean(finite)) if len(finite) else None
            )
            metrics[f"{column}_std"] = (
                _finite_or_none(np.std(finite)) if len(finite) else None
            )
    return metrics


def _htr_effect_metrics(predictions: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"n_rows": int(len(predictions))}
    for column in [
        "tau_hat_r_stage",
        "tau_logit_modifier",
        "r_pseudo_outcome",
        "r_loss",
        "effect_loss",
        "effect_loss_at_zero_tau",
    ]:
        if column not in predictions.columns:
            continue
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        metrics[f"{column}_mean"] = (
            _finite_or_none(np.mean(finite)) if len(finite) else None
        )
        metrics[f"{column}_std"] = (
            _finite_or_none(np.std(finite)) if len(finite) else None
        )
    loss = metrics.get("r_loss_mean")
    zero = metrics.get("effect_loss_at_zero_tau_mean")
    if zero is not None and zero > 0.0 and loss is not None:
        metrics["r_loss_relative_improvement"] = float(1.0 - loss / zero)
    if "effect_objective" in predictions.columns:
        objectives = sorted(
            {str(value) for value in predictions["effect_objective"].dropna() if str(value)}
        )
        if objectives:
            metrics["effect_objectives"] = objectives
    if "target_source" in predictions.columns:
        target_sources = sorted(
            {str(value) for value in predictions["target_source"].dropna() if str(value)}
        )
        if target_sources:
            metrics["target_sources"] = target_sources
    return metrics


def _summarize_multi_model_extractions(
    df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for spec in specs:
        value_col = f"explicit_feat_{spec.name}"
        missing_col = f"{value_col}_missing"
        if value_col not in df.columns:
            summaries.append(
                {
                    "name": spec.name,
                    "roles": list(spec.roles),
                    "coverage": 0.0,
                    "top_values": {},
                }
            )
            continue
        if missing_col in df.columns:
            missing = df[missing_col].astype(bool)
        else:
            missing = df[value_col].isna()
        observed = df.loc[~missing, value_col]
        summaries.append(
            {
                "name": spec.name,
                "roles": list(spec.roles),
                "coverage": float(1.0 - missing.mean()),
                "n_unique_observed": int(observed.nunique(dropna=True)),
                "top_values": observed.astype(str).value_counts().head(8).to_dict(),
            }
        )
    return summaries


def _feature_redundancy_review(
    *,
    train_df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
    corr_threshold: float,
) -> Dict[str, Any]:
    specs = list(specs)
    continuous_correlations: List[Dict[str, Any]] = []
    categorical_contingency: List[Dict[str, Any]] = []
    missingness_overlap: List[Dict[str, Any]] = []
    for left, right in combinations(specs, 2):
        left_values = _explicit_feature_series(train_df, left)
        right_values = _explicit_feature_series(train_df, right)
        left_missing = _explicit_feature_missing_mask(train_df, left, left_values)
        right_missing = _explicit_feature_missing_mask(train_df, right, right_values)
        missingness_overlap.append(
            {
                "a": left.name,
                "b": right.name,
                "both_missing": float(np.mean(left_missing & right_missing)),
                "either_missing": float(np.mean(left_missing | right_missing)),
            }
        )
        if left.type == "continuous" and right.type == "continuous":
            x = pd.to_numeric(left_values, errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(right_values, errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            corr = None
            if int(np.sum(mask)) >= 3 and np.std(x[mask]) > 0.0 and np.std(y[mask]) > 0.0:
                corr = _finite_or_none(np.corrcoef(x[mask], y[mask])[0, 1])
            if corr is not None and abs(corr) >= float(corr_threshold):
                continuous_correlations.append(
                    {
                        "a": left.name,
                        "b": right.name,
                        "correlation": float(corr),
                        "n_pairwise_complete": int(np.sum(mask)),
                    }
                )
        elif left.type == "categorical" and right.type == "categorical":
            left_cat = left_values.astype("object").where(~left_missing, "__MISSING__")
            right_cat = right_values.astype("object").where(~right_missing, "__MISSING__")
            table = pd.crosstab(left_cat, right_cat, dropna=False)
            total = float(table.to_numpy().sum())
            max_cell_fraction = None if total <= 0 else float(table.to_numpy().max() / total)
            categorical_contingency.append(
                {
                    "a": left.name,
                    "b": right.name,
                    "shape": [int(table.shape[0]), int(table.shape[1])],
                    "max_cell_fraction": max_cell_fraction,
                }
            )
    return {
        "continuous_correlations_abs_ge_threshold": continuous_correlations,
        "categorical_contingency": categorical_contingency,
        "missingness_overlap": missingness_overlap,
        "corr_threshold": float(corr_threshold),
    }


def _explicit_feature_series(df: pd.DataFrame, spec: ExplicitFeatureSpec) -> pd.Series:
    value_col = f"explicit_feat_{spec.name}"
    legacy_col = f"explicit_conf_{spec.name}"
    if value_col in df.columns:
        return df[value_col]
    if legacy_col in df.columns:
        return df[legacy_col]
    return pd.Series([np.nan] * len(df), index=df.index, dtype="object")


def _explicit_feature_missing_mask(
    df: pd.DataFrame,
    spec: ExplicitFeatureSpec,
    values: Optional[pd.Series] = None,
) -> np.ndarray:
    value_col = f"explicit_feat_{spec.name}"
    legacy_col = f"explicit_conf_{spec.name}"
    source_col = value_col if value_col in df.columns else legacy_col
    missing_col = f"{source_col}_missing"
    if missing_col in df.columns:
        return df[missing_col].astype(bool).to_numpy()
    if values is None:
        values = _explicit_feature_series(df, spec)
    return values.isna().to_numpy()


def _parsimony_role_guard(
    current_specs: Sequence[ExplicitFeatureSpec],
    trial_specs: Sequence[ExplicitFeatureSpec],
) -> Optional[str]:
    if not trial_specs:
        return "would_remove_all_features"
    for role in ["confounder", "effect_modifier"]:
        had_role = any(role in spec.roles for spec in current_specs)
        keeps_role = any(role in spec.roles for spec in trial_specs)
        if had_role and not keeps_role:
            return f"would_remove_all_{role}_features"
    return None


def _parsimony_metric_snapshot(diagnostic: Dict[str, Any]) -> Dict[str, Any]:
    metrics = diagnostic.get("metrics", {}) if isinstance(diagnostic, dict) else {}
    keys = [
        "n_selected_features",
        "n_w_features",
        "n_x_features",
        "treatment_auroc",
        "treatment_brier",
        "treatment_log_loss",
        "outcome_auroc",
        "outcome_brier",
        "outcome_log_loss",
        "outcome_rmse",
        "r_loss_mean",
        "r_loss_relative_improvement",
        "tau_hat_pseudo_target_corr",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def _parsimony_removal_decision(
    *,
    base_diagnostic: Dict[str, Any],
    trial_diagnostic: Dict[str, Any],
    base_gate: Dict[str, Any],
    trial_gate: Dict[str, Any],
    nn_config: MultiModelAgenticForestConfig,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    base_metrics = base_diagnostic.get("metrics", {})
    trial_metrics = trial_diagnostic.get("metrics", {})
    auc_tolerance = float(getattr(nn_config, "parsimony_review_auc_tolerance", 0.01))
    loss_tolerance = float(
        getattr(nn_config, "parsimony_review_loss_relative_tolerance", 0.03)
    )
    reasons: List[str] = []
    base_failures = int(base_gate.get("n_failed_criteria", 0) or 0)
    trial_failures = int(trial_gate.get("n_failed_criteria", 0) or 0)
    if trial_failures > base_failures:
        reasons.append("review_gate_would_worsen")
    deltas: Dict[str, Any] = {}
    for metric in ["treatment_auroc", "outcome_auroc"]:
        base_value = _finite_or_none(base_metrics.get(metric))
        trial_value = _finite_or_none(trial_metrics.get(metric))
        if base_value is None or trial_value is None:
            continue
        delta = float(trial_value - base_value)
        deltas[metric] = delta
        if delta < -auc_tolerance:
            reasons.append(f"{metric}_drop_exceeds_tolerance")
    for metric in ["treatment_log_loss", "outcome_log_loss", "outcome_rmse", "r_loss_mean"]:
        base_value = _finite_or_none(base_metrics.get(metric))
        trial_value = _finite_or_none(trial_metrics.get(metric))
        if base_value is None or trial_value is None or base_value <= 0.0:
            continue
        relative_change = float((trial_value - base_value) / base_value)
        deltas[f"{metric}_relative_change"] = relative_change
        if relative_change > loss_tolerance:
            reasons.append(f"{metric}_increase_exceeds_tolerance")
    if not reasons:
        reasons.append("within_parsimony_tolerances")
    return reasons == ["within_parsimony_tolerances"], reasons, deltas


def _extracted_feature_review_benchmarks(
    bow_metrics: Dict[str, Any],
    embedding_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    treatment_auc_values = _collect_metric_values(bow_metrics, "treatment_auroc")
    outcome_auc_values = _collect_metric_values(bow_metrics, "outcome_auroc")
    treatment_log_losses = _collect_metric_values(bow_metrics, "treatment_log_loss")
    outcome_log_losses = _collect_metric_values(bow_metrics, "outcome_log_loss")
    outcome_rmses = _collect_metric_values(bow_metrics, "outcome_rmse")
    r_losses = _collect_metric_values(bow_metrics, "r_loss_mean")

    embedding_probe_auc = _embedding_probe_auc_benchmarks(embedding_evidence)
    if embedding_probe_auc.get("treatment_probe_auc") is not None:
        treatment_auc_values.append(float(embedding_probe_auc["treatment_probe_auc"]))
    if embedding_probe_auc.get("outcome_probe_auc") is not None:
        outcome_auc_values.append(float(embedding_probe_auc["outcome_probe_auc"]))

    return {
        "treatment_auroc": _max_or_none(treatment_auc_values),
        "outcome_auroc": _max_or_none(outcome_auc_values),
        "treatment_log_loss": _min_or_none(treatment_log_losses),
        "outcome_log_loss": _min_or_none(outcome_log_losses),
        "outcome_rmse": _min_or_none(outcome_rmses),
        "r_loss_mean": _min_or_none(r_losses),
        "embedding_probe_auc": embedding_probe_auc,
    }


def _collect_metric_values(payload: Any, metric_name: str) -> List[float]:
    values: List[float] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == metric_name or str(key).endswith(f"_{metric_name}"):
                numeric = _finite_or_none(value)
                if numeric is not None:
                    values.append(float(numeric))
            values.extend(_collect_metric_values(value, metric_name))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_collect_metric_values(item, metric_name))
    return values


def _embedding_probe_auc_benchmarks(evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(evidence, dict):
        return {}
    treatment_values: List[float] = []
    outcome_values: List[float] = []
    effect_values: List[float] = []
    for contrast in evidence.get("contrasts", []) or []:
        if not isinstance(contrast, dict):
            continue
        auc = _finite_or_none(contrast.get("probe_auc"))
        if auc is None:
            continue
        name = str(contrast.get("name", ""))
        family = str(contrast.get("contrast_family", ""))
        role_hint = str(contrast.get("role_hint", ""))
        if name == "treatment":
            treatment_values.append(float(auc))
        elif name == "outcome":
            outcome_values.append(float(auc))
        elif role_hint == "effect_modifier" or "r_pseudo" in family:
            effect_values.append(float(auc))
    return {
        "treatment_probe_auc": _max_or_none(treatment_values),
        "outcome_probe_auc": _max_or_none(outcome_values),
        "effect_modifier_probe_auc": _max_or_none(effect_values),
    }


def _extracted_feature_review_gate(
    *,
    diagnostic: Dict[str, Any],
    nn_config: MultiModelAgenticForestConfig,
) -> Dict[str, Any]:
    metrics = diagnostic.get("metrics", {})
    benchmark = diagnostic.get("benchmark", {})
    failures: List[Dict[str, Any]] = []
    if metrics.get("status") != "ok":
        failures.append(
            {
                "metric": "status",
                "observed": metrics.get("status"),
                "benchmark": "ok",
                "reason": "diagnostic_status_not_ok",
            }
        )

    auc_margin = float(getattr(nn_config, "extracted_feature_review_auc_margin", 0.02))
    loss_margin = float(
        getattr(nn_config, "extracted_feature_review_loss_relative_margin", 0.05)
    )
    min_auc = float(
        getattr(nn_config, "extracted_feature_review_min_benchmark_auc", 0.55)
    )

    for metric in ["treatment_auroc", "outcome_auroc"]:
        observed = _finite_or_none(metrics.get(metric))
        target = _finite_or_none(benchmark.get(metric))
        if target is None or target < min_auc:
            continue
        if observed is None or observed < target - auc_margin:
            failures.append(
                {
                    "metric": metric,
                    "observed": observed,
                    "benchmark": target,
                    "required_min": target - auc_margin,
                    "reason": "auc_under_benchmark",
                }
            )

    for metric in ["treatment_log_loss", "outcome_log_loss", "outcome_rmse", "r_loss_mean"]:
        observed = _finite_or_none(metrics.get(metric))
        target = _finite_or_none(benchmark.get(metric))
        if target is None or target <= 0.0:
            continue
        max_allowed = target * (1.0 + loss_margin)
        if observed is None or observed > max_allowed:
            failures.append(
                {
                    "metric": metric,
                    "observed": observed,
                    "benchmark": target,
                    "required_max": max_allowed,
                    "reason": "loss_over_benchmark",
                }
            )

    return {
        "passed": not failures,
        "failed_criteria": failures,
        "n_failed_criteria": int(len(failures)),
    }


def _extracted_review_selection_score(
    diagnostic: Dict[str, Any],
    gate: Dict[str, Any],
) -> Tuple[int, float, float, float]:
    metrics = diagnostic.get("metrics", {})
    fail_count = int(gate.get("n_failed_criteria", 0))
    r_loss = _finite_or_none(metrics.get("r_loss_mean"))
    treatment_auc = _finite_or_none(metrics.get("treatment_auroc"))
    outcome_auc = _finite_or_none(metrics.get("outcome_auroc"))
    return (
        fail_count,
        float("inf") if r_loss is None else float(r_loss),
        float("inf") if treatment_auc is None else -float(treatment_auc),
        float("inf") if outcome_auc is None else -float(outcome_auc),
    )


def _extracted_review_summary(
    *,
    diagnostic: Optional[Dict[str, Any]],
    status: str,
    passed: bool,
    rounds: int,
) -> Dict[str, Any]:
    metrics = diagnostic.get("metrics", {}) if diagnostic else {}
    gate = diagnostic.get("gate", {}) if diagnostic else {}
    return {
        "enabled": True,
        "review_status": status,
        "review_passed": bool(passed),
        "review_rounds": int(rounds),
        "n_failed_criteria": int(gate.get("n_failed_criteria", 0) or 0),
        "failed_criteria": gate.get("failed_criteria", []),
        "treatment_auroc": metrics.get("treatment_auroc"),
        "outcome_auroc": metrics.get("outcome_auroc"),
        "outcome_rmse": metrics.get("outcome_rmse"),
        "r_loss_mean": metrics.get("r_loss_mean"),
        "r_loss_relative_improvement": metrics.get("r_loss_relative_improvement"),
    }


def _protect_required_feature_proposals(
    proposals: Sequence[AgenticFeatureProposal],
    required_names: set,
) -> Tuple[List[AgenticFeatureProposal], List[Dict[str, Any]]]:
    kept: List[AgenticFeatureProposal] = []
    rejected: List[Dict[str, Any]] = []
    for proposal in proposals:
        if proposal.action == "remove" and proposal.name in required_names:
            rejected.append(
                {
                    "proposal": _proposal_to_dict(proposal),
                    "reason": "cannot_remove_required_feature",
                }
            )
            continue
        kept.append(proposal)
    return kept, rejected


def _spec_sets_differ(
    left: Sequence[ExplicitFeatureSpec],
    right: Sequence[ExplicitFeatureSpec],
) -> bool:
    return [_spec_to_dict(spec) for spec in left] != [_spec_to_dict(spec) for spec in right]


def _redact_review_artifact(
    diagnostic: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> Dict[str, Any]:
    del search_config
    return diagnostic


def _compact_extracted_feature_review_context(context: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(context)
    original = compact.get("original_bow_context")
    if isinstance(original, dict) and isinstance(original.get("feature_importance"), dict):
        compact["original_bow_context"] = {
            **original,
            "feature_importance": _compact_multi_model_importance(
                original["feature_importance"]
            ),
        }
    if isinstance(compact.get("embedding_contrast_evidence"), dict):
        compact["embedding_contrast_evidence"] = _compact_embedding_contrast_evidence(
            compact["embedding_contrast_evidence"]
        )
    if isinstance(compact.get("htr_attention_evidence"), dict):
        compact["htr_attention_evidence"] = _compact_htr_attention_evidence(
            compact["htr_attention_evidence"]
        )
    return _round_floats(compact)


def _max_or_none(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    return max(finite) if finite else None


def _min_or_none(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    return min(finite) if finite else None


def _fit_transform_bow_plus_explicit(
    *,
    texts: Sequence[str],
    fit_pos: np.ndarray,
    heldout_pos: np.ndarray,
    vectorizer_params: Dict[str, Any],
    explicit_feature_dicts: Optional[List[Dict[str, Any]]],
    explicit_specs: Optional[List[ExplicitFeatureSpec]],
):
    vectorizer = _make_bow_vectorizer(vectorizer_params)
    x_fit = vectorizer.fit_transform([texts[i] for i in fit_pos])
    x_heldout = vectorizer.transform([texts[i] for i in heldout_pos])
    if not explicit_feature_dicts or not explicit_specs:
        return x_fit, x_heldout

    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    fit_dicts = [explicit_feature_dicts[int(i)] for i in fit_pos]
    heldout_dicts = [explicit_feature_dicts[int(i)] for i in heldout_pos]
    fit_explicit, _ = get_raw_explicit_features(
        fit_dicts,
        explicit_specs,
        continuous_means=means,
        continuous_stds=stds,
        role=None,
    )
    heldout_explicit, _ = get_raw_explicit_features(
        heldout_dicts,
        explicit_specs,
        continuous_means=means,
        continuous_stds=stds,
        role=None,
    )
    return (
        _hstack_sparse_and_dense(x_fit, fit_explicit),
        _hstack_sparse_and_dense(x_heldout, heldout_explicit),
    )


def _append_explicit_features_full(
    x_text,
    text_feature_names: np.ndarray,
    *,
    explicit_feature_dicts: Optional[List[Dict[str, Any]]],
    explicit_specs: Optional[List[ExplicitFeatureSpec]],
) -> Tuple[Any, np.ndarray, List[str]]:
    if not explicit_feature_dicts or not explicit_specs:
        return x_text, text_feature_names, []
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    explicit_features, explicit_names = get_raw_explicit_features(
        explicit_feature_dicts,
        explicit_specs,
        continuous_means=means,
        continuous_stds=stds,
        role=None,
    )
    prefixed_names = [f"explicit:{name}" for name in explicit_names]
    features = np.concatenate(
        [text_feature_names, np.asarray(prefixed_names, dtype=object)]
    )
    return _hstack_sparse_and_dense(x_text, explicit_features), features, prefixed_names


def _hstack_sparse_and_dense(x_text: Any, explicit_features: Sequence[Sequence[float]]):
    explicit_matrix = np.asarray(explicit_features, dtype=np.float32)
    if explicit_matrix.ndim != 2 or explicit_matrix.shape[1] == 0:
        return x_text
    return sparse.hstack(
        [x_text, sparse.csr_matrix(explicit_matrix)],
        format="csr",
        dtype=np.float32,
    )


def _fit_binary_bow_fold(
    texts: Sequence[str],
    labels: np.ndarray,
    fit_pos: np.ndarray,
    heldout_pos: np.ndarray,
    vectorizer_params: Dict[str, Any],
    model_params: Dict[str, Any],
    *,
    explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
    explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(int)
    fit_pos = np.asarray(fit_pos)
    heldout_pos = np.asarray(heldout_pos)
    if len(np.unique(labels[fit_pos])) < 2:
        return heldout_pos, np.full(
            len(heldout_pos),
            float(np.mean(labels[fit_pos])),
            dtype=float,
        )
    x_fit, x_heldout = _fit_transform_bow_plus_explicit(
        texts=texts,
        fit_pos=fit_pos,
        heldout_pos=heldout_pos,
        vectorizer_params=vectorizer_params,
        explicit_feature_dicts=explicit_feature_dicts,
        explicit_specs=explicit_specs,
    )
    model = _make_bow_classifier(model_params, random_state=random_state)
    model.fit(x_fit, labels[fit_pos])
    return heldout_pos, model.predict_proba(x_heldout)[:, 1]


def _fit_regression_bow_fold(
    texts: Sequence[str],
    values: np.ndarray,
    fit_pos: np.ndarray,
    heldout_pos: np.ndarray,
    vectorizer_params: Dict[str, Any],
    model_params: Dict[str, Any],
    *,
    explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
    explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    fit_pos = np.asarray(fit_pos)
    heldout_pos = np.asarray(heldout_pos)
    x_fit, x_heldout = _fit_transform_bow_plus_explicit(
        texts=texts,
        fit_pos=fit_pos,
        heldout_pos=heldout_pos,
        vectorizer_params=vectorizer_params,
        explicit_feature_dicts=explicit_feature_dicts,
        explicit_specs=explicit_specs,
    )
    model = _make_bow_regressor(model_params, random_state=random_state)
    fold_weight = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        fold_weight = weights[fit_pos]
    _fit_regressor(model, x_fit, values[fit_pos], sample_weight=fold_weight)
    return heldout_pos, model.predict(x_heldout)


def _fit_regressor(
    model: Any,
    x: Any,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
) -> Any:
    if sample_weight is None:
        return model.fit(x, y)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape[0] != len(y):
        raise ValueError("sample_weight must have one value per training row")
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        return model.fit(x, y)
    try:
        return model.fit(x, y, sample_weight=weights)
    except TypeError:
        logger.warning(
            "BoW regressor %s does not accept sample_weight; fitting unweighted",
            type(model).__name__,
        )
        return model.fit(x, y)


def _make_bow_vectorizer(params: Dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)[a-z0-9%<>+=-]+",
        ngram_range=(
            int(params["ngram_range_min"]),
            int(params["ngram_range_max"]),
        ),
        min_df=int(params["min_df"]),
        max_df=float(params["max_df"]),
        sublinear_tf=bool(params["sublinear_tf"]),
        max_features=int(params["max_features"]),
        dtype=np.float32,
    )


def _make_bow_classifier(params: Dict[str, Any], *, random_state: int = 17):
    model_name = str(params["bow_model"]).strip().lower()
    if model_name == "linear":
        return LogisticRegression(
            C=float(params["logistic_c"]),
            solver="liblinear",
            max_iter=int(params["logistic_max_iter"]),
            random_state=random_state,
        )
    if model_name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=1,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=1,
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "bow_model='xgboost' requires the xgboost package"
            ) from exc
        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.6,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported bow_model: {model_name}")


def _make_bow_regressor(params: Dict[str, Any], *, random_state: int = 17):
    model_name = str(params["bow_model"]).strip().lower()
    if model_name == "linear":
        return Ridge(alpha=float(params["ridge_alpha"]), random_state=random_state)
    if model_name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=1,
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=1,
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "bow_model='xgboost' requires the xgboost package"
            ) from exc
        return XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.6,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported bow_model: {model_name}")


def _bounded_fold_count(requested: int, n_rows: int) -> int:
    if n_rows < 2:
        raise ValueError("At least two rows are required for cross-fitting")
    return max(2, min(int(requested), int(n_rows)))


def _bounded_stratified_folds(labels: np.ndarray, requested: int) -> int:
    values, counts = np.unique(labels.astype(int), return_counts=True)
    if len(values) < 2:
        raise ValueError("Binary cross-fitting requires both treatment/outcome classes")
    return max(2, min(int(requested), int(np.min(counts)), int(len(labels))))


def _binary_split_items(
    labels: np.ndarray,
    *,
    requested_folds: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    values, counts = np.unique(labels.astype(int), return_counts=True)
    if len(values) >= 2 and int(np.min(counts)) >= 2:
        folds = _bounded_stratified_folds(labels, requested_folds)
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )
        return [
            (np.asarray(fit_pos), np.asarray(heldout_pos))
            for fit_pos, heldout_pos in splitter.split(np.zeros(len(labels)), labels)
        ]

    folds = _bounded_fold_count(requested_folds, len(labels))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    return [
        (np.asarray(fit_pos), np.asarray(heldout_pos))
        for fit_pos, heldout_pos in splitter.split(np.zeros(len(labels)))
    ]


def _top_feature_rows(
    features: np.ndarray,
    scores: np.ndarray,
    top_n: int,
    *,
    descending: bool = True,
    treatment_coef: Optional[np.ndarray] = None,
    outcome_coef: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    if len(features) == 0:
        return []
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    rows: List[Dict[str, Any]] = []
    for idx in order[:top_n]:
        row = {
            "feature": str(features[idx]),
            "score": _finite_or_none(scores[idx]),
        }
        if treatment_coef is not None:
            row["treatment_score"] = _finite_or_none(treatment_coef[idx])
            row["abs_treatment_score"] = _finite_or_none(abs(treatment_coef[idx]))
        if outcome_coef is not None:
            row["outcome_score"] = _finite_or_none(outcome_coef[idx])
            row["abs_outcome_score"] = _finite_or_none(abs(outcome_coef[idx]))
        rows.append(row)
    return rows


def _top_phrase_feature_rows(
    features: np.ndarray,
    *,
    top_n: int,
    treatment_coef: np.ndarray,
    outcome_coef: np.ndarray,
    pseudo_target_coef: np.ndarray,
    confounder_score: np.ndarray,
) -> List[Dict[str, Any]]:
    """Return agent-facing phrase evidence from 2-4 token n-grams.

    The predictive models can still use unigrams. This summary gives the
    proposal agent a phrase-biased view that is easier to map to extractable
    clinical variables.
    """
    if len(features) == 0:
        return []

    phrase_indices = [
        idx
        for idx, feature in enumerate(features)
        if 2 <= _feature_token_count(str(feature)) <= 4
    ]
    if not phrase_indices:
        return []

    phrase_indices_array = np.asarray(phrase_indices, dtype=int)
    treatment_abs = np.abs(treatment_coef)
    outcome_abs = np.abs(outcome_coef)
    pseudo_abs = np.abs(pseudo_target_coef)

    combined_score = np.maximum.reduce(
        [
            _scale_scores_for_phrase_ranking(treatment_abs),
            _scale_scores_for_phrase_ranking(outcome_abs),
            _scale_scores_for_phrase_ranking(pseudo_abs),
            _scale_scores_for_phrase_ranking(confounder_score),
        ]
    )
    order = phrase_indices_array[
        np.argsort(combined_score[phrase_indices_array])[::-1]
    ]

    rows: List[Dict[str, Any]] = []
    for idx in order[:top_n]:
        row = {
            "feature": str(features[idx]),
            "token_count": int(_feature_token_count(str(features[idx]))),
            "combined_score": _finite_or_none(combined_score[idx]),
            "confounder_overlap_score": _finite_or_none(confounder_score[idx]),
            "treatment_score": _finite_or_none(treatment_coef[idx]),
            "abs_treatment_score": _finite_or_none(treatment_abs[idx]),
            "outcome_score": _finite_or_none(outcome_coef[idx]),
            "abs_outcome_score": _finite_or_none(outcome_abs[idx]),
            "pseudo_target_score": _finite_or_none(pseudo_target_coef[idx]),
            "abs_pseudo_target_score": _finite_or_none(pseudo_abs[idx]),
        }
        rows.append(row)
    return rows


def _feature_token_count(feature: str) -> int:
    return len([token for token in str(feature).split() if token])


def _scale_scores_for_phrase_ranking(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    max_abs = float(np.nanmax(np.abs(values))) if len(values) else 0.0
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        return np.zeros_like(values, dtype=float)
    return np.abs(values) / max_abs


def _model_feature_scores(model: Any, n_features: int) -> np.ndarray:
    coef = getattr(model, "coef_", None)
    if coef is not None:
        values = np.asarray(coef, dtype=float).ravel()
        return _resize_scores(values, n_features)
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        values = np.asarray(importances, dtype=float).ravel()
        return _resize_scores(values, n_features)
    booster = getattr(model, "get_booster", None)
    if booster is not None:
        try:
            score = booster().get_score(importance_type="gain")
            values = np.zeros(n_features, dtype=float)
            for key, value in score.items():
                if key.startswith("f"):
                    index = int(key[1:])
                    if 0 <= index < n_features:
                        values[index] = float(value)
            return values
        except Exception:
            pass
    return np.zeros(n_features, dtype=float)


def _resize_scores(values: np.ndarray, n_features: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    if len(values) == n_features:
        return values
    resized = np.zeros(n_features, dtype=float)
    limit = min(n_features, len(values))
    resized[:limit] = values[:limit]
    return resized


def _merge_proposals(
    left: AgenticFeatureProposal,
    right: AgenticFeatureProposal,
) -> AgenticFeatureProposal:
    categories = _merge_ordered_values(left.categories, right.categories) or None
    feature_type = (
        "categorical"
        if left.type == "categorical" or right.type == "categorical" or categories
        else (left.type or right.type)
    )
    return AgenticFeatureProposal(
        action="add",
        name=left.name,
        type=feature_type,
        categories=categories,
        description=_merge_text_values(left.description, right.description),
        roles=_merge_ordered_values(left.roles, right.roles),
        rationale=_merge_text_values(left.rationale, right.rationale),
        expected_signal=_merge_text_values(left.expected_signal, right.expected_signal),
    )


def _merge_ordered_values(left: Any, right: Any) -> List[str]:
    values: List[str] = []
    for item in _as_list(left) + _as_list(right):
        text = str(item).strip()
        if text and text not in values:
            values.append(text)
    return values


def _merge_value_aliases(left: Any, right: Any) -> Optional[Dict[str, List[str]]]:
    merged: Dict[str, List[str]] = {}
    for source in [left, right]:
        if not isinstance(source, dict):
            continue
        for category, aliases in source.items():
            category_text = str(category).strip()
            if not category_text:
                continue
            merged[category_text] = _merge_ordered_values(
                merged.get(category_text, []),
                aliases,
            )
    return merged or None


def _merge_text_values(left: Any, right: Any) -> Optional[str]:
    left_text = str(left).strip() if left is not None else ""
    right_text = str(right).strip() if right is not None else ""
    if not left_text:
        return right_text or None
    if not right_text or right_text == left_text:
        return left_text
    return f"{left_text} / {right_text}"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_spec(spec: ExplicitFeatureSpec) -> ExplicitFeatureSpec:
    normalized_name = _normalize_feature_name(spec.name)
    if normalized_name == spec.name:
        return spec
    return ExplicitFeatureSpec(
        name=normalized_name,
        type=spec.type,
        categories=spec.categories,
        description=spec.description,
        value_aliases=getattr(spec, "value_aliases", None),
        roles=spec.roles,
    )


def _dedupe_specs(specs: Sequence[ExplicitFeatureSpec]) -> List[ExplicitFeatureSpec]:
    by_name: Dict[str, ExplicitFeatureSpec] = {}
    for spec in specs:
        spec = _normalize_spec(spec)
        name = _normalize_feature_name(spec.name)
        if not name:
            continue
        if name not in by_name:
            by_name[name] = spec
            continue
        existing = by_name[name]
        roles = list(dict.fromkeys([*existing.roles, *spec.roles]))
        categories = _merge_ordered_values(existing.categories, spec.categories) or None
        value_aliases = _merge_value_aliases(
            getattr(existing, "value_aliases", None),
            getattr(spec, "value_aliases", None),
        )
        if existing.type == "categorical" or spec.type == "categorical" or categories:
            feature_type = "categorical"
        else:
            feature_type = existing.type
        by_name[name] = ExplicitFeatureSpec(
            name=name,
            type=feature_type,
            categories=categories,
            description=_merge_text_values(existing.description, spec.description),
            value_aliases=value_aliases,
            roles=roles,
        )
    return list(by_name.values())


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _bow_view_to_dict(view: BoWViewConfig) -> Dict[str, Any]:
    return {
        "name": str(view.name),
        "bow_model": str(view.bow_model),
        "ngram_range_min": int(view.ngram_range_min),
        "ngram_range_max": int(view.ngram_range_max),
        "min_df": int(view.min_df),
        "max_df": float(view.max_df),
        "max_features": int(view.max_features),
        "sublinear_tf": bool(view.sublinear_tf),
        "logistic_c": float(view.logistic_c),
        "logistic_max_iter": int(view.logistic_max_iter),
        "ridge_alpha": float(view.ridge_alpha),
    }


def _compact_multi_model_agent_context(context: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(context)
    if isinstance(context.get("feature_importance"), dict):
        compact["feature_importance"] = _compact_multi_model_importance(
            context["feature_importance"]
        )
    if isinstance(context.get("embedding_contrast_evidence"), dict):
        compact["embedding_contrast_evidence"] = _compact_embedding_contrast_evidence(
            context["embedding_contrast_evidence"]
        )
    if isinstance(context.get("htr_attention_evidence"), dict):
        compact["htr_attention_evidence"] = _compact_htr_attention_evidence(
            context["htr_attention_evidence"]
        )
    compact["prompt_compaction"] = {
        "feature_importance": (
            f"per-view feature lists capped at {_AGENT_PROMPT_VIEW_TOP_N}; "
            f"consensus capped at {_AGENT_PROMPT_CONSENSUS_TOP_N}"
        ),
        "embedding_contrast_evidence": (
            f"retrieved chunks capped at {_AGENT_PROMPT_EMBEDDING_CHUNKS_PER_TAIL} "
            f"per tail and {_AGENT_PROMPT_EMBEDDING_CHUNK_CHARS} chars each"
        ),
        "htr_attention_evidence": (
            f"attention rows capped at {_AGENT_PROMPT_HTR_ROWS_PER_STAGE} per stage; "
            f"snippets capped at {_AGENT_PROMPT_HTR_SNIPPET_CHARS} chars"
        ),
    }
    return compact


def _compact_multi_model_importance(importance: Dict[str, Any]) -> Dict[str, Any]:
    consensus = _compact_feature_rows(
        importance.get("phrase_consensus")
        or importance.get("phrase_features")
        or [],
        _AGENT_PROMPT_CONSENSUS_TOP_N,
    )
    compact_views = []
    for view in importance.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        compact_view: Dict[str, Any] = {
            "view_name": view.get("view_name"),
            "view_index": view.get("view_index"),
            "view_config": view.get("view_config"),
            "metrics": view.get("metrics"),
            "n_features": view.get("n_features"),
            "n_bow_features": view.get("n_bow_features"),
            "n_prespecified_features": view.get("n_prespecified_features"),
            "n_prespecified_raw_features": view.get("n_prespecified_raw_features"),
            "prespecified_raw_feature_names": _clip_list(
                view.get("prespecified_raw_feature_names", []),
                50,
            ),
        }
        for key in [
            "phrase_features",
            "confounder_overlap",
            "treatment_positive",
            "treatment_negative",
            "outcome_positive",
            "outcome_negative",
            "pseudo_target_positive",
            "pseudo_target_negative",
        ]:
            compact_view[key] = _compact_feature_rows(
                view.get(key, []) or [],
                _AGENT_PROMPT_VIEW_TOP_N,
            )
        compact_views.append(compact_view)

    compact_importance = {
        "n_views": importance.get("n_views", len(compact_views)),
        "views": compact_views,
        "phrase_features": consensus,
        "phrase_consensus": consensus,
        "prompt_compaction": {
            "consensus_top_n": _AGENT_PROMPT_CONSENSUS_TOP_N,
            "per_view_list_top_n": _AGENT_PROMPT_VIEW_TOP_N,
        },
    }
    if isinstance(importance.get("ensemble_r"), dict):
        compact_importance["ensemble_r"] = _compact_multi_model_importance(
            importance["ensemble_r"]
        )
    for key in ["target_source", "pseudo_target_construction", "nuisance_sources"]:
        if key in importance:
            compact_importance[key] = importance[key]
    return compact_importance


def _compact_embedding_contrast_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        key: evidence.get(key)
        for key in [
            "enabled",
            "model_name",
            "unit",
            "chunking",
            "residualized_columns",
            "n_patients",
                "n_concept_phrases",
                "skipped",
                "error",
                "disabled_reason",
            ]
            if key in evidence
    }
    contrasts = []
    for contrast in evidence.get("contrasts", []) or []:
        if not isinstance(contrast, dict):
            continue
        compact_contrast = {
            key: _round_floats(contrast.get(key))
            for key in [
                "name",
                "positive_label",
                "negative_label",
                "role_hint",
                "contrast_family",
                "n_positive",
                "n_negative",
                "mean_difference_norm",
                "probe_auc",
                "min_probe_auc",
                "direction_source",
                "direction_formula",
                "score_formula",
                "probe_auc_role",
                "direction_norm",
                "retrieval_skipped",
                "component_counts",
                "positive_cell_labels",
                "negative_cell_labels",
            ]
            if key in contrast
        }
        compact_contrast["positive_aligned_chunks"] = _compact_embedding_chunks(
            contrast.get("positive_aligned_chunks", []) or []
        )
        compact_contrast["negative_aligned_chunks"] = _compact_embedding_chunks(
            contrast.get("negative_aligned_chunks", []) or []
        )
        compact_contrast["concept_probe_scores"] = _compact_concept_scores(
            contrast.get("concept_probe_scores", []) or []
        )
        contrasts.append(compact_contrast)
    compact["contrasts"] = contrasts
    compact["prompt_compaction"] = {
        "chunks_per_tail": _AGENT_PROMPT_EMBEDDING_CHUNKS_PER_TAIL,
        "chunk_text_chars": _AGENT_PROMPT_EMBEDDING_CHUNK_CHARS,
        "concept_top_n": _AGENT_PROMPT_CONCEPT_TOP_N,
    }
    return compact


def _compact_htr_attention_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for stage_key in ["nuisance", "effect"]:
        stage_evidence = evidence.get(stage_key)
        if not isinstance(stage_evidence, dict):
            continue
        compact[stage_key] = {
            "metrics": _round_floats(stage_evidence.get("metrics", {})),
            "attention": _compact_htr_attention_rows(
                stage_evidence.get("attention", []) or [],
                max_rows=_AGENT_PROMPT_HTR_ROWS_PER_STAGE,
            ),
        }
    if compact:
        compact["prompt_compaction"] = {
            "rows_per_stage": _AGENT_PROMPT_HTR_ROWS_PER_STAGE,
            "snippet_chars": _AGENT_PROMPT_HTR_SNIPPET_CHARS,
            "summary_chars": _AGENT_PROMPT_HTR_SUMMARY_CHARS,
        }
    return compact


def _compact_htr_attention_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    max_rows: int,
) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        already_compact_text = bool(
            row.get("evidence_snippet")
            or row.get("top_token_spans")
            or row.get("attended_token_summary")
        )
        if not _attention_row_has_usable_text(row) and not already_compact_text:
            continue
        spans = _parse_top_token_spans(
            row.get("top_token_spans") or row.get("top_token_spans_json")
        )
        item: Dict[str, Any] = {}
        for key in [
            "row_id",
            "_oci_row_id",
            "outer_fold",
            "fold",
            "stage",
            "model_family",
            "chunk_index",
            "effect_objective",
            "target_source",
            "view_name",
        ]:
            if key in row:
                item[key] = _round_floats(row[key])
        for key in [
            "attention",
            "attention_score",
            "chunk_attention",
            "e_hat",
            "m_hat",
            "e_hat_raw",
            "m_hat_raw",
            "y_residual",
            "t_residual",
            "tau_hat_r_stage",
            "tau_logit_modifier",
            "r_pseudo_outcome",
            "r_loss",
            "effect_loss",
            "effect_loss_at_zero_tau",
        ]:
            if key in row:
                item[key] = _round_floats(row[key])
        snippet = _clip_text(row.get("evidence_snippet"), _AGENT_PROMPT_HTR_SNIPPET_CHARS)
        if not snippet:
            snippet = _attention_evidence_snippet(
                row.get("chunk_text"),
                spans,
                row.get("highlighted_chunk_text"),
            )
        if snippet:
            item["evidence_snippet"] = _clip_text(snippet, _AGENT_PROMPT_HTR_SNIPPET_CHARS)
        if spans:
            item["top_token_spans"] = _compact_token_spans(spans)
        summary = _clip_text(
            row.get("attended_token_summary"),
            _AGENT_PROMPT_HTR_SUMMARY_CHARS,
        )
        if summary:
            item["attended_token_summary"] = summary
        if item:
            compact.append(item)
        if len(compact) >= max(0, int(max_rows)):
            break
    return compact


def _redact_htr_attention_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    compact = _compact_htr_attention_evidence(evidence)
    for stage_key in ["nuisance", "effect"]:
        stage_evidence = compact.get(stage_key)
        if not isinstance(stage_evidence, dict):
            continue
        redacted_rows = []
        for row in stage_evidence.get("attention", []) or []:
            if not isinstance(row, dict):
                continue
            redacted = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "evidence_snippet",
                    "top_token_spans",
                    "attended_token_summary",
                }
            }
            redacted["text_redacted"] = True
            redacted_rows.append(redacted)
        stage_evidence["attention"] = redacted_rows
    return compact


def _compact_feature_rows(rows: Sequence[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    compact = []
    for row in list(rows)[: max(0, int(top_n))]:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                key: _round_floats(value)
                for key, value in row.items()
                if key
                in {
                    "feature",
                    "token_count",
                    "score",
                    "combined_score",
                    "confounder_overlap_score",
                    "treatment_score",
                    "abs_treatment_score",
                    "outcome_score",
                    "abs_outcome_score",
                    "pseudo_target_score",
                    "abs_pseudo_target_score",
                    "supporting_view_count",
                    "supporting_views",
                    "best_abs_confounder_score",
                    "mean_abs_confounder_score",
                    "best_abs_effect_score",
                    "mean_abs_effect_score",
                }
            }
        )
    return compact


def _compact_embedding_chunks(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for row in list(rows)[:_AGENT_PROMPT_EMBEDDING_CHUNKS_PER_TAIL]:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "row_id": row.get("row_id"),
                "chunk_index": row.get("chunk_index"),
                "score": _round_floats(row.get("score")),
                "text": _clip_text(row.get("text"), _AGENT_PROMPT_EMBEDDING_CHUNK_CHARS),
            }
        )
    return compact


def _compact_concept_scores(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for row in list(rows)[:_AGENT_PROMPT_CONCEPT_TOP_N]:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "concept": row.get("concept"),
                "score": _round_floats(row.get("score")),
            }
        )
    return compact


def _clip_list(values: Any, max_items: int) -> List[Any]:
    if not isinstance(values, list):
        return []
    return values[: max(0, int(max_items))]


def _clip_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    limit = max(0, int(max_chars))
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return round(float(value), 5)
    if isinstance(value, np.floating):
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        return round(numeric, 5)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _multi_view_metrics(view_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not view_results:
        return {"n_bow_views": 0}
    primary = _select_primary_bow_view(view_results)
    primary_metrics = primary.get("metrics", {})
    metrics: Dict[str, Any] = {
        "n_bow_views": int(len(view_results)),
        "primary_view": str(primary["view"].name),
        "primary_view_index": int(primary["view_index"]),
        "views": [
            {
                "view_name": str(result.get("view_name") or result["view"].name),
                "view_index": int(result["view_index"]),
                "view_config": _bow_view_to_dict(result["view"]),
                "metrics": _agent_visible_metrics(result.get("metrics", {})),
            }
            for result in view_results
        ],
    }
    for key, value in _scalar_metrics(primary_metrics).items():
        metrics[f"primary_{key}"] = value
    best_improvement = max(
        (
            value
            for value in (
                _finite_or_none(result.get("metrics", {}).get("r_loss_relative_improvement"))
                for result in view_results
            )
            if value is not None
        ),
        default=None,
    )
    metrics["best_r_loss_relative_improvement"] = best_improvement
    return metrics


def _select_primary_bow_view(view_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def score(result: Dict[str, Any]) -> Tuple[float, float]:
        metrics = result.get("metrics", {})
        improvement = _finite_or_none(metrics.get("r_loss_relative_improvement"))
        tau_corr = _finite_or_none(metrics.get("tau_hat_pseudo_target_corr"))
        return (
            float("-inf") if improvement is None else improvement,
            float("-inf") if tau_corr is None else abs(tau_corr),
        )

    return max(view_results, key=score)


def _multi_view_importance(
    view_results: Sequence[Dict[str, Any]],
    *,
    top_n: int,
) -> Dict[str, Any]:
    views = []
    for result in view_results:
        importance = dict(result.get("importance", {}))
        importance["view_name"] = str(result.get("view_name") or result["view"].name)
        importance["view_index"] = int(result["view_index"])
        importance["view_config"] = _bow_view_to_dict(result["view"])
        importance["metrics"] = _agent_visible_metrics(result.get("metrics", {}))
        views.append(importance)
    consensus = _consensus_phrase_feature_rows(views, top_n=top_n)
    return {
        "n_views": int(len(views)),
        "views": views,
        "phrase_features": consensus,
        "phrase_consensus": consensus,
    }


def _consensus_phrase_feature_rows(
    view_importances: Sequence[Dict[str, Any]],
    *,
    top_n: int,
) -> List[Dict[str, Any]]:
    accumulator: Dict[str, Dict[str, Any]] = {}
    for view in view_importances:
        view_name = str(view.get("view_name", "view"))
        for row in view.get("phrase_features", []) or []:
            feature = str(row.get("feature", "")).strip()
            if not feature:
                continue
            key = _normalize_text(feature)
            entry = accumulator.setdefault(
                key,
                {
                    "feature": feature,
                    "supporting_views": set(),
                    "view_scores": [],
                    "abs_confounder_scores": [],
                    "abs_effect_scores": [],
                },
            )
            entry["supporting_views"].add(view_name)
            confounder_score = abs(
                float(row.get("confounder_overlap_score") or 0.0)
            )
            effect_score = abs(float(row.get("abs_pseudo_target_score") or 0.0))
            entry["abs_confounder_scores"].append(confounder_score)
            entry["abs_effect_scores"].append(effect_score)
            entry["view_scores"].append(
                {
                    "view_name": view_name,
                    "combined_score": row.get("combined_score"),
                    "confounder_overlap_score": row.get("confounder_overlap_score"),
                    "treatment_score": row.get("treatment_score"),
                    "outcome_score": row.get("outcome_score"),
                    "pseudo_target_score": row.get("pseudo_target_score"),
                }
            )

    rows: List[Dict[str, Any]] = []
    for entry in accumulator.values():
        confounder_scores = entry["abs_confounder_scores"]
        effect_scores = entry["abs_effect_scores"]
        supporting_views = sorted(entry["supporting_views"])
        best_confounder = max(confounder_scores) if confounder_scores else 0.0
        best_effect = max(effect_scores) if effect_scores else 0.0
        mean_confounder = float(np.mean(confounder_scores)) if confounder_scores else 0.0
        mean_effect = float(np.mean(effect_scores)) if effect_scores else 0.0
        rows.append(
            {
                "feature": entry["feature"],
                "supporting_view_count": int(len(supporting_views)),
                "supporting_views": supporting_views,
                "best_abs_confounder_score": _finite_or_none(best_confounder),
                "mean_abs_confounder_score": _finite_or_none(mean_confounder),
                "best_abs_effect_score": _finite_or_none(best_effect),
                "mean_abs_effect_score": _finite_or_none(mean_effect),
                "view_scores": entry["view_scores"],
            }
        )

    rows.sort(
        key=lambda row: (
            int(row["supporting_view_count"]),
            float(row.get("best_abs_confounder_score") or 0.0),
            float(row.get("best_abs_effect_score") or 0.0),
            float(row.get("mean_abs_confounder_score") or 0.0),
            float(row.get("mean_abs_effect_score") or 0.0),
        ),
        reverse=True,
    )
    return rows[:top_n]


def _agent_visible_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if not _is_oracle_metric_name(key)
    }


def _is_oracle_metric_name(key: Any) -> bool:
    name = str(key).lower()
    return (
        name.startswith("oracle_")
        or name.startswith("true_")
        or "true_" in name
    )


def _scalar_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, (list, tuple, dict))
    }


def _prefix_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        f"{prefix}{key}": value
        for key, value in _scalar_metrics(metrics).items()
    }


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")
