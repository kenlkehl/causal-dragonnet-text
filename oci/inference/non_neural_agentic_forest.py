"""Non-neural BoW-guided agentic variable discovery plus causal forest."""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from ..config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    NonNeuralAgenticForestConfig,
)
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
    validate_agentic_proposals,
)


logger = logging.getLogger(__name__)


_DASH_TRANSLATION = dict.fromkeys(
    map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"),
    "-",
)


def run_non_neural_agentic_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device=None,
    num_workers: int = 1,
    proposal_agent: Optional[Any] = None,
    extraction_provider: Optional[Any] = None,
    evaluator: Optional[Any] = None,
) -> None:
    """Run BoW-guided agentic variable discovery and final explicit-feature forest."""
    del device
    runner = NonNeuralAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        num_workers=num_workers,
        proposal_agent=proposal_agent,
        extraction_provider=extraction_provider,
        evaluator=evaluator,
    )
    runner.run()


class NonNeuralAgenticForestRunner:
    """Sparse-text discovery path for explicit-variable causal forests."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        config: AppliedInferenceConfig,
        output_path: Path,
        num_workers: int = 1,
        proposal_agent: Optional[Any] = None,
        extraction_provider: Optional[Any] = None,
        evaluator: Optional[Any] = None,
    ) -> None:
        self.dataset = dataset.reset_index(drop=True).copy()
        self.dataset["_oci_row_id"] = np.arange(len(self.dataset), dtype=int)
        self.config = config
        self.output_path = Path(output_path)
        self.artifact_dir = self.output_path.parent / "non_neural_agentic_forest"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = 1 if num_workers is None else int(num_workers)
        self._has_external_components = (
            proposal_agent is not None
            or extraction_provider is not None
            or evaluator is not None
        )

        self.nn_config: NonNeuralAgenticForestConfig = getattr(
            config.architecture,
            "non_neural_agentic_forest",
            NonNeuralAgenticForestConfig(),
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

        self.bow_prediction_frames: List[pd.DataFrame] = []
        self.importance_rows: List[Dict[str, Any]] = []
        self.agent_rows: List[Dict[str, Any]] = []
        self.feature_set_rows: List[Dict[str, Any]] = []
        self.outer_metric_rows: List[Dict[str, Any]] = []
        self.alias_reference_specs: List[ExplicitFeatureSpec] = self._initial_specs()

    def run(self) -> None:
        logger.info("=" * 80)
        logger.info("NON-NEURAL AGENTIC FEATURE CAUSAL FOREST")
        logger.info("=" * 80)

        splits = self._analysis_splits()
        outer_n_jobs = self._outer_n_jobs(len(splits))
        if outer_n_jobs > 1 and self._has_external_components:
            logger.warning(
                "Outer fold parallelism disabled because custom agent/extractor/"
                "evaluator objects were supplied and may not be thread-safe."
            )
            outer_n_jobs = 1

        if outer_n_jobs > 1:
            logger.info(
                "Running %s non-neural outer fold(s) with outer_parallelism=%s",
                len(splits),
                outer_n_jobs,
            )
            fold_results = Parallel(n_jobs=outer_n_jobs, prefer="threads")(
                delayed(self._run_one_analysis_split_isolated)(
                    int(outer_fold),
                    np.asarray(train_idx),
                    np.asarray(test_idx),
                    outer_n_jobs,
                )
                for outer_fold, train_idx, test_idx in splits
            )
            fold_results = sorted(fold_results, key=lambda item: item["outer_fold"])
            prediction_frames = [item["predictions"] for item in fold_results]
            for item in fold_results:
                self.bow_prediction_frames.extend(item["bow_prediction_frames"])
                self.importance_rows.extend(item["importance_rows"])
                self.agent_rows.extend(item["agent_rows"])
                self.feature_set_rows.extend(item["feature_set_rows"])
                self.outer_metric_rows.extend(item["outer_metric_rows"])
        else:
            prediction_frames: List[pd.DataFrame] = []
            for outer_fold, train_idx, test_idx in splits:
                logger.info(
                    "Non-neural agentic fold %s: train=%s test=%s",
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
            "Non-neural agentic isolated fold %s: train=%s test=%s",
            outer_fold,
            len(train_idx),
            len(test_idx),
        )
        fold_runner = NonNeuralAgenticForestRunner(
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
            "importance_rows": fold_runner.importance_rows,
            "agent_rows": fold_runner.agent_rows,
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
            "No held-out split configured for non_neural_agentic_forest; "
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
        bow_result = self._fit_bow_discovery(discovery_df, outer_fold)
        self.bow_prediction_frames.append(bow_result["predictions"])
        self.importance_rows.append(
            {
                "outer_fold": int(outer_fold),
                "context": bow_result["context"],
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

        e_hat = self._crossfit_binary(texts, t, "treatment", outer_fold)
        if self.config.outcome_type == "continuous":
            m_hat = self._crossfit_continuous(texts, y, "outcome", outer_fold)
        else:
            m_hat = self._crossfit_binary(texts, y, "outcome", outer_fold)

        e_clipped = np.clip(e_hat, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip)
        t_resid = t - e_clipped
        y_resid = y - m_hat
        pseudo_target = y_resid / t_resid

        tau_hat = self._crossfit_pseudo_target(
            texts,
            pseudo_target,
            outer_fold,
        )
        r_loss = (y_resid - tau_hat * t_resid) ** 2
        r_loss_at_zero = y_resid**2

        predictions = pd.DataFrame(
            {
                "_oci_row_id": discovery_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "e_hat": e_hat,
                "m_hat": m_hat,
                "y_residual": y_resid,
                "t_residual": t_resid,
                "pseudo_target": pseudo_target,
                "tau_hat_non_neural": tau_hat,
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
        )
        context = self._build_agent_context(
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            metrics=metrics,
            importance=importance,
        )
        return {
            "predictions": predictions,
            "metrics": metrics,
            "importance": importance,
            "context": context,
        }

    def _crossfit_binary(
        self,
        texts: Sequence[str],
        labels: np.ndarray,
        label_name: str,
        outer_fold: int,
    ) -> np.ndarray:
        labels = labels.astype(int)
        oof = np.full(len(labels), np.nan, dtype=float)
        random_state = 11_000 + 100 * outer_fold + (1 if label_name == "outcome" else 2)
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

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            logger.info(
                "Outer fold %s BoW %s nuisance fold %s/%s: train=%s heldout=%s",
                outer_fold,
                label_name,
                fold,
                folds,
                len(fit_pos),
                len(heldout_pos),
            )
            if len(np.unique(labels[fit_pos])) < 2:
                return heldout_pos, np.full(
                    len(heldout_pos),
                    float(np.mean(labels[fit_pos])),
                    dtype=float,
                )
            model = Pipeline(
                [
                    ("tfidf", self._make_vectorizer()),
                    ("model", self._make_classifier(random_state=17 + fold)),
                ]
            )
            model.fit([texts[i] for i in fit_pos], labels[fit_pos])
            return heldout_pos, model.predict_proba([texts[i] for i in heldout_pos])[:, 1]

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
    ) -> np.ndarray:
        oof = np.full(len(values), np.nan, dtype=float)
        folds = _bounded_fold_count(self.nn_config.nuisance_folds, len(values))
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=12_000 + 100 * outer_fold,
        )
        split_items = list(enumerate(splitter.split(texts), start=1))

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            logger.info(
                "Outer fold %s BoW %s nuisance fold %s/%s: train=%s heldout=%s",
                outer_fold,
                label_name,
                fold,
                folds,
                len(fit_pos),
                len(heldout_pos),
            )
            model = Pipeline(
                [
                    ("tfidf", self._make_vectorizer()),
                    ("model", self._make_regressor(random_state=17 + fold)),
                ]
            )
            model.fit([texts[i] for i in fit_pos], values[fit_pos])
            return heldout_pos, model.predict([texts[i] for i in heldout_pos])

        results = self._run_fold_tasks(run_fold, split_items)
        for heldout_pos, fold_values in results:
            oof[heldout_pos] = fold_values
        return oof

    def _crossfit_pseudo_target(
        self,
        texts: Sequence[str],
        pseudo_target: np.ndarray,
        outer_fold: int,
    ) -> np.ndarray:
        oof = np.full(len(pseudo_target), np.nan, dtype=float)
        folds = _bounded_fold_count(self.nn_config.effect_folds, len(pseudo_target))
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=13_000 + outer_fold,
        )
        split_items = list(enumerate(splitter.split(texts), start=1))

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            logger.info(
                "Outer fold %s BoW pseudo-target fold %s/%s: train=%s heldout=%s",
                outer_fold,
                fold,
                folds,
                len(fit_pos),
                len(heldout_pos),
            )
            model = Pipeline(
                [
                    ("tfidf", self._make_vectorizer()),
                    ("model", self._make_regressor(random_state=17 + fold)),
                ]
            )
            model.fit([texts[i] for i in fit_pos], pseudo_target[fit_pos])
            return heldout_pos, model.predict([texts[i] for i in heldout_pos])

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
    ) -> Dict[str, Any]:
        vectorizer = self._make_vectorizer()
        x_text = vectorizer.fit_transform(texts)
        features = np.asarray(vectorizer.get_feature_names_out())

        if len(np.unique(t.astype(int))) < 2:
            treatment_coef = np.zeros(len(features), dtype=float)
        else:
            treatment_model = self._make_classifier(random_state=101)
            treatment_model.fit(x_text, t.astype(int))
            treatment_coef = _model_feature_scores(treatment_model, len(features))

        if self.config.outcome_type == "continuous":
            outcome_model = self._make_regressor(random_state=202)
            outcome_model.fit(x_text, y)
            outcome_coef = _model_feature_scores(outcome_model, len(features))
        else:
            if len(np.unique(y.astype(int))) < 2:
                outcome_coef = np.zeros(len(features), dtype=float)
            else:
                outcome_model = self._make_classifier(random_state=202)
                outcome_model.fit(x_text, y.astype(int))
                outcome_coef = _model_feature_scores(outcome_model, len(features))

        effect_model = self._make_regressor(random_state=303)
        effect_model.fit(x_text, pseudo_target)
        effect_coef = _model_feature_scores(effect_model, len(features))

        top_n = int(self.nn_config.top_n_features)
        confounder_score = np.abs(treatment_coef) * np.abs(outcome_coef)
        return {
            "n_features": int(len(features)),
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
    ) -> Dict[str, Any]:
        return {
            "prompt_version": "non_neural_agentic_forest_v1",
            "outer_fold": int(outer_fold),
            "max_proposals": int(self.nn_config.candidate_proposals_per_fold),
            "clinical_question": self.config.clinical_question,
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "instructions": [
                "Review sparse bag-of-words feature weights from honest nuisance models and an unweighted R pseudo-target model.",
                "Suggest explicit pre-treatment patient-level variables, not raw text tokens.",
                "Use variables predictive of both treatment and outcome as confounders.",
                "Use variables predictive of the pseudo-target as effect modifiers.",
                "Avoid near-duplicate aliases for the same extraction target; a separate alias-resolution pass may merge proposal names.",
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
            "Non-neural candidate consistency parallelism: outer_fold=%s "
            "inner_folds=%s n_jobs=%s setting=%s",
            outer_fold,
            len(split_items),
            n_jobs,
            self.nn_config.candidate_consistency_parallelism,
        )
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self._build_inner_consistency_candidate_bundle_isolated)(
                int(outer_fold),
                discovery_df,
                int(inner_fold),
                fit_pos,
                heldout_pos,
                int(fold_count),
                int(n_jobs),
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
        worker = NonNeuralAgenticForestRunner(
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
                "Skipping non-neural candidate consistency inner fold %s/%s "
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
            "prompt_version": "non_neural_agentic_consistency_v1",
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
        allowed_names = {str(item.get("name")) for item in candidate_summaries}
        fallback = _fallback_consistency_proposals(
            candidate_summaries,
            canonical_proposals,
        )
        try:
            raw_selection = self.proposal_agent.propose(context)
            selection_trace = _get_agent_response_trace(self.proposal_agent)
            selected, rejected = validate_agentic_proposals(
                raw_selection,
                current_specs=self._initial_specs(),
                search_config=self.search_config,
                allow_removals=False,
                max_additions=self.nn_config.candidate_proposals_per_fold,
            )
            filtered = [
                proposal
                for proposal in selected
                if proposal.action == "add" and proposal.name in allowed_names
            ]
            rejected.extend(
                {
                    "proposal": _proposal_to_dict(proposal),
                    "reason": "not_in_consistency_candidates",
                }
                for proposal in selected
                if proposal.action == "add" and proposal.name not in allowed_names
            )
            if not filtered:
                filtered = fallback
                used_fallback = True
            else:
                filtered = [
                    _merge_proposals(canonical_proposals.get(p.name, p), p)
                    for p in filtered
                ]
                used_fallback = False
            result: Dict[str, Any] = {
                "raw_selection": raw_selection,
                "valid_proposals": [_proposal_to_dict(p) for p in filtered],
                "rejected_proposals": rejected,
                "used_fallback": used_fallback,
            }
            if self.search_config.save_agent_raw_output:
                result["agent_raw_output"] = selection_trace
            return filtered, result
        except Exception as exc:
            logger.warning(
                "Non-neural candidate consistency selection failed; using gate fallback",
                exc_info=True,
            )
            return fallback, {
                "error": str(exc),
                "valid_proposals": [_proposal_to_dict(p) for p in fallback],
                "used_fallback": True,
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
            "prompt_version": "non_neural_agentic_alias_resolution_v1",
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
                "Non-neural alias resolution failed; using unmerged proposal names",
                exc_info=True,
            )
            return proposals, {"error": str(exc), "applied_aliases": []}

        resolved, applied_aliases = _apply_alias_resolution(
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
            "prompt_version": "non_neural_agentic_value_harmonization_v1",
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
                "Non-neural value harmonization failed; using unharmonized specs",
                exc_info=True,
            )
            return selected_specs, {"error": str(exc), "applied": []}

        harmonized, applied = _apply_value_harmonization(
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
            logger.info("Dropped low-coverage non-neural agentic features: %s", dropped)
            self.agent_rows.append({"event": "coverage_filter", "dropped": dropped})
        return kept

    def _initial_specs(self) -> List[ExplicitFeatureSpec]:
        if getattr(self.config.explicit_features, "features", None):
            return _dedupe_specs(list(self.config.explicit_features.features))
        return []

    def _make_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)[a-z0-9%<>+=-]+",
            ngram_range=(
                int(self.nn_config.ngram_range_min),
                int(self.nn_config.ngram_range_max),
            ),
            min_df=int(self.nn_config.min_df),
            max_df=float(self.nn_config.max_df),
            sublinear_tf=bool(self.nn_config.sublinear_tf),
            max_features=int(self.nn_config.max_features),
            dtype=np.float32,
        )

    def _make_classifier(self, random_state: int = 17):
        model_name = str(self.nn_config.bow_model).strip().lower()
        if model_name == "linear":
            return self._make_logistic_regression(random_state=random_state)
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

    def _make_regressor(self, random_state: int = 17):
        model_name = str(self.nn_config.bow_model).strip().lower()
        if model_name == "linear":
            return self._make_ridge()
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

    def _make_logistic_regression(self, random_state: int = 17) -> LogisticRegression:
        return LogisticRegression(
            C=float(self.nn_config.logistic_c),
            solver="liblinear",
            max_iter=int(self.nn_config.logistic_max_iter),
            random_state=random_state,
        )

    def _make_ridge(self) -> Ridge:
        return Ridge(alpha=float(self.nn_config.ridge_alpha), random_state=17)

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

    def _run_fold_tasks(self, run_fold: Any, split_items: Sequence[Any]) -> List[Any]:
        n_jobs = self._fold_n_jobs(len(split_items))
        if n_jobs <= 1:
            return [
                run_fold(int(fold), np.asarray(fit_pos), np.asarray(heldout_pos))
                for fold, (fit_pos, heldout_pos) in split_items
            ]
        logger.info(
            "Non-neural BoW cross-fit parallelism: folds=%s n_jobs=%s setting=%s",
            len(split_items),
            n_jobs,
            self.nn_config.fold_parallelism,
        )
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(run_fold)(int(fold), np.asarray(fit_pos), np.asarray(heldout_pos))
            for fold, (fit_pos, heldout_pos) in split_items
        )

    def _save_predictions(self, results_df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(self.output_path, index=False)
        logger.info("Non-neural agentic forest predictions saved to: %s", self.output_path)

    def _save_artifacts(self) -> None:
        if self.bow_prediction_frames:
            pd.concat(self.bow_prediction_frames).to_parquet(
                self.artifact_dir / "bow_oof_predictions.parquet",
                index=False,
            )
        pd.DataFrame(self.outer_metric_rows).to_csv(
            self.artifact_dir / "outer_cv_metrics.csv",
            index=False,
        )
        _write_jsonl(self.artifact_dir / "bow_feature_importance_by_fold.jsonl", self.importance_rows)
        _write_jsonl(self.artifact_dir / "agent_candidate_proposals.jsonl", self.agent_rows)
        with open(self.artifact_dir / "selected_feature_sets.json", "w") as f:
            json.dump(self.feature_set_rows, f, indent=2, default=_json_default)
        logger.info("Non-neural agentic forest artifacts saved to: %s", self.artifact_dir)


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
        for item in candidate_summaries
        if item.get("passes_consistency_gate") and item.get("name") in canonical_proposals
    ]
    if selected:
        return selected
    full_supported = [
        canonical_proposals[item["name"]]
        for item in candidate_summaries
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


_MISSING_VALUE_LABELS = {
    "",
    "na",
    "n/a",
    "null",
    "missing",
    "unknown",
    "unk",
    "unclear",
    "unavailable",
    "not_available",
    "not available",
    "not_reported",
    "not reported",
    "not_documented",
    "not documented",
    "not_assessed",
    "not assessed",
    "not_tested",
    "not tested",
    "not_applicable",
    "not applicable",
    "indeterminate",
}


def _apply_value_harmonization(
    *,
    specs: Sequence[ExplicitFeatureSpec],
    response: Any,
) -> Tuple[List[ExplicitFeatureSpec], List[Dict[str, Any]]]:
    if not isinstance(response, dict):
        return list(specs), [{"error": "response_not_object"}]
    raw_features = response.get("features")
    if not isinstance(raw_features, list):
        return list(specs), [{"error": "missing_features_list"}]

    by_name = {spec.name: spec for spec in specs}
    harmonized = {spec.name: spec for spec in specs}
    applied: List[Dict[str, Any]] = []
    for item in raw_features:
        if not isinstance(item, dict):
            applied.append({"ignored": item, "reason": "feature_not_object"})
            continue
        name = _normalize_feature_name(item.get("name", ""))
        spec = by_name.get(name)
        if spec is None:
            applied.append({"name": name, "reason": "unknown_feature"})
            continue

        feature_type = str(item.get("type") or spec.type).strip().lower()
        if feature_type not in {"categorical", "continuous"}:
            feature_type = spec.type

        description = str(item.get("description") or spec.description or "").strip()
        if feature_type == "categorical":
            categories = _canonical_value_categories(
                item.get("categories") or spec.categories
            )
            if not categories:
                applied.append({"name": name, "reason": "empty_categorical_categories"})
                continue
            value_aliases = _canonical_value_aliases(
                item.get("value_aliases"),
                categories,
            )
            description = _append_value_policy(
                description,
                categories=categories,
                value_aliases=value_aliases,
                continuous=False,
            )
        else:
            categories = None
            value_aliases = None
            description = _append_value_policy(
                description,
                categories=None,
                value_aliases=None,
                continuous=True,
            )

        new_spec = ExplicitFeatureSpec(
            name=spec.name,
            type=feature_type,
            categories=categories,
            description=description or spec.description,
            value_aliases=value_aliases,
            roles=spec.roles,
        )
        harmonized[spec.name] = new_spec
        applied.append(
            {
                "name": spec.name,
                "from": _spec_to_dict(spec),
                "to": _spec_to_dict(new_spec),
                "rationale": item.get("rationale"),
            }
        )

    return _dedupe_specs(list(harmonized.values())), applied


def _canonical_value_categories(value: Any) -> Optional[List[str]]:
    categories: List[str] = []
    seen = set()
    for raw in _as_list(value):
        text = _canonical_category_text(raw)
        if _is_missing_value_label(text):
            continue
        key = _category_equivalence_key(text)
        if not text or key in seen:
            continue
        seen.add(key)
        categories.append(text)
    return categories or None


def _canonical_value_aliases(
    value_aliases: Any,
    categories: Sequence[str],
) -> Optional[Dict[str, List[str]]]:
    if not isinstance(value_aliases, dict):
        return None
    by_key = {_category_equivalence_key(category): category for category in categories}
    result: Dict[str, List[str]] = {}
    for raw_key, raw_aliases in value_aliases.items():
        category = by_key.get(_category_equivalence_key(raw_key))
        if category is None:
            continue
        aliases = []
        for alias in _as_list(raw_aliases):
            text = _canonical_category_text(alias)
            if text and not _is_missing_value_label(text):
                aliases.append(text)
        if aliases:
            result[category] = list(dict.fromkeys(aliases))
    return result or None


def _canonical_category_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).translate(_DASH_TRANSLATION)
    text = text.strip().replace("\u2265", ">=").replace("\u2264", "<=")
    text = " ".join(text.split())
    return text


def _category_equivalence_key(value: Any) -> str:
    text = _canonical_category_text(value).lower()
    text = text.replace("_", " ").replace(" ", "")
    return text


def _is_missing_value_label(value: Any) -> bool:
    key = _canonical_category_text(value).lower().replace("-", "_")
    key = " ".join(key.split())
    return key in _MISSING_VALUE_LABELS or key.replace(" ", "_") in _MISSING_VALUE_LABELS


def _append_value_policy(
    description: str,
    *,
    categories: Optional[Sequence[str]],
    value_aliases: Any,
    continuous: bool,
) -> str:
    base = description.strip()
    policy_parts: List[str] = []
    if continuous:
        policy_parts.append(
            "Value policy: return a numeric value only; return null for unknown, "
            "not reported, not assessed, unavailable, or qualitative-only values "
            "such as high/low without a numeric value."
        )
    else:
        cats = ", ".join(str(cat) for cat in categories or [])
        policy_parts.append(
            f"Value policy: return exactly one of [{cats}]; return null for "
            "unknown, not reported, not assessed, not tested, or unavailable values."
        )
        alias_text = _format_value_aliases(value_aliases, categories or [])
        if alias_text:
            policy_parts.append(f"Map value aliases as follows: {alias_text}.")

    policy = " ".join(policy_parts)
    if not base:
        return policy
    if "Value policy:" in base:
        return base
    return f"{base} {policy}"


def _format_value_aliases(value_aliases: Any, categories: Sequence[str]) -> str:
    if not isinstance(value_aliases, dict):
        return ""
    allowed = set(categories)
    chunks: List[str] = []
    for category, raw_aliases in value_aliases.items():
        if category not in allowed:
            continue
        aliases = [
            _canonical_category_text(alias)
            for alias in _as_list(raw_aliases)
            if _canonical_category_text(alias)
            and not _is_missing_value_label(alias)
        ]
        if aliases:
            chunks.append(f"{category}: {', '.join(dict.fromkeys(aliases))}")
    return "; ".join(chunks)


def _apply_alias_resolution(
    *,
    proposals: Sequence[AgenticFeatureProposal],
    known_specs: Sequence[ExplicitFeatureSpec],
    response: Any,
) -> Tuple[List[AgenticFeatureProposal], List[Dict[str, str]]]:
    if not isinstance(response, dict):
        return list(proposals), []

    proposal_by_name = {
        proposal.name: proposal for proposal in proposals if proposal.action == "add"
    }
    known_by_name = {spec.name: spec for spec in known_specs}
    allowed_names = set(proposal_by_name) | set(known_by_name)
    alias_to_canonical: Dict[str, str] = {}
    applied_aliases: List[Dict[str, str]] = []

    groups = response.get("groups") or []
    if not isinstance(groups, list):
        return list(proposals), []

    for group in groups:
        if not isinstance(group, dict):
            continue
        canonical = _normalize_feature_name(group.get("canonical_name", ""))
        raw_members = group.get("member_names") or []
        if not isinstance(raw_members, list):
            continue
        members = [
            _normalize_feature_name(member)
            for member in raw_members
            if _normalize_feature_name(member) in allowed_names
        ]
        if canonical not in allowed_names:
            continue
        if canonical not in members and canonical not in known_by_name:
            continue
        if len(set([canonical, *members])) < 2:
            continue
        for member in members:
            if member in proposal_by_name and member != canonical:
                alias_to_canonical[member] = canonical
                applied_aliases.append(
                    {
                        "from": member,
                        "to": canonical,
                        "rationale": str(group.get("rationale") or ""),
                    }
                )

    if not alias_to_canonical:
        return list(proposals), []

    rewritten: List[AgenticFeatureProposal] = []
    emitted_add_names: set = set()
    for proposal in proposals:
        if proposal.action != "add":
            rewritten.append(proposal)
            continue
        target_name = alias_to_canonical.get(proposal.name, proposal.name)
        known_spec = known_by_name.get(target_name)
        retargeted = _retarget_proposal(
            proposal=proposal,
            target_name=target_name,
            known_spec=known_spec,
        )
        if target_name in emitted_add_names:
            for index, existing in enumerate(rewritten):
                if existing.action == "add" and existing.name == target_name:
                    rewritten[index] = _merge_proposals(existing, retargeted)
                    break
            continue
        rewritten.append(retargeted)
        emitted_add_names.add(target_name)

    return rewritten, applied_aliases


def _retarget_proposal(
    *,
    proposal: AgenticFeatureProposal,
    target_name: str,
    known_spec: Optional[ExplicitFeatureSpec],
) -> AgenticFeatureProposal:
    if known_spec is None:
        return AgenticFeatureProposal(
            action=proposal.action,
            name=target_name,
            type=proposal.type,
            categories=proposal.categories,
            description=proposal.description,
            roles=proposal.roles,
            rationale=proposal.rationale,
            expected_signal=proposal.expected_signal,
        )

    categories = _merge_ordered_values(known_spec.categories, proposal.categories) or None
    feature_type = (
        "categorical"
        if known_spec.type == "categorical" or proposal.type == "categorical" or categories
        else known_spec.type
    )
    return AgenticFeatureProposal(
        action=proposal.action,
        name=target_name,
        type=feature_type,
        categories=categories,
        description=_merge_text_values(known_spec.description, proposal.description),
        roles=_merge_ordered_values(known_spec.roles, proposal.roles),
        rationale=proposal.rationale,
        expected_signal=proposal.expected_signal,
    )


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
