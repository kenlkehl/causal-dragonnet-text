"""Non-neural BoW-guided agentic variable discovery plus causal forest."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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
    del device, num_workers
    runner = NonNeuralAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
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

    def run(self) -> None:
        logger.info("=" * 80)
        logger.info("NON-NEURAL AGENTIC FEATURE CAUSAL FOREST")
        logger.info("=" * 80)

        prediction_frames: List[pd.DataFrame] = []
        for outer_fold, train_idx, test_idx in self._analysis_splits():
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
        split_items = _binary_split_items(
            labels,
            requested_folds=self.nn_config.nuisance_folds,
            random_state=random_state,
        )
        folds = len(split_items)
        for fold, (fit_pos, heldout_pos) in enumerate(split_items, start=1):
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
                oof[heldout_pos] = float(np.mean(labels[fit_pos]))
                continue
            model = Pipeline(
                [
                    ("tfidf", self._make_vectorizer()),
                    ("logreg", self._make_logistic_regression()),
                ]
            )
            model.fit([texts[i] for i in fit_pos], labels[fit_pos])
            oof[heldout_pos] = model.predict_proba([texts[i] for i in heldout_pos])[:, 1]
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
        for fold, (fit_pos, heldout_pos) in enumerate(splitter.split(texts), start=1):
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
                    ("ridge", self._make_ridge()),
                ]
            )
            model.fit([texts[i] for i in fit_pos], values[fit_pos])
            oof[heldout_pos] = model.predict([texts[i] for i in heldout_pos])
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
        for fold, (fit_pos, heldout_pos) in enumerate(splitter.split(texts), start=1):
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
                    ("ridge", self._make_ridge()),
                ]
            )
            model.fit([texts[i] for i in fit_pos], pseudo_target[fit_pos])
            oof[heldout_pos] = model.predict([texts[i] for i in heldout_pos])
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
            treatment_model = self._make_logistic_regression()
            treatment_model.fit(x_text, t.astype(int))
            treatment_coef = treatment_model.coef_.ravel().astype(float)

        if self.config.outcome_type == "continuous":
            outcome_model = self._make_ridge()
            outcome_model.fit(x_text, y)
            outcome_coef = outcome_model.coef_.ravel().astype(float)
        else:
            if len(np.unique(y.astype(int))) < 2:
                outcome_coef = np.zeros(len(features), dtype=float)
            else:
                outcome_model = self._make_logistic_regression()
                outcome_model.fit(x_text, y.astype(int))
                outcome_coef = outcome_model.coef_.ravel().astype(float)

        effect_model = self._make_ridge()
        effect_model.fit(x_text, pseudo_target)
        effect_coef = effect_model.coef_.ravel().astype(float)

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
            ],
            "current_features": [_spec_to_dict(spec) for spec in self._initial_specs()],
            "model_diagnostics": metrics,
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
        del discovery_df
        raw_proposals = self.proposal_agent.propose(bow_context)
        proposals, rejected = validate_agentic_proposals(
            raw_proposals,
            current_specs=self._initial_specs(),
            search_config=self.search_config,
            allow_removals=False,
            max_additions=self.nn_config.candidate_proposals_per_fold,
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
        row: Dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "raw_proposals": raw_proposals,
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
            row["agent_raw_output"] = _get_agent_response_trace(self.proposal_agent)
        self.agent_rows.append(row)
        return selected_specs

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
            return list(self.config.explicit_features.features)
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

    def _make_logistic_regression(self) -> LogisticRegression:
        return LogisticRegression(
            C=float(self.nn_config.logistic_c),
            solver="liblinear",
            max_iter=int(self.nn_config.logistic_max_iter),
            random_state=17,
        )

    def _make_ridge(self) -> Ridge:
        return Ridge(alpha=float(self.nn_config.ridge_alpha), random_state=17)

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
    text = re.sub(r"pd\s*-\s*l1", "pd-l1", text, flags=re.IGNORECASE)
    return text.lower()


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
            row["treatment_coef"] = _finite_or_none(treatment_coef[idx])
            row["abs_treatment_coef"] = _finite_or_none(abs(treatment_coef[idx]))
        if outcome_coef is not None:
            row["outcome_coef"] = _finite_or_none(outcome_coef[idx])
            row["abs_outcome_coef"] = _finite_or_none(abs(outcome_coef[idx]))
        rows.append(row)
    return rows


def _dedupe_specs(specs: Sequence[ExplicitFeatureSpec]) -> List[ExplicitFeatureSpec]:
    by_name: Dict[str, ExplicitFeatureSpec] = {}
    for spec in specs:
        name = _normalize_feature_name(spec.name)
        if not name:
            continue
        if name not in by_name:
            by_name[name] = spec
            continue
        existing = by_name[name]
        roles = list(dict.fromkeys([*existing.roles, *spec.roles]))
        by_name[name] = ExplicitFeatureSpec(
            name=existing.name,
            type=existing.type,
            categories=existing.categories,
            description=existing.description or spec.description,
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
