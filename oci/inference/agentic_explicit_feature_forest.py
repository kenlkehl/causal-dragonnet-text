"""Agentic explicit-feature causal forest search.

This module runs an adaptive, LLM-guided variable search around the existing
explicit-feature causal forest. The reported performance comes from outer CV;
all feature-set decisions are made with inner CV on each outer-training split.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from ..config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
)
from ..extraction import ExtractionCache, VLLMFeatureExtractor
from ..models.causal_forest_head import CausalForestHead
from .applied_explicit_feature_forest import _build_features, _hstack_present


logger = logging.getLogger(__name__)

AGENT_PROMPT_VERSION = "agentic_explicit_feature_search_v1"
BROAD_AGENT_PROMPT_VERSION = "agentic_explicit_feature_broad_screen_v1"
EXTRACTION_PROMPT_VERSION = "explicit_features_v2"
VALID_ACTIONS = {"add", "remove", "update_role", "none"}
VALID_ROLES = {"confounder", "effect_modifier"}
VALID_TYPES = {"categorical", "continuous"}


@dataclass
class AgenticFeatureProposal:
    """Validated proposal emitted by the feature-search agent."""

    action: str
    name: str
    type: Optional[str] = None
    categories: Optional[List[str]] = None
    description: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    rationale: Optional[str] = None
    expected_signal: Optional[str] = None


@dataclass
class SplitEvaluation:
    """Predictions and metrics for one train/test split."""

    predictions: pd.DataFrame
    metrics: Dict[str, Any]


def run_agentic_explicit_feature_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device=None,
    num_workers: int = 1,
    proposal_agent: Optional[Any] = None,
    extraction_provider: Optional[Any] = None,
    evaluator: Optional[Any] = None,
) -> None:
    """Run nested-CV agentic explicit-feature causal forest inference."""
    del device, num_workers
    runner = AgenticFeatureSearchRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        proposal_agent=proposal_agent,
        extraction_provider=extraction_provider,
        evaluator=evaluator,
    )
    runner.run()


class AgenticFeatureSearchRunner:
    """Nested-CV runner for adaptive explicit-feature search."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        config: AppliedInferenceConfig,
        output_path: Path,
        proposal_agent: Optional[Any] = None,
        extraction_provider: Optional[Any] = None,
        evaluator: Optional[Any] = None,
    ):
        self.dataset = dataset.reset_index(drop=True).copy()
        self.config = config
        self.output_path = Path(output_path)
        self.artifact_dir = self.output_path.parent / "agentic_feature_search"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.search_config = getattr(
            config.architecture,
            "agentic_feature_search",
            AgenticFeatureSearchConfig(),
        )
        self.cf_config = getattr(
            config.architecture,
            "explicit_feature_forest",
            ExplicitFeatureForestConfig(),
        )
        self.initial_specs = (
            list(config.explicit_features.features)
            if getattr(config.explicit_features, "enabled", False)
            else []
        )
        if config.explicit_features.features and not config.explicit_features.enabled:
            logger.info(
                "Ignoring configured explicit_features.features because "
                "explicit_features.enabled=False"
            )
        if not self.initial_specs:
            logger.info(
                "Agentic explicit-feature search is starting from an empty feature set"
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

        self.decision_events: List[Dict[str, Any]] = []
        self.inner_metric_rows: List[Dict[str, Any]] = []
        self.outer_metric_rows: List[Dict[str, Any]] = []
        self.feature_set_rows: List[Dict[str, Any]] = []
        self.screening_metric_rows: List[Dict[str, Any]] = []

    def run(self) -> None:
        """Execute outer CV, inner adaptive search, and final reporting."""
        logger.info("=" * 80)
        logger.info("AGENTIC EXPLICIT FEATURE CAUSAL FOREST")
        logger.info("=" * 80)

        # Ensure initial variables are available before the first inner search.
        self.dataset = self.extraction_provider.ensure_features(self.dataset, self.initial_specs)

        outer_splits = _make_splits(
            self.dataset,
            self.config,
            n_splits=self.search_config.outer_folds,
            random_state=self.search_config.random_state,
        )

        outer_predictions = []
        for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            logger.info(
                "Outer fold %s/%s: train=%s test=%s",
                outer_fold,
                len(outer_splits),
                len(train_idx),
                len(test_idx),
            )
            selected_specs = self._search_outer_train(outer_fold, train_idx)
            self.dataset = self.extraction_provider.ensure_features(self.dataset, selected_specs)

            train_df = self.dataset.iloc[train_idx].copy()
            test_df = self.dataset.iloc[test_idx].copy()
            final_eval = self.evaluator.evaluate_split(
                train_df=train_df,
                test_df=test_df,
                specs=selected_specs,
                fold_id=outer_fold,
            )
            preds = final_eval.predictions.copy()
            preds["outer_fold"] = outer_fold
            preds["selected_feature_names"] = ",".join(spec.name for spec in selected_specs)
            outer_predictions.append(preds)

            metrics = {
                "outer_fold": outer_fold,
                "stage": "outer_final",
                "n_selected_features": len(selected_specs),
                **_without_list_values(final_eval.metrics),
            }
            self.outer_metric_rows.append(metrics)
            self.feature_set_rows.append(
                {
                    "outer_fold": outer_fold,
                    "stage": "selected",
                    "features": [_spec_to_dict(spec) for spec in selected_specs],
                }
            )

        results_df = pd.concat(outer_predictions).sort_index()
        self._save_predictions(results_df)
        self._save_artifacts()

    def _search_outer_train(
        self,
        outer_fold: int,
        outer_train_idx: np.ndarray,
    ) -> List[ExplicitFeatureSpec]:
        """Run the configured feature search for one outer-training split."""
        if self.search_config.search_mode == "broad_screen":
            return self._search_outer_train_broad_screen(outer_fold, outer_train_idx)
        return self._search_outer_train_iterative(outer_fold, outer_train_idx)

    def _search_outer_train_iterative(
        self,
        outer_fold: int,
        outer_train_idx: np.ndarray,
    ) -> List[ExplicitFeatureSpec]:
        """Run the inner adaptive search for one outer-training split."""
        current_specs = list(self.initial_specs)
        accepted_additions = 0

        baseline_rows, baseline_summary = self._evaluate_inner_cv(
            outer_fold=outer_fold,
            iteration=0,
            candidate_name="initial",
            train_idx=outer_train_idx,
            specs=current_specs,
        )
        self._record_inner_rows(baseline_rows, accepted=True)

        for iteration in range(1, self.search_config.max_iterations + 1):
            context = self._build_agent_context(
                outer_fold=outer_fold,
                iteration=iteration,
                train_idx=outer_train_idx,
                current_specs=current_specs,
                current_summary=baseline_summary,
            )
            try:
                raw_proposals = self.proposal_agent.propose(context)
            except Exception as exc:
                error_payload = {
                    "error": repr(exc),
                    "context": context,
                }
                if self.search_config.save_agent_raw_output:
                    error_payload["agent_raw_output"] = _get_agent_response_trace(
                        self.proposal_agent
                    )
                self._record_decision(
                    outer_fold,
                    iteration,
                    "agent_proposal_error",
                    error_payload,
                )
                self._save_decision_events()
                raise
            proposal_payload = {
                "raw_count": len(raw_proposals),
                "valid_count": None,
                "rejected": None,
                "context": context,
                "raw_proposals": raw_proposals,
            }
            if self.search_config.save_agent_raw_output:
                proposal_payload["agent_raw_output"] = _get_agent_response_trace(
                    self.proposal_agent
                )
            proposals, rejected = validate_agentic_proposals(
                raw_proposals,
                current_specs=current_specs,
                search_config=self.search_config,
                allow_removals=accepted_additions > 0,
            )
            proposal_payload["valid_count"] = len(proposals)
            proposal_payload["rejected"] = rejected
            self._record_decision(
                outer_fold,
                iteration,
                "agent_proposals",
                proposal_payload,
            )

            if not proposals:
                logger.info("Outer fold %s iteration %s: no valid proposals", outer_fold, iteration)
                break

            candidate_results = []
            for candidate_id, proposal_group in _candidate_groups(proposals):
                candidate_specs = apply_proposals(current_specs, proposal_group)
                if _spec_names(candidate_specs) == _spec_names(current_specs):
                    continue
                self.dataset = self.extraction_provider.ensure_features(
                    self.dataset,
                    candidate_specs,
                )
                proposal_specs = _candidate_proposal_specs(
                    current_specs=current_specs,
                    candidate_specs=candidate_specs,
                    proposal_group=proposal_group,
                )
                role_diagnostics = evaluate_candidate_role_diagnostics(
                    dataset=self.dataset.iloc[outer_train_idx],
                    current_specs=current_specs,
                    candidate_specs=proposal_specs,
                    config=self.config,
                    search_config=self.search_config,
                )
                coverage_failures = _coverage_failures(
                    self.dataset.iloc[outer_train_idx],
                    proposal_specs,
                    self.search_config.min_feature_coverage,
                )
                if coverage_failures:
                    summary = {"coverage_failures": coverage_failures}
                    if role_diagnostics:
                        summary["role_diagnostics"] = role_diagnostics
                    candidate_results.append(
                        {
                            "candidate_id": candidate_id,
                            "proposal_group": proposal_group,
                            "specs": candidate_specs,
                            "rows": [],
                            "summary": summary,
                            "comparison": {
                                "passes_acceptance": False,
                                "rejection_reason": "low_feature_coverage",
                                "coverage_failures": coverage_failures,
                            },
                        }
                    )
                    continue
                rows, summary = self._evaluate_inner_cv(
                    outer_fold=outer_fold,
                    iteration=iteration,
                    candidate_name=candidate_id,
                    train_idx=outer_train_idx,
                    specs=candidate_specs,
                )
                if role_diagnostics:
                    summary = dict(summary)
                    summary["role_diagnostics"] = role_diagnostics
                comparison = compare_candidate_to_baseline(
                    baseline_rows=baseline_rows,
                    candidate_rows=rows,
                    search_config=self.search_config,
                )
                candidate_results.append(
                    {
                        "candidate_id": candidate_id,
                        "proposal_group": proposal_group,
                        "specs": candidate_specs,
                        "rows": rows,
                        "summary": summary,
                        "comparison": comparison,
                    }
                )
                self._record_inner_rows(rows, accepted=False)

            accepted = _choose_accepted_candidate(candidate_results)
            self._record_decision(
                outer_fold,
                iteration,
                "candidate_evaluations",
                [
                    {
                        "candidate_id": item["candidate_id"],
                        "proposals": [asdict(p) for p in item["proposal_group"]],
                        "summary": item["summary"],
                        "comparison": item["comparison"],
                        "accepted": accepted is item,
                    }
                    for item in candidate_results
                ],
            )

            if accepted is None:
                logger.info(
                    "Outer fold %s iteration %s: no candidate passed acceptance thresholds",
                    outer_fold,
                    iteration,
                )
                if self.search_config.stop_after_rejected_iteration:
                    break
                continue

            current_specs = accepted["specs"]
            baseline_rows = accepted["rows"]
            baseline_summary = accepted["summary"]
            accepted_additions += sum(
                1 for proposal in accepted["proposal_group"] if proposal.action == "add"
            )
            self._record_inner_rows(accepted["rows"], accepted=True)
            self.feature_set_rows.append(
                {
                    "outer_fold": outer_fold,
                    "iteration": iteration,
                    "stage": "accepted_inner",
                    "candidate_id": accepted["candidate_id"],
                    "features": [_spec_to_dict(spec) for spec in current_specs],
                }
            )
            logger.info(
                "Outer fold %s iteration %s: accepted %s",
                outer_fold,
                iteration,
                accepted["candidate_id"],
            )

        return current_specs

    def _search_outer_train_broad_screen(
        self,
        outer_fold: int,
        outer_train_idx: np.ndarray,
    ) -> List[ExplicitFeatureSpec]:
        """Run broad initial proposal, train-fold screening, and greedy CV refinement."""
        current_specs = list(self.initial_specs)

        baseline_rows, baseline_summary = self._evaluate_inner_cv(
            outer_fold=outer_fold,
            iteration=0,
            candidate_name="initial",
            train_idx=outer_train_idx,
            specs=current_specs,
        )
        self._record_inner_rows(baseline_rows, accepted=True)

        if self.search_config.agent_max_tokens < 8000:
            logger.warning(
                "broad_screen mode asks for up to %s proposals but agent_max_tokens=%s; "
                "consider increasing agent_max_tokens to reduce truncation risk",
                self.search_config.broad_candidate_count,
                self.search_config.agent_max_tokens,
            )

        context = self._build_agent_context(
            outer_fold=outer_fold,
            iteration=1,
            train_idx=outer_train_idx,
            current_specs=current_specs,
            current_summary=baseline_summary,
        )
        context["search_mode"] = "broad_screen"
        context["prompt_version"] = BROAD_AGENT_PROMPT_VERSION
        context["broad_candidate_count"] = self.search_config.broad_candidate_count

        try:
            raw_proposals = self.proposal_agent.propose(context)
        except Exception as exc:
            error_payload = {
                "error": repr(exc),
                "context": context,
            }
            if self.search_config.save_agent_raw_output:
                error_payload["agent_raw_output"] = _get_agent_response_trace(
                    self.proposal_agent
                )
            self._record_decision(
                outer_fold,
                1,
                "agent_proposal_error",
                error_payload,
            )
            self._save_decision_events()
            raise

        proposal_payload = {
            "raw_count": len(raw_proposals),
            "valid_count": None,
            "rejected": None,
            "context": context,
            "raw_proposals": raw_proposals,
        }
        if self.search_config.save_agent_raw_output:
            proposal_payload["agent_raw_output"] = _get_agent_response_trace(
                self.proposal_agent
            )
        proposals, rejected = validate_agentic_proposals(
            raw_proposals,
            current_specs=current_specs,
            search_config=self.search_config,
            allow_removals=False,
            max_additions=self.search_config.broad_candidate_count,
        )
        proposals, duplicate_rejected = _dedupe_proposals_by_name(proposals)
        rejected.extend(duplicate_rejected)
        proposal_payload["valid_count"] = len(proposals)
        proposal_payload["rejected"] = rejected
        self._record_decision(
            outer_fold,
            1,
            "agent_proposals",
            proposal_payload,
        )

        if not proposals:
            logger.info("Outer fold %s broad_screen: no valid proposals", outer_fold)
            return current_specs

        proposed_specs = [_proposal_to_spec(proposal) for proposal in proposals]
        self.dataset = self.extraction_provider.ensure_features(self.dataset, proposed_specs)
        screened = screen_agentic_candidate_specs(
            dataset=self.dataset.iloc[outer_train_idx],
            current_specs=current_specs,
            proposals=proposals,
            config=self.config,
            search_config=self.search_config,
        )
        kept = select_screened_candidates(
            screened,
            top_k=self.search_config.broad_screen_top_k,
        )
        kept_ids = {item["candidate_id"] for item in kept}
        for item in screened:
            item["kept_for_cv"] = item["candidate_id"] in kept_ids
            if not item["kept_for_cv"]:
                self.screening_metric_rows.append(
                    _screening_metric_row(outer_fold=outer_fold, iteration=1, item=item)
                )

        if not kept:
            self._record_decision(
                outer_fold,
                1,
                "broad_screening",
                [
                    _screening_decision_payload(item)
                    for item in sorted(screened, key=lambda row: row["rank"])
                ],
            )
            logger.info(
                "Outer fold %s broad_screen: no candidates passed screening",
                outer_fold,
            )
            return current_specs

        candidate_payloads = []
        for item in kept:
            screened_spec = item["screened_spec"]
            if screened_spec.name in _spec_names(current_specs):
                continue
            candidate_specs = _dedupe_feature_specs([*current_specs, screened_spec])
            self.dataset = self.extraction_provider.ensure_features(
                self.dataset,
                candidate_specs,
            )
            rows, summary = self._evaluate_inner_cv(
                outer_fold=outer_fold,
                iteration=1,
                candidate_name=screened_spec.name,
                train_idx=outer_train_idx,
                specs=candidate_specs,
            )
            summary = dict(summary)
            if item.get("role_diagnostics"):
                summary["role_diagnostics"] = item["role_diagnostics"]
            comparison = compare_candidate_to_baseline(
                baseline_rows=baseline_rows,
                candidate_rows=rows,
                search_config=self.search_config,
            )
            accepted = bool(comparison.get("passes_acceptance", False))
            self._record_inner_rows(rows, accepted=False)
            item["cv_comparison"] = comparison
            item["cv_accepted"] = accepted
            candidate_payloads.append(
                {
                    "candidate_id": item["candidate_id"],
                    "proposals": [
                        asdict(_screened_spec_to_proposal(screened_spec, item["proposal"]))
                    ],
                    "summary": summary,
                    "comparison": comparison,
                    "accepted": accepted,
                    "screening": _screening_decision_payload(item),
                }
            )
            self.screening_metric_rows.append(
                _screening_metric_row(outer_fold=outer_fold, iteration=1, item=item)
            )
            if not accepted:
                continue

            current_specs = candidate_specs
            baseline_rows = rows
            self._record_inner_rows(rows, accepted=True)
            self.feature_set_rows.append(
                {
                    "outer_fold": outer_fold,
                    "iteration": 1,
                    "stage": "accepted_inner",
                    "candidate_id": item["candidate_id"],
                    "features": [_spec_to_dict(spec) for spec in current_specs],
                }
            )
            logger.info(
                "Outer fold %s broad_screen: accepted %s",
                outer_fold,
                item["candidate_id"],
            )

        self._record_decision(
            outer_fold,
            1,
            "broad_screening",
            [
                _screening_decision_payload(item)
                for item in sorted(screened, key=lambda row: row["rank"])
            ],
        )
        self._record_decision(
            outer_fold,
            1,
            "candidate_evaluations",
            candidate_payloads,
        )
        return current_specs

    def _evaluate_inner_cv(
        self,
        outer_fold: int,
        iteration: int,
        candidate_name: str,
        train_idx: np.ndarray,
        specs: List[ExplicitFeatureSpec],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate a feature set with inner CV over the outer-training rows."""
        train_df = self.dataset.iloc[train_idx].reset_index(drop=False)
        splits = _make_splits(
            train_df,
            self.config,
            n_splits=self.search_config.inner_folds,
            random_state=self.search_config.random_state + 1000 * outer_fold + iteration,
        )

        rows = []
        for inner_fold, (inner_train_pos, inner_val_pos) in enumerate(splits, start=1):
            inner_train = train_df.iloc[inner_train_pos].set_index("index", drop=True)
            inner_val = train_df.iloc[inner_val_pos].set_index("index", drop=True)
            split_eval = self.evaluator.evaluate_split(
                train_df=inner_train,
                test_df=inner_val,
                specs=specs,
                fold_id=inner_fold,
            )
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "iteration": iteration,
                    "candidate_name": candidate_name,
                    "inner_fold": inner_fold,
                    "feature_names": ",".join(spec.name for spec in specs),
                    **_without_list_values(split_eval.metrics),
                }
            )

        return rows, aggregate_metric_rows(rows)

    def _build_agent_context(
        self,
        outer_fold: int,
        iteration: int,
        train_idx: np.ndarray,
        current_specs: List[ExplicitFeatureSpec],
        current_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the train-only summary sent to the proposal agent."""
        train_only_df = self.dataset.iloc[train_idx]
        recent_decisions = [
            event for event in self.decision_events if event.get("outer_fold") == outer_fold
        ][-8:]
        return {
            "outer_fold": outer_fold,
            "iteration": iteration,
            "prompt_version": AGENT_PROMPT_VERSION,
            "clinical_question": _clinical_question_text(self.config),
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "current_features": [_spec_to_dict(spec) for spec in current_specs],
            "current_inner_cv_metrics": _non_oracle_metrics(current_summary),
            "extraction_summary": summarize_extractions(train_only_df, current_specs),
            "clinical_text_examples": _clinical_text_examples(
                train_only_df,
                self.config.text_column,
                n_examples=self.search_config.clinical_text_examples_per_prompt,
                max_chars=self.search_config.clinical_text_example_chars,
            ),
            "iteration_feedback": build_iteration_feedback(
                recent_decisions,
                self.search_config,
            ),
            "recent_decisions": recent_decisions,
        }

    def _record_inner_rows(self, rows: List[Dict[str, Any]], accepted: bool) -> None:
        for row in rows:
            copied = dict(row)
            copied["accepted_feature_set"] = bool(accepted)
            self.inner_metric_rows.append(copied)

    def _record_decision(
        self,
        outer_fold: int,
        iteration: int,
        event: str,
        payload: Any,
    ) -> None:
        payload = _scrub_decision_payload(
            payload,
            save_agent_context=self.search_config.save_agent_context,
        )
        self.decision_events.append(
            {
                "outer_fold": outer_fold,
                "iteration": iteration,
                "event": event,
                "payload": payload,
            }
        )

    def _save_predictions(self, results_df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(self.output_path, index=False)
        logger.info("Agentic predictions saved to: %s", self.output_path)

    def _save_artifacts(self) -> None:
        pd.DataFrame(self.inner_metric_rows).to_csv(
            self.artifact_dir / "inner_cv_metrics.csv",
            index=False,
        )
        pd.DataFrame(self.outer_metric_rows).to_csv(
            self.artifact_dir / "outer_cv_metrics.csv",
            index=False,
        )
        pd.DataFrame(self.screening_metric_rows).to_csv(
            self.artifact_dir / "screening_metrics.csv",
            index=False,
        )
        with open(self.artifact_dir / "feature_sets.json", "w") as f:
            json.dump(self.feature_set_rows, f, indent=2, default=_json_default)
        self._save_decision_events()
        logger.info("Agentic search artifacts saved to: %s", self.artifact_dir)

    def _save_decision_events(self) -> None:
        with open(self.artifact_dir / "agent_decisions.jsonl", "w") as f:
            for event in self.decision_events:
                f.write(json.dumps(event, default=_json_default) + "\n")


class OpenAICompatibleFeatureSearchAgent:
    """LLM proposal agent using an OpenAI-compatible chat completion endpoint."""

    def __init__(self, search_config: AgenticFeatureSearchConfig):
        self.search_config = search_config
        self._client = None
        self.last_raw_response: Optional[str] = None
        self.last_response_trace: Optional[Dict[str, Any]] = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for agentic feature proposals. "
                "Install the extraction extra or inject a custom proposal_agent."
            ) from exc
        self._client = OpenAI(
            base_url=self.search_config.agent_server_url,
            api_key=self.search_config.agent_api_key,
            max_retries=0,
        )

    def propose(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._ensure_client()
        self.last_raw_response = None
        self.last_response_trace = None
        prompt = build_agent_prompt(context, self.search_config)
        messages = [{"role": "user", "content": prompt}]
        attempts: List[Dict[str, Any]] = []
        max_repair_attempts = max(
            0,
            int(getattr(self.search_config, "agent_schema_repair_attempts", 1)),
        )

        for attempt_idx in range(max_repair_attempts + 1):
            response = self._client.chat.completions.create(
                model=self.search_config.agent_model_name,
                messages=messages,
                temperature=self.search_config.agent_temperature,
                max_tokens=self.search_config.agent_max_tokens,
            )
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            self.last_raw_response = content
            trace = _chat_completion_trace(
                response=response,
                choice=choice,
                message=message,
                content=content,
            )
            attempts.append(trace)
            self.last_response_trace = _trace_with_repair_attempts(trace, attempts)

            try:
                proposals = parse_agent_response(content)
            except Exception as exc:
                issues = [f"malformed JSON: {exc}"]
                if attempt_idx < max_repair_attempts:
                    messages.extend(
                        [
                            {"role": "assistant", "content": content},
                            {"role": "user", "content": build_agent_repair_prompt(issues)},
                        ]
                    )
                    continue
                raise ValueError(
                    "Agent response could not be parsed after "
                    f"{attempt_idx + 1} attempt(s): {issues[0]}"
                ) from exc

            issues = agent_response_schema_issues(proposals)
            if not issues:
                return proposals

            if attempt_idx < max_repair_attempts:
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": build_agent_repair_prompt(issues)},
                    ]
                )
                continue

            logger.warning(
                "Agent proposal response still has schema issues after %s attempt(s): %s",
                attempt_idx + 1,
                "; ".join(issues),
            )
            return proposals

        return []


class VLLMExplicitFeatureExtractionProvider:
    """Ensure requested explicit feature columns exist, one variable at a time."""

    def __init__(self, config: AppliedInferenceConfig, output_dir: Path):
        self.config = config
        self.feature_config = config.explicit_features
        self.output_dir = Path(output_dir)
        self.cache = ExtractionCache(
            cache_dir=self.feature_config.cache_dir or str(self.output_dir)
        )

    def ensure_features(
        self,
        dataset: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
    ) -> pd.DataFrame:
        dataset = dataset.copy()
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if value_col in dataset.columns and missing_col in dataset.columns:
                continue
            extracted_df = self._extract_one_spec(dataset, spec)
            for col in extracted_df.columns:
                dataset[col] = extracted_df[col].values
        return dataset

    def _extract_one_spec(self, dataset: pd.DataFrame, spec: ExplicitFeatureSpec) -> pd.DataFrame:
        cache_config = {
            "features": [spec],
            "prompt_template_version": EXTRACTION_PROMPT_VERSION,
            "vllm_model_name": self.feature_config.vllm_model_name,
            "vllm_max_model_len": self.feature_config.vllm_max_model_len,
            "extraction_temperature": self.feature_config.extraction_temperature,
            "extraction_max_tokens": self.feature_config.extraction_max_tokens,
            "extraction_max_text_length": self.feature_config.extraction_max_text_length,
        }
        cached = None
        if self.feature_config.cache_enabled:
            cached = self.cache.load_if_valid(
                self.config.dataset_path or "in_memory_dataset",
                cache_config,
                expected_rows=len(dataset),
            )
        if cached is not None:
            return cached

        logger.info("Extracting agentic feature with LLM: %s", spec.name)
        extractor = VLLMFeatureExtractor(
            specs=[spec],
            mode=self.feature_config.vllm_mode,
            server_url=self.feature_config.vllm_server_url or "http://localhost:8000/v1",
            model_name=self.feature_config.vllm_model_name,
            tensor_parallel_size=self.feature_config.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.feature_config.vllm_gpu_memory_utilization,
            download_dir=self.feature_config.vllm_download_dir,
            max_model_len=self.feature_config.vllm_max_model_len,
            max_retries=self.feature_config.extraction_max_retries,
            temperature=self.feature_config.extraction_temperature,
            max_tokens=self.feature_config.extraction_max_tokens,
            max_text_length=self.feature_config.extraction_max_text_length,
        )
        try:
            extracted_df = extractor.extract_to_dataframe(
                dataset[self.config.text_column].tolist(),
                batch_size=self.feature_config.extraction_batch_size,
            )
        finally:
            extractor.cleanup()

        if self.feature_config.cache_enabled:
            self.cache.save(
                self.config.dataset_path or "in_memory_dataset",
                cache_config,
                extracted_df,
            )
        return extracted_df


class CausalForestExplicitEvaluator:
    """Fit/evaluate one explicit-feature causal forest split."""

    def __init__(
        self,
        config: AppliedInferenceConfig,
        cf_config: ExplicitFeatureForestConfig,
    ):
        self.config = config
        self.cf_config = cf_config

    def evaluate_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
        fold_id: int,
    ) -> SplitEvaluation:
        train_T = np.asarray(train_df[self.config.treatment_column].values).flatten()
        train_Y = np.asarray(train_df[self.config.outcome_column].values).flatten()
        test_T = np.asarray(test_df[self.config.treatment_column].values).flatten()
        test_Y = np.asarray(test_df[self.config.outcome_column].values).flatten()

        X_train, W_train, x_names, w_names, means, stds = _build_features(train_df, specs)
        X_test, W_test, _, _, _, _ = _build_features(test_df, specs, means, stds)
        actual_x_dim = 0 if X_train is None else X_train.shape[1]
        if X_train is None:
            X_train = np.zeros((len(train_df), 1), dtype=np.float32)
            X_test = np.zeros((len(test_df), 1), dtype=np.float32)
            x_names = ["intercept_effect"]

        nuisance_train = _hstack_present(X_train, W_train)
        nuisance_test = _hstack_present(X_test, W_test)
        if nuisance_train is None or nuisance_test is None:
            raise ValueError("Unable to build explicit-feature nuisance matrices")

        forest = CausalForestHead(
            n_estimators=self.cf_config.n_estimators,
            max_depth=self.cf_config.max_depth,
            min_samples_leaf=self.cf_config.min_samples_leaf,
            max_features=self.cf_config.max_features,
            honest=self.cf_config.honest,
            inference=self.cf_config.inference,
            random_state=42 + fold_id,
        )
        forest.fit(X_train, train_T, train_Y, W=W_train)
        cf_preds = forest.predict(X_test, return_ci=True)
        tau = cf_preds["tau_pred"]

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

        y0_prob = outcome_pred - propensity * tau
        y1_prob = outcome_pred + (1.0 - propensity) * tau
        if self.config.outcome_type == "binary":
            y0_prob = np.clip(y0_prob, 0, 1)
            y1_prob = np.clip(y1_prob, 0, 1)

        predictions = test_df.copy()
        predictions["pred_ite_prob"] = tau
        predictions["pred_y0_prob"] = y0_prob
        predictions["pred_y1_prob"] = y1_prob
        predictions["pred_propensity_prob"] = propensity
        predictions["pred_outcome_prob"] = outcome_pred
        predictions["cv_fold"] = fold_id
        if "tau_lower" in cf_preds:
            predictions["pred_ite_lower"] = cf_preds["tau_lower"]
            predictions["pred_ite_upper"] = cf_preds["tau_upper"]

        metrics = {
            "fold": fold_id,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_explicit_features": len(specs),
            "n_x_features": actual_x_dim,
            "n_w_features": 0 if W_train is None else W_train.shape[1],
            "ate_estimate": float(np.mean(tau)),
            "r_loss": float(_r_loss(test_Y, test_T, outcome_pred, propensity, tau)),
            "treatment_auroc": _safe_roc_auc(test_T, propensity),
            "outcome_auroc": (
                _safe_roc_auc(test_Y, outcome_pred)
                if self.config.outcome_type == "binary"
                else None
            ),
            "x_feature_names": x_names,
            "w_feature_names": w_names,
        }
        if "true_ite_prob" in test_df.columns:
            metrics["oracle_true_ite_corr"] = _safe_corr(test_df["true_ite_prob"].values, tau)
            metrics["oracle_true_ite_mae"] = float(
                np.mean(np.abs(np.asarray(test_df["true_ite_prob"].values) - tau))
            )

        return SplitEvaluation(predictions=predictions, metrics=metrics)


def build_agent_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct the proposal prompt sent to the LLM agent."""
    if context.get("search_mode") == "broad_screen":
        return build_broad_agent_prompt(context, search_config)

    context_json = json.dumps(context, indent=2, default=_json_default)
    current_feature_count = len(context.get("current_features", []))
    feature_status = (
        "No variables are currently included; propose an initial variable to extract."
        if current_feature_count == 0
        else (
            "The current variables are already included; propose additions, "
            "removals, or role updates only when the nested-CV context and "
            "prior feedback justify them."
        )
    )
    return f"""You are helping design a causal inference feature set for a causal forest.

{feature_status}
Propose variables that are plausibly extractable from the text and could improve confounding adjustment or CATE heterogeneity.
Use the clinical question, estimand metadata, nested-CV context, extraction summaries, clinical text examples, and prior feedback to decide which variables are worth trying. Define each extraction target precisely enough that the downstream feature extractor can operationalize it.
Candidate feedback may include role_diagnostics from train-fold regressions adjusted for current confounders. Treat a candidate with both treatment and outcome association as a likely confounder. Treat a candidate with treatment-by-candidate interaction evidence in the outcome model as a likely effect modifier. Use those diagnostics to revise roles or extraction targets when a prior candidate was rejected.

Return JSON only with this shape:
{{
  "proposals": [
    {{
      "action": "add|remove|update_role|none",
      "name": "snake_case_variable_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "roles": ["confounder", "effect_modifier"],
      "description": "exact extraction target",
      "rationale": "why this may help",
      "expected_signal": "treatment, outcome, or tau signal expected"
    }}
  ]
}}

Limits:
- At most {search_config.max_additions_per_iter} add proposals.
- At most {search_config.max_removals_per_iter} remove proposals.
- Use "none" if no defensible variable is available.
- For categorical variables, provide 2-8 mutually exclusive categories.
- Review iteration_feedback and recent_decisions before proposing. Do not repeat
  a rejected feature unchanged; if revisiting a rejected concept, change the
  extraction target, type/categories, or role to directly address failed_checks.

Current nested-CV context:
{context_json}
"""


def build_broad_agent_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct a high-recall initial proposal prompt for broad-screen mode."""
    context_json = json.dumps(context, indent=2, default=_json_default)
    candidate_count = int(
        context.get("broad_candidate_count", search_config.broad_candidate_count)
    )
    return f"""You are helping design a high-recall baseline variable inventory for causal inference.

Propose a broad list of variables that are plausibly extractable from the text and could act as pre-treatment confounders or treatment effect modifiers.
The next stage will statistically screen every candidate for treatment association, outcome association, and treatment-by-candidate interaction, so prioritize recall and precise extraction definitions over narrow confidence.

Return JSON only with this shape:
{{
  "proposals": [
    {{
      "action": "add",
      "name": "snake_case_variable_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "roles": ["confounder", "effect_modifier"],
      "description": "exact pre-treatment extraction target",
      "rationale": "why this may help",
      "expected_signal": "treatment, outcome, or tau signal expected"
    }}
  ]
}}

Limits:
- Return up to {candidate_count} add proposals.
- Do not return remove, update_role, or none actions in this mode.
- Each variable must be measurable before or at treatment initiation.
- Do not propose treatment, post-treatment response, survival, toxicity, or outcome-derived variables.
- For categorical variables, provide 2-8 mutually exclusive categories.
- Use distinct variable names; avoid near-duplicate aliases for the same concept.

Current train-fold context:
{context_json}
"""


def build_agent_repair_prompt(issues: Sequence[str]) -> str:
    """Construct a follow-up prompt asking the agent to repair proposal JSON."""
    issue_lines = "\n".join(f"- {issue}" for issue in issues)
    return f"""The previous response could not be used because it failed these schema checks:
{issue_lines}

Return corrected JSON only. Use this exact top-level shape:
{{
  "proposals": [
    {{
      "action": "add|remove|update_role|none",
      "name": "snake_case_variable_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "roles": ["confounder", "effect_modifier"],
      "description": "exact extraction target",
      "rationale": "why this may help",
      "expected_signal": "treatment, outcome, or tau signal expected"
    }}
  ]
}}

Repair the same candidate concepts when possible. Do not add prose, markdown,
comments, or code fences. For every add proposal, include a non-empty roles
array containing only "confounder" and/or "effect_modifier".
"""


def agent_response_schema_issues(proposals: Sequence[Any]) -> List[str]:
    """Return schema-level issues that are worth asking the LLM to repair."""
    issues: List[str] = []
    for idx, raw in enumerate(proposals, start=1):
        if not isinstance(raw, dict):
            issues.append(f"proposal {idx}: expected an object, got {type(raw).__name__}")
            continue

        label = _proposal_schema_label(idx, raw)
        action_raw = raw.get("action")
        action = str(action_raw).strip().lower() if action_raw is not None else ""
        if not action:
            issues.append(f"{label}: missing action")
            continue
        if action not in VALID_ACTIONS:
            issues.append(
                f"{label}: invalid action {action_raw!r}; expected one of {sorted(VALID_ACTIONS)}"
            )
            continue
        if action == "none":
            continue

        if _missing_or_empty(raw.get("name")):
            issues.append(f"{label}: missing name")

        if action == "add":
            proposal_type = raw.get("type")
            if _missing_or_empty(proposal_type):
                issues.append(f"{label}: missing type")
            elif proposal_type not in VALID_TYPES:
                issues.append(
                    f"{label}: invalid type {proposal_type!r}; expected one of {sorted(VALID_TYPES)}"
                )

            roles_issue = _roles_schema_issue(raw.get("roles"))
            if roles_issue is not None:
                issues.append(f"{label}: {roles_issue}")

            if proposal_type == "categorical" and not raw.get("categories"):
                issues.append(f"{label}: missing categories for categorical proposal")
            if _missing_or_empty(raw.get("description")):
                issues.append(f"{label}: missing description")
        elif action == "update_role":
            roles_issue = _roles_schema_issue(raw.get("roles"))
            if roles_issue is not None:
                issues.append(f"{label}: {roles_issue}")

    return issues


def _roles_schema_issue(roles: Any) -> Optional[str]:
    if roles is None or roles == []:
        return "missing roles"
    role_values = [roles] if isinstance(roles, str) else roles
    if not isinstance(role_values, list):
        return "roles must be a string or list"
    normalized = [str(role).strip() for role in role_values if str(role).strip()]
    if not normalized:
        return "missing roles"
    invalid = sorted(set(normalized) - VALID_ROLES)
    if invalid:
        return f"invalid roles {invalid}; expected one or both of {sorted(VALID_ROLES)}"
    return None


def _proposal_schema_label(idx: int, raw: Dict[str, Any]) -> str:
    name = raw.get("name")
    if _missing_or_empty(name):
        return f"proposal {idx}"
    return f"proposal {idx} ({name})"


def _missing_or_empty(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def parse_agent_response(response: str) -> List[Dict[str, Any]]:
    """Parse JSON proposals from an LLM response."""
    response = response.strip()
    match = re.search(r"\{.*\}", response, re.DOTALL)
    json_str = match.group(0) if match else response
    parsed = json.loads(json_str)
    if isinstance(parsed, list):
        return parsed
    proposals = parsed.get("proposals", [])
    if not isinstance(proposals, list):
        raise ValueError("Agent response JSON must contain a proposals list")
    return proposals


def _chat_completion_trace(
    response: Any,
    choice: Any,
    message: Any,
    content: str,
) -> Dict[str, Any]:
    """Build a JSON-serializable trace of the proposal agent response."""
    trace = {
        "raw_content": content,
        "reasoning_content": _message_field(message, "reasoning_content"),
        "finish_reason": getattr(choice, "finish_reason", None),
        "model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "created": getattr(response, "created", None),
        "usage": _to_jsonable(getattr(response, "usage", None)),
    }
    return {key: value for key, value in trace.items() if value is not None}


def _trace_with_repair_attempts(
    trace: Dict[str, Any],
    attempts: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(attempts) <= 1:
        return trace
    enriched = dict(trace)
    enriched["repair_attempts"] = list(attempts)
    return enriched


def _message_field(message: Any, key: str) -> Any:
    value = getattr(message, key, None)
    if value is not None:
        return _to_jsonable(value)

    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict) and extra.get(key) is not None:
        return _to_jsonable(extra[key])

    dumped = _model_dump(message)
    if isinstance(dumped, dict) and dumped.get(key) is not None:
        return _to_jsonable(dumped[key])

    return None


def _get_agent_response_trace(agent: Any) -> Dict[str, Any]:
    trace = getattr(agent, "last_response_trace", None)
    if trace is not None:
        return _to_jsonable(trace)

    raw_response = getattr(agent, "last_raw_response", None)
    if raw_response is not None:
        return {"raw_content": str(raw_response)}

    return {"available": False}


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return None


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]

    dumped = _model_dump(value)
    if dumped is not None and dumped is not value:
        return _to_jsonable(dumped)

    return str(value)


def validate_agentic_proposals(
    raw_proposals: Sequence[Dict[str, Any]],
    current_specs: List[ExplicitFeatureSpec],
    search_config: AgenticFeatureSearchConfig,
    allow_removals: bool,
    max_additions: Optional[int] = None,
) -> Tuple[List[AgenticFeatureProposal], List[Dict[str, Any]]]:
    """Validate raw LLM proposals against schema, role, and leakage guards."""
    current_names = {spec.name for spec in current_specs}
    valid: List[AgenticFeatureProposal] = []
    rejected = []
    additions = 0
    removals = 0
    max_additions = (
        search_config.max_additions_per_iter
        if max_additions is None
        else int(max_additions)
    )

    for raw in raw_proposals:
        try:
            proposal = _coerce_proposal(raw)
            reason = _proposal_rejection_reason(proposal, current_names, allow_removals)
            if reason is None and proposal.action == "add":
                additions += 1
                if additions > max_additions:
                    reason = "too_many_additions"
            if reason is None and proposal.action == "remove":
                removals += 1
                if removals > search_config.max_removals_per_iter:
                    reason = "too_many_removals"
            if reason is None and proposal.action != "none":
                valid.append(proposal)
                if proposal.action == "add":
                    current_names.add(proposal.name)
            elif reason is not None:
                rejected.append({"proposal": raw, "reason": reason})
        except Exception as exc:
            rejected.append({"proposal": raw, "reason": str(exc)})

    return valid, rejected


def _dedupe_proposals_by_name(
    proposals: Sequence[AgenticFeatureProposal],
) -> Tuple[List[AgenticFeatureProposal], List[Dict[str, Any]]]:
    seen = set()
    deduped = []
    rejected = []
    for proposal in proposals:
        if proposal.name in seen:
            rejected.append({"proposal": asdict(proposal), "reason": "duplicate_feature"})
            continue
        seen.add(proposal.name)
        deduped.append(proposal)
    return deduped, rejected


def apply_proposals(
    current_specs: List[ExplicitFeatureSpec],
    proposals: Sequence[AgenticFeatureProposal],
) -> List[ExplicitFeatureSpec]:
    """Apply one or more validated proposals to a feature spec list."""
    specs = list(current_specs)
    for proposal in proposals:
        if proposal.action == "add":
            specs.append(
                ExplicitFeatureSpec(
                    name=proposal.name,
                    type=proposal.type or "continuous",
                    categories=proposal.categories,
                    description=proposal.description,
                    roles=proposal.roles,
                )
            )
        elif proposal.action == "remove":
            specs = [spec for spec in specs if spec.name != proposal.name]
        elif proposal.action == "update_role":
            updated = []
            for spec in specs:
                if spec.name == proposal.name:
                    updated.append(
                        ExplicitFeatureSpec(
                            name=spec.name,
                            type=spec.type,
                            categories=spec.categories,
                            description=spec.description,
                            roles=proposal.roles,
                        )
                    )
                else:
                    updated.append(spec)
            specs = updated
    return specs


def _proposal_to_spec(proposal: AgenticFeatureProposal) -> ExplicitFeatureSpec:
    return ExplicitFeatureSpec(
        name=proposal.name,
        type=proposal.type or "continuous",
        categories=proposal.categories,
        description=proposal.description,
        roles=proposal.roles,
    )


def _screened_spec_to_proposal(
    spec: ExplicitFeatureSpec,
    original: AgenticFeatureProposal,
) -> AgenticFeatureProposal:
    return AgenticFeatureProposal(
        action="add",
        name=spec.name,
        type=spec.type,
        categories=spec.categories,
        description=spec.description,
        roles=spec.roles,
        rationale=original.rationale,
        expected_signal=original.expected_signal,
    )


def compare_candidate_to_baseline(
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    search_config: AgenticFeatureSearchConfig,
) -> Dict[str, Any]:
    """Compare candidate inner-CV metrics to the current baseline feature set."""
    base = aggregate_metric_rows(baseline_rows)
    cand = aggregate_metric_rows(candidate_rows)
    base_r = base.get("r_loss_mean")
    cand_r = cand.get("r_loss_mean")
    if base_r is None or cand_r is None:
        r_loss_improvement = 0.0
    else:
        r_loss_improvement = (base_r - cand_r) / max(abs(base_r), 1e-8)

    outcome_delta = _metric_delta(cand, base, "outcome_auroc_mean")
    treatment_delta = _metric_delta(cand, base, "treatment_auroc_mean")
    improved_fold_fraction = _improved_fold_fraction(
        baseline_rows,
        candidate_rows,
        metric="r_loss",
        lower_is_better=True,
    )
    passes = (
        r_loss_improvement >= search_config.min_r_loss_improvement
        and outcome_delta >= -search_config.max_outcome_auroc_drop
        and treatment_delta >= -search_config.max_treatment_auroc_drop
        and improved_fold_fraction >= search_config.min_improvement_fold_fraction
    )
    return {
        "r_loss_improvement": float(r_loss_improvement),
        "outcome_auroc_delta": float(outcome_delta),
        "treatment_auroc_delta": float(treatment_delta),
        "improved_fold_fraction": float(improved_fold_fraction),
        "passes_acceptance": bool(passes),
        "baseline": _non_oracle_metrics(base),
        "candidate": _non_oracle_metrics(cand),
    }


def evaluate_candidate_role_diagnostics(
    dataset: pd.DataFrame,
    current_specs: List[ExplicitFeatureSpec],
    candidate_specs: List[ExplicitFeatureSpec],
    config: AppliedInferenceConfig,
    search_config: AgenticFeatureSearchConfig,
) -> List[Dict[str, Any]]:
    """Evaluate proposed feature roles using train-fold regressions.

    For each candidate feature, regress treatment and outcome on the current
    confounder-role features plus the candidate main effect. Then regress
    outcome on the same terms plus treatment-by-candidate interactions. These
    diagnostics are advisory and train-only; nested CV still decides acceptance.
    """
    if not getattr(search_config, "role_diagnostics_enabled", True):
        return []
    if not candidate_specs:
        return []

    diagnostics = []
    for candidate_spec in candidate_specs:
        diagnostics.append(
            _evaluate_single_candidate_role_diagnostic(
                dataset=dataset,
                current_specs=current_specs,
                candidate_spec=candidate_spec,
                config=config,
                search_config=search_config,
            )
        )
    return diagnostics


def screen_agentic_candidate_specs(
    dataset: pd.DataFrame,
    current_specs: List[ExplicitFeatureSpec],
    proposals: Sequence[AgenticFeatureProposal],
    config: AppliedInferenceConfig,
    search_config: AgenticFeatureSearchConfig,
) -> List[Dict[str, Any]]:
    """Screen broad candidate proposals with train-fold role diagnostics."""
    screened: List[Dict[str, Any]] = []
    for original_index, proposal in enumerate(proposals):
        proposed_spec = _proposal_to_spec(proposal)
        coverage_failures = _coverage_failures(
            dataset,
            [proposed_spec],
            search_config.min_feature_coverage,
        )
        diagnostics = evaluate_candidate_role_diagnostics(
            dataset=dataset,
            current_specs=current_specs,
            candidate_specs=[proposed_spec],
            config=config,
            search_config=search_config,
        )
        diagnostic = diagnostics[0] if diagnostics else {}
        recommended_roles = [
            role
            for role in diagnostic.get("recommended_roles", [])
            if role in VALID_ROLES
        ]
        confounder_score, modifier_score, screening_score = _role_screen_scores(
            diagnostic
        )
        rejection_reason = None
        if coverage_failures:
            rejection_reason = "low_feature_coverage"
        elif not diagnostic:
            rejection_reason = "missing_role_diagnostic"
        elif diagnostic.get("status") != "ok":
            rejection_reason = str(diagnostic.get("status", "role_diagnostic_failed"))
        elif not recommended_roles:
            rejection_reason = "no_role_signal"

        screened_spec = None
        if rejection_reason is None:
            screened_spec = ExplicitFeatureSpec(
                name=proposed_spec.name,
                type=proposed_spec.type,
                categories=proposed_spec.categories,
                description=proposed_spec.description,
                roles=recommended_roles,
            )

        screened.append(
            {
                "candidate_id": proposed_spec.name,
                "proposal": proposal,
                "proposed_spec": proposed_spec,
                "screened_spec": screened_spec,
                "role_diagnostics": diagnostics,
                "coverage_failures": coverage_failures,
                "screening_rejection_reason": rejection_reason,
                "confounder_score": confounder_score,
                "modifier_score": modifier_score,
                "screening_score": screening_score,
                "original_index": original_index,
                "kept_for_cv": False,
                "cv_accepted": False,
            }
        )

    ranked = sorted(
        screened,
        key=lambda item: (
            item["screening_rejection_reason"] is not None,
            -float(item["screening_score"]),
            int(item["original_index"]),
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def select_screened_candidates(
    screened: Sequence[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Select a balanced high-signal shortlist for inner-CV refinement."""
    top_k = max(1, int(top_k))
    eligible = [
        item for item in screened if item.get("screened_spec") is not None
    ]
    eligible = sorted(
        eligible,
        key=lambda item: (
            -float(item.get("screening_score", 0.0)),
            int(item.get("rank", 0)),
        ),
    )
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    per_role_quota = min(10, max(1, top_k // 2))

    def add_for_role(role: str) -> None:
        for item in eligible:
            if len(selected) >= top_k:
                return
            spec = item.get("screened_spec")
            if spec is None or role not in spec.roles or item["candidate_id"] in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item["candidate_id"])
            if sum(
                1
                for selected_item in selected
                if role in selected_item["screened_spec"].roles
            ) >= per_role_quota:
                return

    add_for_role("confounder")
    add_for_role("effect_modifier")
    for item in eligible:
        if len(selected) >= top_k:
            break
        if item["candidate_id"] in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item["candidate_id"])

    return sorted(selected, key=lambda item: int(item.get("rank", 0)))


def _role_screen_scores(diagnostic: Dict[str, Any]) -> Tuple[float, float, float]:
    treatment_delta = _diagnostic_score_delta(diagnostic, "treatment_association")
    outcome_delta = _diagnostic_score_delta(diagnostic, "outcome_association")
    interaction_delta = _diagnostic_score_delta(diagnostic, "treatment_interaction")
    confounder_score = (
        min(treatment_delta, outcome_delta)
        if treatment_delta is not None and outcome_delta is not None
        else 0.0
    )
    modifier_score = interaction_delta if interaction_delta is not None else 0.0
    return (
        float(max(confounder_score, 0.0)),
        float(max(modifier_score, 0.0)),
        float(max(confounder_score, modifier_score, 0.0)),
    )


def _diagnostic_score_delta(
    diagnostic: Dict[str, Any],
    key: str,
) -> Optional[float]:
    section = diagnostic.get(key)
    if not isinstance(section, dict):
        return None
    delta = section.get("score_delta")
    if not _is_number(delta):
        return None
    return float(delta)


def _evaluate_single_candidate_role_diagnostic(
    dataset: pd.DataFrame,
    current_specs: List[ExplicitFeatureSpec],
    candidate_spec: ExplicitFeatureSpec,
    config: AppliedInferenceConfig,
    search_config: AgenticFeatureSearchConfig,
) -> Dict[str, Any]:
    df = dataset.reset_index(drop=True)
    coverage = _feature_coverage(df, candidate_spec.name)
    non_missing_n = int(round(coverage * len(df)))
    base_payload = {
        "name": candidate_spec.name,
        "proposed_roles": list(candidate_spec.roles),
        "type": candidate_spec.type,
        "n": int(len(df)),
        "coverage": float(coverage),
        "non_missing_n": non_missing_n,
        "adjustment": "current_confounders",
    }

    min_n = int(getattr(search_config, "role_diagnostic_min_n", 20))
    min_non_missing = int(getattr(search_config, "role_diagnostic_min_non_missing", 10))
    if len(df) < min_n:
        return {
            **base_payload,
            "status": "insufficient_sample",
            "recommended_roles": [],
        }
    if non_missing_n < min_non_missing:
        return {
            **base_payload,
            "status": "insufficient_non_missing",
            "recommended_roles": [],
        }

    current_confounders = [
        spec
        for spec in current_specs
        if spec.name != candidate_spec.name and "confounder" in spec.roles
    ]
    _, w_matrix, _, w_names, _, _ = _build_features(df, current_confounders)
    z_spec = ExplicitFeatureSpec(
        name=candidate_spec.name,
        type=candidate_spec.type,
        categories=candidate_spec.categories,
        description=candidate_spec.description,
        roles=["confounder"],
    )
    _, z_matrix, _, z_names, _, _ = _build_features(df, [z_spec])

    n_rows = len(df)
    w_matrix = _feature_matrix_or_empty(w_matrix, n_rows)
    z_matrix = _feature_matrix_or_empty(z_matrix, n_rows)
    if z_matrix.shape[1] == 0 or not _has_any_variation(z_matrix):
        return {
            **base_payload,
            "status": "constant_candidate",
            "covariate_feature_names": w_names,
            "candidate_feature_names": z_names,
            "recommended_roles": [],
        }

    treatment = np.asarray(df[config.treatment_column].values).flatten()
    outcome = np.asarray(df[config.outcome_column].values).flatten()
    treatment_col = treatment.astype(float).reshape(-1, 1)

    treatment_diag = _nested_regression_diagnostic(
        base_x=w_matrix,
        full_x=np.hstack([w_matrix, z_matrix]),
        target=treatment,
        target_kind="binary",
        block_start=w_matrix.shape[1],
        block_width=z_matrix.shape[1],
    )

    outcome_base_x = np.hstack([w_matrix, treatment_col])
    outcome_main_x = np.hstack([outcome_base_x, z_matrix])
    outcome_diag = _nested_regression_diagnostic(
        base_x=outcome_base_x,
        full_x=outcome_main_x,
        target=outcome,
        target_kind=config.outcome_type,
        block_start=outcome_base_x.shape[1],
        block_width=z_matrix.shape[1],
    )

    interaction_matrix = z_matrix * treatment_col
    interaction_full_x = np.hstack([outcome_main_x, interaction_matrix])
    interaction_diag = _nested_regression_diagnostic(
        base_x=outcome_main_x,
        full_x=interaction_full_x,
        target=outcome,
        target_kind=config.outcome_type,
        block_start=outcome_main_x.shape[1],
        block_width=interaction_matrix.shape[1],
    )

    threshold = float(getattr(search_config, "role_diagnostic_score_delta_threshold", 0.001))
    treatment_signal = _score_delta_at_least(treatment_diag, threshold)
    outcome_signal = _score_delta_at_least(outcome_diag, threshold)
    interaction_signal = _score_delta_at_least(interaction_diag, threshold)
    recommended_roles = []
    if treatment_signal and outcome_signal:
        recommended_roles.append("confounder")
    if interaction_signal:
        recommended_roles.append("effect_modifier")

    return {
        **base_payload,
        "status": "ok",
        "covariate_feature_names": w_names,
        "candidate_feature_names": z_names,
        "treatment_association": treatment_diag,
        "outcome_association": outcome_diag,
        "treatment_interaction": interaction_diag,
        "confounder_signal": bool(treatment_signal and outcome_signal),
        "effect_modifier_signal": bool(interaction_signal),
        "recommended_roles": recommended_roles,
    }


def _nested_regression_diagnostic(
    base_x: np.ndarray,
    full_x: np.ndarray,
    target: np.ndarray,
    target_kind: str,
    block_start: int,
    block_width: int,
) -> Dict[str, Any]:
    target_kind = "continuous" if target_kind == "continuous" else "binary"
    y = np.asarray(target).flatten()
    base_x = np.asarray(base_x, dtype=np.float64)
    full_x = np.asarray(full_x, dtype=np.float64)
    finite = np.isfinite(y.astype(float, copy=False))
    finite &= np.all(np.isfinite(base_x), axis=1)
    finite &= np.all(np.isfinite(full_x), axis=1)
    if finite.sum() < 2:
        return {"status": "insufficient_finite_target", "target_kind": target_kind}

    base_x = base_x[finite]
    full_x = full_x[finite]
    y = y[finite]

    if target_kind == "binary":
        unique = np.unique(y)
        if len(unique) < 2:
            return {"status": "constant_target", "target_kind": target_kind}
        y_model = (y == unique[-1]).astype(int)
        base_pred, base_score, base_auroc, _ = _fit_binary_regression(base_x, y_model)
        full_pred, full_score, full_auroc, full_coef = _fit_binary_regression(full_x, y_model)
        del base_pred, full_pred
        result = {
            "status": "ok",
            "target_kind": target_kind,
            "score_metric": "neg_log_loss",
            "base_score": base_score,
            "full_score": full_score,
            "score_delta": _safe_subtract(full_score, base_score),
            "base_auroc": base_auroc,
            "full_auroc": full_auroc,
            "auroc_delta": _safe_subtract(full_auroc, base_auroc),
        }
    else:
        y_model = y.astype(float)
        if np.std(y_model) == 0:
            return {"status": "constant_target", "target_kind": target_kind}
        base_pred, base_score, _ = _fit_continuous_regression(base_x, y_model)
        full_pred, full_score, full_coef = _fit_continuous_regression(full_x, y_model)
        del base_pred, full_pred
        result = {
            "status": "ok",
            "target_kind": target_kind,
            "score_metric": "r2",
            "base_score": base_score,
            "full_score": full_score,
            "score_delta": _safe_subtract(full_score, base_score),
        }

    block_coef = np.asarray(full_coef[block_start : block_start + block_width], dtype=float)
    result["coefficient_l2_norm"] = float(np.linalg.norm(block_coef))
    result["n_model_rows"] = int(len(y))
    return result


def _fit_binary_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, float, Optional[float], np.ndarray]:
    x = _ensure_regression_columns(x)
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(x, y)
    pred = model.predict_proba(x)[:, 1]
    clipped = np.clip(pred, 1e-6, 1.0 - 1e-6)
    score = -float(log_loss(y, clipped, labels=[0, 1]))
    return pred, score, _safe_roc_auc(y, pred), model.coef_.reshape(-1)


def _fit_continuous_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    x = _ensure_regression_columns(x)
    model = Ridge(alpha=1.0)
    model.fit(x, y)
    pred = model.predict(x)
    return pred, float(r2_score(y, pred)), np.asarray(model.coef_).reshape(-1)


def _ensure_regression_columns(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], 1), dtype=np.float64)
    return x


def _feature_matrix_or_empty(
    matrix: Optional[np.ndarray],
    n_rows: int,
) -> np.ndarray:
    if matrix is None:
        return np.zeros((n_rows, 0), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _has_any_variation(matrix: np.ndarray) -> bool:
    if matrix.shape[1] == 0:
        return False
    return bool(np.any(np.nanstd(matrix, axis=0) > 1e-12))


def _score_delta_at_least(diagnostic: Dict[str, Any], threshold: float) -> bool:
    delta = diagnostic.get("score_delta")
    return _is_number(delta) and float(delta) >= threshold


def _safe_subtract(left: Any, right: Any) -> Optional[float]:
    if left is None or right is None:
        return None
    if not (_is_number(left) and _is_number(right)):
        return None
    return float(left) - float(right)


def _feature_coverage(dataset: pd.DataFrame, name: str) -> float:
    value_col = f"explicit_feat_{name}"
    missing_col = f"{value_col}_missing"
    if value_col not in dataset.columns:
        return 0.0
    missing = (
        dataset[missing_col].astype(bool)
        if missing_col in dataset.columns
        else dataset[value_col].isna()
    )
    return float(1.0 - missing.mean())


def build_iteration_feedback(
    recent_decisions: List[Dict[str, Any]],
    search_config: AgenticFeatureSearchConfig,
) -> List[Dict[str, Any]]:
    """Distill prior decisions into compact feedback for the next agent prompt."""
    feedback: List[Dict[str, Any]] = []
    for event in recent_decisions:
        event_name = event.get("event")
        payload = event.get("payload")
        if event_name == "agent_proposals" and isinstance(payload, dict):
            for rejected in payload.get("rejected", []):
                if not isinstance(rejected, dict):
                    continue
                raw_proposal = rejected.get("proposal", {})
                feedback.append(
                    {
                        "iteration": event.get("iteration"),
                        "candidate_id": _proposal_feedback_id(raw_proposal),
                        "status": "validation_rejected",
                        "failed_checks": [str(rejected.get("reason", "validation_failed"))],
                        "proposals": [_proposal_feedback_summary(raw_proposal)],
                        "instruction": (
                            "Do not repeat this proposal unchanged; fix the validation "
                            "failure or propose a different variable."
                        ),
                    }
                )
        elif event_name == "candidate_evaluations" and isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                comparison = item.get("comparison", {})
                accepted = bool(item.get("accepted", False))
                passed = bool(comparison.get("passes_acceptance", False))
                if accepted:
                    status = "accepted"
                elif passed:
                    status = "not_selected"
                else:
                    status = "rejected"
                entry = {
                    "iteration": event.get("iteration"),
                    "candidate_id": item.get("candidate_id"),
                    "status": status,
                    "proposals": [
                        _proposal_feedback_summary(proposal)
                        for proposal in item.get("proposals", [])
                    ],
                    "metrics": _candidate_feedback_metrics(comparison),
                }
                role_diagnostics = _candidate_feedback_role_diagnostics(item.get("summary"))
                if role_diagnostics:
                    entry["role_diagnostics"] = role_diagnostics
                if accepted:
                    entry["instruction"] = (
                        "This candidate became the current baseline; build on it "
                        "unless later feedback indicates a problem."
                    )
                elif passed:
                    entry["failed_checks"] = ["passed_thresholds_but_lower_ranked"]
                    entry["instruction"] = (
                        "This candidate passed acceptance thresholds but was not "
                        "selected because another candidate had stronger R-loss "
                        "improvement."
                    )
                else:
                    entry["failed_checks"] = _candidate_failed_checks(
                        comparison,
                        search_config,
                    )
                    entry["instruction"] = (
                        "Do not repeat this candidate unchanged; propose a different "
                        "baseline variable, extraction target, or role that addresses "
                        "the failed_checks."
                    )
                feedback.append(entry)

    return feedback[-20:]


def _proposal_feedback_id(proposal: Any) -> str:
    if isinstance(proposal, dict):
        name = proposal.get("name")
        if name:
            return _normalize_feature_name(name)
        action = proposal.get("action")
        if action:
            return str(action)
    return str(proposal)


def _proposal_feedback_summary(proposal: Any) -> Dict[str, Any]:
    if not isinstance(proposal, dict):
        return {"raw": str(proposal)}
    return {
        key: proposal.get(key)
        for key in ["action", "name", "type", "roles", "description"]
        if proposal.get(key) is not None
    }


def _candidate_feedback_metrics(comparison: Any) -> Dict[str, Any]:
    if not isinstance(comparison, dict):
        return {}
    keys = [
        "r_loss_improvement",
        "outcome_auroc_delta",
        "treatment_auroc_delta",
        "improved_fold_fraction",
        "passes_acceptance",
        "rejection_reason",
        "coverage_failures",
    ]
    return {
        key: comparison[key]
        for key in keys
        if key in comparison
    }


def _candidate_feedback_role_diagnostics(summary: Any) -> List[Dict[str, Any]]:
    if not isinstance(summary, dict):
        return []
    diagnostics = summary.get("role_diagnostics")
    if not isinstance(diagnostics, list):
        return []
    compact = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        entry = {
            key: diagnostic.get(key)
            for key in [
                "name",
                "status",
                "proposed_roles",
                "recommended_roles",
                "confounder_signal",
                "effect_modifier_signal",
                "coverage",
                "non_missing_n",
            ]
            if diagnostic.get(key) is not None
        }
        treatment = diagnostic.get("treatment_association", {})
        outcome = diagnostic.get("outcome_association", {})
        interaction = diagnostic.get("treatment_interaction", {})
        if isinstance(treatment, dict):
            entry["treatment_score_delta"] = treatment.get("score_delta")
            entry["treatment_auroc_delta"] = treatment.get("auroc_delta")
        if isinstance(outcome, dict):
            entry["outcome_score_delta"] = outcome.get("score_delta")
            entry["outcome_auroc_delta"] = outcome.get("auroc_delta")
        if isinstance(interaction, dict):
            entry["interaction_score_delta"] = interaction.get("score_delta")
            entry["interaction_auroc_delta"] = interaction.get("auroc_delta")
        compact.append({key: value for key, value in entry.items() if value is not None})
    return compact


def _candidate_failed_checks(
    comparison: Any,
    search_config: AgenticFeatureSearchConfig,
) -> List[str]:
    if not isinstance(comparison, dict):
        return ["candidate_evaluation_missing"]

    failed = []
    rejection_reason = comparison.get("rejection_reason")
    if rejection_reason:
        failed.append(f"rejection_reason: {rejection_reason}")

    for item in comparison.get("coverage_failures", []) or []:
        if not isinstance(item, dict):
            continue
        coverage = item.get("coverage")
        name = item.get("name", "feature")
        if _is_number(coverage):
            failed.append(
                f"coverage {name} {float(coverage):.4g} "
                f"< required {search_config.min_feature_coverage:.4g}"
            )

    r_loss_improvement = comparison.get("r_loss_improvement")
    if (
        _is_number(r_loss_improvement)
        and float(r_loss_improvement) < search_config.min_r_loss_improvement
    ):
        failed.append(
            f"r_loss_improvement {float(r_loss_improvement):.4g} "
            f"< required {search_config.min_r_loss_improvement:.4g}"
        )

    outcome_delta = comparison.get("outcome_auroc_delta")
    outcome_floor = -search_config.max_outcome_auroc_drop
    if _is_number(outcome_delta) and float(outcome_delta) < outcome_floor:
        failed.append(
            f"outcome_auroc_delta {float(outcome_delta):.4g} "
            f"< allowed {outcome_floor:.4g}"
        )

    treatment_delta = comparison.get("treatment_auroc_delta")
    treatment_floor = -search_config.max_treatment_auroc_drop
    if _is_number(treatment_delta) and float(treatment_delta) < treatment_floor:
        failed.append(
            f"treatment_auroc_delta {float(treatment_delta):.4g} "
            f"< allowed {treatment_floor:.4g}"
        )

    improved_fold_fraction = comparison.get("improved_fold_fraction")
    if (
        _is_number(improved_fold_fraction)
        and float(improved_fold_fraction) < search_config.min_improvement_fold_fraction
    ):
        failed.append(
            f"improved_fold_fraction {float(improved_fold_fraction):.4g} "
            f"< required {search_config.min_improvement_fold_fraction:.4g}"
        )

    if not failed:
        failed.append("did_not_pass_acceptance_thresholds")
    return failed


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def aggregate_metric_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate numeric split metrics as mean/std, ignoring oracle-only keys."""
    if not rows:
        return {}
    df = pd.DataFrame([_non_oracle_metrics(row) for row in rows])
    result: Dict[str, Any] = {}
    for col in df.columns:
        if col in {"outer_fold", "iteration", "inner_fold", "fold"}:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        result[f"{col}_mean"] = float(values.mean())
        result[f"{col}_std"] = float(values.std(ddof=0))
    return result


def summarize_extractions(
    dataset: pd.DataFrame,
    specs: List[ExplicitFeatureSpec],
) -> List[Dict[str, Any]]:
    """Summarize extraction coverage and observed values for the current features."""
    summaries = []
    for spec in specs:
        value_col = f"explicit_feat_{spec.name}"
        missing_col = f"{value_col}_missing"
        if value_col not in dataset.columns:
            summaries.append({"name": spec.name, "coverage": 0.0, "top_values": {}})
            continue
        missing = dataset[missing_col].astype(bool) if missing_col in dataset.columns else dataset[value_col].isna()
        observed = dataset.loc[~missing, value_col]
        summaries.append(
            {
                "name": spec.name,
                "coverage": float(1.0 - missing.mean()),
                "top_values": observed.astype(str).value_counts().head(8).to_dict(),
            }
        )
    return summaries


def _clinical_text_examples(
    dataset: pd.DataFrame,
    text_column: str,
    n_examples: int = 3,
    max_chars: int = 1600,
) -> List[str]:
    if text_column not in dataset.columns or len(dataset) == 0:
        return []
    sample = dataset.sample(
        n=min(n_examples, len(dataset)),
        random_state=17,
    )
    return [
        str(text)[:max_chars]
        for text in sample[text_column].fillna("").tolist()
        if str(text).strip()
    ]


def _coverage_failures(
    dataset: pd.DataFrame,
    specs: List[ExplicitFeatureSpec],
    min_coverage: float,
) -> List[Dict[str, Any]]:
    failures = []
    for item in summarize_extractions(dataset, specs):
        if item["coverage"] < min_coverage:
            failures.append({"name": item["name"], "coverage": item["coverage"]})
    return failures


def _coerce_proposal(raw: Dict[str, Any]) -> AgenticFeatureProposal:
    action = str(raw.get("action", "")).strip().lower()
    name = _normalize_feature_name(raw.get("name", ""))
    roles = raw.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    categories = raw.get("categories")
    if categories is not None:
        categories = [str(cat) for cat in categories]
    return AgenticFeatureProposal(
        action=action,
        name=name,
        type=raw.get("type"),
        categories=categories,
        description=raw.get("description"),
        roles=[str(role).strip() for role in roles],
        rationale=raw.get("rationale"),
        expected_signal=raw.get("expected_signal"),
    )


def _proposal_rejection_reason(
    proposal: AgenticFeatureProposal,
    current_names: set,
    allow_removals: bool,
) -> Optional[str]:
    if proposal.action not in VALID_ACTIONS:
        return "invalid_action"
    if proposal.action == "none":
        return None
    if not proposal.name or not re.match(r"^[a-z][a-z0-9_]*$", proposal.name):
        return "invalid_name"
    if proposal.action == "add":
        if proposal.name in current_names:
            return "duplicate_feature"
        if proposal.type not in VALID_TYPES:
            return "invalid_type"
        if not proposal.roles or set(proposal.roles) - VALID_ROLES:
            return "invalid_roles"
        if proposal.type == "categorical" and not proposal.categories:
            return "missing_categories"
        if proposal.type == "categorical" and len(proposal.categories or []) > 8:
            return "too_many_categories"
        if not proposal.description:
            return "missing_description"
    elif proposal.action in {"remove", "update_role"}:
        if not allow_removals:
            return "removal_or_role_update_not_allowed_yet"
        if proposal.name not in current_names:
            return "unknown_existing_feature"
        if proposal.action == "update_role" and (
            not proposal.roles or set(proposal.roles) - VALID_ROLES
        ):
            return "invalid_roles"
    return None


def _candidate_groups(
    proposals: List[AgenticFeatureProposal],
) -> List[Tuple[str, List[AgenticFeatureProposal]]]:
    groups = [(proposal.name, [proposal]) for proposal in proposals]
    if len(proposals) > 1:
        bundled = []
        seen_names = set()
        for proposal in proposals:
            if proposal.name in seen_names:
                continue
            bundled.append(proposal)
            seen_names.add(proposal.name)
        if len(bundled) > 1:
            groups.append(("bundle", bundled))
    return groups


def _candidate_proposal_specs(
    current_specs: List[ExplicitFeatureSpec],
    candidate_specs: List[ExplicitFeatureSpec],
    proposal_group: Sequence[AgenticFeatureProposal],
) -> List[ExplicitFeatureSpec]:
    """Return add/update specs touched by this proposal group."""
    current_by_name = {spec.name: spec for spec in current_specs}
    candidate_by_name = {spec.name: spec for spec in candidate_specs}
    proposal_specs = []
    seen = set()
    for proposal in proposal_group:
        if proposal.action not in {"add", "update_role"}:
            continue
        if proposal.name in seen:
            continue
        spec = candidate_by_name.get(proposal.name) or current_by_name.get(proposal.name)
        if spec is not None:
            proposal_specs.append(spec)
            seen.add(proposal.name)
    return proposal_specs


def _candidate_role_diagnostic_specs(
    current_specs: List[ExplicitFeatureSpec],
    candidate_specs: List[ExplicitFeatureSpec],
    proposal_group: Sequence[AgenticFeatureProposal],
) -> List[ExplicitFeatureSpec]:
    return _candidate_proposal_specs(current_specs, candidate_specs, proposal_group)


def _choose_accepted_candidate(candidate_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    passing = [
        item for item in candidate_results if item["comparison"].get("passes_acceptance")
    ]
    if not passing:
        return None
    return max(
        passing,
        key=lambda item: item["comparison"].get("r_loss_improvement", 0.0),
    )


def _dedupe_feature_specs(specs: Sequence[ExplicitFeatureSpec]) -> List[ExplicitFeatureSpec]:
    deduped = []
    seen = set()
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        deduped.append(spec)
    return deduped


def _screening_decision_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    spec = item.get("screened_spec") or item.get("proposed_spec")
    proposal = item.get("proposal")
    variables = []
    if isinstance(spec, ExplicitFeatureSpec):
        source = proposal if isinstance(proposal, AgenticFeatureProposal) else None
        fallback = AgenticFeatureProposal(
            action="add",
            name=spec.name,
            type=spec.type,
            categories=spec.categories,
            description=spec.description,
            roles=spec.roles,
        )
        variables.append(asdict(_screened_spec_to_proposal(spec, source or fallback)))
    comparison = item.get("cv_comparison", {})
    return {
        "candidate_id": item.get("candidate_id"),
        "rank": item.get("rank"),
        "variables": variables,
        "screening_score": item.get("screening_score"),
        "confounder_score": item.get("confounder_score"),
        "modifier_score": item.get("modifier_score"),
        "kept_for_cv": bool(item.get("kept_for_cv", False)),
        "cv_accepted": bool(item.get("cv_accepted", False)),
        "passes_acceptance": bool(
            isinstance(comparison, dict)
            and comparison.get("passes_acceptance", False)
        ),
        "screening_rejection_reason": item.get("screening_rejection_reason"),
        "coverage_failures": item.get("coverage_failures", []),
        "role_diagnostics": item.get("role_diagnostics", []),
    }


def _screening_metric_row(
    outer_fold: int,
    iteration: int,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    diagnostic = (
        item.get("role_diagnostics", [{}])[0]
        if item.get("role_diagnostics")
        else {}
    )
    comparison = item.get("cv_comparison", {})
    screened_spec = item.get("screened_spec")
    proposed_spec = item.get("proposed_spec")
    spec = screened_spec or proposed_spec
    return {
        "outer_fold": outer_fold,
        "iteration": iteration,
        "candidate_id": item.get("candidate_id"),
        "rank": item.get("rank"),
        "kept_for_cv": bool(item.get("kept_for_cv", False)),
        "cv_accepted": bool(item.get("cv_accepted", False)),
        "screening_rejection_reason": item.get("screening_rejection_reason"),
        "coverage": diagnostic.get("coverage"),
        "diagnostic_status": diagnostic.get("status"),
        "proposed_roles": ",".join(getattr(spec, "roles", []) or []),
        "recommended_roles": ",".join(
            diagnostic.get("recommended_roles", [])
            if isinstance(diagnostic, dict)
            else []
        ),
        "screening_score": item.get("screening_score"),
        "confounder_score": item.get("confounder_score"),
        "modifier_score": item.get("modifier_score"),
        "treatment_score_delta": _diagnostic_score_delta(
            diagnostic,
            "treatment_association",
        ),
        "outcome_score_delta": _diagnostic_score_delta(
            diagnostic,
            "outcome_association",
        ),
        "interaction_score_delta": _diagnostic_score_delta(
            diagnostic,
            "treatment_interaction",
        ),
        "r_loss_improvement": (
            comparison.get("r_loss_improvement")
            if isinstance(comparison, dict)
            else None
        ),
        "passes_acceptance": (
            comparison.get("passes_acceptance")
            if isinstance(comparison, dict)
            else None
        ),
    }


def _make_splits(
    df: pd.DataFrame,
    config: AppliedInferenceConfig,
    n_splits: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_splits > len(df):
        raise ValueError(f"n_splits={n_splits} exceeds n={len(df)}")
    y = (
        df[config.treatment_column].astype(str)
        + "_"
        + df[config.outcome_column].astype(str)
    )
    counts = y.value_counts()
    if len(counts) >= 2 and counts.min() >= n_splits:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        return list(splitter.split(df, y))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(df))


def _fit_predict_propensity(
    train_x: np.ndarray,
    train_t: np.ndarray,
    test_x: np.ndarray,
    cf_config: ExplicitFeatureForestConfig,
    random_state: int,
) -> np.ndarray:
    if len(np.unique(train_t)) < 2:
        return np.full(len(test_x), float(train_t[0]), dtype=np.float32)
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
        return np.full(len(test_x), float(train_y[0]), dtype=np.float32)
    model = RandomForestClassifier(
        n_estimators=max(50, cf_config.n_estimators // 2),
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def _r_loss(
    y: np.ndarray,
    t: np.ndarray,
    outcome_pred: np.ndarray,
    propensity: np.ndarray,
    tau: np.ndarray,
) -> float:
    residual_y = np.asarray(y) - np.asarray(outcome_pred)
    residual_t = np.asarray(t) - np.asarray(propensity)
    return float(np.mean((residual_y - np.asarray(tau) * residual_t) ** 2))


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


def _metric_delta(candidate: Dict[str, Any], baseline: Dict[str, Any], key: str) -> float:
    cand = candidate.get(key)
    base = baseline.get(key)
    if cand is None or base is None:
        return 0.0
    return float(cand - base)


def _improved_fold_fraction(
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    metric: str,
    lower_is_better: bool,
) -> float:
    baseline_by_fold = {row.get("inner_fold", row.get("fold")): row for row in baseline_rows}
    candidate_by_fold = {row.get("inner_fold", row.get("fold")): row for row in candidate_rows}
    common = sorted(set(baseline_by_fold) & set(candidate_by_fold))
    if not common:
        return 0.0
    improved = 0
    for fold in common:
        base = baseline_by_fold[fold].get(metric)
        cand = candidate_by_fold[fold].get(metric)
        if base is None or cand is None:
            continue
        improved += cand < base if lower_is_better else cand > base
    return improved / len(common)


def _without_list_values(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, (list, dict, tuple))
    }


def _non_oracle_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if not str(key).startswith("oracle_") and not str(key).startswith("true_")
    }


def _clinical_question_text(config: AppliedInferenceConfig) -> str:
    configured = str(getattr(config, "clinical_question", "") or "").strip()
    if configured:
        return configured
    return (
        "What is the causal effect of "
        f"{config.treatment_column} on {config.outcome_column}?"
    )


def _scrub_decision_payload(payload: Any, save_agent_context: bool) -> Any:
    """Remove raw clinical text examples from persisted decision artifacts."""
    if save_agent_context:
        return payload
    if isinstance(payload, dict):
        scrubbed = {}
        for key, value in payload.items():
            if key == "clinical_text_examples":
                scrubbed[key] = []
            else:
                scrubbed[key] = _scrub_decision_payload(value, save_agent_context)
        return scrubbed
    if isinstance(payload, list):
        return [
            _scrub_decision_payload(item, save_agent_context)
            for item in payload
        ]
    return payload


def _spec_to_dict(spec: ExplicitFeatureSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "type": spec.type,
        "categories": spec.categories,
        "description": spec.description,
        "roles": spec.roles,
    }


def _spec_names(specs: List[ExplicitFeatureSpec]) -> List[str]:
    return [spec.name for spec in specs]


def _normalize_feature_name(name: Any) -> str:
    name = str(name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, ExplicitFeatureSpec):
        return _spec_to_dict(value)
    return str(value)
