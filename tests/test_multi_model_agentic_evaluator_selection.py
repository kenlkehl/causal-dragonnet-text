import json
from pathlib import Path

import numpy as np
import pandas as pd

from oci.config import (
    AppliedInferenceConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    ModelArchitectureConfig,
    MultiModelForestConfig,
)
from oci.inference.agentic_explicit_feature_forest import (
    CausalForestExplicitEvaluator,
    StructuredInteractionExplicitEvaluator,
)
from oci.inference.multi_model_agentic_forest import (
    MultiModelAgenticForestRunner,
    PrecomputedDiscoveryMultiModelAgenticForestRunner,
    _without_oracle_columns,
)


def _dataset(n_rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(41)
    biomarker = rng.normal(size=n_rows)
    treatment = rng.binomial(1, 0.5, size=n_rows)
    logit = -0.3 + 0.2 * biomarker + treatment * (0.1 + biomarker)
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    return pd.DataFrame(
        {
            "clinical_text": [f"document {index}" for index in range(n_rows)],
            "treatment_indicator": treatment,
            "outcome_indicator": outcome,
            "explicit_feat_biomarker": biomarker,
            "explicit_feat_biomarker_missing": False,
            "true_ite_prob": biomarker,
            "oracle_debug_value": np.arange(n_rows),
        }
    )


def _config(estimator: str) -> AppliedInferenceConfig:
    return AppliedInferenceConfig(
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                structured_effect_estimator=estimator,
            ),
            explicit_feature_forest=ExplicitFeatureForestConfig(
                interaction_regularization_grid=[0.1, 1.0],
                interaction_inner_folds=2,
                interaction_max_iter=250,
            ),
        ),
    )


def _runner(tmp_path: Path, config: AppliedInferenceConfig, **kwargs):
    return MultiModelAgenticForestRunner(
        dataset=_dataset(),
        config=config,
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=object(),
        extraction_provider=object(),
        **kwargs,
    )


def test_legacy_runner_selects_configured_structured_or_forest_evaluator(tmp_path):
    structured = _runner(tmp_path / "structured", _config("interaction_s_learner"))
    forest = _runner(tmp_path / "forest", _config("causal_forest"))

    assert isinstance(structured.evaluator, StructuredInteractionExplicitEvaluator)
    assert isinstance(forest.evaluator, CausalForestExplicitEvaluator)


def test_legacy_runner_without_integrated_config_preserves_causal_forest(tmp_path):
    config = _config("interaction_s_learner")
    config.architecture.multi_model_forest = None

    runner = _runner(tmp_path, config)

    assert isinstance(runner.evaluator, CausalForestExplicitEvaluator)


def test_explicit_evaluator_override_still_takes_precedence(tmp_path):
    override = object()
    runner = _runner(
        tmp_path,
        _config("interaction_s_learner"),
        evaluator=override,
    )

    assert runner.evaluator is override


def test_precomputed_runner_inherits_structured_evaluator_selection(tmp_path):
    handoff_path = tmp_path / "handoff.jsonl"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": "multi_model_agentic_discovery_handoff_v1",
                "fold_key": 1,
                "outer_fold": 1,
                "scope": "full_outer_train",
                "n_rows": 80,
            }
        )
        + "\n"
    )
    runner = PrecomputedDiscoveryMultiModelAgenticForestRunner(
        dataset=_dataset(),
        config=_config("interaction_s_learner"),
        output_path=tmp_path / "predictions.parquet",
        handoff_path=handoff_path,
        proposal_agent=object(),
        extraction_provider=object(),
    )

    assert isinstance(runner.evaluator, StructuredInteractionExplicitEvaluator)


def test_selected_structured_evaluator_does_not_propagate_oracle_columns(tmp_path):
    runner = _runner(tmp_path, _config("interaction_s_learner"))
    frame = runner.dataset
    spec = ExplicitFeatureSpec(
        name="biomarker",
        type="continuous",
        roles=["effect_modifier"],
    )

    model_train = _without_oracle_columns(frame.iloc[:90])
    model_test = _without_oracle_columns(frame.iloc[90:])
    evaluation = runner.evaluator.evaluate_split(
        train_df=model_train,
        test_df=model_test,
        specs=[spec],
        fold_id=1,
    )

    assert "true_ite_prob" not in evaluation.predictions.columns
    assert "oracle_debug_value" not in evaluation.predictions.columns
    assert not any(
        str(column).startswith(("true_", "oracle_"))
        for column in evaluation.predictions.columns
    )
    assert evaluation.metrics["effect_estimator"] == "interaction_s_learner"
