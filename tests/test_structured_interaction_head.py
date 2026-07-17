import numpy as np
import pandas as pd
import pytest

from oci.config import (
    AppliedInferenceConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
)
from oci.inference.agentic_explicit_feature_forest import (
    StructuredInteractionExplicitEvaluator,
)
from oci.models.structured_interaction_head import StructuredInteractionHead


def test_generic_defaults_explore_stronger_logistic_regularization():
    expected = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

    assert StructuredInteractionHead().regularization_grid == expected
    assert (
        tuple(ExplicitFeatureForestConfig().interaction_regularization_grid)
        == expected
    )


def test_binary_interaction_head_recovers_heterogeneous_effect_without_oracle_tuning():
    rng = np.random.default_rng(17)
    n = 1800
    confounder = rng.normal(size=n)
    modifier = rng.normal(size=n)
    features = np.column_stack((confounder, modifier))
    propensity = 1.0 / (1.0 + np.exp(-0.9 * confounder))
    treatment = rng.binomial(1, propensity)
    baseline_logit = -0.5 + 0.7 * confounder + 0.2 * modifier
    treated_logit = baseline_logit + 0.25 + 1.25 * modifier
    y0 = 1.0 / (1.0 + np.exp(-baseline_logit))
    y1 = 1.0 / (1.0 + np.exp(-treated_logit))
    outcome_probability = np.where(treatment == 1, y1, y0)
    outcome = rng.binomial(1, outcome_probability)

    fit = np.arange(1300)
    heldout = np.arange(1300, n)
    head = StructuredInteractionHead(
        outcome_type="binary",
        regularization_grid=(0.03, 0.1, 0.3, 1.0, 3.0),
        inner_folds=4,
        interact_all_features=False,
        random_state=31,
    ).fit(
        features[fit],
        treatment[fit],
        outcome[fit],
        modifier_indices=[1],
    )

    prediction = head.predict_effect(features[heldout])
    oracle_effect = y1[heldout] - y0[heldout]
    assert np.corrcoef(prediction, oracle_effect)[0, 1] > 0.95
    assert head.tuning_result_.selection_metric == "log_loss"
    assert head.tuning_result_.n_splits == 4
    assert set(head.tuning_result_.mean_validation_loss) == {
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
    }


def test_interaction_head_can_treat_all_discovered_features_as_modifier_candidates():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(500, 3))
    treatment = rng.binomial(1, 0.5, size=500)
    logit = -0.4 + 0.2 * features[:, 0] + treatment * features[:, 2]
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    head = StructuredInteractionHead(
        interact_all_features=True,
        inner_folds=3,
        random_state=8,
    ).fit(features, treatment, outcome)

    assert head.modifier_indices_.tolist() == [0, 1, 2]
    assert head.predict_effect(features[:7]).shape == (7,)


def test_inner_fold_standardization_makes_regularized_fit_affine_invariant():
    rng = np.random.default_rng(901)
    features = rng.normal(size=(700, 3))
    treatment = rng.binomial(1, 0.5, size=700)
    logits = (
        -0.3
        + 0.35 * features[:, 0]
        + treatment * (0.15 + 0.9 * features[:, 2])
    )
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))
    shifted = features * np.asarray([1000.0, 0.02, 17.0]) + np.asarray(
        [80000.0, -12.0, 250.0]
    )
    kwargs = {
        "outcome_type": "binary",
        "regularization_grid": (0.03, 0.1, 0.3, 1.0),
        "inner_folds": 4,
        "interact_all_features": False,
        "random_state": 44,
    }
    original = StructuredInteractionHead(**kwargs).fit(
        features, treatment, outcome, modifier_indices=[2]
    )
    transformed = StructuredInteractionHead(**kwargs).fit(
        shifted, treatment, outcome, modifier_indices=[2]
    )

    assert (
        original.tuning_result_.selected_regularization
        == transformed.tuning_result_.selected_regularization
    )
    np.testing.assert_allclose(
        original.predict_effect(features[:80]),
        transformed.predict_effect(shifted[:80]),
        atol=1e-10,
        rtol=1e-10,
    )


def test_continuous_interaction_head_uses_ridge_and_returns_potential_outcomes():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(600, 2))
    treatment = rng.binomial(1, 0.5, size=600)
    outcome = (
        0.6 * features[:, 0]
        + treatment * (0.3 + 1.4 * features[:, 1])
        + rng.normal(scale=0.15, size=600)
    )
    head = StructuredInteractionHead(
        outcome_type="continuous",
        regularization_grid=(0.01, 0.1, 1.0, 10.0),
        inner_folds=3,
        interact_all_features=False,
    ).fit(features, treatment, outcome, modifier_indices=[1])

    y0, y1 = head.predict_potential_outcomes(features)
    assert np.corrcoef(y1 - y0, 0.3 + 1.4 * features[:, 1])[0, 1] > 0.99
    assert head.tuning_result_.selection_metric == "mean_squared_error"


def test_constant_outcome_and_input_validation_fail_closed():
    features = np.arange(60, dtype=float).reshape(30, 2)
    treatment = np.tile([0, 1], 15)
    head = StructuredInteractionHead(inner_folds=3).fit(
        features, treatment, np.zeros(30)
    )
    assert np.all(head.predict_effect(features) == 0.0)

    with pytest.raises(ValueError, match="binary"):
        StructuredInteractionHead().fit(features, np.full(30, 0.5), np.zeros(30))
    with pytest.raises(ValueError, match="non-finite"):
        bad = features.copy()
        bad[0, 0] = np.nan
        StructuredInteractionHead().fit(bad, treatment, np.zeros(30))
    with pytest.raises(ValueError, match="modifier_indices"):
        StructuredInteractionHead(interact_all_features=False).fit(
            features, treatment, np.zeros(30)
        )


def test_explicit_evaluator_freezes_outer_predictions_without_oracle_columns():
    rng = np.random.default_rng(23)
    n = 900
    inlet_pressure = rng.normal(65.0, 8.0, size=n)
    surface_index = rng.normal(size=n)
    pressure_z = (inlet_pressure - inlet_pressure.mean()) / inlet_pressure.std()
    propensity = 1.0 / (1.0 + np.exp(-0.8 * pressure_z))
    treatment = rng.binomial(1, propensity)
    baseline = -0.6 + 0.5 * pressure_z + 0.15 * surface_index
    treated = baseline + 0.2 + 1.1 * surface_index
    y0 = 1.0 / (1.0 + np.exp(-baseline))
    y1 = 1.0 / (1.0 + np.exp(-treated))
    outcome = rng.binomial(1, np.where(treatment == 1, y1, y0))
    frame = pd.DataFrame(
        {
            "treatment_indicator": treatment,
            "outcome_indicator": outcome,
            "explicit_feat_inlet_pressure": inlet_pressure,
            "explicit_feat_inlet_pressure_missing": False,
            "explicit_feat_surface_index": surface_index,
            "explicit_feat_surface_index_missing": False,
            "true_ite_prob": y1 - y0,
        }
    )
    specs = [
        ExplicitFeatureSpec(
            name="inlet_pressure", type="continuous", roles=["confounder"]
        ),
        ExplicitFeatureSpec(
            name="surface_index", type="continuous", roles=["effect_modifier"]
        ),
    ]
    config = AppliedInferenceConfig(
        outcome_type="binary",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
    )
    evaluator = StructuredInteractionExplicitEvaluator(
        config,
        ExplicitFeatureForestConfig(
            interaction_regularization_grid=[0.1, 0.3, 1.0],
            interaction_inner_folds=3,
            interaction_interact_all_features=False,
        ),
    )
    evaluation = evaluator.evaluate_split(
        frame.iloc[:700], frame.iloc[700:], specs, fold_id=4
    )

    assert "true_ite_prob" not in evaluation.predictions
    assert evaluation.metrics["effect_estimator"] == "interaction_s_learner"
    assert evaluation.metrics["interaction_inner_folds"] == 3
    assert np.corrcoef(
        evaluation.predictions["pred_ite_prob"], frame.iloc[700:]["true_ite_prob"]
    )[0, 1] > 0.9
