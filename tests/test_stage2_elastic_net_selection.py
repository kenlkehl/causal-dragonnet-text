from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

import oci.inference.stage2_elastic_net_selection as selection_module
from oci.inference.stage2_elastic_net_selection import (
    SCHEMA_VERSION,
    Stage2ElasticNetSelectionConfig,
    _encode_design,
    select_stage2_features_elastic_net,
)


def _continuous(feature_id: str, name: str) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "name": name,
        "description": name,
        "value_type": "continuous",
        "modeling_strategy": "continuous",
        "categories_or_unit": ["unitless"],
        "measurement_definition": name,
        "missing_value_rule": "null when absent",
    }


def _categorical(feature_id: str, name: str, levels: list[str]) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "name": name,
        "description": name,
        "value_type": "categorical",
        "categories_or_unit": levels,
        "measurement_definition": name,
        "missing_value_rule": "null when absent",
    }


def test_any_fold_nuisance_union_and_univariable_modifier_selection():
    rng = np.random.default_rng(141)
    rows = 700
    confounder = rng.normal(size=rows)
    treatment_only = rng.normal(size=rows)
    outcome_only = rng.normal(size=rows)
    modifier = rng.normal(size=rows)
    noise = rng.normal(size=rows)
    treatment_probability = 1.0 / (
        1.0 + np.exp(-(1.5 * confounder + 1.4 * treatment_only))
    )
    treatment = rng.binomial(1, treatment_probability)
    outcome = (
        1.8 * confounder
        + 1.7 * outcome_only
        + treatment * (2.4 * modifier)
        + 0.45 * noise
    )
    dataset = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
        }
    )
    extracted = pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "confounder": confounder,
            "treatment_only": treatment_only,
            "outcome_only": outcome_only,
            "modifier": modifier,
            "noise": noise,
        }
    )
    definitions = [
        _continuous("conf", "confounder"),
        _continuous("t_only", "treatment_only"),
        _continuous("y_only", "outcome_only"),
        _continuous("mod", "modifier"),
        _continuous("noise", "noise"),
    ]
    splitter = KFold(n_splits=5, shuffle=True, random_state=19)
    inner_splits = [
        {
            "inner_fold": index,
            "fit_row_ids": fit.tolist(),
            "heldout_row_ids": heldout.tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(extracted), start=1)
    ]
    policy = Stage2ElasticNetSelectionConfig(
        l1_ratio=0.9,
        internal_cv_folds=3,
        regularization_grid_size=8,
        nuisance_forest_trees=60,
        nuisance_forest_min_samples_leaf=6,
        modifier_top_n_per_inner_fold=1,
        max_iter=3_000,
    )

    selected, report, dependencies, latent_states = select_stage2_features_elastic_net(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=definitions,
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="continuous",
        seed=91,
        policy=policy,
    )

    selected_by_id = {row["feature_id"]: row for row in selected}
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["latent_construction"] == "disabled"
    assert report["pairwise_association_screen"] == "disabled"
    assert report["nuisance_screen"]["intersection_is_not_a_selection_gate"] is True
    assert "treatment" in selected_by_id["conf"]["nuisance_model_roles"]
    assert "outcome" in selected_by_id["conf"]["nuisance_model_roles"]
    assert selected_by_id["t_only"]["nuisance_model_roles"] == [
        "treatment",
        "outcome",
    ]
    assert selected_by_id["y_only"]["nuisance_model_roles"] == [
        "treatment",
        "outcome",
    ]
    assert (
        report["nuisance_screen"]["stable_treatment_feature_ids"]
        != report["nuisance_screen"]["stable_outcome_feature_ids"]
    )
    assert report["nuisance_screen"]["required_votes"] == 1
    assert report["nuisance_screen"]["union_is_used_by_both_nuisance_models"] is True
    assert (
        report["cross_fitted_nuisance_models"]["treatment_feature_ids"]
        == report["cross_fitted_nuisance_models"]["outcome_feature_ids"]
        == report["nuisance_screen"]["union_confounder_feature_ids"]
    )
    assert report["nuisance_screen"]["overall_treatment_auroc"] is not None
    assert report["nuisance_screen"]["overall_outcome_auroc"] is None
    for fold in report["nuisance_screen"]["folds"]:
        assert fold["treatment"]["heldout_auroc"] is not None
        assert fold["outcome"]["heldout_auroc"] is None
    assert "effect_modifier" in selected_by_id["mod"]["roles"]
    assert report["effect_modifier_screen"]["model_family"] == (
        "gaussian_linear_regression"
    )
    assert report["effect_modifier_screen"]["top_n_per_inner_fold"] == 1
    assert report["effect_modifier_screen"]["required_votes"] == 1
    assert report["effect_modifier_screen"]["p_value_threshold_is_not_a_selection_gate"]
    assert all(
        fold["selected_count"] == 1
        and fold["nuisance_predictions_are_cross_fitted"] is True
        for fold in report["effect_modifier_screen"]["folds"]
    )
    assert report["cross_fitted_nuisance_models"][
        "predictions_are_inner_fold_out_of_fold"
    ] is True
    assert [row["feature_id"] for row in dependencies] == [
        row["feature_id"] for row in selected
    ]
    assert latent_states == []


def test_investigator_locked_roles_are_preserved_without_latents():
    rng = np.random.default_rng(22)
    rows = 160
    treatment = rng.binomial(1, 0.5, size=rows)
    dataset = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": rng.normal(size=rows),
        }
    )
    extracted = pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "locked": rng.normal(size=rows),
        }
    )
    definition = {
        **_continuous("locked", "locked"),
        "configured_explicit_feature": True,
        "roles": ["confounder", "effect_modifier"],
    }
    splitter = KFold(n_splits=4, shuffle=True, random_state=3)
    inner_splits = [
        {
            "inner_fold": index,
            "fit_row_ids": fit.tolist(),
            "heldout_row_ids": heldout.tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(extracted), start=1)
    ]
    policy = Stage2ElasticNetSelectionConfig(
        regularization_grid_size=5,
        nuisance_forest_trees=20,
        nuisance_forest_min_samples_leaf=5,
    )

    selected, _report, dependencies, latent_states = select_stage2_features_elastic_net(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=[definition],
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="continuous",
        seed=4,
        policy=policy,
    )

    assert selected[0]["roles"] == ["confounder", "effect_modifier"]
    assert selected[0]["nuisance_model_roles"] == ["treatment", "outcome"]
    assert dependencies[0]["feature_id"] == "locked"
    assert latent_states == []


def test_ordinal_measurement_is_one_standardized_score_not_dummy_columns():
    definition = {
        **_categorical("ecog", "ecog", ["0", "1", "2", "3"]),
        "value_type": "ordinal",
    }
    train = pd.DataFrame({"ecog": [0, 1, 2, 3, 1, 2, None, 0]})
    valid = pd.DataFrame({"ecog": [3, None, 1]})

    design = _encode_design(
        train,
        valid,
        [definition],
        categorical_min_count=1,
    )

    assert design.column_names == ("ecog:ordered_score", "ecog:missing")
    assert design.column_feature_ids == ("ecog", "ecog")
    np.testing.assert_allclose(np.mean(design.train, axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(design.train, axis=0), 1.0, atol=1e-12)


def test_nominal_factor_is_selected_as_one_group_in_both_nuisance_tasks():
    rng = np.random.default_rng(221)
    rows = 480
    levels = np.asarray(["A", "B", "C", "D"])
    category_index = rng.integers(0, len(levels), size=rows)
    category = levels[category_index]
    treatment_probability = 1.0 / (
        1.0 + np.exp(-np.asarray([-2.0, -0.7, 0.8, 2.0])[category_index])
    )
    treatment = rng.binomial(1, treatment_probability)
    outcome = np.asarray([-3.0, -1.0, 1.0, 3.0])[category_index] + rng.normal(
        scale=0.45,
        size=rows,
    )
    extracted = pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "factor": category,
            "noise": rng.normal(size=rows),
        }
    )
    dataset = pd.DataFrame({"treatment": treatment, "outcome": outcome})
    definitions = [
        _categorical("factor", "factor", levels.tolist()),
        _continuous("noise", "noise"),
    ]
    splitter = KFold(n_splits=4, shuffle=True, random_state=12)
    inner_splits = [
        {
            "inner_fold": index,
            "fit_row_ids": fit.tolist(),
            "heldout_row_ids": heldout.tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(extracted), start=1)
    ]
    policy = Stage2ElasticNetSelectionConfig(
        regularization_grid_size=7,
        nuisance_forest_trees=20,
        nuisance_forest_min_samples_leaf=5,
        max_iter=2_000,
    )

    selected, report, _dependencies, _latents = select_stage2_features_elastic_net(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=definitions,
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="continuous",
        seed=18,
        policy=policy,
    )

    selected_factor = next(row for row in selected if row["feature_id"] == "factor")
    assert selected_factor["nuisance_model_roles"] == ["treatment", "outcome"]
    assert report["penalized_model_family"] == "group_lasso_plus_ridge"
    assert report["encoding"]["missing_indicator"] == (
        "same_penalty_group_as_measurement"
    )
    for fold in report["nuisance_screen"]["folds"]:
        assert "factor" in fold["treatment"]["feature_group_l2_norms"]
        assert "factor" in fold["outcome"]["feature_group_l2_norms"]


def test_categorical_modifier_uses_one_grouped_interaction_p_value():
    rng = np.random.default_rng(733)
    rows = 900
    levels = np.asarray(["A", "B", "C"])
    category_index = rng.integers(0, len(levels), size=rows)
    category = levels[category_index]
    treatment = rng.binomial(1, 0.5, size=rows)
    interaction_effect = np.asarray([-2.8, 0.0, 2.8])[category_index]
    outcome_probability = 1.0 / (
        1.0 + np.exp(-(-0.3 + 0.15 * category_index + treatment * interaction_effect))
    )
    outcome = rng.binomial(1, outcome_probability)
    extracted = pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "factor": category,
            "noise": rng.normal(size=rows),
        }
    )
    dataset = pd.DataFrame({"treatment": treatment, "outcome": outcome})
    definitions = [
        _categorical("factor", "factor", levels.tolist()),
        _continuous("noise", "noise"),
    ]
    splitter = KFold(n_splits=3, shuffle=True, random_state=44)
    inner_splits = [
        {
            "inner_fold": index,
            "fit_row_ids": fit.tolist(),
            "heldout_row_ids": heldout.tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(extracted), start=1)
    ]

    selected, report, _dependencies, _latents = select_stage2_features_elastic_net(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=definitions,
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        seed=8,
        policy=Stage2ElasticNetSelectionConfig(
            regularization_grid_size=5,
            nuisance_forest_trees=20,
            nuisance_forest_min_samples_leaf=5,
            modifier_top_n_per_inner_fold=1,
            max_iter=2_000,
        ),
    )

    selected_factor = next(row for row in selected if row["feature_id"] == "factor")
    assert "effect_modifier" in selected_factor["roles"]
    assert report["effect_modifier_screen"]["model_family"] == (
        "binomial_logistic_regression"
    )
    for fold in report["effect_modifier_screen"]["folds"]:
        assert fold["selected_feature_ids"] == ["factor"]
        factor_test = next(
            row for row in fold["tests"] if row["feature_id"] == "factor"
        )
        assert factor_test["categorical_interactions_are_grouped"] is True
        assert len(factor_test["tested_interaction_columns"]) == 2
        assert factor_test["interaction_test"]["test"] == (
            "likelihood_ratio_chi_square"
        )
        assert factor_test["interaction_test"]["degrees_of_freedom"] == 2


def test_modifier_top_n_defaults_to_five_and_must_be_positive():
    assert Stage2ElasticNetSelectionConfig().modifier_top_n_per_inner_fold == 5
    with pytest.raises(ValueError, match="modifier_top_n_per_inner_fold"):
        Stage2ElasticNetSelectionConfig(modifier_top_n_per_inner_fold=0).validate()


def test_binary_nuisance_screens_report_inner_and_pooled_aurocs():
    rng = np.random.default_rng(808)
    rows = 300
    signal = rng.normal(size=rows)
    treatment = rng.binomial(1, 1.0 / (1.0 + np.exp(-1.8 * signal)))
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-1.4 * signal)))
    dataset = pd.DataFrame({"treatment": treatment, "outcome": outcome})
    extracted = pd.DataFrame(
        {"_oci_row_id": np.arange(rows), "signal": signal}
    )
    splitter = KFold(n_splits=3, shuffle=True, random_state=31)
    inner_splits = [
        {
            "inner_fold": index,
            "fit_row_ids": fit.tolist(),
            "heldout_row_ids": heldout.tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(extracted), start=1)
    ]

    _selected, report, _dependencies, _latents = select_stage2_features_elastic_net(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=[_continuous("signal", "signal")],
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        seed=73,
        policy=Stage2ElasticNetSelectionConfig(
            regularization_grid_size=5,
            nuisance_forest_trees=20,
            nuisance_forest_min_samples_leaf=5,
            max_iter=2_000,
        ),
    )

    assert 0.5 < report["nuisance_screen"]["overall_treatment_auroc"] <= 1.0
    assert 0.5 < report["nuisance_screen"]["overall_outcome_auroc"] <= 1.0
    assert 0.5 < report["cross_fitted_nuisance_models"][
        "overall_treatment_auroc"
    ] <= 1.0
    assert 0.5 < report["cross_fitted_nuisance_models"][
        "overall_outcome_auroc"
    ] <= 1.0
    for fold in report["nuisance_screen"]["folds"]:
        assert fold["treatment"]["heldout_auroc"] is not None
        assert fold["outcome"]["heldout_auroc"] is not None


def test_outer_modifier_set_is_union_of_each_inner_folds_top_n(monkeypatch):
    rng = np.random.default_rng(99)
    rows = 90
    treatment = np.tile([0, 1], rows // 2)
    dataset = pd.DataFrame(
        {"treatment": treatment, "outcome": rng.normal(size=rows)}
    )
    extracted = pd.DataFrame({
        "_oci_row_id": np.arange(rows),
        "candidate_a": rng.normal(size=rows),
        "candidate_b": rng.normal(size=rows),
        "candidate_c": rng.normal(size=rows),
    })
    splitter = KFold(n_splits=3, shuffle=True, random_state=7)
    inner_splits = [
        {
            "inner_fold": index,
            "fit_row_ids": fit.tolist(),
            "heldout_row_ids": heldout.tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(extracted), start=1)
    ]

    def constant_logistic(train_x, train_y, valid_x, _column_feature_ids, **_kwargs):
        probability = float(np.mean(train_y))
        return selection_module._PenalizedFit(
            coefficients=np.zeros(train_x.shape[1]),
            train_prediction=np.full(len(train_y), probability),
            valid_prediction=np.full(len(valid_x), probability),
            regularization=0.1,
            cv_folds=2,
            status="ok",
            iterations=1,
            converged=True,
        )

    def controlled_squared(
        train_x,
        train_y,
        valid_x,
        _column_feature_ids,
        *,
        fit_intercept=True,
        **_kwargs,
    ):
        coefficients = np.zeros(train_x.shape[1])
        train_prediction = np.full(len(train_y), float(np.mean(train_y)))
        valid_prediction = np.full(len(valid_x), float(np.mean(train_y)))
        return selection_module._PenalizedFit(
            coefficients=coefficients,
            train_prediction=train_prediction,
            valid_prediction=valid_prediction,
            regularization=0.1,
            cv_folds=2,
            status="ok",
            iterations=1,
            converged=True,
        )

    monkeypatch.setattr(selection_module, "_logistic_elastic_net", constant_logistic)
    monkeypatch.setattr(selection_module, "_squared_error_elastic_net", controlled_squared)
    p_values = iter(
        [
            0.01, 0.20, 0.30,
            0.20, 0.01, 0.30,
            0.30, 0.20, 0.01,
        ]
    )

    def controlled_interaction_test(*, feature, **_kwargs):
        p_value = next(p_values)
        return {
            "feature_id": feature["feature_id"],
            "name": feature["name"],
            "candidate_strategy": "continuous",
            "encoded_main_columns": [feature["name"]],
            "candidate_interaction_columns": [feature["name"]],
            "tested_interaction_columns": [feature["name"]],
            "categorical_interactions_are_grouped": False,
            "missingness_interactions": False,
            "interaction_p_value": p_value,
            "interaction_test": {
                "status": "ok",
                "test": "partial_f",
                "degrees_of_freedom": [1, 50],
                "tested_column_indices": [0],
            },
        }

    monkeypatch.setattr(
        selection_module,
        "_modifier_interaction_test",
        controlled_interaction_test,
    )

    selected, report, _dependencies, _latents = select_stage2_features_elastic_net(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=[
            _continuous("candidate_a", "candidate_a"),
            _continuous("candidate_b", "candidate_b"),
            _continuous("candidate_c", "candidate_c"),
        ],
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="continuous",
        seed=15,
        policy=Stage2ElasticNetSelectionConfig(
            regularization_grid_size=5,
            nuisance_forest_trees=20,
            nuisance_forest_min_samples_leaf=5,
            modifier_top_n_per_inner_fold=1,
        ),
    )

    selected_by_id = {row["feature_id"]: row for row in selected}
    assert all(
        "effect_modifier" in selected_by_id[feature_id]["roles"]
        for feature_id in ("candidate_a", "candidate_b", "candidate_c")
    )
    assert report["effect_modifier_screen"]["required_votes"] == 1
    assert report["effect_modifier_screen"]["votes"] == {
        "candidate_a": 1,
        "candidate_b": 1,
        "candidate_c": 1,
    }
    assert report["effect_modifier_screen"]["stable_effect_modifier_feature_ids"] == [
        "candidate_a",
        "candidate_b",
        "candidate_c",
    ]
    assert [fold["selected_count"] for fold in report["effect_modifier_screen"]["folds"]] == [
        1,
        1,
        1,
    ]
