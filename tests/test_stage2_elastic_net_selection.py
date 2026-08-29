from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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


def test_separate_nuisance_supports_and_r_learner_modifier_selection():
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
        nuisance_selection_frequency=0.6,
        modifier_selection_frequency=0.6,
        internal_cv_folds=3,
        regularization_grid_size=8,
        nuisance_forest_trees=60,
        nuisance_forest_min_samples_leaf=6,
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
    assert selected_by_id["t_only"]["nuisance_model_roles"] == ["treatment"]
    assert "outcome" in selected_by_id["y_only"]["nuisance_model_roles"]
    assert (
        report["nuisance_screen"]["stable_treatment_feature_ids"]
        != report["nuisance_screen"]["stable_outcome_feature_ids"]
    )
    assert "effect_modifier" in selected_by_id["mod"]["roles"]
    assert report["effect_modifier_screen"]["set_passed_heldout_r_loss_gate"] is True
    assert report["effect_modifier_screen"]["mean_heldout_r_loss_improvement"] > 0
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
