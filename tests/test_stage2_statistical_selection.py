from __future__ import annotations

import math

import numpy as np
import pandas as pd

from oci.inference.stage2_statistical_selection import select_stage2_features


def _definition(
    feature_id: str,
    name: str,
    *,
    value_type: str = "continuous",
    categories: list[str] | None = None,
    roles: list[str] | None = None,
    explicit: bool = False,
) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "name": name,
        "description": f"Pretreatment {name}.",
        "value_type": value_type,
        "categories_or_unit": categories or (["score"] if value_type == "continuous" else []),
        "measurement_definition": f"Extract {name} before treatment.",
        "missing_value_rule": "Use null when undocumented.",
        "roles": roles or [],
        "configured_explicit_feature": explicit,
    }


def test_inner_fold_screens_select_confounder_modifier_and_locked_overrides():
    rng = np.random.default_rng(217)
    rows = 800
    confounder = rng.normal(size=rows)
    modifier = rng.integers(0, 2, size=rows)
    noise = rng.normal(size=rows)
    locked_confounder = rng.normal(size=rows)
    locked_modifier = rng.integers(0, 2, size=rows)

    treatment_probability = 1.0 / (1.0 + np.exp(-1.35 * confounder))
    treatment = rng.binomial(1, treatment_probability)
    outcome_logit = (
        -1.0
        + 0.75 * confounder
        + 0.45 * treatment
        + 1.8 * treatment * modifier
    )
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-outcome_logit)))
    dataset = pd.DataFrame({"treatment": treatment, "outcome": outcome})
    extracted = pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "baseline_severity": confounder,
            "binary_biomarker": np.where(modifier == 1, "present", "absent"),
            "noise_measure": noise,
            "investigator_confounder": locked_confounder,
            "investigator_modifier": np.where(
                locked_modifier == 1,
                "present",
                "absent",
            ),
        }
    )
    definitions = [
        _definition("f1", "baseline_severity"),
        _definition(
            "f2",
            "binary_biomarker",
            value_type="binary",
            categories=["absent", "present"],
        ),
        _definition("f3", "noise_measure"),
        _definition(
            "f4",
            "investigator_confounder",
            roles=["confounder"],
            explicit=True,
        ),
        _definition(
            "f5",
            "investigator_modifier",
            value_type="binary",
            categories=["absent", "present"],
            roles=["effect_modifier"],
            explicit=True,
        ),
    ]
    inner_splits = []
    all_ids = np.arange(rows)
    for fold_index, heldout in enumerate(np.array_split(all_ids, 4), start=1):
        fit = np.setdiff1d(all_ids, heldout, assume_unique=True)
        inner_splits.append(
            {
                "inner_fold": fold_index,
                "fit_row_ids": fit.tolist(),
                "heldout_row_ids": heldout.tolist(),
            }
        )

    selected, report = select_stage2_features(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=definitions,
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        confounder_p_value_threshold=0.01,
        confounder_min_inner_fold_fraction=0.75,
        effect_modifier_p_value_threshold=0.01,
        effect_modifier_min_inner_fold_fraction=0.75,
        workers=2,
    )

    selected_by_name = {str(feature["name"]): feature for feature in selected}
    assert selected_by_name["baseline_severity"]["roles"] == ["confounder"]
    assert selected_by_name["binary_biomarker"]["roles"] == ["effect_modifier"]
    assert selected_by_name["investigator_confounder"]["roles"] == ["confounder"]
    assert selected_by_name["investigator_modifier"]["roles"] == ["effect_modifier"]
    assert "noise_measure" not in selected_by_name

    policy = report["policy"]
    assert policy["confounder_required_votes"] == math.ceil(0.75 * 4)
    assert policy["effect_modifier_required_votes"] == math.ceil(0.75 * 4)
    assert report["confounder_screen"]["votes"]["f1"] == 4
    assert report["effect_modifier_screen"]["votes"]["f2"] == 4
    assert set(report["effect_modifier_screen"]["adjustment_feature_ids"]) == {"f1", "f4"}
    assert report["parallelization"]["backend"] == "loky"
    assert report["parallelization"]["requested_workers"] == 2
    assert report["parallelization"]["confounder_effective_workers"] == 2
    decisions = {row["feature_id"]: row for row in report["decisions"]}
    assert decisions["f4"]["selection_source"] == "investigator_locked"
    assert decisions["f5"]["selection_source"] == "investigator_locked"
    for fold in report["confounder_screen"]["folds"]:
        treatment_p_values = [
            row["p_value"] for row in fold["treatment_p_value_ranking"]
        ]
        outcome_p_values = [row["p_value"] for row in fold["outcome_p_value_ranking"]]
        assert treatment_p_values == sorted(treatment_p_values)
        assert outcome_p_values == sorted(outcome_p_values)


def test_non_evaluable_candidates_do_not_receive_selection_votes():
    rows = 24
    dataset = pd.DataFrame(
        {
            "treatment": [0, 1] * (rows // 2),
            "outcome": [0, 0, 1, 1] * (rows // 4),
        }
    )
    extracted = pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "constant_candidate": [1.0] * rows,
        }
    )
    definitions = [_definition("constant", "constant_candidate")]
    inner_splits = [
        {
            "inner_fold": 1,
            "fit_row_ids": list(range(12)),
            "heldout_row_ids": list(range(12, 24)),
        },
        {
            "inner_fold": 2,
            "fit_row_ids": list(range(12, 24)),
            "heldout_row_ids": list(range(12)),
        },
    ]

    selected, report = select_stage2_features(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=definitions,
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        confounder_p_value_threshold=0.05,
        confounder_min_inner_fold_fraction=0.5,
        effect_modifier_p_value_threshold=0.05,
        effect_modifier_min_inner_fold_fraction=0.5,
    )

    assert selected == []
    assert report["confounder_screen"]["votes"] == {"constant": 0}
    assert report["effect_modifier_screen"]["votes"] == {"constant": 0}
    assert report["policy"]["non_evaluable_counts_as_vote"] is False


def test_categorical_modifier_uses_all_level_interactions_in_one_omnibus_test():
    rng = np.random.default_rng(991)
    rows = 3_000
    levels = np.asarray(["A", "B", "C"] * (rows // 3))
    rng.shuffle(levels)
    treatment = rng.binomial(1, 0.5, size=rows)
    interaction = np.where(levels == "B", 2.1, np.where(levels == "C", -2.1, 0.0))
    outcome_logit = -0.5 + 0.4 * treatment + treatment * interaction
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-outcome_logit)))
    dataset = pd.DataFrame({"treatment": treatment, "outcome": outcome})
    extracted = pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "three_level_marker": levels,
        }
    )
    definitions = [
        _definition(
            "marker",
            "three_level_marker",
            value_type="categorical",
            categories=["A", "B", "C"],
        )
    ]
    all_ids = np.arange(rows)
    inner_splits = []
    for fold_index, heldout in enumerate(np.array_split(all_ids, 3), start=1):
        fit = np.setdiff1d(all_ids, heldout, assume_unique=True)
        inner_splits.append(
            {
                "inner_fold": fold_index,
                "fit_row_ids": fit.tolist(),
                "heldout_row_ids": heldout.tolist(),
            }
        )

    selected, report = select_stage2_features(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=definitions,
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        confounder_p_value_threshold=0.001,
        confounder_min_inner_fold_fraction=1.0,
        effect_modifier_p_value_threshold=0.001,
        effect_modifier_min_inner_fold_fraction=1.0,
        workers=2,
    )

    assert selected[0]["roles"] == ["effect_modifier"]
    assert report["effect_modifier_screen"]["votes"] == {"marker": 3}
    for fold in report["effect_modifier_screen"]["folds"]:
        test = fold["tests"][0]["interaction_test"]
        assert test["test"] == "likelihood_ratio_chi_square"
        assert test["degrees_of_freedom"] == 2
        assert test["categorical_reference_level"] == "A"
        assert test["candidate_interaction_columns"] == [
            "three_level_marker:level=B",
            "three_level_marker:level=C",
        ]
        assert test["tested_interaction_columns"] == [
            "three_level_marker:level=B",
            "three_level_marker:level=C",
        ]
