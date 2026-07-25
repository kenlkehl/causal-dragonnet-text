from __future__ import annotations

import copy

import pytest

from oci.inference.post_extraction_scientific_policy import (
    POST_EXTRACTION_SCIENTIFIC_POLICY_VERSION,
    PostExtractionScientificPolicy,
)


def _mapping() -> dict[str, object]:
    return {
        "schema_version": POST_EXTRACTION_SCIENTIFIC_POLICY_VERSION,
        "extraction_quality": {
            "minimum_coverage": 0.05,
            "maximum_unknown_category_rate": 0.05,
            "continuous_outlier_minimum_rows": 8,
            "continuous_outlier_iqr_multiplier": 6.0,
            "continuous_outlier_warning_rate": 0.10,
            "fold_coverage_range_warning": 0.35,
            "fold_continuous_scale_epsilon": 1e-8,
        },
        "extraction_redundancy": {
            "association_threshold": 0.80,
            "missingness_jaccard_threshold": 0.90,
            "minimum_pairwise_complete_rows": 3,
        },
        "extraction_grounding": {
            "anchor_group_selection": "all_source_attested_unbounded",
            "maximum_group_span_chars": 96,
            "anchor_value_window_chars": 96,
            "category_assertion_prefix_chars": 64,
            "unit_window_min_chars": 12,
            "unit_window_max_chars": 32,
            "unit_window_divisor": 3,
            "minimum_evaluable_rows": 3,
            "maximum_alternative_category_only_rate": 0.50,
            "unsupported_value_warning_rate": 0.25,
            "minimum_unit_support_rate": 0.50,
        },
        "review_estimator": {
            "standardization_scale_epsilon": 1e-8,
            "logistic_alpha_floor": 1e-12,
            "logistic_solver": "liblinear",
            "logistic_max_iter": 1000,
            "logistic_random_seed": 0,
            "logistic_fit_intercept": True,
            "logistic_class_weight": None,
            "binary_no_features_fallback": "prevalence",
            "binary_single_class_fallback": "prevalence",
            "binary_fit_failure_policy": "prevalence",
            "continuous_minimum_fit_rows": 2,
            "continuous_degenerate_fallback": "mean",
            "effect_minimum_usable_rows": 2,
            "effect_no_usable_fallback": "zero",
            "effect_degenerate_fallback": "weighted_mean",
            "ridge_solver": "auto",
            "ridge_fit_intercept": True,
            "ridge_tolerance": 1e-4,
            "ridge_max_iter": None,
            "ridge_positive": False,
            "ridge_random_seed": None,
        },
    }


def test_complete_policy_round_trips_without_defaults() -> None:
    expected = _mapping()
    policy = PostExtractionScientificPolicy.from_mapping(expected)
    assert policy.as_dict() == expected


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("extraction_quality", "minimum_coverage"),
        ("extraction_redundancy", "association_threshold"),
        ("extraction_grounding", "anchor_value_window_chars"),
        ("review_estimator", "logistic_solver"),
    ],
)
def test_every_nested_section_is_closed(section: str, field: str) -> None:
    missing = copy.deepcopy(_mapping())
    del missing[section][field]  # type: ignore[index]
    with pytest.raises(ValueError, match="missing"):
        PostExtractionScientificPolicy.from_mapping(missing)

    extra = copy.deepcopy(_mapping())
    extra[section]["implicit_default"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="extra"):
        PostExtractionScientificPolicy.from_mapping(extra)


def test_top_level_policy_is_closed_and_versioned() -> None:
    missing = _mapping()
    del missing["review_estimator"]
    with pytest.raises(ValueError, match="missing"):
        PostExtractionScientificPolicy.from_mapping(missing)

    extra = _mapping()
    extra["hidden"] = {}
    with pytest.raises(ValueError, match="extra"):
        PostExtractionScientificPolicy.from_mapping(extra)

    wrong_version = _mapping()
    wrong_version["schema_version"] = "older"
    with pytest.raises(ValueError, match="unsupported"):
        PostExtractionScientificPolicy.from_mapping(wrong_version)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("extraction_quality", "minimum_coverage", 1.01, r"\[0, 1\]"),
        (
            "extraction_quality",
            "continuous_outlier_minimum_rows",
            True,
            "integer",
        ),
        (
            "extraction_redundancy",
            "minimum_pairwise_complete_rows",
            1,
            "integer",
        ),
        (
            "extraction_grounding",
            "anchor_group_selection",
            "top_four",
            "preserve all",
        ),
        (
            "extraction_grounding",
            "unsupported_value_warning_rate",
            -0.1,
            r"\[0, 1\]",
        ),
        (
            "review_estimator",
            "binary_fit_failure_policy",
            "silent_missing",
            "must be one of",
        ),
        (
            "review_estimator",
            "ridge_tolerance",
            0.0,
            "positive",
        ),
    ],
)
def test_invalid_scientific_policies_fail_closed(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    mapping = _mapping()
    mapping[section][field] = value  # type: ignore[index]
    with pytest.raises(ValueError, match=message):
        PostExtractionScientificPolicy.from_mapping(mapping)
