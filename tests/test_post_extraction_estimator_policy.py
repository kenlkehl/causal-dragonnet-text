from __future__ import annotations

import numpy as np
import pytest

import oci.inference.all_evidence_post_extraction_review as review
from oci.inference.all_evidence_post_extraction_review import (
    CausalReviewConfig,
)
from oci.inference.post_extraction_scientific_policy import (
    ReviewEstimatorPolicy,
)


def _policy(**overrides) -> ReviewEstimatorPolicy:
    values = {
        "standardization_scale_epsilon": 1e-8,
        "logistic_alpha_floor": 1e-12,
        "logistic_solver": "liblinear",
        "logistic_max_iter": 1000,
        "logistic_random_seed": 17,
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
    }
    values.update(overrides)
    return ReviewEstimatorPolicy(**values)


def test_binary_constructor_receives_complete_configured_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class _Logistic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, x, y):
            assert x.shape == (4, 1)
            assert y.tolist() == [0, 1, 0, 1]
            return self

        def predict_proba(self, x):
            return np.column_stack(
                (np.full(len(x), 0.25), np.full(len(x), 0.75))
            )

    monkeypatch.setattr(review, "LogisticRegression", _Logistic)
    policy = _policy()
    result = review._fit_predict_binary(
        np.arange(4, dtype=float)[:, None],
        np.asarray([0.0, 1.0, 0.0, 1.0]),
        np.asarray([[4.0], [5.0]]),
        alpha=2.0,
        policy=policy,
    )

    assert captured == {
        "C": 0.5,
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": 17,
        "fit_intercept": True,
        "class_weight": None,
    }
    assert result.tolist() == [0.75, 0.75]


def test_binary_fit_failure_policy_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Failing:
        def __init__(self, **_kwargs):
            pass

        def fit(self, _x, _y):
            raise ValueError("configured failure")

    monkeypatch.setattr(review, "LogisticRegression", _Failing)
    args = (
        np.arange(4, dtype=float)[:, None],
        np.asarray([0.0, 1.0, 0.0, 1.0]),
        np.asarray([[4.0], [5.0]]),
    )
    fallback = review._fit_predict_binary(
        *args,
        alpha=1.0,
        policy=_policy(binary_fit_failure_policy="prevalence"),
    )
    assert fallback.tolist() == [0.5, 0.5]
    with pytest.raises(ValueError, match="configured failure"):
        review._fit_predict_binary(
            *args,
            alpha=1.0,
            policy=_policy(binary_fit_failure_policy="abort"),
        )


def test_ridge_constructor_receives_complete_configured_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class _Ridge:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, x, y):
            assert x.shape == (3, 1)
            assert y.shape == (3,)
            return self

        def predict(self, x):
            return np.full(len(x), 4.25)

    monkeypatch.setattr(review, "Ridge", _Ridge)
    result = review._fit_predict_continuous(
        np.arange(3, dtype=float)[:, None],
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([[3.0], [4.0]]),
        alpha=3.0,
        policy=_policy(
            ridge_solver="lsqr",
            ridge_fit_intercept=False,
            ridge_tolerance=0.002,
            ridge_max_iter=77,
            ridge_positive=False,
            ridge_random_seed=19,
        ),
    )

    assert captured == {
        "alpha": 3.0,
        "fit_intercept": False,
        "solver": "lsqr",
        "tol": 0.002,
        "max_iter": 77,
        "positive": False,
        "random_state": 19,
    }
    assert result.tolist() == [4.25, 4.25]


def test_estimator_policy_is_bound_into_causal_review_identity() -> None:
    left = review._causal_review_config_payload(
        CausalReviewConfig(estimator_policy=_policy())
    )
    right = review._causal_review_config_payload(
        CausalReviewConfig(
            estimator_policy=_policy(logistic_random_seed=18)
        )
    )

    assert left["estimator_policy"]["logistic_random_seed"] == 17
    assert right["estimator_policy"]["logistic_random_seed"] == 18
    assert left != right
