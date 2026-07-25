from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from oci.config import (
    BoWViewConfig,
    ForestScientificConfig,
    LogisticRegressionScientificConfig,
    RidgeScientificConfig,
    TfidfNuisanceStackScientificConfig,
    TfidfVectorizerScientificConfig,
)
from oci.inference.multi_model_agentic_forest import (
    _bow_model_params,
    _bow_vectorizer_params,
    _fit_regressor,
    _make_bow_classifier,
    _make_bow_regressor,
    _make_bow_vectorizer,
)
from oci.inference.tfidf_topic_discovery import _fit_stack


def test_nondefault_vectorizer_science_reaches_constructor_exactly():
    scientific = TfidfVectorizerScientificConfig(
        strip_accents="unicode",
        lowercase=False,
        stop_words=["ignoredword"],
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
        ngram_range_min=2,
        ngram_range_max=4,
        max_df=1.0,
        min_df=2,
        max_features=73,
        binary=True,
        dtype="float64",
        norm="l1",
        use_idf=False,
        smooth_idf=False,
        sublinear_tf=False,
    )
    view = BoWViewConfig(
        name="nondefault_vectorizer",
        max_features=73,
        min_df=2,
        max_df=1.0,
        ngram_range_min=2,
        ngram_range_max=4,
        sublinear_tf=False,
        vectorizer_scientific=scientific,
    )

    vectorizer = _make_bow_vectorizer(_bow_vectorizer_params(view))
    params = vectorizer.get_params(deep=False)

    assert params["strip_accents"] == "unicode"
    assert params["lowercase"] is False
    assert params["stop_words"] == ["ignoredword"]
    assert params["ngram_range"] == (2, 4)
    assert params["max_features"] == 73
    assert params["binary"] is True
    assert params["dtype"] is np.float64
    assert params["norm"] == "l1"
    assert params["use_idf"] is False
    assert params["smooth_idf"] is False
    assert params["sublinear_tf"] is False


def test_linear_and_forest_science_reaches_constructors_exactly():
    view = BoWViewConfig(
        name="nondefault_models",
        logistic_c=0.37,
        logistic_max_iter=321,
        ridge_alpha=4.25,
        logistic_scientific=LogisticRegressionScientificConfig(
            tol=3e-5,
            fit_intercept=False,
            solver="liblinear",
            warm_start=True,
        ),
        ridge_scientific=RidgeScientificConfig(
            fit_intercept=False,
            max_iter=211,
            tol=7e-5,
            solver="lsqr",
        ),
    )
    classifier = _make_bow_classifier(_bow_model_params(view), random_state=19)
    regressor = _make_bow_regressor(_bow_model_params(view), random_state=23)
    assert classifier.get_params(deep=False)["C"] == pytest.approx(0.37)
    assert classifier.get_params(deep=False)["max_iter"] == 321
    assert classifier.get_params(deep=False)["fit_intercept"] is False
    assert classifier.get_params(deep=False)["tol"] == pytest.approx(3e-5)
    assert regressor.get_params(deep=False)["alpha"] == pytest.approx(4.25)
    assert regressor.get_params(deep=False)["max_iter"] == 211
    assert regressor.get_params(deep=False)["solver"] == "lsqr"

    tree_view = BoWViewConfig(
        name="nondefault_forest",
        bow_model="random_forest",
        forest_scientific=ForestScientificConfig(
            n_estimators=17,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features=0.7,
            random_forest_bootstrap=False,
            ccp_alpha=0.02,
        ),
    )
    tree = _make_bow_classifier(_bow_model_params(tree_view), random_state=29)
    assert tree.get_params(deep=False)["n_estimators"] == 17
    assert tree.get_params(deep=False)["max_depth"] == 4
    assert tree.get_params(deep=False)["bootstrap"] is False
    assert tree.get_params(deep=False)["ccp_alpha"] == pytest.approx(0.02)


def test_constructor_mappings_and_xgboost_availability_fail_closed(monkeypatch):
    params = _bow_vectorizer_params(BoWViewConfig(name="closed"))
    params.pop("norm")
    with pytest.raises(ValueError, match=r"missing=.*norm"):
        _make_bow_vectorizer(params)

    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "xgboost" else original_find_spec(name),
    )
    with pytest.raises(ValueError, match="xgboost is unavailable"):
        BoWViewConfig(name="unavailable_xgboost", bow_model="xgboost")


def test_weighted_regression_fallback_requires_named_legacy_policy():
    class NoWeightModel:
        def __init__(self):
            self.calls = 0

        def fit(self, _x, _y, **kwargs):
            self.calls += 1
            if kwargs:
                raise TypeError("sample_weight unsupported")
            return self

    x = np.ones((4, 1))
    y = np.arange(4, dtype=float)
    weights = np.ones(4)
    with pytest.raises(TypeError, match="does not accept configured sample weights"):
        _fit_regressor(
            NoWeightModel(),
            x,
            y,
            sample_weight=weights,
            unsupported_sample_weight_policy="fail_closed",
        )
    legacy = NoWeightModel()
    assert (
        _fit_regressor(
            legacy,
            x,
            y,
            sample_weight=weights,
            unsupported_sample_weight_policy="unweighted_legacy_compatibility",
        )
        is legacy
    )
    assert legacy.calls == 2


def test_meta_stack_science_reaches_constructor():
    config = TfidfNuisanceStackScientificConfig(
        meta_logistic_c=0.23,
        meta_logistic_max_iter=444,
        meta_logistic=LogisticRegressionScientificConfig(
            solver="liblinear",
            tol=8e-5,
            fit_intercept=False,
        ),
    )
    matrix = np.column_stack(
        [np.linspace(0.1, 0.9, 20), np.tile([0.2, 0.8], 10)]
    )
    labels = np.tile([0.0, 1.0], 10)

    model, _constant = _fit_stack(
        matrix,
        labels,
        binary=True,
        seed=31,
        config=config,
    )

    assert model is not None
    assert model.get_params(deep=False)["C"] == pytest.approx(0.23)
    assert model.get_params(deep=False)["max_iter"] == 444
    assert model.get_params(deep=False)["tol"] == pytest.approx(8e-5)
    assert model.get_params(deep=False)["fit_intercept"] is False
