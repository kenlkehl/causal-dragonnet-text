import logging

import numpy as np

import oci.models.causal_forest_head as causal_forest_head


class FakeNuisanceModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCausalForestDML:
    instances = []
    fail_tune = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.calls = []
        self.__class__.instances.append(self)

    def tune(self, **kwargs):
        self.calls.append(("tune", kwargs))
        if self.fail_tune:
            raise RuntimeError("tuning failed")
        return self

    def fit(self, **kwargs):
        self.calls.append(("fit", kwargs))
        return self


def _patch_econml_dependencies(monkeypatch):
    FakeCausalForestDML.instances = []
    FakeCausalForestDML.fail_tune = False
    monkeypatch.setattr(causal_forest_head, "ECONML_AVAILABLE", True)
    monkeypatch.setattr(causal_forest_head, "CausalForestDML", FakeCausalForestDML)
    monkeypatch.setattr(
        causal_forest_head, "RandomForestClassifier", FakeNuisanceModel, raising=False
    )
    monkeypatch.setattr(
        causal_forest_head, "RandomForestRegressor", FakeNuisanceModel, raising=False
    )


def test_causal_forest_head_tunes_before_fit(monkeypatch):
    _patch_econml_dependencies(monkeypatch)
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    W = np.array([[0.2], [0.4], [0.6], [0.8]])
    T = np.array([[0], [1], [0], [1]])
    Y = np.array([[0], [1], [1], [0]])

    head = causal_forest_head.CausalForestHead(n_estimators=8, min_samples_leaf=2)
    head.fit(X=X, W=W, T=T, Y=Y)

    model = head.model
    assert model.calls[0][0] == "tune"
    assert model.calls[0][1]["params"] == "auto"
    assert model.calls[1][0] == "fit"
    np.testing.assert_array_equal(model.calls[0][1]["T"], T.flatten())
    np.testing.assert_array_equal(model.calls[0][1]["Y"], Y.flatten())
    np.testing.assert_array_equal(model.calls[0][1]["X"], X)
    np.testing.assert_array_equal(model.calls[0][1]["W"], W)
    np.testing.assert_array_equal(model.calls[1][1]["W"], W)
    audit = head.fit_audit()
    assert audit["tuning_attempted"] is True
    assert audit["tuning_succeeded"] is True
    assert audit["tuning_failure_fell_back_to_configured_parameters"] is False
    assert audit["effective_parameters"]["n_estimators"] == 8


def test_causal_forest_head_warns_rebuilds_and_fits_when_tuning_fails(monkeypatch, caplog):
    _patch_econml_dependencies(monkeypatch)
    FakeCausalForestDML.fail_tune = True
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    W = np.array([[1.0], [1.0], [0.0], [0.0]])
    T = np.array([0, 1, 0, 1])
    Y = np.array([0, 1, 1, 0])

    head = causal_forest_head.CausalForestHead(n_estimators=8, min_samples_leaf=2)

    with caplog.at_level(logging.WARNING):
        head.fit(X=X, W=W, T=T, Y=Y)

    first_model, second_model = FakeCausalForestDML.instances
    assert first_model.calls[0][0] == "tune"
    assert second_model is head.model
    assert second_model.calls[0][0] == "fit"
    np.testing.assert_array_equal(second_model.calls[0][1]["W"], W)
    assert "CausalForestDML hyperparameter tuning failed" in caplog.text
    audit = head.fit_audit()
    assert audit["tuning_attempted"] is True
    assert audit["tuning_succeeded"] is False
    assert audit["tuning_failure_fell_back_to_configured_parameters"] is True
    assert audit["effective_parameters"] == audit["configured_parameters"]


def test_causal_forest_head_can_use_fixed_configuration_without_tuning(monkeypatch):
    _patch_econml_dependencies(monkeypatch)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    W = np.array([[1.0], [1.0], [0.0], [0.0]])
    T = np.array([0, 1, 0, 1])
    Y = np.array([0, 1, 1, 0])

    head = causal_forest_head.CausalForestHead(
        n_estimators=8,
        min_samples_leaf=2,
        tune_model=False,
    )
    head.fit(X=X, W=W, T=T, Y=Y)

    assert len(FakeCausalForestDML.instances) == 1
    assert head.model.calls[0][0] == "fit"
    audit = head.fit_audit()
    assert audit["tuning_attempted"] is False
    assert audit["tuning_succeeded"] is None
    assert audit["tuning_failure_fell_back_to_configured_parameters"] is False


def test_causal_forest_head_forwards_and_audits_explicit_nuisance_settings(
    monkeypatch,
):
    _patch_econml_dependencies(monkeypatch)
    head = causal_forest_head.CausalForestHead(
        n_estimators=12,
        max_depth=7,
        min_samples_leaf=3,
        max_features=0.75,
        honest=True,
        inference=True,
        random_state=19,
        tune_model=False,
        subforest_size=4,
        nuisance_n_estimators=9,
        nuisance_max_depth=5,
        nuisance_min_samples_leaf=2,
        nuisance_treatment_max_features="sqrt",
        nuisance_outcome_max_features=1.0,
        n_jobs=3,
    )
    head.fit(
        X=np.arange(8, dtype=float).reshape(4, 2),
        W=np.arange(4, dtype=float).reshape(4, 1),
        T=np.array([0, 1, 0, 1]),
        Y=np.array([0, 1, 1, 0]),
    )

    model = head.model
    assert model.kwargs["subforest_size"] == 4
    assert model.kwargs["n_jobs"] == 3
    assert model.kwargs["model_t"].kwargs == {
        "n_estimators": 9,
        "max_depth": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 19,
        "n_jobs": 3,
    }
    assert model.kwargs["model_y"].kwargs["max_features"] == 1.0
    audit = head.fit_audit()
    assert audit["configured_parameters"] == audit["effective_parameters"]
    assert audit["configured_nuisance_parameters"] == audit["effective_nuisance_parameters"]
    assert audit["operational_parameters"] == {"n_jobs": 3}


def test_tune_causal_forest_model_returns_false_on_failure(caplog):
    FakeCausalForestDML.instances = []
    FakeCausalForestDML.fail_tune = True
    model = FakeCausalForestDML()

    with caplog.at_level(logging.WARNING):
        tuned = causal_forest_head.tune_causal_forest_model(
            model,
            Y=np.array([0, 1]),
            T=np.array([0, 1]),
            X=np.array([[0.0], [1.0]]),
            W=np.array([[1.0], [0.0]]),
        )

    assert tuned is False
    assert model.calls[0][0] == "tune"
    assert "W" in model.calls[0][1]
    assert "fitting with configured hyperparameters" in caplog.text
