import copy
import inspect

import numpy as np
import pytest
from econml.dml import CausalForestDML
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

from oci.models.causal_forest_head import CausalForestHead
from oci.models.elastic_net_nuisance import (
    ElasticNetLogisticClassifier,
    ElasticNetRegressor,
)
from oci.models.strict_causal_forest_runtime import (
    CAUSAL_FOREST_IMPLEMENTATION,
    OUTCOME_FOREST_IMPLEMENTATION,
    STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
    STRATIFIED_CROSSFIT_IMPLEMENTATION,
    TREATMENT_FOREST_IMPLEMENTATION,
    StrictCausalForestRuntimeConfig,
    assert_supported_constructor_signatures,
)


def _runtime_mapping(*, n_jobs=1):
    return {
        "schema_version": STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
        "causal_forest": {
            "implementation": CAUSAL_FOREST_IMPLEMENTATION,
            "tune_model": False,
            "featurizer": None,
            "treatment_featurizer": None,
            "discrete_outcome": False,
            "discrete_treatment": True,
            "categories": "auto",
            "crossfit": {
                "implementation": STRATIFIED_CROSSFIT_IMPLEMENTATION,
                "n_splits": 2,
                "shuffle": True,
                "random_seed": 42,
            },
            "mc_iters": None,
            "mc_agg": "mean",
            "drate": True,
            "n_estimators": 8,
            "criterion": "mse",
            "max_depth": None,
            "min_samples_split": 10,
            "min_samples_leaf": 2,
            "min_weight_fraction_leaf": 0.0,
            "min_var_fraction_leaf": None,
            "min_var_leaf_on_val": False,
            "max_features": "sqrt",
            "min_impurity_decrease": 0.0,
            "max_samples": 0.45,
            "min_balancedness_tol": 0.45,
            "honest": True,
            "inference": True,
            "fit_intercept": True,
            "subforest_size": 4,
            "random_seed": 42,
            "allow_missing": False,
            "treatment_model": {
                "implementation": TREATMENT_FOREST_IMPLEMENTATION,
                "n_estimators": 5,
                "criterion": "gini",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 2,
                "min_weight_fraction_leaf": 0.0,
                "max_features": "sqrt",
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.0,
                "bootstrap": True,
                "oob_score": False,
                "random_seed": 42,
                "warm_start": False,
                "class_weight": None,
                "ccp_alpha": 0.0,
                "max_samples": None,
                "monotonic_cst": None,
            },
            "outcome_model": {
                "implementation": OUTCOME_FOREST_IMPLEMENTATION,
                "n_estimators": 5,
                "criterion": "squared_error",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 2,
                "min_weight_fraction_leaf": 0.0,
                "max_features": 1.0,
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.0,
                "bootstrap": True,
                "oob_score": False,
                "random_seed": 42,
                "warm_start": False,
                "ccp_alpha": 0.0,
                "max_samples": None,
                "monotonic_cst": None,
            },
        },
        "operational": {
            "requested_host_cpu_budget": n_jobs,
            "verbose": 0,
            "use_ray": False,
            "ray_remote_func_options": None,
        },
    }


def _runtime(*, n_jobs=1):
    return StrictCausalForestRuntimeConfig.from_mapping(_runtime_mapping(n_jobs=n_jobs))


def _data():
    rng = np.random.RandomState(91)
    n = 96
    effect = rng.normal(size=(n, 3))
    control = rng.normal(size=(n, 2))
    treatment = np.tile(np.array([0, 1]), n // 2)
    outcome = (
        0.25 * treatment + effect[:, 0] + control[:, 0] + rng.normal(scale=0.8, size=n) > 0.1
    ).astype(float)
    return effect, control, treatment, outcome


def test_runtime_config_is_closed_round_trippable_and_path_neutral():
    config = _runtime(n_jobs=1)
    assert StrictCausalForestRuntimeConfig.from_mapping(config.as_dict()) == config

    extra = config.as_dict()
    extra["causal_forest"]["unclassified_parameter"] = 1
    with pytest.raises(ValueError, match="closed schema"):
        StrictCausalForestRuntimeConfig.from_mapping(extra)

    missing = config.as_dict()
    del missing["causal_forest"]["outcome_model"]["criterion"]
    with pytest.raises(ValueError, match="closed schema"):
        StrictCausalForestRuntimeConfig.from_mapping(missing)

    other_jobs = _runtime(n_jobs=3)
    assert config.scientific_identity() == other_jobs.scientific_identity()
    assert config.scientific_identity_sha256() == other_jobs.scientific_identity_sha256()
    assert config.operational_attestation() != other_jobs.operational_attestation()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("causal_forest", "tune_model"), True, "tune_model=false"),
        (("causal_forest", "honest"), False, "honest=true"),
        (("causal_forest", "max_samples"), 0.75, "cannot exceed 0.5"),
        (
            ("causal_forest", "treatment_model", "warm_start"),
            True,
            "warm_start=false",
        ),
        (
            ("causal_forest", "outcome_model", "oob_score"),
            True,
            "oob_score=false",
        ),
        (
            ("operational", "requested_host_cpu_budget"),
            0,
            "at least 1",
        ),
        (("operational", "use_ray"), True, "use_ray=false"),
    ],
)
def test_runtime_config_rejects_non_strict_values(path, value, message):
    mapping = _runtime_mapping()
    target = mapping
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises((TypeError, ValueError), match=message):
        StrictCausalForestRuntimeConfig.from_mapping(mapping)


def test_installed_constructor_signatures_are_exhaustively_classified():
    result = assert_supported_constructor_signatures(
        causal_forest_class=CausalForestDML,
        treatment_forest_class=RandomForestClassifier,
        outcome_forest_class=RandomForestRegressor,
        stratified_crossfit_class=StratifiedKFold,
    )
    assert tuple(result["causal_forest"]) == tuple(inspect.signature(CausalForestDML).parameters)

    class DriftedCausalForest:
        pass

    signature = inspect.signature(CausalForestDML)
    parameters = list(signature.parameters.values())
    parameters.append(
        inspect.Parameter(
            "new_unclassified_parameter",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
        )
    )
    DriftedCausalForest.__signature__ = signature.replace(parameters=parameters)
    with pytest.raises(RuntimeError, match="unsupported causal_forest"):
        assert_supported_constructor_signatures(
            causal_forest_class=DriftedCausalForest,
            treatment_forest_class=RandomForestClassifier,
            outcome_forest_class=RandomForestRegressor,
            stratified_crossfit_class=StratifiedKFold,
        )

    class DriftedParameterKind:
        pass

    original = list(signature.parameters.values())
    original[0] = original[0].replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    DriftedParameterKind.__signature__ = signature.replace(parameters=original)
    with pytest.raises(RuntimeError, match="parameter kinds"):
        assert_supported_constructor_signatures(
            causal_forest_class=DriftedParameterKind,
            treatment_forest_class=RandomForestClassifier,
            outcome_forest_class=RandomForestRegressor,
            stratified_crossfit_class=StratifiedKFold,
        )


def test_causal_forest_head_exposes_only_elastic_net_nuisance_configuration():
    parameters = inspect.signature(CausalForestHead).parameters
    retired = {
        "runtime_config",
        "nuisance_model_family",
        "nuisance_n_estimators",
        "nuisance_max_depth",
        "nuisance_min_samples_leaf",
        "nuisance_treatment_max_features",
        "nuisance_outcome_max_features",
    }

    assert retired.isdisjoint(parameters)
    with pytest.raises(TypeError, match="runtime_config"):
        CausalForestHead(runtime_config=_runtime())
    with pytest.raises(TypeError, match="nuisance_model_family"):
        CausalForestHead(nuisance_model_family="random_forest")


def test_prediction_calls_explicit_binary_treatment_contrast():
    class EffectSpy:
        def __init__(self):
            self.calls = []

        def effect(self, X, **kwargs):
            self.calls.append((X, kwargs))
            return np.zeros(len(X), dtype=float)

    head = object.__new__(CausalForestHead)
    head._fitted = True
    head.model = EffectSpy()
    head.inference = False
    values = np.ones((3, 2), dtype=float)
    result = head.predict(values, return_ci=False)
    np.testing.assert_array_equal(result["tau_pred"], np.zeros(3))
    assert head.model.calls[0][1] == {"T0": 0, "T1": 1}


@pytest.mark.parametrize(
    ("outcome_type", "expected_class", "discrete_outcome", "prediction_interface"),
    [
        ("binary", ElasticNetLogisticClassifier, True, "predict_proba"),
        ("continuous", ElasticNetRegressor, False, "predict"),
    ],
)
def test_convenience_head_uses_outcome_typed_nuisance_model(
    outcome_type,
    expected_class,
    discrete_outcome,
    prediction_interface,
):
    head = CausalForestHead(
        outcome_type=outcome_type,
        n_estimators=8,
        subforest_size=4,
        min_samples_leaf=2,
        nuisance_regularization_grid_size=5,
        nuisance_max_iter=500,
        tune_model=False,
        n_jobs=1,
    )

    model = head._create_model()

    assert type(model.model_y) is expected_class
    assert model.discrete_outcome is discrete_outcome
    assert head._configured_forest_parameters()["discrete_outcome"] is discrete_outcome
    assert (
        head._configured_nuisance_parameters()["outcome_model_contract"][
            "prediction_interface"
        ]
        == prediction_interface
    )


def test_fitted_causal_forest_crossfits_elastic_net_nuisance_models():
    effect, control, treatment, outcome = _data()
    head = CausalForestHead(
        outcome_type="binary",
        n_estimators=8,
        subforest_size=4,
        min_samples_leaf=2,
        nuisance_regularization_grid_size=5,
        nuisance_maximum_log10_c=2.0,
        nuisance_max_iter=10_000,
        nuisance_tolerance=1e-4,
        tune_model=False,
        n_jobs=1,
    ).fit(
        X=effect,
        W=control,
        T=treatment,
        Y=outcome,
    )

    assert all(
        type(model) is ElasticNetLogisticClassifier
        for monte_carlo_models in head.model.models_t
        for model in monte_carlo_models
    )
    assert all(
        type(model) is ElasticNetLogisticClassifier
        for monte_carlo_models in head.model.models_y
        for model in monte_carlo_models
    )
    assert head.fit_audit()["effective_nuisance_parameters"]["model_family"] == (
        "elastic_net"
    )


def test_elastic_net_nuisance_wrappers_are_clone_safe_and_handle_empty_designs():
    design = np.empty((4, 0), dtype=float)
    classifier = clone(ElasticNetLogisticClassifier()).fit(
        design,
        np.array([0, 1, 0, 1]),
    )
    regressor = clone(ElasticNetRegressor()).fit(
        design,
        np.array([1.0, 2.0, 3.0, 4.0]),
    )

    np.testing.assert_allclose(classifier.predict_proba(design)[:, 1], 0.5)
    np.testing.assert_allclose(regressor.predict(design), 2.5)


def test_binary_convenience_head_rejects_nonbinary_outcomes():
    head = CausalForestHead(
        outcome_type="binary",
        n_estimators=8,
        subforest_size=4,
        tune_model=False,
    )
    effect = np.ones((4, 1), dtype=float)
    treatment = np.array([0, 1, 0, 1], dtype=float)

    with pytest.raises(ValueError, match="must contain exactly both 0 and 1"):
        head.fit(effect, treatment, np.array([0.0, 0.2, 0.8, 1.0]))


def test_scientific_mutations_change_identity_but_operations_do_not():
    baseline = _runtime_mapping()
    baseline_config = StrictCausalForestRuntimeConfig.from_mapping(baseline)
    mutations = (
        (("causal_forest", "random_seed"), 17),
        (("causal_forest", "criterion"), "het"),
        (("causal_forest", "crossfit", "random_seed"), 17),
        (
            ("causal_forest", "treatment_model", "criterion"),
            "entropy",
        ),
        (
            ("causal_forest", "outcome_model", "criterion"),
            "absolute_error",
        ),
    )
    for path, value in mutations:
        changed = copy.deepcopy(baseline)
        target = changed
        for component in path[:-1]:
            target = target[component]
        target[path[-1]] = value
        changed_config = StrictCausalForestRuntimeConfig.from_mapping(changed)
        assert (
            changed_config.scientific_identity_sha256()
            != baseline_config.scientific_identity_sha256()
        )

    operations = copy.deepcopy(baseline)
    operations["operational"]["verbose"] = 2
    operations["operational"]["requested_host_cpu_budget"] = 4
    operational_config = StrictCausalForestRuntimeConfig.from_mapping(operations)
    assert (
        operational_config.scientific_identity_sha256()
        == baseline_config.scientific_identity_sha256()
    )
