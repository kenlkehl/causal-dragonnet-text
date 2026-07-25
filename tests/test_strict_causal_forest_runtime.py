import copy
import inspect

import numpy as np
import pytest
from econml.dml import CausalForestDML
from econml.grf import CausalForest as EconMLCausalForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

import oci.models.causal_forest_head as head_module
from oci.inference.final_context_fit_causal_forest_adapter import (
    FixedCausalForestHeadBackend,
)
from oci.models.causal_forest_head import CausalForestHead
from oci.models.strict_causal_forest_runtime import (
    CAUSAL_FOREST_IMPLEMENTATION,
    OUTCOME_FOREST_IMPLEMENTATION,
    STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
    STRATIFIED_CROSSFIT_IMPLEMENTATION,
    TREATMENT_FOREST_IMPLEMENTATION,
    StrictCausalForestRuntimeConfig,
    assert_supported_constructor_signatures,
    audit_strict_fitted_estimator,
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


def test_strict_model_constructor_receives_every_explicit_kwarg(monkeypatch):
    real_cf_signature = inspect.signature(CausalForestDML)
    real_t_signature = inspect.signature(RandomForestClassifier)
    real_y_signature = inspect.signature(RandomForestRegressor)
    real_cv_signature = inspect.signature(StratifiedKFold)

    class FakeTreatmentForest:
        __signature__ = real_t_signature

        def __init__(self, **kwargs):
            self.parameters = dict(kwargs)

        def get_params(self, deep=False):
            assert deep is False
            return dict(self.parameters)

    class FakeOutcomeForest:
        __signature__ = real_y_signature

        def __init__(self, **kwargs):
            self.parameters = dict(kwargs)

        def get_params(self, deep=False):
            assert deep is False
            return dict(self.parameters)

    class FakeCrossfit:
        __signature__ = real_cv_signature

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeCausalForest:
        __signature__ = real_cf_signature

        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            self.model_t = kwargs["model_t"]
            self.model_y = kwargs["model_y"]
            self.cv = kwargs["cv"]
            for key, value in kwargs.items():
                if key not in {"model_t", "model_y", "cv"}:
                    setattr(self, key, value)

    monkeypatch.setattr(head_module, "CausalForestDML", FakeCausalForest)
    monkeypatch.setattr(head_module, "RandomForestClassifier", FakeTreatmentForest)
    monkeypatch.setattr(head_module, "RandomForestRegressor", FakeOutcomeForest)
    monkeypatch.setattr(head_module, "StratifiedKFold", FakeCrossfit)

    config = _runtime()
    model = CausalForestHead(runtime_config=config)._create_model()
    assert tuple(model.kwargs) == tuple(inspect.signature(CausalForestDML).parameters)
    assert tuple(model.model_t.parameters) == tuple(
        inspect.signature(RandomForestClassifier).parameters
    )
    assert tuple(model.model_y.parameters) == tuple(
        inspect.signature(RandomForestRegressor).parameters
    )
    assert set(vars(model.cv)) == set(inspect.signature(StratifiedKFold).parameters)


def test_real_strict_fit_audits_attributes_clones_grfs_and_split_hashes():
    effect, control, treatment, outcome = _data()
    config = _runtime()
    head = CausalForestHead(runtime_config=config).fit(
        X=effect,
        W=control,
        T=treatment,
        Y=outcome,
    )
    audit = head.fit_audit()

    assert audit["configuration_mode"] == ("portable_strict_runtime_config_v1")
    assert audit["scientific_identity_sha256"] == (config.scientific_identity_sha256())
    assert audit["operational_attestation"] == (config.operational_attestation())
    assert audit["fit_call_contract"] == {
        "sample_weight": None,
        "groups": None,
        "cache_values": False,
        "inference": "auto",
        "fit_call_count": 1,
    }
    assert audit["prediction_contrast"] == {"T0": 0, "T1": 1}
    split_audit = audit["crossfit_split_audit"]
    assert len(split_audit["splits"]) == 2
    assert split_audit == config.split_audit(treatment)
    fitted = audit["fitted_estimator_audit"]
    assert len(fitted["fitted_treatment_models"]) == 1
    assert len(fitted["fitted_treatment_models"][0]) == 2
    assert len(fitted["fitted_outcome_models"][0]) == 2
    assert len(fitted["fitted_grf_parameters"]) == 1
    assert fitted["fitted_grf_parameters"][0]["warm_start"] is False

    top_level_trees = head.model.n_estimators
    del head.model.n_estimators
    with pytest.raises(RuntimeError, match="does not expose"):
        audit_strict_fitted_estimator(
            model=head.model,
            config=config,
            causal_forest_class=CausalForestDML,
            treatment_forest_class=RandomForestClassifier,
            outcome_forest_class=RandomForestRegressor,
            stratified_crossfit_class=StratifiedKFold,
            grf_class=EconMLCausalForest,
        )
    head.model.n_estimators = top_level_trees

    head.model.models_t[0][0].min_samples_leaf += 1
    with pytest.raises(RuntimeError, match="effective parameters differ"):
        audit_strict_fitted_estimator(
            model=head.model,
            config=config,
            causal_forest_class=CausalForestDML,
            treatment_forest_class=RandomForestClassifier,
            outcome_forest_class=RandomForestRegressor,
            stratified_crossfit_class=StratifiedKFold,
            grf_class=EconMLCausalForest,
        )
    head.model.models_t[0][0].min_samples_leaf -= 1

    head.model.model_cate.estimators_[0].min_samples_leaf += 1
    with pytest.raises(RuntimeError, match="effective parameters differ"):
        audit_strict_fitted_estimator(
            model=head.model,
            config=config,
            causal_forest_class=CausalForestDML,
            treatment_forest_class=RandomForestClassifier,
            outcome_forest_class=RandomForestRegressor,
            stratified_crossfit_class=StratifiedKFold,
            grf_class=EconMLCausalForest,
        )


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


def test_explicit_crossfit_preserves_implicit_cv2_predictions_and_n_jobs():
    effect, control, treatment, outcome = _data()
    strict_one = CausalForestHead(runtime_config=_runtime(n_jobs=1)).fit(
        X=effect,
        W=control,
        T=treatment,
        Y=outcome,
    )
    strict_two = CausalForestHead(runtime_config=_runtime(n_jobs=2)).fit(
        X=effect,
        W=control,
        T=treatment,
        Y=outcome,
    )
    strict_repeat = CausalForestHead(runtime_config=_runtime(n_jobs=1)).fit(
        X=effect,
        W=control,
        T=treatment,
        Y=outcome,
    )
    legacy = CausalForestHead(
        n_estimators=8,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        honest=True,
        inference=True,
        random_state=42,
        tune_model=False,
        subforest_size=4,
        nuisance_n_estimators=5,
        nuisance_max_depth=None,
        nuisance_min_samples_leaf=2,
        nuisance_treatment_max_features="sqrt",
        nuisance_outcome_max_features=1.0,
        n_jobs=1,
    ).fit(
        X=effect,
        W=control,
        T=treatment,
        Y=outcome,
    )
    legacy_parallel = CausalForestHead(
        n_estimators=8,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        honest=True,
        inference=True,
        random_state=42,
        tune_model=False,
        subforest_size=4,
        nuisance_n_estimators=5,
        nuisance_max_depth=None,
        nuisance_min_samples_leaf=2,
        nuisance_treatment_max_features="sqrt",
        nuisance_outcome_max_features=1.0,
        n_jobs=2,
    ).fit(
        X=effect,
        W=control,
        T=treatment,
        Y=outcome,
    )
    heldout = effect[:12]
    strict_one_tau = strict_one.predict(heldout, return_ci=False)["tau_pred"]
    strict_two_tau = strict_two.predict(heldout, return_ci=False)["tau_pred"]
    strict_repeat_tau = strict_repeat.predict(heldout, return_ci=False)["tau_pred"]
    legacy_tau = legacy.predict(heldout, return_ci=False)["tau_pred"]
    legacy_parallel_tau = legacy_parallel.predict(heldout, return_ci=False)["tau_pred"]
    np.testing.assert_array_equal(strict_one_tau, strict_two_tau)
    np.testing.assert_array_equal(strict_one_tau, strict_repeat_tau)
    np.testing.assert_array_equal(strict_one_tau, legacy_tau)
    assert not np.array_equal(legacy_tau, legacy_parallel_tau)
    np.testing.assert_allclose(
        legacy_tau,
        legacy_parallel_tau,
        rtol=0.0,
        atol=2e-15,
    )
    assert strict_two.fit_audit()["operational_attestation"]["requested_host_cpu_budget"] == 2
    assert strict_two.fit_audit()["operational_attestation"]["effective_estimator_n_jobs"] == 1


def test_backend_identity_distinguishes_portable_and_legacy_modes():
    portable_one = FixedCausalForestHeadBackend(runtime_config=_runtime(n_jobs=1))
    portable_two = FixedCausalForestHeadBackend(runtime_config=_runtime(n_jobs=2))
    legacy = FixedCausalForestHeadBackend()

    assert portable_one.identity() == portable_two.identity()
    assert portable_one.identity()["configuration_mode"] == ("portable_strict_runtime_config_v1")
    assert legacy.identity()["configuration_mode"] == ("legacy_compatibility_shim_v1")
    assert portable_one.identity() != legacy.identity()
    with pytest.raises(ValueError, match="cannot be combined"):
        FixedCausalForestHeadBackend(
            n_estimators=8,
            runtime_config=_runtime(),
        )


def test_portable_backend_runs_only_the_authenticated_strict_path():
    effect, control, treatment, outcome = _data()
    config = _runtime(n_jobs=3)
    backend = FixedCausalForestHeadBackend(runtime_config=config)
    tau = backend.fit_predict(
        effect_train=effect,
        control_train=control,
        treatment=treatment,
        outcome=outcome,
        effect_heldout=effect[:9],
        control_heldout=control[:9],
    )
    assert tau.shape == (9,)
    assert np.isfinite(tau).all()
    audit = backend.fit_audit()
    assert audit["configuration_mode"] == ("portable_strict_runtime_config_v1")
    assert audit["scientific_identity_sha256"] == (config.scientific_identity_sha256())
    assert audit["operational_parameters"]["requested_host_cpu_budget"] == 3
    assert audit["operational_parameters"]["effective_estimator_n_jobs"] == 1
    assert audit["outer_train_labels_only"] is True
    assert audit["outer_heldout_labels_accepted"] is False


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
