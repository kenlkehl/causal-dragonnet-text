"""Closed, path-neutral runtime contract for the final causal forest.

The portable final estimator must not inherit scientific behavior from
EconML or scikit-learn constructor defaults.  This module therefore makes
every constructor parameter explicit, separates scientific settings from
operational parallelism, checks the installed constructor signatures, and
audits the fitted estimator graph.

``StrictCausalForestRuntimeConfig`` deliberately has no field defaults.
Callers must supply the complete contract.  The older convenience arguments
on :class:`oci.models.causal_forest_head.CausalForestHead` remain a separately
labelled compatibility path and are not a portable scientific configuration.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA = "strict_causal_forest_runtime_config_v1"
STRICT_CAUSAL_FOREST_SCIENTIFIC_SCHEMA = "strict_causal_forest_scientific_identity_v1"
STRICT_CAUSAL_FOREST_OPERATIONAL_SCHEMA = "strict_causal_forest_operational_attestation_v1"

CAUSAL_FOREST_IMPLEMENTATION = "econml.dml.CausalForestDML"
TREATMENT_FOREST_IMPLEMENTATION = "sklearn.ensemble.RandomForestClassifier"
OUTCOME_CLASSIFIER_IMPLEMENTATION = "sklearn.ensemble.RandomForestClassifier"
OUTCOME_FOREST_IMPLEMENTATION = "sklearn.ensemble.RandomForestRegressor"
STRATIFIED_CROSSFIT_IMPLEMENTATION = "sklearn.model_selection.StratifiedKFold"

_CAUSAL_FOREST_SIGNATURE = (
    "model_y",
    "model_t",
    "featurizer",
    "treatment_featurizer",
    "discrete_outcome",
    "discrete_treatment",
    "categories",
    "cv",
    "mc_iters",
    "mc_agg",
    "drate",
    "n_estimators",
    "criterion",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "min_weight_fraction_leaf",
    "min_var_fraction_leaf",
    "min_var_leaf_on_val",
    "max_features",
    "min_impurity_decrease",
    "max_samples",
    "min_balancedness_tol",
    "honest",
    "inference",
    "fit_intercept",
    "subforest_size",
    "n_jobs",
    "random_state",
    "verbose",
    "allow_missing",
    "use_ray",
    "ray_remote_func_options",
)
_TREATMENT_FOREST_SIGNATURE = (
    "n_estimators",
    "criterion",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "min_weight_fraction_leaf",
    "max_features",
    "max_leaf_nodes",
    "min_impurity_decrease",
    "bootstrap",
    "oob_score",
    "n_jobs",
    "random_state",
    "verbose",
    "warm_start",
    "class_weight",
    "ccp_alpha",
    "max_samples",
    "monotonic_cst",
)
_OUTCOME_FOREST_SIGNATURE = (
    "n_estimators",
    "criterion",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "min_weight_fraction_leaf",
    "max_features",
    "max_leaf_nodes",
    "min_impurity_decrease",
    "bootstrap",
    "oob_score",
    "n_jobs",
    "random_state",
    "verbose",
    "warm_start",
    "ccp_alpha",
    "max_samples",
    "monotonic_cst",
)
_STRATIFIED_CROSSFIT_SIGNATURE = (
    "n_splits",
    "shuffle",
    "random_state",
)
_CAUSAL_FOREST_SIGNATURE_KINDS = tuple(
    inspect.Parameter.KEYWORD_ONLY for _ in _CAUSAL_FOREST_SIGNATURE
)
_TREATMENT_FOREST_SIGNATURE_KINDS = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    *(inspect.Parameter.KEYWORD_ONLY for _ in _TREATMENT_FOREST_SIGNATURE[1:]),
)
_OUTCOME_FOREST_SIGNATURE_KINDS = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    *(inspect.Parameter.KEYWORD_ONLY for _ in _OUTCOME_FOREST_SIGNATURE[1:]),
)
_STRATIFIED_CROSSFIT_SIGNATURE_KINDS = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
    inspect.Parameter.KEYWORD_ONLY,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _closed_mapping(
    value: Mapping[str, Any],
    *,
    required: Sequence[str],
    path: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    expected = set(required)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{path} must contain exactly the closed schema; " f"missing={missing}, extra={extra}"
        )
    return dict(value)


def _bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def _int(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
    nonzero: bool = False,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    if nonzero and result == 0:
        raise ValueError(f"{name} must be nonzero")
    return result


def _seed(value: Any, *, name: str) -> int:
    return _int(value, name=name, minimum=0, maximum=(2**32 - 1))


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        if result < minimum or (result == minimum and not minimum_inclusive):
            relation = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None:
        if result > maximum or (result == maximum and not maximum_inclusive):
            relation = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{name} must be {relation} {maximum}")
    return result


def _optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _int(value, name=name, minimum=1)


def _split_count(value: Any, *, name: str) -> int | float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} cannot be boolean")
    if isinstance(value, (int, np.integer)):
        return _int(value, name=name, minimum=2)
    return _finite_float(
        value,
        name=name,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )


def _leaf_count(
    value: Any,
    *,
    name: str,
    fractional_maximum: float,
) -> int | float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} cannot be boolean")
    if isinstance(value, (int, np.integer)):
        return _int(value, name=name, minimum=1)
    return _finite_float(
        value,
        name=name,
        minimum=0.0,
        maximum=fractional_maximum,
        minimum_inclusive=False,
    )


def _max_features(
    value: Any,
    *,
    name: str,
    allowed_strings: frozenset[str],
    allow_none: bool,
) -> str | int | float | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} cannot be null")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} cannot be boolean")
    if isinstance(value, str):
        if value not in allowed_strings:
            raise ValueError(f"{name} must be one of {sorted(allowed_strings)}")
        return value
    if isinstance(value, (int, np.integer)):
        return _int(value, name=name, minimum=1)
    return _finite_float(
        value,
        name=name,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )


def _max_samples(
    value: Any,
    *,
    name: str,
    allow_none: bool,
) -> int | float | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} cannot be null")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} cannot be boolean")
    if isinstance(value, (int, np.integer)):
        return _int(value, name=name, minimum=1)
    return _finite_float(
        value,
        name=name,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )


def _exact_literal(value: Any, *, expected: Any, name: str) -> Any:
    if value != expected or type(value) is not type(expected):
        raise ValueError(f"{name} must be exactly {expected!r}")
    return value


@dataclass(frozen=True)
class StrictStratifiedKFoldSpec:
    implementation: str
    n_splits: int
    shuffle: bool
    random_seed: int

    def __post_init__(self) -> None:
        _exact_literal(
            self.implementation,
            expected=STRATIFIED_CROSSFIT_IMPLEMENTATION,
            name="crossfit.implementation",
        )
        _int(self.n_splits, name="crossfit.n_splits", minimum=2)
        if not _bool(self.shuffle, name="crossfit.shuffle"):
            raise ValueError(
                "strict cross-fitting requires shuffle=true so its explicit "
                "seed controls the split"
            )
        _seed(self.random_seed, name="crossfit.random_seed")

    def as_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "n_splits": int(self.n_splits),
            "shuffle": bool(self.shuffle),
            "random_seed": int(self.random_seed),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StrictStratifiedKFoldSpec":
        data = _closed_mapping(
            value,
            required=(
                "implementation",
                "n_splits",
                "shuffle",
                "random_seed",
            ),
            path="crossfit",
        )
        return cls(**data)


@dataclass(frozen=True)
class StrictRandomForestClassifierSpec:
    implementation: str
    n_estimators: int
    criterion: str
    max_depth: int | None
    min_samples_split: int | float
    min_samples_leaf: int | float
    min_weight_fraction_leaf: float
    max_features: str | int | float | None
    max_leaf_nodes: int | None
    min_impurity_decrease: float
    bootstrap: bool
    oob_score: bool
    random_seed: int
    warm_start: bool
    class_weight: None
    ccp_alpha: float
    max_samples: int | float | None
    monotonic_cst: None

    def __post_init__(self) -> None:
        _exact_literal(
            self.implementation,
            expected=TREATMENT_FOREST_IMPLEMENTATION,
            name="treatment_model.implementation",
        )
        _int(
            self.n_estimators,
            name="treatment_model.n_estimators",
            minimum=1,
        )
        if self.criterion not in {"gini", "entropy", "log_loss"}:
            raise ValueError("unsupported treatment_model.criterion")
        _optional_positive_int(self.max_depth, name="treatment_model.max_depth")
        _split_count(
            self.min_samples_split,
            name="treatment_model.min_samples_split",
        )
        _leaf_count(
            self.min_samples_leaf,
            name="treatment_model.min_samples_leaf",
            fractional_maximum=1.0,
        )
        _finite_float(
            self.min_weight_fraction_leaf,
            name="treatment_model.min_weight_fraction_leaf",
            minimum=0.0,
            maximum=0.5,
        )
        _max_features(
            self.max_features,
            name="treatment_model.max_features",
            allowed_strings=frozenset({"sqrt", "log2"}),
            allow_none=True,
        )
        if self.max_leaf_nodes is not None:
            _int(
                self.max_leaf_nodes,
                name="treatment_model.max_leaf_nodes",
                minimum=2,
            )
        _finite_float(
            self.min_impurity_decrease,
            name="treatment_model.min_impurity_decrease",
            minimum=0.0,
        )
        bootstrap = _bool(self.bootstrap, name="treatment_model.bootstrap")
        if _bool(self.oob_score, name="treatment_model.oob_score"):
            raise ValueError("strict nuisance forests require oob_score=false")
        _seed(self.random_seed, name="treatment_model.random_seed")
        if _bool(self.warm_start, name="treatment_model.warm_start"):
            raise ValueError("strict nuisance forests require warm_start=false")
        if self.class_weight is not None:
            raise ValueError("strict v1 treatment_model.class_weight must be null")
        _finite_float(
            self.ccp_alpha,
            name="treatment_model.ccp_alpha",
            minimum=0.0,
        )
        max_samples = _max_samples(
            self.max_samples,
            name="treatment_model.max_samples",
            allow_none=True,
        )
        if not bootstrap and max_samples is not None:
            raise ValueError("treatment_model.max_samples requires bootstrap=true")
        if self.monotonic_cst is not None:
            raise ValueError("strict v1 treatment_model.monotonic_cst must be null")

    def scientific_kwargs(self) -> dict[str, Any]:
        return {
            "n_estimators": int(self.n_estimators),
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": float(self.min_weight_fraction_leaf),
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": float(self.min_impurity_decrease),
            "bootstrap": bool(self.bootstrap),
            "oob_score": bool(self.oob_score),
            "random_state": int(self.random_seed),
            "warm_start": bool(self.warm_start),
            "class_weight": self.class_weight,
            "ccp_alpha": float(self.ccp_alpha),
            "max_samples": self.max_samples,
            "monotonic_cst": self.monotonic_cst,
        }

    def as_dict(self) -> dict[str, Any]:
        result = {"implementation": self.implementation}
        result.update(self.scientific_kwargs())
        result["random_seed"] = result.pop("random_state")
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StrictRandomForestClassifierSpec":
        fields = tuple(cls.__dataclass_fields__)
        return cls(
            **_closed_mapping(
                value,
                required=fields,
                path="treatment_model",
            )
        )


@dataclass(frozen=True)
class StrictOutcomeRandomForestClassifierSpec(StrictRandomForestClassifierSpec):
    """Closed classifier specification used for a discrete outcome nuisance."""

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "StrictOutcomeRandomForestClassifierSpec":
        fields = tuple(cls.__dataclass_fields__)
        return cls(
            **_closed_mapping(
                value,
                required=fields,
                path="outcome_model",
            )
        )


@dataclass(frozen=True)
class StrictRandomForestRegressorSpec:
    implementation: str
    n_estimators: int
    criterion: str
    max_depth: int | None
    min_samples_split: int | float
    min_samples_leaf: int | float
    min_weight_fraction_leaf: float
    max_features: str | int | float | None
    max_leaf_nodes: int | None
    min_impurity_decrease: float
    bootstrap: bool
    oob_score: bool
    random_seed: int
    warm_start: bool
    ccp_alpha: float
    max_samples: int | float | None
    monotonic_cst: None

    def __post_init__(self) -> None:
        _exact_literal(
            self.implementation,
            expected=OUTCOME_FOREST_IMPLEMENTATION,
            name="outcome_model.implementation",
        )
        _int(
            self.n_estimators,
            name="outcome_model.n_estimators",
            minimum=1,
        )
        if self.criterion not in {
            "squared_error",
            "absolute_error",
            "friedman_mse",
            "poisson",
        }:
            raise ValueError("unsupported outcome_model.criterion")
        _optional_positive_int(self.max_depth, name="outcome_model.max_depth")
        _split_count(
            self.min_samples_split,
            name="outcome_model.min_samples_split",
        )
        _leaf_count(
            self.min_samples_leaf,
            name="outcome_model.min_samples_leaf",
            fractional_maximum=1.0,
        )
        _finite_float(
            self.min_weight_fraction_leaf,
            name="outcome_model.min_weight_fraction_leaf",
            minimum=0.0,
            maximum=0.5,
        )
        _max_features(
            self.max_features,
            name="outcome_model.max_features",
            allowed_strings=frozenset({"sqrt", "log2"}),
            allow_none=True,
        )
        if self.max_leaf_nodes is not None:
            _int(
                self.max_leaf_nodes,
                name="outcome_model.max_leaf_nodes",
                minimum=2,
            )
        _finite_float(
            self.min_impurity_decrease,
            name="outcome_model.min_impurity_decrease",
            minimum=0.0,
        )
        bootstrap = _bool(self.bootstrap, name="outcome_model.bootstrap")
        if _bool(self.oob_score, name="outcome_model.oob_score"):
            raise ValueError("strict nuisance forests require oob_score=false")
        _seed(self.random_seed, name="outcome_model.random_seed")
        if _bool(self.warm_start, name="outcome_model.warm_start"):
            raise ValueError("strict nuisance forests require warm_start=false")
        _finite_float(
            self.ccp_alpha,
            name="outcome_model.ccp_alpha",
            minimum=0.0,
        )
        max_samples = _max_samples(
            self.max_samples,
            name="outcome_model.max_samples",
            allow_none=True,
        )
        if not bootstrap and max_samples is not None:
            raise ValueError("outcome_model.max_samples requires bootstrap=true")
        if self.monotonic_cst is not None:
            raise ValueError("strict v1 outcome_model.monotonic_cst must be null")

    def scientific_kwargs(self) -> dict[str, Any]:
        return {
            "n_estimators": int(self.n_estimators),
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": float(self.min_weight_fraction_leaf),
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": float(self.min_impurity_decrease),
            "bootstrap": bool(self.bootstrap),
            "oob_score": bool(self.oob_score),
            "random_state": int(self.random_seed),
            "warm_start": bool(self.warm_start),
            "ccp_alpha": float(self.ccp_alpha),
            "max_samples": self.max_samples,
            "monotonic_cst": self.monotonic_cst,
        }

    def as_dict(self) -> dict[str, Any]:
        result = {"implementation": self.implementation}
        result.update(self.scientific_kwargs())
        result["random_seed"] = result.pop("random_state")
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StrictRandomForestRegressorSpec":
        fields = tuple(cls.__dataclass_fields__)
        return cls(
            **_closed_mapping(
                value,
                required=fields,
                path="outcome_model",
            )
        )


@dataclass(frozen=True)
class StrictCausalForestDMLSpec:
    implementation: str
    tune_model: bool
    featurizer: None
    treatment_featurizer: None
    discrete_outcome: bool
    discrete_treatment: bool
    categories: str
    crossfit: StrictStratifiedKFoldSpec
    mc_iters: int | None
    mc_agg: str
    drate: bool
    n_estimators: int
    criterion: str
    max_depth: int | None
    min_samples_split: int | float
    min_samples_leaf: int | float
    min_weight_fraction_leaf: float
    min_var_fraction_leaf: float | None
    min_var_leaf_on_val: bool
    max_features: str | int | float
    min_impurity_decrease: float
    max_samples: int | float
    min_balancedness_tol: float
    honest: bool
    inference: bool
    fit_intercept: bool
    subforest_size: int
    random_seed: int
    allow_missing: bool
    treatment_model: StrictRandomForestClassifierSpec
    outcome_model: (
        StrictOutcomeRandomForestClassifierSpec | StrictRandomForestRegressorSpec
    )

    def __post_init__(self) -> None:
        _exact_literal(
            self.implementation,
            expected=CAUSAL_FOREST_IMPLEMENTATION,
            name="causal_forest.implementation",
        )
        if _bool(self.tune_model, name="causal_forest.tune_model"):
            raise ValueError("portable strict causal forest requires tune_model=false")
        if self.featurizer is not None:
            raise ValueError("causal_forest.featurizer must be null")
        if self.treatment_featurizer is not None:
            raise ValueError("causal_forest.treatment_featurizer must be null")
        discrete_outcome = _bool(
            self.discrete_outcome,
            name="causal_forest.discrete_outcome",
        )
        if not _bool(
            self.discrete_treatment,
            name="causal_forest.discrete_treatment",
        ):
            raise ValueError("strict v1 requires discrete_treatment=true")
        _exact_literal(
            self.categories,
            expected="auto",
            name="causal_forest.categories",
        )
        if not isinstance(self.crossfit, StrictStratifiedKFoldSpec):
            raise TypeError("causal_forest.crossfit must be StrictStratifiedKFoldSpec")
        if self.mc_iters is not None:
            raise ValueError(
                "strict v1 requires causal_forest.mc_iters=null so the "
                "single authenticated cross-fit split plan is exhaustive"
            )
        if self.mc_agg != "mean":
            raise ValueError("strict v1 requires causal_forest.mc_agg='mean'")
        _bool(self.drate, name="causal_forest.drate")
        trees = _int(
            self.n_estimators,
            name="causal_forest.n_estimators",
            minimum=1,
        )
        if self.criterion not in {"mse", "het"}:
            raise ValueError("causal_forest.criterion must be 'mse' or 'het'")
        _optional_positive_int(self.max_depth, name="causal_forest.max_depth")
        _split_count(
            self.min_samples_split,
            name="causal_forest.min_samples_split",
        )
        _leaf_count(
            self.min_samples_leaf,
            name="causal_forest.min_samples_leaf",
            fractional_maximum=0.5,
        )
        _finite_float(
            self.min_weight_fraction_leaf,
            name="causal_forest.min_weight_fraction_leaf",
            minimum=0.0,
            maximum=0.5,
        )
        if self.min_var_fraction_leaf is not None:
            _finite_float(
                self.min_var_fraction_leaf,
                name="causal_forest.min_var_fraction_leaf",
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            )
        _bool(
            self.min_var_leaf_on_val,
            name="causal_forest.min_var_leaf_on_val",
        )
        _max_features(
            self.max_features,
            name="causal_forest.max_features",
            allowed_strings=frozenset({"auto", "sqrt", "log2"}),
            allow_none=False,
        )
        _finite_float(
            self.min_impurity_decrease,
            name="causal_forest.min_impurity_decrease",
            minimum=0.0,
        )
        max_samples = _max_samples(
            self.max_samples,
            name="causal_forest.max_samples",
            allow_none=False,
        )
        _finite_float(
            self.min_balancedness_tol,
            name="causal_forest.min_balancedness_tol",
            minimum=0.0,
            maximum=0.5,
        )
        if not _bool(self.honest, name="causal_forest.honest"):
            raise ValueError("strict final causal forest requires honest=true")
        inference = _bool(self.inference, name="causal_forest.inference")
        if not inference:
            raise ValueError("strict final causal forest requires inference=true")
        _bool(self.fit_intercept, name="causal_forest.fit_intercept")
        subforest = _int(
            self.subforest_size,
            name="causal_forest.subforest_size",
            minimum=2,
        )
        if trees % subforest:
            raise ValueError(
                "causal_forest.n_estimators must be divisible by "
                "subforest_size when inference is enabled"
            )
        if isinstance(max_samples, float) and max_samples > 0.5:
            raise ValueError("causal_forest.max_samples cannot exceed 0.5 with inference")
        _seed(self.random_seed, name="causal_forest.random_seed")
        if _bool(self.allow_missing, name="causal_forest.allow_missing"):
            raise ValueError("strict v1 requires allow_missing=false")
        if not isinstance(self.treatment_model, StrictRandomForestClassifierSpec):
            raise TypeError("causal_forest.treatment_model has the wrong type")
        expected_outcome_spec = (
            StrictOutcomeRandomForestClassifierSpec
            if discrete_outcome
            else StrictRandomForestRegressorSpec
        )
        if type(self.outcome_model) is not expected_outcome_spec:
            expected_family = "classifier" if discrete_outcome else "regressor"
            raise TypeError(
                "causal_forest.outcome_model must be a random-forest "
                f"{expected_family} when discrete_outcome={discrete_outcome}"
            )

    def scientific_constructor_kwargs(self) -> dict[str, Any]:
        return {
            "featurizer": self.featurizer,
            "treatment_featurizer": self.treatment_featurizer,
            "discrete_outcome": bool(self.discrete_outcome),
            "discrete_treatment": bool(self.discrete_treatment),
            "categories": self.categories,
            "mc_iters": self.mc_iters,
            "mc_agg": self.mc_agg,
            "drate": bool(self.drate),
            "n_estimators": int(self.n_estimators),
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": float(self.min_weight_fraction_leaf),
            "min_var_fraction_leaf": self.min_var_fraction_leaf,
            "min_var_leaf_on_val": bool(self.min_var_leaf_on_val),
            "max_features": self.max_features,
            "min_impurity_decrease": float(self.min_impurity_decrease),
            "max_samples": self.max_samples,
            "min_balancedness_tol": float(self.min_balancedness_tol),
            "honest": bool(self.honest),
            "inference": bool(self.inference),
            "fit_intercept": bool(self.fit_intercept),
            "subforest_size": int(self.subforest_size),
            "random_state": int(self.random_seed),
            "allow_missing": bool(self.allow_missing),
        }

    def as_dict(self) -> dict[str, Any]:
        result = {
            "implementation": self.implementation,
            "tune_model": bool(self.tune_model),
            **self.scientific_constructor_kwargs(),
            "crossfit": self.crossfit.as_dict(),
            "treatment_model": self.treatment_model.as_dict(),
            "outcome_model": self.outcome_model.as_dict(),
        }
        result["random_seed"] = result.pop("random_state")
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StrictCausalForestDMLSpec":
        fields = tuple(cls.__dataclass_fields__)
        data = _closed_mapping(
            value,
            required=fields,
            path="causal_forest",
        )
        data["crossfit"] = StrictStratifiedKFoldSpec.from_mapping(data["crossfit"])
        data["treatment_model"] = StrictRandomForestClassifierSpec.from_mapping(
            data["treatment_model"]
        )
        outcome_spec = (
            StrictOutcomeRandomForestClassifierSpec
            if _bool(data["discrete_outcome"], name="causal_forest.discrete_outcome")
            else StrictRandomForestRegressorSpec
        )
        data["outcome_model"] = outcome_spec.from_mapping(data["outcome_model"])
        return cls(**data)


@dataclass(frozen=True)
class StrictCausalForestOperationalSpec:
    requested_host_cpu_budget: int
    verbose: int
    use_ray: bool
    ray_remote_func_options: None

    def __post_init__(self) -> None:
        _int(
            self.requested_host_cpu_budget,
            name="operational.requested_host_cpu_budget",
            minimum=1,
        )
        _int(self.verbose, name="operational.verbose", minimum=0)
        if _bool(self.use_ray, name="operational.use_ray"):
            raise ValueError("single-node strict runtime requires use_ray=false")
        if self.ray_remote_func_options is not None:
            raise ValueError("single-node strict runtime requires " "ray_remote_func_options=null")

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_host_cpu_budget": int(self.requested_host_cpu_budget),
            "verbose": int(self.verbose),
            "use_ray": bool(self.use_ray),
            "ray_remote_func_options": self.ray_remote_func_options,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StrictCausalForestOperationalSpec":
        return cls(
            **_closed_mapping(
                value,
                required=tuple(cls.__dataclass_fields__),
                path="operational",
            )
        )


@dataclass(frozen=True)
class StrictCausalForestRuntimeConfig:
    schema_version: str
    causal_forest: StrictCausalForestDMLSpec
    operational: StrictCausalForestOperationalSpec

    def __post_init__(self) -> None:
        _exact_literal(
            self.schema_version,
            expected=STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
            name="schema_version",
        )
        if not isinstance(self.causal_forest, StrictCausalForestDMLSpec):
            raise TypeError("causal_forest must be StrictCausalForestDMLSpec")
        if not isinstance(self.operational, StrictCausalForestOperationalSpec):
            raise TypeError("operational must be StrictCausalForestOperationalSpec")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "causal_forest": self.causal_forest.as_dict(),
            "operational": self.operational.as_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StrictCausalForestRuntimeConfig":
        data = _closed_mapping(
            value,
            required=("schema_version", "causal_forest", "operational"),
            path="strict_causal_forest_runtime",
        )
        data["causal_forest"] = StrictCausalForestDMLSpec.from_mapping(data["causal_forest"])
        data["operational"] = StrictCausalForestOperationalSpec.from_mapping(data["operational"])
        return cls(**data)

    def scientific_identity(self) -> dict[str, Any]:
        return {
            "schema_version": STRICT_CAUSAL_FOREST_SCIENTIFIC_SCHEMA,
            "causal_forest": self.causal_forest.as_dict(),
            "fit_contract": {
                "groups": None,
                "sample_weight": None,
                "cache_values": False,
                "inference": "auto",
                "fit_call_count": 1,
                "prediction_contrast": {"T0": 0, "T1": 1},
            },
        }

    def scientific_identity_sha256(self) -> str:
        return _sha256_json(self.scientific_identity())

    def operational_attestation(self) -> dict[str, Any]:
        return {
            "schema_version": STRICT_CAUSAL_FOREST_OPERATIONAL_SCHEMA,
            **self.operational.as_dict(),
            "effective_estimator_n_jobs": 1,
            "estimator_parallelism_policy": ("serial_estimator_determinism_v1"),
            "host_parallelism_location": "fold_or_scope_scheduler",
        }

    def treatment_constructor_kwargs(self) -> dict[str, Any]:
        result = self.causal_forest.treatment_model.scientific_kwargs()
        result["n_jobs"] = 1
        result["verbose"] = int(self.operational.verbose)
        return {name: result[name] for name in _TREATMENT_FOREST_SIGNATURE}

    def outcome_constructor_kwargs(self) -> dict[str, Any]:
        result = self.causal_forest.outcome_model.scientific_kwargs()
        result["n_jobs"] = 1
        result["verbose"] = int(self.operational.verbose)
        signature = (
            _TREATMENT_FOREST_SIGNATURE
            if self.causal_forest.discrete_outcome
            else _OUTCOME_FOREST_SIGNATURE
        )
        return {name: result[name] for name in signature}

    def crossfit_constructor_kwargs(self) -> dict[str, Any]:
        return {
            "n_splits": int(self.causal_forest.crossfit.n_splits),
            "shuffle": bool(self.causal_forest.crossfit.shuffle),
            "random_state": int(self.causal_forest.crossfit.random_seed),
        }

    def causal_forest_constructor_kwargs(
        self,
        *,
        model_t: Any,
        model_y: Any,
        cv: Any,
    ) -> dict[str, Any]:
        result = {
            "model_y": model_y,
            "model_t": model_t,
            "cv": cv,
            **self.causal_forest.scientific_constructor_kwargs(),
            "n_jobs": 1,
            "verbose": int(self.operational.verbose),
            "use_ray": bool(self.operational.use_ray),
            "ray_remote_func_options": (self.operational.ray_remote_func_options),
        }
        return {name: result[name] for name in _CAUSAL_FOREST_SIGNATURE}

    def validate_fit_inputs(
        self,
        *,
        effect: np.ndarray,
        controls: np.ndarray | None,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> None:
        x = np.asarray(effect)
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError(
                "strict causal forest effect input must be a nonempty " "two-dimensional matrix"
            )
        if not np.isfinite(x).all():
            raise ValueError("strict causal forest effect input must be finite")
        w = None if controls is None else np.asarray(controls)
        if w is not None:
            if w.ndim != 2 or w.shape[0] != x.shape[0] or not np.isfinite(w).all():
                raise ValueError(
                    "strict causal forest controls must be a finite matrix "
                    "aligned to effect rows"
                )
        t = np.asarray(treatment).reshape(-1)
        y = np.asarray(outcome).reshape(-1)
        if t.shape[0] != x.shape[0] or y.shape[0] != x.shape[0]:
            raise ValueError("strict causal forest labels must align to effect rows")
        if not np.isfinite(t).all() or not np.isfinite(y).all():
            raise ValueError("strict causal forest labels must be finite")
        if set(np.unique(t).tolist()) != {0, 1}:
            raise ValueError("strict causal forest treatment must contain exactly 0 and 1")
        if self.causal_forest.discrete_outcome and set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError("strict binary outcome must contain exactly both 0 and 1")
        strata = _strict_crossfit_strata(
            treatment=t,
            outcome=y,
            discrete_outcome=bool(self.causal_forest.discrete_outcome),
        )
        _, class_counts = np.unique(strata, return_counts=True)
        n_splits = int(self.causal_forest.crossfit.n_splits)
        if int(class_counts.min()) < n_splits:
            raise ValueError("crossfit.n_splits exceeds the smallest discrete-stratum count")
        if (
            isinstance(self.causal_forest.outcome_model, StrictRandomForestRegressorSpec)
            and self.causal_forest.outcome_model.criterion == "poisson"
            and ((y < 0).any() or float(y.sum()) <= 0.0)
        ):
            raise ValueError(
                "Poisson nuisance outcome fitting requires nonnegative "
                "outcomes with positive total"
            )
        _validate_feature_count(
            self.causal_forest.max_features,
            feature_count=int(x.shape[1]),
            name="causal_forest.max_features",
        )
        nuisance_feature_count = int(x.shape[1]) + (0 if w is None else int(w.shape[1]))
        _validate_feature_count(
            self.causal_forest.treatment_model.max_features,
            feature_count=nuisance_feature_count,
            name="treatment_model.max_features",
        )
        _validate_feature_count(
            self.causal_forest.outcome_model.max_features,
            feature_count=nuisance_feature_count,
            name="outcome_model.max_features",
        )
        n_rows = int(x.shape[0])
        max_samples = self.causal_forest.max_samples
        if isinstance(max_samples, (int, np.integer)) and int(max_samples) > n_rows // 2:
            raise ValueError(
                "integer causal_forest.max_samples cannot exceed half "
                "the fit rows when inference is enabled"
            )
        smallest_crossfit_train = min(
            len(train)
            for train, _ in _crossfit_splits(
                self,
                treatment=t.astype(np.int64),
                outcome=y,
            )
        )
        for label, spec in (
            ("treatment_model", self.causal_forest.treatment_model),
            ("outcome_model", self.causal_forest.outcome_model),
        ):
            if (
                isinstance(spec.max_samples, (int, np.integer))
                and int(spec.max_samples) > smallest_crossfit_train
            ):
                raise ValueError(
                    f"integer {label}.max_samples exceeds a cross-fit " "training partition"
                )

    def split_audit(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray | None = None,
    ) -> dict[str, Any]:
        splits = _crossfit_splits(self, treatment=treatment, outcome=outcome)
        records = []
        for fold_index, (train, test) in enumerate(splits):
            records.append(
                {
                    "fold_index": fold_index,
                    "train_count": int(len(train)),
                    "test_count": int(len(test)),
                    "train_index_sha256": _index_sha256(train),
                    "test_index_sha256": _index_sha256(test),
                }
            )
        result = {
            "implementation": STRATIFIED_CROSSFIT_IMPLEMENTATION,
            "parameters": self.crossfit_constructor_kwargs(),
            "splits": records,
        }
        result["split_plan_sha256"] = _sha256_json(result)
        return result


def _validate_feature_count(
    value: Any,
    *,
    feature_count: int,
    name: str,
) -> None:
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        if int(value) > feature_count:
            raise ValueError(
                f"integer {name} exceeds the actual feature dimension " f"({feature_count})"
            )


def _crossfit_splits(
    config: StrictCausalForestRuntimeConfig,
    *,
    treatment: np.ndarray,
    outcome: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold

    vector = np.asarray(treatment).reshape(-1)
    strata = _strict_crossfit_strata(
        treatment=vector,
        outcome=outcome,
        discrete_outcome=bool(config.causal_forest.discrete_outcome),
    )
    splitter = StratifiedKFold(**config.crossfit_constructor_kwargs())
    return [
        (
            np.asarray(train, dtype=np.int64),
            np.asarray(test, dtype=np.int64),
        )
        for train, test in splitter.split(
            np.zeros((len(vector), 1), dtype=np.uint8),
            strata,
        )
    ]


def _strict_crossfit_strata(
    *,
    treatment: np.ndarray,
    outcome: np.ndarray | None,
    discrete_outcome: bool,
) -> np.ndarray:
    """Match EconML's deterministic encoding of its discrete stratification arrays."""

    treatment_vector = np.asarray(treatment).reshape(-1)
    arrays = []
    if discrete_outcome:
        if outcome is None:
            raise ValueError("outcome is required to audit discrete-outcome cross-fitting")
        outcome_vector = np.asarray(outcome).reshape(-1)
        if len(outcome_vector) != len(treatment_vector):
            raise ValueError("outcome must align to treatment for cross-fitting")
        arrays.append(outcome_vector)
    arrays.append(treatment_vector)
    strata = np.zeros(len(treatment_vector), dtype=np.int64)
    for values in arrays:
        classes, encoded = np.unique(values, return_inverse=True)
        strata = encoded + strata * len(classes)
    return strata


def _index_sha256(values: np.ndarray) -> str:
    vector = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": "<i8", "shape": list(vector.shape)}).encode("utf-8"))
    digest.update(b"\0")
    digest.update(vector.tobytes(order="C"))
    return digest.hexdigest()


def _signature_names(owner: type[Any]) -> tuple[str, ...]:
    return tuple(inspect.signature(owner).parameters)


def assert_supported_constructor_signatures(
    *,
    causal_forest_class: type[Any],
    treatment_forest_class: type[Any],
    outcome_forest_class: type[Any],
    stratified_crossfit_class: type[Any],
    outcome_forest_is_classifier: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Fail if an installed constructor parameter has not been classified."""

    expected = {
        "causal_forest": _CAUSAL_FOREST_SIGNATURE,
        "treatment_forest": _TREATMENT_FOREST_SIGNATURE,
        "outcome_forest": (
            _TREATMENT_FOREST_SIGNATURE
            if outcome_forest_is_classifier
            else _OUTCOME_FOREST_SIGNATURE
        ),
        "stratified_crossfit": _STRATIFIED_CROSSFIT_SIGNATURE,
    }
    actual = {
        "causal_forest": _signature_names(causal_forest_class),
        "treatment_forest": _signature_names(treatment_forest_class),
        "outcome_forest": _signature_names(outcome_forest_class),
        "stratified_crossfit": _signature_names(stratified_crossfit_class),
    }
    expected_kinds = {
        "causal_forest": _CAUSAL_FOREST_SIGNATURE_KINDS,
        "treatment_forest": _TREATMENT_FOREST_SIGNATURE_KINDS,
        "outcome_forest": (
            _TREATMENT_FOREST_SIGNATURE_KINDS
            if outcome_forest_is_classifier
            else _OUTCOME_FOREST_SIGNATURE_KINDS
        ),
        "stratified_crossfit": (_STRATIFIED_CROSSFIT_SIGNATURE_KINDS),
    }
    owners = {
        "causal_forest": causal_forest_class,
        "treatment_forest": treatment_forest_class,
        "outcome_forest": outcome_forest_class,
        "stratified_crossfit": stratified_crossfit_class,
    }
    for label in expected:
        if actual[label] != expected[label]:
            raise RuntimeError(
                f"unsupported {label} constructor signature; "
                f"expected={expected[label]!r}, actual={actual[label]!r}"
            )
        actual_kinds = tuple(
            parameter.kind for parameter in inspect.signature(owners[label]).parameters.values()
        )
        if actual_kinds != expected_kinds[label]:
            raise RuntimeError(
                f"unsupported {label} constructor parameter kinds; "
                f"expected={expected_kinds[label]!r}, "
                f"actual={actual_kinds!r}"
            )
    return actual


def _require_exact_class(
    value: Any,
    *,
    expected: type[Any],
    path: str,
) -> None:
    if type(value) is not expected:
        raise RuntimeError(
            f"{path} must be exactly "
            f"{expected.__module__}.{expected.__qualname__}; got "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )


def _normalize_parameter(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_exact_get_params(
    estimator: Any,
    *,
    expected_class: type[Any],
    expected_parameters: Mapping[str, Any],
    path: str,
) -> dict[str, Any]:
    _require_exact_class(estimator, expected=expected_class, path=path)
    get_params = getattr(estimator, "get_params", None)
    if not callable(get_params):
        raise RuntimeError(f"{path} does not expose get_params(deep=False)")
    actual_raw = get_params(deep=False)
    if not isinstance(actual_raw, Mapping):
        raise RuntimeError(f"{path}.get_params(deep=False) was not a mapping")
    actual = {str(key): _normalize_parameter(value) for key, value in actual_raw.items()}
    expected = {str(key): _normalize_parameter(value) for key, value in expected_parameters.items()}
    if actual != expected:
        raise RuntimeError(
            f"{path} effective parameters differ from the strict "
            f"configuration; expected={expected!r}, actual={actual!r}"
        )
    return actual


def _causal_forest_top_level_expected(
    config: StrictCausalForestRuntimeConfig,
) -> dict[str, Any]:
    expected = config.causal_forest.scientific_constructor_kwargs()
    expected.update(
        {
            "n_jobs": 1,
            "verbose": int(config.operational.verbose),
            "use_ray": bool(config.operational.use_ray),
            "ray_remote_func_options": (config.operational.ray_remote_func_options),
        }
    )
    return expected


def _audit_top_level_causal_forest(
    model: Any,
    *,
    config: StrictCausalForestRuntimeConfig,
    causal_forest_class: type[Any],
) -> dict[str, Any]:
    _require_exact_class(
        model,
        expected=causal_forest_class,
        path="causal_forest",
    )
    expected = _causal_forest_top_level_expected(config)
    actual: dict[str, Any] = {}
    for name, expected_value in expected.items():
        if not hasattr(model, name):
            raise RuntimeError(f"causal_forest does not expose configured attribute {name}")
        actual_value = _normalize_parameter(getattr(model, name))
        if actual_value != expected_value:
            raise RuntimeError(
                f"causal_forest.{name} differs from the strict "
                f"configuration: expected={expected_value!r}, "
                f"actual={actual_value!r}"
            )
        actual[name] = actual_value
    return actual


def audit_strict_unfitted_estimator(
    *,
    model: Any,
    config: StrictCausalForestRuntimeConfig,
    causal_forest_class: type[Any],
    treatment_forest_class: type[Any],
    outcome_forest_class: type[Any],
    stratified_crossfit_class: type[Any],
) -> dict[str, Any]:
    """Audit every configured object before any labels are fitted."""

    top_level = _audit_top_level_causal_forest(
        model,
        config=config,
        causal_forest_class=causal_forest_class,
    )
    treatment = _require_exact_get_params(
        getattr(model, "model_t", None),
        expected_class=treatment_forest_class,
        expected_parameters=config.treatment_constructor_kwargs(),
        path="causal_forest.model_t",
    )
    outcome = _require_exact_get_params(
        getattr(model, "model_y", None),
        expected_class=outcome_forest_class,
        expected_parameters=config.outcome_constructor_kwargs(),
        path="causal_forest.model_y",
    )
    crossfit = getattr(model, "cv", None)
    _require_exact_class(
        crossfit,
        expected=stratified_crossfit_class,
        path="causal_forest.cv",
    )
    crossfit_actual = {}
    for name, expected_value in config.crossfit_constructor_kwargs().items():
        if not hasattr(crossfit, name):
            raise RuntimeError(f"causal_forest.cv does not expose {name}")
        actual_value = _normalize_parameter(getattr(crossfit, name))
        if actual_value != expected_value:
            raise RuntimeError(f"causal_forest.cv.{name} differs from the strict " f"configuration")
        crossfit_actual[name] = actual_value
    return {
        "top_level_attributes": top_level,
        "model_t_parameters": treatment,
        "model_y_parameters": outcome,
        "crossfit_parameters": crossfit_actual,
    }


def _grf_expected_parameters(
    config: StrictCausalForestRuntimeConfig,
) -> dict[str, Any]:
    scientific = config.causal_forest
    return {
        "criterion": scientific.criterion,
        "fit_intercept": bool(scientific.fit_intercept),
        "honest": bool(scientific.honest),
        "inference": bool(scientific.inference),
        "max_depth": scientific.max_depth,
        "max_features": scientific.max_features,
        "max_samples": scientific.max_samples,
        "min_balancedness_tol": float(scientific.min_balancedness_tol),
        "min_impurity_decrease": float(scientific.min_impurity_decrease),
        "min_samples_leaf": scientific.min_samples_leaf,
        "min_samples_split": scientific.min_samples_split,
        "min_var_fraction_leaf": scientific.min_var_fraction_leaf,
        "min_var_leaf_on_val": bool(scientific.min_var_leaf_on_val),
        "min_weight_fraction_leaf": float(scientific.min_weight_fraction_leaf),
        "n_estimators": int(scientific.n_estimators),
        "n_jobs": 1,
        "random_state": int(scientific.random_seed),
        "subforest_size": int(scientific.subforest_size),
        "verbose": int(config.operational.verbose),
        "warm_start": False,
    }


def _nested_fitted_models(
    value: Any,
    *,
    path: str,
    expected_repetitions: int,
    expected_folds: int,
) -> list[list[Any]]:
    if not isinstance(value, list) or len(value) != expected_repetitions:
        raise RuntimeError(
            f"{path} must contain exactly {expected_repetitions} " "Monte-Carlo repetitions"
        )
    result: list[list[Any]] = []
    for repetition, models in enumerate(value):
        if not isinstance(models, list) or len(models) != expected_folds:
            raise RuntimeError(
                f"{path}[{repetition}] must contain exactly "
                f"{expected_folds} fitted cross-fit models"
            )
        result.append(models)
    return result


def audit_strict_fitted_estimator(
    *,
    model: Any,
    config: StrictCausalForestRuntimeConfig,
    causal_forest_class: type[Any],
    treatment_forest_class: type[Any],
    outcome_forest_class: type[Any],
    stratified_crossfit_class: type[Any],
    grf_class: type[Any],
) -> dict[str, Any]:
    """Reopen all fitted nuisance clones and all final GRF estimators."""

    unfitted_graph = audit_strict_unfitted_estimator(
        model=model,
        config=config,
        causal_forest_class=causal_forest_class,
        treatment_forest_class=treatment_forest_class,
        outcome_forest_class=outcome_forest_class,
        stratified_crossfit_class=stratified_crossfit_class,
    )
    repetitions = 1 if config.causal_forest.mc_iters is None else int(config.causal_forest.mc_iters)
    folds = int(config.causal_forest.crossfit.n_splits)
    models_t = _nested_fitted_models(
        getattr(model, "models_t", None),
        path="causal_forest.models_t",
        expected_repetitions=repetitions,
        expected_folds=folds,
    )
    models_y = _nested_fitted_models(
        getattr(model, "models_y", None),
        path="causal_forest.models_y",
        expected_repetitions=repetitions,
        expected_folds=folds,
    )
    fitted_treatment = []
    fitted_outcome = []
    for repetition in range(repetitions):
        treatment_row = []
        outcome_row = []
        for fold in range(folds):
            treatment_row.append(
                _require_exact_get_params(
                    models_t[repetition][fold],
                    expected_class=treatment_forest_class,
                    expected_parameters=(config.treatment_constructor_kwargs()),
                    path=(f"causal_forest.models_t[{repetition}][{fold}]"),
                )
            )
            outcome_row.append(
                _require_exact_get_params(
                    models_y[repetition][fold],
                    expected_class=outcome_forest_class,
                    expected_parameters=config.outcome_constructor_kwargs(),
                    path=(f"causal_forest.models_y[{repetition}][{fold}]"),
                )
            )
        fitted_treatment.append(treatment_row)
        fitted_outcome.append(outcome_row)

    model_cate = getattr(model, "model_cate", None)
    if model_cate is None:
        raise RuntimeError("causal_forest does not expose fitted model_cate")
    expected_grf = _grf_expected_parameters(config)
    template = _require_exact_get_params(
        getattr(model_cate, "estimator", None),
        expected_class=grf_class,
        expected_parameters=expected_grf,
        path="causal_forest.model_cate.estimator",
    )
    fitted_grfs = getattr(model_cate, "estimators_", None)
    if not isinstance(fitted_grfs, list) or len(fitted_grfs) != 1:
        raise RuntimeError(
            "binary-treatment causal forest must contain exactly one " "fitted GRF estimator"
        )
    fitted_grf_parameters = [
        _require_exact_get_params(
            estimator,
            expected_class=grf_class,
            expected_parameters=expected_grf,
            path=f"causal_forest.model_cate.estimators_[{index}]",
        )
        for index, estimator in enumerate(fitted_grfs)
    ]
    return {
        "unfitted_estimator_graph": unfitted_graph,
        "fitted_treatment_models": fitted_treatment,
        "fitted_outcome_models": fitted_outcome,
        "model_cate_template_parameters": template,
        "fitted_grf_parameters": fitted_grf_parameters,
    }


__all__ = [
    "CAUSAL_FOREST_IMPLEMENTATION",
    "OUTCOME_CLASSIFIER_IMPLEMENTATION",
    "OUTCOME_FOREST_IMPLEMENTATION",
    "STRICT_CAUSAL_FOREST_OPERATIONAL_SCHEMA",
    "STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA",
    "STRICT_CAUSAL_FOREST_SCIENTIFIC_SCHEMA",
    "STRATIFIED_CROSSFIT_IMPLEMENTATION",
    "StrictCausalForestDMLSpec",
    "StrictCausalForestOperationalSpec",
    "StrictCausalForestRuntimeConfig",
    "StrictRandomForestClassifierSpec",
    "StrictOutcomeRandomForestClassifierSpec",
    "StrictRandomForestRegressorSpec",
    "StrictStratifiedKFoldSpec",
    "TREATMENT_FOREST_IMPLEMENTATION",
    "assert_supported_constructor_signatures",
    "audit_strict_fitted_estimator",
    "audit_strict_unfitted_estimator",
]
