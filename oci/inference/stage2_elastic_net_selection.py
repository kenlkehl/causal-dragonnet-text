"""Fold-local group-elastic-net nuisance and R-learner selection for Stage 2.

The selector has one deliberately narrow job: turn the frozen, extracted
outer-training candidate matrix into three empirical feature sets without
consulting the outer-heldout partition:

* stable predictors of treatment for the propensity nuisance model;
* stable predictors of the marginal outcome for the outcome nuisance model;
* stable treatment-interaction terms selected by an R-learner.

Treatment and outcome supports are retained separately.  Their union is the
adjustment set used by the final causal forest, but the separate supports are
persisted on each definition so the final propensity and outcome forests can
use their own inputs.  R-learner targets are built from inner-fold out-of-fold
nuisance predictions.  No pairwise clustering or latent construction occurs.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

SCHEMA_VERSION = "stage2_group_elastic_net_rlearner_selection_v2"
TEMPORAL_SCOPE = "pre_index_treatment"


@dataclass(frozen=True)
class Stage2ElasticNetSelectionConfig:
    """Scientific and numerical policy for deterministic grouped selection."""

    l1_ratio: float = 0.8
    nuisance_selection_frequency: float = 0.6
    modifier_selection_frequency: float = 0.6
    internal_cv_folds: int = 3
    regularization_grid_size: int = 16
    minimum_log10_alpha: float = -5.0
    maximum_log10_alpha: float = -1.0
    coefficient_tolerance: float = 1e-7
    optimization_tolerance: float = 1e-6
    categorical_min_count: int = 5
    max_iter: int = 5_000
    one_standard_error_rule: bool = True
    modifier_one_standard_error_rule: bool = False
    nuisance_forest_trees: int = 200
    nuisance_forest_min_samples_leaf: int = 10
    modifier_min_mean_r_loss_improvement: float = 0.0
    modifier_min_positive_fold_fraction: float = 0.4

    def validate(self) -> None:
        for name in (
            "l1_ratio",
            "nuisance_selection_frequency",
            "modifier_selection_frequency",
            "modifier_min_positive_fold_fraction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= 1.0
            ):
                raise ValueError(f"stage2.statistical_selection.{name} must be in (0, 1]")
        for name in (
            "internal_cv_folds",
            "regularization_grid_size",
            "categorical_min_count",
            "max_iter",
            "nuisance_forest_trees",
            "nuisance_forest_min_samples_leaf",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"stage2.statistical_selection.{name} must be a positive integer"
                )
        if self.internal_cv_folds < 2:
            raise ValueError(
                "stage2.statistical_selection.internal_cv_folds must be at least 2"
            )
        if self.regularization_grid_size < 3:
            raise ValueError(
                "stage2.statistical_selection.regularization_grid_size must be at least 3"
            )
        if self.nuisance_forest_trees < 10:
            raise ValueError(
                "stage2.statistical_selection.nuisance_forest_trees must be at least 10"
            )
        for name in (
            "minimum_log10_alpha",
            "maximum_log10_alpha",
            "coefficient_tolerance",
            "optimization_tolerance",
            "modifier_min_mean_r_loss_improvement",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"stage2.statistical_selection.{name} must be a finite number"
                )
        if self.minimum_log10_alpha >= self.maximum_log10_alpha:
            raise ValueError(
                "stage2.statistical_selection.minimum_log10_alpha must be smaller than "
                "maximum_log10_alpha"
            )
        if self.coefficient_tolerance < 0.0:
            raise ValueError(
                "stage2.statistical_selection.coefficient_tolerance must be nonnegative"
            )
        if self.optimization_tolerance <= 0.0:
            raise ValueError(
                "stage2.statistical_selection.optimization_tolerance must be positive"
            )
        if self.modifier_min_mean_r_loss_improvement < 0.0:
            raise ValueError(
                "stage2.statistical_selection.modifier_min_mean_r_loss_improvement "
                "must be nonnegative"
            )
        for name in ("one_standard_error_rule", "modifier_one_standard_error_rule"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(
                    f"stage2.statistical_selection.{name} must be boolean"
                )

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def statistical_selection_config_from_mapping(
    value: Mapping[str, Any] | None,
) -> Stage2ElasticNetSelectionConfig:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("stage2.statistical_selection must be an object")
    raw = dict(value or {})
    known = set(Stage2ElasticNetSelectionConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            "stage2.statistical_selection contains unsupported fields: " f"{unknown}"
        )
    config = Stage2ElasticNetSelectionConfig(**raw)
    config.validate()
    return config


def _feature_key(feature: Mapping[str, Any]) -> str:
    return str(feature.get("feature_id") or feature["name"])


def _feature_strategy(feature: Mapping[str, Any]) -> str:
    value_type = str(feature.get("value_type") or "ambiguous").strip().lower()
    if value_type == "ordinal":
        return "ordinal"
    plan = feature.get("harmonization_plan")
    if isinstance(plan, Mapping):
        target = str(plan.get("target_representation") or "").strip().lower()
        if target in {"continuous", "categorical"}:
            return target
    if value_type == "continuous":
        if (
            feature.get("modeling_strategy") == "continuous_with_categorical_fallback"
            or isinstance(feature.get("harmonization_fallback"), Mapping)
        ):
            return "continuous_with_categorical_fallback"
        return "continuous"
    return "categorical"


@dataclass(frozen=True)
class _EncodedDesign:
    train: np.ndarray
    valid: np.ndarray
    column_feature_ids: tuple[str, ...]
    column_names: tuple[str, ...]


def _append_variable_column(
    train_columns: list[np.ndarray],
    valid_columns: list[np.ndarray],
    column_feature_ids: list[str],
    column_names: list[str],
    *,
    feature_id: str,
    name: str,
    train_values: np.ndarray,
    valid_values: np.ndarray,
) -> None:
    train_values = np.asarray(train_values, dtype=float)
    valid_values = np.asarray(valid_values, dtype=float)
    if not np.isfinite(train_values).all() or not np.isfinite(valid_values).all():
        raise ValueError(f"nonfinite encoded values for Stage 2 feature {feature_id!r}")
    if len(train_values) == 0:
        return
    mean = float(np.mean(train_values))
    scale = float(np.std(train_values, ddof=0))
    if not math.isfinite(scale) or scale <= 1e-12:
        return
    # Every penalized column has unit training-fold variance.  Without this,
    # an indicator for a rare category is charged much more heavily than a
    # standardized continuous measurement.
    train_columns.append((train_values - mean) / scale)
    valid_columns.append((valid_values - mean) / scale)
    column_feature_ids.append(feature_id)
    column_names.append(name)


def _ordered_values(
    train_series: pd.Series,
    valid_series: pd.Series,
    definition: Mapping[str, Any],
) -> tuple[pd.Series, pd.Series]:
    """Map an ordinal measurement to one ordered numerical score."""

    train_numeric = pd.to_numeric(train_series, errors="coerce")
    valid_numeric = pd.to_numeric(valid_series, errors="coerce")
    categories = definition.get("categories_or_unit")
    if isinstance(categories, Sequence) and not isinstance(categories, (str, bytes)):
        category_order = {
            str(category).strip().casefold(): float(index)
            for index, category in enumerate(categories)
        }
        train_text = train_series.astype(str).str.strip().str.casefold()
        valid_text = valid_series.astype(str).str.strip().str.casefold()
        train_numeric = train_numeric.where(
            train_numeric.notna(), train_text.map(category_order)
        )
        valid_numeric = valid_numeric.where(
            valid_numeric.notna(), valid_text.map(category_order)
        )
    return train_numeric, valid_numeric


def _encode_design(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    *,
    categorical_min_count: int,
) -> _EncodedDesign:
    """Fit a feature-group-preserving design on one training partition."""

    train_columns: list[np.ndarray] = []
    valid_columns: list[np.ndarray] = []
    column_feature_ids: list[str] = []
    column_names: list[str] = []
    for definition in definitions:
        feature_id = _feature_key(definition)
        name = str(definition["name"])
        train_series = (
            train[name].reset_index(drop=True)
            if name in train
            else pd.Series([None] * len(train), dtype=object)
        )
        valid_series = (
            valid[name].reset_index(drop=True)
            if name in valid
            else pd.Series([None] * len(valid), dtype=object)
        )
        strategy = _feature_strategy(definition)
        if strategy in {
            "continuous",
            "continuous_with_categorical_fallback",
            "ordinal",
        }:
            if strategy == "ordinal":
                train_numeric, valid_numeric = _ordered_values(
                    train_series,
                    valid_series,
                    definition,
                )
            else:
                train_numeric = pd.to_numeric(train_series, errors="coerce")
                valid_numeric = pd.to_numeric(valid_series, errors="coerce")
            observed = train_numeric.dropna()
            median = float(observed.median()) if len(observed) else 0.0
            _append_variable_column(
                train_columns,
                valid_columns,
                column_feature_ids,
                column_names,
                feature_id=feature_id,
                name=(
                    f"{name}:ordered_score"
                    if strategy == "ordinal"
                    else f"{name}:value"
                ),
                train_values=train_numeric.fillna(median).to_numpy(),
                valid_values=valid_numeric.fillna(median).to_numpy(),
            )
            if strategy == "continuous_with_categorical_fallback":
                train_fallback = train_series.notna() & train_numeric.isna()
                valid_fallback = valid_series.notna() & valid_numeric.isna()
                counts = train_series.loc[train_fallback].astype(str).value_counts()
                frequent = sorted(
                    str(level)
                    for level, count in counts.items()
                    if int(count) >= int(categorical_min_count)
                )
                train_text = train_series.astype(str)
                valid_text = valid_series.astype(str)
                for level in frequent:
                    _append_variable_column(
                        train_columns,
                        valid_columns,
                        column_feature_ids,
                        column_names,
                        feature_id=feature_id,
                        name=f"{name}:fallback={level}",
                        train_values=(train_fallback & (train_text == level)).to_numpy(),
                        valid_values=(valid_fallback & (valid_text == level)).to_numpy(),
                    )
            _append_variable_column(
                train_columns,
                valid_columns,
                column_feature_ids,
                column_names,
                feature_id=feature_id,
                name=f"{name}:missing",
                train_values=(
                    train_numeric.isna().to_numpy()
                    if strategy == "ordinal"
                    else train_series.isna().to_numpy()
                ),
                valid_values=(
                    valid_numeric.isna().to_numpy()
                    if strategy == "ordinal"
                    else valid_series.isna().to_numpy()
                ),
            )
            continue

        train_missing = train_series.isna()
        valid_missing = valid_series.isna()
        train_text = train_series.astype(str)
        valid_text = valid_series.astype(str)
        counts = train_text.loc[~train_missing].value_counts()
        frequent = sorted(
            str(level)
            for level, count in counts.items()
            if int(count) >= int(categorical_min_count)
        )
        rare_train = (~train_missing) & (~train_text.isin(frequent))
        rare_valid = (~valid_missing) & (~valid_text.isin(frequent))
        levels = list(frequent)
        if bool(rare_train.any()):
            levels.append("__OTHER__")
        # Reference contrasts are standardized individually and penalized as
        # one feature group.  The group penalty, rather than an arbitrary
        # individual dummy coefficient, determines whether the factor survives.
        for level in levels[1:]:
            if level == "__OTHER__":
                train_values, valid_values = rare_train.to_numpy(), rare_valid.to_numpy()
            else:
                train_values = ((~train_missing) & (train_text == level)).to_numpy()
                valid_values = ((~valid_missing) & (valid_text == level)).to_numpy()
            _append_variable_column(
                train_columns,
                valid_columns,
                column_feature_ids,
                column_names,
                feature_id=feature_id,
                name=f"{name}:level={level}",
                train_values=train_values,
                valid_values=valid_values,
            )
        _append_variable_column(
            train_columns,
            valid_columns,
            column_feature_ids,
            column_names,
            feature_id=feature_id,
            name=f"{name}:missing",
            train_values=train_missing.to_numpy(),
            valid_values=valid_missing.to_numpy(),
        )
    return _EncodedDesign(
        train=(
            np.column_stack(train_columns).astype(float, copy=False)
            if train_columns
            else np.empty((len(train), 0), dtype=float)
        ),
        valid=(
            np.column_stack(valid_columns).astype(float, copy=False)
            if valid_columns
            else np.empty((len(valid), 0), dtype=float)
        ),
        column_feature_ids=tuple(column_feature_ids),
        column_names=tuple(column_names),
    )


@dataclass(frozen=True)
class _PenalizedFit:
    coefficients: np.ndarray
    train_prediction: np.ndarray
    valid_prediction: np.ndarray
    regularization: float | None
    cv_folds: int
    status: str
    iterations: int
    converged: bool


@dataclass(frozen=True)
class _GroupStructure:
    starts: np.ndarray
    sizes: np.ndarray
    weights: np.ndarray
    feature_ids: tuple[str, ...]


@dataclass(frozen=True)
class _SolverState:
    coefficients: np.ndarray
    intercept: float
    iterations: int
    converged: bool


def _group_structure(column_feature_ids: Sequence[str]) -> _GroupStructure:
    if not column_feature_ids:
        return _GroupStructure(
            starts=np.asarray([], dtype=int),
            sizes=np.asarray([], dtype=int),
            weights=np.asarray([], dtype=float),
            feature_ids=(),
        )
    starts = [0]
    feature_ids = [str(column_feature_ids[0])]
    seen = {feature_ids[0]}
    for position, raw_feature_id in enumerate(column_feature_ids[1:], start=1):
        feature_id = str(raw_feature_id)
        if feature_id == feature_ids[-1]:
            continue
        if feature_id in seen:
            raise ValueError("group elastic-net columns must be contiguous by feature")
        starts.append(position)
        feature_ids.append(feature_id)
        seen.add(feature_id)
    start_values = np.asarray(starts, dtype=int)
    sizes = np.diff(np.append(start_values, len(column_feature_ids))).astype(int)
    return _GroupStructure(
        starts=start_values,
        sizes=sizes,
        weights=np.sqrt(sizes.astype(float)),
        feature_ids=tuple(feature_ids),
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _spectral_norm_squared(
    design: np.ndarray,
    *,
    fit_intercept: bool,
    iterations: int = 30,
) -> float:
    """Estimate ||[1, X]||_2^2 deterministically by power iteration."""

    columns = int(design.shape[1]) + int(fit_intercept)
    if columns == 0 or len(design) == 0:
        return 0.0
    vector = np.random.default_rng(0).normal(size=columns)
    vector /= float(np.linalg.norm(vector))
    for _ in range(iterations):
        if fit_intercept:
            projected = vector[0] + design @ vector[1:]
            back = np.concatenate(
                ([float(np.sum(projected))], design.T @ projected)
            )
        else:
            projected = design @ vector
            back = design.T @ projected
        norm = float(np.linalg.norm(back))
        if not math.isfinite(norm) or norm <= 1e-15:
            return 0.0
        vector = back / norm
    if fit_intercept:
        projected = vector[0] + design @ vector[1:]
    else:
        projected = design @ vector
    return float(np.dot(projected, projected))


def _group_ridge_prox(
    values: np.ndarray,
    *,
    step: float,
    regularization: float,
    group_ratio: float,
    groups: _GroupStructure,
) -> np.ndarray:
    if len(values) == 0:
        return values.copy()
    squared = np.square(values)
    norms = np.sqrt(np.add.reduceat(squared, groups.starts))
    thresholds = (
        float(step)
        * float(regularization)
        * float(group_ratio)
        * groups.weights
    )
    scales = np.maximum(0.0, 1.0 - thresholds / np.maximum(norms, 1e-30))
    scales /= 1.0 + (
        float(step) * float(regularization) * (1.0 - float(group_ratio))
    )
    return values * np.repeat(scales, groups.sizes)


def _fit_group_elastic_net(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    groups: _GroupStructure,
    regularization: float,
    group_ratio: float,
    binary: bool,
    fit_intercept: bool,
    max_iter: int,
    tolerance: float,
    initial: _SolverState | None = None,
) -> _SolverState:
    """Fit a convex group-lasso-plus-ridge model with accelerated proximal steps."""

    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    rows, columns = train_x.shape
    if initial is not None and len(initial.coefficients) == columns:
        coefficients = initial.coefficients.copy()
        intercept = float(initial.intercept) if fit_intercept else 0.0
    else:
        coefficients = np.zeros(columns, dtype=float)
        if fit_intercept and binary:
            mean = float(np.clip(np.mean(train_y), 1e-6, 1.0 - 1e-6))
            intercept = float(math.log(mean / (1.0 - mean)))
        elif fit_intercept:
            intercept = float(np.mean(train_y))
        else:
            intercept = 0.0
    spectral_squared = _spectral_norm_squared(
        train_x,
        fit_intercept=fit_intercept,
    )
    loss_curvature = 0.25 if binary else 1.0
    lipschitz = max(
        1.01 * loss_curvature * spectral_squared / max(rows, 1),
        1e-12,
    )
    step = 1.0 / lipschitz
    extrapolated = coefficients.copy()
    extrapolated_intercept = intercept
    momentum = 1.0
    converged = False
    iterations_used = 0
    for iteration in range(1, int(max_iter) + 1):
        linear = train_x @ extrapolated + extrapolated_intercept
        residual = _sigmoid(linear) - train_y if binary else linear - train_y
        gradient = train_x.T @ residual / rows
        intercept_gradient = float(np.mean(residual)) if fit_intercept else 0.0
        updated = _group_ridge_prox(
            extrapolated - step * gradient,
            step=step,
            regularization=regularization,
            group_ratio=group_ratio,
            groups=groups,
        )
        updated_intercept = (
            extrapolated_intercept - step * intercept_gradient
            if fit_intercept
            else 0.0
        )
        delta = math.sqrt(
            float(np.dot(updated - coefficients, updated - coefficients))
            + (updated_intercept - intercept) ** 2
        )
        scale = 1.0 + math.sqrt(
            float(np.dot(coefficients, coefficients)) + intercept**2
        )
        iterations_used = iteration
        if delta <= float(tolerance) * scale:
            coefficients = updated
            intercept = updated_intercept
            converged = True
            break
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        acceleration = (momentum - 1.0) / next_momentum
        next_extrapolated = updated + acceleration * (updated - coefficients)
        next_extrapolated_intercept = updated_intercept + acceleration * (
            updated_intercept - intercept
        )
        # Adaptive restart avoids oscillation in highly correlated clinical groups.
        restart_direction = float(
            np.dot(extrapolated - updated, updated - coefficients)
        ) + (extrapolated_intercept - updated_intercept) * (
            updated_intercept - intercept
        )
        coefficients = updated
        intercept = updated_intercept
        if restart_direction > 0.0:
            extrapolated = updated.copy()
            extrapolated_intercept = updated_intercept
            momentum = 1.0
        else:
            extrapolated = next_extrapolated
            extrapolated_intercept = next_extrapolated_intercept
            momentum = next_momentum
    return _SolverState(
        coefficients=np.asarray(coefficients, dtype=float),
        intercept=float(intercept),
        iterations=iterations_used,
        converged=converged,
    )


def _state_prediction(
    state: _SolverState,
    design: np.ndarray,
    *,
    binary: bool,
) -> np.ndarray:
    linear = np.asarray(design, dtype=float) @ state.coefficients + state.intercept
    if binary:
        return np.clip(_sigmoid(linear), 1e-6, 1.0 - 1e-6)
    return np.asarray(linear, dtype=float)


def _constant_fit(
    train_y: np.ndarray,
    valid_rows: int,
    columns: int,
    *,
    binary: bool,
    status: str,
) -> _PenalizedFit:
    mean = float(np.mean(train_y)) if len(train_y) else 0.0
    if binary:
        mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
    return _PenalizedFit(
        coefficients=np.zeros(columns, dtype=float),
        train_prediction=np.full(len(train_y), mean, dtype=float),
        valid_prediction=np.full(valid_rows, mean, dtype=float),
        regularization=None,
        cv_folds=0,
        status=status,
        iterations=0,
        converged=True,
    )


def _group_elastic_net_cv(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    column_feature_ids: Sequence[str],
    *,
    config: Stage2ElasticNetSelectionConfig,
    seed: int,
    binary: bool,
    fit_intercept: bool,
    one_standard_error_rule: bool,
) -> _PenalizedFit:
    train_y = np.asarray(train_y, dtype=float)
    if binary:
        class_counts = np.bincount(train_y.astype(int), minlength=2)
        cv_folds = min(int(config.internal_cv_folds), int(class_counts.min()))
        splitter: Any = StratifiedKFold(
            n_splits=max(cv_folds, 2),
            shuffle=True,
            random_state=seed,
        )
    else:
        cv_folds = min(int(config.internal_cv_folds), len(train_y))
        splitter = KFold(
            n_splits=max(cv_folds, 2),
            shuffle=True,
            random_state=seed,
        )
    if train_x.shape[1] == 0 or cv_folds < 2:
        if not fit_intercept:
            return _PenalizedFit(
                coefficients=np.zeros(train_x.shape[1], dtype=float),
                train_prediction=np.zeros(len(train_y), dtype=float),
                valid_prediction=np.zeros(len(valid_x), dtype=float),
                regularization=None,
                cv_folds=0,
                status="empty_design",
                iterations=0,
                converged=True,
            )
        return _constant_fit(
            train_y,
            len(valid_x),
            train_x.shape[1],
            binary=binary,
            status="insufficient_rows_or_empty_design",
        )
    groups = _group_structure(column_feature_ids)
    alphas = np.logspace(
        float(config.minimum_log10_alpha),
        float(config.maximum_log10_alpha),
        int(config.regularization_grid_size),
    )
    losses = np.empty((cv_folds, len(alphas)), dtype=float)
    split_iterator = (
        splitter.split(train_x, train_y.astype(int))
        if binary
        else splitter.split(train_x)
    )
    for fold_index, (fit, heldout) in enumerate(split_iterator):
        initial: _SolverState | None = None
        for alpha_index in range(len(alphas) - 1, -1, -1):
            state = _fit_group_elastic_net(
                train_x[fit],
                train_y[fit],
                groups=groups,
                regularization=float(alphas[alpha_index]),
                group_ratio=float(config.l1_ratio),
                binary=binary,
                fit_intercept=fit_intercept,
                max_iter=int(config.max_iter),
                tolerance=float(config.optimization_tolerance),
                initial=initial,
            )
            initial = state
            prediction = _state_prediction(
                state,
                train_x[heldout],
                binary=binary,
            )
            losses[fold_index, alpha_index] = (
                log_loss(train_y[heldout], prediction, labels=[0, 1])
                if binary
                else mean_squared_error(train_y[heldout], prediction)
            )
    means = np.mean(losses, axis=0)
    errors = np.std(losses, axis=0, ddof=1) / math.sqrt(losses.shape[0])
    best = int(np.argmin(means))
    chosen_index = best
    if one_standard_error_rule:
        eligible = np.flatnonzero(means <= means[best] + errors[best])
        if len(eligible):
            chosen_index = int(eligible[-1])
    chosen_alpha = float(alphas[chosen_index])
    state = _fit_group_elastic_net(
        train_x,
        train_y,
        groups=groups,
        regularization=chosen_alpha,
        group_ratio=float(config.l1_ratio),
        binary=binary,
        fit_intercept=fit_intercept,
        max_iter=int(config.max_iter),
        tolerance=float(config.optimization_tolerance),
    )
    return _PenalizedFit(
        coefficients=state.coefficients,
        train_prediction=_state_prediction(state, train_x, binary=binary),
        valid_prediction=_state_prediction(state, valid_x, binary=binary),
        regularization=chosen_alpha,
        cv_folds=cv_folds,
        status="ok" if state.converged else "maximum_iterations_reached",
        iterations=state.iterations,
        converged=state.converged,
    )


def _logistic_elastic_net(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    column_feature_ids: Sequence[str],
    *,
    config: Stage2ElasticNetSelectionConfig,
    seed: int,
) -> _PenalizedFit:
    train_y = np.asarray(train_y, dtype=int)
    if train_x.shape[1] == 0 or len(np.unique(train_y)) < 2:
        return _constant_fit(
            train_y,
            len(valid_x),
            train_x.shape[1],
            binary=True,
            status="constant_or_empty_design",
        )
    return _group_elastic_net_cv(
        train_x,
        train_y,
        valid_x,
        column_feature_ids,
        config=config,
        seed=seed,
        binary=True,
        fit_intercept=True,
        one_standard_error_rule=bool(config.one_standard_error_rule),
    )


def _squared_error_elastic_net(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    column_feature_ids: Sequence[str],
    *,
    config: Stage2ElasticNetSelectionConfig,
    seed: int,
    fit_intercept: bool = True,
    one_standard_error_rule: bool | None = None,
) -> _PenalizedFit:
    use_one_standard_error_rule = (
        config.one_standard_error_rule
        if one_standard_error_rule is None
        else bool(one_standard_error_rule)
    )
    return _group_elastic_net_cv(
        train_x,
        train_y,
        valid_x,
        column_feature_ids,
        config=config,
        seed=seed,
        binary=False,
        fit_intercept=fit_intercept,
        one_standard_error_rule=use_one_standard_error_rule,
    )


def _selected_feature_ids(
    coefficients: np.ndarray,
    column_feature_ids: Sequence[str],
    *,
    tolerance: float,
) -> tuple[list[str], dict[str, float]]:
    squared_magnitudes: dict[str, float] = {}
    for coefficient, feature_id in zip(coefficients, column_feature_ids):
        key = str(feature_id)
        squared_magnitudes[key] = squared_magnitudes.get(key, 0.0) + float(
            coefficient
        ) ** 2
    magnitudes = {
        feature_id: math.sqrt(value)
        for feature_id, value in squared_magnitudes.items()
    }
    selected = sorted(
        feature_id
        for feature_id, magnitude in magnitudes.items()
        if magnitude > float(tolerance)
    )
    return selected, dict(sorted(magnitudes.items()))


def _required_votes(frequency: float, folds: int) -> int:
    return int(math.ceil(float(frequency) * int(folds)))


def _stable_ids(
    votes: Mapping[str, int],
    *,
    frequency: float,
    folds: int,
) -> tuple[set[str], int]:
    required = _required_votes(frequency, folds)
    return {key for key, value in votes.items() if int(value) >= required}, required


class _ConstantProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 1e-6, 1 - 1e-6))

    def predict(self, rows: int) -> np.ndarray:
        return np.full(rows, self.probability, dtype=float)


def _forest_probability(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    *,
    config: Stage2ElasticNetSelectionConfig,
    seed: int,
) -> np.ndarray:
    if train_x.shape[1] == 0 or len(np.unique(train_y)) < 2:
        return _ConstantProbabilityModel(float(np.mean(train_y))).predict(len(valid_x))
    model = RandomForestClassifier(
        n_estimators=int(config.nuisance_forest_trees),
        min_samples_leaf=int(config.nuisance_forest_min_samples_leaf),
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
    )
    model.fit(train_x, np.asarray(train_y, dtype=int))
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(valid_x), dtype=float)
    return np.clip(model.predict_proba(valid_x)[:, classes.index(1)], 1e-6, 1 - 1e-6)


def _forest_outcome(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    *,
    binary: bool,
    config: Stage2ElasticNetSelectionConfig,
    seed: int,
) -> np.ndarray:
    if train_x.shape[1] == 0:
        mean = float(np.mean(train_y))
        return np.full(len(valid_x), mean, dtype=float)
    if binary:
        return _forest_probability(
            train_x,
            train_y,
            valid_x,
            config=config,
            seed=seed,
        )
    model = RandomForestRegressor(
        n_estimators=int(config.nuisance_forest_trees),
        min_samples_leaf=int(config.nuisance_forest_min_samples_leaf),
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
    )
    model.fit(train_x, np.asarray(train_y, dtype=float))
    return np.asarray(model.predict(valid_x), dtype=float)


def _loss(observed: np.ndarray, predicted: np.ndarray, *, binary: bool) -> float:
    if binary:
        return float(
            log_loss(observed, np.clip(predicted, 1e-6, 1 - 1e-6), labels=[0, 1])
        )
    return float(mean_squared_error(observed, predicted))


def select_stage2_features_elastic_net(
    *,
    dataset: pd.DataFrame,
    extracted_fit: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    inner_splits: Sequence[Mapping[str, Any]],
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    seed: int,
    policy: Stage2ElasticNetSelectionConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[Any]]:
    """Select nuisance inputs and effect modifiers inside one outer fold."""

    policy.validate()
    original = [copy.deepcopy(dict(feature)) for feature in definitions]
    if not original:
        report = {
            "schema_version": SCHEMA_VERSION,
            "temporal_scope": TEMPORAL_SCOPE,
            "status": "complete_no_candidates",
            "policy": policy.public_dict(),
            "decisions": [],
        }
        return [], report, [], []
    folds = list(inner_splits)
    if not folds:
        raise ValueError("Stage 2 group-elastic-net selection requires inner folds")
    if str(outcome_type) not in {"binary", "continuous"}:
        raise ValueError("Stage 2 outcome_type must be binary or continuous")
    by_id = {_feature_key(feature): feature for feature in original}
    if len(by_id) != len(original):
        raise ValueError(
            "Stage 2 group-elastic-net selection requires unique feature IDs"
        )
    extracted_by_id = extracted_fit.set_index("_oci_row_id", drop=False)
    treatment_votes = {feature_id: 0 for feature_id in by_id}
    outcome_votes = {feature_id: 0 for feature_id in by_id}
    screen_folds: list[dict[str, Any]] = []
    binary_outcome = str(outcome_type) == "binary"

    for position, split in enumerate(folds, start=1):
        train_ids = [int(value) for value in split.get("fit_row_ids") or []]
        valid_ids = [int(value) for value in split.get("heldout_row_ids") or []]
        if not train_ids or not valid_ids:
            raise ValueError("each Stage 2 inner fold must have fit and heldout rows")
        train = extracted_by_id.loc[train_ids].reset_index(drop=True)
        valid = extracted_by_id.loc[valid_ids].reset_index(drop=True)
        design = _encode_design(
            train,
            valid,
            original,
            categorical_min_count=int(policy.categorical_min_count),
        )
        t_train = dataset.iloc[train_ids][treatment_column].to_numpy(dtype=float)
        t_valid = dataset.iloc[valid_ids][treatment_column].to_numpy(dtype=float)
        y_train = dataset.iloc[train_ids][outcome_column].to_numpy(dtype=float)
        y_valid = dataset.iloc[valid_ids][outcome_column].to_numpy(dtype=float)
        treatment_fit = _logistic_elastic_net(
            design.train,
            t_train,
            design.valid,
            design.column_feature_ids,
            config=policy,
            seed=seed + 100 * position,
        )
        outcome_fit = (
            _logistic_elastic_net(
                design.train,
                y_train,
                design.valid,
                design.column_feature_ids,
                config=policy,
                seed=seed + 100 * position + 1,
            )
            if binary_outcome
            else _squared_error_elastic_net(
                design.train,
                y_train,
                design.valid,
                design.column_feature_ids,
                config=policy,
                seed=seed + 100 * position + 1,
            )
        )
        treatment_selected, treatment_magnitudes = _selected_feature_ids(
            treatment_fit.coefficients,
            design.column_feature_ids,
            tolerance=float(policy.coefficient_tolerance),
        )
        outcome_selected, outcome_magnitudes = _selected_feature_ids(
            outcome_fit.coefficients,
            design.column_feature_ids,
            tolerance=float(policy.coefficient_tolerance),
        )
        for feature_id in treatment_selected:
            treatment_votes[feature_id] += 1
        for feature_id in outcome_selected:
            outcome_votes[feature_id] += 1
        screen_folds.append(
            {
                "inner_fold": int(split.get("inner_fold", position)),
                "fit_rows": len(train_ids),
                "heldout_rows": len(valid_ids),
                "encoded_columns": int(design.train.shape[1]),
                "treatment": {
                    "status": treatment_fit.status,
                    "selected_feature_ids": treatment_selected,
                    "feature_group_l2_norms": treatment_magnitudes,
                    "regularization_alpha": treatment_fit.regularization,
                    "internal_cv_folds": treatment_fit.cv_folds,
                    "solver_iterations": treatment_fit.iterations,
                    "solver_converged": treatment_fit.converged,
                    "heldout_log_loss": float(
                        log_loss(
                            t_valid,
                            treatment_fit.valid_prediction,
                            labels=[0, 1],
                        )
                    ),
                },
                "outcome": {
                    "status": outcome_fit.status,
                    "selected_feature_ids": outcome_selected,
                    "feature_group_l2_norms": outcome_magnitudes,
                    "regularization": outcome_fit.regularization,
                    "internal_cv_folds": outcome_fit.cv_folds,
                    "solver_iterations": outcome_fit.iterations,
                    "solver_converged": outcome_fit.converged,
                    "heldout_loss": _loss(
                        y_valid,
                        outcome_fit.valid_prediction,
                        binary=binary_outcome,
                    ),
                },
            }
        )

    stable_treatment, nuisance_required = _stable_ids(
        treatment_votes,
        frequency=float(policy.nuisance_selection_frequency),
        folds=len(folds),
    )
    stable_outcome, _ = _stable_ids(
        outcome_votes,
        frequency=float(policy.nuisance_selection_frequency),
        folds=len(folds),
    )
    locked_confounders = {
        _feature_key(feature)
        for feature in original
        if feature.get("configured_explicit_feature") is True
        and "confounder" in set(map(str, feature.get("roles") or []))
    }
    stable_treatment.update(locked_confounders)
    stable_outcome.update(locked_confounders)

    treatment_definitions = [
        by_id[feature_id] for feature_id in by_id if feature_id in stable_treatment
    ]
    outcome_definitions = [
        by_id[feature_id] for feature_id in by_id if feature_id in stable_outcome
    ]
    all_fit_ids = sorted(
        {
            int(value)
            for split in folds
            for key in ("fit_row_ids", "heldout_row_ids")
            for value in split.get(key) or []
        }
    )
    id_position = {row_id: position for position, row_id in enumerate(all_fit_ids)}
    oof_e = np.full(len(all_fit_ids), np.nan, dtype=float)
    oof_m = np.full(len(all_fit_ids), np.nan, dtype=float)
    nuisance_folds: list[dict[str, Any]] = []
    for position, split in enumerate(folds, start=1):
        train_ids = [int(value) for value in split.get("fit_row_ids") or []]
        valid_ids = [int(value) for value in split.get("heldout_row_ids") or []]
        train = extracted_by_id.loc[train_ids].reset_index(drop=True)
        valid = extracted_by_id.loc[valid_ids].reset_index(drop=True)
        t_design = _encode_design(
            train,
            valid,
            treatment_definitions,
            categorical_min_count=int(policy.categorical_min_count),
        )
        y_design = _encode_design(
            train,
            valid,
            outcome_definitions,
            categorical_min_count=int(policy.categorical_min_count),
        )
        t_train = dataset.iloc[train_ids][treatment_column].to_numpy(dtype=float)
        t_valid = dataset.iloc[valid_ids][treatment_column].to_numpy(dtype=float)
        y_train = dataset.iloc[train_ids][outcome_column].to_numpy(dtype=float)
        y_valid = dataset.iloc[valid_ids][outcome_column].to_numpy(dtype=float)
        e_valid = _forest_probability(
            t_design.train,
            t_train,
            t_design.valid,
            config=policy,
            seed=seed + 10_000 + position,
        )
        m_valid = _forest_outcome(
            y_design.train,
            y_train,
            y_design.valid,
            binary=binary_outcome,
            config=policy,
            seed=seed + 20_000 + position,
        )
        locations = [id_position[row_id] for row_id in valid_ids]
        if np.isfinite(oof_e[locations]).any() or np.isfinite(oof_m[locations]).any():
            raise ValueError("Stage 2 inner folds predict an outer-training row more than once")
        oof_e[locations] = e_valid
        oof_m[locations] = m_valid
        nuisance_folds.append(
            {
                "inner_fold": int(split.get("inner_fold", position)),
                "fit_rows": len(train_ids),
                "heldout_rows": len(valid_ids),
                "treatment_features": len(treatment_definitions),
                "outcome_features": len(outcome_definitions),
                "treatment_encoded_columns": int(t_design.train.shape[1]),
                "outcome_encoded_columns": int(y_design.train.shape[1]),
                "heldout_treatment_log_loss": float(
                    log_loss(t_valid, e_valid, labels=[0, 1])
                ),
                "heldout_outcome_loss": _loss(y_valid, m_valid, binary=binary_outcome),
            }
        )
    if np.isnan(oof_e).any() or np.isnan(oof_m).any():
        raise ValueError(
            "Stage 2 inner splits must provide one out-of-fold nuisance prediction "
            "for every outer-training row"
        )
    t_all = dataset.iloc[all_fit_ids][treatment_column].to_numpy(dtype=float)
    y_all = dataset.iloc[all_fit_ids][outcome_column].to_numpy(dtype=float)
    residual_t = t_all - oof_e
    residual_y = y_all - oof_m

    modifier_votes = {feature_id: 0 for feature_id in by_id}
    modifier_folds: list[dict[str, Any]] = []
    for position, split in enumerate(folds, start=1):
        train_ids = [int(value) for value in split.get("fit_row_ids") or []]
        valid_ids = [int(value) for value in split.get("heldout_row_ids") or []]
        train = extracted_by_id.loc[train_ids].reset_index(drop=True)
        valid = extracted_by_id.loc[valid_ids].reset_index(drop=True)
        design = _encode_design(
            train,
            valid,
            original,
            categorical_min_count=int(policy.categorical_min_count),
        )
        train_positions = np.asarray([id_position[row_id] for row_id in train_ids], dtype=int)
        valid_positions = np.asarray([id_position[row_id] for row_id in valid_ids], dtype=int)
        rt_train, rt_valid = residual_t[train_positions], residual_t[valid_positions]
        ry_train, ry_valid = residual_y[train_positions], residual_y[valid_positions]
        weights = rt_train**2
        centers = (
            np.average(design.train, axis=0, weights=weights)
            if design.train.shape[1] and float(np.sum(weights)) > 1e-12
            else np.zeros(design.train.shape[1], dtype=float)
        )
        centered_train = design.train - centers
        centered_valid = design.valid - centers
        denominator = float(np.sum(weights))
        tau0 = (
            float(np.sum(rt_train * ry_train) / denominator)
            if denominator > 1e-12
            else 0.0
        )
        modifier_target = ry_train - rt_train * tau0
        modifier_fit = _squared_error_elastic_net(
            rt_train.reshape(-1, 1) * centered_train,
            modifier_target,
            rt_valid.reshape(-1, 1) * centered_valid,
            design.column_feature_ids,
            config=policy,
            seed=seed + 30_000 + position,
            fit_intercept=False,
            one_standard_error_rule=policy.modifier_one_standard_error_rule,
        )
        selected_ids, magnitudes = _selected_feature_ids(
            modifier_fit.coefficients,
            design.column_feature_ids,
            tolerance=float(policy.coefficient_tolerance),
        )
        for feature_id in selected_ids:
            modifier_votes[feature_id] += 1
        null_prediction = rt_valid * tau0
        full_prediction = null_prediction + modifier_fit.valid_prediction
        null_loss = float(mean_squared_error(ry_valid, null_prediction))
        full_loss = float(mean_squared_error(ry_valid, full_prediction))
        modifier_folds.append(
            {
                "inner_fold": int(split.get("inner_fold", position)),
                "fit_rows": len(train_ids),
                "heldout_rows": len(valid_ids),
                "encoded_modifier_columns": int(design.train.shape[1]),
                "constant_effect": tau0,
                "status": modifier_fit.status,
                "regularization_alpha": modifier_fit.regularization,
                "internal_cv_folds": modifier_fit.cv_folds,
                "selected_feature_ids": selected_ids,
                "feature_group_l2_norms": magnitudes,
                "solver_iterations": modifier_fit.iterations,
                "solver_converged": modifier_fit.converged,
                "null_r_loss": null_loss,
                "selected_r_loss": full_loss,
                "heldout_r_loss_improvement": null_loss - full_loss,
            }
        )

    stable_modifiers, modifier_required = _stable_ids(
        modifier_votes,
        frequency=float(policy.modifier_selection_frequency),
        folds=len(folds),
    )
    locked_modifiers = {
        _feature_key(feature)
        for feature in original
        if feature.get("configured_explicit_feature") is True
        and "effect_modifier" in set(map(str, feature.get("roles") or []))
    }
    r_improvements = [float(row["heldout_r_loss_improvement"]) for row in modifier_folds]
    mean_r_improvement = float(np.mean(r_improvements))
    positive_fraction = float(np.mean(np.asarray(r_improvements) > 0.0))
    modifier_set_supported = bool(
        mean_r_improvement
        >= float(policy.modifier_min_mean_r_loss_improvement)
        and positive_fraction >= float(policy.modifier_min_positive_fold_fraction)
    )
    if not modifier_set_supported:
        stable_modifiers.clear()
    stable_modifiers.update(locked_modifiers)

    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for feature in original:
        feature_id = _feature_key(feature)
        nuisance_roles: list[str] = []
        if feature_id in stable_treatment:
            nuisance_roles.append("treatment")
        if feature_id in stable_outcome:
            nuisance_roles.append("outcome")
        roles: list[str] = []
        if nuisance_roles:
            roles.append("confounder")
        if feature_id in stable_modifiers:
            roles.append("effect_modifier")
        configured = feature.get("configured_explicit_feature") is True
        if configured:
            configured_roles = list(dict.fromkeys(map(str, feature.get("roles") or [])))
            roles = configured_roles
            if "confounder" in configured_roles:
                nuisance_roles = ["treatment", "outcome"]
        retained = bool(roles)
        decisions.append(
            {
                "feature_id": feature_id,
                "name": str(feature["name"]),
                "configured_explicit_feature": configured,
                "treatment_votes": int(treatment_votes[feature_id]),
                "outcome_votes": int(outcome_votes[feature_id]),
                "modifier_votes": int(modifier_votes[feature_id]),
                "treatment_selection_frequency": float(
                    treatment_votes[feature_id] / len(folds)
                ),
                "outcome_selection_frequency": float(outcome_votes[feature_id] / len(folds)),
                "modifier_selection_frequency": float(
                    modifier_votes[feature_id] / len(folds)
                ),
                "nuisance_model_roles": nuisance_roles,
                "roles": roles,
                "retained": retained,
                "selection_source": (
                    "investigator_locked"
                    if configured
                    else "group_elastic_net_inner_fold_stability"
                ),
            }
        )
        if retained:
            updated = copy.deepcopy(feature)
            updated["roles"] = roles
            updated["nuisance_model_roles"] = nuisance_roles
            updated["selection_source"] = (
                "investigator_locked"
                if configured
                else "group_elastic_net_inner_fold_stability"
            )
            selected.append(updated)

    report = {
        "schema_version": SCHEMA_VERSION,
        "temporal_scope": TEMPORAL_SCOPE,
        "status": "complete",
        "selection_method": (
            "separate_group_elastic_net_nuisance_supports_and_group_r_learner"
        ),
        "penalized_model_family": "group_lasso_plus_ridge",
        "encoding": {
            "ordinal": "single_training_standardized_ordered_score",
            "nominal": "training_standardized_reference_contrasts",
            "missing_indicator": "same_penalty_group_as_measurement",
            "group_weight": "square_root_of_encoded_group_rank",
        },
        "latent_construction": "disabled",
        "pairwise_association_screen": "disabled",
        "policy": policy.public_dict(),
        "inner_folds": len(folds),
        "nuisance_screen": {
            "folds": screen_folds,
            "required_votes": nuisance_required,
            "treatment_votes": dict(sorted(treatment_votes.items())),
            "outcome_votes": dict(sorted(outcome_votes.items())),
            "stable_treatment_feature_ids": sorted(stable_treatment),
            "stable_outcome_feature_ids": sorted(stable_outcome),
            "intersection_is_not_a_selection_gate": True,
        },
        "cross_fitted_nuisance_models": {
            "model_family": "random_forest",
            "treatment_feature_ids": sorted(stable_treatment),
            "outcome_feature_ids": sorted(stable_outcome),
            "folds": nuisance_folds,
            "overall_treatment_log_loss": float(
                log_loss(t_all, oof_e, labels=[0, 1])
            ),
            "overall_outcome_loss": _loss(y_all, oof_m, binary=binary_outcome),
            "propensity_min": float(np.min(oof_e)),
            "propensity_max": float(np.max(oof_e)),
            "predictions_are_inner_fold_out_of_fold": True,
        },
        "effect_modifier_screen": {
            "objective": (
                "squared_R_loss_on_residualized_outcome_and_treatment; "
                "per-row squared loss is not used as a regression target"
            ),
            "folds": modifier_folds,
            "required_votes": modifier_required,
            "votes": dict(sorted(modifier_votes.items())),
            "mean_heldout_r_loss_improvement": mean_r_improvement,
            "positive_fold_fraction": positive_fraction,
            "set_passed_heldout_r_loss_gate": modifier_set_supported,
            "stable_effect_modifier_feature_ids": sorted(stable_modifiers),
        },
        "decisions": decisions,
        "retained_feature_ids": [_feature_key(feature) for feature in selected],
        "measurement_dependency_feature_ids": [
            _feature_key(feature) for feature in selected
        ],
    }
    return selected, report, [copy.deepcopy(feature) for feature in selected], []


__all__ = [
    "SCHEMA_VERSION",
    "TEMPORAL_SCOPE",
    "Stage2ElasticNetSelectionConfig",
    "select_stage2_features_elastic_net",
    "statistical_selection_config_from_mapping",
]
