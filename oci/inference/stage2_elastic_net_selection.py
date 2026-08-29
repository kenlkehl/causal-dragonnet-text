"""Fold-local nuisance selection and interaction screening for Stage 2.

The selector has one deliberately narrow job: turn the frozen, extracted
outer-training candidate matrix into three empirical feature sets without
consulting the outer-heldout partition:

* predictors selected in any inner fold for treatment or marginal outcome;
* candidate modifiers with a top-ranked treatment interaction in any inner fold.

The union of treatment and outcome selections is the outer-fold confounder set
and is used by both nuisance models.  Those models produce cross-fitted
treatment and outcome predictions.  Within each inner fold, every candidate is
then screened in its own outcome model containing those predictions, observed
treatment, the candidate main effect, and the treatment-by-candidate
interaction.  The candidates with the N smallest interaction p-values are
retained, with categorical interactions tested jointly.  The union of those
inner-fold top-N sets is passed to the final causal forest.  No pairwise
clustering or latent construction occurs here.
"""

from __future__ import annotations

import copy
import logging
import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = (
    "stage2_group_elastic_net_univariable_modifier_selection_v4_top_n_union"
)
TEMPORAL_SCOPE = "pre_index_treatment"


@dataclass(frozen=True)
class Stage2ElasticNetSelectionConfig:
    """Scientific and numerical policy for deterministic grouped selection."""

    l1_ratio: float = 0.8
    nuisance_selection_rule: str = "any_inner_fold_union"
    modifier_selection_rule: str = "any_inner_fold_union"
    # Retained only so older configuration files remain loadable. Selection no
    # longer uses frequency thresholds; public_dict deliberately omits them.
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
    modifier_top_n_per_inner_fold: int = 5
    # Retained only so configurations written for the R-learner selector remain
    # loadable.  These settings no longer affect modifier discovery and are
    # omitted from public_dict so they do not misdescribe the scientific policy.
    modifier_min_mean_r_loss_improvement: float = 0.0
    modifier_min_positive_fold_fraction: float = 0.4

    def validate(self) -> None:
        for name in ("nuisance_selection_rule", "modifier_selection_rule"):
            if getattr(self, name) != "any_inner_fold_union":
                raise ValueError(
                    f"stage2.statistical_selection.{name} must be "
                    "'any_inner_fold_union'"
                )
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
            "modifier_top_n_per_inner_fold",
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
        result = asdict(self)
        for retired in (
            "nuisance_selection_frequency",
            "modifier_selection_frequency",
            "modifier_one_standard_error_rule",
            "modifier_min_mean_r_loss_improvement",
            "modifier_min_positive_fold_fraction",
        ):
            result.pop(retired, None)
        return result


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


def _selected_in_any_inner_fold(votes: Mapping[str, int]) -> set[str]:
    return {key for key, value in votes.items() if int(value) >= 1}


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


def _safe_auroc(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    observed = np.asarray(observed)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if int(np.sum(mask)) < 2 or len(np.unique(observed[mask])) < 2:
        return None
    return float(roc_auc_score(observed[mask].astype(int), predicted[mask]))


def _rank_safe_columns(
    base: np.ndarray,
    additions: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Append columns only when they add rank, preserving source order."""

    current = np.asarray(base, dtype=float)
    additions = np.asarray(additions, dtype=float)
    if additions.ndim != 2:
        raise ValueError("Stage 2 modifier additions must be a two-dimensional matrix")
    rank = int(np.linalg.matrix_rank(current))
    kept: list[int] = []
    for index in range(additions.shape[1]):
        candidate = np.column_stack([current, additions[:, index]])
        next_rank = int(np.linalg.matrix_rank(candidate))
        if next_rank > rank:
            current = candidate
            rank = next_rank
            kept.append(index)
    return current, kept


def _binary_nested_interaction_p_value(
    target: np.ndarray,
    reduced: np.ndarray,
    interactions: np.ndarray,
) -> tuple[float | None, dict[str, Any]]:
    """Likelihood-ratio p-value for one or more logistic interaction terms."""

    target = np.asarray(target, dtype=float)
    if len(np.unique(target)) != 2:
        return None, {"status": "not_evaluable", "reason": "outcome_has_one_class"}
    full, kept = _rank_safe_columns(reduced, interactions)
    degrees = int(full.shape[1] - reduced.shape[1])
    if degrees < 1:
        return None, {
            "status": "not_evaluable",
            "reason": "no_independent_interaction_columns",
        }
    try:
        import statsmodels.api as sm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reduced_fit = sm.GLM(
                target,
                reduced,
                family=sm.families.Binomial(),
            ).fit(disp=0)
            full_fit = sm.GLM(
                target,
                full,
                family=sm.families.Binomial(),
            ).fit(disp=0)
        if not bool(getattr(reduced_fit, "converged", True)) or not bool(
            getattr(full_fit, "converged", True)
        ):
            raise ValueError("logistic regression did not converge")
        reduced_llf = float(reduced_fit.llf)
        full_llf = float(full_fit.llf)
        if not math.isfinite(reduced_llf) or not math.isfinite(full_llf):
            raise ValueError("nonfinite logistic log likelihood")
        statistic = max(0.0, 2.0 * (full_llf - reduced_llf))
        p_value = float(stats.chi2.sf(statistic, degrees))
        if not math.isfinite(p_value):
            raise ValueError("nonfinite likelihood-ratio p-value")
    except Exception as exc:
        return None, {
            "status": "not_evaluable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return p_value, {
        "status": "ok",
        "test": "likelihood_ratio_chi_square",
        "statistic": statistic,
        "degrees_of_freedom": degrees,
        "tested_column_indices": kept,
    }


def _continuous_nested_interaction_p_value(
    target: np.ndarray,
    reduced: np.ndarray,
    interactions: np.ndarray,
) -> tuple[float | None, dict[str, Any]]:
    """Partial-F p-value preserving support for continuous-outcome workflows."""

    target = np.asarray(target, dtype=float)
    full, kept = _rank_safe_columns(reduced, interactions)
    degrees = int(full.shape[1] - reduced.shape[1])
    residual_degrees = int(len(target) - full.shape[1])
    if degrees < 1 or residual_degrees < 1:
        return None, {
            "status": "not_evaluable",
            "reason": "insufficient_independent_columns_or_residual_degrees_of_freedom",
        }
    try:
        reduced_coefficients = np.linalg.lstsq(reduced, target, rcond=None)[0]
        full_coefficients = np.linalg.lstsq(full, target, rcond=None)[0]
        reduced_residual = target - reduced @ reduced_coefficients
        full_residual = target - full @ full_coefficients
        reduced_ss = float(reduced_residual @ reduced_residual)
        full_ss = float(full_residual @ full_residual)
        if full_ss <= 1e-15:
            statistic = math.inf if reduced_ss > full_ss + 1e-15 else 0.0
        else:
            numerator = max(0.0, (reduced_ss - full_ss) / degrees)
            statistic = numerator / (full_ss / residual_degrees)
        p_value = float(stats.f.sf(statistic, degrees, residual_degrees))
        if not math.isfinite(p_value):
            raise ValueError("nonfinite partial-F p-value")
    except Exception as exc:
        return None, {
            "status": "not_evaluable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return p_value, {
        "status": "ok",
        "test": "partial_f",
        "statistic": statistic,
        "degrees_of_freedom": [degrees, residual_degrees],
        "tested_column_indices": kept,
    }


def _modifier_interaction_test(
    *,
    frame: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    reduced_base: np.ndarray,
    feature: Mapping[str, Any],
    binary_outcome: bool,
    categorical_min_count: int,
) -> dict[str, Any]:
    """Fit the candidate-specific outcome model and test its interaction group."""

    design = _encode_design(
        frame,
        frame,
        [feature],
        categorical_min_count=int(categorical_min_count),
    )
    reduced, kept_main = _rank_safe_columns(reduced_base, design.train)
    interaction_indices = [
        index
        for index, name in enumerate(design.column_names)
        if not str(name).endswith(":missing")
    ]
    interaction_columns = (
        treatment.reshape(-1, 1) * design.train[:, interaction_indices]
        if interaction_indices
        else np.empty((len(frame), 0), dtype=float)
    )
    if binary_outcome:
        p_value, test = _binary_nested_interaction_p_value(
            outcome,
            reduced,
            interaction_columns,
        )
    else:
        p_value, test = _continuous_nested_interaction_p_value(
            outcome,
            reduced,
            interaction_columns,
        )
    interaction_names = [
        str(design.column_names[index]) for index in interaction_indices
    ]
    tested_indices = [int(value) for value in test.get("tested_column_indices") or []]
    strategy = _feature_strategy(feature)
    return {
        "feature_id": _feature_key(feature),
        "name": str(feature["name"]),
        "candidate_strategy": strategy,
        "encoded_main_columns": [
            str(design.column_names[index]) for index in kept_main
        ],
        "candidate_interaction_columns": interaction_names,
        "tested_interaction_columns": [
            interaction_names[index]
            for index in tested_indices
            if 0 <= index < len(interaction_names)
        ],
        "categorical_interactions_are_grouped": strategy == "categorical",
        "missingness_interactions": False,
        "interaction_p_value": p_value,
        "interaction_test": test,
    }


def _rank_modifier_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    evaluable = [row for row in rows if row.get("interaction_p_value") is not None]
    return [
        {
            "rank": rank,
            "feature_id": str(row["feature_id"]),
            "name": str(row["name"]),
            "p_value": float(row["interaction_p_value"]),
        }
        for rank, row in enumerate(
            sorted(
                evaluable,
                key=lambda row: (
                    float(row["interaction_p_value"]),
                    str(row["feature_id"]),
                ),
            ),
            start=1,
        )
    ]


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
    screen_t_observed: list[np.ndarray] = []
    screen_t_predicted: list[np.ndarray] = []
    screen_y_observed: list[np.ndarray] = []
    screen_y_predicted: list[np.ndarray] = []

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
        screen_t_observed.append(t_valid)
        screen_t_predicted.append(treatment_fit.valid_prediction)
        screen_y_observed.append(y_valid)
        screen_y_predicted.append(outcome_fit.valid_prediction)
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
                    "heldout_auroc": _safe_auroc(
                        t_valid,
                        treatment_fit.valid_prediction,
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
                    "heldout_auroc": (
                        _safe_auroc(y_valid, outcome_fit.valid_prediction)
                        if binary_outcome
                        else None
                    ),
                },
            }
        )

    any_treatment = _selected_in_any_inner_fold(treatment_votes)
    any_outcome = _selected_in_any_inner_fold(outcome_votes)
    locked_confounders = {
        _feature_key(feature)
        for feature in original
        if feature.get("configured_explicit_feature") is True
        and "confounder" in set(map(str, feature.get("roles") or []))
    }
    confounder_union = any_treatment | any_outcome | locked_confounders

    nuisance_definitions = [
        by_id[feature_id] for feature_id in by_id if feature_id in confounder_union
    ]
    treatment_definitions = nuisance_definitions
    outcome_definitions = nuisance_definitions
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
                "heldout_treatment_auroc": _safe_auroc(t_valid, e_valid),
                "heldout_outcome_loss": _loss(y_valid, m_valid, binary=binary_outcome),
                "heldout_outcome_auroc": (
                    _safe_auroc(y_valid, m_valid) if binary_outcome else None
                ),
            }
        )
    if np.isnan(oof_e).any() or np.isnan(oof_m).any():
        raise ValueError(
            "Stage 2 inner splits must provide one out-of-fold nuisance prediction "
            "for every outer-training row"
        )
    t_all = dataset.iloc[all_fit_ids][treatment_column].to_numpy(dtype=float)
    y_all = dataset.iloc[all_fit_ids][outcome_column].to_numpy(dtype=float)

    modifier_votes = {feature_id: 0 for feature_id in by_id}
    modifier_p_values = {feature_id: [] for feature_id in by_id}
    modifier_folds: list[dict[str, Any]] = []
    for position, split in enumerate(folds, start=1):
        train_ids = [int(value) for value in split.get("fit_row_ids") or []]
        valid_ids = [int(value) for value in split.get("heldout_row_ids") or []]
        train = extracted_by_id.loc[train_ids].reset_index(drop=True)
        train_positions = np.asarray(
            [id_position[row_id] for row_id in train_ids],
            dtype=int,
        )
        treatment = dataset.iloc[train_ids][treatment_column].to_numpy(dtype=float)
        outcome = dataset.iloc[train_ids][outcome_column].to_numpy(dtype=float)
        base_names = ["treatment_prediction", "outcome_prediction", "treatment"]
        base_inputs = np.column_stack(
            [oof_e[train_positions], oof_m[train_positions], treatment]
        )
        reduced_base, kept_base = _rank_safe_columns(
            np.ones((len(train), 1), dtype=float),
            base_inputs,
        )
        tests = [
            _modifier_interaction_test(
                frame=train,
                treatment=treatment,
                outcome=outcome,
                reduced_base=reduced_base,
                feature=feature,
                binary_outcome=binary_outcome,
                categorical_min_count=int(policy.categorical_min_count),
            )
            for feature in original
        ]
        ranking = _rank_modifier_rows(tests)
        selected_ids = [
            str(row["feature_id"])
            for row in ranking[: int(policy.modifier_top_n_per_inner_fold)]
        ]
        selected_set = set(selected_ids)
        rank_by_id = {
            str(row["feature_id"]): int(row["rank"])
            for row in ranking
        }
        for row in tests:
            feature_id = str(row["feature_id"])
            row["rank"] = rank_by_id.get(feature_id)
            row["selected_top_n"] = feature_id in selected_set
            p_value = row.get("interaction_p_value")
            if p_value is not None:
                modifier_p_values[feature_id].append(float(p_value))
        for feature_id in selected_ids:
            modifier_votes[feature_id] += 1
        modifier_folds.append(
            {
                "inner_fold": int(split.get("inner_fold", position)),
                "fit_rows": len(train_ids),
                "excluded_inner_heldout_rows": len(valid_ids),
                "model_family": (
                    "binomial_logistic_regression"
                    if binary_outcome
                    else "gaussian_linear_regression"
                ),
                "base_input_columns": [
                    base_names[index] for index in kept_base
                ],
                "nuisance_predictions_are_cross_fitted": True,
                "candidate_models": len(tests),
                "evaluable_candidates": len(ranking),
                "requested_top_n": int(policy.modifier_top_n_per_inner_fold),
                "selected_feature_ids": selected_ids,
                "selected_count": len(selected_ids),
                "interaction_p_value_ranking": ranking,
                "tests": tests,
            }
        )
        LOGGER.info(
            "Stage 2 modifier screen inner_fold=%s candidates=%s "
            "evaluable=%s selected=%s top_n=%s",
            int(split.get("inner_fold", position)),
            len(tests),
            len(ranking),
            len(selected_ids),
            int(policy.modifier_top_n_per_inner_fold),
        )

    modifier_union = _selected_in_any_inner_fold(modifier_votes)
    locked_modifiers = {
        _feature_key(feature)
        for feature in original
        if feature.get("configured_explicit_feature") is True
        and "effect_modifier" in set(map(str, feature.get("roles") or []))
    }
    modifier_union.update(locked_modifiers)

    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for feature in original:
        feature_id = _feature_key(feature)
        nuisance_roles = (
            ["treatment", "outcome"] if feature_id in confounder_union else []
        )
        roles: list[str] = []
        if nuisance_roles:
            roles.append("confounder")
        if feature_id in modifier_union:
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
                "modifier_min_interaction_p_value": (
                    float(min(modifier_p_values[feature_id]))
                    if modifier_p_values[feature_id]
                    else None
                ),
                "nuisance_model_roles": nuisance_roles,
                "roles": roles,
                "retained": retained,
                "selection_source": (
                    "investigator_locked"
                    if configured
                    else (
                        "group_elastic_net_nuisance_and_univariable_"
                        "interaction_any_inner_fold_union"
                    )
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
                else (
                    "group_elastic_net_nuisance_and_univariable_"
                    "interaction_any_inner_fold_union"
                )
            )
            selected.append(updated)

    report = {
        "schema_version": SCHEMA_VERSION,
        "temporal_scope": TEMPORAL_SCOPE,
        "status": "complete",
        "selection_method": (
            "any_inner_fold_union_group_elastic_net_nuisance_and_top_n_"
            "univariable_treatment_interactions"
        ),
        "penalized_nuisance_screen_model_family": "group_lasso_plus_ridge",
        # Backward-compatible report key; penalization applies to nuisance
        # screening only, not to the candidate-wise modifier regressions.
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
            "selection_rule": "selected_in_any_inner_fold_for_either_task",
            "required_votes": 1,
            "treatment_votes": dict(sorted(treatment_votes.items())),
            "outcome_votes": dict(sorted(outcome_votes.items())),
            "stable_treatment_feature_ids": sorted(any_treatment),
            "stable_outcome_feature_ids": sorted(any_outcome),
            "union_confounder_feature_ids": sorted(confounder_union),
            "intersection_is_not_a_selection_gate": True,
            "union_is_used_by_both_nuisance_models": True,
            "overall_treatment_auroc": _safe_auroc(
                np.concatenate(screen_t_observed),
                np.concatenate(screen_t_predicted),
            ),
            "overall_outcome_auroc": (
                _safe_auroc(
                    np.concatenate(screen_y_observed),
                    np.concatenate(screen_y_predicted),
                )
                if binary_outcome
                else None
            ),
        },
        "cross_fitted_nuisance_models": {
            "model_family": "random_forest",
            "treatment_feature_ids": sorted(confounder_union),
            "outcome_feature_ids": sorted(confounder_union),
            "folds": nuisance_folds,
            "overall_treatment_log_loss": float(
                log_loss(t_all, oof_e, labels=[0, 1])
            ),
            "overall_outcome_loss": _loss(y_all, oof_m, binary=binary_outcome),
            "overall_treatment_auroc": _safe_auroc(t_all, oof_e),
            "overall_outcome_auroc": (
                _safe_auroc(y_all, oof_m) if binary_outcome else None
            ),
            "propensity_min": float(np.min(oof_e)),
            "propensity_max": float(np.max(oof_e)),
            "predictions_are_inner_fold_out_of_fold": True,
        },
        "effect_modifier_screen": {
            "objective": (
                "candidate-wise outcome regression adjusted for cross-fitted "
                "treatment and outcome predictions, observed treatment, and the "
                "candidate main effect"
            ),
            "model_family": (
                "binomial_logistic_regression"
                if binary_outcome
                else "gaussian_linear_regression"
            ),
            "binary_outcome_formula": (
                "outcome ~ treatment_prediction + outcome_prediction + treatment + "
                "candidate + treatment:candidate"
            ),
            "categorical_interaction_test": (
                "one_joint_likelihood_ratio_test_over_all_estimable_"
                "treatment_by_nonreference_level_terms"
                if binary_outcome
                else (
                    "one_joint_partial_f_test_over_all_estimable_treatment_by_"
                    "nonreference_level_terms"
                )
            ),
            "candidate_scope": "one_candidate_group_per_model",
            "screening_rows": (
                "inner_fold_fit_rows_with_cross_fitted_nuisance_predictions"
            ),
            "folds": modifier_folds,
            "selection_rule": (
                "union_of_top_n_interaction_p_values_from_each_inner_fold"
            ),
            "top_n_per_inner_fold": int(policy.modifier_top_n_per_inner_fold),
            "required_votes": 1,
            "votes": dict(sorted(modifier_votes.items())),
            "p_values_are_raw": True,
            "p_value_threshold_is_not_a_selection_gate": True,
            "non_evaluable_candidates_are_not_ranked": True,
            "missingness_interactions": False,
            "stable_effect_modifier_feature_ids": sorted(modifier_union),
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
