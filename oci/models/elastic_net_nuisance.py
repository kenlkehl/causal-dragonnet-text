"""Clone-safe elastic-net estimators for causal nuisance functions.

The causal workflows repeatedly fit nuisance models inside an outer/inner
cross-fitting scheme.  These wrappers keep the model family and tuning rule
identical in direct Stage 2 fits and in estimators cloned by EconML.  They also
handle the small-fold edge cases where an internal regularization CV would not
have enough observations from both classes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonzero_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result == 0:
        raise ValueError(f"{name} must be nonzero")
    return result


class ElasticNetLogisticClassifier(ClassifierMixin, BaseEstimator):
    """Logistic elastic net with deterministic, fold-adaptive internal CV."""

    def __init__(
        self,
        *,
        l1_ratio: float = 0.8,
        cv_folds: int = 3,
        regularization_grid_size: int = 16,
        minimum_log10_c: float = -2.0,
        maximum_log10_c: float = 4.0,
        max_iter: int = 5_000,
        tolerance: float = 1e-4,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.l1_ratio = l1_ratio
        self.cv_folds = cv_folds
        self.regularization_grid_size = regularization_grid_size
        self.minimum_log10_c = minimum_log10_c
        self.maximum_log10_c = maximum_log10_c
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _validated_parameters(self) -> tuple[float, int, np.ndarray, int, float, int, int]:
        ratio = _finite_float(self.l1_ratio, name="l1_ratio")
        if not 0.0 < ratio <= 1.0:
            raise ValueError("l1_ratio must be in (0, 1]")
        folds = _positive_int(self.cv_folds, name="cv_folds", minimum=2)
        grid_size = _positive_int(
            self.regularization_grid_size,
            name="regularization_grid_size",
            minimum=3,
        )
        minimum = _finite_float(self.minimum_log10_c, name="minimum_log10_c")
        maximum = _finite_float(self.maximum_log10_c, name="maximum_log10_c")
        if maximum <= minimum:
            raise ValueError("maximum_log10_c must exceed minimum_log10_c")
        max_iter = _positive_int(self.max_iter, name="max_iter")
        tolerance = _finite_float(self.tolerance, name="tolerance")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        seed = _positive_int(self.random_state, name="random_state", minimum=0)
        jobs = _nonzero_int(self.n_jobs, name="n_jobs")
        return (
            ratio,
            folds,
            np.logspace(minimum, maximum, grid_size),
            max_iter,
            tolerance,
            seed,
            jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetLogisticClassifier":
        (
            ratio,
            requested_folds,
            grid,
            max_iter,
            tolerance,
            seed,
            jobs,
        ) = self._validated_parameters()
        design, target = check_X_y(
            X,
            y,
            accept_sparse=True,
            dtype=float,
            ensure_min_features=0,
        )
        target = np.asarray(target).reshape(-1)
        values, counts = np.unique(target, return_counts=True)
        if not set(values.tolist()).issubset({0, 1}):
            raise ValueError("elastic-net logistic target must contain only 0 and 1")
        self.n_features_in_ = int(design.shape[1])
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.constant_probability_ = None
        self.model_ = None
        self.fit_mode_ = "constant"
        self.effective_cv_folds_ = 0
        if len(values) < 2 or design.shape[1] == 0:
            self.constant_probability_ = float(np.clip(np.mean(target), 1e-6, 1 - 1e-6))
            return self

        folds = min(requested_folds, int(counts.min()))
        common = {
            "penalty": "elasticnet",
            "solver": "saga",
            "fit_intercept": True,
            "max_iter": max_iter,
            "tol": tolerance,
            "random_state": seed,
            "n_jobs": jobs,
        }
        if folds >= 2:
            self.fit_mode_ = "cross_validated"
            self.effective_cv_folds_ = int(folds)
            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=seed,
            )
            self.model_ = LogisticRegressionCV(
                Cs=grid,
                cv=splitter,
                scoring="neg_log_loss",
                refit=True,
                l1_ratios=[ratio],
                **common,
            )
        else:
            self.fit_mode_ = "fixed_grid_midpoint"
            self.model_ = LogisticRegression(
                C=float(grid[len(grid) // 2]),
                l1_ratio=ratio,
                **common,
            )
        self.model_.fit(design, target.astype(int))
        self.classes_ = np.asarray(self.model_.classes_, dtype=int)
        return self

    def fit_audit(self) -> dict[str, Any]:
        """Return JSON-safe details from the fitted nuisance clone."""

        check_is_fitted(
            self,
            ("classes_", "model_", "fit_mode_", "effective_cv_folds_"),
        )
        selected_regularization = None
        iteration_values = np.asarray([], dtype=int)
        if self.model_ is not None:
            selected_c = (
                np.asarray(self.model_.C_, dtype=float).reshape(-1)[0]
                if hasattr(self.model_, "C_")
                else float(self.model_.C)
            )
            selected_regularization = {
                "parameter": "C",
                "value": float(selected_c),
            }
            iteration_values = np.asarray(
                getattr(self.model_, "n_iter_", []), dtype=int
            ).reshape(-1)
        maximum_iterations = (
            int(iteration_values.max()) if iteration_values.size else 0
        )
        return {
            "estimator": (
                "oci.models.elastic_net_nuisance."
                "ElasticNetLogisticClassifier"
            ),
            "fit_mode": str(self.fit_mode_),
            "n_features": int(self.n_features_in_),
            "requested_cv_folds": int(self.cv_folds),
            "effective_cv_folds": int(self.effective_cv_folds_),
            "selected_regularization": selected_regularization,
            "constant_prediction": (
                float(self.constant_probability_)
                if self.constant_probability_ is not None
                else None
            ),
            "optimization": {
                "configured_max_iter": int(self.max_iter),
                "maximum_iterations_observed": maximum_iterations,
                "iteration_limit_reached": bool(
                    iteration_values.size
                    and maximum_iterations >= int(self.max_iter)
                ),
            },
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("classes_", "model_"))
        design = check_array(
            X,
            accept_sparse=True,
            dtype=float,
            ensure_min_features=0,
        )
        if self.model_ is not None:
            return np.asarray(self.model_.predict_proba(design), dtype=float)
        probability = np.full(
            len(design),
            float(self.constant_probability_),
            dtype=float,
        )
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ElasticNetRegressor(RegressorMixin, BaseEstimator):
    """Squared-error elastic net with deterministic, adaptive internal CV."""

    def __init__(
        self,
        *,
        l1_ratio: float = 0.8,
        cv_folds: int = 3,
        regularization_grid_size: int = 16,
        minimum_log10_alpha: float = -5.0,
        maximum_log10_alpha: float = -1.0,
        max_iter: int = 5_000,
        tolerance: float = 1e-4,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.l1_ratio = l1_ratio
        self.cv_folds = cv_folds
        self.regularization_grid_size = regularization_grid_size
        self.minimum_log10_alpha = minimum_log10_alpha
        self.maximum_log10_alpha = maximum_log10_alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetRegressor":
        ratio = _finite_float(self.l1_ratio, name="l1_ratio")
        if not 0.0 < ratio <= 1.0:
            raise ValueError("l1_ratio must be in (0, 1]")
        requested_folds = _positive_int(self.cv_folds, name="cv_folds", minimum=2)
        grid_size = _positive_int(
            self.regularization_grid_size,
            name="regularization_grid_size",
            minimum=3,
        )
        minimum = _finite_float(
            self.minimum_log10_alpha,
            name="minimum_log10_alpha",
        )
        maximum = _finite_float(
            self.maximum_log10_alpha,
            name="maximum_log10_alpha",
        )
        if maximum <= minimum:
            raise ValueError("maximum_log10_alpha must exceed minimum_log10_alpha")
        max_iter = _positive_int(self.max_iter, name="max_iter")
        tolerance = _finite_float(self.tolerance, name="tolerance")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        seed = _positive_int(self.random_state, name="random_state", minimum=0)
        jobs = _nonzero_int(self.n_jobs, name="n_jobs")

        design, target = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=float,
            ensure_min_features=0,
        )
        target = np.asarray(target, dtype=float).reshape(-1)
        self.n_features_in_ = int(design.shape[1])
        self.constant_mean_ = None
        self.model_ = None
        self.fit_mode_ = "constant"
        self.effective_cv_folds_ = 0
        if design.shape[1] == 0 or len(target) < 3 or float(np.var(target)) <= 1e-15:
            self.constant_mean_ = float(np.mean(target))
            return self

        folds = min(requested_folds, len(target))
        self.fit_mode_ = "cross_validated"
        self.effective_cv_folds_ = int(folds)
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        self.model_ = ElasticNetCV(
            l1_ratio=ratio,
            alphas=np.logspace(minimum, maximum, grid_size),
            fit_intercept=True,
            max_iter=max_iter,
            tol=tolerance,
            cv=splitter,
            n_jobs=jobs,
            random_state=seed,
            selection="cyclic",
        )
        self.model_.fit(design, target)
        return self

    def fit_audit(self) -> dict[str, Any]:
        """Return JSON-safe details from the fitted nuisance clone."""

        check_is_fitted(
            self,
            ("model_", "fit_mode_", "effective_cv_folds_"),
        )
        selected_regularization = None
        iterations = 0
        duality_gap = None
        if self.model_ is not None:
            selected_regularization = {
                "parameter": "alpha",
                "value": float(self.model_.alpha_),
            }
            iterations = int(np.asarray(self.model_.n_iter_).reshape(-1).max())
            duality_gap = float(np.asarray(self.model_.dual_gap_).reshape(-1).max())
        return {
            "estimator": "oci.models.elastic_net_nuisance.ElasticNetRegressor",
            "fit_mode": str(self.fit_mode_),
            "n_features": int(self.n_features_in_),
            "requested_cv_folds": int(self.cv_folds),
            "effective_cv_folds": int(self.effective_cv_folds_),
            "selected_regularization": selected_regularization,
            "constant_prediction": (
                float(self.constant_mean_)
                if self.constant_mean_ is not None
                else None
            ),
            "optimization": {
                "configured_max_iter": int(self.max_iter),
                "maximum_iterations_observed": iterations,
                "iteration_limit_reached": bool(iterations >= int(self.max_iter)),
                "duality_gap": duality_gap,
            },
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        design = check_array(
            X,
            accept_sparse=False,
            dtype=float,
            ensure_min_features=0,
        )
        if self.model_ is not None:
            return np.asarray(self.model_.predict(design), dtype=float)
        return np.full(len(design), float(self.constant_mean_), dtype=float)


__all__ = ["ElasticNetLogisticClassifier", "ElasticNetRegressor"]
