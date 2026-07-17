"""Regularized outcome-regression head for structured treatment effects.

The head is an S-learner with explicit treatment-by-covariate interactions.
Regularization is selected using only observed outcomes in cross-validation on
the fit sample.  Potential outcomes are evaluated by setting treatment to zero
and one after the final model is fit.

This intentionally contains no knowledge of clinical variable names or of a
synthetic data-generating process. It operates on a numeric feature matrix and
fits standardization separately inside every tuning split, then once on the
complete outer-fit sample. That keeps validation-fold covariate statistics out
of each inner fit while preserving a compact, auditable estimator boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass(frozen=True)
class InteractionTuningResult:
    """Observed-label tuning result recorded for auditability."""

    selected_regularization: float
    mean_validation_loss: Dict[float, float]
    validation_loss_by_fold: Dict[float, Tuple[float, ...]]
    n_splits: int
    selection_metric: str


class StructuredInteractionHead:
    """Fit a regularized main-effect plus treatment-interaction outcome model.

    Parameters
    ----------
    outcome_type:
        ``"binary"`` fits logistic outcome regressions; ``"continuous"`` fits
        ridge regressions.
    regularization_grid:
        Logistic ``C`` values for binary outcomes or ridge ``alpha`` values for
        continuous outcomes.  Selection minimizes inner held-out outcome loss.
    inner_folds:
        Number of fit-sample folds used to select regularization.
    interact_all_features:
        If true, every supplied feature may modify treatment.  If false,
        ``modifier_indices`` supplied to :meth:`fit` define interaction columns.
        Treating all pre-treatment candidates as possible modifiers is useful
        when role labels are themselves noisy discovery outputs.
    """

    def __init__(
        self,
        *,
        outcome_type: str = "binary",
        regularization_grid: Sequence[float] = (
            0.003,
            0.01,
            0.03,
            0.1,
            0.3,
            1.0,
            3.0,
            10.0,
        ),
        inner_folds: int = 3,
        interact_all_features: bool = True,
        random_state: int = 42,
        max_iter: int = 3000,
    ) -> None:
        outcome_type = str(outcome_type).strip().lower()
        if outcome_type not in {"binary", "continuous"}:
            raise ValueError("outcome_type must be 'binary' or 'continuous'")
        grid = tuple(float(value) for value in regularization_grid)
        if not grid or any(not np.isfinite(value) or value <= 0.0 for value in grid):
            raise ValueError("regularization_grid must contain positive finite values")
        if int(inner_folds) < 2:
            raise ValueError("inner_folds must be at least 2")
        if int(max_iter) < 1:
            raise ValueError("max_iter must be positive")
        self.outcome_type = outcome_type
        self.regularization_grid = tuple(dict.fromkeys(grid))
        self.inner_folds = int(inner_folds)
        self.interact_all_features = bool(interact_all_features)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)

        self.model_: Optional[Any] = None
        self.modifier_indices_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.feature_means_: Optional[np.ndarray] = None
        self.feature_scales_: Optional[np.ndarray] = None
        self.constant_outcome_: Optional[float] = None
        self.tuning_result_: Optional[InteractionTuningResult] = None

    @staticmethod
    def _validate_inputs(
        features: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.asarray(features, dtype=float)
        treatment = np.asarray(treatment, dtype=float).reshape(-1)
        outcome = np.asarray(outcome, dtype=float).reshape(-1)
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional numeric matrix")
        if len(features) != len(treatment) or len(features) != len(outcome):
            raise ValueError("features, treatment, and outcome must have equal row counts")
        if len(features) < 2:
            raise ValueError("at least two fit rows are required")
        if not np.all(np.isfinite(features)):
            raise ValueError("features contain non-finite values")
        if not np.all(np.isfinite(treatment)) or not np.all(np.isfinite(outcome)):
            raise ValueError("treatment and outcome must be finite")
        if not set(np.unique(treatment)).issubset({0.0, 1.0}):
            raise ValueError("treatment must be binary and encoded as 0/1")
        return features, treatment, outcome

    def _resolve_modifier_indices(
        self,
        n_features: int,
        modifier_indices: Optional[Iterable[int]],
    ) -> np.ndarray:
        if self.interact_all_features:
            return np.arange(n_features, dtype=int)
        if modifier_indices is None:
            raise ValueError(
                "modifier_indices are required when interact_all_features=False"
            )
        indices = np.asarray(list(modifier_indices), dtype=int)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("modifier_indices must contain at least one column index")
        if len(set(indices.tolist())) != len(indices):
            raise ValueError("modifier_indices contain duplicates")
        if int(indices.min()) < 0 or int(indices.max()) >= n_features:
            raise ValueError("modifier_indices are outside the feature matrix")
        return indices

    @staticmethod
    def _design(
        features: np.ndarray,
        treatment: np.ndarray,
        modifier_indices: np.ndarray,
    ) -> np.ndarray:
        treatment = np.asarray(treatment, dtype=float).reshape(-1, 1)
        interactions = features[:, modifier_indices] * treatment
        return np.column_stack((features, treatment, interactions))

    @staticmethod
    def _fit_feature_scaler(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit deterministic preprocessing on one estimation partition only."""

        means = np.mean(features, axis=0)
        scales = np.std(features, axis=0, ddof=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        return np.asarray(means, dtype=float), np.asarray(scales, dtype=float)

    @staticmethod
    def _scale_features(
        features: np.ndarray,
        means: np.ndarray,
        scales: np.ndarray,
    ) -> np.ndarray:
        return (features - means) / scales

    def _splits(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_splits = min(self.inner_folds, len(outcome))
        if n_splits < 2:
            raise ValueError("not enough rows for interaction-head tuning")
        if self.outcome_type == "binary":
            labels = np.asarray(
                [f"{int(t)}_{int(y)}" for t, y in zip(treatment, outcome)],
                dtype=object,
            )
            _, counts = np.unique(labels, return_counts=True)
            if len(counts) >= 2 and int(counts.min()) >= n_splits:
                splitter = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                return list(splitter.split(np.zeros(len(labels)), labels))
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        return list(splitter.split(np.zeros(len(outcome))))

    def _new_model(self, regularization: float) -> Any:
        if self.outcome_type == "binary":
            return LogisticRegression(
                C=float(regularization),
                max_iter=self.max_iter,
                solver="lbfgs",
                random_state=self.random_state,
            )
        return Ridge(alpha=float(regularization))

    def _validation_loss(
        self,
        model: Any,
        design: np.ndarray,
        outcome: np.ndarray,
    ) -> float:
        if self.outcome_type == "binary":
            probability = model.predict_proba(design)[:, 1]
            return float(log_loss(outcome, probability, labels=[0.0, 1.0]))
        return float(mean_squared_error(outcome, model.predict(design)))

    def fit(
        self,
        features: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        *,
        modifier_indices: Optional[Iterable[int]] = None,
    ) -> "StructuredInteractionHead":
        features, treatment, outcome = self._validate_inputs(
            features, treatment, outcome
        )
        if self.outcome_type == "binary" and not set(np.unique(outcome)).issubset(
            {0.0, 1.0}
        ):
            raise ValueError("binary outcomes must be encoded as 0/1")

        self.n_features_in_ = int(features.shape[1])
        self.modifier_indices_ = self._resolve_modifier_indices(
            self.n_features_in_, modifier_indices
        )
        self.feature_means_, self.feature_scales_ = self._fit_feature_scaler(features)
        unique_outcomes = np.unique(outcome)
        if len(unique_outcomes) == 1:
            self.constant_outcome_ = float(unique_outcomes[0])
            self.model_ = None
            metric = "log_loss" if self.outcome_type == "binary" else "mean_squared_error"
            self.tuning_result_ = InteractionTuningResult(
                selected_regularization=float(self.regularization_grid[0]),
                mean_validation_loss={
                    value: 0.0 for value in self.regularization_grid
                },
                validation_loss_by_fold={
                    value: tuple() for value in self.regularization_grid
                },
                n_splits=0,
                selection_metric=metric,
            )
            return self

        self.constant_outcome_ = None
        losses: Dict[float, List[float]] = {
            value: [] for value in self.regularization_grid
        }
        splits = self._splits(treatment, outcome)
        for fit_idx, validation_idx in splits:
            fit_y = outcome[fit_idx]
            inner_means, inner_scales = self._fit_feature_scaler(features[fit_idx])
            inner_fit_features = self._scale_features(
                features[fit_idx], inner_means, inner_scales
            )
            inner_validation_features = self._scale_features(
                features[validation_idx], inner_means, inner_scales
            )
            for regularization in self.regularization_grid:
                if self.outcome_type == "binary" and len(np.unique(fit_y)) < 2:
                    probability = np.full(
                        len(validation_idx), float(np.mean(fit_y)), dtype=float
                    )
                    loss = float(
                        log_loss(
                            outcome[validation_idx],
                            np.clip(probability, 1e-8, 1.0 - 1e-8),
                            labels=[0.0, 1.0],
                        )
                    )
                else:
                    model = self._new_model(regularization)
                    model.fit(
                        self._design(
                            inner_fit_features,
                            treatment[fit_idx],
                            self.modifier_indices_,
                        ),
                        fit_y,
                    )
                    loss = self._validation_loss(
                        model,
                        self._design(
                            inner_validation_features,
                            treatment[validation_idx],
                            self.modifier_indices_,
                        ),
                        outcome[validation_idx],
                    )
                losses[regularization].append(loss)

        mean_losses = {
            value: float(np.mean(fold_losses))
            for value, fold_losses in losses.items()
        }
        # The second key gives deterministic ties to the stronger penalty for
        # logistic regression (smaller C) and ridge regression (larger alpha).
        if self.outcome_type == "binary":
            selected = min(
                self.regularization_grid,
                key=lambda value: (mean_losses[value], value),
            )
        else:
            selected = min(
                self.regularization_grid,
                key=lambda value: (mean_losses[value], -value),
            )
        self.model_ = self._new_model(selected)
        final_features = self._scale_features(
            features,
            self.feature_means_,
            self.feature_scales_,
        )
        self.model_.fit(
            self._design(final_features, treatment, self.modifier_indices_), outcome
        )
        metric = "log_loss" if self.outcome_type == "binary" else "mean_squared_error"
        self.tuning_result_ = InteractionTuningResult(
            selected_regularization=float(selected),
            mean_validation_loss=mean_losses,
            validation_loss_by_fold={
                value: tuple(float(loss) for loss in fold_losses)
                for value, fold_losses in losses.items()
            },
            n_splits=len(splits),
            selection_metric=metric,
        )
        return self

    def _check_predict_features(self, features: np.ndarray) -> np.ndarray:
        if (
            self.tuning_result_ is None
            or self.modifier_indices_ is None
            or self.feature_means_ is None
            or self.feature_scales_ is None
        ):
            raise RuntimeError("StructuredInteractionHead must be fit before prediction")
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected a two-dimensional matrix with {self.n_features_in_} columns"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("features contain non-finite values")
        return self._scale_features(features, self.feature_means_, self.feature_scales_)

    def predict_potential_outcomes(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        features = self._check_predict_features(features)
        if self.constant_outcome_ is not None:
            constant = np.full(len(features), self.constant_outcome_, dtype=float)
            return constant.copy(), constant.copy()
        assert self.model_ is not None
        assert self.modifier_indices_ is not None
        treatment_zero = np.zeros(len(features), dtype=float)
        treatment_one = np.ones(len(features), dtype=float)
        design_zero = self._design(features, treatment_zero, self.modifier_indices_)
        design_one = self._design(features, treatment_one, self.modifier_indices_)
        if self.outcome_type == "binary":
            y0 = self.model_.predict_proba(design_zero)[:, 1]
            y1 = self.model_.predict_proba(design_one)[:, 1]
        else:
            y0 = self.model_.predict(design_zero)
            y1 = self.model_.predict(design_one)
        return np.asarray(y0, dtype=float), np.asarray(y1, dtype=float)

    def predict_effect(self, features: np.ndarray) -> np.ndarray:
        y0, y1 = self.predict_potential_outcomes(features)
        return y1 - y0

    def predict_observed_outcome(
        self, features: np.ndarray, treatment: np.ndarray
    ) -> np.ndarray:
        features = self._check_predict_features(features)
        treatment = np.asarray(treatment, dtype=float).reshape(-1)
        if len(treatment) != len(features):
            raise ValueError("treatment and features must have equal row counts")
        if self.constant_outcome_ is not None:
            return np.full(len(features), self.constant_outcome_, dtype=float)
        assert self.model_ is not None
        assert self.modifier_indices_ is not None
        design = self._design(features, treatment, self.modifier_indices_)
        if self.outcome_type == "binary":
            return np.asarray(self.model_.predict_proba(design)[:, 1], dtype=float)
        return np.asarray(self.model_.predict(design), dtype=float)
