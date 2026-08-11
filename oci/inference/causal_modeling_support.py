"""Small causal-modeling primitives shared by active Stage 1 lanes."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

from ..config import ExplicitFeatureForestConfig


def hstack_present(*matrices: Optional[np.ndarray]) -> Optional[np.ndarray]:
    present = [matrix for matrix in matrices if matrix is not None and matrix.shape[1] > 0]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return np.hstack(present)


def fit_predict_propensity(
    train_x: np.ndarray,
    train_t: np.ndarray,
    test_x: np.ndarray,
    cf_config: ExplicitFeatureForestConfig,
    random_state: int,
) -> np.ndarray:
    if len(np.unique(train_t)) < 2:
        return np.full(len(test_x), float(train_t[0]), dtype=np.float32)
    model = RandomForestClassifier(
        n_estimators=max(50, cf_config.n_estimators // 2),
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_x, train_t)
    return model.predict_proba(test_x)[:, 1]


def fit_predict_outcome(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    outcome_type: str,
    cf_config: ExplicitFeatureForestConfig,
    random_state: int,
) -> np.ndarray:
    if outcome_type == "continuous":
        model = RandomForestRegressor(
            n_estimators=max(50, cf_config.n_estimators // 2),
            max_depth=cf_config.max_depth,
            min_samples_leaf=cf_config.min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(train_x, train_y)
        return model.predict(test_x)
    if len(np.unique(train_y)) < 2:
        return np.full(len(test_x), float(train_y[0]), dtype=np.float32)
    model = RandomForestClassifier(
        n_estimators=max(50, cf_config.n_estimators // 2),
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def r_loss(
    y: np.ndarray,
    t: np.ndarray,
    outcome_pred: np.ndarray,
    propensity: np.ndarray,
    tau: np.ndarray,
) -> float:
    residual_y = np.asarray(y) - np.asarray(outcome_pred)
    residual_t = np.asarray(t) - np.asarray(propensity)
    return float(np.mean((residual_y - np.asarray(tau) * residual_t) ** 2))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


_hstack_present = hstack_present
_fit_predict_propensity = fit_predict_propensity
_fit_predict_outcome = fit_predict_outcome
_r_loss = r_loss
_safe_roc_auc = safe_roc_auc


__all__ = [
    "fit_predict_outcome",
    "fit_predict_propensity",
    "hstack_present",
    "r_loss",
    "safe_roc_auc",
]
