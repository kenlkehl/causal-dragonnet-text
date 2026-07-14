"""Honest TF-IDF nuisance modeling and three-bank topic discovery.

This module is intentionally independent of the legacy HTR, embedding, uplift,
R-learner, and raw-text causal-forest implementations.  A fitted context owns
all vectorizers, nuisance stacks, topic models, and vocabulary state needed to
transform an external held-out row set without refitting.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
from sklearn.decomposition import NMF
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from ..config import BoWViewConfig, TfidfTopicDiscoveryConfig
from .tfidf_topic_score_selection import (
    TOPIC_SCORE_TEST_SCHEMA_VERSION,
    score_topic_banks,
)

logger = logging.getLogger(__name__)

HANDOFF_SCHEMA_VERSION = "multi_model_forest_handoff_v2"
DISCOVERY_SCHEMA_VERSION = "tfidf_topic_discovery_v2"


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_set_fingerprint(row_ids: Sequence[Any]) -> str:
    """Fingerprint a row set without making its input order meaningful."""
    values = sorted(str(value) for value in row_ids)
    return stable_hash(values)


def _normalize_texts(values: Sequence[Any]) -> List[str]:
    return [str(value or "").lower() for value in values]


def _bounded_folds(requested: int, labels: np.ndarray, *, stratified: bool) -> int:
    n_rows = int(len(labels))
    if n_rows < 4:
        raise ValueError("At least four rows are required for nested nuisance cross-fitting")
    upper = n_rows
    if stratified:
        counts = np.unique(labels.astype(int), return_counts=True)[1]
        if len(counts) < 2:
            return 2
        upper = int(np.min(counts))
    return max(2, min(int(requested), upper, n_rows // 2))


def _splitter(labels: np.ndarray, folds: int, *, stratified: bool, seed: int):
    counts = np.unique(labels.astype(int), return_counts=True)[1] if stratified else []
    if stratified and len(counts) >= 2 and int(np.min(counts)) >= int(folds):
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(
            np.zeros(len(labels)), labels.astype(int)
        )
    return KFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(labels)))


def _vectorizer(view: BoWViewConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)[a-z0-9%<>+=-]+",
        ngram_range=(int(view.ngram_range_min), int(view.ngram_range_max)),
        min_df=int(view.min_df),
        max_df=float(view.max_df),
        sublinear_tf=bool(view.sublinear_tf),
        max_features=int(view.max_features),
        dtype=np.float32,
    )


def _classifier(view: BoWViewConfig, seed: int):
    if view.bow_model == "linear":
        return LogisticRegression(
            C=float(view.logistic_c),
            solver="liblinear",
            max_iter=int(view.logistic_max_iter),
            random_state=seed,
        )
    cls = ExtraTreesClassifier if view.bow_model == "extratrees" else RandomForestClassifier
    return cls(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )


def _regressor(view: BoWViewConfig, seed: int):
    if view.bow_model == "linear":
        return Ridge(alpha=float(view.ridge_alpha))
    cls = ExtraTreesRegressor if view.bow_model == "extratrees" else RandomForestRegressor
    return cls(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )


def _constant_prediction(values: np.ndarray, n_rows: int) -> np.ndarray:
    mean = float(np.mean(values)) if len(values) else 0.0
    return np.full(int(n_rows), mean, dtype=float)


def _fit_base_predict(
    *,
    fit_texts: Sequence[str],
    fit_values: np.ndarray,
    predict_texts: Sequence[str],
    view: BoWViewConfig,
    binary: bool,
    seed: int,
) -> Tuple[np.ndarray, Optional[TfidfVectorizer], Optional[Any]]:
    if binary and len(np.unique(fit_values.astype(int))) < 2:
        return _constant_prediction(fit_values, len(predict_texts)), None, None
    vectorizer = _vectorizer(view)
    try:
        x_fit = vectorizer.fit_transform(fit_texts)
    except ValueError:
        return _constant_prediction(fit_values, len(predict_texts)), None, None
    model = _classifier(view, seed) if binary else _regressor(view, seed)
    model.fit(x_fit, fit_values.astype(int) if binary else fit_values)
    if len(predict_texts) == 0:
        return np.zeros(0, dtype=float), vectorizer, model
    x_predict = vectorizer.transform(predict_texts)
    if binary:
        predictions = model.predict_proba(x_predict)[:, 1]
    else:
        predictions = model.predict(x_predict)
    return np.asarray(predictions, dtype=float), vectorizer, model


@dataclass
class CrossFittedStack:
    """Fitted complete-context bases and a stack trained on context OOF scores."""

    views: List[BoWViewConfig]
    binary: bool
    base_models: List[Tuple[Optional[TfidfVectorizer], Optional[Any], float]]
    stack_model: Optional[Any]
    stack_constant: float
    config_hash: str

    def predict(self, texts: Sequence[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        texts = _normalize_texts(texts)
        columns: List[np.ndarray] = []
        by_view: Dict[str, np.ndarray] = {}
        for view, (vectorizer, model, constant) in zip(self.views, self.base_models):
            if vectorizer is None or model is None:
                prediction = np.full(len(texts), constant, dtype=float)
            else:
                matrix = vectorizer.transform(texts)
                prediction = (
                    model.predict_proba(matrix)[:, 1]
                    if self.binary
                    else model.predict(matrix)
                )
            prediction = np.asarray(prediction, dtype=float)
            columns.append(prediction)
            by_view[view.name] = prediction
        meta = np.column_stack(columns)
        if self.stack_model is None:
            stacked = np.full(len(texts), self.stack_constant, dtype=float)
        elif self.binary:
            stacked = self.stack_model.predict_proba(meta)[:, 1]
        else:
            stacked = self.stack_model.predict(meta)
        return np.asarray(stacked, dtype=float), by_view


def _fit_stack(meta: np.ndarray, values: np.ndarray, *, binary: bool, seed: int):
    constant = float(np.mean(values))
    if meta.shape[1] == 0 or (binary and len(np.unique(values.astype(int))) < 2):
        return None, constant
    if binary:
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
        model.fit(meta, values.astype(int))
    else:
        model = Ridge(alpha=1.0)
        model.fit(meta, values)
    return model, constant


def fit_cross_fitted_nuisance_stack(
    *,
    texts: Sequence[str],
    values: Sequence[float],
    views: Sequence[BoWViewConfig],
    folds: int,
    binary: bool,
    random_state: int,
) -> Dict[str, Any]:
    """Fit a two-level honest OOF stack within one analysis context.

    Each top-level held-out nuisance fold receives base predictions from models
    fitted without that fold.  The stack for that fold is itself trained on
    sub-inner OOF predictions from only the top-level fitting portion.
    """
    texts = _normalize_texts(texts)
    values = np.asarray(values, dtype=float)
    views = list(views)
    n_rows = len(values)
    fold_count = _bounded_folds(folds, values, stratified=binary)
    base_oof = np.full((n_rows, len(views)), np.nan, dtype=float)
    stack_oof = np.full(n_rows, np.nan, dtype=float)
    fold_ids = np.full(n_rows, -1, dtype=int)
    fit_ids_by_row: List[List[int]] = [[] for _ in range(n_rows)]

    for fold, (fit_pos, heldout_pos) in enumerate(
        _splitter(values, fold_count, stratified=binary, seed=random_state), start=1
    ):
        fit_pos = np.asarray(fit_pos, dtype=int)
        heldout_pos = np.asarray(heldout_pos, dtype=int)
        sub_values = values[fit_pos]
        sub_folds = _bounded_folds(
            min(fold_count, len(fit_pos) // 2), sub_values, stratified=binary
        )
        meta_fit = np.full((len(fit_pos), len(views)), np.nan, dtype=float)
        for view_index, view in enumerate(views):
            for sub_fold, (sub_fit_local, sub_hold_local) in enumerate(
                _splitter(
                    sub_values,
                    sub_folds,
                    stratified=binary,
                    seed=random_state + 1000 * fold + view_index,
                ),
                start=1,
            ):
                sub_fit_local = np.asarray(sub_fit_local, dtype=int)
                sub_hold_local = np.asarray(sub_hold_local, dtype=int)
                prediction, _, _ = _fit_base_predict(
                    fit_texts=[texts[fit_pos[index]] for index in sub_fit_local],
                    fit_values=sub_values[sub_fit_local],
                    predict_texts=[texts[fit_pos[index]] for index in sub_hold_local],
                    view=view,
                    binary=binary,
                    seed=random_state + 10_000 * fold + 100 * view_index + sub_fold,
                )
                meta_fit[sub_hold_local, view_index] = prediction
            held_prediction, _, _ = _fit_base_predict(
                fit_texts=[texts[index] for index in fit_pos],
                fit_values=values[fit_pos],
                predict_texts=[texts[index] for index in heldout_pos],
                view=view,
                binary=binary,
                seed=random_state + 20_000 * fold + view_index,
            )
            base_oof[heldout_pos, view_index] = held_prediction
        if not np.isfinite(meta_fit).all():
            raise RuntimeError("Nested nuisance meta-features are incomplete")
        stack_model, stack_constant = _fit_stack(
            meta_fit, sub_values, binary=binary, seed=random_state + 30_000 + fold
        )
        held_meta = base_oof[heldout_pos]
        if stack_model is None:
            stack_oof[heldout_pos] = stack_constant
        elif binary:
            stack_oof[heldout_pos] = stack_model.predict_proba(held_meta)[:, 1]
        else:
            stack_oof[heldout_pos] = stack_model.predict(held_meta)
        fold_ids[heldout_pos] = fold
        fit_list = fit_pos.astype(int).tolist()
        for row in heldout_pos:
            fit_ids_by_row[int(row)] = fit_list

    if not np.isfinite(base_oof).all() or not np.isfinite(stack_oof).all():
        raise RuntimeError("Cross-fitted nuisance predictions are incomplete")

    complete_stack, complete_constant = _fit_stack(
        base_oof, values, binary=binary, seed=random_state + 40_000
    )
    complete_bases: List[Tuple[Optional[TfidfVectorizer], Optional[Any], float]] = []
    for view_index, view in enumerate(views):
        _, vectorizer, model = _fit_base_predict(
            fit_texts=texts,
            fit_values=values,
            predict_texts=[],
            view=view,
            binary=binary,
            seed=random_state + 50_000 + view_index,
        )
        complete_bases.append((vectorizer, model, float(np.mean(values))))
    fitted = CrossFittedStack(
        views=views,
        binary=binary,
        base_models=complete_bases,
        stack_model=complete_stack,
        stack_constant=complete_constant,
        config_hash=stable_hash(
            {
                "views": [asdict(view) for view in views],
                "folds": fold_count,
                "binary": binary,
                "random_state": random_state,
            }
        ),
    )
    return {
        "base_oof": base_oof,
        "stacked_oof": stack_oof,
        "fold_ids": fold_ids,
        "fit_positions_by_row": fit_ids_by_row,
        "fitted": fitted,
        "metrics": nuisance_metrics(values, stack_oof, binary=binary),
        "view_metrics": {
            view.name: nuisance_metrics(values, base_oof[:, index], binary=binary)
            for index, view in enumerate(views)
        },
    }


def _vectorizer_key(view: BoWViewConfig) -> Tuple[Any, ...]:
    return (
        int(view.max_features),
        int(view.min_df),
        float(view.max_df),
        int(view.ngram_range_min),
        int(view.ngram_range_max),
        bool(view.sublinear_tf),
    )


def _matrix_model_prediction(
    x_fit: Any,
    x_predict: Any,
    values: np.ndarray,
    view: BoWViewConfig,
    *,
    binary: bool,
    seed: int,
) -> Tuple[np.ndarray, Optional[Any]]:
    if binary and len(np.unique(values.astype(int))) < 2:
        return _constant_prediction(values, x_predict.shape[0]), None
    model = _classifier(view, seed) if binary else _regressor(view, seed)
    model.fit(x_fit, values.astype(int) if binary else values)
    if x_predict.shape[0] == 0:
        return np.zeros(0, dtype=float), model
    prediction = (
        model.predict_proba(x_predict)[:, 1] if binary else model.predict(x_predict)
    )
    return np.asarray(prediction, dtype=float), model


def fit_joint_cross_fitted_nuisance_stacks(
    *,
    texts: Sequence[str],
    treatment: Sequence[float],
    outcome: Sequence[float],
    outcome_binary: bool,
    strata: Sequence[int],
    views: Sequence[BoWViewConfig],
    folds: int,
    random_state: int,
) -> Dict[str, Dict[str, Any]]:
    """Fit treatment/outcome stacks while sharing every label-free TF-IDF fit."""
    texts = _normalize_texts(texts)
    targets = {
        "treatment": (np.asarray(treatment, dtype=float), True),
        "outcome": (np.asarray(outcome, dtype=float), bool(outcome_binary)),
    }
    split_labels = np.asarray(strata, dtype=int)
    views = list(views)
    n_rows = len(texts)
    fold_count = _bounded_folds(folds, split_labels, stratified=True)
    base_oof = {
        target: np.full((n_rows, len(views)), np.nan, dtype=float) for target in targets
    }
    stack_oof = {target: np.full(n_rows, np.nan, dtype=float) for target in targets}
    fold_ids = np.full(n_rows, -1, dtype=int)
    fit_positions_by_row: List[List[int]] = [[] for _ in range(n_rows)]
    grouped: Dict[Tuple[Any, ...], List[Tuple[int, BoWViewConfig]]] = {}
    for index, view in enumerate(views):
        grouped.setdefault(_vectorizer_key(view), []).append((index, view))

    for fold, (fit_pos, heldout_pos) in enumerate(
        _splitter(split_labels, fold_count, stratified=True, seed=random_state), start=1
    ):
        fit_pos = np.asarray(fit_pos, dtype=int)
        heldout_pos = np.asarray(heldout_pos, dtype=int)
        fit_strata = split_labels[fit_pos]
        sub_folds = _bounded_folds(
            min(fold_count, len(fit_pos) // 2), fit_strata, stratified=True
        )
        meta_fit = {
            target: np.full((len(fit_pos), len(views)), np.nan, dtype=float)
            for target in targets
        }
        for sub_fold, (sub_fit_local, sub_hold_local) in enumerate(
            _splitter(
                fit_strata,
                sub_folds,
                stratified=True,
                seed=random_state + 1000 * fold,
            ),
            start=1,
        ):
            sub_fit_local = np.asarray(sub_fit_local, dtype=int)
            sub_hold_local = np.asarray(sub_hold_local, dtype=int)
            sub_fit_pos = fit_pos[sub_fit_local]
            sub_hold_pos = fit_pos[sub_hold_local]
            for group_index, group_views in enumerate(grouped.values()):
                vectorizer = _vectorizer(group_views[0][1])
                try:
                    x_sub_fit = vectorizer.fit_transform([texts[index] for index in sub_fit_pos])
                    x_sub_hold = vectorizer.transform([texts[index] for index in sub_hold_pos])
                except ValueError:
                    x_sub_fit = x_sub_hold = None
                for view_index, view in group_views:
                    for target_index, (target, (values, binary)) in enumerate(targets.items()):
                        if x_sub_fit is None:
                            prediction = _constant_prediction(
                                values[sub_fit_pos], len(sub_hold_pos)
                            )
                        else:
                            prediction, _ = _matrix_model_prediction(
                                x_sub_fit,
                                x_sub_hold,
                                values[sub_fit_pos],
                                view,
                                binary=binary,
                                seed=(
                                    random_state
                                    + 10_000 * fold
                                    + 1000 * sub_fold
                                    + 100 * group_index
                                    + 10 * view_index
                                    + target_index
                                ),
                            )
                        meta_fit[target][sub_hold_local, view_index] = prediction

        for group_index, group_views in enumerate(grouped.values()):
            vectorizer = _vectorizer(group_views[0][1])
            try:
                x_fit = vectorizer.fit_transform([texts[index] for index in fit_pos])
                x_heldout = vectorizer.transform([texts[index] for index in heldout_pos])
            except ValueError:
                x_fit = x_heldout = None
            for view_index, view in group_views:
                for target_index, (target, (values, binary)) in enumerate(targets.items()):
                    if x_fit is None:
                        prediction = _constant_prediction(values[fit_pos], len(heldout_pos))
                    else:
                        prediction, _ = _matrix_model_prediction(
                            x_fit,
                            x_heldout,
                            values[fit_pos],
                            view,
                            binary=binary,
                            seed=(
                                random_state
                                + 30_000 * fold
                                + 100 * group_index
                                + 10 * view_index
                                + target_index
                            ),
                        )
                    base_oof[target][heldout_pos, view_index] = prediction

        for target_index, (target, (values, binary)) in enumerate(targets.items()):
            if not np.isfinite(meta_fit[target]).all():
                raise RuntimeError(f"Nested {target} nuisance meta-features are incomplete")
            stack_model, stack_constant = _fit_stack(
                meta_fit[target],
                values[fit_pos],
                binary=binary,
                seed=random_state + 40_000 + 10 * fold + target_index,
            )
            held_meta = base_oof[target][heldout_pos]
            if stack_model is None:
                stack_oof[target][heldout_pos] = stack_constant
            elif binary:
                stack_oof[target][heldout_pos] = stack_model.predict_proba(held_meta)[:, 1]
            else:
                stack_oof[target][heldout_pos] = stack_model.predict(held_meta)
        fold_ids[heldout_pos] = fold
        fit_list = fit_pos.astype(int).tolist()
        for position in heldout_pos:
            fit_positions_by_row[int(position)] = fit_list

    fitted_bases: Dict[str, List[Tuple[Optional[TfidfVectorizer], Optional[Any], float]]] = {
        target: [(None, None, float(np.mean(values))) for _ in views]
        for target, (values, _binary) in targets.items()
    }
    for group_index, group_views in enumerate(grouped.values()):
        vectorizer = _vectorizer(group_views[0][1])
        try:
            x_full = vectorizer.fit_transform(texts)
        except ValueError:
            vectorizer = None
            x_full = None
        for view_index, view in group_views:
            for target_index, (target, (values, binary)) in enumerate(targets.items()):
                model = None
                if x_full is not None and not (binary and len(np.unique(values.astype(int))) < 2):
                    _, model = _matrix_model_prediction(
                        x_full,
                        x_full[:0],
                        values,
                        view,
                        binary=binary,
                        seed=random_state + 50_000 + 100 * group_index + 10 * view_index + target_index,
                    )
                fitted_bases[target][view_index] = (
                    vectorizer,
                    model,
                    float(np.mean(values)),
                )

    output: Dict[str, Dict[str, Any]] = {}
    for target_index, (target, (values, binary)) in enumerate(targets.items()):
        if not np.isfinite(base_oof[target]).all() or not np.isfinite(stack_oof[target]).all():
            raise RuntimeError(f"Cross-fitted {target} nuisance predictions are incomplete")
        complete_stack, complete_constant = _fit_stack(
            base_oof[target],
            values,
            binary=binary,
            seed=random_state + 60_000 + target_index,
        )
        fitted = CrossFittedStack(
            views=views,
            binary=binary,
            base_models=fitted_bases[target],
            stack_model=complete_stack,
            stack_constant=complete_constant,
            config_hash=stable_hash(
                {
                    "views": [asdict(view) for view in views],
                    "folds": fold_count,
                    "target": target,
                    "joint_label_free_vectorization": True,
                    "random_state": random_state,
                }
            ),
        )
        output[target] = {
            "base_oof": base_oof[target],
            "stacked_oof": stack_oof[target],
            "fold_ids": fold_ids.copy(),
            "fit_positions_by_row": [list(values) for values in fit_positions_by_row],
            "fitted": fitted,
            "metrics": nuisance_metrics(values, stack_oof[target], binary=binary),
            "view_metrics": {
                view.name: nuisance_metrics(values, base_oof[target][:, index], binary=binary)
                for index, view in enumerate(views)
            },
        }
    return output


def calibration_intercept_slope(labels: np.ndarray, probabilities: np.ndarray) -> Tuple[Any, Any]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(np.unique(labels)) < 2:
        return None, None
    logits = np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    try:
        model.fit(logits, labels)
    except ValueError:
        return None, None
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, n_bins: int = 10
) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bins = np.minimum(np.digitize(probabilities, edges[1:-1]), n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        mask = bins == index
        if np.any(mask):
            error += float(np.mean(mask)) * abs(
                float(np.mean(labels[mask])) - float(np.mean(probabilities[mask]))
            )
    return float(error)


def nuisance_metrics(values: np.ndarray, predictions: np.ndarray, *, binary: bool) -> Dict[str, Any]:
    values = np.asarray(values, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if not binary:
        return {
            "rmse": float(np.sqrt(mean_squared_error(values, predictions))),
            "mae": float(mean_absolute_error(values, predictions)),
        }
    probabilities = np.clip(predictions, 1e-6, 1.0 - 1e-6)
    try:
        auroc = float(roc_auc_score(values.astype(int), probabilities))
    except ValueError:
        auroc = None
    intercept, slope = calibration_intercept_slope(values, probabilities)
    return {
        "auroc": auroc,
        "brier": float(brier_score_loss(values.astype(int), probabilities)),
        "log_loss": float(log_loss(values.astype(int), probabilities, labels=[0, 1])),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece": expected_calibration_error(values, probabilities),
    }


def cohort_contrast_scores(
    matrix: Any,
    feature_names: Sequence[str],
    treatment: Sequence[float],
    outcome: Sequence[float],
    propensity_prediction: Sequence[float],
    outcome_prediction: Sequence[float],
) -> pd.DataFrame:
    """Calculate signed orthogonal cohort moments and robust standard errors."""
    x = sparse.csr_matrix(matrix, dtype=float)
    names = np.asarray(feature_names, dtype=object)
    t = np.asarray(treatment, dtype=float)
    y = np.asarray(outcome, dtype=float)
    e = np.asarray(propensity_prediction, dtype=float)
    m = np.asarray(outcome_prediction, dtype=float)
    if x.shape != (len(t), len(names)):
        raise ValueError("matrix shape must match rows and feature_names")
    u = t - e
    v = y - m
    denominator = float(np.dot(u, u))
    constant_effect = 0.0 if denominator <= 0.0 else float(np.dot(u, v) / denominator)
    contribution = u * (v - constant_effect * u)
    n_rows = max(1, len(t))
    moments = np.asarray(x.T @ contribution).ravel() / n_rows
    squared_sum = np.asarray(x.power(2).T @ np.square(contribution)).ravel()
    variances = np.maximum(0.0, squared_sum / n_rows - np.square(moments))
    robust_se = np.sqrt(variances / n_rows)
    scores = np.divide(
        moments,
        robust_se,
        out=np.zeros_like(moments),
        where=robust_se > 0.0,
    )
    presence = x.copy()
    presence.data = np.ones_like(presence.data)
    support_control = np.asarray(presence[t < 0.5].sum(axis=0)).ravel().astype(int)
    support_treated = np.asarray(presence[t >= 0.5].sum(axis=0)).ravel().astype(int)
    return pd.DataFrame(
        {
            "feature": names.astype(str),
            "moment": moments,
            "robust_se": robust_se,
            "signed_score": scores,
            "unsigned_score": np.abs(scores),
            "support_control": support_control,
            "support_treated": support_treated,
            "constant_residual_effect": constant_effect,
        }
    )


def unsigned_linear_screen(
    matrix: Any,
    feature_names: Sequence[str],
    values: Sequence[float],
    *,
    binary: bool,
    logistic_c: float = 1.0,
    ridge_alpha: float = 10.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return unsigned selection importance while preserving coefficient sign."""
    x = sparse.csr_matrix(matrix)
    values = np.asarray(values, dtype=float)
    if binary:
        if len(np.unique(values.astype(int))) < 2:
            coefficients = np.zeros(x.shape[1], dtype=float)
        else:
            model = LogisticRegression(
                C=float(logistic_c), solver="liblinear", max_iter=1000,
                random_state=random_state,
            ).fit(x, values.astype(int))
            coefficients = np.asarray(model.coef_).reshape(-1)
    else:
        model = Ridge(alpha=float(ridge_alpha)).fit(x, values)
        coefficients = np.asarray(model.coef_).reshape(-1)
    frame = pd.DataFrame(
        {
            "feature": np.asarray(feature_names, dtype=str),
            "signed_score": coefficients,
            "unsigned_score": np.abs(coefficients),
        }
    )
    return frame.sort_values(
        ["unsigned_score", "feature"], ascending=[False, True], ignore_index=True
    )


def _subsample_indices(
    strata: np.ndarray, fraction: float, rng: np.random.Generator
) -> np.ndarray:
    selected: List[int] = []
    for value in np.unique(strata):
        positions = np.where(strata == value)[0]
        count = max(1, min(len(positions), int(np.ceil(len(positions) * fraction))))
        selected.extend(rng.choice(positions, size=count, replace=False).astype(int).tolist())
    return np.asarray(sorted(selected), dtype=int)


def add_linear_stability(
    screen: pd.DataFrame,
    matrix: Any,
    values: np.ndarray,
    *,
    binary: bool,
    strata: np.ndarray,
    config: TfidfTopicDiscoveryConfig,
    logistic_c: float,
    ridge_alpha: float,
    random_state: int,
) -> pd.DataFrame:
    result = screen.copy()
    n_features = len(result)
    feature_to_index = {name: index for index, name in enumerate(result["feature"])}
    selection = np.zeros(n_features, dtype=float)
    rank_score = np.zeros(n_features, dtype=float)
    signs = np.zeros(n_features, dtype=float)
    repeats = int(config.stability_repeats)
    rng = np.random.default_rng(random_state)
    top_count = max(1, int(np.ceil(config.top_fraction * n_features)))
    for repeat in range(repeats):
        positions = _subsample_indices(strata, config.stability_fraction, rng)
        sample = unsigned_linear_screen(
            sparse.csr_matrix(matrix)[positions],
            result["feature"].tolist(),
            values[positions],
            binary=binary,
            logistic_c=logistic_c,
            ridge_alpha=ridge_alpha,
            random_state=random_state + repeat + 1,
        )
        for rank, row in sample.iterrows():
            index = feature_to_index[str(row["feature"])]
            if rank < top_count:
                selection[index] += 1.0
            rank_score[index] += 1.0 - rank / max(1, n_features - 1)
            signs[index] += np.sign(float(row["signed_score"]))
    divisor = max(1, repeats)
    result["selection_stability"] = selection / divisor
    result["rank_stability"] = rank_score / divisor
    result["sign_stability"] = np.abs(signs) / divisor
    result["combined_importance"] = result["unsigned_score"] * (
        0.5 + 0.25 * result["selection_stability"] + 0.25 * result["rank_stability"]
    )
    return result.sort_values(
        ["combined_importance", "unsigned_score", "feature"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def add_effect_stability(
    scores: pd.DataFrame,
    matrix: Any,
    treatment: np.ndarray,
    outcome: np.ndarray,
    stacked_e: np.ndarray,
    stacked_m: np.ndarray,
    nuisance_sources: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    strata: np.ndarray,
    config: TfidfTopicDiscoveryConfig,
    random_state: int,
) -> pd.DataFrame:
    result = scores.copy()
    x = sparse.csr_matrix(matrix)
    names = result["feature"].tolist()
    reference_sign = np.sign(result["signed_score"].to_numpy(dtype=float))
    source_agreements: List[np.ndarray] = []
    for source_e, source_m in nuisance_sources:
        source = cohort_contrast_scores(
            x, names, treatment, outcome, source_e, source_m
        )["signed_score"].to_numpy(dtype=float)
        source_agreements.append((np.sign(source) == reference_sign).astype(float))
    result["nuisance_source_agreement"] = (
        np.mean(np.vstack(source_agreements), axis=0) if source_agreements
        else np.ones(len(result), dtype=float)
    )

    repeats = int(config.stability_repeats)
    selection = np.zeros(len(result), dtype=float)
    signs = np.zeros(len(result), dtype=float)
    rng = np.random.default_rng(random_state)
    eligible_support = (
        (result["support_control"].to_numpy() >= config.minimum_arm_document_support)
        & (result["support_treated"].to_numpy() >= config.minimum_arm_document_support)
    )
    eligible_count = max(1, int(np.sum(eligible_support)))
    top_count = max(1, int(np.ceil(config.top_fraction * eligible_count)))
    for _ in range(repeats):
        positions = _subsample_indices(strata, config.stability_fraction, rng)
        sample = cohort_contrast_scores(
            x[positions], names, treatment[positions], outcome[positions],
            stacked_e[positions], stacked_m[positions],
        )
        magnitude = sample["unsigned_score"].to_numpy(dtype=float)
        valid = np.where(eligible_support)[0]
        chosen = valid[np.argsort(-magnitude[valid], kind="stable")[:top_count]]
        selection[chosen] += 1.0
        signs += (
            np.sign(sample["signed_score"].to_numpy(dtype=float)) == reference_sign
        ).astype(float)
    divisor = max(1, repeats)
    result["subsample_selection_stability"] = selection / divisor
    result["subsample_sign_agreement"] = signs / divisor

    contribution = (treatment - stacked_e) * (
        (outcome - stacked_m)
        - float(result["constant_residual_effect"].iloc[0]) * (treatment - stacked_e)
    )
    tail_agreement = np.zeros(len(result), dtype=float)
    csc = x.tocsc()
    for index in range(x.shape[1]):
        values = np.asarray(csc[:, index].toarray()).ravel()
        present = values > 0.0
        if np.any(present) and np.any(~present):
            tail_sign = np.sign(np.mean(contribution[present]) - np.mean(contribution[~present]))
            tail_agreement[index] = float(tail_sign == reference_sign[index])
    result["tail_contrast_sign_agreement"] = tail_agreement
    result["eligible"] = (
        eligible_support
        & (result["nuisance_source_agreement"] >= config.minimum_nuisance_source_agreement)
        & (
            result["subsample_selection_stability"]
            >= config.minimum_subsample_selection_fraction
        )
        & (result["tail_contrast_sign_agreement"] >= config.minimum_tail_sign_agreement)
    )
    result["combined_importance"] = result["unsigned_score"] * (
        0.4
        + 0.2 * result["nuisance_source_agreement"]
        + 0.2 * result["subsample_selection_stability"]
        + 0.2 * result["subsample_sign_agreement"]
    )
    return result.sort_values(
        ["eligible", "combined_importance", "unsigned_score", "feature"],
        ascending=[False, False, False, True],
        ignore_index=True,
    )


def align_topic_components(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return candidate indices aligned to reference topics by cosine/Hungarian match."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate topic matrices must have equal 2D shapes")
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    candidate_norm = np.linalg.norm(candidate, axis=1, keepdims=True)
    reference_unit = reference / np.where(reference_norm > 0.0, reference_norm, 1.0)
    candidate_unit = candidate / np.where(candidate_norm > 0.0, candidate_norm, 1.0)
    reference_rows, candidate_rows = linear_sum_assignment(-(reference_unit @ candidate_unit.T))
    permutation = np.empty(reference.shape[0], dtype=int)
    permutation[reference_rows] = candidate_rows
    return permutation


@dataclass
class ConsensusNMFTopicBank:
    bank_name: str
    feature_names: List[str]
    selected_indices: np.ndarray
    feature_weights: np.ndarray
    models: List[NMF]
    component_norms: List[np.ndarray]
    alignments: List[np.ndarray]
    consensus_loadings: np.ndarray
    topic_terms: List[List[Dict[str, Any]]]
    requested_components: int
    actual_components: int
    seeds: List[int]
    reduction_reason: Optional[str] = None

    @classmethod
    def fit(
        cls,
        *,
        bank_name: str,
        matrix: Any,
        feature_names: Sequence[str],
        scores: pd.DataFrame,
        config: TfidfTopicDiscoveryConfig,
    ) -> Tuple[Optional["ConsensusNMFTopicBank"], np.ndarray]:
        x = sparse.csr_matrix(matrix, dtype=np.float64)
        names = np.asarray(feature_names, dtype=str)
        score_by_name = scores.set_index("feature")
        eligible = scores
        if "eligible" in eligible.columns:
            eligible = eligible[eligible["eligible"].astype(bool)]
        if eligible.empty or x.shape[1] == 0 or x.shape[0] < 2:
            return None, np.zeros((x.shape[0], 0), dtype=float)
        target_count = max(
            min(config.topic_count, len(eligible)),
            min(config.terms_per_topic, len(eligible)),
        )
        select_count = min(
            len(eligible),
            max(target_count, int(np.ceil(config.top_fraction * len(eligible)))),
        )
        selected_names = eligible.head(select_count)["feature"].astype(str).tolist()
        index_by_name = {name: index for index, name in enumerate(names)}
        selected_indices = np.asarray(
            [index_by_name[name] for name in selected_names if name in index_by_name], dtype=int
        )
        if len(selected_indices) == 0:
            return None, np.zeros((x.shape[0], 0), dtype=float)
        importance = np.asarray(
            [
                float(score_by_name.loc[names[index], "combined_importance"])
                for index in selected_indices
            ],
            dtype=float,
        )
        positive = importance[np.isfinite(importance) & (importance > 0.0)]
        median = float(np.median(positive)) if len(positive) else 1.0
        weights = np.clip(
            np.sqrt(np.maximum(importance, 0.0) / max(median, 1e-12)),
            config.importance_weight_min,
            config.importance_weight_max,
        )
        weighted = x[:, selected_indices].multiply(weights).tocsr()
        actual = min(config.topic_count, x.shape[0] - 1, len(selected_indices))
        if actual < 1:
            return None, np.zeros((x.shape[0], 0), dtype=float)
        reason = None
        if actual < config.topic_count:
            reason = (
                f"requested={config.topic_count}; rows={x.shape[0]}; "
                f"selected_terms={len(selected_indices)}"
            )
        models: List[NMF] = []
        normalized_h: List[np.ndarray] = []
        scaled_w: List[np.ndarray] = []
        norms: List[np.ndarray] = []
        for seed in config.topic_seeds:
            model = NMF(
                n_components=actual,
                init=config.nmf_init,
                solver=config.nmf_solver,
                beta_loss=config.nmf_beta_loss,
                max_iter=config.nmf_max_iter,
                tol=config.nmf_tol,
                random_state=int(seed),
            )
            w = model.fit_transform(weighted)
            h = np.asarray(model.components_, dtype=float)
            component_norm = np.linalg.norm(h, axis=1)
            component_norm = np.where(component_norm > 0.0, component_norm, 1.0)
            models.append(model)
            normalized_h.append(h / component_norm[:, None])
            scaled_w.append(w * component_norm[None, :])
            norms.append(component_norm)
        reference = normalized_h[0]
        alignments: List[np.ndarray] = [np.arange(actual, dtype=int)]
        aligned_h = [reference]
        aligned_w = [scaled_w[0]]
        for index in range(1, len(models)):
            permutation = align_topic_components(reference, normalized_h[index])
            alignments.append(permutation)
            aligned_h.append(normalized_h[index][permutation])
            aligned_w.append(scaled_w[index][:, permutation])
        consensus_h = np.mean(np.stack(aligned_h), axis=0)
        consensus_w = np.mean(np.stack(aligned_w), axis=0)

        selected_name_array = names[selected_indices]
        all_ranked_names = scores["feature"].astype(str).tolist()
        terms: List[List[Dict[str, Any]]] = []
        for topic_index in range(actual):
            order = np.argsort(-consensus_h[topic_index], kind="stable")
            topic_rows: List[Dict[str, Any]] = []
            used = set()
            for selected_position in order:
                name = str(selected_name_array[selected_position])
                used.add(name)
                topic_rows.append(
                    {
                        "term": name,
                        "loading": float(consensus_h[topic_index, selected_position]),
                        "screen_rank": int(score_by_name.index.get_loc(name)) + 1,
                        "signed_score": float(score_by_name.loc[name, "signed_score"]),
                    }
                )
                if len(topic_rows) == config.terms_per_topic:
                    break
            # A tiny selected set can still yield a traceable 15-term label
            # prompt by appending the highest-ranked zero-loading vocabulary.
            for name in all_ranked_names:
                if len(topic_rows) == config.terms_per_topic:
                    break
                if name in used:
                    continue
                used.add(name)
                topic_rows.append(
                    {
                        "term": name,
                        "loading": 0.0,
                        "screen_rank": int(score_by_name.index.get_loc(name)) + 1,
                        "signed_score": float(score_by_name.loc[name, "signed_score"]),
                    }
                )
            terms.append(topic_rows)
        bank = cls(
            bank_name=bank_name,
            feature_names=names.tolist(),
            selected_indices=selected_indices,
            feature_weights=weights,
            models=models,
            component_norms=norms,
            alignments=alignments,
            consensus_loadings=consensus_h,
            topic_terms=terms,
            requested_components=int(config.topic_count),
            actual_components=int(actual),
            seeds=list(config.topic_seeds),
            reduction_reason=reason,
        )
        return bank, consensus_w

    def transform(self, matrix: Any) -> np.ndarray:
        x = sparse.csr_matrix(matrix, dtype=np.float64)
        weighted = x[:, self.selected_indices].multiply(self.feature_weights).tocsr()
        aligned = []
        for model, norms, permutation in zip(
            self.models, self.component_norms, self.alignments
        ):
            values = model.transform(weighted) * norms[None, :]
            aligned.append(values[:, permutation])
        return np.mean(np.stack(aligned), axis=0)

    def metadata(self) -> Dict[str, Any]:
        return {
            "bank": self.bank_name,
            "requested_topic_count": self.requested_components,
            "actual_topic_count": self.actual_components,
            "component_reduction_reason": self.reduction_reason,
            "seeds": self.seeds,
            "selected_term_count": int(len(self.selected_indices)),
            "selected_terms": [self.feature_names[index] for index in self.selected_indices],
            "feature_weights": self.feature_weights.astype(float).tolist(),
            "alignments": [alignment.astype(int).tolist() for alignment in self.alignments],
            "topics": [
                {
                    "topic_id": f"{self.bank_name}_topic_{index + 1:03d}",
                    "bank": self.bank_name,
                    "terms": topic_terms,
                }
                for index, topic_terms in enumerate(self.topic_terms)
            ],
        }


@dataclass
class FittedTopicContext:
    common_vectorizer: TfidfVectorizer
    treatment_stack: CrossFittedStack
    outcome_stack: CrossFittedStack
    topic_banks: Dict[str, ConsensusNMFTopicBank]
    config_hash: str

    def transform_topics(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        matrix = self.common_vectorizer.transform(_normalize_texts(texts))
        return {name: bank.transform(matrix) for name, bank in self.topic_banks.items()}


def _common_vectorizer(config: TfidfTopicDiscoveryConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)[a-z0-9%<>+=-]+",
        ngram_range=(config.ngram_range_min, config.ngram_range_max),
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        max_features=config.max_features,
        dtype=np.float32,
    )


def _strata(treatment: np.ndarray, outcome: np.ndarray, *, outcome_binary: bool) -> np.ndarray:
    if outcome_binary:
        outcome_group = outcome.astype(int)
    else:
        try:
            outcome_group = pd.qcut(outcome, q=min(4, len(np.unique(outcome))), labels=False,
                                    duplicates="drop").to_numpy()
        except AttributeError:
            outcome_group = np.asarray(
                pd.qcut(outcome, q=min(4, len(np.unique(outcome))), labels=False,
                        duplicates="drop")
            )
        outcome_group = np.nan_to_num(outcome_group, nan=0).astype(int)
    return treatment.astype(int) * (int(np.max(outcome_group)) + 1) + outcome_group


def compact_topic_score_tests(score_tests: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the small handoff view while retaining auditable score evidence."""
    compact: Dict[str, Any] = {
        "schema_version": score_tests.get(
            "schema_version", TOPIC_SCORE_TEST_SCHEMA_VERSION
        ),
        "status": score_tests.get("status", "not_run"),
        "reason": score_tests.get("reason"),
        "uses_heldout_treatment_and_outcome": bool(
            score_tests.get("uses_heldout_treatment_and_outcome", False)
        ),
        "banks": {},
    }
    for bank_name, bank_result in (score_tests.get("banks") or {}).items():
        compact["banks"][bank_name] = {
            key: bank_result.get(key)
            for key in (
                "selected_topic_ids",
                "selection_count",
                "selection_rule",
                "fdr_level",
                "p_threshold",
                "minimum_topics",
                "maximum_topics",
                "complete_family_multiplier_bootstrap",
                "complete_term_group_multiplier_bootstrap",
                "selected_ngram_terms",
                "ngram_selection_count",
                "ngram_selection_rule",
                "unique_ngram_count",
                "testable_unique_ngram_count",
            )
        }
        compact["banks"][bank_name]["bootstrap_calibration"] = (
            bank_result.get("bootstrap_calibration") or {}
        )
        compact["banks"][bank_name]["topic_ranking"] = [
            {
                key: topic_result.get(key)
                for key in (
                    "topic_id",
                    "evidence_rank",
                    "selected_for_agent",
                    "selection_reason",
                    "primary_p",
                    "primary_p_source",
                    "fdr_q",
                    "familywise_p",
                    "topic_score_testable",
                    "topic_score_moment",
                    "topic_standardized_score",
                    "topic_unadjusted_two_sided_p",
                    "topic_familywise_p",
                    "term_group_primary_p",
                    "term_group_primary_p_source",
                    "term_group_fdr_q",
                    "term_group_familywise_p",
                    "quadratic_statistic_per_rank",
                    "maximum_absolute_standardized_score",
                    "selected_ngram_count",
                    "selected_ngram_terms",
                )
            }
            for topic_result in bank_result.get("topic_tests", [])
        ]
    orphan = score_tests.get("effect_orphan_ngram_branch") or {}
    compact["effect_orphan_ngram_branch"] = {
        key: orphan.get(key)
        for key in (
            "status",
            "candidate_definition",
            "topic_term_exclusion_is_fit_side",
            "cluster_construction_uses_heldout_rows_or_labels",
            "candidate_count_before_topic_exclusion",
            "represented_topic_term_exclusion_count",
            "candidate_count_before_nested_deduplication",
            "deduplicated_alias_count",
            "representative_count",
            "cluster_count",
            "selected_cluster_ids",
            "selected_clusters",
            "selection_count",
            "selection_rule",
            "minimum_selected_clusters",
            "maximum_selected_clusters",
        )
    }
    return compact


def fit_tfidf_topic_context(
    *,
    fit_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    views: Sequence[BoWViewConfig],
    nuisance_folds: int,
    config: TfidfTopicDiscoveryConfig,
    artifact_dir: Path,
    scope_id: str,
    enable_heldout_score_tests: bool = False,
) -> Dict[str, Any]:
    """Fit one exact context and transform its externally held-out row set."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fit_texts = _normalize_texts(fit_df[text_column].fillna(""))
    heldout_texts = _normalize_texts(heldout_df[text_column].fillna(""))
    treatment = fit_df[treatment_column].to_numpy(dtype=float)
    outcome = fit_df[outcome_column].to_numpy(dtype=float)
    outcome_binary = str(outcome_type).lower() != "continuous"
    strata = _strata(treatment, outcome, outcome_binary=outcome_binary)
    joint_nuisance = fit_joint_cross_fitted_nuisance_stacks(
        texts=fit_texts,
        treatment=treatment,
        outcome=outcome,
        outcome_binary=outcome_binary,
        strata=strata,
        views=views,
        folds=nuisance_folds,
        random_state=config.random_state + 101,
    )
    treatment_result = joint_nuisance["treatment"]
    outcome_result = joint_nuisance["outcome"]
    external_e, external_e_views = treatment_result["fitted"].predict(heldout_texts)
    external_m, external_m_views = outcome_result["fitted"].predict(heldout_texts)

    vectorizer = _common_vectorizer(config)
    common_fit = vectorizer.fit_transform(fit_texts)
    common_heldout = vectorizer.transform(heldout_texts)
    feature_names = vectorizer.get_feature_names_out()

    linear_view = next(
        (
            view for view in views
            if view.bow_model == "linear"
            and view.ngram_range_min == 1
            and view.ngram_range_max == 3
        ),
        BoWViewConfig(
            name="linear_1_3_topic_basis",
            max_features=config.max_features,
            min_df=config.min_df,
            max_df=config.max_df,
            ngram_range_min=1,
            ngram_range_max=3,
            sublinear_tf=config.sublinear_tf,
            bow_model="linear",
        ),
    )
    treatment_scores = add_linear_stability(
        unsigned_linear_screen(
            common_fit, feature_names, treatment, binary=True,
            logistic_c=linear_view.logistic_c, random_state=config.random_state + 301,
        ),
        common_fit,
        treatment,
        binary=True,
        strata=strata,
        config=config,
        logistic_c=linear_view.logistic_c,
        ridge_alpha=linear_view.ridge_alpha,
        random_state=config.random_state + 311,
    )
    treatment_scores["eligible"] = True
    outcome_scores = add_linear_stability(
        unsigned_linear_screen(
            common_fit, feature_names, outcome, binary=outcome_binary,
            logistic_c=linear_view.logistic_c,
            ridge_alpha=linear_view.ridge_alpha,
            random_state=config.random_state + 401,
        ),
        common_fit,
        outcome,
        binary=outcome_binary,
        strata=strata,
        config=config,
        logistic_c=linear_view.logistic_c,
        ridge_alpha=linear_view.ridge_alpha,
        random_state=config.random_state + 411,
    )
    outcome_scores["eligible"] = True

    nuisance_sources = [
        (treatment_result["base_oof"][:, index], outcome_result["base_oof"][:, index])
        for index in range(min(treatment_result["base_oof"].shape[1],
                               outcome_result["base_oof"].shape[1]))
    ]
    effect_scores = add_effect_stability(
        cohort_contrast_scores(
            common_fit,
            feature_names,
            treatment,
            outcome,
            treatment_result["stacked_oof"],
            outcome_result["stacked_oof"],
        ),
        common_fit,
        treatment,
        outcome,
        treatment_result["stacked_oof"],
        outcome_result["stacked_oof"],
        nuisance_sources,
        strata=strata,
        config=config,
        random_state=config.random_state + 501,
    )

    score_frames = {
        "treatment": treatment_scores,
        "outcome": outcome_scores,
        "effect": effect_scores,
    }
    topic_banks: Dict[str, ConsensusNMFTopicBank] = {}
    fit_topic_values: Dict[str, np.ndarray] = {}
    heldout_topic_values: Dict[str, np.ndarray] = {}
    bank_metadata: Dict[str, Dict[str, Any]] = {}
    score_paths: Dict[str, str] = {}
    for bank_name, score_frame in score_frames.items():
        score_path = artifact_dir / f"{bank_name}_ngram_scores.parquet"
        score_frame.to_parquet(score_path, index=False)
        score_paths[bank_name] = str(score_path)
        bank, fit_values = ConsensusNMFTopicBank.fit(
            bank_name=bank_name,
            matrix=common_fit,
            feature_names=feature_names,
            scores=score_frame,
            config=config,
        )
        if bank is None:
            bank_metadata[bank_name] = {
                "bank": bank_name,
                "requested_topic_count": config.topic_count,
                "actual_topic_count": 0,
                "weak_or_unstable_raw_evidence": bank_name == "effect",
                "topics": [],
            }
            continue
        topic_banks[bank_name] = bank
        fit_topic_values[bank_name] = fit_values
        heldout_topic_values[bank_name] = bank.transform(common_heldout)
        bank_metadata[bank_name] = bank.metadata()
        bank_metadata[bank_name]["weak_or_unstable_raw_evidence"] = False

    topic_score_tests: Dict[str, Any] = {
        "schema_version": TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "status": "not_run",
        "reason": "heldout_labels_reserved_or_score_tests_disabled",
        "uses_heldout_treatment_and_outcome": False,
        "banks": {},
    }
    topic_score_tests_path: Optional[Path] = None
    if bool(config.score_test_enabled) and bool(enable_heldout_score_tests):
        # This block is used only for exact candidate-selection inner contexts.
        # Full outer contexts never read outer-held-out treatment or outcome.
        topic_score_tests = score_topic_banks(
            fit_matrix=common_fit,
            heldout_matrix=common_heldout,
            feature_names=feature_names,
            topic_banks=bank_metadata,
            fit_topic_values=fit_topic_values,
            heldout_topic_values=heldout_topic_values,
            fit_treatment=treatment,
            fit_outcome=outcome,
            heldout_treatment=heldout_df[treatment_column].to_numpy(dtype=float),
            heldout_outcome=heldout_df[outcome_column].to_numpy(dtype=float),
            fit_propensity=treatment_result["stacked_oof"],
            fit_outcome_prediction=outcome_result["stacked_oof"],
            heldout_propensity=external_e,
            heldout_outcome_prediction=external_m,
            config=config,
            scope_id=scope_id,
            raw_ngram_scores=score_frames,
        )
        topic_score_tests["status"] = "completed"
        topic_score_tests_path = artifact_dir / "topic_score_tests.json"
        topic_score_tests_path.write_text(
            json.dumps(topic_score_tests, indent=2, default=str),
            encoding="utf-8",
        )

    compact_score_tests = compact_topic_score_tests(topic_score_tests)

    fitted = FittedTopicContext(
        common_vectorizer=vectorizer,
        treatment_stack=treatment_result["fitted"],
        outcome_stack=outcome_result["fitted"],
        topic_banks=topic_banks,
        config_hash=stable_hash(asdict(config)),
    )
    model_path = artifact_dir / "fitted_context.joblib"
    joblib.dump(fitted, model_path)
    fit_topics_path = artifact_dir / "fit_topic_values.npz"
    heldout_topics_path = artifact_dir / "heldout_topic_values.npz"
    np.savez_compressed(fit_topics_path, **fit_topic_values)
    np.savez_compressed(heldout_topics_path, **heldout_topic_values)

    fit_row_ids = fit_df["_oci_row_id"].astype(int).tolist()
    heldout_row_ids = heldout_df["_oci_row_id"].astype(int).tolist()
    oof_rows: List[Dict[str, Any]] = []
    for position, row_id in enumerate(fit_row_ids):
        row = {
            "_oci_row_id": int(row_id),
            "prediction_scope": "fit_oof",
            "nuisance_fold": int(treatment_result["fold_ids"][position]),
            "treatment_stacked": float(treatment_result["stacked_oof"][position]),
            "outcome_stacked": float(outcome_result["stacked_oof"][position]),
            "fit_row_ids": [
                int(fit_row_ids[index])
                for index in treatment_result["fit_positions_by_row"][position]
            ],
        }
        for index, view in enumerate(views):
            row[f"treatment_view__{view.name}"] = float(
                treatment_result["base_oof"][position, index]
            )
            row[f"outcome_view__{view.name}"] = float(
                outcome_result["base_oof"][position, index]
            )
        oof_rows.append(row)
    external_rows: List[Dict[str, Any]] = []
    for position, row_id in enumerate(heldout_row_ids):
        row = {
            "_oci_row_id": int(row_id),
            "prediction_scope": "external_heldout",
            "treatment_stacked": float(external_e[position]),
            "outcome_stacked": float(external_m[position]),
            "fit_row_ids": fit_row_ids,
        }
        for view in views:
            row[f"treatment_view__{view.name}"] = float(external_e_views[view.name][position])
            row[f"outcome_view__{view.name}"] = float(external_m_views[view.name][position])
        external_rows.append(row)
    nuisance_path = artifact_dir / "nuisance_predictions.parquet"
    pd.DataFrame([*oof_rows, *external_rows]).to_parquet(nuisance_path, index=False)

    metadata = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "scope_id": scope_id,
        "fit_row_fingerprint": row_set_fingerprint(fit_row_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_row_ids),
        "fit_row_ids": fit_row_ids,
        "heldout_row_ids": heldout_row_ids,
        "config_hash": fitted.config_hash,
        "common_vocabulary_size": int(len(feature_names)),
        "common_vocabulary": feature_names.astype(str).tolist(),
        "nuisance": {
            "treatment": {
                "stacked_metrics": treatment_result["metrics"],
                "view_metrics": treatment_result["view_metrics"],
                "stack_config_hash": treatment_result["fitted"].config_hash,
            },
            "outcome": {
                "stacked_metrics": outcome_result["metrics"],
                "view_metrics": outcome_result["view_metrics"],
                "stack_config_hash": outcome_result["fitted"].config_hash,
            },
        },
        "topic_banks": bank_metadata,
        "heldout_score_tests_enabled": bool(enable_heldout_score_tests),
        "topic_score_tests": compact_score_tests,
        "artifacts": {
            "fitted_context": str(model_path),
            "ngram_scores": score_paths,
            "fit_topic_values": str(fit_topics_path),
            "heldout_topic_values": str(heldout_topics_path),
            "nuisance_predictions": str(nuisance_path),
            "topic_score_tests": (
                None if topic_score_tests_path is None else str(topic_score_tests_path)
            ),
        },
    }
    metadata_path = artifact_dir / "context_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata
