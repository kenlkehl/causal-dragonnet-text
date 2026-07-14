#!/usr/bin/env python
"""One-off cohort-level TF-IDF contrast followed by a residualized causal forest."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from econml.grf import CausalForest
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


LOGGER = logging.getLogger("tfidf_cohort_contrast")
DEFAULT_DATASET = (
    "synthetic_data/example_synthetic_datasets/"
    "one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet"
)
DEFAULT_NUISANCE = (
    "../pcori_experiments/one_conf_one_mod_agent_refactor_7-9-26/"
    "multi_model_forest/33843981024b/text_model_feature_predictions.parquet"
)
DEFAULT_OUTPUT = "one_off_tfidf_cohort_contrast/results_one_conf_one_mod"


@dataclass(frozen=True)
class ContrastResult:
    tau_constant: float
    treatment_residual: np.ndarray
    outcome_residual: np.ndarray
    patient_contribution: np.ndarray
    raw_moment: np.ndarray
    standard_error: np.ndarray
    z_score: np.ndarray


def parse_int_list(value: str) -> List[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def normalize_text(value: object) -> str:
    text = str(value or "").lower()
    replacements = {
        "≥": ">=",
        "≤": "<=",
        "–": "-",
        "—": "-",
        "−": "-",
        "‑": "-",
        "％": "%",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def make_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)[a-z0-9%<>+=.-]+",
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def clip_probabilities(values: np.ndarray, clip: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), clip, 1.0 - clip)


def cohort_contrast(
    x: sparse.csr_matrix,
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    outcome_prediction: np.ndarray,
    *,
    probability_clip: float,
) -> ContrastResult:
    """Return a standardized score-test vector without fitting patient targets."""
    n = x.shape[0]
    if n < 3:
        raise ValueError("cohort contrast requires at least three rows")
    e_hat = clip_probabilities(propensity, probability_clip)
    m_hat = clip_probabilities(outcome_prediction, probability_clip)
    u = np.asarray(treatment, dtype=float) - e_hat
    v = np.asarray(outcome, dtype=float) - m_hat
    denominator = float(np.dot(u, u))
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("treatment residuals contain no usable variation")
    tau_constant = float(np.dot(u, v) / denominator)
    contribution = u * (v - tau_constant * u)

    # Because tau_constant solves the global residual moment, contribution sums
    # to approximately zero.  Centering X is retained explicitly in the robust
    # standard-error calculation below.
    sum_x_contribution = np.asarray(x.T @ contribution).reshape(-1)
    raw_moment = sum_x_contribution / float(n)

    x_mean = np.asarray(x.mean(axis=0)).reshape(-1)
    contribution_sq = np.square(contribution)
    sum_contribution_sq = float(np.sum(contribution_sq))
    sum_x_contribution_sq = np.asarray(x.T @ contribution_sq).reshape(-1)
    sum_x2_contribution_sq = np.asarray(x.power(2).T @ contribution_sq).reshape(-1)
    sum_centered_sq = (
        sum_x2_contribution_sq
        - 2.0 * x_mean * sum_x_contribution_sq
        + np.square(x_mean) * sum_contribution_sq
    )
    # Remove the squared sample mean to obtain the sample variance of each
    # feature's influence values, then convert it to the SE of their mean.
    variance_numerator = np.maximum(
        sum_centered_sq - float(n) * np.square(raw_moment),
        0.0,
    )
    standard_error = np.sqrt(variance_numerator / float(n * (n - 1)))
    z_score = np.divide(
        raw_moment,
        standard_error,
        out=np.zeros_like(raw_moment),
        where=standard_error > 1e-12,
    )
    z_score[~np.isfinite(z_score)] = 0.0
    return ContrastResult(
        tau_constant=tau_constant,
        treatment_residual=u,
        outcome_residual=v,
        patient_contribution=contribution,
        raw_moment=raw_moment,
        standard_error=standard_error,
        z_score=z_score,
    )


def tail_contrast(
    x: sparse.csr_matrix,
    contribution: np.ndarray,
    *,
    quantile: float,
) -> np.ndarray:
    low_cut, high_cut = np.quantile(contribution, [quantile, 1.0 - quantile])
    low = contribution <= low_cut
    high = contribution >= high_cut
    if int(np.sum(low)) < 2 or int(np.sum(high)) < 2:
        return np.zeros(x.shape[1], dtype=float)
    delta = np.asarray(x[high].mean(axis=0) - x[low].mean(axis=0)).reshape(-1)
    second_moment = np.asarray(x.power(2).mean(axis=0)).reshape(-1)
    mean = np.asarray(x.mean(axis=0)).reshape(-1)
    feature_sd = np.sqrt(np.maximum(second_moment - np.square(mean), 0.0))
    return np.divide(
        delta,
        feature_sd,
        out=np.zeros_like(delta),
        where=feature_sd > 1e-12,
    )


def stratified_subsample_indices(
    treatment: np.ndarray,
    outcome: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = 2 * np.asarray(treatment, dtype=int) + np.asarray(outcome, dtype=int)
    chosen: List[np.ndarray] = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        count = max(2, int(math.floor(fraction * len(indices))))
        count = min(count, len(indices))
        chosen.append(rng.choice(indices, size=count, replace=False))
    result = np.concatenate(chosen)
    rng.shuffle(result)
    return result


def stability_diagnostics(
    x: sparse.csr_matrix,
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    outcome_prediction: np.ndarray,
    primary_z: np.ndarray,
    *,
    repeats: int,
    fraction: float,
    top_pool: int,
    probability_clip: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sign_matches = np.zeros(x.shape[1], dtype=float)
    selected = np.zeros(x.shape[1], dtype=float)
    rng = np.random.default_rng(random_state)
    pool_size = min(max(1, int(top_pool)), x.shape[1])
    for _ in range(repeats):
        indices = stratified_subsample_indices(treatment, outcome, fraction, rng)
        result = cohort_contrast(
            x[indices],
            treatment[indices],
            outcome[indices],
            propensity[indices],
            outcome_prediction[indices],
            probability_clip=probability_clip,
        )
        sign_matches += (np.sign(result.z_score) == np.sign(primary_z)).astype(float)
        if pool_size == x.shape[1]:
            top_indices = np.arange(x.shape[1])
        else:
            top_indices = np.argpartition(np.abs(result.z_score), -pool_size)[-pool_size:]
        selected[top_indices] += 1.0
    denominator = float(max(1, repeats))
    return sign_matches / denominator, selected / denominator


def source_sign_agreement(
    x: sparse.csr_matrix,
    treatment: np.ndarray,
    outcome: np.ndarray,
    source_predictions: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    primary_z: np.ndarray,
    *,
    probability_clip: float,
) -> Tuple[np.ndarray, int]:
    matches: List[np.ndarray] = []
    for source_name, (propensity, outcome_prediction) in source_predictions.items():
        result = cohort_contrast(
            x,
            treatment,
            outcome,
            propensity,
            outcome_prediction,
            probability_clip=probability_clip,
        )
        matches.append((np.sign(result.z_score) == np.sign(primary_z)).astype(float))
    if not matches:
        return np.ones_like(primary_z), 0
    return np.mean(np.vstack(matches), axis=0), len(matches)


def feature_support(
    x: sparse.csr_matrix,
    treatment: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    present = x.copy()
    present.data = np.ones_like(present.data)
    document_frequency = np.asarray(present.sum(axis=0)).reshape(-1).astype(int)
    treated = np.asarray(present.T @ np.asarray(treatment, dtype=float)).reshape(-1)
    treated = np.rint(treated).astype(int)
    control = document_frequency - treated
    return document_frequency, treated, control


def build_feature_frame(
    *,
    outer_fold: int,
    feature_names: np.ndarray,
    result: ContrastResult,
    tail_score: np.ndarray,
    source_agreement: np.ndarray,
    subsample_agreement: np.ndarray,
    selection_frequency: np.ndarray,
    document_frequency: np.ndarray,
    treated_count: np.ndarray,
    control_count: np.ndarray,
    min_arm_count: int,
) -> pd.DataFrame:
    sign_tail_agreement = (np.sign(tail_score) == np.sign(result.z_score)).astype(float)
    eligible = (
        np.isfinite(result.z_score)
        & (result.standard_error > 1e-12)
        & (treated_count >= min_arm_count)
        & (control_count >= min_arm_count)
    )
    stability_multiplier = (
        (0.50 + 0.50 * source_agreement)
        * (0.50 + 0.50 * subsample_agreement)
        * (0.25 + 0.75 * selection_frequency)
        * (0.90 + 0.10 * sign_tail_agreement)
    )
    ranking_score = np.abs(result.z_score) * stability_multiplier
    ranking_score[~eligible] = -np.inf
    order = np.argsort(ranking_score)[::-1]
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    frame = pd.DataFrame(
        {
            "outer_fold": int(outer_fold),
            "feature_index": np.arange(len(feature_names), dtype=int),
            "feature": feature_names.astype(str),
            "document_frequency": document_frequency,
            "treated_count": treated_count,
            "control_count": control_count,
            "raw_moment": result.raw_moment,
            "standard_error": result.standard_error,
            "z_score": result.z_score,
            "tail_score": tail_score,
            "source_sign_agreement": source_agreement,
            "subsample_sign_agreement": subsample_agreement,
            "subsample_top_pool_frequency": selection_frequency,
            "tail_sign_agreement": sign_tail_agreement,
            "eligible": eligible,
            "ranking_score": ranking_score,
            "rank": ranks,
        }
    )
    return frame


def _aligned_source_arrays(
    rows: pd.DataFrame,
    row_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    indexed = rows.drop_duplicates("_oci_row_id").set_index("_oci_row_id")
    missing = sorted(set(map(int, row_ids)) - set(map(int, indexed.index)))
    if missing:
        raise ValueError(f"nuisance source is missing {len(missing)} required rows")
    aligned = indexed.loc[row_ids]
    return (
        aligned["e_hat"].to_numpy(dtype=float),
        aligned["m_hat"].to_numpy(dtype=float),
    )


def nuisance_rows_for_fold(
    nuisance_df: pd.DataFrame,
    outer_fold: int,
    split_role: str,
    source_name: str,
) -> pd.DataFrame:
    rows = nuisance_df[
        (nuisance_df["outer_fold"] == int(outer_fold))
        & (nuisance_df["split_role"] == split_role)
        & (nuisance_df["source_name"] == source_name)
        & nuisance_df["e_hat"].notna()
        & nuisance_df["m_hat"].notna()
    ].copy()
    if rows.empty:
        raise ValueError(
            f"no nuisance rows for fold={outer_fold}, split={split_role}, source={source_name}"
        )
    rows["_oci_row_id"] = rows["_oci_row_id"].astype(int)
    return rows


def individual_nuisance_sources(nuisance_df: pd.DataFrame, outer_fold: int) -> List[str]:
    subset = nuisance_df[
        (nuisance_df["outer_fold"] == int(outer_fold))
        & (nuisance_df["split_role"] == "train_inner_oof")
        & nuisance_df["e_hat"].notna()
        & nuisance_df["m_hat"].notna()
    ]
    names = sorted(set(subset["source_name"].astype(str)))
    return [name for name in names if name != "ensemble_mean_nuisance"]


def _logit_features(probability_columns: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probability_columns, dtype=float), 1e-4, 1.0 - 1e-4)
    return np.log(probabilities / (1.0 - probabilities))


def crossfit_logistic_stack(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    *,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return meta-level OOF train predictions and an outer-test prediction."""
    labels = np.asarray(train_labels, dtype=int)
    min_class = int(np.min(np.bincount(labels)))
    folds = min(5, min_class)
    if folds < 2:
        raise ValueError("stacked nuisance model needs at least two rows per class")
    oof = np.full(len(labels), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    for fit_idx, heldout_idx in splitter.split(train_features, labels):
        model = LogisticRegression(C=0.5, solver="lbfgs", max_iter=2000)
        model.fit(train_features[fit_idx], labels[fit_idx])
        oof[heldout_idx] = model.predict_proba(train_features[heldout_idx])[:, 1]
    final_model = LogisticRegression(C=0.5, solver="lbfgs", max_iter=2000)
    final_model.fit(train_features, labels)
    test_prediction = final_model.predict_proba(test_features)[:, 1]
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("incomplete stacked nuisance OOF predictions")
    return oof, test_prediction


def stacked_nuisance_predictions(
    nuisance_df: pd.DataFrame,
    outer_fold: int,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    source_names: Sequence[str],
    *,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_e: List[np.ndarray] = []
    train_m: List[np.ndarray] = []
    test_e: List[np.ndarray] = []
    test_m: List[np.ndarray] = []
    for source_name in source_names:
        train_rows = nuisance_rows_for_fold(
            nuisance_df,
            outer_fold,
            "train_inner_oof",
            source_name,
        )
        test_rows = nuisance_rows_for_fold(
            nuisance_df,
            outer_fold,
            "test_outer_train_fit",
            source_name,
        )
        e_train, m_train = _aligned_source_arrays(train_rows, train_ids)
        e_test, m_test = _aligned_source_arrays(test_rows, test_ids)
        train_e.append(e_train)
        train_m.append(m_train)
        test_e.append(e_test)
        test_m.append(m_test)
    e_train_features = _logit_features(np.column_stack(train_e))
    m_train_features = _logit_features(np.column_stack(train_m))
    e_test_features = _logit_features(np.column_stack(test_e))
    m_test_features = _logit_features(np.column_stack(test_m))
    e_train, e_test = crossfit_logistic_stack(
        e_train_features,
        treatment,
        e_test_features,
        random_state=random_state + 17,
    )
    m_train, m_test = crossfit_logistic_stack(
        m_train_features,
        outcome,
        m_test_features,
        random_state=random_state + 29,
    )
    return e_train, m_train, e_test, m_test


def safe_corr(function, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(function(x, y).statistic)


def prediction_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    true_tau = frame["true_ite_prob"].to_numpy(dtype=float)
    pred_tau = frame["pred_ite_prob"].to_numpy(dtype=float)
    high_pdl1 = (frame["true_pdl1_expression"] == ">=50%").to_numpy(dtype=int)
    if not np.any(high_pdl1):
        high_pdl1 = (frame["true_pdl1_expression"] == "≥50%").to_numpy(dtype=int)
    low = frame["true_pdl1_expression"].isin(["<1%"])
    high = frame["true_pdl1_expression"].isin([">=50%", "≥50%"])
    true_gap = float(frame.loc[high, "true_ite_prob"].mean() - frame.loc[low, "true_ite_prob"].mean())
    predicted_gap = float(
        frame.loc[high, "pred_ite_prob"].mean() - frame.loc[low, "pred_ite_prob"].mean()
    )
    result = {
        "n": int(len(frame)),
        "true_ate": float(np.mean(true_tau)),
        "predicted_ate": float(np.mean(pred_tau)),
        "ate_bias": float(np.mean(pred_tau) - np.mean(true_tau)),
        "pehe": float(np.sqrt(mean_squared_error(true_tau, pred_tau))),
        "ite_mae": float(mean_absolute_error(true_tau, pred_tau)),
        "ite_r2": float(r2_score(true_tau, pred_tau)),
        "ite_pearson": safe_corr(pearsonr, true_tau, pred_tau),
        "ite_spearman": safe_corr(spearmanr, true_tau, pred_tau),
        "predicted_ite_sd": float(np.std(pred_tau, ddof=1)),
        "true_ite_sd": float(np.std(true_tau, ddof=1)),
        "high_pdl1_auroc": float(roc_auc_score(high_pdl1, pred_tau)),
        "pdl1_high_vs_low_true_gap": true_gap,
        "pdl1_high_vs_low_predicted_gap": predicted_gap,
        "pdl1_gap_fraction_recovered": predicted_gap / true_gap,
    }
    if {"pred_ite_lower", "pred_ite_upper"}.issubset(frame.columns):
        lower = frame["pred_ite_lower"].to_numpy(dtype=float)
        upper = frame["pred_ite_upper"].to_numpy(dtype=float)
        result["oracle_row_ite_interval_coverage"] = float(
            np.mean((lower <= true_tau) & (true_tau <= upper))
        )
        result["mean_interval_width"] = float(np.mean(upper - lower))
    if {"test_treatment_residual", "test_outcome_residual"}.issubset(frame.columns):
        u = frame["test_treatment_residual"].to_numpy(dtype=float)
        v = frame["test_outcome_residual"].to_numpy(dtype=float)
        result["r_loss"] = float(np.mean(np.square(v - u * pred_tau)))
        tau_constant = frame["train_constant_effect"].to_numpy(dtype=float)
        baseline = float(np.mean(np.square(v - u * tau_constant)))
        result["constant_effect_r_loss"] = baseline
        result["r_loss_improvement_vs_constant"] = baseline - result["r_loss"]
    return result


def fit_residual_causal_forest(
    x_train: np.ndarray,
    x_test: np.ndarray,
    treatment_residual: np.ndarray,
    outcome_residual: np.ndarray,
    *,
    args: argparse.Namespace,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forest = CausalForest(
        n_estimators=args.forest_estimators,
        criterion="mse",
        min_samples_split=max(2 * args.forest_min_leaf, 10),
        min_samples_leaf=args.forest_min_leaf,
        min_var_fraction_leaf=args.min_var_fraction_leaf,
        max_features=args.forest_max_features,
        max_samples=args.forest_max_samples,
        honest=True,
        inference=True,
        fit_intercept=True,
        subforest_size=4,
        n_jobs=args.n_jobs,
        random_state=random_state,
    )
    forest.fit(
        np.asarray(x_train, dtype=np.float64),
        np.asarray(treatment_residual, dtype=np.float64).reshape(-1, 1),
        np.asarray(outcome_residual, dtype=np.float64).reshape(-1, 1),
    )
    point, lower, upper = forest.predict(
        np.asarray(x_test, dtype=np.float64),
        interval=True,
        alpha=0.05,
    )
    return point.reshape(-1), lower.reshape(-1), upper.reshape(-1)


def _input_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> None:
    start = time.time()
    dataset_path = Path(args.dataset)
    nuisance_path = Path(args.nuisance_predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_parquet(dataset_path).reset_index(drop=True)
    required = {
        args.text_column,
        args.treatment_column,
        args.outcome_column,
        "true_ite_prob",
        "true_pdl1_expression",
        "true_age",
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    dataset["_oci_row_id"] = np.arange(len(dataset), dtype=int)
    nuisance_df = pd.read_parquet(nuisance_path)
    top_k_values = args.top_k
    if args.primary_top_k not in top_k_values:
        raise ValueError("primary_top_k must be included in top_k")

    run_config = {
        "args": vars(args),
        "dataset_sha256": _input_sha256(dataset_path),
        "nuisance_predictions_sha256": _input_sha256(nuisance_path),
        "started_at_unix": start,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )

    all_feature_frames: List[pd.DataFrame] = []
    all_selected_frames: List[pd.DataFrame] = []
    all_predictions: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, float]] = []
    outer_folds = sorted(int(value) for value in nuisance_df["outer_fold"].dropna().unique())

    for outer_fold in outer_folds:
        LOGGER.info("Outer fold %s/%s", outer_fold, len(outer_folds))
        # Use a fixed source only to recover the exact outer-fold membership.
        # The primary nuisance values can come from a cross-fitted stack below.
        anchor_train_rows = nuisance_rows_for_fold(
            nuisance_df,
            outer_fold,
            "train_inner_oof",
            "ensemble_mean_nuisance",
        )
        anchor_test_rows = nuisance_rows_for_fold(
            nuisance_df,
            outer_fold,
            "test_outer_train_fit",
            "ensemble_mean_nuisance",
        )
        train_ids = np.sort(anchor_train_rows["_oci_row_id"].unique())
        test_ids = np.sort(anchor_test_rows["_oci_row_id"].unique())
        if set(train_ids) & set(test_ids):
            raise ValueError(f"outer fold {outer_fold} has overlapping train/test row ids")
        if len(train_ids) + len(test_ids) != len(dataset):
            raise ValueError(f"outer fold {outer_fold} does not partition the dataset")

        train_df = dataset.set_index("_oci_row_id").loc[train_ids].copy()
        test_df = dataset.set_index("_oci_row_id").loc[test_ids].copy()
        train_texts = [normalize_text(value) for value in train_df[args.text_column]]
        test_texts = [normalize_text(value) for value in test_df[args.text_column]]
        vectorizer = make_vectorizer(args)
        x_train = vectorizer.fit_transform(train_texts).tocsr()
        x_test = vectorizer.transform(test_texts).tocsr()
        feature_names = vectorizer.get_feature_names_out()
        LOGGER.info("  vocabulary=%s, train=%s, test=%s", len(feature_names), len(train_ids), len(test_ids))

        treatment_train = train_df[args.treatment_column].to_numpy(dtype=float)
        outcome_train = train_df[args.outcome_column].to_numpy(dtype=float)
        source_names = individual_nuisance_sources(nuisance_df, outer_fold)
        if args.nuisance_source == "stacked":
            e_train, m_train, e_test, m_test = stacked_nuisance_predictions(
                nuisance_df,
                outer_fold,
                train_ids,
                test_ids,
                treatment_train,
                outcome_train,
                source_names,
                random_state=args.random_state + 10_000 * outer_fold,
            )
            resolved_nuisance_source = "crossfit_logistic_stack"
        else:
            primary_train_rows = nuisance_rows_for_fold(
                nuisance_df,
                outer_fold,
                "train_inner_oof",
                args.nuisance_source,
            )
            primary_test_rows = nuisance_rows_for_fold(
                nuisance_df,
                outer_fold,
                "test_outer_train_fit",
                args.nuisance_source,
            )
            e_train, m_train = _aligned_source_arrays(primary_train_rows, train_ids)
            e_test, m_test = _aligned_source_arrays(primary_test_rows, test_ids)
            resolved_nuisance_source = args.nuisance_source
        primary = cohort_contrast(
            x_train,
            treatment_train,
            outcome_train,
            e_train,
            m_train,
            probability_clip=args.probability_clip,
        )
        tail_score = tail_contrast(
            x_train,
            primary.patient_contribution,
            quantile=args.tail_quantile,
        )

        source_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for source_name in source_names:
            rows = nuisance_rows_for_fold(
                nuisance_df,
                outer_fold,
                "train_inner_oof",
                source_name,
            )
            source_predictions[source_name] = _aligned_source_arrays(rows, train_ids)
        source_agreement, n_sources = source_sign_agreement(
            x_train,
            treatment_train,
            outcome_train,
            source_predictions,
            primary.z_score,
            probability_clip=args.probability_clip,
        )
        subsample_agreement, selection_frequency = stability_diagnostics(
            x_train,
            treatment_train,
            outcome_train,
            e_train,
            m_train,
            primary.z_score,
            repeats=args.stability_repeats,
            fraction=args.stability_fraction,
            top_pool=args.stability_top_pool,
            probability_clip=args.probability_clip,
            random_state=args.random_state + 10_000 * outer_fold,
        )
        document_frequency, treated_count, control_count = feature_support(
            x_train,
            treatment_train,
        )
        feature_frame = build_feature_frame(
            outer_fold=outer_fold,
            feature_names=feature_names,
            result=primary,
            tail_score=tail_score,
            source_agreement=source_agreement,
            subsample_agreement=subsample_agreement,
            selection_frequency=selection_frequency,
            document_frequency=document_frequency,
            treated_count=treated_count,
            control_count=control_count,
            min_arm_count=args.min_arm_count,
        )
        all_feature_frames.append(feature_frame)
        eligible_frame = feature_frame[feature_frame["eligible"]].sort_values(
            "ranking_score", ascending=False
        )
        if len(eligible_frame) < max(top_k_values):
            raise ValueError(
                f"fold {outer_fold} has only {len(eligible_frame)} eligible features, "
                f"less than requested top_k={max(top_k_values)}"
            )

        test_treatment = test_df[args.treatment_column].to_numpy(dtype=float)
        test_outcome = test_df[args.outcome_column].to_numpy(dtype=float)
        test_u = test_treatment - clip_probabilities(e_test, args.probability_clip)
        test_v = test_outcome - clip_probabilities(m_test, args.probability_clip)

        for top_k in top_k_values:
            selected = eligible_frame.head(top_k).copy()
            selected["top_k"] = int(top_k)
            all_selected_frames.append(selected)
            indices = selected["feature_index"].to_numpy(dtype=int)
            LOGGER.info("  fitting top_k=%s residual forest", top_k)
            point, lower, upper = fit_residual_causal_forest(
                x_train[:, indices].toarray(),
                x_test[:, indices].toarray(),
                primary.treatment_residual,
                primary.outcome_residual,
                args=args,
                random_state=args.random_state + 1_000 * outer_fold + top_k,
            )
            predictions = test_df.reset_index().copy()
            keep_columns = [
                "_oci_row_id",
                "patient_id",
                args.treatment_column,
                args.outcome_column,
                "true_treatment_prob",
                "true_y0_prob",
                "true_y1_prob",
                "true_ite_prob",
                "true_age",
                "true_pdl1_expression",
            ]
            predictions = predictions[
                [column for column in keep_columns if column in predictions.columns]
            ].copy()
            predictions["outer_fold"] = int(outer_fold)
            predictions["top_k"] = int(top_k)
            predictions["pred_ite_prob"] = point
            predictions["pred_ite_lower"] = lower
            predictions["pred_ite_upper"] = upper
            predictions["test_e_hat"] = e_test
            predictions["test_m_hat"] = m_test
            predictions["test_treatment_residual"] = test_u
            predictions["test_outcome_residual"] = test_v
            predictions["train_constant_effect"] = primary.tau_constant
            all_predictions.append(predictions)
            metrics = prediction_metrics(predictions)
            metrics.update(
                {
                    "outer_fold": int(outer_fold),
                    "top_k": int(top_k),
                    "n_train": int(len(train_ids)),
                    "n_test": int(len(test_ids)),
                    "vocabulary_size": int(len(feature_names)),
                    "n_nuisance_stability_sources": int(n_sources),
                    "nuisance_source": resolved_nuisance_source,
                    "nuisance_treatment_auroc": float(
                        roc_auc_score(treatment_train, e_train)
                    ),
                    "nuisance_treatment_brier": float(
                        brier_score_loss(treatment_train, e_train)
                    ),
                    "nuisance_outcome_auroc": float(
                        roc_auc_score(outcome_train, m_train)
                    ),
                    "nuisance_outcome_brier": float(
                        brier_score_loss(outcome_train, m_train)
                    ),
                    "train_constant_effect": float(primary.tau_constant),
                }
            )
            fold_metrics.append(metrics)

    feature_scores = pd.concat(all_feature_frames, ignore_index=True)
    selected_features = pd.concat(all_selected_frames, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metrics).sort_values(["top_k", "outer_fold"])

    aggregate_rows: List[Dict[str, float]] = []
    for top_k, group in predictions.groupby("top_k", sort=True):
        metrics = prediction_metrics(group)
        metrics["top_k"] = int(top_k)
        aggregate_rows.append(metrics)
    aggregate = pd.DataFrame(aggregate_rows).sort_values("top_k")

    stability = (
        selected_features.groupby(["top_k", "feature"], as_index=False)
        .agg(
            outer_fold_count=("outer_fold", "nunique"),
            mean_rank=("rank", "mean"),
            mean_z_score=("z_score", "mean"),
            mean_abs_z_score=("z_score", lambda values: float(np.mean(np.abs(values)))),
            mean_ranking_score=("ranking_score", "mean"),
        )
        .sort_values(["top_k", "outer_fold_count", "mean_ranking_score"], ascending=[True, False, False])
    )

    eligible_scores = feature_scores[feature_scores["eligible"]].copy()
    relevance = (
        eligible_scores.groupby("feature", as_index=False)
        .agg(
            outer_fold_count=("outer_fold", "nunique"),
            mean_z_score=("z_score", "mean"),
            mean_abs_z_score=("z_score", lambda values: float(np.mean(np.abs(values)))),
            positive_fold_fraction=("z_score", lambda values: float(np.mean(np.asarray(values) > 0))),
            mean_ranking_score=("ranking_score", "mean"),
            best_rank=("rank", "min"),
            mean_rank=("rank", "mean"),
            mean_document_frequency=("document_frequency", "mean"),
            mean_source_sign_agreement=("source_sign_agreement", "mean"),
            mean_subsample_sign_agreement=("subsample_sign_agreement", "mean"),
            mean_subsample_top_pool_frequency=("subsample_top_pool_frequency", "mean"),
        )
    )
    relevance["direction_consistency"] = np.maximum(
        relevance["positive_fold_fraction"],
        1.0 - relevance["positive_fold_fraction"],
    )
    relevance["stable_relevance_score"] = (
        relevance["mean_ranking_score"]
        * relevance["direction_consistency"]
        * relevance["outer_fold_count"]
        / float(len(outer_folds))
    )
    relevance = relevance.sort_values(
        ["stable_relevance_score", "outer_fold_count", "mean_ranking_score"],
        ascending=[False, False, False],
    )

    feature_scores.to_parquet(output_dir / "contrast_feature_scores.parquet", index=False)
    selected_features.to_csv(output_dir / "selected_features.csv", index=False)
    predictions.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    fold_metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    stability.to_csv(output_dir / "selection_stability.csv", index=False)
    relevance.to_csv(output_dir / "ngram_relevance_summary.csv", index=False)

    primary_metrics = aggregate[aggregate["top_k"] == args.primary_top_k].iloc[0].to_dict()
    primary_stable = stability[
        (stability["top_k"] == args.primary_top_k)
        & (stability["outer_fold_count"] >= args.stable_outer_fold_min)
    ].head(50)
    summary = {
        "primary_top_k": int(args.primary_top_k),
        "primary_metrics": {
            key: (None if pd.isna(value) else float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, (np.integer, int)) else value)
            for key, value in primary_metrics.items()
        },
        "stable_primary_features": primary_stable.to_dict(orient="records"),
        "elapsed_seconds": float(time.time() - start),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    LOGGER.info("Complete in %.1f seconds. Results: %s", time.time() - start, output_dir)
    print(aggregate.to_string(index=False))
    print("\nPrimary stable features")
    print(primary_stable.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--nuisance-predictions", default=DEFAULT_NUISANCE)
    parser.add_argument(
        "--nuisance-source",
        default="stacked",
        help=(
            "Use 'stacked' for an honest logistic stack of all individual nuisance "
            "views, or name one source from the nuisance artifact."
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--top-k", type=parse_int_list, default=parse_int_list("25,50,100,200,400"))
    parser.add_argument("--primary-top-k", type=int, default=100)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.98)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--min-arm-count", type=int, default=3)
    parser.add_argument("--probability-clip", type=float, default=0.02)
    parser.add_argument("--tail-quantile", type=float, default=0.20)
    parser.add_argument("--stability-repeats", type=int, default=30)
    parser.add_argument("--stability-fraction", type=float, default=0.75)
    parser.add_argument("--stability-top-pool", type=int, default=500)
    parser.add_argument("--stable-outer-fold-min", type=int, default=2)
    parser.add_argument("--forest-estimators", type=int, default=400)
    parser.add_argument("--forest-min-leaf", type=int, default=10)
    parser.add_argument("--forest-max-features", type=float, default=1.0)
    parser.add_argument("--forest-max-samples", type=float, default=0.45)
    parser.add_argument("--min-var-fraction-leaf", type=float, default=0.01)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.ngram_min < 1 or args.ngram_max < args.ngram_min:
        raise ValueError("invalid n-gram range")
    if args.min_df < 1 or args.max_features < 1:
        raise ValueError("min_df and max_features must be positive")
    if not 0.0 < args.max_df <= 1.0:
        raise ValueError("max_df must be in (0, 1]")
    if not 0.0 < args.probability_clip < 0.5:
        raise ValueError("probability_clip must be in (0, 0.5)")
    if not 0.0 < args.tail_quantile < 0.5:
        raise ValueError("tail_quantile must be in (0, 0.5)")
    if args.stability_repeats < 1:
        raise ValueError("stability_repeats must be positive")
    if not 0.0 < args.stability_fraction <= 1.0:
        raise ValueError("stability_fraction must be in (0, 1]")
    if args.forest_estimators < 4 or args.forest_estimators % 4 != 0:
        raise ValueError("forest_estimators must be positive and divisible by 4")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_parser().parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
