#!/usr/bin/env python
"""Train fresh cross-fitted BoW nuisances and benchmark probability calibration."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold

try:
    from one_off_tfidf_cohort_contrast.run_experiment import (
        DEFAULT_DATASET,
        DEFAULT_NUISANCE,
        _aligned_source_arrays,
        _logit_features,
        crossfit_logistic_stack,
        individual_nuisance_sources,
        normalize_text,
        nuisance_rows_for_fold,
        stacked_nuisance_predictions,
    )
except ModuleNotFoundError:
    # Also support direct execution from the repository root.
    from run_experiment import (
        DEFAULT_DATASET,
        DEFAULT_NUISANCE,
        _aligned_source_arrays,
        _logit_features,
        crossfit_logistic_stack,
        individual_nuisance_sources,
        normalize_text,
        nuisance_rows_for_fold,
        stacked_nuisance_predictions,
    )


LOGGER = logging.getLogger("fresh_bow_nuisance")
DEFAULT_OUTPUT = "one_off_tfidf_cohort_contrast/results_fresh_bow_nuisance"


@dataclass(frozen=True)
class ViewSpec:
    name: str
    ngram_range: Tuple[int, int]
    min_df: int
    max_features: int


VIEWS = (
    ViewSpec("word_1", (1, 1), 3, 20_000),
    ViewSpec("word_1_2", (1, 2), 3, 30_000),
    ViewSpec("word_1_3", (1, 3), 3, 40_000),
    ViewSpec("word_2_4", (2, 4), 3, 40_000),
)
REGULARIZATION_VALUES = (0.03, 0.10, 0.30, 1.00)


def make_vectorizer(view: ViewSpec) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)[a-z0-9%<>+=.-]+",
        ngram_range=view.ngram_range,
        min_df=view.min_df,
        max_df=0.98,
        max_features=view.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def fit_classifier(x, labels: np.ndarray, c_value: float) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value),
        solver="liblinear",
        max_iter=2000,
        random_state=17,
    )
    model.fit(x, np.asarray(labels, dtype=int))
    return model


def model_name(view: ViewSpec, c_value: float) -> str:
    c_text = str(c_value).replace(".", "p")
    return f"fresh_bow__{view.name}__c{c_text}__nuisance"


def fresh_base_predictions(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    outer_fold: int,
    random_state: int,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    names = [model_name(view, c_value) for view in VIEWS for c_value in REGULARIZATION_VALUES]
    train_e = {name: np.full(len(train_texts), np.nan, dtype=float) for name in names}
    train_m = {name: np.full(len(train_texts), np.nan, dtype=float) for name in names}
    splitter = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state + 1000 * outer_fold,
    )
    for inner_fold, (fit_idx, heldout_idx) in enumerate(splitter.split(train_texts), start=1):
        LOGGER.info("  inner fold %s/5", inner_fold)
        fit_texts = [train_texts[index] for index in fit_idx]
        heldout_texts = [train_texts[index] for index in heldout_idx]
        for view in VIEWS:
            vectorizer = make_vectorizer(view)
            x_fit = vectorizer.fit_transform(fit_texts)
            x_heldout = vectorizer.transform(heldout_texts)
            for c_value in REGULARIZATION_VALUES:
                name = model_name(view, c_value)
                treatment_model = fit_classifier(x_fit, treatment[fit_idx], c_value)
                outcome_model = fit_classifier(x_fit, outcome[fit_idx], c_value)
                train_e[name][heldout_idx] = treatment_model.predict_proba(x_heldout)[:, 1]
                train_m[name][heldout_idx] = outcome_model.predict_proba(x_heldout)[:, 1]

    test_e: Dict[str, np.ndarray] = {}
    test_m: Dict[str, np.ndarray] = {}
    for view in VIEWS:
        LOGGER.info("  fitting full outer-train view=%s", view.name)
        vectorizer = make_vectorizer(view)
        x_train = vectorizer.fit_transform(train_texts)
        x_test = vectorizer.transform(test_texts)
        for c_value in REGULARIZATION_VALUES:
            name = model_name(view, c_value)
            treatment_model = fit_classifier(x_train, treatment, c_value)
            outcome_model = fit_classifier(x_train, outcome, c_value)
            test_e[name] = treatment_model.predict_proba(x_test)[:, 1]
            test_m[name] = outcome_model.predict_proba(x_test)[:, 1]

    train = {name: (train_e[name], train_m[name]) for name in names}
    test = {name: (test_e[name], test_m[name]) for name in names}
    for name, (e_hat, m_hat) in train.items():
        if not np.all(np.isfinite(e_hat)) or not np.all(np.isfinite(m_hat)):
            raise RuntimeError(f"incomplete OOF predictions for {name}")
    return train, test


def stack_fresh_predictions(
    train_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = sorted(train_predictions)
    e_train_features = _logit_features(
        np.column_stack([train_predictions[name][0] for name in names])
    )
    m_train_features = _logit_features(
        np.column_stack([train_predictions[name][1] for name in names])
    )
    e_test_features = _logit_features(
        np.column_stack([test_predictions[name][0] for name in names])
    )
    m_test_features = _logit_features(
        np.column_stack([test_predictions[name][1] for name in names])
    )
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


def calibration_intercept_slope(labels: np.ndarray, probability: np.ndarray) -> Tuple[float, float]:
    logits = _logit_features(np.asarray(probability, dtype=float).reshape(-1, 1))
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logits, np.asarray(labels, dtype=int))
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def expected_calibration_error(
    labels: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    frame = pd.DataFrame({"label": labels, "probability": probability})
    frame["bin"] = pd.qcut(
        frame["probability"],
        q=min(bins, len(frame)),
        labels=False,
        duplicates="drop",
    )
    total = float(len(frame))
    value = 0.0
    for _, group in frame.groupby("bin", observed=True):
        value += len(group) / total * abs(group["label"].mean() - group["probability"].mean())
    return float(value)


def calibration_metrics(
    *,
    outer_fold: int,
    split_role: str,
    model: str,
    treatment: np.ndarray,
    outcome: np.ndarray,
    e_hat: np.ndarray,
    m_hat: np.ndarray,
    true_e: np.ndarray,
    true_m: np.ndarray,
    true_ite: np.ndarray,
) -> Dict[str, float]:
    e_intercept, e_slope = calibration_intercept_slope(treatment, e_hat)
    m_intercept, m_slope = calibration_intercept_slope(outcome, m_hat)
    u = np.asarray(treatment, dtype=float) - e_hat
    v = np.asarray(outcome, dtype=float) - m_hat
    tau = float(np.dot(u, v) / np.dot(u, u))
    return {
        "outer_fold": int(outer_fold),
        "split_role": split_role,
        "model": model,
        "n": int(len(treatment)),
        "treatment_auroc": float(roc_auc_score(treatment, e_hat)),
        "treatment_brier": float(brier_score_loss(treatment, e_hat)),
        "treatment_log_loss": float(log_loss(treatment, e_hat)),
        "treatment_ece": expected_calibration_error(treatment, e_hat),
        "treatment_calibration_intercept": e_intercept,
        "treatment_calibration_slope": e_slope,
        "treatment_true_probability_rmse": float(np.sqrt(np.mean(np.square(e_hat - true_e)))),
        "outcome_auroc": float(roc_auc_score(outcome, m_hat)),
        "outcome_brier": float(brier_score_loss(outcome, m_hat)),
        "outcome_log_loss": float(log_loss(outcome, m_hat)),
        "outcome_ece": expected_calibration_error(outcome, m_hat),
        "outcome_calibration_intercept": m_intercept,
        "outcome_calibration_slope": m_slope,
        "outcome_true_probability_rmse": float(np.sqrt(np.mean(np.square(m_hat - true_m)))),
        "residual_constant_effect": tau,
        "true_ate": float(np.mean(true_ite)),
    }


def append_prediction_rows(
    rows: List[pd.DataFrame],
    *,
    row_ids: np.ndarray,
    outer_fold: int,
    split_role: str,
    source_name: str,
    e_hat: np.ndarray,
    m_hat: np.ndarray,
) -> None:
    rows.append(
        pd.DataFrame(
            {
                "_oci_row_id": row_ids.astype(int),
                "outer_fold": int(outer_fold),
                "split_role": split_role,
                "source_name": source_name,
                "e_hat": e_hat,
                "m_hat": m_hat,
            }
        )
    )


def run(args: argparse.Namespace) -> None:
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.dataset).reset_index(drop=True)
    dataset["_oci_row_id"] = np.arange(len(dataset), dtype=int)
    old_predictions = pd.read_parquet(args.existing_nuisance_predictions)
    dataset_indexed = dataset.set_index("_oci_row_id")

    artifact_rows: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, float]] = []
    for outer_fold in sorted(int(value) for value in old_predictions["outer_fold"].dropna().unique()):
        LOGGER.info("Outer fold %s", outer_fold)
        anchor_train = nuisance_rows_for_fold(
            old_predictions,
            outer_fold,
            "train_inner_oof",
            "ensemble_mean_nuisance",
        )
        anchor_test = nuisance_rows_for_fold(
            old_predictions,
            outer_fold,
            "test_outer_train_fit",
            "ensemble_mean_nuisance",
        )
        train_ids = np.sort(anchor_train["_oci_row_id"].unique())
        test_ids = np.sort(anchor_test["_oci_row_id"].unique())
        train_df = dataset_indexed.loc[train_ids]
        test_df = dataset_indexed.loc[test_ids]
        train_texts = [normalize_text(value) for value in train_df["clinical_text"]]
        test_texts = [normalize_text(value) for value in test_df["clinical_text"]]
        treatment_train = train_df["treatment_indicator"].to_numpy(dtype=int)
        outcome_train = train_df["outcome_indicator"].to_numpy(dtype=int)

        fresh_train, fresh_test = fresh_base_predictions(
            train_texts,
            test_texts,
            treatment_train,
            outcome_train,
            outer_fold=outer_fold,
            random_state=args.random_state,
        )
        e_fresh_train, m_fresh_train, e_fresh_test, m_fresh_test = stack_fresh_predictions(
            fresh_train,
            fresh_test,
            treatment_train,
            outcome_train,
            random_state=args.random_state + 10_000 * outer_fold,
        )

        old_sources = individual_nuisance_sources(old_predictions, outer_fold)
        e_old_train, m_old_train, e_old_test, m_old_test = stacked_nuisance_predictions(
            old_predictions,
            outer_fold,
            train_ids,
            test_ids,
            treatment_train,
            outcome_train,
            old_sources,
            random_state=args.random_state + 20_000 * outer_fold,
        )
        e_mean_train, m_mean_train = _aligned_source_arrays(anchor_train, train_ids)
        e_mean_test, m_mean_test = _aligned_source_arrays(anchor_test, test_ids)

        for name, values in fresh_train.items():
            append_prediction_rows(
                artifact_rows,
                row_ids=train_ids,
                outer_fold=outer_fold,
                split_role="train_inner_oof",
                source_name=name,
                e_hat=values[0],
                m_hat=values[1],
            )
            test_values = fresh_test[name]
            append_prediction_rows(
                artifact_rows,
                row_ids=test_ids,
                outer_fold=outer_fold,
                split_role="test_outer_train_fit",
                source_name=name,
                e_hat=test_values[0],
                m_hat=test_values[1],
            )
        append_prediction_rows(
            artifact_rows,
            row_ids=train_ids,
            outer_fold=outer_fold,
            split_role="train_inner_oof",
            source_name="fresh_bow_stacked_nuisance",
            e_hat=e_fresh_train,
            m_hat=m_fresh_train,
        )
        append_prediction_rows(
            artifact_rows,
            row_ids=test_ids,
            outer_fold=outer_fold,
            split_role="test_outer_train_fit",
            source_name="fresh_bow_stacked_nuisance",
            e_hat=e_fresh_test,
            m_hat=m_fresh_test,
        )

        models = {
            "fresh_bow_stacked": (e_fresh_train, m_fresh_train, e_fresh_test, m_fresh_test),
            "reused_crossfit_stack": (e_old_train, m_old_train, e_old_test, m_old_test),
            "reused_probability_mean": (e_mean_train, m_mean_train, e_mean_test, m_mean_test),
        }
        for split_role, frame, row_ids in (
            ("train_inner_oof", train_df, train_ids),
            ("test_outer_train_fit", test_df, test_ids),
        ):
            treatment = frame["treatment_indicator"].to_numpy(dtype=int)
            outcome = frame["outcome_indicator"].to_numpy(dtype=int)
            true_e = frame["true_treatment_prob"].to_numpy(dtype=float)
            true_m = (
                true_e * frame["true_y1_prob"].to_numpy(dtype=float)
                + (1.0 - true_e) * frame["true_y0_prob"].to_numpy(dtype=float)
            )
            true_ite = frame["true_ite_prob"].to_numpy(dtype=float)
            for model, values in models.items():
                e_hat, m_hat = (values[0], values[1]) if split_role == "train_inner_oof" else (values[2], values[3])
                metric_rows.append(
                    calibration_metrics(
                        outer_fold=outer_fold,
                        split_role=split_role,
                        model=model,
                        treatment=treatment,
                        outcome=outcome,
                        e_hat=e_hat,
                        m_hat=m_hat,
                        true_e=true_e,
                        true_m=true_m,
                        true_ite=true_ite,
                    )
                )

        # Keep completed folds inspectable even if a later fold is interrupted.
        pd.concat(artifact_rows, ignore_index=True).to_parquet(
            output_dir / "fresh_nuisance_predictions.parquet",
            index=False,
        )
        pd.DataFrame(metric_rows).to_csv(output_dir / "calibration_by_fold.csv", index=False)
        (output_dir / "run_status.json").write_text(
            json.dumps(
                {
                    "complete": False,
                    "completed_outer_folds": sorted(
                        int(value)
                        for value in pd.DataFrame(metric_rows)["outer_fold"].unique()
                    ),
                    "elapsed_seconds": float(time.time() - start),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    nuisance_artifact = pd.concat(artifact_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    aggregate = (
        metrics.groupby(["split_role", "model"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["outer_fold"], errors="ignore")
    )
    nuisance_artifact.to_parquet(output_dir / "fresh_nuisance_predictions.parquet", index=False)
    metrics.to_csv(output_dir / "calibration_by_fold.csv", index=False)
    aggregate.to_csv(output_dir / "calibration_aggregate.csv", index=False)
    config = {
        "args": vars(args),
        "views": [view.__dict__ for view in VIEWS],
        "regularization_values": list(REGULARIZATION_VALUES),
        "elapsed_seconds": float(time.time() - start),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output_dir / "run_status.json").write_text(
        json.dumps(
            {
                "complete": True,
                "completed_outer_folds": sorted(
                    int(value) for value in metrics["outer_fold"].unique()
                ),
                "elapsed_seconds": float(time.time() - start),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    LOGGER.info("Complete in %.1f seconds", time.time() - start)
    print(aggregate.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--existing-nuisance-predictions", default=DEFAULT_NUISANCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
