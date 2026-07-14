#!/usr/bin/env python
"""One-fold oracle-assisted diagnostic for the TF-IDF/NMF topic forest.

Oracle covariates from the outer-training rows are used only to select topics
that encode the known confounders and effect modifiers. The selected topic
scores, rather than the oracle covariates themselves, are passed to the causal
forest. Outer-held-out oracle values are used only after predictions are saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.models.causal_forest_head import CausalForestHead  # noqa: E402

TRUE_CONFOUNDERS = (
    "true_age",
    "true_sex",
    "true_ecog_performance_status",
    "true_creatinine_clearance",
    "true_prior_platinum_therapy",
)
TRUE_MODIFIERS = (
    "true_histology_type",
    "true_egfr_mutation_status",
    "true_baseline_nlr",
    "true_brain_metastases_status",
    "true_baseline_hemoglobin",
)


def categorical_eta(values: np.ndarray, categories: Sequence[Any]) -> float:
    values = np.asarray(values, dtype=float)
    categories = pd.Series(categories).fillna("<missing>").astype(str).to_numpy()
    center = float(np.mean(values))
    denominator = float(np.sum(np.square(values - center)))
    if denominator <= 1e-12:
        return 0.0
    between = 0.0
    for category in np.unique(categories):
        group = values[categories == category]
        between += len(group) * float(np.square(np.mean(group) - center))
    return float(np.sqrt(max(0.0, min(1.0, between / denominator))))


def association_strengths(topic_scores: np.ndarray, target: pd.Series) -> np.ndarray:
    strengths = np.zeros(topic_scores.shape[1], dtype=float)
    numeric = pd.api.types.is_numeric_dtype(target)
    for topic_index in range(topic_scores.shape[1]):
        values = topic_scores[:, topic_index]
        if numeric:
            statistic = spearmanr(values, target.to_numpy(dtype=float)).statistic
            strengths[topic_index] = abs(float(statistic)) if np.isfinite(statistic) else 0.0
        else:
            strengths[topic_index] = categorical_eta(values, target)
    return strengths


def _read_handoff(path: Path, outer_fold: int) -> Dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    matches = [
        row
        for row in rows
        if row.get("scope") == "full_outer_train"
        and int(row.get("outer_fold", -1)) == int(outer_fold)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one full_outer_train context for fold {outer_fold}; found {len(matches)}"
        )
    return matches[0]


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: np.asarray(archive[name], dtype=float) for name in archive.files}


def _standardize(
    fit_values: np.ndarray,
    heldout_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(fit_values, axis=0)
    scales = np.std(fit_values, axis=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    return (
        ((fit_values - means) / scales).astype(np.float32),
        ((heldout_values - means) / scales).astype(np.float32),
        means,
        scales,
    )


def _ordered_nuisance(
    frame: pd.DataFrame,
    row_ids: Sequence[int],
    prediction_scope: str,
) -> pd.DataFrame:
    selected = frame[frame["prediction_scope"] == prediction_scope].set_index("_oci_row_id")
    return selected.loc[[int(value) for value in row_ids]].reset_index()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _metrics(truth: np.ndarray, estimate: np.ndarray) -> Dict[str, float]:
    error = estimate - truth
    return {
        "pearson_correlation": float(np.corrcoef(truth, estimate)[0, 1]),
        "spearman_correlation": float(spearmanr(truth, estimate).statistic),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "estimated_ate": float(np.mean(estimate)),
        "oracle_ate": float(np.mean(truth)),
        "ate_bias": float(np.mean(error)),
        "estimated_ite_standard_deviation": float(np.std(estimate)),
        "oracle_ite_standard_deviation": float(np.std(truth)),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.dataset).reset_index(drop=True)
    dataset["_oci_row_id"] = np.arange(len(dataset), dtype=int)
    context = _read_handoff(Path(args.stage1_handoff), args.outer_fold)
    fit_ids = [int(value) for value in context["fit_row_ids"]]
    heldout_ids = [int(value) for value in context["heldout_row_ids"]]
    indexed = dataset.set_index("_oci_row_id", drop=False)
    fit_data = indexed.loc[fit_ids].reset_index(drop=True)
    heldout_data = indexed.loc[heldout_ids].reset_index(drop=True)
    artifacts = context["discovery"]["artifacts"]
    fit_scores = _load_npz(Path(artifacts["fit_topic_values"]))
    heldout_scores = _load_npz(Path(artifacts["heldout_topic_values"]))

    variables_by_bank = {
        "treatment": TRUE_CONFOUNDERS,
        "outcome": TRUE_CONFOUNDERS,
        "effect": TRUE_MODIFIERS,
    }
    selected_by_bank: Dict[str, List[int]] = {}
    selection_rows: List[Dict[str, Any]] = []
    topic_banks = context["discovery"]["topic_banks"]
    for bank, variables in variables_by_bank.items():
        selected = set()
        topics = topic_banks[bank]["topics"]
        for variable in variables:
            strengths = association_strengths(fit_scores[bank], fit_data[variable])
            order = np.argsort(-strengths, kind="stable")[: args.top_topics_per_variable]
            for rank, topic_index in enumerate(order, start=1):
                topic_index = int(topic_index)
                selected.add(topic_index)
                selection_rows.append(
                    {
                        "bank": bank,
                        "model_role": "X_heterogeneity" if bank == "effect" else "W_adjustment",
                        "oracle_variable": variable,
                        "within_variable_rank": rank,
                        "association_kind": (
                            "absolute_spearman"
                            if pd.api.types.is_numeric_dtype(fit_data[variable])
                            else "categorical_eta"
                        ),
                        "association_strength": float(strengths[topic_index]),
                        "topic_index": topic_index,
                        "topic_id": topics[topic_index]["topic_id"],
                        "terms": topics[topic_index].get("terms", []),
                    }
                )
        selected_by_bank[bank] = sorted(selected)

    transformed_fit: Dict[str, np.ndarray] = {}
    transformed_heldout: Dict[str, np.ndarray] = {}
    preprocessing: Dict[str, Any] = {}
    for bank, selected in selected_by_bank.items():
        fit_selected = fit_scores[bank][:, selected]
        heldout_selected = heldout_scores[bank][:, selected]
        fit_matrix, heldout_matrix, means, scales = _standardize(fit_selected, heldout_selected)
        transformed_fit[bank] = fit_matrix
        transformed_heldout[bank] = heldout_matrix
        preprocessing[bank] = {
            "selected_indices": selected,
            "topic_ids": [topic_banks[bank]["topics"][index]["topic_id"] for index in selected],
            "means": means,
            "scales": scales,
        }

    x_fit = transformed_fit["effect"]
    x_heldout = transformed_heldout["effect"]
    w_fit = np.column_stack([transformed_fit["treatment"], transformed_fit["outcome"]])
    forest = CausalForestHead(
        n_estimators=args.cf_n_estimators,
        max_depth=args.cf_max_depth,
        min_samples_leaf=args.cf_min_samples_leaf,
        max_features=args.cf_max_features,
        honest=True,
        inference=True,
        random_state=args.seed + args.outer_fold,
        tune_model=False,
    )
    forest.fit(
        x_fit,
        fit_data[args.treatment_column].to_numpy(dtype=float),
        fit_data[args.outcome_column].to_numpy(dtype=float),
        W=w_fit,
    )
    forest_result = forest.predict(x_heldout, return_ci=True)
    tau = np.asarray(forest_result["tau_pred"], dtype=float)

    nuisance = pd.read_parquet(artifacts["nuisance_predictions"])
    fit_nuisance = _ordered_nuisance(nuisance, fit_ids, "fit_oof")
    heldout_nuisance = _ordered_nuisance(nuisance, heldout_ids, "external_heldout")
    propensity = heldout_nuisance["treatment_stacked"].to_numpy(dtype=float)
    outcome_prediction = heldout_nuisance["outcome_stacked"].to_numpy(dtype=float)
    y0 = np.clip(outcome_prediction - propensity * tau, 0.0, 1.0)
    y1 = np.clip(outcome_prediction + (1.0 - propensity) * tau, 0.0, 1.0)

    predictions = heldout_data[
        ["_oci_row_id", "patient_id", args.treatment_column, args.outcome_column]
    ].copy()
    predictions["pred_ite_prob"] = tau
    predictions["pred_y0_prob"] = y0
    predictions["pred_y1_prob"] = y1
    predictions["pred_propensity_prob"] = propensity
    predictions["pred_outcome_prob"] = outcome_prediction
    predictions["pred_ite_lower"] = forest_result["tau_lower"]
    predictions["pred_ite_upper"] = forest_result["tau_upper"]
    predictions["outer_fold"] = args.outer_fold
    predictions["oracle_assisted_topic_selection"] = True
    predictions["heldout_oracle_used_for_selection_or_fitting"] = False
    prediction_path = output_dir / "predictions_without_heldout_oracle.parquet"
    predictions.to_parquet(prediction_path, index=False)
    frozen_hash = _sha256(prediction_path)

    evaluated = predictions.merge(
        heldout_data[["_oci_row_id", "true_ite_prob"]],
        on="_oci_row_id",
        validate="one_to_one",
    )
    evaluated.to_parquet(output_dir / "posthoc_predictions_with_oracle.parquet", index=False)
    oracle_metrics = _metrics(
        evaluated["true_ite_prob"].to_numpy(dtype=float),
        evaluated["pred_ite_prob"].to_numpy(dtype=float),
    )
    heldout_treatment = heldout_data[args.treatment_column].to_numpy(dtype=float)
    heldout_outcome = heldout_data[args.outcome_column].to_numpy(dtype=float)
    heldout_u = heldout_treatment - propensity
    heldout_v = heldout_outcome - outcome_prediction
    fit_u = fit_data[args.treatment_column].to_numpy(dtype=float) - fit_nuisance[
        "treatment_stacked"
    ].to_numpy(dtype=float)
    fit_v = fit_data[args.outcome_column].to_numpy(dtype=float) - fit_nuisance[
        "outcome_stacked"
    ].to_numpy(dtype=float)
    heldout_pseudo_target = heldout_v / heldout_u
    fit_pseudo_target = fit_v / fit_u
    train_fitted_constant_effect = float(np.dot(fit_u, fit_v) / np.dot(fit_u, fit_u))
    train_fitted_pseudo_target_mean = float(np.mean(fit_pseudo_target))
    r_loss = float(np.mean(np.square(heldout_v - heldout_u * tau)))
    constant_r_loss = float(
        np.mean(np.square(heldout_v - heldout_u * train_fitted_constant_effect))
    )
    pseudo_target_mse = float(np.mean(np.square(tau - heldout_pseudo_target)))
    pseudo_target_correlation = float(np.corrcoef(tau, heldout_pseudo_target)[0, 1])
    oracle_metrics.update(
        {
            "n": int(len(evaluated)),
            "outer_fold": int(args.outer_fold),
            "n_x_effect_topics": int(x_fit.shape[1]),
            "n_w_treatment_topics": int(transformed_fit["treatment"].shape[1]),
            "n_w_outcome_topics": int(transformed_fit["outcome"].shape[1]),
            "selection_top_topics_per_oracle_variable": int(args.top_topics_per_variable),
            "r_loss_with_stage1_nuisance": r_loss,
            "train_fitted_constant_effect": train_fitted_constant_effect,
            "train_fitted_constant_r_loss": constant_r_loss,
            "r_loss_change_from_train_fitted_constant": r_loss - constant_r_loss,
            "pseudo_target_mse": pseudo_target_mse,
            "tau_hat_pseudo_target_correlation": pseudo_target_correlation,
            "pseudo_target_standard_deviation": float(np.std(heldout_pseudo_target)),
            "pseudo_target_maximum_absolute_value": float(np.max(np.abs(heldout_pseudo_target))),
            "pseudo_target_zero_effect_mse": float(np.mean(np.square(heldout_pseudo_target))),
            "train_fitted_pseudo_target_mean": train_fitted_pseudo_target_mean,
            "train_fitted_pseudo_target_mean_mse": float(
                np.mean(np.square(heldout_pseudo_target - train_fitted_pseudo_target_mean))
            ),
            "pseudo_target_true_ite_correlation": float(
                np.corrcoef(
                    heldout_pseudo_target,
                    evaluated["true_ite_prob"].to_numpy(dtype=float),
                )[0, 1]
            ),
            "frozen_prediction_sha256": frozen_hash,
        }
    )
    if args.baseline_predictions:
        baseline = pd.read_parquet(args.baseline_predictions)
        baseline = baseline[baseline["outer_fold"] == args.outer_fold].merge(
            heldout_data[["_oci_row_id", "true_ite_prob"]],
            on="_oci_row_id",
            validate="one_to_one",
        )
        baseline_metrics = _metrics(
            baseline["true_ite_prob"].to_numpy(dtype=float),
            baseline["pred_ite_prob"].to_numpy(dtype=float),
        )
        baseline_tau = baseline["pred_ite_prob"].to_numpy(dtype=float)
        baseline_metrics["r_loss_with_stage1_nuisance"] = float(
            np.mean(np.square(heldout_v - heldout_u * baseline_tau))
        )
        baseline_metrics["pseudo_target_mse"] = float(
            np.mean(np.square(baseline_tau - heldout_pseudo_target))
        )
        baseline_metrics["tau_hat_pseudo_target_correlation"] = float(
            np.corrcoef(baseline_tau, heldout_pseudo_target)[0, 1]
        )
        oracle_metrics["all_topic_fold_baseline"] = baseline_metrics
        oracle_metrics["pearson_change_from_all_topics"] = float(
            oracle_metrics["pearson_correlation"] - baseline_metrics["pearson_correlation"]
        )
        oracle_metrics["r_loss_change_from_all_topics"] = float(
            r_loss - baseline_metrics["r_loss_with_stage1_nuisance"]
        )
        oracle_metrics["pseudo_target_mse_change_from_all_topics"] = float(
            pseudo_target_mse - baseline_metrics["pseudo_target_mse"]
        )

    pd.DataFrame(selection_rows).to_json(
        output_dir / "oracle_topic_selection.jsonl", orient="records", lines=True
    )
    joblib.dump(forest, output_dir / "causal_forest.joblib")
    joblib.dump(preprocessing, output_dir / "topic_preprocessing.joblib")
    _write_json(output_dir / "posthoc_oracle_metrics.json", oracle_metrics)
    _write_json(
        output_dir / "run_config.json",
        {
            "outer_fold": args.outer_fold,
            "top_topics_per_variable": args.top_topics_per_variable,
            "true_confounders": TRUE_CONFOUNDERS,
            "true_effect_modifiers": TRUE_MODIFIERS,
            "selected_topic_indices": selected_by_bank,
            "selection_rows_are_outer_training_only": True,
            "heldout_oracle_used_only_after_prediction_freeze": True,
            "frozen_prediction_sha256": frozen_hash,
        },
    )
    return oracle_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage1-handoff", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outer-fold", type=int, default=1)
    parser.add_argument("--top-topics-per-variable", type=int, default=3)
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--cf-n-estimators", type=int, default=200)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=10)
    parser.add_argument("--cf-max-depth", type=int, default=None)
    parser.add_argument("--cf-max-features", default="sqrt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-predictions", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
