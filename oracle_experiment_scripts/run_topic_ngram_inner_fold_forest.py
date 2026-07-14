#!/usr/bin/env python
"""Evaluate one topic's top n-grams as causal-forest heterogeneity features.

This is an oracle-assisted diagnostic for synthetic data.  Topic selection uses
only the requested inner-fit rows.  The causal forest is trained on those rows,
and R-loss is evaluated once on the exact inner-held-out rows using Stage-1's
external held-out nuisance predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.inference.tfidf_topic_discovery import _normalize_texts  # noqa: E402
from oci.models.causal_forest_head import CausalForestHead  # noqa: E402


def _context(path: Path, outer_fold: int, inner_fold: int) -> Dict[str, Any]:
    matches = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("scope") == "candidate_selection_inner_fit"
                and int(row.get("outer_fold", -1)) == outer_fold
                and int(row.get("inner_fold", -1)) == inner_fold
            ):
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one candidate-selection context for "
            f"outer={outer_fold}, inner={inner_fold}; found {len(matches)}"
        )
    return matches[0]


def _ordered_nuisance(
    frame: pd.DataFrame,
    row_ids: Sequence[int],
    prediction_scope: str,
) -> pd.DataFrame:
    selected = frame.loc[frame["prediction_scope"] == prediction_scope].copy()
    selected["_oci_row_id"] = selected["_oci_row_id"].astype(int)
    selected = selected.set_index("_oci_row_id", drop=False)
    ordered = selected.loc[[int(value) for value in row_ids]].reset_index(drop=True)
    if len(ordered) != len(row_ids):
        raise ValueError(f"Missing {prediction_scope} nuisance rows")
    return ordered


def _select_topic(
    scores: np.ndarray,
    target: pd.Series,
    *,
    positive_category: str,
) -> tuple[int, np.ndarray, str]:
    if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 2:
        strengths = []
        for index in range(scores.shape[1]):
            value = spearmanr(scores[:, index], target.to_numpy(dtype=float)).statistic
            strengths.append(abs(float(value)) if np.isfinite(value) else 0.0)
        return int(np.argmax(strengths)), np.asarray(strengths), "absolute_spearman"

    normalized = target.fillna("<missing>").astype(str).str.lower()
    positive = normalized.eq(str(positive_category).lower()).astype(int).to_numpy()
    if len(np.unique(positive)) != 2:
        raise ValueError(
            f"Positive category {positive_category!r} does not define a binary target; "
            f"observed categories are {sorted(normalized.unique())}"
        )
    strengths = []
    for index in range(scores.shape[1]):
        auc = float(roc_auc_score(positive, scores[:, index]))
        strengths.append(max(auc, 1.0 - auc))
    return int(np.argmax(strengths)), np.asarray(strengths), "orientation_free_auc"


def _paired_loss_metrics(
    constant_loss: np.ndarray,
    forest_loss: np.ndarray,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> Dict[str, Any]:
    differences = np.asarray(constant_loss - forest_loss, dtype=float)
    absolute = float(np.mean(differences))
    constant_mean = float(np.mean(constant_loss))
    forest_mean = float(np.mean(forest_loss))
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
    if standard_error > 0:
        z_value = absolute / standard_error
        one_sided_p = float(norm.sf(z_value))
    else:
        z_value = float("nan")
        one_sided_p = float("nan")

    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_repeats, dtype=float)
    for repeat in range(bootstrap_repeats):
        indices = rng.integers(0, len(differences), size=len(differences))
        bootstrap[repeat] = float(np.mean(differences[indices]))
    bootstrap_low, bootstrap_high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "constant_r_loss": constant_mean,
        "forest_r_loss": forest_mean,
        "absolute_r_loss_reduction": absolute,
        "relative_r_loss_reduction": (
            float(absolute / constant_mean) if constant_mean > 0 else float("nan")
        ),
        "paired_standard_error": standard_error,
        "paired_wald_95_ci": [
            float(absolute - 1.96 * standard_error),
            float(absolute + 1.96 * standard_error),
        ],
        "conditional_patient_bootstrap_95_ci": [
            float(bootstrap_low),
            float(bootstrap_high),
        ],
        "one_sided_p_for_positive_reduction": one_sided_p,
        "paired_z_value": float(z_value),
        "positive_loss_difference_fraction": float(np.mean(differences > 0)),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = _context(Path(args.stage1_handoff), args.outer_fold, args.inner_fold)
    discovery = context["discovery"]
    artifacts = discovery["artifacts"]
    fit_ids = [int(value) for value in context["fit_row_ids"]]
    heldout_ids = [int(value) for value in context["heldout_row_ids"]]

    dataset = pd.read_parquet(args.dataset).reset_index(drop=True)
    dataset["_oci_row_id"] = np.arange(len(dataset), dtype=int)
    indexed = dataset.set_index("_oci_row_id", drop=False)
    fit_data = indexed.loc[fit_ids].reset_index(drop=True)
    heldout_data = indexed.loc[heldout_ids].reset_index(drop=True)

    with np.load(artifacts["fit_topic_values"]) as archive:
        effect_scores = np.asarray(archive["effect"], dtype=float)
    topic_index, strengths, association_kind = _select_topic(
        effect_scores,
        fit_data[args.oracle_modifier_column],
        positive_category=args.positive_category,
    )
    topics = discovery["topic_banks"]["effect"]["topics"]
    topic = topics[topic_index]
    term_records = list(topic.get("terms") or [])[: args.terms_per_topic]
    if len(term_records) != args.terms_per_topic:
        raise ValueError(
            f"Topic {topic.get('topic_id')} has {len(term_records)} terms; "
            f"expected {args.terms_per_topic}"
        )
    terms = [str(record["term"]) for record in term_records]

    fitted_context = joblib.load(artifacts["fitted_context"])
    vectorizer = fitted_context.common_vectorizer
    missing_terms = [term for term in terms if term not in vectorizer.vocabulary_]
    if missing_terms:
        raise ValueError(f"Topic terms are absent from the fitted vocabulary: {missing_terms}")
    term_indices = [int(vectorizer.vocabulary_[term]) for term in terms]
    fit_common = vectorizer.transform(
        _normalize_texts(fit_data[args.text_column].fillna(""))
    )
    heldout_common = vectorizer.transform(
        _normalize_texts(heldout_data[args.text_column].fillna(""))
    )
    x_fit = np.asarray(fit_common[:, term_indices].toarray(), dtype=np.float32)
    x_heldout = np.asarray(heldout_common[:, term_indices].toarray(), dtype=np.float32)

    nuisance = pd.read_parquet(artifacts["nuisance_predictions"])
    fit_nuisance = _ordered_nuisance(nuisance, fit_ids, "fit_oof")
    heldout_nuisance = _ordered_nuisance(nuisance, heldout_ids, "external_heldout")
    nuisance_columns = ["treatment_stacked", "outcome_stacked"]
    w_fit = fit_nuisance[nuisance_columns].to_numpy(dtype=np.float32)

    treatment_fit = fit_data[args.treatment_column].to_numpy(dtype=float)
    outcome_fit = fit_data[args.outcome_column].to_numpy(dtype=float)
    forest = CausalForestHead(
        n_estimators=args.cf_n_estimators,
        max_depth=args.cf_max_depth,
        min_samples_leaf=args.cf_min_samples_leaf,
        max_features=args.cf_max_features,
        honest=True,
        inference=True,
        random_state=args.seed,
        tune_model=False,
    )
    forest.fit(x_fit, treatment_fit, outcome_fit, W=w_fit)
    forest_predictions = forest.predict(x_heldout, return_ci=True)
    tau = np.asarray(forest_predictions["tau_pred"], dtype=float)

    fit_u = treatment_fit - fit_nuisance["treatment_stacked"].to_numpy(dtype=float)
    fit_v = outcome_fit - fit_nuisance["outcome_stacked"].to_numpy(dtype=float)
    constant_effect = float(np.dot(fit_u, fit_v) / np.dot(fit_u, fit_u))

    treatment_heldout = heldout_data[args.treatment_column].to_numpy(dtype=float)
    outcome_heldout = heldout_data[args.outcome_column].to_numpy(dtype=float)
    heldout_e = heldout_nuisance["treatment_stacked"].to_numpy(dtype=float)
    heldout_m = heldout_nuisance["outcome_stacked"].to_numpy(dtype=float)
    heldout_u = treatment_heldout - heldout_e
    heldout_v = outcome_heldout - heldout_m
    constant_loss = np.square(heldout_v - heldout_u * constant_effect)
    forest_loss = np.square(heldout_v - heldout_u * tau)
    loss_metrics = _paired_loss_metrics(
        constant_loss,
        forest_loss,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 1,
    )

    predictions = pd.DataFrame(
        {
            "_oci_row_id": heldout_ids,
            "patient_id": heldout_data["patient_id"].to_numpy(),
            "treatment": treatment_heldout,
            "outcome": outcome_heldout,
            "propensity_oof": heldout_e,
            "outcome_prediction_oof": heldout_m,
            "treatment_residual": heldout_u,
            "outcome_residual": heldout_v,
            "constant_effect": constant_effect,
            "forest_cate": tau,
            "constant_r_loss": constant_loss,
            "forest_r_loss": forest_loss,
            "paired_r_loss_reduction": constant_loss - forest_loss,
        }
    )
    if "tau_lower" in forest_predictions:
        predictions["forest_cate_lower"] = forest_predictions["tau_lower"]
        predictions["forest_cate_upper"] = forest_predictions["tau_upper"]
    predictions.to_parquet(output_dir / "heldout_predictions.parquet", index=False)
    joblib.dump(forest, output_dir / "causal_forest.joblib")

    oracle_metrics: Dict[str, float] = {}
    if "true_ite_prob" in heldout_data:
        truth = heldout_data["true_ite_prob"].to_numpy(dtype=float)
        oracle_metrics = {
            "posthoc_cate_pearson": float(np.corrcoef(tau, truth)[0, 1]),
            "posthoc_cate_spearman": float(spearmanr(tau, truth).statistic),
            "posthoc_oracle_ite_sd": float(np.std(truth)),
        }

    support_fit = np.sum(x_fit > 0, axis=0).astype(int)
    support_heldout = np.sum(x_heldout > 0, axis=0).astype(int)
    result = {
        "scope": {
            "outer_fold": int(args.outer_fold),
            "inner_fold": int(args.inner_fold),
            "scope_id": discovery["scope_id"],
            "fit_n": int(len(fit_data)),
            "heldout_n": int(len(heldout_data)),
        },
        "selection": {
            "oracle_assisted": True,
            "heldout_labels_used_for_selection": False,
            "oracle_modifier_column": args.oracle_modifier_column,
            "positive_category": args.positive_category,
            "association_kind": association_kind,
            "selected_topic_index_zero_based": int(topic_index),
            "selected_topic_id": topic["topic_id"],
            "training_association_strength": float(strengths[topic_index]),
            "terms": [
                {
                    **record,
                    "fit_document_support": int(support_fit[index]),
                    "heldout_document_support": int(support_heldout[index]),
                }
                for index, record in enumerate(term_records)
            ],
        },
        "forest": {
            "heterogeneity_feature_count": int(x_fit.shape[1]),
            "heterogeneity_features": terms,
            "adjustment_inputs": nuisance_columns,
            "n_estimators": int(args.cf_n_estimators),
            "max_depth": args.cf_max_depth,
            "min_samples_leaf": int(args.cf_min_samples_leaf),
            "max_features": args.cf_max_features,
            "honest": True,
            "tune_model": False,
            "random_state": int(args.seed),
            "train_fitted_constant_effect": constant_effect,
            "heldout_cate_mean": float(np.mean(tau)),
            "heldout_cate_sd": float(np.std(tau)),
        },
        "r_loss_evaluation": loss_metrics,
        "oracle_posthoc_evaluation": oracle_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage1-handoff", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outer-fold", type=int, default=1)
    parser.add_argument("--inner-fold", type=int, default=1)
    parser.add_argument(
        "--oracle-modifier-column", default="true_brain_metastases_status"
    )
    parser.add_argument("--positive-category", default="Yes")
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--terms-per-topic", type=int, default=15)
    parser.add_argument("--cf-n-estimators", type=int, default=200)
    parser.add_argument("--cf-max-depth", type=int, default=None)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=10)
    parser.add_argument("--cf-max-features", default="sqrt")
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1043)
    return parser.parse_args()


if __name__ == "__main__":
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, default=str))
