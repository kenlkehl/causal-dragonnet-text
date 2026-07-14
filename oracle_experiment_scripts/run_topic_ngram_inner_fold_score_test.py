#!/usr/bin/env python
"""Held-out group score test for one NMF topic's top TF-IDF n-grams.

The test requires nuisance models and a fit-side constant treatment effect, but
does not fit a CATE model or create patient-level pseudo-outcomes. It evaluates
whether the topic's n-grams are jointly associated with residualized treatment-
effect contributions on an exact inner-held-out split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.inference.tfidf_topic_discovery import _normalize_texts  # noqa: E402
from oracle_experiment_scripts.run_topic_ngram_inner_fold_forest import (  # noqa: E402
    _context,
    _ordered_nuisance,
    _select_topic,
)


def _multiplier_test(
    row_scores: np.ndarray,
    *,
    repeats: int,
    chunk_size: int,
    seed: int,
) -> Dict[str, Any]:
    scores = np.asarray(row_scores, dtype=float)
    n_rows = scores.shape[0]
    means = np.mean(scores, axis=0)
    scales = np.std(scores, axis=0, ddof=1)
    retained = scales > 1e-12
    scores = scores[:, retained]
    means = means[retained]
    scales = scales[retained]
    if not scores.shape[1]:
        raise ValueError("Every topic score contribution is constant")

    score_vector = np.sqrt(n_rows) * means
    covariance = np.atleast_2d(np.cov(scores, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = max(float(np.max(eigenvalues)) * 1e-8, 1e-12)
    nonzero = eigenvalues > tolerance
    covariance_rank = int(np.sum(nonzero))
    inverse = (eigenvectors[:, nonzero] / eigenvalues[nonzero]) @ eigenvectors[
        :, nonzero
    ].T
    quadratic = float(score_vector @ inverse @ score_vector)
    standardized = score_vector / scales
    maximum = float(np.max(np.abs(standardized)))

    centered = scores - means
    root_n = np.sqrt(n_rows)
    rng = np.random.default_rng(seed)
    quadratic_null = np.empty(repeats, dtype=float)
    maximum_null = np.empty(repeats, dtype=float)
    for start in range(0, repeats, chunk_size):
        stop = min(repeats, start + chunk_size)
        multipliers = rng.choice(
            np.asarray([-1.0, 1.0]), size=(stop - start, n_rows)
        )
        bootstrap_scores = multipliers @ centered / root_n
        quadratic_null[start:stop] = np.einsum(
            "bi,ij,bj->b", bootstrap_scores, inverse, bootstrap_scores
        )
        maximum_null[start:stop] = np.max(
            np.abs(bootstrap_scores / scales), axis=1
        )

    quadratic_p = float((1 + np.sum(quadratic_null >= quadratic)) / (repeats + 1))
    maximum_p = float((1 + np.sum(maximum_null >= maximum)) / (repeats + 1))
    return {
        "retained_columns": retained,
        "column_means": means,
        "column_standard_deviations": scales,
        "column_standardized_scores": standardized,
        "quadratic_statistic": quadratic,
        "quadratic_covariance_rank": covariance_rank,
        "quadratic_statistic_per_rank": float(quadratic / covariance_rank),
        "quadratic_asymptotic_chi_square_p": float(
            chi2.sf(quadratic, covariance_rank)
        ),
        "quadratic_multiplier_p": quadratic_p,
        "quadratic_null_95th_percentile": float(
            np.quantile(quadratic_null, 0.95)
        ),
        "maximum_absolute_standardized_score": maximum,
        "maximum_multiplier_p": maximum_p,
        "maximum_null_95th_percentile": float(np.quantile(maximum_null, 0.95)),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
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
        effect_topics = np.asarray(archive["effect"], dtype=float)
    topic_index, associations, association_kind = _select_topic(
        effect_topics,
        fit_data[args.oracle_modifier_column],
        positive_category=args.positive_category,
    )
    topic = discovery["topic_banks"]["effect"]["topics"][topic_index]
    term_records = list(topic["terms"])[: args.terms_per_topic]
    if len(term_records) != args.terms_per_topic:
        raise ValueError(
            f"Topic {topic['topic_id']} has {len(term_records)} terms; "
            f"expected {args.terms_per_topic}"
        )
    terms = [str(record["term"]) for record in term_records]

    fitted_context = joblib.load(artifacts["fitted_context"])
    vectorizer = fitted_context.common_vectorizer
    term_indices = [int(vectorizer.vocabulary_[term]) for term in terms]
    fit_matrix = vectorizer.transform(
        _normalize_texts(fit_data[args.text_column].fillna(""))
    )[:, term_indices].toarray()
    heldout_matrix = vectorizer.transform(
        _normalize_texts(heldout_data[args.text_column].fillna(""))
    )[:, term_indices].toarray()
    fit_means = np.mean(fit_matrix, axis=0)
    fit_scales = np.std(fit_matrix, axis=0)
    fit_scales = np.where(fit_scales > 1e-12, fit_scales, 1.0)
    standardized_heldout = (heldout_matrix - fit_means) / fit_scales

    nuisance = pd.read_parquet(artifacts["nuisance_predictions"])
    fit_nuisance = _ordered_nuisance(nuisance, fit_ids, "fit_oof")
    heldout_nuisance = _ordered_nuisance(
        nuisance, heldout_ids, "external_heldout"
    )
    fit_treatment = fit_data[args.treatment_column].to_numpy(dtype=float)
    fit_outcome = fit_data[args.outcome_column].to_numpy(dtype=float)
    fit_u = fit_treatment - fit_nuisance["treatment_stacked"].to_numpy(dtype=float)
    fit_v = fit_outcome - fit_nuisance["outcome_stacked"].to_numpy(dtype=float)
    constant_effect = float(np.dot(fit_u, fit_v) / np.dot(fit_u, fit_u))

    treatment = heldout_data[args.treatment_column].to_numpy(dtype=float)
    outcome = heldout_data[args.outcome_column].to_numpy(dtype=float)
    treatment_residual = treatment - heldout_nuisance[
        "treatment_stacked"
    ].to_numpy(dtype=float)
    outcome_residual = outcome - heldout_nuisance["outcome_stacked"].to_numpy(
        dtype=float
    )
    constant_residual = outcome_residual - treatment_residual * constant_effect
    cohort_contribution = treatment_residual * constant_residual

    # Orthogonalize interaction columns against the constant-effect score.
    # This uses held-out features and treatment residuals, but no outcomes.
    weights = np.square(treatment_residual)
    weighted_means = np.sum(
        weights[:, None] * standardized_heldout, axis=0
    ) / np.sum(weights)
    interaction_features = standardized_heldout - weighted_means
    row_scores = interaction_features * cohort_contribution[:, None]
    test = _multiplier_test(
        row_scores,
        repeats=args.bootstrap_repeats,
        chunk_size=args.bootstrap_chunk_size,
        seed=args.seed,
    )

    retained = np.asarray(test.pop("retained_columns"), dtype=bool)
    score_means = np.asarray(test.pop("column_means"), dtype=float)
    score_scales = np.asarray(test.pop("column_standard_deviations"), dtype=float)
    standardized_scores = np.asarray(
        test.pop("column_standardized_scores"), dtype=float
    )
    retained_records = [record for record, keep in zip(term_records, retained) if keep]
    term_results = []
    for record, mean, scale, statistic in zip(
        retained_records, score_means, score_scales, standardized_scores
    ):
        term_results.append(
            {
                **record,
                "heldout_score_moment": float(mean),
                "heldout_score_standard_deviation": float(scale),
                "heldout_standardized_score": float(statistic),
                "unadjusted_two_sided_normal_p": float(
                    2.0 * norm.sf(abs(statistic))
                ),
            }
        )
    term_results.sort(
        key=lambda row: abs(row["heldout_standardized_score"]), reverse=True
    )

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
            "heldout_outcomes_used_for_selection": False,
            "oracle_modifier_column": args.oracle_modifier_column,
            "positive_category": args.positive_category,
            "association_kind": association_kind,
            "selected_topic_index_zero_based": int(topic_index),
            "selected_topic_id": topic["topic_id"],
            "training_association_strength": float(associations[topic_index]),
            "terms_per_topic": int(args.terms_per_topic),
        },
        "test_definition": {
            "cate_model_fitted": False,
            "patient_pseudo_target_constructed": False,
            "train_fitted_constant_effect": constant_effect,
            "cohort_contribution": (
                "treatment_residual * (outcome_residual - "
                "train_constant_effect * treatment_residual)"
            ),
            "feature_scaling": "fit_mean_and_standard_deviation",
            "constant_effect_orthogonalization": (
                "heldout_treatment_residual_squared_weighted_feature_centering"
            ),
            "bootstrap_kind": "rademacher_multiplier_on_centered_row_scores",
            "bootstrap_repeats": int(args.bootstrap_repeats),
        },
        "cohort_contribution_summary": {
            "mean": float(np.mean(cohort_contribution)),
            "standard_deviation": float(np.std(cohort_contribution, ddof=1)),
        },
        "group_score_test": test,
        "term_scores": term_results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--bootstrap-repeats", type=int, default=100000)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1043)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, default=str))
