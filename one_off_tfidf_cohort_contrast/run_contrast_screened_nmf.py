#!/usr/bin/env python
"""Factor top contrast n-grams into stable topics and re-score the topics."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import NMF

try:
    from one_off_tfidf_cohort_contrast.run_experiment import (
        cohort_contrast,
        individual_nuisance_sources,
        make_vectorizer,
        normalize_text,
        nuisance_rows_for_fold,
        source_sign_agreement,
        stability_diagnostics,
        stacked_nuisance_predictions,
    )
    from one_off_tfidf_cohort_contrast.run_one_fold_contrast import (
        DEFAULT_DATASET,
        DEFAULT_NUISANCE,
    )
except ModuleNotFoundError:
    from run_experiment import (
        cohort_contrast,
        individual_nuisance_sources,
        make_vectorizer,
        normalize_text,
        nuisance_rows_for_fold,
        source_sign_agreement,
        stability_diagnostics,
        stacked_nuisance_predictions,
    )
    from run_one_fold_contrast import DEFAULT_DATASET, DEFAULT_NUISANCE


LOGGER = logging.getLogger("contrast_screened_nmf")
DEFAULT_CONTRAST_SCORES = (
    "one_off_tfidf_cohort_contrast/results_five_conf_five_mod_fold1/"
    "contrast_feature_scores.parquet"
)
DEFAULT_OUTPUT = (
    "one_off_tfidf_cohort_contrast/results_five_conf_five_mod_fold1_nmf"
)
TRUE_MODIFIERS = (
    "true_histology_type",
    "true_egfr_mutation_status",
    "true_baseline_nlr",
    "true_brain_metastases_status",
    "true_baseline_hemoglobin",
)
TRUE_CONFOUNDERS = (
    "true_age",
    "true_sex",
    "true_ecog_performance_status",
    "true_creatinine_clearance",
    "true_prior_platinum_therapy",
)


def parse_int_list(value: str) -> List[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def l2_normalize_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=0, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def align_topics(reference_h: np.ndarray, candidate_h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    similarity = l2_normalize_rows(reference_h) @ l2_normalize_rows(candidate_h).T
    reference_indices, candidate_indices = linear_sum_assignment(-similarity)
    mapping = np.full(reference_h.shape[0], -1, dtype=int)
    matched_similarity = np.full(reference_h.shape[0], np.nan, dtype=float)
    mapping[reference_indices] = candidate_indices
    matched_similarity[reference_indices] = similarity[reference_indices, candidate_indices]
    if np.any(mapping < 0):
        raise RuntimeError("failed to align every NMF topic")
    return mapping, matched_similarity


def categorical_eta(values: np.ndarray, categories: Sequence[object]) -> float:
    values = np.asarray(values, dtype=float)
    categories = pd.Series(categories).fillna("<missing>").astype(str).to_numpy()
    total = float(np.sum(np.square(values - np.mean(values))))
    if total <= 1e-12:
        return 0.0
    between = 0.0
    for category in np.unique(categories):
        group = values[categories == category]
        between += len(group) * float(np.square(np.mean(group) - np.mean(values)))
    return float(np.sqrt(max(0.0, min(1.0, between / total))))


def oracle_associations(
    topic_scores: np.ndarray,
    frame: pd.DataFrame,
    variables: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for topic_index in range(topic_scores.shape[1]):
        values = topic_scores[:, topic_index]
        for variable in variables:
            target = frame[variable]
            if pd.api.types.is_numeric_dtype(target):
                statistic = float(spearmanr(values, target.to_numpy(dtype=float)).statistic)
                kind = "spearman"
                strength = abs(statistic)
            else:
                statistic = categorical_eta(values, target)
                kind = "categorical_eta"
                strength = statistic
            rows.append(
                {
                    "topic_index": topic_index,
                    "variable": variable,
                    "variable_role": "effect_modifier" if variable in TRUE_MODIFIERS else "confounder",
                    "association_kind": kind,
                    "association": statistic,
                    "association_strength": strength,
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.dataset).reset_index(drop=True)
    dataset["_oci_row_id"] = np.arange(len(dataset), dtype=int)
    indexed = dataset.set_index("_oci_row_id")
    nuisance = pd.read_parquet(args.nuisance_predictions)
    contrast_scores = pd.read_parquet(args.contrast_scores)

    anchor_train = nuisance_rows_for_fold(
        nuisance, args.outer_fold, "train_inner_oof", "ensemble_mean_nuisance"
    )
    anchor_test = nuisance_rows_for_fold(
        nuisance, args.outer_fold, "test_outer_train_fit", "ensemble_mean_nuisance"
    )
    train_ids = np.sort(anchor_train["_oci_row_id"].unique())
    test_ids = np.sort(anchor_test["_oci_row_id"].unique())
    train = indexed.loc[train_ids].copy()
    train_texts = [normalize_text(value) for value in train[args.text_column]]
    vectorizer = make_vectorizer(args)
    x_all = vectorizer.fit_transform(train_texts).tocsr()
    feature_names = vectorizer.get_feature_names_out().astype(str)

    score_order = contrast_scores.sort_values("feature_index")
    if len(score_order) != len(feature_names) or not np.array_equal(
        score_order["feature"].to_numpy(dtype=str), feature_names
    ):
        raise ValueError("contrast scores do not match the reconstructed TF-IDF vocabulary")
    eligible = contrast_scores[contrast_scores["eligible"]].sort_values("rank")
    candidate_count = int(math.ceil(args.top_fraction * len(eligible)))
    candidate_count = max(args.n_components, candidate_count)
    candidates = eligible.head(candidate_count).copy().sort_values("feature_index")
    selected_indices = candidates["feature_index"].to_numpy(dtype=int)
    selected_names = feature_names[selected_indices]
    x_candidates = x_all[:, selected_indices].tocsr()

    raw_weights = np.sqrt(
        candidates["ranking_score"].to_numpy(dtype=float)
        / max(float(candidates["ranking_score"].median()), 1e-12)
    )
    feature_weights = np.clip(raw_weights, args.min_feature_weight, args.max_feature_weight)
    x_weighted = x_candidates.multiply(feature_weights).tocsr()
    LOGGER.info(
        "fold=%s train=%s vocabulary=%s eligible=%s candidates=%s topics=%s seeds=%s",
        args.outer_fold,
        len(train),
        len(feature_names),
        len(eligible),
        candidate_count,
        args.n_components,
        args.seeds,
    )

    treatment = train[args.treatment_column].to_numpy(dtype=float)
    outcome = train[args.outcome_column].to_numpy(dtype=float)
    source_names = individual_nuisance_sources(nuisance, args.outer_fold)
    e_train, m_train, _, _ = stacked_nuisance_predictions(
        nuisance,
        args.outer_fold,
        train_ids,
        test_ids,
        treatment,
        outcome,
        source_names,
        random_state=args.random_state + 10_000 * args.outer_fold,
    )
    true_e = train["true_treatment_prob"].to_numpy(dtype=float)
    true_m = (
        true_e * train["true_y1_prob"].to_numpy(dtype=float)
        + (1.0 - true_e) * train["true_y0_prob"].to_numpy(dtype=float)
    )

    nmf_runs: List[Dict[str, object]] = []
    for seed in args.seeds:
        LOGGER.info("fitting NMF seed=%s", seed)
        model = NMF(
            n_components=args.n_components,
            init="nndsvdar",
            solver="cd",
            beta_loss="frobenius",
            tol=args.nmf_tolerance,
            max_iter=args.nmf_max_iter,
            random_state=seed,
        )
        w = model.fit_transform(x_weighted)
        h = model.components_
        learned = cohort_contrast(
            sparse.csr_matrix(w),
            treatment,
            outcome,
            e_train,
            m_train,
            probability_clip=args.probability_clip,
        )
        oracle = cohort_contrast(
            sparse.csr_matrix(w),
            treatment,
            outcome,
            true_e,
            true_m,
            probability_clip=args.probability_clip,
        )
        nmf_runs.append(
            {
                "seed": seed,
                "w": w,
                "h": h,
                "learned_z": learned.z_score,
                "oracle_z": oracle.z_score,
                "reconstruction_error": float(model.reconstruction_err_),
                "n_iter": int(model.n_iter_),
            }
        )

    reference_h = np.asarray(nmf_runs[0]["h"])
    mappings = [np.arange(args.n_components, dtype=int)]
    similarities = [np.ones(args.n_components, dtype=float)]
    for run_data in nmf_runs[1:]:
        mapping, similarity = align_topics(reference_h, np.asarray(run_data["h"]))
        mappings.append(mapping)
        similarities.append(similarity)

    aligned_h = []
    aligned_w = []
    aligned_learned_z = []
    aligned_oracle_z = []
    for run_data, mapping in zip(nmf_runs, mappings):
        aligned_h.append(l2_normalize_rows(np.asarray(run_data["h"])[mapping]))
        aligned_w.append(l2_normalize_columns(np.asarray(run_data["w"])[:, mapping]))
        aligned_learned_z.append(np.asarray(run_data["learned_z"])[mapping])
        aligned_oracle_z.append(np.asarray(run_data["oracle_z"])[mapping])
    mean_h = np.mean(np.stack(aligned_h), axis=0)
    consensus_w = np.mean(np.stack(aligned_w), axis=0)
    learned_z_by_seed = np.stack(aligned_learned_z)
    oracle_z_by_seed = np.stack(aligned_oracle_z)

    consensus_learned = cohort_contrast(
        sparse.csr_matrix(consensus_w),
        treatment,
        outcome,
        e_train,
        m_train,
        probability_clip=args.probability_clip,
    )
    consensus_oracle = cohort_contrast(
        sparse.csr_matrix(consensus_w),
        treatment,
        outcome,
        true_e,
        true_m,
        probability_clip=args.probability_clip,
    )
    source_predictions = {}
    for source_name in source_names:
        rows = nuisance_rows_for_fold(
            nuisance, args.outer_fold, "train_inner_oof", source_name
        )
        values = rows.drop_duplicates("_oci_row_id").set_index("_oci_row_id").loc[train_ids]
        source_predictions[source_name] = (
            values["e_hat"].to_numpy(dtype=float),
            values["m_hat"].to_numpy(dtype=float),
        )
    source_agreement, n_sources = source_sign_agreement(
        sparse.csr_matrix(consensus_w),
        treatment,
        outcome,
        source_predictions,
        consensus_learned.z_score,
        probability_clip=args.probability_clip,
    )
    subsample_agreement, selection_frequency = stability_diagnostics(
        sparse.csr_matrix(consensus_w),
        treatment,
        outcome,
        e_train,
        m_train,
        consensus_learned.z_score,
        repeats=args.stability_repeats,
        fraction=args.stability_fraction,
        top_pool=min(args.topic_stability_top_pool, args.n_components),
        probability_clip=args.probability_clip,
        random_state=args.random_state + 20_000 * args.outer_fold,
    )

    mean_seed_z = np.mean(learned_z_by_seed, axis=0)
    mean_seed_oracle_z = np.mean(oracle_z_by_seed, axis=0)
    seed_sign_agreement = np.mean(
        np.sign(learned_z_by_seed) == np.sign(mean_seed_z)[None, :], axis=0
    )
    loading_similarity = np.mean(np.stack(similarities), axis=0)
    ranking_score = (
        np.abs(consensus_learned.z_score)
        * seed_sign_agreement
        * (0.50 + 0.50 * source_agreement)
        * (0.50 + 0.50 * subsample_agreement)
        * (0.25 + 0.75 * selection_frequency)
        * (0.50 + 0.50 * loading_similarity)
    )
    order = np.argsort(ranking_score)[::-1]
    ranks = np.empty(args.n_components, dtype=int)
    ranks[order] = np.arange(1, args.n_components + 1)
    topic_summary = pd.DataFrame(
        {
            "topic_index": np.arange(args.n_components, dtype=int),
            "topic_rank": ranks,
            "learned_z_score": consensus_learned.z_score,
            "mean_seed_learned_z": mean_seed_z,
            "oracle_z_score": consensus_oracle.z_score,
            "mean_seed_oracle_z": mean_seed_oracle_z,
            "learned_oracle_sign_agreement": (
                np.sign(consensus_learned.z_score) == np.sign(consensus_oracle.z_score)
            ),
            "seed_sign_agreement": seed_sign_agreement,
            "source_sign_agreement": source_agreement,
            "subsample_sign_agreement": subsample_agreement,
            "subsample_top_pool_frequency": selection_frequency,
            "mean_loading_similarity": loading_similarity,
            "ranking_score": ranking_score,
        }
    ).sort_values("topic_rank")

    candidate_metadata = candidates.set_index("feature_index")
    term_rows: List[Dict[str, object]] = []
    for topic_index in range(args.n_components):
        term_order = np.argsort(mean_h[topic_index])[::-1][: args.terms_per_topic]
        for term_rank, candidate_position in enumerate(term_order, start=1):
            original_index = int(selected_indices[candidate_position])
            metadata = candidate_metadata.loc[original_index]
            term_rows.append(
                {
                    "topic_index": topic_index,
                    "topic_rank": int(ranks[topic_index]),
                    "term_rank": term_rank,
                    "feature": selected_names[candidate_position],
                    "topic_loading": float(mean_h[topic_index, candidate_position]),
                    "ngram_contrast_rank": int(metadata["rank"]),
                    "ngram_z_score": float(metadata["z_score"]),
                    "ngram_ranking_score": float(metadata["ranking_score"]),
                }
            )
    topic_terms = pd.DataFrame(term_rows).sort_values(
        ["topic_rank", "term_rank"]
    )

    association_variables = [
        variable
        for variable in (*TRUE_MODIFIERS, *TRUE_CONFOUNDERS)
        if variable in train.columns
    ]
    associations = oracle_associations(consensus_w, train, association_variables)
    associations = associations.merge(
        topic_summary[["topic_index", "topic_rank"]], on="topic_index", how="left"
    ).sort_values(["variable", "association_strength"], ascending=[True, False])

    patient_topics = pd.DataFrame(
        consensus_w,
        columns=[f"topic_{index:03d}" for index in range(args.n_components)],
    )
    patient_topics.insert(0, "_oci_row_id", train_ids)
    topic_summary.to_csv(output_dir / "topic_summary.csv", index=False)
    topic_terms.to_csv(output_dir / "topic_terms.csv", index=False)
    associations.to_csv(output_dir / "topic_oracle_associations.csv", index=False)
    patient_topics.to_parquet(output_dir / "patient_topic_scores.parquet", index=False)
    candidates.sort_values("rank").to_csv(
        output_dir / "candidate_ngrams_top_decile.csv", index=False
    )

    config = {
        "args": vars(args),
        "candidate_count": candidate_count,
        "eligible_feature_count": len(eligible),
        "n_nuisance_sources": n_sources,
        "nuisance_sources": source_names,
        "nmf_runs": [
            {
                "seed": run_data["seed"],
                "reconstruction_error": run_data["reconstruction_error"],
                "n_iter": run_data["n_iter"],
            }
            for run_data in nmf_runs
        ],
        "elapsed_seconds": time.time() - start,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    LOGGER.info("complete in %.1f seconds: %s", time.time() - start, output_dir)
    preview = topic_summary.head(args.agent_topic_count).merge(
        topic_terms[topic_terms["term_rank"] <= min(8, args.terms_per_topic)]
        .groupby("topic_index")["feature"]
        .apply(lambda values: " | ".join(values)),
        on="topic_index",
        how="left",
    )
    print(
        preview[
            [
                "topic_rank",
                "topic_index",
                "learned_z_score",
                "oracle_z_score",
                "ranking_score",
                "feature",
            ]
        ].to_string(index=False)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--nuisance-predictions", default=DEFAULT_NUISANCE)
    parser.add_argument("--contrast-scores", default=DEFAULT_CONTRAST_SCORES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-fold", type=int, default=1)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("42,43,44"))
    parser.add_argument("--nmf-max-iter", type=int, default=400)
    parser.add_argument("--nmf-tolerance", type=float, default=1e-4)
    parser.add_argument("--min-feature-weight", type=float, default=0.5)
    parser.add_argument("--max-feature-weight", type=float, default=2.0)
    parser.add_argument("--terms-per-topic", type=int, default=15)
    parser.add_argument("--agent-topic-count", type=int, default=25)
    parser.add_argument("--topic-stability-top-pool", type=int, default=25)
    parser.add_argument("--stability-repeats", type=int, default=30)
    parser.add_argument("--stability-fraction", type=float, default=0.75)
    parser.add_argument("--probability-clip", type=float, default=0.02)
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.98)
    parser.add_argument("--max-features", type=int, default=30_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
