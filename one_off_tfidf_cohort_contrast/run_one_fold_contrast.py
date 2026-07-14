#!/usr/bin/env python
"""Compute and save a cohort TF-IDF modifier contrast for one outer fold."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

try:
    from one_off_tfidf_cohort_contrast.run_experiment import (
        build_feature_frame,
        cohort_contrast,
        feature_support,
        individual_nuisance_sources,
        make_vectorizer,
        normalize_text,
        nuisance_rows_for_fold,
        source_sign_agreement,
        stability_diagnostics,
        stacked_nuisance_predictions,
        tail_contrast,
    )
except ModuleNotFoundError:
    from run_experiment import (
        build_feature_frame,
        cohort_contrast,
        feature_support,
        individual_nuisance_sources,
        make_vectorizer,
        normalize_text,
        nuisance_rows_for_fold,
        source_sign_agreement,
        stability_diagnostics,
        stacked_nuisance_predictions,
        tail_contrast,
    )


LOGGER = logging.getLogger("one_fold_cohort_contrast")
DEFAULT_DATASET = (
    "synthetic_data/example_synthetic_datasets/"
    "five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
)
DEFAULT_NUISANCE = (
    "../pcori_experiments/five_conf_five_mod_agent_refactor_7-9-26/"
    "multi_model_forest/25366977da7d/text_model_feature_predictions.parquet"
)
DEFAULT_OUTPUT = (
    "one_off_tfidf_cohort_contrast/results_five_conf_five_mod_fold1"
)


def run(args: argparse.Namespace) -> None:
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.dataset).reset_index(drop=True)
    dataset["_oci_row_id"] = np.arange(len(dataset), dtype=int)
    indexed = dataset.set_index("_oci_row_id")
    nuisance = pd.read_parquet(args.nuisance_predictions)

    available_folds = sorted(int(value) for value in nuisance["outer_fold"].dropna().unique())
    if args.outer_fold not in available_folds:
        raise ValueError(
            f"outer fold {args.outer_fold} is unavailable; choices are {available_folds}"
        )
    anchor_train = nuisance_rows_for_fold(
        nuisance, args.outer_fold, "train_inner_oof", "ensemble_mean_nuisance"
    )
    anchor_test = nuisance_rows_for_fold(
        nuisance, args.outer_fold, "test_outer_train_fit", "ensemble_mean_nuisance"
    )
    train_ids = np.sort(anchor_train["_oci_row_id"].unique())
    test_ids = np.sort(anchor_test["_oci_row_id"].unique())
    train = indexed.loc[train_ids].copy()
    test = indexed.loc[test_ids].copy()

    train_texts = [normalize_text(value) for value in train[args.text_column]]
    vectorizer = make_vectorizer(args)
    x_train = vectorizer.fit_transform(train_texts).tocsr()
    feature_names = vectorizer.get_feature_names_out()
    LOGGER.info(
        "fold=%s train=%s test=%s vocabulary=%s",
        args.outer_fold,
        len(train_ids),
        len(test_ids),
        len(feature_names),
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
    primary = cohort_contrast(
        x_train,
        treatment,
        outcome,
        e_train,
        m_train,
        probability_clip=args.probability_clip,
    )
    tail_score = tail_contrast(
        x_train,
        primary.patient_contribution,
        quantile=args.tail_quantile,
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
        x_train,
        treatment,
        outcome,
        source_predictions,
        primary.z_score,
        probability_clip=args.probability_clip,
    )
    subsample_agreement, selection_frequency = stability_diagnostics(
        x_train,
        treatment,
        outcome,
        e_train,
        m_train,
        primary.z_score,
        repeats=args.stability_repeats,
        fraction=args.stability_fraction,
        top_pool=args.stability_top_pool,
        probability_clip=args.probability_clip,
        random_state=args.random_state + 10_000 * args.outer_fold,
    )
    document_frequency, treated_count, control_count = feature_support(
        x_train, treatment
    )
    scores = build_feature_frame(
        outer_fold=args.outer_fold,
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
    ranked = scores[scores["eligible"]].sort_values("rank").copy()
    top = ranked.head(args.top_n).copy()
    scores.to_parquet(output_dir / "contrast_feature_scores.parquet", index=False)
    top.to_csv(output_dir / f"top{args.top_n}_ranked_phrases.csv", index=False)

    summary = {
        "outer_fold": args.outer_fold,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "vocabulary_size": len(feature_names),
        "eligible_features": len(ranked),
        "nuisance_sources": source_names,
        "n_nuisance_sources": n_sources,
        "nuisance_treatment_auroc": float(roc_auc_score(treatment, e_train)),
        "nuisance_treatment_brier": float(brier_score_loss(treatment, e_train)),
        "nuisance_outcome_auroc": float(roc_auc_score(outcome, m_train)),
        "nuisance_outcome_brier": float(brier_score_loss(outcome, m_train)),
        "residual_constant_effect": primary.tau_constant,
        "elapsed_seconds": time.time() - start,
        "args": vars(args),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    LOGGER.info("complete in %.1f seconds: %s", time.time() - start, output_dir)
    print(
        top[
            [
                "rank",
                "feature",
                "z_score",
                "ranking_score",
                "document_frequency",
                "source_sign_agreement",
                "subsample_sign_agreement",
                "subsample_top_pool_frequency",
            ]
        ].to_string(index=False)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--nuisance-predictions", default=DEFAULT_NUISANCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-fold", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.98)
    parser.add_argument("--max-features", type=int, default=30_000)
    parser.add_argument("--min-arm-count", type=int, default=3)
    parser.add_argument("--probability-clip", type=float, default=0.02)
    parser.add_argument("--tail-quantile", type=float, default=0.20)
    parser.add_argument("--stability-repeats", type=int, default=30)
    parser.add_argument("--stability-fraction", type=float, default=0.75)
    parser.add_argument("--stability-top-pool", type=int, default=500)
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
