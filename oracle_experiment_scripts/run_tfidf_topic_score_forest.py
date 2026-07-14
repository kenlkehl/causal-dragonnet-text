#!/usr/bin/env python
"""Run the fast, non-agentic Stage 2 over persisted TF-IDF/NMF topic scores."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Union

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.inference.tfidf_topic_score_forest import (  # noqa: E402
    TopicScoreForestConfig,
    run_tfidf_topic_score_forest,
)


def _max_features(value: str) -> Union[str, int, float]:
    normalized = str(value).strip().lower()
    if normalized in {"sqrt", "log2", "auto"}:
        return normalized
    try:
        parsed_int = int(normalized)
        if str(parsed_int) == normalized:
            return parsed_int
    except ValueError:
        pass
    try:
        return float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "max features must be sqrt, log2, auto, an integer, or a float"
        ) from exc


def _dataset_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit honest outer-fold causal forests with treatment/outcome NMF topics "
            "as W and effect NMF topics as X. No LLM server is used."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage1-handoff", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--outcome-type", choices=["binary", "continuous"], default="binary")
    parser.add_argument("--patient-id-column", default="patient_id")
    parser.add_argument("--oracle-ite-column", default="true_ite_prob")
    parser.add_argument("--no-oracle-evaluation", action="store_true")
    parser.add_argument("--cf-n-estimators", type=int, default=200)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=10)
    parser.add_argument("--cf-max-depth", type=int, default=None)
    parser.add_argument("--cf-max-features", type=_max_features, default="sqrt")
    parser.add_argument("--cf-no-inference", action="store_true")
    parser.add_argument(
        "--cf-tune",
        action="store_true",
        help="Run EconML automatic tuning before every fold (off by default).",
    )
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument(
        "--include-stacked-nuisance-in-w",
        action="store_true",
        help=(
            "Also put the honest Stage 1 stacked propensity/outcome predictions in W. "
            "The default benchmark uses topic scores only."
        ),
    )
    parser.add_argument("--no-persist-fold-models", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    dataset_path = _dataset_path(args.dataset)
    dataset = pd.read_parquet(dataset_path)
    config = TopicScoreForestConfig(
        n_estimators=args.cf_n_estimators,
        max_depth=args.cf_max_depth,
        min_samples_leaf=args.cf_min_samples_leaf,
        max_features=args.cf_max_features,
        inference=not args.cf_no_inference,
        tune_model=args.cf_tune,
        standardize=not args.no_standardize,
        include_stacked_nuisance_in_w=args.include_stacked_nuisance_in_w,
        random_state=args.seed,
        persist_fold_models=not args.no_persist_fold_models,
    )
    result = run_tfidf_topic_score_forest(
        dataset=dataset,
        handoff_path=Path(args.stage1_handoff),
        output_dir=Path(args.output_dir),
        treatment_column=args.treatment_column,
        outcome_column=args.outcome_column,
        outcome_type=args.outcome_type,
        id_columns=(args.patient_id_column,),
        oracle_ite_column=(None if args.no_oracle_evaluation else args.oracle_ite_column),
        config=config,
        force=args.force,
    )
    print(json.dumps(result.get("oracle_metrics") or result, indent=2, default=str))


if __name__ == "__main__":
    main()
