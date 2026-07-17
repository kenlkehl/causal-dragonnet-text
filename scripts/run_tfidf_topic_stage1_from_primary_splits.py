#!/usr/bin/env python3
"""Create a registry-sealed TF-IDF Stage 1 from authoritative outer folds.

The workflow is classical CPU-only: projected Parquet reads, stratified split
construction, sparse nuisance models, TF-IDF contrasts, and consensus NMF. It
does not construct an LLM, embedding model, transformer, or GPU client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, MutableMapping

# A fork pool should be created only after discouraging numeric and allocator
# libraries from eagerly creating background thread teams.  This pre-scan runs
# before pandas, PyArrow, NumPy, sklearn, or OCI are imported and affects only
# the explicitly requested Linux multiprocessing backend.  Existing user
# values win; per-worker threadpoolctl limits in Stage 1 provide a second bound.
_FORK_GUARD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "ARROW_DEFAULT_MEMORY_POOL": "system",
    "MALLOC_CONF": "background_thread:false",
}


def _fork_backend_requested(argv: list[str]) -> bool:
    for index, token in enumerate(argv):
        if token == "--parallel-backend" and index + 1 < len(argv):
            return argv[index + 1].strip().lower() in {"fork", "multiprocessing"}
        if token.startswith("--parallel-backend="):
            return token.split("=", 1)[1].strip().lower() in {"fork", "multiprocessing"}
    return False


def _apply_fork_guard_environment(environment: MutableMapping[str, str]) -> dict[str, str]:
    for variable, default_value in _FORK_GUARD_ENVIRONMENT.items():
        environment.setdefault(variable, default_value)
    return {name: environment[name] for name in _FORK_GUARD_ENVIRONMENT}


if _fork_backend_requested(sys.argv[1:]):
    _apply_fork_guard_environment(os.environ)

import pandas as pd

from oci.config import (
    AppliedInferenceConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
    TfidfTopicDiscoveryConfig,
)
from oci.inference.tfidf_topic_agentic_forest import (
    validate_tfidf_topic_stage2_handoff,
)
from oci.inference.tfidf_topic_split_registry import (
    TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
    load_tfidf_topic_split_registry,
)
from oci.inference.tfidf_topic_stage1 import (
    make_joint_treatment_outcome_splits,
    run_tfidf_topic_stage1,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_registry(
    *,
    data: pd.DataFrame,
    primary_predictions_path: Path,
    treatment_column: str,
    outcome_column: str,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> dict[str, Any]:
    split_rows = pd.read_parquet(
        primary_predictions_path,
        columns=["_oci_row_id", "outer_fold"],
    )
    if len(split_rows) != len(data) or split_rows["_oci_row_id"].duplicated().any():
        raise ValueError("primary prediction split rows do not match the dataset")
    expected_ids = set(range(len(data)))
    if set(map(int, split_rows["_oci_row_id"])) != expected_ids:
        raise ValueError("primary prediction split rows lack canonical positional IDs")
    fold_values = set(map(int, split_rows["outer_fold"]))
    if fold_values != set(range(1, int(outer_folds) + 1)):
        raise ValueError("primary prediction outer folds are incomplete")

    folds: list[dict[str, Any]] = []
    heldout_counts: dict[int, int] = {}
    indexed = split_rows.set_index("_oci_row_id")
    for outer_fold in range(1, int(outer_folds) + 1):
        heldout_ids = sorted(
            map(
                int,
                indexed.index[indexed["outer_fold"].astype(int) == outer_fold],
            )
        )
        fit_ids = sorted(expected_ids - set(heldout_ids))
        for row_id in heldout_ids:
            heldout_counts[row_id] = heldout_counts.get(row_id, 0) + 1
        outer_train = data.iloc[fit_ids].copy()
        inner_splits, _metadata = make_joint_treatment_outcome_splits(
            outer_train,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            outcome_type="binary",
            n_splits=int(inner_folds),
            seed=int(seed) + 51_000 + outer_fold,
        )
        inner = []
        for inner_fold, (fit_local, heldout_local) in enumerate(inner_splits, start=1):
            inner.append(
                {
                    "inner_fold": inner_fold,
                    "fit_row_ids": [fit_ids[int(position)] for position in fit_local],
                    "heldout_row_ids": [fit_ids[int(position)] for position in heldout_local],
                }
            )
        folds.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": fit_ids,
                "heldout_row_ids": heldout_ids,
                "inner_folds": inner,
            }
        )
    if set(heldout_counts) != expected_ids or set(heldout_counts.values()) != {1}:
        raise ValueError("primary prediction heldouts do not partition the dataset once")
    return {
        "schema_version": TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
        "dataset_row_count": len(data),
        "outer_folds": folds,
    }


def build_config(args: argparse.Namespace, registry_path: Path) -> AppliedInferenceConfig:
    topic = TfidfTopicDiscoveryConfig(
        max_features=int(args.max_features),
        topic_count=int(args.topic_count),
        topic_seeds=list(map(int, args.topic_seeds)),
        stability_repeats=int(args.stability_repeats),
        score_test_bootstrap_repeats=int(args.bootstrap_repeats),
        random_state=int(args.seed),
    )
    forest = MultiModelForestConfig(
        feature_discovery_methods=["bow", "tfidf_topic_contrast"],
        candidate_consistency_inner_folds=int(args.inner_folds),
        nuisance_folds=int(args.nuisance_folds),
        cpus_total=int(args.workers),
        outer_parallel_backend=str(args.parallel_backend),
        split_registry_path=str(registry_path.resolve()),
        tfidf_topic=topic,
    )
    architecture = ModelArchitectureConfig(
        model_type="multi_model_forest",
        multi_model_forest=forest,
    )
    config = AppliedInferenceConfig(
        clinical_question=(
            "Discover fold-local pre-treatment text variables for causal adjustment "
            "and heterogeneous treatment effects."
        ),
        outcome_type="binary",
        dataset_path=str(args.dataset.resolve()),
        text_column=args.text_column,
        treatment_column=args.treatment_column,
        outcome_column=args.outcome_column,
        cv_folds=int(args.outer_folds),
        architecture=architecture,
    )
    setattr(config, "seed", int(args.seed))
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--nuisance-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--parallel-backend",
        choices=("threads", "processes", "multiprocessing", "fork"),
        default="threads",
        help=(
            "Joblib backend for independent classical TF-IDF/NMF contexts. "
            "'processes' uses loky; 'multiprocessing' (alias 'fork') uses the "
            "Linux fork backend and avoids worker re-imports."
        ),
    )
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--topic-count", type=int, default=100)
    parser.add_argument("--topic-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--stability-repeats", type=int, default=30)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve(strict=True)
    primary_path = args.primary_predictions.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    projected_columns = [
        args.text_column,
        args.treatment_column,
        args.outcome_column,
    ]
    data = pd.read_parquet(dataset_path, columns=projected_columns).reset_index(drop=True)
    registry_body = build_registry(
        data=data,
        primary_predictions_path=primary_path,
        treatment_column=args.treatment_column,
        outcome_column=args.outcome_column,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
    )
    registry_path = output / "split_registry.json"
    _write_json(registry_path, registry_body)
    registry = load_tfidf_topic_split_registry(
        registry_path,
        dataset_row_count=len(data),
        outer_fold_count=int(args.outer_folds),
        inner_fold_count=int(args.inner_folds),
    )
    config = build_config(args, registry_path)
    effective_parallel_backend = str(config.architecture.multi_model_forest.outer_parallel_backend)
    fork_thread_environment = (
        {name: os.environ.get(name) for name in _FORK_GUARD_ENVIRONMENT}
        if effective_parallel_backend == "multiprocessing"
        else None
    )
    audit = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "primary_predictions_path": str(primary_path),
        "primary_predictions_sha256": _sha256_file(primary_path),
        "dataset_columns_read": projected_columns,
        "primary_prediction_columns_read": ["_oci_row_id", "outer_fold"],
        "dataset_rows": len(data),
        "split_registry_content_hash": registry["content_hash"],
        "outer_folds": int(args.outer_folds),
        "inner_folds": int(args.inner_folds),
        "workers": int(args.workers),
        "parallel_backend": effective_parallel_backend,
        "parallel_backend_requested": str(args.parallel_backend),
        "fork_thread_environment": fork_thread_environment,
        "llm_or_model_client_constructed": False,
        "gpu_or_transformer_used": False,
        "config": asdict(config),
    }
    _write_json(output / "stage1_invocation_audit.json", audit)
    if args.dry_run:
        return

    handoff_path = output / "handoff" / "discovery_contexts.jsonl"
    run_tfidf_topic_stage1(
        dataset=data,
        config=config,
        output_path=output / "primary_predictions.parquet",
        artifact_dir=output,
        handoff_path=handoff_path,
    )
    preflight = validate_tfidf_topic_stage2_handoff(
        dataset=data,
        config=config,
        handoff_path=handoff_path,
    )
    _write_json(output / "handoff" / "stage2_preflight.json", preflight)
    result = {
        **audit,
        "handoff_path": str(handoff_path),
        "handoff_sha256": _sha256_file(handoff_path),
        "primary_predictions_sha256": _sha256_file(output / "primary_predictions.parquet"),
        "stage2_preflight": preflight,
    }
    _write_json(output / "stage1_result.json", result)


if __name__ == "__main__":
    main()
