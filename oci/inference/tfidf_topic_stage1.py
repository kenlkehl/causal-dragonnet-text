"""Stage 1 orchestration for exact-scope TF-IDF topic discovery."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import KFold

from ..config import AppliedInferenceConfig, MultiModelForestConfig
from .tfidf_topic_discovery import (
    HANDOFF_SCHEMA_VERSION,
    fit_tfidf_topic_context,
    row_set_fingerprint,
    stable_hash,
)

logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _outer_splits(dataset: pd.DataFrame, config: AppliedInferenceConfig):
    if int(config.cv_folds) > 1:
        yield from enumerate(
            KFold(n_splits=int(config.cv_folds), shuffle=True, random_state=42).split(dataset),
            start=1,
        )
        return
    split_column = config.split_column
    if split_column in dataset.columns and "test" in set(dataset[split_column]):
        train = np.where(dataset[split_column].isin(["train", "val"]).to_numpy())[0]
        test = np.where((dataset[split_column] == "test").to_numpy())[0]
        yield 1, (train, test)
        return
    raise ValueError(
        "multi_model_forest v2 requires cv_folds > 1 or an explicit held-out test split"
    )


def run_tfidf_topic_stage1(
    *,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    artifact_dir: Path,
    handoff_path: Path,
) -> None:
    """Fit every exact outer/inner context and write a v2 fail-closed handoff."""
    data = dataset.reset_index(drop=True).copy()
    data["_oci_row_id"] = np.arange(len(data), dtype=int)
    nn_config: MultiModelForestConfig = config.architecture.multi_model_forest
    topic_config = nn_config.tfidf_topic
    contexts_dir = Path(artifact_dir) / "stage1_tfidf_topics" / "contexts"
    contexts_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    primary_predictions: List[pd.DataFrame] = []
    stage1_hash = stable_hash(
        {
            "schema": HANDOFF_SCHEMA_VERSION,
            "views": [asdict(view) for view in nn_config.bow_views],
            "nuisance_folds": nn_config.nuisance_folds,
            "topic": asdict(topic_config),
            "text_column": config.text_column,
            "treatment_column": config.treatment_column,
            "outcome_column": config.outcome_column,
            "outcome_type": config.outcome_type,
        }
    )

    for outer_fold, (train_idx, test_idx) in _outer_splits(data, config):
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        outer_train = data.iloc[train_idx].copy()
        outer_test = data.iloc[test_idx].copy()
        split_rows.append(
            {
                "outer_fold": int(outer_fold),
                "fit_row_ids": outer_train["_oci_row_id"].astype(int).tolist(),
                "heldout_row_ids": outer_test["_oci_row_id"].astype(int).tolist(),
                "fit_row_fingerprint": row_set_fingerprint(outer_train["_oci_row_id"]),
                "heldout_row_fingerprint": row_set_fingerprint(outer_test["_oci_row_id"]),
                "honest_outer_holdout": True,
            }
        )

        inner_count = min(
            int(nn_config.candidate_consistency_inner_folds),
            max(2, len(outer_train) // 4),
        )
        inner_splitter = KFold(
            n_splits=inner_count,
            shuffle=True,
            random_state=51_000 + int(outer_fold),
        )
        context_specs: List[Dict[str, Any]] = []
        for inner_fold, (fit_local, heldout_local) in enumerate(
            inner_splitter.split(outer_train), start=1
        ):
            context_specs.append(
                {
                    "inner_fold": int(inner_fold),
                    "scope": "candidate_selection_inner_fit",
                    "fold_key": 1000 * int(outer_fold) + int(inner_fold),
                    "fit_df": outer_train.iloc[np.asarray(fit_local, dtype=int)].copy(),
                    "heldout_df": outer_train.iloc[np.asarray(heldout_local, dtype=int)].copy(),
                    "scope_id": f"outer_{outer_fold:03d}_inner_{inner_fold:03d}",
                }
            )
        context_specs.append(
            {
                "inner_fold": None,
                "scope": "full_outer_train",
                "fold_key": int(outer_fold),
                "fit_df": outer_train,
                "heldout_df": outer_test,
                "scope_id": f"outer_{outer_fold:03d}_full_train",
            }
        )

        def fit_spec(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            metadata_path = contexts_dir / spec["scope_id"] / "context_metadata.json"
            if metadata_path.exists():
                try:
                    cached = json.loads(metadata_path.read_text(encoding="utf-8"))
                    expected_fit = row_set_fingerprint(spec["fit_df"]["_oci_row_id"])
                    expected_heldout = row_set_fingerprint(spec["heldout_df"]["_oci_row_id"])
                    artifact_paths = [
                        cached.get("artifacts", {}).get("fitted_context"),
                        cached.get("artifacts", {}).get("fit_topic_values"),
                        cached.get("artifacts", {}).get("heldout_topic_values"),
                        cached.get("artifacts", {}).get("nuisance_predictions"),
                        *(cached.get("artifacts", {}).get("ngram_scores", {}).values()),
                    ]
                    if (
                        cached.get("fit_row_fingerprint") == expected_fit
                        and cached.get("heldout_row_fingerprint") == expected_heldout
                        and cached.get("config_hash") == stable_hash(asdict(topic_config))
                        and all(path and Path(path).exists() for path in artifact_paths)
                    ):
                        logger.info(
                            "Reusing complete exact Stage 1 context scope_id=%s",
                            spec["scope_id"],
                        )
                        return spec, cached
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    logger.warning(
                        "Ignoring incomplete exact-context checkpoint scope_id=%s",
                        spec["scope_id"],
                    )
            logger.info(
                "Stage 1 fitting exact context scope_id=%s fit=%s heldout=%s",
                spec["scope_id"],
                len(spec["fit_df"]),
                len(spec["heldout_df"]),
            )
            metadata = fit_tfidf_topic_context(
                fit_df=spec["fit_df"],
                heldout_df=spec["heldout_df"],
                text_column=config.text_column,
                treatment_column=config.treatment_column,
                outcome_column=config.outcome_column,
                outcome_type=config.outcome_type,
                views=nn_config.bow_views,
                nuisance_folds=int(nn_config.nuisance_folds),
                config=topic_config,
                artifact_dir=contexts_dir / spec["scope_id"],
                scope_id=spec["scope_id"],
            )
            logger.info("Stage 1 completed exact context scope_id=%s", spec["scope_id"])
            return spec, metadata

        requested_workers = int(nn_config.cpus_total or 1)
        context_workers = max(1, min(len(context_specs), requested_workers, 4))
        with parallel_config(
            backend="loky",
            n_jobs=context_workers,
            inner_max_num_threads=1,
        ):
            completed_contexts = Parallel(batch_size=1, pre_dispatch="all")(
                delayed(fit_spec)(spec) for spec in context_specs
            )

        full_metadata: Optional[Dict[str, Any]] = None
        for spec, metadata in completed_contexts:
            rows.append(
                {
                    "schema_version": HANDOFF_SCHEMA_VERSION,
                    "stage1_config_hash": stage1_hash,
                    "fold_key": int(spec["fold_key"]),
                    "outer_fold": int(outer_fold),
                    "inner_fold": spec["inner_fold"],
                    "scope": spec["scope"],
                    "fit_row_ids": metadata["fit_row_ids"],
                    "heldout_row_ids": metadata["heldout_row_ids"],
                    "fit_row_fingerprint": metadata["fit_row_fingerprint"],
                    "heldout_row_fingerprint": metadata["heldout_row_fingerprint"],
                    "discovery": metadata,
                }
            )
            if spec["scope"] == "full_outer_train":
                full_metadata = metadata
        if full_metadata is None:
            raise RuntimeError(f"Full outer context did not complete for fold {outer_fold}")
        nuisance = pd.read_parquet(full_metadata["artifacts"]["nuisance_predictions"])
        nuisance = nuisance[nuisance["prediction_scope"] == "external_heldout"].copy()
        nuisance["outer_fold"] = int(outer_fold)
        nuisance["honest_outer_holdout"] = True
        nuisance["estimation_provenance"] = "outer_train_tfidf_nuisance_only"
        primary_predictions.append(nuisance)

    rows.sort(key=lambda row: (int(row["outer_fold"]), int(row.get("inner_fold") or 999)))
    required_inner = int(nn_config.candidate_consistency_inner_folds)
    for outer_fold in sorted({int(row["outer_fold"]) for row in rows}):
        inner_rows = [
            row for row in rows
            if int(row["outer_fold"]) == outer_fold
            and row["scope"] == "candidate_selection_inner_fit"
        ]
        full_rows = [
            row for row in rows
            if int(row["outer_fold"]) == outer_fold and row["scope"] == "full_outer_train"
        ]
        if len(inner_rows) != required_inner or len(full_rows) != 1:
            raise RuntimeError(
                f"Exact Stage 1 context set is incomplete for outer_fold={outer_fold}: "
                f"inner={len(inner_rows)}/{required_inner}, full={len(full_rows)}/1"
            )

    handoff_path = Path(handoff_path)
    _write_jsonl(handoff_path, rows)
    _write_json(
        handoff_path.parent / "manifest.json",
        {
            "schema_version": HANDOFF_SCHEMA_VERSION,
            "stage1_config_hash": stage1_hash,
            "path": str(handoff_path),
            "n_rows": len(rows),
            "n_outer_folds": len(split_rows),
            "inner_contexts_per_outer": required_inner,
            "exact_inner_contexts": True,
            "stage1_raw_text_forest_prediction": False,
            "stage2_raw_text_modeling_required": False,
            "feature_discovery_methods": ["bow", "tfidf_topic_contrast"],
        },
    )
    _write_jsonl(Path(artifact_dir) / "split_provenance.jsonl", split_rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(primary_predictions, ignore_index=True).sort_values("_oci_row_id").to_parquet(
        output_path, index=False
    )
    logger.info(
        "Saved Stage 1 nuisance/topic handoff contexts=%s path=%s",
        len(rows),
        handoff_path,
    )
