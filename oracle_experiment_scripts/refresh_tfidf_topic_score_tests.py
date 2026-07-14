#!/usr/bin/env python
"""Refresh honest inner-held-out topic score tests from fitted Stage 1 artifacts.

This utility intentionally does not refit a vectorizer, nuisance model, or NMF
bank.  It is useful when score-test semantics change but the exact-scope Stage
1 fitted artifacts remain valid.  Outer-test labels are never read: full outer
contexts receive only an updated ``not_run`` schema marker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oci.config import ExperimentConfig  # noqa: E402
from oci.inference.tfidf_topic_discovery import (  # noqa: E402
    compact_topic_score_tests,
)
from oci.inference.tfidf_topic_score_selection import (  # noqa: E402
    TOPIC_SCORE_TEST_SCHEMA_VERSION,
    reselect_persisted_topic_scores,
    score_topic_banks,
)
from oci.inference.tfidf_topic_stage1 import (  # noqa: E402
    tfidf_topic_stage1_config_hash,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json_atomic(path: Path, payload: Any) -> None:
    pending = path.with_name(f"{path.name}.pending")
    pending.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pending.replace(path)


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    pending = path.with_name(f"{path.name}.pending")
    with pending.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    pending.replace(path)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], dtype=float) for key in archive.files}


def _ordered_predictions(
    frame: pd.DataFrame,
    *,
    scope: str,
    row_ids: Sequence[int],
) -> pd.DataFrame:
    subset = frame.loc[frame["prediction_scope"] == scope].copy()
    if subset["_oci_row_id"].duplicated().any():
        raise RuntimeError(f"Duplicate nuisance rows for prediction_scope={scope!r}")
    subset["_oci_row_id"] = subset["_oci_row_id"].astype(int)
    subset = subset.set_index("_oci_row_id")
    expected = [int(value) for value in row_ids]
    missing = sorted(set(expected) - set(subset.index))
    if missing:
        raise RuntimeError(
            f"Missing {scope} nuisance predictions for row ids {missing[:5]}"
        )
    return subset.loc[expected]


def _context_rows(
    data_by_id: pd.DataFrame,
    row_ids: Sequence[int],
) -> pd.DataFrame:
    expected = [int(value) for value in row_ids]
    missing = sorted(set(expected) - set(data_by_id.index))
    if missing:
        raise RuntimeError(f"Dataset is missing row ids {missing[:5]}")
    return data_by_id.loc[expected].copy()


def _not_run_score_marker() -> Dict[str, Any]:
    return {
        "schema_version": TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "status": "not_run",
        "reason": "outer_test_labels_reserved",
        "uses_heldout_treatment_and_outcome": False,
        "banks": {},
    }


def validate_committed_handoff(handoff_path: Path) -> Dict[str, Any]:
    """Fail closed unless every exact context has the expected score state."""
    rows = _read_jsonl(handoff_path)
    inner = [
        row
        for row in rows
        if row.get("scope") == "candidate_selection_inner_fit"
    ]
    full = [row for row in rows if row.get("scope") == "full_outer_train"]
    if not rows or len(inner) + len(full) != len(rows):
        raise RuntimeError("Handoff contains missing or unknown context scopes")
    hashes = {str(row.get("stage1_config_hash")) for row in rows}
    if len(hashes) != 1:
        raise RuntimeError("Handoff rows disagree on the Stage 1 config hash")
    selection_counts: Dict[str, List[int]] = {
        "treatment": [],
        "outcome": [],
        "effect": [],
    }
    for row in inner:
        if set(row["fit_row_ids"]) & set(row["heldout_row_ids"]):
            raise RuntimeError(f"Fit/held-out overlap in fold {row.get('fold_key')}")
        score_path = Path(row["discovery"]["artifacts"]["topic_score_tests"])
        payload = json.loads(score_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != TOPIC_SCORE_TEST_SCHEMA_VERSION
            or payload.get("status") != "completed"
        ):
            raise RuntimeError(f"Incomplete score artifact: {score_path}")
        for bank in ("treatment", "outcome", "effect"):
            bank_payload = (payload.get("banks") or {}).get(bank) or {}
            topics = list(bank_payload.get("topic_tests") or [])
            expected_topics = len(
                (
                    row["discovery"].get("topic_banks", {}).get(bank, {})
                ).get("topics", [])
            )
            if len(topics) != expected_topics:
                raise RuntimeError(
                    f"{bank} score/topic count mismatch in {score_path}"
                )
            if not bool(
                (bank_payload.get("bootstrap_calibration") or {}).get(
                    "complete_topic_family"
                )
            ):
                raise RuntimeError(f"Incomplete topic bootstrap family: {score_path}")
            if any(
                "topic_standardized_score" not in topic
                or "_topic_bootstrap_rows" in topic
                for topic in topics
            ):
                raise RuntimeError(f"Invalid scalar topic evidence: {score_path}")
            selection_counts[bank].append(
                int(bank_payload.get("selection_count") or 0)
            )
    for row in full:
        discovery = row["discovery"]
        compact = discovery.get("topic_score_tests") or {}
        if discovery.get("artifacts", {}).get("topic_score_tests") is not None:
            raise RuntimeError("A full-outer context has a score-test artifact")
        if (
            compact.get("schema_version") != TOPIC_SCORE_TEST_SCHEMA_VERSION
            or compact.get("status") != "not_run"
            or bool(compact.get("uses_heldout_treatment_and_outcome"))
        ):
            raise RuntimeError("A full-outer context exposes held-out score tests")
    pending = list(
        handoff_path.parent.parent.rglob("*.score_refresh.pending")
    )
    if pending:
        raise RuntimeError(f"Uncommitted score files remain: {pending[:3]}")
    return {
        "schema_version": TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "context_count": len(rows),
        "inner_context_count": len(inner),
        "outer_context_count": len(full),
        "stage1_config_hash": next(iter(hashes)),
        "selection_ranges": {
            bank: [min(values), max(values)] if values else [0, 0]
            for bank, values in selection_counts.items()
        },
        "pending_file_count": 0,
        "outer_test_score_artifact_count": 0,
    }


def refresh(
    *,
    dataset_path: Path,
    stage1_config_path: Path,
    handoff_path: Path,
    reuse_existing_statistics: bool = False,
) -> Dict[str, Any]:
    saved = json.loads(stage1_config_path.read_text(encoding="utf-8"))
    applied_payload = saved.get("config")
    if not isinstance(applied_payload, dict):
        raise ValueError("stage1_config.json does not contain an applied config")
    # Old runner snapshots may contain the integrated forest payload under
    # both architecture keys.  Reconstruct only the v2 pathway needed here so
    # unrelated legacy configuration cannot affect the refresh.
    architecture = applied_payload.get("architecture") or {}
    forest_payload = architecture.get("multi_model_forest")
    if not isinstance(forest_payload, dict):
        raise ValueError("Saved config has no multi_model_forest payload")
    minimal_applied = {
        key: applied_payload[key]
        for key in (
            "clinical_question",
            "outcome_type",
            "dataset_path",
            "text_column",
            "outcome_column",
            "treatment_column",
            "split_column",
            "cv_folds",
        )
        if key in applied_payload
    }
    minimal_applied["architecture"] = {
        "model_type": "multi_model_forest",
        "multi_model_forest": forest_payload,
    }
    applied = ExperimentConfig.from_dict(
        {"applied_inference": minimal_applied}
    ).applied_inference
    topic_config = applied.architecture.multi_model_forest.tfidf_topic

    data = pd.read_parquet(dataset_path).reset_index(drop=True)
    data["_oci_row_id"] = np.arange(len(data), dtype=int)
    data_by_id = data.set_index("_oci_row_id", drop=False)
    rows = _read_jsonl(handoff_path)
    if not rows:
        raise RuntimeError("Stage 1 handoff is empty")

    staged: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for row in rows:
        discovery = dict(row.get("discovery") or {})
        artifacts = dict(discovery.get("artifacts") or {})
        fitted_path = Path(str(artifacts.get("fitted_context") or ""))
        if not fitted_path.is_file():
            raise RuntimeError(f"Missing fitted context: {fitted_path}")
        metadata_path = fitted_path.parent / "context_metadata.json"
        scope = str(row.get("scope") or "")
        if scope != "candidate_selection_inner_fit":
            marker = _not_run_score_marker()
            discovery["topic_score_tests"] = compact_topic_score_tests(marker)
            staged.append(
                {
                    "row": row,
                    "discovery": discovery,
                    "metadata_path": metadata_path,
                    "pending_score_path": None,
                    "target_score_path": None,
                }
            )
            continue

        if reuse_existing_statistics:
            target_score_path = Path(str(artifacts["topic_score_tests"]))
            persisted = json.loads(
                target_score_path.read_text(encoding="utf-8")
            )
            if (
                bool(topic_config.orphan_ngram_enabled)
                and (persisted.get("effect_orphan_ngram_branch") or {}).get(
                    "status"
                )
                != "completed"
            ):
                raise RuntimeError(
                    "--reuse-existing-statistics cannot create the orphan "
                    "n-gram branch from a legacy score artifact; rerun this "
                    "refresh without that flag. Fitted vectorizers, NMF, and "
                    "nuisance models will still be reused."
                )
            score_tests = reselect_persisted_topic_scores(
                persisted, topic_config
            )
            score_tests["status"] = "completed"
            pending_score_path = target_score_path.with_name(
                f"{target_score_path.name}.score_refresh.pending"
            )
            pending_score_path.write_text(
                json.dumps(score_tests, indent=2, default=str), encoding="utf-8"
            )
            discovery["topic_score_tests"] = compact_topic_score_tests(
                score_tests
            )
            summaries.append(
                {
                    "scope_id": discovery.get("scope_id"),
                    "banks": {
                        bank: {
                            "selected_topics": int(
                                result.get("selection_count", 0)
                            ),
                            "selected_ngrams": int(
                                result.get("ngram_selection_count", 0)
                            ),
                        }
                        for bank, result in score_tests["banks"].items()
                    },
                }
            )
            staged.append(
                {
                    "row": row,
                    "discovery": discovery,
                    "metadata_path": metadata_path,
                    "pending_score_path": pending_score_path,
                    "target_score_path": target_score_path,
                }
            )
            continue

        fit_ids = [int(value) for value in row["fit_row_ids"]]
        heldout_ids = [int(value) for value in row["heldout_row_ids"]]
        fit_df = _context_rows(data_by_id, fit_ids)
        heldout_df = _context_rows(data_by_id, heldout_ids)
        fitted = joblib.load(fitted_path)
        fit_texts = [
            str(value or "").lower()
            for value in fit_df[applied.text_column].fillna("").tolist()
        ]
        heldout_texts = [
            str(value or "").lower()
            for value in heldout_df[applied.text_column].fillna("").tolist()
        ]
        fit_matrix = sparse.csr_matrix(
            fitted.common_vectorizer.transform(fit_texts)
        )
        heldout_matrix = sparse.csr_matrix(
            fitted.common_vectorizer.transform(heldout_texts)
        )
        feature_names = fitted.common_vectorizer.get_feature_names_out()
        fit_topic_values = _load_npz(Path(artifacts["fit_topic_values"]))
        heldout_topic_values = _load_npz(
            Path(artifacts["heldout_topic_values"])
        )
        nuisance = pd.read_parquet(Path(artifacts["nuisance_predictions"]))
        fit_nuisance = _ordered_predictions(
            nuisance, scope="fit_oof", row_ids=fit_ids
        )
        heldout_nuisance = _ordered_predictions(
            nuisance, scope="external_heldout", row_ids=heldout_ids
        )

        score_tests = score_topic_banks(
            fit_matrix=fit_matrix,
            heldout_matrix=heldout_matrix,
            feature_names=feature_names,
            topic_banks=discovery.get("topic_banks") or {},
            fit_topic_values=fit_topic_values,
            heldout_topic_values=heldout_topic_values,
            fit_treatment=fit_df[applied.treatment_column].to_numpy(dtype=float),
            fit_outcome=fit_df[applied.outcome_column].to_numpy(dtype=float),
            heldout_treatment=heldout_df[applied.treatment_column].to_numpy(
                dtype=float
            ),
            heldout_outcome=heldout_df[applied.outcome_column].to_numpy(
                dtype=float
            ),
            fit_propensity=fit_nuisance["treatment_stacked"].to_numpy(
                dtype=float
            ),
            fit_outcome_prediction=fit_nuisance["outcome_stacked"].to_numpy(
                dtype=float
            ),
            heldout_propensity=heldout_nuisance["treatment_stacked"].to_numpy(
                dtype=float
            ),
            heldout_outcome_prediction=heldout_nuisance[
                "outcome_stacked"
            ].to_numpy(dtype=float),
            config=topic_config,
            scope_id=str(discovery.get("scope_id") or row.get("fold_key")),
            raw_ngram_scores={
                bank: pd.read_parquet(Path(artifacts["ngram_scores"][bank]))
                for bank in ("treatment", "outcome", "effect")
            },
        )
        score_tests["status"] = "completed"
        target_score_path = Path(
            str(
                artifacts.get("topic_score_tests")
                or fitted_path.parent / "topic_score_tests.json"
            )
        )
        pending_score_path = target_score_path.with_name(
            f"{target_score_path.name}.score_refresh.pending"
        )
        pending_score_path.write_text(
            json.dumps(score_tests, indent=2, default=str), encoding="utf-8"
        )
        discovery["topic_score_tests"] = compact_topic_score_tests(score_tests)
        artifacts["topic_score_tests"] = str(target_score_path)
        discovery["artifacts"] = artifacts
        summaries.append(
            {
                "scope_id": discovery.get("scope_id"),
                "banks": {
                    bank: {
                        "selected_topics": int(result.get("selection_count", 0)),
                        "selected_ngrams": int(
                            result.get("ngram_selection_count", 0)
                        ),
                    }
                    for bank, result in score_tests["banks"].items()
                },
            }
        )
        staged.append(
            {
                "row": row,
                "discovery": discovery,
                "metadata_path": metadata_path,
                "pending_score_path": pending_score_path,
                "target_score_path": target_score_path,
            }
        )

    expected_inner = sum(
        str(row.get("scope")) == "candidate_selection_inner_fit" for row in rows
    )
    if len(summaries) != expected_inner:
        raise RuntimeError(
            f"Staged {len(summaries)}/{expected_inner} inner score contexts"
        )

    stage1_hash = tfidf_topic_stage1_config_hash(applied)
    for item in staged:
        pending_score = item["pending_score_path"]
        target_score = item["target_score_path"]
        if pending_score is not None:
            pending_score.replace(target_score)
        discovery = item["discovery"]
        _write_json_atomic(item["metadata_path"], discovery)
        item["row"]["discovery"] = discovery
        item["row"]["stage1_config_hash"] = stage1_hash

    _write_jsonl_atomic(handoff_path, rows)
    manifest_path = handoff_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stage1_config_hash"] = stage1_hash
    manifest["inner_topic_and_ngram_score_test_schema"] = (
        TOPIC_SCORE_TEST_SCHEMA_VERSION
    )
    manifest["outer_test_labels_used_for_topic_score_tests"] = False
    _write_json_atomic(manifest_path, manifest)
    summary = {
        "schema_version": TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "stage1_config_hash": stage1_hash,
        "handoff_path": str(handoff_path),
        "context_count": len(rows),
        "inner_context_count": len(summaries),
        "outer_context_count": len(rows) - len(summaries),
        "refit_vectorizers_or_nmf_or_nuisance": False,
        "reused_existing_score_statistics": bool(reuse_existing_statistics),
        "score_statistics_recomputed": not bool(reuse_existing_statistics),
        "outer_test_labels_read_for_score_tests": False,
        "contexts": summaries,
    }
    _write_json_atomic(
        handoff_path.parent / "topic_score_test_refresh_summary.json", summary
    )
    summary["integrity_validation"] = validate_committed_handoff(handoff_path)
    _write_json_atomic(
        handoff_path.parent / "topic_score_test_refresh_summary.json", summary
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--stage1-config", type=Path)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--reuse-existing-statistics", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_committed_handoff(args.handoff), indent=2))
        return
    if args.dataset is None or args.stage1_config is None:
        parser.error("--dataset and --stage1-config are required for a refresh")
    summary = refresh(
        dataset_path=args.dataset,
        stage1_config_path=args.stage1_config,
        handoff_path=args.handoff,
        reuse_existing_statistics=args.reuse_existing_statistics,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
