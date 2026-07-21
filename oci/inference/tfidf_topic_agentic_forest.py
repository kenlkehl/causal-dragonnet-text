"""Stage 2 structured-feature workflow for TF-IDF topic handoffs."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error

from ..config import (
    AppliedInferenceConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    MultiModelForestConfig,
    load_explicit_feature_specs_json,
)
from ..models.explicit_feature_featurizer import get_raw_explicit_features
from .agentic_explicit_feature_forest import (
    CausalForestExplicitEvaluator,
    SplitEvaluation,
    StructuredInteractionExplicitEvaluator,
    _get_agent_response_trace,
    _normalize_feature_name,
    make_explicit_feature_extraction_provider,
    make_feature_search_agent,
)
from .multi_model_agentic_forest import (
    _build_value_driven_feature_clusters,
    _columns_to_feature_dicts,
    _parsimony_feature_contract_document,
    _parsimony_tfidf_semantic_vectors,
    _validate_parsimony_factor_candidate,
)
from .tfidf_topic_discovery import (
    HANDOFF_SCHEMA_VERSION,
    cohort_contrast_scores,
    nuisance_metrics,
    row_set_fingerprint,
    stable_hash,
)
from .tfidf_topic_score_selection import TOPIC_SCORE_TEST_SCHEMA_VERSION
from .tfidf_topic_split_registry import (
    load_tfidf_topic_split_registry,
    validate_handoff_rows_against_split_registry,
)

logger = logging.getLogger(__name__)

TOPIC_LABEL_PROMPT_VERSION = "tfidf_topic_label_v2"
TOPIC_RECOVERY_PROMPT_VERSION = "tfidf_topic_recovery_v2"
ORPHAN_NGRAM_LABEL_PROMPT_VERSION = "tfidf_orphan_ngram_label_v1"
TOPIC_NAME_HARMONIZATION_PROMPT_VERSION = "tfidf_topic_name_harmonization_v2"
TOPIC_GLOBAL_DEDUP_PROMPT_VERSION = "tfidf_topic_global_dedup_v2"
TOPIC_VALUE_HARMONIZATION_PROMPT_VERSION = "tfidf_topic_value_harmonization_v2"
TOPIC_VALUE_REPAIR_PROMPT_VERSION = "tfidf_topic_value_repair_v2"
CANONICAL_REGISTRY_SCHEMA_VERSION = "tfidf_topic_canonical_registry_v4"
PARSIMONY_SCHEMA_VERSION = "tfidf_topic_parsimony_v2"

_NAME_HARMONIZATION_ACTIONS = {"extract", "derive", "alias/drop", "drop"}
_DERIVATION_OPERATIONS = {
    "copy",
    "sum",
    "difference",
    "product",
    "ratio",
    "mean",
    "minimum",
    "maximum",
}
_HARMONIZATION_BATCH_SIZE = 8
_VALUE_HARMONIZATION_BATCH_SIZE = 8
_GLOBAL_DEDUP_BLOCK_SIZE = 8
_GLOBAL_DEDUP_MIN_SIMILARITY = 0.22
_GLOBAL_DEDUP_MAX_NEIGHBORS = 4
_RECOVERY_NGRAM_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "has",
    "have",
    "not",
    "of",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Checkpoint a JSONL collection without exposing a partially rewritten file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_rank_correlation(
    first: Sequence[float], second: Sequence[float], *, rank: bool
) -> Optional[float]:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return None
    left = left[mask]
    right = right[mask]
    if rank:
        left = pd.Series(left).rank(method="average").to_numpy(dtype=float)
        right = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    if float(np.std(left)) <= 0.0 or float(np.std(right)) <= 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def evaluate_frozen_structured_predictions(
    *,
    prediction_path: Path,
    oracle_frame: pd.DataFrame,
    output_dir: Path,
    oracle_ite_column: str = "true_ite_prob",
) -> Dict[str, Any]:
    """Join synthetic oracle ITEs only after every outer prediction is frozen."""
    prediction_path = Path(prediction_path)
    frozen_hash = _sha256_file(prediction_path)
    predictions = pd.read_parquet(prediction_path)
    if any(str(column).startswith("true_") for column in predictions.columns):
        raise RuntimeError("Frozen structured predictions contain an oracle column")
    oracle = oracle_frame[["_oci_row_id", oracle_ite_column]].copy()
    if oracle["_oci_row_id"].duplicated().any():
        raise ValueError("Oracle frame contains duplicate _oci_row_id values")
    evaluated = predictions.merge(
        oracle,
        on="_oci_row_id",
        how="left",
        validate="one_to_one",
    )
    if evaluated[oracle_ite_column].isna().any():
        raise ValueError("Oracle ITE is missing for one or more frozen predictions")

    def metrics_for(frame: pd.DataFrame) -> Dict[str, Any]:
        truth = frame[oracle_ite_column].to_numpy(dtype=float)
        estimate = frame["pred_ite_prob"].to_numpy(dtype=float)
        error = estimate - truth
        return {
            "n": int(len(frame)),
            "pearson_correlation": _safe_rank_correlation(truth, estimate, rank=False),
            "spearman_correlation": _safe_rank_correlation(truth, estimate, rank=True),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mean_error": float(np.mean(error)),
            "estimated_ate": float(np.mean(estimate)),
            "oracle_ate": float(np.mean(truth)),
            "ate_bias": float(np.mean(estimate) - np.mean(truth)),
            "estimated_ite_standard_deviation": float(np.std(estimate)),
            "oracle_ite_standard_deviation": float(np.std(truth)),
        }

    payload = {
        "schema_version": "tfidf_topic_agentic_forest_v7",
        "evaluation_is_post_hoc": True,
        "all_outer_predictions_frozen_before_oracle_join": True,
        "frozen_prediction_path": str(prediction_path),
        "frozen_prediction_sha256": frozen_hash,
        "oracle_ite_column": oracle_ite_column,
        "overall": metrics_for(evaluated),
        "per_fold": [
            {"outer_fold": int(fold), **metrics_for(frame)}
            for fold, frame in evaluated.groupby("outer_fold", sort=True)
        ],
    }
    output_dir = Path(output_dir)
    evaluated.to_parquet(output_dir / "posthoc_predictions_with_oracle.parquet", index=False)
    _write_json(output_dir / "posthoc_oracle_metrics.json", payload)
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if row.get("schema_version") != HANDOFF_SCHEMA_VERSION:
                raise ValueError(
                    "multi_model_forest v2 rejects legacy handoffs. "
                    f"Line {line_number} has schema {row.get('schema_version')!r}; "
                    "rerun Stage 1 to generate exact TF-IDF topic contexts."
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Stage 1 handoff is empty: {path}")
    return rows


def validate_tfidf_topic_stage2_handoff(
    *,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    handoff_path: Path,
) -> Dict[str, Any]:
    """Audit exact-scope Stage 1 artifacts without constructing an LLM client."""
    from .tfidf_topic_stage1 import (
        tfidf_topic_stage1_config_hash,
        tfidf_topic_stage1_identity,
    )

    data = dataset.reset_index(drop=True)
    n_rows = int(len(data))
    valid_row_ids = set(range(n_rows))
    handoff_path = Path(handoff_path)
    rows = _read_jsonl(handoff_path)
    expected_hash = tfidf_topic_stage1_config_hash(config, data)
    expected_identity = tfidf_topic_stage1_identity(config, data)
    required_inner = int(config.architecture.multi_model_forest.candidate_consistency_inner_folds)
    registry_path = getattr(
        config.architecture.multi_model_forest,
        "split_registry_path",
        None,
    )
    split_registry = None
    if registry_path:
        split_registry = load_tfidf_topic_split_registry(
            registry_path,
            dataset_row_count=n_rows,
            outer_fold_count=int(config.cv_folds),
            inner_fold_count=required_inner,
        )
        validate_handoff_rows_against_split_registry(rows, split_registry)
    max_variables = int(config.explicit_features.max_variables_per_extraction_request)
    if not 1 <= max_variables <= 10:
        raise RuntimeError(
            "Stage 2 extraction preflight requires "
            "max_variables_per_extraction_request in [1, 10]"
        )

    def artifact_path(value: Any) -> Path:
        requested = Path(str(value or "")).expanduser()
        candidates = [requested, handoff_path.parent / requested]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise RuntimeError(f"Missing exact-scope Stage 1 artifact: {value!r}")

    selected_counts: Dict[str, List[int]] = {
        "treatment": [],
        "outcome": [],
        "effect": [],
    }
    topic_counts: Dict[str, List[int]] = {
        "treatment": [],
        "outcome": [],
        "effect": [],
    }
    selected_ngram_counts: Dict[str, List[int]] = {
        "treatment": [],
        "outcome": [],
        "effect": [],
    }
    orphan_cluster_counts: List[int] = []
    selected_orphan_cluster_counts: List[int] = []
    honest_nuisance_rows = 0
    inner_score_contexts = 0
    outer_contexts = 0
    referenced_paths: List[str] = []
    rows_by_outer: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        outer_fold = int(row["outer_fold"])
        rows_by_outer[outer_fold].append(row)
        if str(row.get("stage1_config_hash")) != expected_hash:
            raise RuntimeError(
                f"Stage 1 hash mismatch in outer fold {outer_fold}: "
                f"{row.get('stage1_config_hash')!r} != {expected_hash!r}"
            )
        if split_registry is not None:
            expected_fields = {
                "dataset_content_fingerprint": expected_identity["dataset"]["content_fingerprint"],
                "dataset_ordered_row_fingerprint": expected_identity["dataset"][
                    "ordered_row_fingerprint"
                ],
                "split_semantics_hash": expected_identity["split_semantics_hash"],
                "split_registry_content_hash": split_registry["content_hash"],
            }
            discovery = row.get("discovery") or {}
            for key, expected_value in expected_fields.items():
                if row.get(key) != expected_value:
                    raise RuntimeError(
                        f"Registry-sealed handoff {key} mismatch in outer fold " f"{outer_fold}"
                    )
                if key != "split_registry_content_hash" and discovery.get(key) != expected_value:
                    raise RuntimeError(
                        f"Registry-sealed discovery {key} mismatch in outer fold " f"{outer_fold}"
                    )
        fit_ids = [int(value) for value in row.get("fit_row_ids", [])]
        heldout_ids = [int(value) for value in row.get("heldout_row_ids", [])]
        if len(fit_ids) != len(set(fit_ids)) or len(heldout_ids) != len(set(heldout_ids)):
            raise RuntimeError(f"Duplicate row ids in fold {row.get('fold_key')}")
        if not (set(fit_ids) | set(heldout_ids)) <= valid_row_ids:
            raise RuntimeError(f"Out-of-range row id in fold {row.get('fold_key')}")
        if set(fit_ids) & set(heldout_ids):
            raise RuntimeError(f"Fit/held-out overlap in fold {row.get('fold_key')}")
        discovery = row.get("discovery") or {}
        for key, values in (
            ("fit", fit_ids),
            ("heldout", heldout_ids),
        ):
            expected_fingerprint = row_set_fingerprint(values)
            if row.get(f"{key}_row_fingerprint") != expected_fingerprint:
                raise RuntimeError(
                    f"Handoff {key} fingerprint mismatch in fold {row.get('fold_key')}"
                )
            if discovery.get(f"{key}_row_fingerprint") != expected_fingerprint:
                raise RuntimeError(
                    f"Discovery {key} fingerprint mismatch in fold {row.get('fold_key')}"
                )
        vocabulary = list(discovery.get("common_vocabulary") or [])
        if int(discovery.get("common_vocabulary_size") or 0) != len(vocabulary):
            raise RuntimeError(f"Vocabulary size mismatch in fold {row.get('fold_key')}")
        if any(str(term).startswith("true_") for term in vocabulary):
            raise RuntimeError("An oracle-prefixed token entered a Stage 1 vocabulary")

        artifacts = discovery.get("artifacts") or {}
        fitted_path = artifact_path(artifacts.get("fitted_context"))
        fit_topics_path = artifact_path(artifacts.get("fit_topic_values"))
        heldout_topics_path = artifact_path(artifacts.get("heldout_topic_values"))
        nuisance_path = artifact_path(artifacts.get("nuisance_predictions"))
        referenced_paths.extend(
            map(
                str,
                [fitted_path, fit_topics_path, heldout_topics_path, nuisance_path],
            )
        )
        ngram_paths = artifacts.get("ngram_scores") or {}
        if set(ngram_paths) != {"treatment", "outcome", "effect"}:
            raise RuntimeError(f"Incomplete n-gram banks in fold {row.get('fold_key')}")
        referenced_paths.extend(
            str(artifact_path(ngram_paths[bank])) for bank in ("treatment", "outcome", "effect")
        )

        bank_metadata = discovery.get("topic_banks") or {}
        with np.load(fit_topics_path) as fit_archive, np.load(
            heldout_topics_path
        ) as heldout_archive:
            for bank in ("treatment", "outcome", "effect"):
                if bank not in bank_metadata:
                    raise RuntimeError(f"Missing {bank} topic bank in fold {row.get('fold_key')}")
                topics = list((bank_metadata.get(bank) or {}).get("topics") or [])
                topic_counts[bank].append(len(topics))
                if any(len(topic.get("terms") or []) != 15 for topic in topics):
                    raise RuntimeError(f"A {bank} topic does not contain exactly 15 terms")
                fit_values = (
                    np.asarray(fit_archive[bank])
                    if bank in fit_archive.files
                    else np.zeros((len(fit_ids), 0))
                )
                heldout_values = (
                    np.asarray(heldout_archive[bank])
                    if bank in heldout_archive.files
                    else np.zeros((len(heldout_ids), 0))
                )
                if fit_values.shape != (len(fit_ids), len(topics)):
                    raise RuntimeError(
                        f"Misaligned fit {bank} topic values in fold {row.get('fold_key')}"
                    )
                if heldout_values.shape != (len(heldout_ids), len(topics)):
                    raise RuntimeError(
                        f"Misaligned held-out {bank} topic values in fold " f"{row.get('fold_key')}"
                    )

        nuisance = pd.read_parquet(nuisance_path)
        fit_nuisance = nuisance.loc[nuisance["prediction_scope"] == "fit_oof"].copy()
        external_nuisance = nuisance.loc[nuisance["prediction_scope"] == "external_heldout"].copy()
        if set(map(int, fit_nuisance["_oci_row_id"])) != set(fit_ids):
            raise RuntimeError(f"Misaligned OOF nuisance rows in fold {row.get('fold_key')}")
        if set(map(int, external_nuisance["_oci_row_id"])) != set(heldout_ids):
            raise RuntimeError(f"Misaligned external nuisance rows in fold {row.get('fold_key')}")
        required_prediction_columns = {"treatment_stacked", "outcome_stacked"}
        if not required_prediction_columns <= set(nuisance.columns):
            raise RuntimeError(f"Missing stacked nuisances in fold {row.get('fold_key')}")
        if not np.isfinite(nuisance[list(required_prediction_columns)].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite nuisance prediction in fold {row.get('fold_key')}")
        fit_set = set(fit_ids)
        selection_policy = str(
            discovery.get("score_selection_label_policy")
            or config.architecture.multi_model_forest.tfidf_topic.score_selection_label_policy
        )
        model_fit_set = set(map(int, discovery.get("model_fit_row_ids") or fit_ids))
        if selection_policy == "nested_fit_calibration" and (
            not model_fit_set
            or not model_fit_set < fit_set
            or discovery.get("registered_heldout_labels_accessed") is not False
            or not discovery.get("selection_frozen_sha256")
        ):
            raise RuntimeError(
                f"Invalid nested TF-IDF selection provenance in fold {row.get('fold_key')}"
            )
        for record in fit_nuisance[["_oci_row_id", "fit_row_ids"]].to_dict(orient="records"):
            training_ids = set(map(int, record["fit_row_ids"]))
            if int(record["_oci_row_id"]) in training_ids or not training_ids <= fit_set:
                raise RuntimeError(
                    f"A fit OOF nuisance prediction includes its own label in fold "
                    f"{row.get('fold_key')}"
                )
        for record in external_nuisance[["_oci_row_id", "fit_row_ids"]].to_dict(orient="records"):
            training_ids = set(map(int, record["fit_row_ids"]))
            expected_training = (
                model_fit_set if selection_policy == "nested_fit_calibration" else fit_set
            )
            if int(record["_oci_row_id"]) in training_ids or training_ids != expected_training:
                raise RuntimeError(
                    f"An external nuisance prediction has invalid fit provenance in "
                    f"fold {row.get('fold_key')}"
                )
        honest_nuisance_rows += int(len(nuisance))

        score_artifact = artifacts.get("topic_score_tests")
        if row.get("scope") == "candidate_selection_inner_fit":
            inner_score_contexts += 1
            score_path = artifact_path(score_artifact)
            referenced_paths.append(str(score_path))
            score_tests = json.loads(score_path.read_text(encoding="utf-8"))
            nested_policy = selection_policy == "nested_fit_calibration"
            if (
                score_tests.get("schema_version") != TOPIC_SCORE_TEST_SCHEMA_VERSION
                or score_tests.get("status") != "completed"
                or (
                    nested_policy
                    and (
                        score_tests.get("uses_heldout_treatment_and_outcome") is not False
                        or score_tests.get("uses_registered_heldout_treatment_and_outcome")
                        is not False
                        or score_tests.get("uses_nested_fit_calibration_treatment_and_outcome")
                        is not True
                        or score_tests.get("score_selection_label_policy")
                        != "nested_fit_calibration"
                        or score_tests.get("selection_frozen_sha256")
                        != discovery.get("selection_frozen_sha256")
                    )
                )
                or (
                    not nested_policy
                    and not bool(score_tests.get("uses_heldout_treatment_and_outcome"))
                )
                or bool(score_tests.get("fits_patient_level_cate_model"))
                or bool(score_tests.get("constructs_divided_pseudo_target"))
            ):
                raise RuntimeError(f"Invalid inner score-test artifact: {score_path}")
            for bank in ("treatment", "outcome", "effect"):
                bank_score = (score_tests.get("banks") or {}).get(bank) or {}
                tests = list(bank_score.get("topic_tests") or [])
                topics = list((bank_metadata.get(bank) or {}).get("topics") or [])
                if len(tests) != len(topics):
                    raise RuntimeError(f"{bank} score/topic mismatch in fold {row.get('fold_key')}")
                topic_ids = {str(topic["topic_id"]) for topic in topics}
                selected_ids = set(map(str, bank_score.get("selected_topic_ids") or []))
                if not selected_ids <= topic_ids:
                    raise RuntimeError(
                        f"Unknown selected {bank} topic in fold {row.get('fold_key')}"
                    )
                calibration = bank_score.get("bootstrap_calibration") or {}
                if not all(
                    bool(calibration.get(field))
                    for field in (
                        "complete_topic_family",
                        "complete_term_group_family",
                        "complete_ngram_family",
                    )
                ):
                    raise RuntimeError(
                        f"Incomplete {bank} multiplier family in fold {row.get('fold_key')}"
                    )
                if any(
                    len(test.get("term_scores") or []) != 15
                    or "topic_standardized_score" not in test
                    or "term_group_primary_p" not in test
                    or "_topic_bootstrap_rows" in test
                    for test in tests
                ):
                    raise RuntimeError(
                        f"Incomplete {bank} topic evidence in fold {row.get('fold_key')}"
                    )
                selected_counts[bank].append(len(selected_ids))
                selected_ngram_counts[bank].append(
                    int(bank_score.get("ngram_selection_count") or 0)
                )
            if bool(config.architecture.multi_model_forest.tfidf_topic.orphan_ngram_enabled):
                orphan = score_tests.get("effect_orphan_ngram_branch") or {}
                clusters = list(orphan.get("clusters") or [])
                selected_clusters = list(orphan.get("selected_clusters") or [])
                selected_ids = set(map(str, orphan.get("selected_cluster_ids") or []))
                all_ids = {str(cluster.get("cluster_id")) for cluster in clusters}
                effect_topic_terms = {
                    str(term.get("term"))
                    for topic in (bank_metadata.get("effect") or {}).get("topics", [])
                    for term in topic.get("terms", [])
                }
                calibration = orphan.get("bootstrap_calibration") or {}
                if (
                    orphan.get("status") != "completed"
                    or (
                        nested_policy
                        and (
                            orphan.get("uses_heldout_treatment_and_outcome") is not False
                            or orphan.get("uses_registered_heldout_treatment_and_outcome")
                            is not False
                            or orphan.get("uses_nested_fit_calibration_treatment_and_outcome")
                            is not True
                        )
                    )
                    or (
                        not nested_policy
                        and not bool(orphan.get("uses_heldout_treatment_and_outcome"))
                    )
                    or bool(orphan.get("fits_patient_level_cate_model"))
                    or not bool(orphan.get("topic_term_exclusion_is_fit_side"))
                    or bool(orphan.get("cluster_construction_uses_heldout_rows_or_labels"))
                    or selected_ids
                    != {str(cluster.get("cluster_id")) for cluster in selected_clusters}
                    or not selected_ids <= all_ids
                    or (
                        bool(clusters)
                        and not bool(calibration.get("complete_term_group_family", False))
                    )
                ):
                    raise RuntimeError(f"Invalid inner orphan n-gram branch: {score_path}")
                for cluster in clusters:
                    terms = list(cluster.get("term_scores") or [])
                    if not 1 <= len(terms) <= 15:
                        raise RuntimeError("An orphan n-gram cluster does not contain 1-15 terms")
                    if effect_topic_terms & {str(term.get("term")) for term in terms}:
                        raise RuntimeError(
                            "An orphan n-gram cluster overlaps a fitted topic summary"
                        )
                orphan_cluster_counts.append(len(clusters))
                selected_orphan_cluster_counts.append(len(selected_clusters))
        elif row.get("scope") == "full_outer_train":
            outer_contexts += 1
            compact_score = discovery.get("topic_score_tests") or {}
            if selection_policy == "nested_fit_calibration":
                score_path = artifact_path(score_artifact)
                referenced_paths.append(str(score_path))
                score_tests = json.loads(score_path.read_text(encoding="utf-8"))
                if (
                    compact_score.get("status") != "completed"
                    or compact_score.get("uses_heldout_treatment_and_outcome") is not False
                    or compact_score.get("uses_registered_heldout_treatment_and_outcome")
                    is not False
                    or compact_score.get("uses_nested_fit_calibration_treatment_and_outcome")
                    is not True
                    or score_tests.get("selection_frozen_sha256")
                    != discovery.get("selection_frozen_sha256")
                ):
                    raise RuntimeError(
                        f"A full-outer context lacks nested label-safe selection in fold "
                        f"{outer_fold}"
                    )
            elif (
                score_artifact is not None
                or compact_score.get("status") != "not_run"
                or bool(compact_score.get("uses_heldout_treatment_and_outcome"))
            ):
                raise RuntimeError(
                    f"A full-outer context exposes score-test labels in fold {outer_fold}"
                )
        else:
            raise RuntimeError(f"Unknown handoff scope: {row.get('scope')!r}")

    outer_test_occurrences: Counter = Counter()
    for outer_fold, fold_rows in sorted(rows_by_outer.items()):
        full = [row for row in fold_rows if row.get("scope") == "full_outer_train"]
        inner = [row for row in fold_rows if row.get("scope") == "candidate_selection_inner_fit"]
        if len(full) != 1 or len(inner) != required_inner:
            raise RuntimeError(
                f"Incomplete exact contexts for outer fold {outer_fold}: "
                f"full={len(full)}/1 inner={len(inner)}/{required_inner}"
            )
        outer_fit = set(map(int, full[0]["fit_row_ids"]))
        outer_heldout = set(map(int, full[0]["heldout_row_ids"]))
        if outer_fit | outer_heldout != valid_row_ids:
            raise RuntimeError(f"Outer fold {outer_fold} does not partition the dataset")
        outer_test_occurrences.update(outer_heldout)
        for inner_row in inner:
            inner_fit = set(map(int, inner_row["fit_row_ids"]))
            inner_heldout = set(map(int, inner_row["heldout_row_ids"]))
            if inner_fit | inner_heldout != outer_fit:
                raise RuntimeError(
                    f"Inner fold {outer_fold}/{inner_row.get('inner_fold')} does not "
                    "partition its outer-training rows"
                )
    if set(outer_test_occurrences) != valid_row_ids or set(outer_test_occurrences.values()) != {1}:
        raise RuntimeError("Outer held-out folds do not form a once-only row partition")

    forbidden_reference_tokens = (
        "htr",
        "sentence_transform",
        "pseudo_target",
        "matched_pair",
        "uplift",
        "raw_text_forest",
    )
    forbidden_references = sorted(
        path
        for path in referenced_paths
        if any(token in Path(path).name.lower() for token in forbidden_reference_tokens)
    )
    if forbidden_references:
        raise RuntimeError(
            f"Forbidden Stage 1 artifact references are present: {forbidden_references[:3]}"
        )

    def value_range(values: Sequence[int]) -> List[int]:
        return [min(values), max(values)] if values else [0, 0]

    return {
        "schema_version": "tfidf_topic_stage2_preflight_v1",
        "status": "passed",
        "handoff_schema_version": HANDOFF_SCHEMA_VERSION,
        "topic_score_test_schema_version": TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "stage1_config_hash": expected_hash,
        "dataset_row_count": n_rows,
        "outer_fold_count": len(rows_by_outer),
        "exact_context_count": len(rows),
        "inner_score_context_count": inner_score_contexts,
        "full_outer_context_count": outer_contexts,
        "topic_count_ranges": {bank: value_range(values) for bank, values in topic_counts.items()},
        "selected_topic_count_ranges": {
            bank: value_range(values) for bank, values in selected_counts.items()
        },
        "selected_ngram_count_ranges": {
            bank: value_range(values) for bank, values in selected_ngram_counts.items()
        },
        "orphan_cluster_count_range": value_range(orphan_cluster_counts),
        "selected_orphan_cluster_count_range": value_range(selected_orphan_cluster_counts),
        "honest_nuisance_prediction_rows_checked": honest_nuisance_rows,
        "outer_test_rows_predicted_once": True,
        "outer_test_score_artifact_count": 0,
        "forbidden_artifact_reference_count": 0,
        "oracle_columns_consumed": False,
        "llm_or_extraction_client_constructed": False,
        "max_variables_per_extraction_request": max_variables,
    }


def build_topic_label_context(
    *,
    outer_fold: int,
    scope: str,
    inner_fold: Optional[int],
    bank: str,
    topic: Dict[str, Any],
    prompt_version: str = TOPIC_LABEL_PROMPT_VERSION,
    uncovered_raw_ngrams: Optional[Sequence[str]] = None,
    current_definitions: Optional[Sequence[Dict[str, Any]]] = None,
    extraction_failures: Optional[Sequence[Dict[str, Any]]] = None,
    score_test_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    terms = list(topic.get("terms") or [])
    if prompt_version == ORPHAN_NGRAM_LABEL_PROMPT_VERSION:
        if not 1 <= len(terms) <= 15:
            raise ValueError("Every orphan n-gram prompt must receive 1-15 terms")
    elif len(terms) != 15:
        raise ValueError("Every topic-label prompt must receive exactly 15 supplied terms")
    role = "effect_modifier" if bank == "effect" else "confounder"
    return {
        "prompt_version": prompt_version,
        "outer_fold": int(outer_fold),
        "inner_fold": None if inner_fold is None else int(inner_fold),
        "scope": str(scope),
        "bank": str(bank),
        "mechanical_role": role,
        "topic_id": str(topic["topic_id"]),
        "evidence_kind": str(
            topic.get("evidence_kind")
            or (
                "orphan_raw_ngram_cluster"
                if prompt_version == ORPHAN_NGRAM_LABEL_PROMPT_VERSION
                else "nmf_topic"
            )
        ),
        "topic_terms": terms,
        "uncovered_raw_ngrams": list(uncovered_raw_ngrams or []),
        "current_canonical_definitions": list(current_definitions or []),
        "extraction_failures": list(extraction_failures or []),
        "heldout_relevance_evidence": dict(score_test_evidence or {}),
        "max_features": 20 if prompt_version == TOPIC_RECOVERY_PROMPT_VERSION else 12,
        "response_contract": {
            "general_topic": "short neutral label, including mixed/weak/artifact when appropriate",
            "topic_quality": "coherent|mixed|weak|administrative_or_artifactual",
            "proposals": [
                {
                    "action": "add",
                    "name": "snake_case_pre_treatment_variable",
                    "type": "categorical|continuous",
                    "categories": ["permitted", "canonical", "values"],
                    "roles": [role],
                    "description": "operational pre-decision extraction definition",
                    "supporting_terms": ["exact supplied term"],
                    "rationale": "how the cited terms represent this feature",
                    "expected_signal": f"evidence organized in the {bank} topic bank",
                }
            ],
        },
    }


def select_topic_recovery_raw_ngrams(
    raw_scores: pd.DataFrame,
    topic: Dict[str, Any],
    *,
    excluded_terms: Sequence[str] = (),
    limit: int = 20,
) -> List[str]:
    """Return bounded, fit-ranked raw evidence tied to one uncovered topic.

    The score table is already ordered using fit-side contrast evidence.  We
    reserve half the request for unrepresented n-grams sharing a meaningful
    token with the topic and half for the global fit-side ranking.  Reserving a
    global share prevents a broad/noisy topic vocabulary from consuming all 20
    slots before the highest-ranked uncovered raw evidence is revisited.
    Held-out labels are never consulted here.
    """
    if int(limit) < 1 or raw_scores.empty:
        return []
    frame = raw_scores
    if "eligible" in frame.columns:
        frame = frame[frame["eligible"].astype(bool)]
    excluded = {str(term) for term in excluded_terms}

    def tokens(value: Any) -> set:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
            if len(token) >= 3 and token not in _RECOVERY_NGRAM_STOPWORDS and not token.isdigit()
        }

    topic_terms = {str(term.get("term") or "") for term in topic.get("terms", [])}
    topic_tokens = set().union(*(tokens(term) for term in topic_terms))
    candidates: List[Tuple[int, int, str]] = []
    for fit_rank, feature in enumerate(frame["feature"].astype(str), start=1):
        if feature in excluded or feature in topic_terms:
            continue
        overlap = len(tokens(feature) & topic_tokens)
        candidates.append((-overlap, fit_rank, feature))
    focused = [row for row in candidates if row[0] < 0]
    focused.sort()
    global_fill = sorted(candidates, key=lambda row: (row[1], row[2]))
    global_slots = max(1, int(limit) // 2)
    focused_slots = max(0, int(limit) - global_slots)
    selected: List[str] = []
    seen: set = set()
    for _negative_overlap, _fit_rank, feature in focused:
        if feature in seen:
            continue
        seen.add(feature)
        selected.append(feature)
        if len(selected) >= focused_slots:
            break
    for _negative_overlap, _fit_rank, feature in global_fill:
        if feature in seen:
            continue
        seen.add(feature)
        selected.append(feature)
        if len(selected) >= int(limit):
            break
    return selected


def render_topic_label_prompt(context: Dict[str, Any]) -> str:
    """Render the one-topic prompt; intentionally contains no forbidden jargon."""
    terms = list(context.get("topic_terms") or [])
    is_orphan = context.get("prompt_version") == ORPHAN_NGRAM_LABEL_PROMPT_VERSION
    if is_orphan:
        if not 1 <= len(terms) <= 15:
            raise ValueError("Orphan n-gram prompt rendering requires 1-15 terms")
    elif len(terms) != 15:
        raise ValueError("Topic prompt rendering requires exactly 15 supplied terms")
    payload = json.dumps(context, indent=2, default=str)
    evidence_description = (
        "These raw text phrases formed a stable, held-out-supported evidence "
        "group that was not represented in the fitted topic summaries. Review "
        f"this one group and its {len(terms)} supplied phrases."
        if is_orphan
        else "Topic modeling was used only to organize candidate text signals. "
        "Review this one topic and its 15 highest-loading supplied terms."
    )
    prompt = f"""You are organizing candidate pre-treatment clinical information for a study of treatment choice, baseline outcome risk, and variation in outcomes after treatment.

{evidence_description} First identify the general clinical concept. Then list the specific, operational clinical features actually represented by the supplied terms.

Rules:
- Cite one or more exact supplied terms for every proposed feature.
- Automated held-out relevance evidence may be supplied; use it to focus on
  represented terms with reproducible signal, while keeping every definition
  grounded in the exact supplied terms.
- A topic may be mixed, weak, administrative, or artifactual; say so and return no features when appropriate.
- Do not decide the analytic role of a feature. The role is assigned mechanically from the topic bank.
- Do not exclude response, survival, toxicity, or outcome-related concepts merely because of their content.
- Every extraction definition must use only information documented before the treatment decision.
- Return JSON only, following the response contract.

Return exactly one JSON object with this top-level shape (never return a bare list):
{{
  "general_topic": "short neutral topic label",
  "topic_quality": "coherent|mixed|weak|administrative_or_artifactual",
  "proposals": [{{
    "action": "add",
    "name": "snake_case_pre_treatment_variable",
    "type": "categorical|continuous",
    "categories": ["canonical category values for categorical variables"],
    "roles": ["{context['mechanical_role']}"],
    "description": "operational extraction definition",
    "supporting_terms": ["one or more exact supplied terms"],
    "rationale": "how those exact terms support the feature",
    "expected_signal": "evidence organized in the {context['bank']} topic bank"
  }}]
}}

Topic context:
{payload}
"""
    if "causal" in prompt.lower():
        raise RuntimeError("Topic-label prompt unexpectedly contains forbidden terminology")
    return prompt


def topic_label_response_issues(response: Any, context: Dict[str, Any]) -> List[str]:
    """Validate the two-part topic label and exact supporting-term provenance."""
    if not isinstance(response, dict):
        return ["topic response must be one JSON object, not a bare list"]
    issues: List[str] = []
    if not str(response.get("general_topic") or "").strip():
        issues.append("missing non-empty general_topic")
    if response.get("topic_quality") not in {
        "coherent",
        "mixed",
        "weak",
        "administrative_or_artifactual",
    }:
        issues.append("invalid or missing topic_quality")
    proposals = response.get("proposals")
    if not isinstance(proposals, list):
        issues.append("proposals must be a list")
        return issues
    supplied = {str(row["term"]) for row in context.get("topic_terms", [])} | {
        str(term) for term in context.get("uncovered_raw_ngrams", [])
    }
    for index, proposal in enumerate(proposals, start=1):
        if not isinstance(proposal, dict):
            issues.append(f"proposal {index} must be an object")
            continue
        supporting = proposal.get("supporting_terms")
        if not isinstance(supporting, list) or not supporting:
            issues.append(f"proposal {index} must cite supporting_terms")
        elif any(str(term) not in supplied for term in supporting):
            issues.append(f"proposal {index} cites a term not supplied in this topic")
    return issues


def _compact_harmonization_candidate(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return label-free source evidence needed for name/value adjudication."""
    supporting_terms: List[str] = []
    topic_ids: List[str] = []
    for provenance in entry.get("provenance", []):
        topic_id = str(provenance.get("topic_id") or "").strip()
        if topic_id:
            topic_ids.append(topic_id)
        for term in provenance.get("supporting_terms", []):
            value = term.get("term") if isinstance(term, dict) else term
            value = str(value or "").strip()
            if value:
                supporting_terms.append(value)
    return {
        "candidate_id": entry["candidate_id"],
        "name": entry["name"],
        "type": entry["type"],
        "categories": entry.get("categories"),
        "description": entry.get("description"),
        "clinical_domain": entry.get("clinical_domain"),
        "parent_object": entry.get("parent_object"),
        "supporting_terms": list(dict.fromkeys(supporting_terms))[:20],
        "source_topic_ids": list(dict.fromkeys(topic_ids))[:20],
        "required_or_prespecified": bool(entry.get("required_or_prespecified", False)),
    }


def render_topic_name_harmonization_prompt(context: Dict[str, Any]) -> str:
    """Render a bounded, role-blind name harmonization/adjudication request."""
    candidates = list(context.get("candidates") or [])
    payload = json.dumps(context, indent=2, default=str)
    return f"""You are harmonizing names for candidate pre-treatment clinical variables. The candidates came from topic-organized text evidence in one fixed training context.

For every supplied candidate, identify its clinical domain and reusable parent object, then make exactly one final decision. Resolve spelling variants, true aliases, base variables, deterministic derivations, subfields, and genuinely distinct concepts. Do not decide or change any analytic role; roles are assigned mechanically elsewhere from source provenance.

Return one JSON object:
{{
  "decisions": [
    {{
      "candidate_id": "exact supplied id",
      "action": "extract|derive|alias/drop|drop",
      "canonical_name": "snake_case executable name or null",
      "clinical_domain": "stable broad clinical domain",
      "parent_object": "specific reusable clinical object",
      "alias_of": "exact supplied candidate name or null",
      "source_names": ["exact supplied candidate names used by a derivation"],
      "derivation": {{"operation": "copy|sum|difference|product|ratio|mean|minimum|maximum", "parameters": {{}}}},
      "reason": "brief evidence-grounded reason"
    }}
  ]
}}

Rules:
- Return exactly one decision for each of the {len(candidates)} supplied candidate_id values and no others.
- Use only final actions extract, derive, alias/drop, or drop. Never return review or defer.
- Preserve a valid, operationally distinct variable as extract. Do not collapse related but different timing, assay, anatomic, severity, or subfield targets.
- Use alias/drop only for a true duplicate and set alias_of to an exact supplied candidate name that should own the combined provenance.
- Use derive only for a deterministic numeric calculation from exact supplied source_names and one allowed operation. Otherwise use extract.
- Drop administrative/artifactual, incoherent, post-decision-only, or non-operational definitions with a concrete reason.
- A required_or_prespecified candidate must remain extract unless it is a valid deterministic derivation; never drop it or make it an alias.
- Use only information documented before the treatment decision.

Training-context candidates:
{payload}
"""


def render_topic_global_dedup_prompt(context: Dict[str, Any]) -> str:
    """Render the final cross-domain deduplication pass."""
    payload = json.dumps(context, indent=2, default=str)
    return f"""You are performing the final global deduplication of already domain-harmonized pre-treatment clinical variables from one fixed training context.

Find only cross-group mistakes: true aliases assigned to different domains or parent objects, a clearly invalid definition, or a valid deterministic derivation that replaces redundant extracted members. Do not restate variables that are already distinct. Do not decide or change analytic roles.

Return one JSON object:
{{
  "resolutions": [
    {{
      "action": "extract|derive|alias/drop|drop",
      "member_names": ["exact supplied canonical names"],
      "canonical_name": "one exact supplied member name",
      "source_names": ["exact supplied canonical names"],
      "derivation": {{"operation": "copy|sum|difference|product|ratio|mean|minimum|maximum", "parameters": {{}}}},
      "reason": "brief reason"
    }}
  ]
}}

Rules:
- Use only supplied names. An omitted name remains a distinct extract target.
- Use alias/drop only for exact clinical equivalence, with canonical_name chosen from member_names.
- Do not merge a base variable with a clinically meaningful subfield, different timing, different unit/assay, different anatomy, or different severity construct.
- Use derive only when all source_names are supplied and the calculation is deterministic.
- Never return review or defer. Return an empty resolutions list when no cross-group change is justified.
- Required/prespecified variables cannot be removed or aliased.

Global registry candidates:
{payload}
"""


def render_topic_value_harmonization_prompt(context: Dict[str, Any]) -> str:
    """Render value-contract fitting or training-value repair for fixed names."""
    candidates = list(context.get("candidates") or [])
    payload = json.dumps(context, indent=2, default=str)
    return f"""You are defining machine-usable value contracts for fixed pre-treatment clinical variables in one training context. Keep every name and its source-defined role unchanged.

Return one JSON object:
{{
  "features": [
    {{
      "name": "exact supplied canonical name",
      "data_type": "categorical|continuous",
      "permitted_categories": ["mutually exclusive canonical values"] or null,
      "canonical_unit": "single canonical unit or null",
      "unit_conversions": {{"source unit": {{"multiply": 1.0, "add": 0.0}}}},
      "category_synonyms": {{"canonical category": ["source synonym"]}},
      "ordinal_order": ["lowest", "...", "highest"] or null,
      "missing_semantics": {{
        "missing": "schema/request failure or unusable value",
        "unknown": "explicitly stated unknown or indeterminate",
        "absent": "explicitly stated absence",
        "not_documented": "not stated before the cutoff"
      }},
      "deterministic_derivation": null or {{"operation": "allowed operation", "source_names": ["exact names"], "parameters": {{}}}},
      "temporal_cutoff": "use only information documented before the treatment decision",
      "description": "operational target, canonical value policy, and unit policy",
      "reason": "brief reason"
    }}
  ]
}}

Rules:
- Return exactly one feature for each of the {len(candidates)} supplied names and no others. Never add, remove, or rename a variable.
- Categorical variables need at least two mutually exclusive categories. Category synonym keys must be permitted categories.
- Continuous variables must have null categories and numeric-only output; qualitative, unknown, or undocumented values are not numbers.
- Define deterministic unit conversions only when clinically valid. Do not guess a unit when the evidence cannot support one.
- Keep missing, unknown, absent, and not_documented semantically distinct even when some are represented by null plus a missingness indicator downstream.
- Ordinal ordering must contain only permitted categories and no duplicates.
- A derivation is allowed only when its exact source names and operation are supplied and deterministic.
- Use the fixed pre-treatment cutoff verbatim.

Value-contract context:
{payload}
"""


def topic_harmonization_response_issues(response: Any, context: Dict[str, Any]) -> List[str]:
    """Validate name/global/value harmonization responses before execution."""
    if not isinstance(response, dict):
        return ["harmonization response must be one JSON object"]
    version = str(context.get("prompt_version") or "")
    candidates = list(context.get("candidates") or [])
    by_id = {str(item.get("candidate_id")): item for item in candidates}
    by_name = {str(item.get("name")): item for item in candidates}
    issues: List[str] = []

    if version == TOPIC_GLOBAL_DEDUP_PROMPT_VERSION:
        resolutions = response.get("resolutions")
        if not isinstance(resolutions, list):
            return ["resolutions must be a list"]
        for index, resolution in enumerate(resolutions, start=1):
            if not isinstance(resolution, dict):
                issues.append(f"resolution {index} must be an object")
                continue
            action = str(resolution.get("action") or "")
            if action not in _NAME_HARMONIZATION_ACTIONS:
                issues.append(f"resolution {index} has invalid action")
            members = resolution.get("member_names")
            if not isinstance(members, list) or not members:
                issues.append(f"resolution {index} needs member_names")
                continue
            unknown = [str(name) for name in members if str(name) not in by_name]
            if unknown:
                issues.append(f"resolution {index} contains unknown member names")
            canonical = str(resolution.get("canonical_name") or "")
            if action in {"extract", "alias/drop", "derive"} and canonical not in members:
                issues.append(f"resolution {index} canonical_name must be a member")
            if action == "derive":
                derivation = resolution.get("derivation")
                sources = resolution.get("source_names")
                if (
                    not isinstance(derivation, dict)
                    or derivation.get("operation") not in _DERIVATION_OPERATIONS
                ):
                    issues.append(f"resolution {index} has invalid derivation")
                if (
                    not isinstance(sources, list)
                    or not sources
                    or any(str(name) not in by_name for name in sources)
                    or canonical in sources
                ):
                    issues.append(f"resolution {index} has invalid source_names")
        return issues

    if version in {
        TOPIC_NAME_HARMONIZATION_PROMPT_VERSION,
    }:
        decisions = response.get("decisions")
        if not isinstance(decisions, list):
            return ["decisions must be a list"]
        seen: List[str] = []
        for index, decision in enumerate(decisions, start=1):
            if not isinstance(decision, dict):
                issues.append(f"decision {index} must be an object")
                continue
            candidate_id = str(decision.get("candidate_id") or "")
            seen.append(candidate_id)
            if candidate_id not in by_id:
                issues.append(f"decision {index} has an unknown candidate_id")
            action = str(decision.get("action") or "")
            if action not in _NAME_HARMONIZATION_ACTIONS:
                issues.append(f"decision {index} has invalid action")
            if not str(decision.get("clinical_domain") or "").strip():
                issues.append(f"decision {index} lacks clinical_domain")
            if not str(decision.get("parent_object") or "").strip():
                issues.append(f"decision {index} lacks parent_object")
            if action in {"extract", "derive"} and not _normalize_feature_name(
                decision.get("canonical_name", "")
            ):
                issues.append(f"decision {index} lacks canonical_name")
            if action == "alias/drop":
                alias = str(decision.get("alias_of") or "")
                if alias not in by_name:
                    issues.append(f"decision {index} alias_of is not supplied")
            if action == "derive":
                derivation = decision.get("derivation")
                sources = decision.get("source_names")
                if (
                    not isinstance(derivation, dict)
                    or derivation.get("operation") not in _DERIVATION_OPERATIONS
                ):
                    issues.append(f"decision {index} has invalid derivation")
                if not isinstance(sources, list) or any(
                    str(name) not in by_name for name in sources
                ):
                    issues.append(f"decision {index} has invalid source_names")
        if len(seen) != len(set(seen)):
            issues.append("candidate_id decisions must be unique")
        if set(seen) != set(by_id):
            issues.append("decisions must cover every supplied candidate_id exactly once")
        return issues

    if version in {
        TOPIC_VALUE_HARMONIZATION_PROMPT_VERSION,
        TOPIC_VALUE_REPAIR_PROMPT_VERSION,
    }:
        features = response.get("features")
        if not isinstance(features, list):
            return ["features must be a list"]
        seen_names: List[str] = []
        for index, feature in enumerate(features, start=1):
            if not isinstance(feature, dict):
                issues.append(f"feature {index} must be an object")
                continue
            name = str(feature.get("name") or "")
            seen_names.append(name)
            if name not in by_name:
                issues.append(f"feature {index} has unknown name")
            data_type = str(feature.get("data_type") or "")
            categories = feature.get("permitted_categories")
            if data_type not in {"categorical", "continuous"}:
                issues.append(f"feature {index} has invalid data_type")
            elif data_type == "categorical" and (
                not isinstance(categories, list)
                or len({str(value) for value in categories if str(value).strip()}) < 2
            ):
                issues.append(f"feature {index} needs at least two categories")
            elif data_type == "continuous" and categories not in (None, []):
                issues.append(f"feature {index} continuous categories must be null")
            synonyms = feature.get("category_synonyms") or {}
            if not isinstance(synonyms, dict):
                issues.append(f"feature {index} category_synonyms must be an object")
            elif isinstance(categories, list) and any(
                str(key) not in {str(value) for value in categories} for key in synonyms
            ):
                issues.append(f"feature {index} synonym key is not a permitted category")
            elif any(not isinstance(values, list) for values in synonyms.values()):
                issues.append(f"feature {index} category synonym values must be lists")
            ordinal = feature.get("ordinal_order")
            if ordinal not in (None, []) and (
                not isinstance(ordinal, list)
                or len(ordinal) != len(set(map(str, ordinal)))
                or not set(map(str, ordinal)).issubset(set(map(str, categories or [])))
            ):
                issues.append(f"feature {index} has invalid ordinal_order")
            semantics = feature.get("missing_semantics")
            if not isinstance(semantics, dict) or set(semantics) != {
                "missing",
                "unknown",
                "absent",
                "not_documented",
            }:
                issues.append(f"feature {index} has incomplete missing_semantics")
            elif len({str(value).strip() for value in semantics.values()}) != 4:
                issues.append(f"feature {index} missing semantics must be distinct")
            if str(feature.get("temporal_cutoff") or "").strip() != (
                "use only information documented before the treatment decision"
            ):
                issues.append(f"feature {index} has invalid temporal_cutoff")
        if len(seen_names) != len(set(seen_names)) or set(seen_names) != set(by_name):
            issues.append("features must cover every supplied name exactly once")
        return issues

    return [f"unknown harmonization prompt version: {version}"]


def _proposal_rows(response: Any) -> List[Dict[str, Any]]:
    if isinstance(response, list):
        return [row for row in response if isinstance(row, dict)]
    if isinstance(response, dict):
        for key in ("proposals", "features", "clinical_features"):
            value = response.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _candidate_from_response(
    row: Dict[str, Any],
    *,
    context: Dict[str, Any],
    topic: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if str(row.get("action") or "add").strip().lower() not in {"add", "extract"}:
        return None
    name = _normalize_feature_name(row.get("name", ""))
    feature_type = str(row.get("type") or "").strip().lower()
    if not name or feature_type not in {"categorical", "continuous"}:
        return None
    supplied = {str(term["term"]) for term in context["topic_terms"]} | {
        str(term) for term in context.get("uncovered_raw_ngrams", [])
    }
    supporting = [
        str(value).strip()
        for value in (row.get("supporting_terms") or [])
        if str(value).strip() in supplied
    ]
    if not supporting:
        # A response is not executable unless its provenance is traceable.
        return None
    categories = row.get("categories")
    if feature_type == "categorical":
        if not isinstance(categories, list) or len(categories) < 2:
            categories = ["absent", "present", "unknown", "not_documented"]
        categories = list(
            dict.fromkeys(str(value).strip() for value in categories if str(value).strip())
        )
        if len(categories) < 2:
            return None
    else:
        categories = None
    role = str(context["mechanical_role"])
    term_rows = {str(item["term"]): item for item in topic["terms"]}
    return {
        "action": "extract",
        "name": name,
        "type": feature_type,
        "categories": categories,
        "roles": [role],
        "description": str(row.get("description") or name.replace("_", " ")).strip(),
        "supporting_terms": supporting,
        "rationale": str(row.get("rationale") or "").strip(),
        "provenance": [
            {
                "outer_fold": int(context["outer_fold"]),
                "inner_fold": context.get("inner_fold"),
                "scope": context["scope"],
                "bank": context["bank"],
                "topic_id": context["topic_id"],
                "evidence_kind": context.get("evidence_kind", "nmf_topic"),
                "supporting_terms": [
                    {
                        "term": term,
                        "loading": float(term_rows.get(term, {}).get("loading", 0.0)),
                        "rank": int(
                            term_rows.get(term, {}).get(
                                "screen_rank",
                                term_rows.get(term, {}).get("fit_rank", 0),
                            )
                        ),
                        "signed_score": float(
                            term_rows.get(term, {}).get(
                                "signed_score",
                                term_rows.get(term, {}).get("fit_signed_score", 0.0),
                            )
                        ),
                    }
                    for term in supporting
                ],
                "objective": context["bank"],
                "heldout_relevance_evidence": dict(context.get("heldout_relevance_evidence") or {}),
            }
        ],
    }


def harmonize_topic_candidates(
    candidates: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Perform the deterministic spelling/punctuation pass before agent adjudication."""
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    dropped: List[Dict[str, Any]] = []
    for candidate in candidates:
        name = _normalize_feature_name(candidate.get("name", ""))
        if not name:
            dropped.append({"candidate": candidate, "reason": "empty_normalized_name"})
            continue
        domain = _normalize_feature_name(candidate.get("clinical_domain") or name.split("_", 1)[0])
        grouped[(domain, name)].append({**candidate, "name": name})

    first_pass: List[Dict[str, Any]] = []
    for (_domain, name), members in grouped.items():
        types = Counter(str(member.get("type")) for member in members)
        feature_type = types.most_common(1)[0][0]
        compatible = [member for member in members if member.get("type") == feature_type]
        incompatible = [member for member in members if member.get("type") != feature_type]
        for member in incompatible:
            dropped.append({"candidate": member, "reason": "irreconcilable_type_conflict"})
        roles = list(
            dict.fromkeys(role for member in compatible for role in member.get("roles", []))
        )
        categories = None
        if feature_type == "categorical":
            categories = list(
                dict.fromkeys(
                    str(value)
                    for member in compatible
                    for value in (member.get("categories") or [])
                )
            )
            for required in ("unknown", "not_documented"):
                if required not in categories:
                    categories.append(required)
        provenance = [entry for member in compatible for entry in member.get("provenance", [])]
        descriptions = [str(member.get("description") or "") for member in compatible]
        description = max(descriptions, key=len, default=name.replace("_", " "))
        requested_actions = [str(member.get("action") or "extract") for member in compatible]
        action = "derive" if "derive" in requested_actions else "extract"
        derivation = next(
            (
                member.get("derivation")
                or (member.get("value_contract") or {}).get("deterministic_derivation")
                for member in compatible
                if member.get("derivation")
                or (member.get("value_contract") or {}).get("deterministic_derivation")
            ),
            None,
        )
        parent_object = _normalize_feature_name(
            next(
                (
                    member.get("parent_object")
                    for member in compatible
                    if member.get("parent_object")
                ),
                "_".join(name.split("_")[:2]),
            )
        )
        first_pass.append(
            {
                "action": action,
                "name": name,
                "type": feature_type,
                "categories": categories,
                "roles": roles,
                "description": description,
                "provenance": provenance,
                "clinical_domain": domain,
                "parent_object": parent_object or domain,
                "derivation": derivation,
                "required_or_prespecified": any(
                    bool(member.get("required_or_prespecified"))
                    or any(
                        item.get("bank") == "prespecified" for item in member.get("provenance", [])
                    )
                    for member in compatible
                ),
            }
        )

    # Final global pass deliberately ignores the earlier domain grouping.
    global_by_name: Dict[str, Dict[str, Any]] = {}
    for candidate in first_pass:
        name = candidate["name"]
        if name not in global_by_name:
            global_by_name[name] = candidate
            continue
        current = global_by_name[name]
        current["roles"] = list(dict.fromkeys([*current["roles"], *candidate["roles"]]))
        current["provenance"].extend(candidate["provenance"])
        current["required_or_prespecified"] = bool(
            current.get("required_or_prespecified") or candidate.get("required_or_prespecified")
        )
        if current["type"] == "categorical":
            current["categories"] = list(
                dict.fromkeys([*(current["categories"] or []), *(candidate["categories"] or [])])
            )

    registry: List[Dict[str, Any]] = []
    for candidate in sorted(global_by_name.values(), key=lambda row: row["name"]):
        aliases = None
        if candidate["type"] == "categorical":
            categories = candidate["categories"] or []
            aliases = {}
            if "present" in categories:
                aliases["present"] = ["yes", "documented", "positive"]
            if "absent" in categories:
                aliases["absent"] = ["no", "negative", "explicitly absent"]
            aliases["unknown"] = ["uncertain", "indeterminate"]
            aliases["not_documented"] = ["not stated", "not recorded"]
        value_contract = {
            "data_type": candidate["type"],
            "permitted_categories": candidate.get("categories"),
            "canonical_unit": None,
            "unit_conversions": {},
            "category_synonyms": aliases or {},
            "ordinal_order": None,
            "missing_semantics": {
                "missing": "no executable extraction value was produced",
                "unknown": "the document explicitly says the value is unknown or indeterminate",
                "absent": "the document explicitly states the condition is absent",
                "not_documented": "the document does not state the requested value before cutoff",
            },
            "deterministic_derivation": candidate.get("derivation"),
            "temporal_cutoff": "use only information documented before the treatment decision",
        }
        candidate["candidate_id"] = stable_hash(
            {
                "name": candidate["name"],
                "type": candidate["type"],
                "provenance": candidate["provenance"],
            }
        )[:20]
        candidate["value_contract"] = value_contract
        candidate["contract_hash"] = stable_hash(
            {
                "name": candidate["name"],
                "type": candidate["type"],
                "categories": candidate.get("categories"),
                "description": candidate["description"],
                "value_contract": value_contract,
            }
        )
        registry.append(candidate)
    return registry, dropped


def _refresh_registry_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one executable registry record and refresh its cache contract."""
    result = dict(entry)
    result["name"] = _normalize_feature_name(result.get("name", ""))
    result["clinical_domain"] = _normalize_feature_name(
        result.get("clinical_domain") or result["name"].split("_", 1)[0]
    )
    result["parent_object"] = _normalize_feature_name(
        result.get("parent_object") or "_".join(result["name"].split("_")[:2])
    )
    result["action"] = (
        str(result.get("action"))
        if str(result.get("action")) in {"extract", "derive"}
        else "extract"
    )
    result["roles"] = list(dict.fromkeys(map(str, result.get("roles") or [])))
    result["provenance"] = list(result.get("provenance") or [])
    result["required_or_prespecified"] = bool(result.get("required_or_prespecified"))
    data_type = str(result.get("type") or "categorical")
    result["type"] = data_type if data_type in {"categorical", "continuous"} else "categorical"
    if result["type"] == "continuous":
        result["categories"] = None
    else:
        categories = list(
            dict.fromkeys(
                str(value).strip()
                for value in (result.get("categories") or [])
                if str(value).strip()
            )
        )
        if len(categories) < 2:
            categories = ["absent", "present", "unknown", "not_documented"]
        result["categories"] = categories

    contract = dict(result.get("value_contract") or {})
    semantics = dict(contract.get("missing_semantics") or {})
    semantics.setdefault("missing", "no executable extraction value was produced")
    semantics.setdefault(
        "unknown", "the document explicitly says the value is unknown or indeterminate"
    )
    semantics.setdefault("absent", "the document explicitly states the condition is absent")
    semantics.setdefault(
        "not_documented", "the document does not state the requested value before cutoff"
    )
    contract.update(
        {
            "data_type": result["type"],
            "permitted_categories": result.get("categories"),
            "canonical_unit": contract.get("canonical_unit"),
            "unit_conversions": dict(contract.get("unit_conversions") or {}),
            "category_synonyms": dict(contract.get("category_synonyms") or {}),
            "ordinal_order": contract.get("ordinal_order"),
            "missing_semantics": semantics,
            "deterministic_derivation": (
                result.get("derivation")
                if result["action"] == "derive"
                else contract.get("deterministic_derivation")
            ),
            "temporal_cutoff": ("use only information documented before the treatment decision"),
        }
    )
    result["value_contract"] = contract
    result["derivation"] = contract.get("deterministic_derivation")
    result["candidate_id"] = str(
        result.get("candidate_id")
        or stable_hash({"name": result["name"], "provenance": result["provenance"]})[:20]
    )
    result["contract_hash"] = stable_hash(
        {
            "action": result["action"],
            "name": result["name"],
            "type": result["type"],
            "categories": result.get("categories"),
            "description": result.get("description"),
            "value_contract": result["value_contract"],
        }
    )
    return result


def _registry_entry_state_hash(entry: Dict[str, Any]) -> str:
    """Hash both the extraction contract and its modeling/evidence state.

    ``contract_hash`` deliberately excludes provenance and roles because those
    fields do not change the extraction prompt.  Recovery, however, must still
    recognize a newly supported topic or role as a registry change: it affects
    coverage diagnostics and whether the extracted column enters W, X, or both.
    """
    refreshed = _refresh_registry_entry(entry)
    provenance_hashes = sorted(
        stable_hash(provenance) for provenance in refreshed.get("provenance", [])
    )
    return stable_hash(
        {
            "name": refreshed["name"],
            "contract_hash": refreshed["contract_hash"],
            "roles": sorted(refreshed.get("roles", [])),
            "provenance_hashes": provenance_hashes,
            "required_or_prespecified": bool(refreshed.get("required_or_prespecified")),
        }
    )


def _merge_executable_registry_entries(
    entries: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Merge renamed/aliased executable entries without losing provenance or roles."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for raw in entries:
        entry = _refresh_registry_entry(raw)
        if entry["name"]:
            grouped[entry["name"]].append(entry)
    merged: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for name, members in grouped.items():
        type_counts = Counter(member["type"] for member in members)
        selected_type = type_counts.most_common(1)[0][0]
        compatible = [member for member in members if member["type"] == selected_type]
        for member in members:
            if member["type"] != selected_type:
                dropped.append(
                    {
                        "candidate": member,
                        "action": "drop",
                        "reason": "irreconcilable_type_conflict_after_name_harmonization",
                    }
                )
        base = max(compatible, key=lambda item: len(str(item.get("description") or "")))
        result = dict(base)
        result["name"] = name
        result["roles"] = list(
            dict.fromkeys(role for member in compatible for role in member.get("roles", []))
        )
        result["provenance"] = [
            provenance for member in compatible for provenance in member.get("provenance", [])
        ]
        result["required_or_prespecified"] = any(
            member.get("required_or_prespecified", False) for member in compatible
        )
        result["action"] = (
            "derive"
            if any(member.get("action") == "derive" for member in compatible)
            else "extract"
        )
        if selected_type == "categorical":
            result["categories"] = list(
                dict.fromkeys(
                    str(value)
                    for member in compatible
                    for value in (member.get("categories") or [])
                )
            )
        merged.append(_refresh_registry_entry(result))
    return sorted(merged, key=lambda entry: entry["name"]), dropped


def _valid_derivation(
    derivation: Any,
    source_names: Sequence[str],
    available_names: Sequence[str],
    *,
    target_name: Optional[str] = None,
) -> bool:
    if not isinstance(derivation, dict):
        return False
    operation = str(derivation.get("operation") or "")
    sources = list(dict.fromkeys(_normalize_feature_name(name) for name in source_names))
    if operation not in _DERIVATION_OPERATIONS or not sources:
        return False
    if any(source not in set(available_names) for source in sources):
        return False
    if target_name and _normalize_feature_name(target_name) in sources:
        return False
    if operation == "copy" and len(sources) != 1:
        return False
    if operation in {"difference", "ratio"} and len(sources) != 2:
        return False
    parameters = derivation.get("parameters") or {}
    return isinstance(parameters, dict)


_DISTINCT_SUBFIELD_NAME_TOKENS = {
    "assay",
    "auscultation",
    "bilateral",
    "completed",
    "completion",
    "count",
    "cycle",
    "cycles",
    "date",
    "dosage",
    "dose",
    "doublet",
    "duration",
    "frequency",
    "grade",
    "histology",
    "laterality",
    "left",
    "line",
    "lobe",
    "location",
    "mutation",
    "number",
    "palpation",
    "percentage",
    "postoperative",
    "preoperative",
    "procedure",
    "ratio",
    "regimen",
    "response",
    "right",
    "severity",
    "side",
    "site",
    "size",
    "source",
    "stage",
    "subtype",
    "timing",
    "type",
    "unit",
    "variant",
    "volume",
}


def _alias_preserves_distinct_subfields(source: Dict[str, Any], target: Dict[str, Any]) -> bool:
    """Fail conservative when an alias would erase an explicit subfield."""
    if source.get("type") != target.get("type"):
        return False
    source_markers = set(source.get("name", "").split("_")) & _DISTINCT_SUBFIELD_NAME_TOKENS
    target_markers = set(target.get("name", "").split("_")) & _DISTINCT_SUBFIELD_NAME_TOKENS
    return source_markers == target_markers


def apply_topic_name_harmonization(
    registry: Sequence[Dict[str, Any]],
    responses: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply exhaustive batch decisions while preserving mechanical role unions."""
    current = [_refresh_registry_entry(entry) for entry in registry]
    by_id = {entry["candidate_id"]: entry for entry in current}
    by_name = {entry["name"]: entry for entry in current}
    decisions: Dict[str, Dict[str, Any]] = {}
    for response in responses:
        for decision in response.get("decisions", []):
            decisions[str(decision.get("candidate_id"))] = decision

    # An LLM may still call a base variable and a named subfield aliases despite
    # the prompt prohibition. Preserve both executable variables in that case;
    # later empirical review can evaluate them without irreversible evidence loss.
    for candidate_id, entry in by_id.items():
        decision = decisions.get(candidate_id)
        if not decision or str(decision.get("action") or "") != "alias/drop":
            continue
        alias_name = _normalize_feature_name(decision.get("alias_of") or "")
        alias_target = by_name.get(alias_name)
        if alias_target is None or _alias_preserves_distinct_subfields(entry, alias_target):
            continue
        decisions[candidate_id] = {
            **decision,
            "action": "extract",
            "canonical_name": entry["name"],
            "alias_of": None,
            "reason": "retained_distinct_subfield_guardrail",
            "_guardrail_reason": (f"unsafe alias to {alias_name!r} would erase a named subfield"),
        }

    dropped: List[Dict[str, Any]] = []
    name_targets: Dict[str, Optional[str]] = {}
    for entry in current:
        decision = decisions.get(entry["candidate_id"])
        if decision is None:
            dropped.append(
                {
                    "candidate": entry,
                    "action": "drop",
                    "reason": "missing_name_harmonization_decision",
                }
            )
            name_targets[entry["name"]] = None
            continue
        action = str(decision.get("action") or "")
        if entry.get("required_or_prespecified") and action in {"alias/drop", "drop"}:
            action = "extract"
        if action == "drop":
            dropped.append(
                {
                    "candidate": entry,
                    "action": "drop",
                    "reason": str(decision.get("reason") or "agent_harmonization_drop"),
                }
            )
            name_targets[entry["name"]] = None
        elif action == "alias/drop":
            name_targets[entry["name"]] = _normalize_feature_name(decision.get("alias_of", ""))
        else:
            name_targets[entry["name"]] = _normalize_feature_name(
                decision.get("canonical_name") or entry["name"]
            )

    def resolve_target(name: str) -> Optional[str]:
        seen: set = set()
        target: Optional[str] = name
        while target in name_targets and name_targets[target] != target:
            if target in seen:
                return None
            seen.add(target)
            target = name_targets[target]
            if target is None:
                return None
        return target

    rewritten: List[Dict[str, Any]] = []
    pending_derivations: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for entry in current:
        decision = decisions.get(entry["candidate_id"])
        target = resolve_target(entry["name"])
        if decision is None or target is None:
            continue
        action = str(decision.get("action") or "extract")
        if entry.get("required_or_prespecified") and action in {"alias/drop", "drop"}:
            action = "extract"
        if action == "drop":
            continue
        updated = dict(entry)
        updated["name"] = target
        if decision.get("_guardrail_reason"):
            updated["harmonization_audit"] = [
                *list(updated.get("harmonization_audit") or []),
                {
                    "action": "extract",
                    "reason": decision["_guardrail_reason"],
                },
            ]
        updated["clinical_domain"] = _normalize_feature_name(
            decision.get("clinical_domain") or entry.get("clinical_domain")
        )
        updated["parent_object"] = _normalize_feature_name(
            decision.get("parent_object") or entry.get("parent_object")
        )
        if action == "alias/drop":
            updated["action"] = "extract"
            dropped.append(
                {
                    "candidate": entry,
                    "action": "alias/drop",
                    "alias_of": target,
                    "reason": str(decision.get("reason") or "true_alias"),
                }
            )
        elif action == "derive":
            updated["action"] = "derive"
            pending_derivations.append((updated, decision))
        else:
            updated["action"] = "extract"
        rewritten.append(updated)

    preliminary, conflicts = _merge_executable_registry_entries(rewritten)
    dropped.extend(conflicts)
    available = [entry["name"] for entry in preliminary]
    derivation_by_name: Dict[str, Dict[str, Any]] = {}
    for updated, decision in pending_derivations:
        source_names = [
            resolve_target(_normalize_feature_name(name))
            for name in (decision.get("source_names") or [])
        ]
        source_names = [name for name in source_names if name]
        derivation = dict(decision.get("derivation") or {})
        if _valid_derivation(derivation, source_names, available, target_name=updated["name"]):
            derivation["source_names"] = source_names
            derivation_by_name[updated["name"]] = derivation
        else:
            dropped.append(
                {
                    "candidate": updated,
                    "action": "drop",
                    "reason": "invalid_or_unavailable_deterministic_derivation",
                }
            )
    result: List[Dict[str, Any]] = []
    for entry in preliminary:
        if entry["action"] == "derive":
            derivation = derivation_by_name.get(entry["name"])
            if derivation is None:
                continue
            entry["derivation"] = derivation
        result.append(_refresh_registry_entry(entry))
    return result, dropped


def _global_dedup_document(entry: Dict[str, Any]) -> str:
    """Compact word/character TF-IDF document used only for global blocking."""
    contract = dict(entry.get("value_contract") or {})
    categories = " ".join(str(value) for value in (entry.get("categories") or []))
    synonyms = " ".join(
        str(value)
        for values in (contract.get("category_synonyms") or {}).values()
        for value in values
    )
    return " ".join(
        str(value)
        for value in (
            str(entry.get("name") or "").replace("_", " "),
            entry.get("description") or "",
            entry.get("type") or "",
            categories,
            synonyms,
        )
        if str(value).strip()
    )


def _global_dedup_name_tokens(name: str) -> set:
    generic = {
        "baseline",
        "current",
        "documented",
        "history",
        "indicator",
        "measurement",
        "presence",
        "pretreatment",
        "status",
        "value",
    }
    return {
        token
        for token in _normalize_feature_name(name).split("_")
        if token and token not in generic
    }


def build_topic_global_dedup_blocks(
    registry: Sequence[Dict[str, Any]],
    *,
    max_block_size: int = _GLOBAL_DEDUP_BLOCK_SIZE,
    min_similarity: float = _GLOBAL_DEDUP_MIN_SIMILARITY,
    max_neighbors: int = _GLOBAL_DEDUP_MAX_NEIGHBORS,
) -> List[List[Dict[str, Any]]]:
    """Block likely cross-group aliases without ever creating an unbounded prompt.

    Word/character TF-IDF is used only to propose comparisons. Every final merge is
    still adjudicated by the global harmonization prompt, and omitted entries remain
    distinct. Blocks may overlap so that every retained high-similarity edge is seen.
    """
    if not 2 <= int(max_block_size) <= 10:
        raise ValueError("global dedup max_block_size must be between 2 and 10")
    if int(max_neighbors) < 1:
        raise ValueError("global dedup max_neighbors must be positive")
    entries = sorted(
        (_refresh_registry_entry(entry) for entry in registry),
        key=lambda entry: entry["name"],
    )
    if len(entries) < 2:
        return []
    vectors = np.asarray(
        _parsimony_tfidf_semantic_vectors([_global_dedup_document(entry) for entry in entries]),
        dtype=float,
    )
    similarities = np.asarray(vectors @ vectors.T, dtype=float)
    np.fill_diagonal(similarities, -np.inf)
    groups = [(entry.get("clinical_domain"), entry.get("parent_object")) for entry in entries]
    name_tokens = [_global_dedup_name_tokens(entry["name"]) for entry in entries]

    edges: Dict[Tuple[int, int], float] = {}
    for left, entry in enumerate(entries):
        ranked = np.argsort(-similarities[left], kind="stable")
        retained = 0
        for right_value in ranked:
            right = int(right_value)
            if right == left or groups[right] == groups[left]:
                continue
            # True aliases must have compatible executable value types.
            if entries[right]["type"] != entry["type"]:
                continue
            union = name_tokens[left] | name_tokens[right]
            lexical = len(name_tokens[left] & name_tokens[right]) / len(union) if union else 0.0
            semantic = float(similarities[left, right])
            if semantic < float(min_similarity) and lexical < 0.5:
                continue
            edge = (min(left, right), max(left, right))
            edges[edge] = max(edges.get(edge, -np.inf), semantic, lexical)
            retained += 1
            if retained >= int(max_neighbors):
                break

    uncovered = set(edges)
    blocks: List[List[Dict[str, Any]]] = []
    seen_blocks: set = set()
    while uncovered:
        seed = max(
            uncovered,
            key=lambda edge: (
                edges[edge],
                entries[edge[0]]["name"],
                entries[edge[1]]["name"],
            ),
        )
        members = set(seed)
        while len(members) < int(max_block_size):
            touching = [edge for edge in uncovered if (edge[0] in members) ^ (edge[1] in members)]
            if not touching:
                break
            edge = max(
                touching,
                key=lambda value: (
                    edges[value],
                    entries[value[0]]["name"],
                    entries[value[1]]["name"],
                ),
            )
            members.update(edge)
        block_key = tuple(sorted(members))
        if block_key not in seen_blocks:
            seen_blocks.add(block_key)
            blocks.append([entries[index] for index in block_key])
        uncovered = {edge for edge in uncovered if not (edge[0] in members and edge[1] in members)}
    return blocks


def combine_topic_global_dedup_responses(
    responses: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Combine overlapping sparse blocks conservatively and deterministically."""
    combined: List[Dict[str, Any]] = []
    claimed: set = set()
    for response in responses:
        for raw in response.get("resolutions", []):
            resolution = dict(raw)
            action = str(resolution.get("action") or "")
            if action == "extract":
                continue
            members = list(
                dict.fromkeys(
                    _normalize_feature_name(name)
                    for name in (resolution.get("member_names") or [])
                    if _normalize_feature_name(name)
                )
            )
            if not members or claimed.intersection(members):
                continue
            resolution["member_names"] = members
            resolution["canonical_name"] = _normalize_feature_name(
                resolution.get("canonical_name") or ""
            )
            resolution["source_names"] = [
                _normalize_feature_name(name) for name in (resolution.get("source_names") or [])
            ]
            combined.append(resolution)
            claimed.update(members)
    return {"resolutions": combined}


def apply_topic_global_dedup(
    registry: Sequence[Dict[str, Any]], response: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply sparse final-global resolutions; omitted entries remain distinct."""
    current = [_refresh_registry_entry(entry) for entry in registry]
    by_name = {entry["name"]: entry for entry in current}
    overrides: Dict[str, Dict[str, Any]] = {}
    for resolution in response.get("resolutions", []):
        action = str(resolution.get("action") or "")
        members = [_normalize_feature_name(name) for name in resolution.get("member_names", [])]
        canonical = _normalize_feature_name(resolution.get("canonical_name") or "")
        if action == "alias/drop":
            for name in members:
                if name not in by_name:
                    continue
                overrides[name] = {
                    "candidate_id": by_name[name]["candidate_id"],
                    "action": "extract" if name == canonical else "alias/drop",
                    "canonical_name": canonical if name == canonical else name,
                    "clinical_domain": by_name[name].get("clinical_domain"),
                    "parent_object": by_name[name].get("parent_object"),
                    "alias_of": None if name == canonical else canonical,
                    "source_names": [],
                    "derivation": None,
                    "reason": resolution.get("reason"),
                }
        elif action == "drop":
            for name in members:
                if name not in by_name:
                    continue
                overrides[name] = {
                    "candidate_id": by_name[name]["candidate_id"],
                    "action": "drop",
                    "canonical_name": name,
                    "clinical_domain": by_name[name].get("clinical_domain"),
                    "parent_object": by_name[name].get("parent_object"),
                    "alias_of": None,
                    "source_names": [],
                    "derivation": None,
                    "reason": resolution.get("reason"),
                }
        elif action == "derive" and canonical in by_name:
            overrides[canonical] = {
                "candidate_id": by_name[canonical]["candidate_id"],
                "action": "derive",
                "canonical_name": canonical,
                "clinical_domain": by_name[canonical].get("clinical_domain"),
                "parent_object": by_name[canonical].get("parent_object"),
                "alias_of": None,
                "source_names": resolution.get("source_names") or [],
                "derivation": resolution.get("derivation"),
                "reason": resolution.get("reason"),
            }
    decisions = []
    for entry in current:
        decisions.append(
            overrides.get(
                entry["name"],
                {
                    "candidate_id": entry["candidate_id"],
                    "action": entry["action"],
                    "canonical_name": entry["name"],
                    "clinical_domain": entry.get("clinical_domain"),
                    "parent_object": entry.get("parent_object"),
                    "source_names": (entry.get("derivation") or {}).get("source_names", []),
                    "derivation": entry.get("derivation"),
                    "reason": "confirmed_distinct_by_global_pass",
                },
            )
        )
    return apply_topic_name_harmonization(current, [{"decisions": decisions}])


def _default_category_synonyms(categories: Sequence[str]) -> Dict[str, List[str]]:
    """Return conservative aliases for common structured clinical categories.

    These aliases are deterministic parts of the value contract rather than
    parser-only conveniences, so they are visible in extraction prompts and
    participate in contract/cache hashes.
    """
    available = {_normalize_feature_name(category): str(category) for category in categories}
    defaults: Dict[str, List[str]] = {
        "first_line": ["first line", "1st-line", "1st line", "line 1"],
        "second_line": ["second line", "2nd-line", "2nd line", "line 2"],
        "third_line": ["third line", "3rd-line", "3rd line", "line 3"],
        "subsequent_line": [
            "subsequent line",
            "later-line",
            "later line",
            "fourth-line",
            "fourth line",
            "4th-line",
            "4th line",
            "fifth-line",
            "fifth line",
            "5th-line",
            "5th line",
            "sixth-line",
            "sixth line",
            "6th-line",
            "6th line",
            "seventh-line",
            "seventh line",
            "7th-line",
            "7th line",
            "eighth-line",
            "eighth line",
            "8th-line",
            "8th line",
            "ninth-line",
            "ninth line",
            "9th-line",
            "9th line",
            "tenth-line",
            "tenth line",
            "10th-line",
            "10th line",
            "fourth line or later",
            "4th line or later",
        ],
        "yes": ["true", "present"],
        "no": ["false", "absent"],
        "present": ["yes", "true"],
        "absent": ["no", "false", "not present"],
        "adherent": ["compliant", "adherence documented"],
        "non_adherent": [
            "nonadherent",
            "not adherent",
            "non-compliant",
            "noncompliant",
        ],
    }
    return {available[key]: aliases for key, aliases in defaults.items() if key in available}


def apply_topic_value_harmonization(
    registry: Sequence[Dict[str, Any]], responses: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply complete structured value contracts without changing names or roles."""
    features: Dict[str, Dict[str, Any]] = {}
    for response in responses:
        for feature in response.get("features", []):
            features[str(feature.get("name"))] = feature
    harmonized: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    available_names = [entry["name"] for entry in registry]
    for raw in registry:
        entry = _refresh_registry_entry(raw)
        feature = features.get(entry["name"])
        if feature is None:
            dropped.append(
                {
                    "candidate": entry,
                    "action": "drop",
                    "reason": "missing_value_harmonization_contract",
                }
            )
            continue
        data_type = str(feature.get("data_type") or entry["type"])
        categories = (
            None
            if data_type == "continuous"
            else list(
                dict.fromkeys(
                    str(value).strip()
                    for value in (feature.get("permitted_categories") or [])
                    if str(value).strip()
                )
            )
        )
        if categories is not None:
            # Missing is reserved for schema/request failure.  Categorical
            # contracts must encode an explicitly unknown value separately from
            # a value that was simply not documented before the cutoff.
            for required_state in ("unknown", "not_documented"):
                if required_state not in categories:
                    categories.append(required_state)
        conversions: Dict[str, Dict[str, float]] = {}
        for unit, rule in (feature.get("unit_conversions") or {}).items():
            if not isinstance(rule, dict):
                continue
            try:
                multiply = float(rule.get("multiply", 1.0))
                add = float(rule.get("add", 0.0))
            except (TypeError, ValueError):
                continue
            if np.isfinite(multiply) and np.isfinite(add):
                conversions[str(unit)] = {"multiply": multiply, "add": add}
        category_synonyms = {
            str(key): list(dict.fromkeys(map(str, values or [])))
            for key, values in (feature.get("category_synonyms") or {}).items()
            if categories is not None and str(key) in categories
        }
        if categories is not None:
            for category, aliases in _default_category_synonyms(categories).items():
                category_synonyms[category] = list(
                    dict.fromkeys([*(category_synonyms.get(category) or []), *aliases])
                )
            category_synonyms.setdefault("unknown", ["unknown", "uncertain", "indeterminate"])
            category_synonyms.setdefault(
                "not_documented",
                ["not documented", "not stated", "not recorded"],
            )
        contract = {
            "data_type": data_type,
            "permitted_categories": categories,
            "canonical_unit": (
                str(feature.get("canonical_unit")).strip()
                if feature.get("canonical_unit") not in (None, "")
                else None
            ),
            "unit_conversions": conversions,
            "category_synonyms": category_synonyms,
            "ordinal_order": (list(map(str, feature.get("ordinal_order") or [])) or None),
            "missing_semantics": dict(feature.get("missing_semantics") or {}),
            "deterministic_derivation": (
                feature.get("deterministic_derivation") if entry["action"] == "derive" else None
            ),
            "temporal_cutoff": ("use only information documented before the treatment decision"),
        }
        if entry["action"] == "derive":
            derivation = contract.get("deterministic_derivation") or entry.get("derivation")
            sources = list((derivation or {}).get("source_names") or [])
            if not _valid_derivation(
                derivation, sources, available_names, target_name=entry["name"]
            ):
                dropped.append(
                    {
                        "candidate": entry,
                        "action": "drop",
                        "reason": "invalid_value_contract_derivation",
                    }
                )
                continue
            contract["deterministic_derivation"] = derivation
            entry["derivation"] = derivation
        entry["type"] = data_type
        entry["categories"] = categories
        entry["description"] = str(feature.get("description") or entry.get("description") or "")
        entry["value_contract"] = contract
        harmonized.append(_refresh_registry_entry(entry))
    return sorted(harmonized, key=lambda item: item["name"]), dropped


def registry_specs(registry: Sequence[Dict[str, Any]]) -> List[ExplicitFeatureSpec]:
    specs: List[ExplicitFeatureSpec] = []
    for entry in registry:
        contract = dict(entry.get("value_contract") or {})
        policy_parts: List[str] = []
        if contract.get("canonical_unit"):
            policy_parts.append(f"Return numeric values in {contract['canonical_unit']}")
        if contract.get("unit_conversions"):
            conversions = "; ".join(
                f"{unit}: multiply by {rule.get('multiply', 1.0)} then add {rule.get('add', 0.0)}"
                for unit, rule in contract["unit_conversions"].items()
            )
            policy_parts.append(f"Apply these deterministic unit conversions: {conversions}")
        if contract.get("ordinal_order"):
            policy_parts.append("Use this ordinal order: " + " < ".join(contract["ordinal_order"]))
        if entry.get("action") == "derive" and contract.get("deterministic_derivation"):
            derivation = contract["deterministic_derivation"]
            policy_parts.append(
                "This value is computed deterministically after extracting sources "
                f"{derivation.get('source_names', [])}; do not extract it independently"
            )
        description = (
            f"clinical_domain={entry.get('clinical_domain', 'general')}; "
            f"parent_object={entry.get('parent_object', entry['name'])}: "
            f"{entry['description']} Use only information documented before the treatment "
            "decision. Keep an explicitly unknown value distinct from an explicitly absent "
            "condition and from a value that is not documented. "
            + (". ".join(policy_parts) + "." if policy_parts else "")
        )
        specs.append(
            ExplicitFeatureSpec(
                name=entry["name"],
                type=entry["type"],
                categories=entry.get("categories"),
                roles=list(entry["roles"]),
                description=description,
                value_aliases=contract.get("category_synonyms") or None,
            )
        )
    return specs


def apply_registry_derivations(
    frame: pd.DataFrame, registry: Sequence[Dict[str, Any]]
) -> pd.DataFrame:
    """Materialize validated deterministic derivations from extracted source columns."""
    result = frame.copy()
    pending = {entry["name"]: entry for entry in registry if entry.get("action") == "derive"}
    while pending:
        progressed = False
        for name, entry in list(pending.items()):
            derivation = (
                (entry.get("value_contract") or {}).get("deterministic_derivation")
                or entry.get("derivation")
                or {}
            )
            sources = list(map(str, derivation.get("source_names") or []))
            source_columns = [f"explicit_feat_{source}" for source in sources]
            if not all(column in result.columns for column in source_columns):
                continue
            source_missing = []
            for column in source_columns:
                missing_column = f"{column}_missing"
                if missing_column in result.columns:
                    source_missing.append(result[missing_column].astype(bool).to_numpy())
                else:
                    source_missing.append(result[column].isna().to_numpy())
            missing = (
                np.logical_or.reduce(source_missing)
                if source_missing
                else np.ones(len(result), dtype=bool)
            )
            operation = str(derivation.get("operation") or "")
            parameters = dict(derivation.get("parameters") or {})
            if operation == "copy":
                values = result[source_columns[0]].copy()
            else:
                arrays = [
                    pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
                    for column in source_columns
                ]
                missing = missing | np.logical_or.reduce(
                    [~np.isfinite(values) for values in arrays]
                )
                if operation == "sum":
                    values = np.sum(np.vstack(arrays), axis=0)
                elif operation == "difference":
                    values = arrays[0] - arrays[1]
                elif operation == "product":
                    values = np.prod(np.vstack(arrays), axis=0)
                elif operation == "ratio":
                    power = float(parameters.get("denominator_power", 1.0))
                    denominator = np.power(arrays[1], power)
                    missing = missing | ~np.isfinite(denominator) | (denominator == 0.0)
                    values = np.divide(
                        arrays[0],
                        denominator,
                        out=np.full(len(result), np.nan, dtype=float),
                        where=~missing,
                    )
                elif operation == "mean":
                    values = np.mean(np.vstack(arrays), axis=0)
                elif operation == "minimum":
                    values = np.min(np.vstack(arrays), axis=0)
                elif operation == "maximum":
                    values = np.max(np.vstack(arrays), axis=0)
                else:
                    raise ValueError(f"Unsupported deterministic derivation operation: {operation}")
                scale = float(parameters.get("multiply", 1.0))
                offset = float(parameters.get("add", 0.0))
                values = values * scale + offset
            value_column = f"explicit_feat_{name}"
            missing_column = f"{value_column}_missing"
            if isinstance(values, pd.Series):
                values = values.copy()
                values.loc[missing] = None
            else:
                values = np.asarray(values)
                if np.issubdtype(values.dtype, np.number):
                    values = values.astype(float)
                    values[missing] = np.nan
                else:
                    values = values.astype(object)
                    values[missing] = None
            result[value_column] = values
            result[missing_column] = missing.astype(bool)
            del pending[name]
            progressed = True
        if not progressed:
            raise RuntimeError(
                "Canonical registry contains an unresolved or cyclic deterministic "
                f"derivation: {sorted(pending)}"
            )
    return result


def _role_matrix(
    fit_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
    *,
    role: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    fit_values = _columns_to_feature_dicts(fit_df, list(specs))
    heldout_values = _columns_to_feature_dicts(heldout_df, list(specs))
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    fit_matrix, names = get_raw_explicit_features(
        fit_values, list(specs), continuous_means=means, continuous_stds=stds, role=role
    )
    heldout_matrix, _ = get_raw_explicit_features(
        heldout_values, list(specs), continuous_means=means, continuous_stds=stds, role=role
    )
    if not names:
        return (
            np.zeros((len(fit_df), 0), dtype=float),
            np.zeros((len(heldout_df), 0), dtype=float),
            [],
        )
    return (
        np.asarray(fit_matrix, dtype=float).reshape(len(fit_df), len(names)),
        np.asarray(heldout_matrix, dtype=float).reshape(len(heldout_df), len(names)),
        names,
    )


def _fit_nuisance_from_structured(
    fit_matrix: np.ndarray,
    heldout_matrix: np.ndarray,
    fit_values: np.ndarray,
    *,
    binary: bool,
) -> np.ndarray:
    if fit_matrix.shape[1] == 0 or (binary and len(np.unique(fit_values.astype(int))) < 2):
        return np.full(len(heldout_matrix), float(np.mean(fit_values)), dtype=float)
    if binary:
        model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000).fit(
            fit_matrix, fit_values.astype(int)
        )
        return model.predict_proba(heldout_matrix)[:, 1]
    return Ridge(alpha=10.0).fit(fit_matrix, fit_values).predict(heldout_matrix)


def _nuisance_benchmark(
    metadata: Dict[str, Any],
    heldout_ids: Sequence[int],
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    outcome_binary: bool,
) -> Dict[str, Any]:
    predictions = pd.read_parquet(metadata["artifacts"]["nuisance_predictions"])
    predictions = predictions[predictions["prediction_scope"] == "external_heldout"]
    predictions = predictions.set_index("_oci_row_id").loc[list(heldout_ids)]
    return {
        "predictions": predictions,
        "metrics": {
            "treatment": {
                "stacked_metrics": nuisance_metrics(
                    treatment,
                    predictions["treatment_stacked"].to_numpy(dtype=float),
                    binary=True,
                )
            },
            "outcome": {
                "stacked_metrics": nuisance_metrics(
                    outcome,
                    predictions["outcome_stacked"].to_numpy(dtype=float),
                    binary=outcome_binary,
                )
            },
        },
    }


def _topic_evidence_mass(topic: Dict[str, Any]) -> float:
    """Return unsigned topic evidence mass from fixed Stage 1 loadings/scores."""
    mass = float(
        sum(
            abs(float(term.get("loading", 0.0)) * float(term.get("signed_score", 0.0)))
            for term in topic.get("terms", [])
        )
    )
    if np.isfinite(mass) and mass > 0.0:
        return mass
    # A zero signed diagnostic should not make a fitted topic disappear from
    # the audit.  Loading mass is a deterministic, unsigned fallback.
    loading_mass = float(
        sum(abs(float(term.get("loading", 0.0))) for term in topic.get("terms", []))
    )
    return loading_mass if np.isfinite(loading_mass) and loading_mass > 0.0 else 0.0


def _registry_topic_keys(entry: Dict[str, Any]) -> set:
    return {
        (str(provenance.get("bank")), str(provenance.get("topic_id")))
        for provenance in entry.get("provenance", [])
        if str(provenance.get("bank")) in {"treatment", "outcome", "effect"}
        and str(provenance.get("topic_id") or "")
    }


def _registry_effect_terms(entry: Dict[str, Any]) -> set:
    return {
        str(term.get("term") if isinstance(term, dict) else term)
        for provenance in entry.get("provenance", [])
        if provenance.get("bank") == "effect"
        for term in provenance.get("supporting_terms", [])
        if str(term.get("term") if isinstance(term, dict) else term).strip()
    }


def select_initial_topic_evidence_registry(
    registry: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    *,
    coverage_target: float = 0.80,
    highest_ranked_effect_count: int = 20,
    fixed_policy_priority_names: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Build the additive review's initial working registry without a count cap.

    Topic labeling can produce several alternative operationalizations for one
    mixed topic.  Extracting every alternative before *any* held-out review is
    both wasteful and contrary to the intended evidence-gated recovery loop.
    This deterministic set-cover pass starts review with the smallest greedy
    evidence cover that reaches the configured mass target in every bank and
    the same target among mappable high-ranked effect n-grams.  All remaining
    valid contracts stay in a fold-local deferred pool and may only be added;
    they are never treated as harmonization drops.
    """
    target = float(coverage_target)
    if not 0.0 <= target <= 1.0:
        raise ValueError("initial topic evidence coverage target must be in [0, 1]")
    entries = sorted(
        (_refresh_registry_entry(entry) for entry in registry),
        key=lambda entry: entry["name"],
    )
    banks = ("treatment", "outcome", "effect")
    masses: Dict[Tuple[str, str], float] = {}
    for bank in banks:
        for topic in (metadata.get("topic_banks", {}).get(bank, {}) or {}).get("topics", []):
            masses[(bank, str(topic["topic_id"]))] = _topic_evidence_mass(topic)
    orphan_branch = (metadata.get("topic_score_tests") or {}).get(
        "effect_orphan_ngram_branch"
    ) or {}
    for cluster in orphan_branch.get("selected_clusters") or []:
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        masses[("effect", cluster_id)] = max(
            float(cluster.get("quadratic_statistic_per_rank") or 0.0),
            float(cluster.get("maximum_absolute_standardized_score") or 0.0),
            1e-12,
        )

    topic_sets = [
        {key for key in _registry_topic_keys(entry) if key in masses} for entry in entries
    ]
    operational_topics = set().union(*topic_sets) if topic_sets else set()
    operational_totals = {
        bank: float(
            sum(
                mass for key, mass in masses.items() if key[0] == bank and key in operational_topics
            )
        )
        for bank in banks
    }
    all_totals = {
        bank: float(sum(mass for key, mass in masses.items() if key[0] == bank)) for bank in banks
    }

    raw_path = metadata.get("artifacts", {}).get("ngram_scores", {}).get("effect")
    highest_ranked: List[str] = []
    if (
        orphan_branch.get("status") != "completed"
        and raw_path
        and Path(raw_path).exists()
        and int(highest_ranked_effect_count) > 0
    ):
        raw_scores = pd.read_parquet(raw_path)
        if "eligible" in raw_scores.columns:
            raw_scores = raw_scores[raw_scores["eligible"].astype(bool)]
        highest_ranked = (
            raw_scores.head(int(highest_ranked_effect_count))["feature"].astype(str).tolist()
        )
    effect_term_sets = [_registry_effect_terms(entry) for entry in entries]
    operational_effect_terms = set().union(*effect_term_sets) if effect_term_sets else set()
    mappable_highest = [term for term in highest_ranked if term in operational_effect_terms]

    fixed_priority = {_normalize_feature_name(name) for name in (fixed_policy_priority_names or [])}
    selected: set = {
        index
        for index, entry in enumerate(entries)
        if bool(entry.get("required_or_prespecified")) or entry["name"] in fixed_priority
    }
    covered_topics = set().union(*(topic_sets[index] for index in selected)) if selected else set()
    covered_effect_terms = (
        set().union(*(effect_term_sets[index] for index in selected)) if selected else set()
    )

    def bank_fraction(bank: str) -> float:
        denominator = operational_totals[bank]
        if denominator <= 0.0:
            return 1.0
        return float(sum(masses[key] for key in covered_topics if key[0] == bank) / denominator)

    def raw_fraction() -> float:
        if not mappable_highest:
            return 1.0
        return float(len(set(mappable_highest) & covered_effect_terms) / len(set(mappable_highest)))

    while any(bank_fraction(bank) + 1e-12 < target for bank in banks) or (
        raw_fraction() + 1e-12 < target
    ):
        candidates: List[Tuple[float, int, int, str, int]] = []
        for index, entry in enumerate(entries):
            if index in selected:
                continue
            normalized_gain = 0.0
            new_topics = topic_sets[index] - covered_topics
            for bank in banks:
                if bank_fraction(bank) + 1e-12 >= target:
                    continue
                denominator = operational_totals[bank]
                if denominator > 0.0:
                    normalized_gain += (
                        sum(masses[key] for key in new_topics if key[0] == bank) / denominator
                    )
            new_raw = effect_term_sets[index] & set(mappable_highest) - covered_effect_terms
            if raw_fraction() + 1e-12 < target and mappable_highest:
                normalized_gain += len(new_raw) / len(set(mappable_highest))
            candidates.append(
                (
                    float(normalized_gain),
                    len(new_topics) + len(new_raw),
                    len(entry.get("provenance", [])),
                    entry["name"],
                    index,
                )
            )
        best = max(candidates, default=None)
        if best is None or best[0] <= 0.0:
            break
        index = best[-1]
        selected.add(index)
        covered_topics.update(topic_sets[index])
        covered_effect_terms.update(effect_term_sets[index])

    # A selected deterministic derivation is executable only with all of its
    # source contracts.  Close that dependency graph without applying a cap.
    by_name = {entry["name"]: index for index, entry in enumerate(entries)}
    changed = True
    while changed:
        changed = False
        for index in list(selected):
            derivation = (
                entries[index].get("derivation")
                or (entries[index].get("value_contract") or {}).get("deterministic_derivation")
                or {}
            )
            for source_name in derivation.get("source_names", []):
                source_index = by_name.get(_normalize_feature_name(source_name))
                if source_index is not None and source_index not in selected:
                    selected.add(source_index)
                    covered_topics.update(topic_sets[source_index])
                    covered_effect_terms.update(effect_term_sets[source_index])
                    changed = True

    active = [entry for index, entry in enumerate(entries) if index in selected]
    deferred = [entry for index, entry in enumerate(entries) if index not in selected]
    bank_audit: Dict[str, Any] = {}
    for bank in banks:
        operational = operational_totals[bank]
        covered = float(sum(masses[key] for key in covered_topics if key[0] == bank))
        bank_audit[bank] = {
            "covered_operational_mass": covered,
            "operational_nonartifactual_mass": operational,
            "all_fitted_topic_mass": all_totals[bank],
            "coverage_fraction": 1.0 if operational <= 0.0 else covered / operational,
            "all_topic_coverage_fraction": (
                1.0 if all_totals[bank] <= 0.0 else covered / all_totals[bank]
            ),
            "operational_topic_ids": sorted(key[1] for key in operational_topics if key[0] == bank),
            "excluded_topic_ids_without_operational_candidate": sorted(
                key[1] for key in masses if key[0] == bank and key not in operational_topics
            ),
        }
    audit = {
        "schema_version": "tfidf_topic_initial_review_registry_v1",
        "selection_rule": (
            "greedy_unsigned_topic_and_selected_orphan_cluster_mass_cover"
            if orphan_branch.get("status") == "completed"
            else "greedy_unsigned_topic_and_mappable_effect_ngram_mass_cover"
        ),
        "coverage_target": target,
        "has_global_feature_count_cap": False,
        "n_canonical_candidate_contracts": len(entries),
        "n_initial_review_contracts": len(active),
        "n_deferred_valid_contracts": len(deferred),
        "initial_names": [entry["name"] for entry in active],
        "deferred_names": [entry["name"] for entry in deferred],
        "banks": bank_audit,
        "highest_ranked_effect_ngrams": highest_ranked,
        "mappable_highest_ranked_effect_ngrams": mappable_highest,
        "unmapped_or_nonoperational_highest_ranked_effect_ngrams": [
            term for term in highest_ranked if term not in operational_effect_terms
        ],
        "preserved_mappable_highest_ranked_effect_ngrams": [
            term for term in mappable_highest if term in covered_effect_terms
        ],
        "mappable_highest_ranked_effect_ngram_preservation": raw_fraction(),
        "required_or_prespecified_names": [
            entry["name"] for entry in active if entry.get("required_or_prespecified")
        ],
        "fixed_inner_policy_priority_names": sorted(
            entry["name"] for entry in active if entry["name"] in fixed_priority
        ),
        "deferred_contracts_remain_eligible_for_additive_review": True,
    }
    return active, deferred, audit


def select_deferred_review_additions(
    active_registry: Sequence[Dict[str, Any]],
    deferred_registry: Sequence[Dict[str, Any]],
    gate: Dict[str, Any],
    diagnostic: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    max_additions: int = 20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Choose evidence-relevant deferred contracts for one additive review round."""
    limit = int(max_additions)
    if limit < 0 or limit > 20:
        raise ValueError("one recovery round may add at most 20 deferred contracts")
    failed = [row for row in gate.get("criteria", []) if not row.get("passed", False)]
    relevant_banks: set = set()
    for criterion in failed:
        if criterion.get("family") == "effect":
            relevant_banks.add("effect")
        elif criterion.get("family") == "nuisance" and criterion.get("target") in {
            "treatment",
            "outcome",
        }:
            relevant_banks.add(str(criterion["target"]))
    if not relevant_banks or limit == 0:
        return [], {
            "relevant_banks": sorted(relevant_banks),
            "selected_names": [],
            "reason": "no_failed_diagnostic_family_or_zero_round_limit",
        }

    masses: Dict[Tuple[str, str], float] = {}
    for bank in ("treatment", "outcome", "effect"):
        for topic in (metadata.get("topic_banks", {}).get(bank, {}) or {}).get("topics", []):
            masses[(bank, str(topic["topic_id"]))] = _topic_evidence_mass(topic)
    orphan_branch = (metadata.get("topic_score_tests") or {}).get(
        "effect_orphan_ngram_branch"
    ) or {}
    for cluster in orphan_branch.get("selected_clusters") or []:
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if cluster_id:
            masses[("effect", cluster_id)] = max(
                float(cluster.get("quadratic_statistic_per_rank") or 0.0),
                float(cluster.get("maximum_absolute_standardized_score") or 0.0),
                1e-12,
            )
    covered_topics = (
        set().union(*(_registry_topic_keys(entry) for entry in active_registry))
        if active_registry
        else set()
    )
    coverage = diagnostic.get("effect_coverage", {})
    missing_effect_terms = (
        set()
        if orphan_branch.get("status") == "completed"
        else set(coverage.get("mappable_highest_ranked_raw_ngrams", []))
        - set(coverage.get("preserved_highest_ranked_raw_ngrams", []))
    )

    remaining = [
        _refresh_registry_entry(entry)
        for entry in deferred_registry
        if _registry_topic_keys(entry) & {key for key in masses if key[0] in relevant_banks}
    ]
    selected: List[Dict[str, Any]] = []
    while remaining and len(selected) < limit:
        ranked: List[Tuple[float, int, int, str, Dict[str, Any]]] = []
        for entry in remaining:
            keys = {
                key
                for key in _registry_topic_keys(entry)
                if key[0] in relevant_banks and key in masses
            }
            new_keys = keys - covered_topics
            new_mass = float(sum(masses[key] for key in new_keys))
            # Distinct features from an already-covered mixed topic can still be
            # the variable needed by a nuisance gate.  Their own cited evidence
            # therefore provides a small, deterministic secondary score.
            cited_mass = float(
                sum(
                    abs(float(term.get("loading", 0.0)) * float(term.get("signed_score", 0.0)))
                    for provenance in entry.get("provenance", [])
                    if str(provenance.get("bank")) in relevant_banks
                    for term in provenance.get("supporting_terms", [])
                    if isinstance(term, dict)
                )
            )
            new_effect_terms = _registry_effect_terms(entry) & missing_effect_terms
            score = new_mass + 0.01 * cited_mass + float(len(new_effect_terms))
            ranked.append(
                (
                    score,
                    len(new_keys) + len(new_effect_terms),
                    len(entry.get("provenance", [])),
                    entry["name"],
                    entry,
                )
            )
        best = max(ranked, default=None)
        if best is None or best[0] <= 0.0:
            break
        entry = best[-1]
        selected.append(entry)
        covered_topics.update(_registry_topic_keys(entry))
        missing_effect_terms.difference_update(_registry_effect_terms(entry))
        remaining = [row for row in remaining if row["name"] != entry["name"]]

    # Include deterministic sources when available, while respecting the
    # explicit per-round limit.  A derivation without sources is not added.
    active_names = {entry["name"] for entry in active_registry}
    deferred_by_name = {
        entry["name"]: _refresh_registry_entry(entry) for entry in deferred_registry
    }
    executable: List[Dict[str, Any]] = []
    selected_names: set = set()
    for entry in selected:
        derivation = (
            entry.get("derivation")
            or (entry.get("value_contract") or {}).get("deterministic_derivation")
            or {}
        )
        needed = [
            deferred_by_name[name]
            for raw_name in derivation.get("source_names", [])
            if (name := _normalize_feature_name(raw_name)) not in active_names
            and name in deferred_by_name
            and name not in selected_names
        ]
        if len(executable) + len(needed) + 1 > limit:
            continue
        executable.extend(needed)
        selected_names.update(item["name"] for item in needed)
        if entry["name"] not in selected_names:
            executable.append(entry)
            selected_names.add(entry["name"])
    return executable, {
        "relevant_banks": sorted(relevant_banks),
        "selected_names": [entry["name"] for entry in executable],
        "remaining_deferred_count": len(deferred_registry) - len(executable),
        "selection_rule": "marginal_failed_family_topic_term_evidence",
        "maximum_new_contracts": limit,
    }


def _effect_evidence_coverage(
    registry: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    candidate_evidence_universe: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    bank = metadata["topic_banks"].get("effect") or {}
    topics = bank.get("topics") or []
    supported_topics = {
        provenance["topic_id"]
        for entry in registry
        for provenance in entry.get("provenance", [])
        if provenance.get("bank") == "effect"
    }
    supported_terms = {
        str(term.get("term"))
        for entry in registry
        for provenance in entry.get("provenance", [])
        if provenance.get("bank") == "effect"
        for term in provenance.get("supporting_terms", [])
    }
    masses: Dict[str, float] = {}
    for topic in topics:
        mass = sum(
            abs(float(term.get("loading", 0.0)) * float(term.get("signed_score", 0.0)))
            for term in topic.get("terms", [])
        )
        masses[str(topic["topic_id"])] = mass
    operational_topic_ids: Optional[set] = None
    operational_terms: Optional[set] = None
    if candidate_evidence_universe is not None:
        operational_topic_ids = {
            str(provenance.get("topic_id"))
            for entry in candidate_evidence_universe
            for provenance in entry.get("provenance", [])
            if provenance.get("bank") == "effect"
        }
        operational_terms = {
            str(term.get("term") if isinstance(term, dict) else term)
            for entry in candidate_evidence_universe
            for provenance in entry.get("provenance", [])
            if provenance.get("bank") == "effect"
            for term in provenance.get("supporting_terms", [])
        }
    score_selected_topic_ids = set(
        map(
            str,
            (
                (metadata.get("topic_score_tests") or {})
                .get("banks", {})
                .get("effect", {})
                .get("selected_topic_ids", [])
            ),
        )
    )
    if operational_topic_ids is None:
        denominator_topics = set(masses)
    elif score_selected_topic_ids:
        # The score-test shortlist is the evidence universe handed to the
        # agent.  A topic cannot disappear from the coverage denominator just
        # because the agent failed to produce an executable candidate for it.
        # Additive recovery topics join that universe once operationalized.
        denominator_topics = set(masses) & (score_selected_topic_ids | operational_topic_ids)
    else:
        denominator_topics = set(masses) & operational_topic_ids
    total = float(sum(masses[topic_id] for topic_id in denominator_topics))
    covered = float(
        sum(masses[topic_id] for topic_id in denominator_topics if topic_id in supported_topics)
    )
    score_path = metadata.get("artifacts", {}).get("ngram_scores", {}).get("effect")
    highest_ranked: List[str] = []
    if score_path and Path(score_path).exists():
        raw_scores = pd.read_parquet(score_path)
        if "eligible" in raw_scores.columns:
            raw_scores = raw_scores[raw_scores["eligible"].astype(bool)]
        highest_ranked = raw_scores.head(20)["feature"].astype(str).tolist()
    mappable_highest = (
        highest_ranked
        if operational_terms is None
        else [term for term in highest_ranked if term in operational_terms]
    )
    # Preservation is assessed against the complete stable top-ranked raw
    # evidence, not only terms an initial agent happened to operationalize.
    # Otherwise an omitted but strong n-gram vanishes from the denominator and
    # the recovery gate can pass without ever revisiting it.
    preserved_highest = [term for term in highest_ranked if term in supported_terms]
    orphan_branch = (metadata.get("topic_score_tests") or {}).get(
        "effect_orphan_ngram_branch"
    ) or {}
    selected_orphan_clusters = list(orphan_branch.get("selected_clusters") or [])
    orphan_masses = {
        str(cluster.get("cluster_id")): max(
            float(cluster.get("quadratic_statistic_per_rank") or 0.0),
            float(cluster.get("maximum_absolute_standardized_score") or 0.0),
            1e-12,
        )
        for cluster in selected_orphan_clusters
        if str(cluster.get("cluster_id") or "").strip()
    }
    orphan_total = float(sum(orphan_masses.values()))
    orphan_covered = float(
        sum(mass for cluster_id, mass in orphan_masses.items() if cluster_id in supported_topics)
    )
    uncovered_orphan_ids = sorted(set(orphan_masses) - supported_topics)
    return {
        "covered_mass": covered,
        "total_mass": total,
        "coverage_fraction": 1.0 if total <= 0.0 else covered / total,
        "covered_topic_ids": sorted(supported_topics & denominator_topics),
        "uncovered_topic_ids": sorted(
            (denominator_topics - supported_topics) | set(uncovered_orphan_ids)
        ),
        "score_test_shortlist_topic_ids": sorted(score_selected_topic_ids),
        "all_fitted_topic_mass": float(sum(masses.values())),
        "shortlist_to_all_fitted_mass_fraction": (
            1.0
            if not masses or float(sum(masses.values())) <= 0.0
            else total / float(sum(masses.values()))
        ),
        "excluded_topic_ids_without_operational_candidate": sorted(
            set(masses) - denominator_topics
        ),
        "highest_ranked_raw_ngrams": highest_ranked,
        "mappable_highest_ranked_raw_ngrams": mappable_highest,
        "unmapped_or_nonoperational_highest_ranked_raw_ngrams": [
            term for term in highest_ranked if term not in set(mappable_highest)
        ],
        "preserved_highest_ranked_raw_ngrams": preserved_highest,
        "highest_ranked_raw_ngram_preservation": (
            1.0 if not highest_ranked else len(preserved_highest) / len(highest_ranked)
        ),
        "orphan_ngram_branch_status": orphan_branch.get("status"),
        "selected_orphan_cluster_ids": sorted(orphan_masses),
        "covered_orphan_cluster_ids": sorted(set(orphan_masses) & supported_topics),
        "uncovered_orphan_cluster_ids": uncovered_orphan_ids,
        "orphan_cluster_covered_mass": orphan_covered,
        "orphan_cluster_total_mass": orphan_total,
        "orphan_cluster_coverage_fraction": (
            1.0 if orphan_total <= 0.0 else orphan_covered / orphan_total
        ),
        "raw_effect_evidence_weak": bool(
            bank.get("weak_or_unstable_raw_evidence", False)
            or (total <= 0.0 and orphan_total <= 0.0)
        ),
    }


def structured_heldout_diagnostic(
    *,
    fit_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    registry: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    config: AppliedInferenceConfig,
    candidate_evidence_universe: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    specs = registry_specs(registry)
    extraction_summary: List[Dict[str, Any]] = []
    for spec in specs:
        value_column = f"explicit_feat_{spec.name}"
        missing_column = f"{value_column}_missing"
        missing = (
            fit_df[missing_column].astype(bool)
            if missing_column in fit_df.columns
            else fit_df[value_column].isna()
        )
        observed = fit_df.loc[~missing, value_column]
        extraction_summary.append(
            {
                "name": spec.name,
                "coverage": float(1.0 - missing.mean()),
                "n_unique_observed": int(observed.nunique(dropna=True)),
            }
        )
    w_fit, w_heldout, w_names = _role_matrix(fit_df, heldout_df, specs, role="confounder")
    x_fit, x_heldout, x_names = _role_matrix(fit_df, heldout_df, specs, role="effect_modifier")
    t_fit = fit_df[config.treatment_column].to_numpy(dtype=float)
    y_fit = fit_df[config.outcome_column].to_numpy(dtype=float)
    t_heldout = heldout_df[config.treatment_column].to_numpy(dtype=float)
    y_heldout = heldout_df[config.outcome_column].to_numpy(dtype=float)
    outcome_binary = str(config.outcome_type).lower() != "continuous"
    structured_e = _fit_nuisance_from_structured(w_fit, w_heldout, t_fit, binary=True)
    structured_m = _fit_nuisance_from_structured(w_fit, w_heldout, y_fit, binary=outcome_binary)
    treatment_metrics = nuisance_metrics(t_heldout, structured_e, binary=True)
    outcome_metrics = nuisance_metrics(y_heldout, structured_m, binary=outcome_binary)
    benchmark = _nuisance_benchmark(
        metadata,
        heldout_df["_oci_row_id"].astype(int).tolist(),
        t_heldout,
        y_heldout,
        outcome_binary=outcome_binary,
    )

    coverage = _effect_evidence_coverage(
        registry,
        metadata,
        candidate_evidence_universe=candidate_evidence_universe,
    )
    reconstruction: Dict[str, Any] = {
        "mean_correlation": None,
        "topic_correlations": [],
        "n_topics": 0,
    }
    fit_topics = np.load(metadata["artifacts"]["fit_topic_values"])
    heldout_topics = np.load(metadata["artifacts"]["heldout_topic_values"])
    if "effect" in fit_topics.files and x_fit.shape[1] > 0:
        fit_values = np.asarray(fit_topics["effect"], dtype=float)
        heldout_values = np.asarray(heldout_topics["effect"], dtype=float)
        effect_topics = list(
            (metadata.get("topic_banks", {}).get("effect") or {}).get("topics", [])
        )
        selected_effect_ids = set(
            map(
                str,
                (
                    (metadata.get("topic_score_tests", {}).get("banks", {}))
                    .get("effect", {})
                    .get("selected_topic_ids", [])
                ),
            )
        )
        if candidate_evidence_universe is not None:
            # Recovery rounds can introduce a score-ranked topic that was not
            # in the initial shortlist.  Once it has an operational candidate,
            # its held-out reconstruction belongs in the same diagnostic gate
            # as the initially selected topics.
            selected_effect_ids.update(
                str(provenance.get("topic_id"))
                for entry in candidate_evidence_universe
                for provenance in entry.get("provenance", [])
                if provenance.get("bank") == "effect" and provenance.get("topic_id") is not None
            )
        if selected_effect_ids:
            selected_indices = [
                index
                for index, topic in enumerate(effect_topics)
                if str(topic.get("topic_id")) in selected_effect_ids
            ]
            if selected_indices:
                fit_values = fit_values[:, selected_indices]
                heldout_values = heldout_values[:, selected_indices]
        predicted = np.asarray(
            Ridge(alpha=10.0).fit(x_fit, fit_values).predict(x_heldout),
            dtype=float,
        )
        if predicted.ndim == 1:
            predicted = predicted[:, None]
        if heldout_values.ndim == 1:
            heldout_values = heldout_values[:, None]
        correlations: List[Optional[float]] = []
        for index in range(fit_values.shape[1]):
            if np.std(predicted[:, index]) > 0.0 and np.std(heldout_values[:, index]) > 0.0:
                correlations.append(
                    float(np.corrcoef(predicted[:, index], heldout_values[:, index])[0, 1])
                )
            else:
                correlations.append(None)
        finite = [value for value in correlations if value is not None and np.isfinite(value)]
        reconstruction = {
            "mean_correlation": None if not finite else float(np.mean(finite)),
            "topic_correlations": correlations,
            "n_topics": int(fit_values.shape[1]),
            "selected_topic_ids": [
                str(effect_topics[index].get("topic_id"))
                for index in (
                    selected_indices
                    if selected_effect_ids and selected_indices
                    else range(len(effect_topics))
                )
            ],
            "rmse": float(np.sqrt(mean_squared_error(heldout_values, predicted))),
        }

    contrast = {"mean_sign_agreement": None, "feature_names": x_names}
    nuisance_frame = pd.read_parquet(metadata["artifacts"]["nuisance_predictions"])
    fit_nuisance = (
        nuisance_frame[nuisance_frame["prediction_scope"] == "fit_oof"]
        .set_index("_oci_row_id")
        .loc[fit_df["_oci_row_id"].astype(int)]
    )
    heldout_nuisance = (
        nuisance_frame[nuisance_frame["prediction_scope"] == "external_heldout"]
        .set_index("_oci_row_id")
        .loc[heldout_df["_oci_row_id"].astype(int)]
    )
    if x_fit.shape[1] > 0:
        fit_scores = cohort_contrast_scores(
            x_fit,
            x_names,
            t_fit,
            y_fit,
            fit_nuisance["treatment_stacked"],
            fit_nuisance["outcome_stacked"],
        )
        heldout_scores = cohort_contrast_scores(
            x_heldout,
            x_names,
            t_heldout,
            y_heldout,
            heldout_nuisance["treatment_stacked"],
            heldout_nuisance["outcome_stacked"],
        )
        agreement = np.sign(fit_scores["signed_score"]) == np.sign(heldout_scores["signed_score"])
        contrast = {
            "mean_sign_agreement": float(np.mean(agreement)),
            "feature_names": x_names,
            "fit_signed_scores": fit_scores["signed_score"].astype(float).tolist(),
            "heldout_signed_scores": heldout_scores["signed_score"].astype(float).tolist(),
        }
    return {
        "n_selected_features": len(registry),
        "n_w_encoded": int(w_fit.shape[1]),
        "n_x_encoded": int(x_fit.shape[1]),
        "treatment": treatment_metrics,
        "outcome": outcome_metrics,
        "benchmark": benchmark["metrics"],
        "effect_coverage": coverage,
        "effect_topic_reconstruction": reconstruction,
        "structured_contrast": contrast,
        "extraction_summary": extraction_summary,
    }


def structured_review_gate(
    diagnostic: Dict[str, Any], nn_config: MultiModelForestConfig
) -> Dict[str, Any]:
    criteria: List[Dict[str, Any]] = []
    for target in ("treatment", "outcome"):
        observed = diagnostic[target]
        benchmark = diagnostic["benchmark"][target]["stacked_metrics"]
        for metric, direction in (
            ("auroc", "higher"),
            ("brier", "lower"),
            ("log_loss", "lower"),
            ("rmse", "lower"),
        ):
            base = benchmark.get(metric)
            value = observed.get(metric)
            if base is None or value is None:
                continue
            if direction == "higher":
                passed = float(value) >= float(base) - float(
                    nn_config.extracted_feature_review_auc_margin
                )
            else:
                passed = float(value) <= float(base) * (
                    1.0 + float(nn_config.extracted_feature_review_loss_relative_margin)
                )
            criteria.append(
                {
                    "family": "nuisance",
                    "target": target,
                    "metric": metric,
                    "observed": value,
                    "benchmark": base,
                    "passed": bool(passed),
                }
            )
    coverage = diagnostic["effect_coverage"]
    if coverage["raw_effect_evidence_weak"]:
        criteria.append(
            {
                "family": "effect",
                "metric": "contrast_mass_coverage",
                "passed": True,
                "state": "raw_effect_evidence_weak_or_unstable",
            }
        )
    else:
        criteria.append(
            {
                "family": "effect",
                "metric": "contrast_mass_coverage",
                "observed": coverage["coverage_fraction"],
                "target": nn_config.tfidf_topic.initial_effect_coverage_target,
                "passed": bool(
                    coverage["coverage_fraction"]
                    >= nn_config.tfidf_topic.initial_effect_coverage_target
                ),
            }
        )
        reconstruction = diagnostic.get("effect_topic_reconstruction", {})
        mean_correlation = reconstruction.get("mean_correlation")
        criteria.append(
            {
                "family": "effect",
                "metric": "heldout_topic_reconstruction_mean_correlation",
                "observed": mean_correlation,
                "target": 0.0,
                "passed": bool(
                    mean_correlation is not None
                    and np.isfinite(float(mean_correlation))
                    and float(mean_correlation) >= 0.0
                ),
                "training_fitted_then_heldout_scored": True,
            }
        )
        contrast = diagnostic.get("structured_contrast", {})
        sign_agreement = contrast.get("mean_sign_agreement")
        contrast_target = float(nn_config.tfidf_topic.minimum_tail_sign_agreement)
        criteria.append(
            {
                "family": "effect",
                "metric": "structured_cohort_contrast_sign_agreement",
                "observed": sign_agreement,
                "target": contrast_target,
                "passed": bool(
                    sign_agreement is not None
                    and np.isfinite(float(sign_agreement))
                    and float(sign_agreement) >= contrast_target
                ),
                "patient_level_effect_learner_used": False,
            }
        )
        if coverage.get("orphan_ngram_branch_status") == "completed":
            criteria.append(
                {
                    "family": "effect",
                    "metric": "selected_orphan_ngram_cluster_coverage",
                    "observed": coverage["orphan_cluster_coverage_fraction"],
                    "target": nn_config.tfidf_topic.initial_effect_coverage_target,
                    "passed": bool(
                        coverage["orphan_cluster_coverage_fraction"]
                        >= nn_config.tfidf_topic.initial_effect_coverage_target
                    ),
                    "raw_top20_preservation_reported_but_not_used_as_gate": True,
                }
            )
        else:
            criteria.append(
                {
                    "family": "effect",
                    "metric": "highest_ranked_raw_ngram_preservation",
                    "observed": coverage["highest_ranked_raw_ngram_preservation"],
                    "target": nn_config.tfidf_topic.initial_effect_coverage_target,
                    "passed": bool(
                        coverage["highest_ranked_raw_ngram_preservation"]
                        >= nn_config.tfidf_topic.initial_effect_coverage_target
                    ),
                }
            )
    failed = [criterion for criterion in criteria if not criterion["passed"]]
    return {
        "passed": not failed,
        "n_failed_criteria": len(failed),
        "criteria": criteria,
        "failed_families": sorted({row["family"] for row in failed}),
    }


def parsimony_replacement_passes(
    *,
    base: Dict[str, Any],
    trial: Dict[str, Any],
    base_dimension: int,
    trial_dimension: int,
    source_topic_coverage_loss: float,
    topic_reconstruction_loss: float,
    required_features_preserved: bool,
    role_union_preserved: bool,
    auc_tolerance: float = 0.02,
    loss_relative_tolerance: float = 0.05,
    contrast_coverage_tolerance: float = 0.05,
    reconstruction_tolerance: float = 0.03,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if trial_dimension >= base_dimension:
        reasons.append("encoded_dimension_not_reduced")
    if not required_features_preserved:
        reasons.append("required_feature_removed")
    if not role_union_preserved:
        reasons.append("role_union_changed")
    for target in ("treatment", "outcome"):
        for metric in ("auroc",):
            before = (base.get(target) or {}).get(metric)
            after = (trial.get(target) or {}).get(metric)
            if before is not None and (after is None or after < before - auc_tolerance):
                reasons.append(f"{target}_{metric}_outside_tolerance")
        for metric in ("brier", "log_loss", "rmse"):
            before = (base.get(target) or {}).get(metric)
            after = (trial.get(target) or {}).get(metric)
            if before is not None and (
                after is None or after > before * (1.0 + loss_relative_tolerance)
            ):
                reasons.append(f"{target}_{metric}_outside_tolerance")
    if source_topic_coverage_loss > contrast_coverage_tolerance:
        reasons.append("source_topic_contrast_coverage_loss")
    if topic_reconstruction_loss > reconstruction_tolerance:
        reasons.append("heldout_topic_reconstruction_loss")
    base_contrast = (base.get("structured_contrast") or {}).get("mean_sign_agreement")
    trial_contrast = (trial.get("structured_contrast") or {}).get("mean_sign_agreement")
    if base_contrast is not None and (trial_contrast is None or trial_contrast < base_contrast):
        reasons.append("structured_cohort_contrast_not_preserved")
    return not reasons, reasons


class TfidfTopicAgenticForestRunner:
    def __init__(
        self,
        *,
        dataset: pd.DataFrame,
        config: AppliedInferenceConfig,
        output_path: Path,
        handoff_path: Path,
        proposal_agent: Optional[Any] = None,
        extraction_provider: Optional[Any] = None,
        evaluator: Optional[Any] = None,
        resume: bool = True,
    ) -> None:
        self.dataset = dataset.reset_index(drop=True).copy()
        self.dataset["_oci_row_id"] = np.arange(len(self.dataset), dtype=int)
        self.config = config
        self.output_path = Path(output_path)
        self.artifact_dir = self.output_path.parent / "tfidf_topic_agentic_forest"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.handoff_path = Path(handoff_path)
        self.rows = _read_jsonl(self.handoff_path)
        self.nn_config: MultiModelForestConfig = config.architecture.multi_model_forest
        self.search_config = config.architecture.agentic_feature_search
        self.proposal_agent = proposal_agent or make_feature_search_agent(self.search_config)
        self.extraction_provider = extraction_provider or make_explicit_feature_extraction_provider(
            config=config, output_dir=self.artifact_dir
        )
        if evaluator is not None:
            self.evaluator = evaluator
        else:
            evaluator_class = (
                StructuredInteractionExplicitEvaluator
                if self.nn_config.structured_effect_estimator == "interaction_s_learner"
                else CausalForestExplicitEvaluator
            )
            self.evaluator = evaluator_class(
                config=config,
                cf_config=getattr(
                    config.architecture,
                    "explicit_feature_forest",
                    ExplicitFeatureForestConfig(),
                ),
            )
        self.resume = bool(resume)
        prespecified = [
            *list(self.nn_config.prespecified_features),
            *list(self.nn_config.prespecified_confounders),
            *list(self.nn_config.prespecified_effect_modifiers),
        ]
        if self.nn_config.prespecified_features_json:
            prespecified.extend(
                load_explicit_feature_specs_json(self.nn_config.prespecified_features_json)
            )
        by_name: Dict[str, ExplicitFeatureSpec] = {}
        for spec in prespecified:
            if spec.name not in by_name:
                by_name[spec.name] = spec
                continue
            current = by_name[spec.name]
            current.roles = list(dict.fromkeys([*current.roles, *spec.roles]))
        self.prespecified_specs = list(by_name.values())

    @staticmethod
    def _without_oracle_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """Return the exact modeling frame with every synthetic oracle removed."""
        oracle_columns = [column for column in frame.columns if str(column).startswith("true_")]
        return frame.drop(columns=oracle_columns, errors="ignore")

    def _prespecified_candidates(self) -> List[Dict[str, Any]]:
        return [
            {
                "action": "extract",
                "name": spec.name,
                "type": spec.type,
                "categories": spec.categories,
                "roles": spec.roles,
                "description": spec.description or spec.name.replace("_", " "),
                "required_or_prespecified": True,
                "provenance": [
                    {
                        "bank": "prespecified",
                        "topic_id": "prespecified",
                        "objective": "required",
                        "supporting_terms": [],
                    }
                ],
            }
            for spec in self.prespecified_specs
        ]

    def _rows_for_outer(self, outer_fold: int):
        full = [
            row
            for row in self.rows
            if int(row["outer_fold"]) == outer_fold and row["scope"] == "full_outer_train"
        ]
        inner = sorted(
            [
                row
                for row in self.rows
                if int(row["outer_fold"]) == outer_fold
                and row["scope"] == "candidate_selection_inner_fit"
            ],
            key=lambda row: int(row["inner_fold"]),
        )
        required = int(self.nn_config.candidate_consistency_inner_folds)
        if len(full) != 1 or len(inner) != required:
            raise RuntimeError(
                f"Stage 2 fails closed for outer_fold={outer_fold}: exact contexts "
                f"full={len(full)}/1 inner={len(inner)}/{required}. No evidence fallback is allowed."
            )
        return full[0], inner

    def _load_topic_score_tests(self, row: Dict[str, Any]) -> Dict[str, Any]:
        artifact = row["discovery"].get("artifacts", {}).get("topic_score_tests")
        if not artifact:
            raise RuntimeError(
                "Score-test filtering is enabled but this exact inner context has "
                "no topic_score_tests artifact; rerun Stage 1."
            )
        path = Path(str(artifact))
        if not path.exists():
            raise RuntimeError(f"Missing exact-context topic score tests: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != TOPIC_SCORE_TEST_SCHEMA_VERSION:
            raise RuntimeError(
                "Exact-context topic/n-gram score tests use an incompatible "
                f"schema {payload.get('schema_version')!r}; expected "
                f"{TOPIC_SCORE_TEST_SCHEMA_VERSION!r}. Rerun Stage 1."
            )
        if payload.get("status") != "completed":
            raise RuntimeError(f"Exact-context topic score tests are incomplete: {path}")
        return payload

    def _topic_score_selection_snapshot(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(self.nn_config.tfidf_topic.score_test_enabled):
            return {"enabled": False, "banks": {}}
        payload = self._load_topic_score_tests(row)
        snapshot: Dict[str, Any] = {
            "enabled": True,
            "schema_version": payload.get("schema_version"),
            "scope_id": payload.get("scope_id"),
            "banks": {},
        }
        for bank in ("treatment", "outcome", "effect"):
            bank_payload = (payload.get("banks") or {}).get(bank) or {}
            selected_ids = set(map(str, bank_payload.get("selected_topic_ids") or []))
            selected_topics = []
            for test in bank_payload.get("topic_tests", []):
                if str(test.get("topic_id")) not in selected_ids:
                    continue
                selected_topics.append(
                    {
                        key: test.get(key)
                        for key in (
                            "topic_id",
                            "evidence_rank",
                            "selection_reason",
                            "primary_p",
                            "primary_p_source",
                            "fdr_q",
                            "familywise_p",
                            "topic_score_testable",
                            "topic_score_moment",
                            "topic_standardized_score",
                            "topic_unadjusted_two_sided_p",
                            "topic_familywise_p",
                            "term_group_primary_p",
                            "term_group_primary_p_source",
                            "term_group_fdr_q",
                            "term_group_familywise_p",
                            "candidate_familywise_energy_p",
                            "quadratic_statistic",
                            "quadratic_covariance_rank",
                            "quadratic_statistic_per_rank",
                            "maximum_absolute_standardized_score",
                            "term_scores",
                            "selected_ngram_count",
                            "selected_ngram_terms",
                        )
                    }
                )
            snapshot["banks"][bank] = {
                "selected_topic_ids": sorted(selected_ids),
                "selected_topics": selected_topics,
                "selection_count": int(bank_payload.get("selection_count") or 0),
                "selection_rule": bank_payload.get("selection_rule"),
                "bootstrap_calibration": bank_payload.get("bootstrap_calibration") or {},
                "selected_ngrams": list(bank_payload.get("selected_ngrams") or []),
                "selected_ngram_terms": list(bank_payload.get("selected_ngram_terms") or []),
                "ngram_selection_count": int(bank_payload.get("ngram_selection_count") or 0),
                "ngram_selection_rule": bank_payload.get("ngram_selection_rule"),
            }
        orphan = payload.get("effect_orphan_ngram_branch") or {}
        snapshot["effect_orphan_ngram_branch"] = {
            "status": orphan.get("status"),
            "candidate_definition": orphan.get("candidate_definition"),
            "selected_cluster_ids": list(orphan.get("selected_cluster_ids") or []),
            "selected_clusters": list(orphan.get("selected_clusters") or []),
            "selection_count": int(orphan.get("selection_count") or 0),
            "selection_rule": orphan.get("selection_rule"),
            "bootstrap_calibration": orphan.get("bootstrap_calibration") or {},
            "cluster_count": int(orphan.get("cluster_count") or 0),
        }
        return snapshot

    @staticmethod
    def _topic_document(topic: Dict[str, Any]) -> str:
        return " ; ".join(
            str(term.get("term") or "")
            for term in topic.get("terms", [])
            if str(term.get("term") or "").strip()
        )

    def _full_orphan_jobs_from_policy(
        self,
        row: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
        """Map fixed inner orphan signatures onto full-training raw evidence."""
        orphan_policy = (policy.get("topic_score_selection") or {}).get(
            "effect_orphan_ngram_branch"
        ) or {}
        signatures = [
            dict(signature)
            for signature in orphan_policy.get("signatures") or []
            if signature.get("terms")
        ]
        audit: Dict[str, Any] = {
            "source": "fixed_inner_orphan_ngram_policy_mapping",
            "uses_outer_heldout_labels": False,
            "input_signature_count": len(signatures),
            "jobs": [],
        }
        if not bool(self.nn_config.tfidf_topic.orphan_ngram_enabled):
            audit["status"] = "disabled"
            return [], audit
        if not signatures:
            audit["status"] = "no_selected_inner_orphan_clusters"
            return [], audit

        documents = [" ; ".join(map(str, signature["terms"])) for signature in signatures]
        signature_vectors = _parsimony_tfidf_semantic_vectors(documents)
        signature_similarity = signature_vectors @ signature_vectors.T
        ungrouped = set(range(len(signatures)))
        signature_groups: List[List[int]] = []
        threshold = float(self.nn_config.tfidf_topic.orphan_ngram_cluster_similarity_threshold)
        for seed in range(len(signatures)):
            if seed not in ungrouped:
                continue
            members = [
                index
                for index in sorted(ungrouped)
                if index == seed
                or float(signature_similarity[seed, index]) >= threshold
                or bool(
                    set(map(str, signatures[seed]["terms"]))
                    & set(map(str, signatures[index]["terms"]))
                )
            ]
            for index in members:
                ungrouped.discard(index)
            signature_groups.append(members)
        signature_groups.sort(
            key=lambda members: (
                -len({int(signatures[index]["inner_fold"]) for index in members}),
                min(float(signatures[index].get("primary_p") or 1.0) for index in members),
                min(str(signatures[index].get("cluster_id")) for index in members),
            )
        )

        score_path = Path(str(row["discovery"]["artifacts"]["ngram_scores"]["effect"]))
        raw = pd.read_parquet(score_path).copy()
        raw["feature"] = raw["feature"].astype(str)
        raw["fit_rank"] = np.arange(1, len(raw) + 1, dtype=int)
        if "eligible" in raw.columns:
            raw = raw.loc[raw["eligible"].astype(bool)].copy()
        raw = raw.loc[
            raw["signed_score"].astype(float).abs()
            >= float(self.nn_config.tfidf_topic.orphan_ngram_min_abs_fit_score)
        ].copy()
        represented = {
            str(term.get("term"))
            for topic in (row["discovery"]["topic_banks"].get("effect", {}).get("topics", []))
            for term in topic.get("terms", [])
        }
        raw = raw.loc[~raw["feature"].isin(represented)].copy()
        raw = raw.sort_values(
            ["fit_rank", "unsigned_score", "feature"],
            ascending=[True, False, True],
            kind="stable",
        )
        candidate_records = raw.to_dict(orient="records")
        candidate_terms = [str(record["feature"]) for record in candidate_records]
        signature_terms = sorted(
            {
                str(term)
                for signature in signatures
                for term in signature.get("terms") or []
                if str(term).strip()
            }
        )
        if not candidate_terms or not signature_terms:
            audit["status"] = "no_full_context_fit_side_candidates"
            return [], audit
        semantic_vectors = _parsimony_tfidf_semantic_vectors([*signature_terms, *candidate_terms])
        signature_term_vectors = semantic_vectors[: len(signature_terms)]
        candidate_vectors = semantic_vectors[len(signature_terms) :]
        term_similarity = candidate_vectors @ signature_term_vectors.T
        signature_term_index = {term: index for index, term in enumerate(signature_terms)}
        required_folds = min(
            int(self.nn_config.tfidf_topic.orphan_ngram_full_min_inner_folds),
            int(policy.get("inner_fold_count") or 1),
        )
        maximum_terms = int(self.nn_config.tfidf_topic.orphan_ngram_cluster_max_terms)
        assigned_terms: set[str] = set()
        jobs: List[Tuple[str, Dict[str, Any]]] = []
        for group_index, members in enumerate(signature_groups, start=1):
            inner_folds = sorted({int(signatures[index]["inner_fold"]) for index in members})
            if len(inner_folds) < required_folds:
                continue
            group_terms = sorted(
                {str(term) for index in members for term in signatures[index].get("terms") or []}
            )
            group_columns = [signature_term_index[term] for term in group_terms]
            ranked_candidates: List[Tuple[int, float, int, str, Dict[str, Any]]] = []
            for candidate_index, record in enumerate(candidate_records):
                term = str(record["feature"])
                if term in assigned_terms:
                    continue
                exact = term in group_terms
                similarity = float(np.max(term_similarity[candidate_index, group_columns]))
                if not exact and similarity < threshold:
                    continue
                ranked_candidates.append(
                    (
                        0 if exact else 1,
                        -similarity,
                        int(record["fit_rank"]),
                        term,
                        record,
                    )
                )
            ranked_candidates.sort(key=lambda item: item[:4])
            chosen = ranked_candidates[:maximum_terms]
            if not chosen:
                continue
            term_rows: List[Dict[str, Any]] = []
            for exact_rank, negative_similarity, _rank, term, record in chosen:
                assigned_terms.add(term)
                term_rows.append(
                    {
                        "term": term,
                        "loading": 0.0,
                        "screen_rank": int(record["fit_rank"]),
                        "signed_score": float(record["signed_score"]),
                        "fit_rank": int(record["fit_rank"]),
                        "fit_signed_score": float(record["signed_score"]),
                        "fit_unsigned_score": float(abs(float(record["signed_score"]))),
                        "combined_importance": float(record.get("combined_importance", 0.0)),
                        "mapped_exactly": exact_rank == 0,
                        "mapping_similarity": float(-negative_similarity),
                    }
                )
            cluster_id = f"effect_orphan_cluster_full_{len(jobs) + 1:03d}"
            mapped_evidence = {
                "source": "fixed_inner_orphan_ngram_policy_mapping",
                "uses_outer_heldout_labels": False,
                "cluster_id": cluster_id,
                "matched_inner_folds": inner_folds,
                "matched_inner_cluster_ids": [
                    str(signatures[index].get("cluster_id")) for index in members
                ],
                "inner_signatures": [signatures[index] for index in members],
                "full_context_terms": [row["term"] for row in term_rows],
            }
            job = (
                "effect",
                {
                    "topic_id": cluster_id,
                    "bank": "effect",
                    "evidence_kind": "orphan_raw_ngram_cluster",
                    "terms": term_rows,
                    "_prompt_version": ORPHAN_NGRAM_LABEL_PROMPT_VERSION,
                    "_selection_evidence": mapped_evidence,
                },
            )
            jobs.append(job)
            audit["jobs"].append(mapped_evidence)
        audit.update(
            {
                "status": "completed",
                "signature_group_count": len(signature_groups),
                "required_inner_fold_recurrence": required_folds,
                "full_fit_candidate_count": len(candidate_records),
                "mapped_job_count": len(jobs),
                "mapped_term_count": len(assigned_terms),
            }
        )
        return jobs, audit

    def _full_topic_jobs_from_policy(
        self,
        row: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
        score_policy = policy.get("topic_score_selection") or {}
        policy_banks = score_policy.get("banks") or {}
        jobs: List[Tuple[str, Dict[str, Any]]] = []
        audit: Dict[str, Any] = {
            "source": "fixed_inner_score_test_policy",
            "outer_fold": int(row["outer_fold"]),
            "uses_outer_heldout_labels": False,
            "banks": {},
        }
        for bank in ("treatment", "outcome", "effect"):
            topics = list(row["discovery"]["topic_banks"].get(bank, {}).get("topics", []))
            signatures = list((policy_banks.get(bank) or {}).get("signatures") or [])
            if not topics:
                audit["banks"][bank] = {
                    "selected_topic_ids": [],
                    "reason": "no_full_context_topics",
                }
                continue
            if not signatures:
                audit["banks"][bank] = {
                    "selected_topic_ids": [],
                    "reason": "no_testable_or_selected_inner_topics",
                    "uses_outer_heldout_labels": False,
                }
                continue
            documents = [
                " ; ".join(map(str, signature.get("terms") or [])) for signature in signatures
            ] + [self._topic_document(topic) for topic in topics]
            vectors = _parsimony_tfidf_semantic_vectors(documents)
            signature_vectors = vectors[: len(signatures)]
            topic_vectors = vectors[len(signatures) :]
            similarities = topic_vectors @ signature_vectors.T
            term_fold_recurrence = {
                str(term): int(count)
                for term, count in (
                    (policy_banks.get(bank) or {}).get(
                        "selected_ngram_fold_recurrence",
                        (policy_banks.get(bank) or {}).get("selected_term_fold_recurrence", {}),
                    )
                ).items()
            }
            scored: List[Dict[str, Any]] = []
            for topic_index, topic in enumerate(topics):
                topic_terms = {
                    str(term.get("term"))
                    for term in topic.get("terms", [])
                    if str(term.get("term") or "").strip()
                }
                exact_folds = {
                    int(signature["inner_fold"])
                    for signature in signatures
                    if topic_terms & set(map(str, signature.get("terms") or []))
                }
                semantic_folds = {
                    int(signatures[index]["inner_fold"])
                    for index, value in enumerate(similarities[topic_index])
                    if float(value) >= 0.35
                }
                matched_folds = exact_folds | semantic_folds
                matched_evidence: List[Dict[str, Any]] = []
                for signature_index, signature in enumerate(signatures):
                    signature_terms = set(map(str, signature.get("terms") or []))
                    exact_overlap = sorted(topic_terms & signature_terms)
                    similarity = float(similarities[topic_index, signature_index])
                    if not exact_overlap and similarity < 0.35:
                        continue
                    strongest_terms = sorted(
                        list(signature.get("term_scores") or []),
                        key=lambda term: -abs(float(term.get("heldout_standardized_score") or 0.0)),
                    )[:5]
                    matched_evidence.append(
                        {
                            "inner_fold": int(signature["inner_fold"]),
                            "inner_topic_id": str(signature.get("topic_id")),
                            "exact_term_overlap": exact_overlap,
                            "semantic_similarity": similarity,
                            "primary_p": signature.get("primary_p"),
                            "primary_p_source": signature.get("primary_p_source"),
                            "fdr_q": signature.get("fdr_q"),
                            "familywise_p": signature.get("familywise_p"),
                            "topic_standardized_score": signature.get("topic_standardized_score"),
                            "topic_score_moment": signature.get("topic_score_moment"),
                            "term_group_primary_p": signature.get("term_group_primary_p"),
                            "term_group_fdr_q": signature.get("term_group_fdr_q"),
                            "evidence_rank": signature.get("evidence_rank"),
                            "quadratic_statistic_per_rank": signature.get(
                                "quadratic_statistic_per_rank"
                            ),
                            "strongest_term_scores": strongest_terms,
                        }
                    )
                matched_evidence.sort(
                    key=lambda evidence: (
                        -len(evidence["exact_term_overlap"]),
                        -float(evidence["semantic_similarity"]),
                        float(evidence["primary_p"] if evidence["primary_p"] is not None else 1.0),
                        int(evidence["inner_fold"]),
                    )
                )
                exact_weight = int(sum(term_fold_recurrence.get(term, 0) for term in topic_terms))
                best_similarity = float(np.max(similarities[topic_index]))
                scored.append(
                    {
                        "topic_index": topic_index,
                        "topic_id": str(topic["topic_id"]),
                        "matched_inner_folds": sorted(matched_folds),
                        "matched_inner_fold_count": len(matched_folds),
                        "exact_selected_term_weight": exact_weight,
                        "exact_selected_ngram_weight": exact_weight,
                        "best_signature_similarity": best_similarity,
                        "matched_inner_evidence": matched_evidence[:6],
                        "mapping_score": float(
                            len(matched_folds)
                            + exact_weight / max(1, 15 * int(policy.get("inner_fold_count") or 1))
                            + best_similarity
                        ),
                    }
                )
            scored.sort(
                key=lambda item: (
                    -int(item["matched_inner_fold_count"]),
                    -int(item["exact_selected_term_weight"]),
                    -float(item["best_signature_similarity"]),
                    str(item["topic_id"]),
                )
            )
            maximum = min(
                int(self.nn_config.tfidf_topic.score_test_max_topics_per_bank),
                len(scored),
            )
            minimum = min(
                int(self.nn_config.tfidf_topic.score_test_min_topics_per_bank),
                maximum,
            )
            required_folds = min(
                int(self.nn_config.tfidf_topic.score_test_full_topic_min_inner_folds),
                int(policy.get("inner_fold_count") or 1),
            )
            selected = [
                item for item in scored if int(item["matched_inner_fold_count"]) >= required_folds
            ][:maximum]
            selected_ids = {item["topic_id"] for item in selected}
            for item in scored:
                if len(selected) >= minimum:
                    break
                if item["topic_id"] not in selected_ids:
                    selected.append(item)
                    selected_ids.add(item["topic_id"])
            selected_by_id = {item["topic_id"]: item for item in selected}
            for topic in topics:
                topic_id = str(topic["topic_id"])
                if topic_id not in selected_by_id:
                    continue
                evidence = {
                    "source": "fixed_inner_score_test_policy_mapping",
                    "uses_outer_heldout_labels": False,
                    **selected_by_id[topic_id],
                }
                jobs.append((bank, {**topic, "_selection_evidence": evidence}))
            audit["banks"][bank] = {
                "selected_topic_ids": [str(item["topic_id"]) for item in selected],
                "required_inner_fold_recurrence": required_folds,
                "minimum_topics": minimum,
                "maximum_topics": maximum,
                "topic_mapping": scored,
            }
        orphan_jobs, orphan_audit = self._full_orphan_jobs_from_policy(row, policy)
        jobs.extend(orphan_jobs)
        audit["effect_orphan_ngram_branch"] = orphan_audit
        return jobs, audit

    def _topic_jobs(
        self,
        row: Dict[str, Any],
        selection_policy: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        if not bool(self.nn_config.tfidf_topic.score_test_enabled):
            return [
                (bank, topic)
                for bank in ("treatment", "outcome", "effect")
                for topic in row["discovery"]["topic_banks"].get(bank, {}).get("topics", [])
            ]
        if row.get("scope") == "full_outer_train":
            if selection_policy is None:
                raise RuntimeError(
                    "Full-outer topic labeling requires the fixed inner "
                    "score-test policy; labeling all topics is not allowed."
                )
            jobs, _audit = self._full_topic_jobs_from_policy(row, selection_policy)
            return jobs

        topic_jobs = [
            (bank, topic)
            for bank, topic in self._inner_ranked_topic_jobs(row)
            if bool((topic.get("_selection_evidence") or {}).get("selected_for_agent", False))
        ]
        return [*topic_jobs, *self._inner_orphan_jobs(row)]

    def _inner_orphan_jobs(self, row: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Return held-out-selected groups from the predeclared fit-side universe."""
        if row.get("scope") == "full_outer_train":
            raise ValueError("Inner orphan score evidence is unavailable for full contexts")
        if not bool(self.nn_config.tfidf_topic.orphan_ngram_enabled):
            return []
        score_tests = self._load_topic_score_tests(row)
        branch = score_tests.get("effect_orphan_ngram_branch") or {}
        if branch.get("status") != "completed":
            raise RuntimeError(
                "The orphan n-gram branch is enabled but its exact inner "
                "score-test artifact is incomplete; rerun Stage 1."
            )
        jobs: List[Tuple[str, Dict[str, Any]]] = []
        for cluster in sorted(
            branch.get("selected_clusters") or [],
            key=lambda item: (
                int(item.get("evidence_rank") or 10**9),
                str(item.get("cluster_id")),
            ),
        ):
            term_rows = []
            for term in cluster.get("term_scores") or []:
                term_rows.append(
                    {
                        **dict(term),
                        "loading": 0.0,
                        "screen_rank": int(term.get("fit_rank") or 0),
                        "signed_score": float(term.get("fit_signed_score") or 0.0),
                    }
                )
            if not 1 <= len(term_rows) <= 15:
                raise RuntimeError(
                    f"Selected orphan cluster {cluster.get('cluster_id')} has "
                    f"{len(term_rows)} terms; expected 1-15"
                )
            nested_policy = (
                str(self.nn_config.tfidf_topic.score_selection_label_policy)
                == "nested_fit_calibration"
            )
            evidence = {
                "source": (
                    "nested_fit_calibration_orphan_ngram_group_score_test"
                    if nested_policy
                    else "exact_inner_heldout_orphan_ngram_group_score_test"
                ),
                "uses_heldout_treatment_and_outcome": not nested_policy,
                "uses_registered_heldout_treatment_and_outcome": False,
                "uses_nested_fit_calibration_treatment_and_outcome": nested_policy,
                **dict(cluster),
            }
            jobs.append(
                (
                    "effect",
                    {
                        "topic_id": str(cluster["cluster_id"]),
                        "bank": "effect",
                        "evidence_kind": "orphan_raw_ngram_cluster",
                        "terms": term_rows,
                        "_prompt_version": ORPHAN_NGRAM_LABEL_PROMPT_VERSION,
                        "_selection_evidence": evidence,
                    },
                )
            )
        return jobs

    def _inner_ranked_topic_jobs(self, row: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Return every inner topic in held-out evidence order.

        The initial labeling pass consumes only ``selected_for_agent`` topics.
        Additive review can ask for the next relevant topic when structured
        diagnostics fail, without recomputing or peeking at any new rows.
        """
        if row.get("scope") == "full_outer_train":
            raise ValueError("Ranked held-out topic evidence exists only for inner contexts")
        if not bool(self.nn_config.tfidf_topic.score_test_enabled):
            return [
                (bank, topic)
                for bank in ("treatment", "outcome", "effect")
                for topic in row["discovery"]["topic_banks"].get(bank, {}).get("topics", [])
            ]
        score_tests = self._load_topic_score_tests(row)
        jobs: List[Tuple[str, Dict[str, Any]]] = []
        for bank in ("treatment", "outcome", "effect"):
            bank_tests = (score_tests.get("banks") or {}).get(bank) or {}
            topics_by_id = {
                str(topic["topic_id"]): topic
                for topic in row["discovery"]["topic_banks"].get(bank, {}).get("topics", [])
            }
            ranked_tests = sorted(
                bank_tests.get("topic_tests", []),
                key=lambda test: (
                    int(test.get("evidence_rank") or 10**9),
                    str(test.get("topic_id")),
                ),
            )
            for test in ranked_tests:
                topic_id = str(test["topic_id"])
                topic = topics_by_id.get(topic_id)
                if topic is None:
                    continue
                nested_policy = (
                    str(self.nn_config.tfidf_topic.score_selection_label_policy)
                    == "nested_fit_calibration"
                )
                evidence = {
                    "source": (
                        "nested_fit_calibration_topic_and_ngram_score_test"
                        if nested_policy
                        else "exact_inner_heldout_topic_and_ngram_score_test"
                    ),
                    "uses_heldout_treatment_and_outcome": not nested_policy,
                    "uses_registered_heldout_treatment_and_outcome": False,
                    "uses_nested_fit_calibration_treatment_and_outcome": nested_policy,
                    **test,
                }
                jobs.append((bank, {**topic, "_selection_evidence": evidence}))
        return jobs

    def _label_context_topics(
        self,
        row: Dict[str, Any],
        context_dir: Path,
        selection_policy: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        checkpoint = context_dir / "topic_labels.jsonl"
        model_identity = self._harmonization_agent_model_identity()
        request_settings_hash = stable_hash(
            {
                "agent_provider": getattr(self.search_config, "agent_provider", "openai"),
                "agent_server_url": getattr(self.search_config, "agent_server_url", None),
                "configured_model_name": getattr(self.search_config, "agent_model_name", None),
                "resolved_model_identity": model_identity,
                "agent_temperature": self.search_config.agent_temperature,
                "agent_max_tokens": self.search_config.agent_max_tokens,
                "agent_enable_thinking": getattr(self.search_config, "agent_enable_thinking", None),
                "agent_thinking_token_budget": getattr(
                    self.search_config, "agent_thinking_token_budget", None
                ),
                "agent_schema_repair_attempts": getattr(
                    self.search_config, "agent_schema_repair_attempts", 1
                ),
            }
        )
        completed: Dict[str, Dict[str, Any]] = {}
        if self.resume and checkpoint.exists():
            for item in _read_unversioned_jsonl(checkpoint):
                completed[str(item["topic_id"])] = item
        jobs = self._topic_jobs(row, selection_policy=selection_policy)
        selection_audit: Dict[str, Any]
        if row.get("scope") == "full_outer_train" and selection_policy is not None:
            _jobs, selection_audit = self._full_topic_jobs_from_policy(row, selection_policy)
        else:
            selection_audit = {
                "source": (
                    (
                        "nested_fit_calibration_topic_and_ngram_score_test"
                        if str(self.nn_config.tfidf_topic.score_selection_label_policy)
                        == "nested_fit_calibration"
                        else "exact_inner_heldout_topic_and_ngram_score_test"
                    )
                    if self.nn_config.tfidf_topic.score_test_enabled
                    else "score_test_filter_disabled"
                ),
                "selected_topic_ids_by_bank": {
                    bank: [str(topic["topic_id"]) for job_bank, topic in jobs if job_bank == bank]
                    for bank in ("treatment", "outcome", "effect")
                },
            }
        _write_json(context_dir / "topic_filter_selection.json", selection_audit)

        def run_one(bank: str, topic: Dict[str, Any]) -> Dict[str, Any]:
            context = build_topic_label_context(
                outer_fold=int(row["outer_fold"]),
                scope=str(row["scope"]),
                inner_fold=row.get("inner_fold"),
                bank=bank,
                topic=topic,
                prompt_version=str(topic.get("_prompt_version") or TOPIC_LABEL_PROMPT_VERSION),
                score_test_evidence=topic.get("_selection_evidence"),
            )
            try:
                response = self.proposal_agent.propose(context)
                trace = _get_agent_response_trace(self.proposal_agent)
                candidates = [
                    candidate
                    for proposal in _proposal_rows(response)
                    if (
                        candidate := _candidate_from_response(
                            proposal, context=context, topic=topic
                        )
                    )
                    is not None
                ]
                return {
                    "topic_id": topic["topic_id"],
                    "bank": bank,
                    "context_hash": stable_hash(context),
                    "model_identity": model_identity,
                    "request_settings_hash": request_settings_hash,
                    "response": response,
                    "response_trace": trace if self.search_config.save_agent_raw_output else None,
                    "candidates": candidates,
                    "status": "completed",
                }
            except Exception as exc:
                return {
                    "topic_id": topic["topic_id"],
                    "bank": bank,
                    "context_hash": stable_hash(context),
                    "model_identity": model_identity,
                    "request_settings_hash": request_settings_hash,
                    "candidates": [],
                    "status": "dropped",
                    "drop_reason": repr(exc),
                    "error": repr(exc),
                }

        pending = []
        for bank, topic in jobs:
            expected_context = build_topic_label_context(
                outer_fold=int(row["outer_fold"]),
                scope=str(row["scope"]),
                inner_fold=row.get("inner_fold"),
                bank=bank,
                topic=topic,
                prompt_version=str(topic.get("_prompt_version") or TOPIC_LABEL_PROMPT_VERSION),
                score_test_evidence=topic.get("_selection_evidence"),
            )
            existing = completed.get(str(topic["topic_id"]))
            if (
                existing is None
                or existing.get("context_hash") != stable_hash(expected_context)
                or existing.get("model_identity") != model_identity
                or existing.get("request_settings_hash") != request_settings_hash
            ):
                pending.append((bank, topic))
        if pending:
            # Resolve one shared client/model before concurrent topic requests so
            # lazy endpoint discovery cannot race across worker threads.
            ensure_client = getattr(self.proposal_agent, "_ensure_client", None)
            resolve_model = getattr(self.proposal_agent, "_resolve_agent_model_name", None)
            if callable(ensure_client):
                ensure_client()
            if callable(resolve_model):
                resolve_model()
            workers = min(int(self.nn_config.tfidf_topic.topic_label_parallelism), len(pending))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {executor.submit(run_one, bank, topic): topic for bank, topic in pending}
                for future in as_completed(futures):
                    result = future.result()
                    completed[str(result["topic_id"])] = result
                    _write_jsonl(checkpoint, completed.values())
        missing = [topic["topic_id"] for _bank, topic in jobs if topic["topic_id"] not in completed]
        if missing:
            raise RuntimeError(f"Topic label checkpoint is incomplete: {missing[:5]}")
        ordered = [completed[topic["topic_id"]] for _bank, topic in jobs]
        dropped = [item["topic_id"] for item in ordered if item.get("status") != "completed"]
        if dropped:
            logger.warning(
                "Topic labeling dropped %s unresolved topic(s) after the configured "
                "repair attempts; source evidence remains available to review/recovery: %s",
                len(dropped),
                dropped[:5],
            )
        return ordered

    def _harmonization_agent_model_identity(self) -> str:
        resolver = getattr(self.proposal_agent, "_resolve_agent_model_name", None)
        if callable(resolver):
            return str(resolver())
        return str(
            getattr(self.search_config, "agent_model_name", None)
            or getattr(self.search_config, "codex_cli_model_name", None)
            or self.proposal_agent.__class__.__name__
        )

    def _run_harmonization_requests(
        self,
        *,
        contexts: Sequence[Dict[str, Any]],
        checkpoint: Path,
    ) -> List[Dict[str, Any]]:
        """Run bounded, independently resumable fold-local harmonization calls."""
        model_identity = self._harmonization_agent_model_identity()
        request_settings_hash = stable_hash(
            {
                "agent_provider": getattr(self.search_config, "agent_provider", "openai"),
                "agent_temperature": self.search_config.agent_temperature,
                "agent_max_tokens": self.search_config.agent_max_tokens,
                "agent_enable_thinking": getattr(self.search_config, "agent_enable_thinking", None),
                "agent_thinking_token_budget": getattr(
                    self.search_config, "agent_thinking_token_budget", None
                ),
            }
        )
        completed: Dict[str, Dict[str, Any]] = {}
        if self.resume and checkpoint.exists():
            for record in _read_unversioned_jsonl(checkpoint):
                request_id = str(record.get("request_id") or "")
                if request_id:
                    completed[request_id] = record
        expected = {str(context["request_id"]): context for context in contexts}
        valid: Dict[str, Dict[str, Any]] = {}
        for request_id, record in completed.items():
            context = expected.get(request_id)
            if context is None:
                continue
            if (
                record.get("context_hash") == stable_hash(context)
                and record.get("model_identity") == model_identity
                and record.get("request_settings_hash") == request_settings_hash
            ):
                valid[request_id] = record
        completed = valid
        pending = [context for context in contexts if str(context["request_id"]) not in completed]

        ensure_client = getattr(self.proposal_agent, "_ensure_client", None)
        if pending and callable(ensure_client):
            ensure_client()

        def run_one(context: Dict[str, Any]) -> Dict[str, Any]:
            try:
                response = self.proposal_agent.propose(context)
                issues = topic_harmonization_response_issues(response, context)
                if issues:
                    raise ValueError("; ".join(issues))
                trace = _get_agent_response_trace(self.proposal_agent)
                return {
                    "request_id": context["request_id"],
                    "context_hash": stable_hash(context),
                    "model_identity": model_identity,
                    "request_settings_hash": request_settings_hash,
                    "status": "completed",
                    "response": response,
                    "response_trace": (trace if self.search_config.save_agent_raw_output else None),
                }
            except Exception as exc:
                return {
                    "request_id": context["request_id"],
                    "context_hash": stable_hash(context),
                    "model_identity": model_identity,
                    "request_settings_hash": request_settings_hash,
                    "status": "dropped",
                    "error": repr(exc),
                }

        if pending:
            workers = min(len(pending), int(self.nn_config.tfidf_topic.topic_label_parallelism))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {executor.submit(run_one, context): context for context in pending}
                for future in as_completed(futures):
                    record = future.result()
                    completed[str(record["request_id"])] = record
                    _write_jsonl_atomic(
                        checkpoint,
                        [completed[request_id] for request_id in sorted(completed)],
                    )
        return [completed[str(context["request_id"])] for context in contexts]

    @staticmethod
    def _identity_name_response(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "decisions": [
                {
                    "candidate_id": entry["candidate_id"],
                    "action": entry.get("action", "extract"),
                    "canonical_name": entry["name"],
                    "clinical_domain": entry.get("clinical_domain"),
                    "parent_object": entry.get("parent_object"),
                    "alias_of": None,
                    "source_names": (entry.get("derivation") or {}).get("source_names", []),
                    "derivation": entry.get("derivation"),
                    "reason": "no competing candidate in this harmonization block",
                }
                for entry in entries
            ]
        }

    def _harmonization_context_base(
        self, row: Dict[str, Any], *, prompt_version: str, stage: str
    ) -> Dict[str, Any]:
        return {
            "prompt_version": prompt_version,
            "harmonization_stage": stage,
            "outer_fold": int(row["outer_fold"]),
            "inner_fold": row.get("inner_fold"),
            "scope": row["scope"],
            "fit_row_fingerprint": row["fit_row_fingerprint"],
            "temporal_cutoff": ("use only information documented before the treatment decision"),
            "allowed_final_actions": ["extract", "derive", "alias/drop", "drop"],
            "roles_are_mechanically_assigned": True,
        }

    def _harmonize_context_candidates(
        self,
        *,
        row: Dict[str, Any],
        context_dir: Path,
        candidates: Sequence[Dict[str, Any]],
        include_prespecified: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build one fully fold-local canonical registry before any extraction."""
        harmonization_dir = context_dir / "harmonization"
        registry, dropped = harmonize_topic_candidates(
            [
                *(self._prespecified_candidates() if include_prespecified else []),
                *candidates,
            ]
        )

        # Pass 1: annotate domain/parent and adjudicate nearby sorted definitions.
        annotation_contexts: List[Dict[str, Any]] = []
        for batch_index, start in enumerate(
            range(0, len(registry), _HARMONIZATION_BATCH_SIZE), start=1
        ):
            batch = registry[start : start + _HARMONIZATION_BATCH_SIZE]
            annotation_contexts.append(
                {
                    **self._harmonization_context_base(
                        row,
                        prompt_version=TOPIC_NAME_HARMONIZATION_PROMPT_VERSION,
                        stage="domain_parent_annotation",
                    ),
                    "request_id": f"annotation_{batch_index:04d}",
                    "candidates": [_compact_harmonization_candidate(entry) for entry in batch],
                }
            )
        annotation_records = self._run_harmonization_requests(
            contexts=annotation_contexts,
            checkpoint=harmonization_dir / "name_annotation.jsonl",
        )
        annotation_responses = [
            record["response"]
            for record in annotation_records
            if record.get("status") == "completed"
        ]
        registry, pass_drops = apply_topic_name_harmonization(registry, annotation_responses)
        dropped.extend(pass_drops)
        dropped.extend(
            {
                "action": "drop",
                "reason": "unresolved_name_annotation_request",
                "request_id": record["request_id"],
                "error": record.get("error"),
            }
            for record in annotation_records
            if record.get("status") != "completed"
        )

        # Pass 2: candidates sharing the agent-assigned domain/parent are reviewed
        # together. Singletons receive a deterministic no-change decision.
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for entry in registry:
            grouped[(entry["clinical_domain"], entry["parent_object"])].append(entry)
        domain_contexts: List[Dict[str, Any]] = []
        identity_responses: List[Dict[str, Any]] = []
        request_number = 0
        for group_key in sorted(grouped):
            values = sorted(grouped[group_key], key=lambda entry: entry["name"])
            if len(values) == 1:
                identity_responses.append(self._identity_name_response(values))
                continue
            for start in range(0, len(values), _HARMONIZATION_BATCH_SIZE):
                request_number += 1
                batch = values[start : start + _HARMONIZATION_BATCH_SIZE]
                domain_contexts.append(
                    {
                        **self._harmonization_context_base(
                            row,
                            prompt_version=TOPIC_NAME_HARMONIZATION_PROMPT_VERSION,
                            stage="within_domain_parent_adjudication",
                        ),
                        "request_id": f"domain_{request_number:04d}",
                        "group": {
                            "clinical_domain": group_key[0],
                            "parent_object": group_key[1],
                        },
                        "candidates": [_compact_harmonization_candidate(entry) for entry in batch],
                    }
                )
        domain_records = self._run_harmonization_requests(
            contexts=domain_contexts,
            checkpoint=harmonization_dir / "domain_resolution.jsonl",
        )
        domain_responses = [
            *identity_responses,
            *[
                record["response"]
                for record in domain_records
                if record.get("status") == "completed"
            ],
        ]
        registry, pass_drops = apply_topic_name_harmonization(registry, domain_responses)
        dropped.extend(pass_drops)
        dropped.extend(
            {
                "action": "drop",
                "reason": "unresolved_domain_parent_adjudication",
                "request_id": record["request_id"],
                "error": record.get("error"),
            }
            for record in domain_records
            if record.get("status") != "completed"
        )

        # Pass 3 is global in candidate comparison, but every agent request is
        # bounded. Word/character TF-IDF blocks likely cross-group aliases; the
        # agent returns sparse changes only, and omitted entries remain distinct.
        global_contexts: List[Dict[str, Any]] = []
        for block_index, block in enumerate(build_topic_global_dedup_blocks(registry), start=1):
            global_contexts.append(
                {
                    **self._harmonization_context_base(
                        row,
                        prompt_version=TOPIC_GLOBAL_DEDUP_PROMPT_VERSION,
                        stage="final_global_deduplication",
                    ),
                    "request_id": f"global_{block_index:04d}",
                    "candidates": [
                        {
                            "name": entry["name"],
                            "type": entry["type"],
                            "description": str(entry.get("description") or "")[:500],
                            "clinical_domain": entry["clinical_domain"],
                            "parent_object": entry["parent_object"],
                            "required_or_prespecified": entry.get(
                                "required_or_prespecified", False
                            ),
                        }
                        for entry in block
                    ],
                }
            )
        global_checkpoint = harmonization_dir / "global_dedup.jsonl"
        global_records = self._run_harmonization_requests(
            contexts=global_contexts,
            checkpoint=global_checkpoint,
        )
        if not global_contexts:
            _write_jsonl_atomic(global_checkpoint, [])
        completed_global_responses = [
            record["response"] for record in global_records if record.get("status") == "completed"
        ]
        if completed_global_responses:
            registry, pass_drops = apply_topic_global_dedup(
                registry,
                combine_topic_global_dedup_responses(completed_global_responses),
            )
            dropped.extend(pass_drops)
        dropped.extend(
            {
                "action": "drop",
                "reason": "global_dedup_request_unresolved_no_changes_applied",
                "request_id": record["request_id"],
                "error": record.get("error"),
            }
            for record in global_records
            if record.get("status") != "completed"
        )

        # Fit structured value contracts using only this context's topic evidence.
        value_contexts: List[Dict[str, Any]] = []
        value_checkpoint = harmonization_dir / "value_contracts.jsonl"
        value_layout_path = harmonization_dir / "value_contract_layout.json"
        if value_layout_path.exists():
            value_layout = str(
                json.loads(value_layout_path.read_text(encoding="utf-8")).get(
                    "layout", "domain_packed_v2"
                )
            )
        elif value_checkpoint.exists():
            # Preserve request identities for a checkpoint created before the
            # packed layout was introduced; new contexts use the efficient path.
            value_layout = "domain_parent_legacy_v1"
            _write_json(value_layout_path, {"layout": value_layout})
        else:
            value_layout = "domain_packed_v2"
            _write_json(value_layout_path, {"layout": value_layout})
        value_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for entry in registry:
            key = (
                (entry["clinical_domain"], entry["parent_object"])
                if value_layout == "domain_parent_legacy_v1"
                else (entry["clinical_domain"], "__domain_packed__")
            )
            value_groups[key].append(entry)
        request_number = 0
        for group_key in sorted(value_groups):
            values = sorted(value_groups[group_key], key=lambda entry: entry["name"])
            for start in range(0, len(values), _VALUE_HARMONIZATION_BATCH_SIZE):
                request_number += 1
                batch = values[start : start + _VALUE_HARMONIZATION_BATCH_SIZE]
                value_contexts.append(
                    {
                        **self._harmonization_context_base(
                            row,
                            prompt_version=TOPIC_VALUE_HARMONIZATION_PROMPT_VERSION,
                            stage="initial_value_contract_fit",
                        ),
                        "request_id": f"value_{request_number:04d}",
                        "group": (
                            {
                                "clinical_domain": group_key[0],
                                "parent_object": group_key[1],
                            }
                            if value_layout == "domain_parent_legacy_v1"
                            else {
                                "clinical_domain": group_key[0],
                                "parent_object": None,
                                "parent_objects": sorted(
                                    {str(entry["parent_object"]) for entry in batch}
                                ),
                            }
                        ),
                        "candidates": [_compact_harmonization_candidate(entry) for entry in batch],
                    }
                )
        value_records = self._run_harmonization_requests(
            contexts=value_contexts,
            checkpoint=value_checkpoint,
        )
        value_responses = [
            record["response"] for record in value_records if record.get("status") == "completed"
        ]
        registry, value_drops = apply_topic_value_harmonization(registry, value_responses)
        dropped.extend(value_drops)
        dropped.extend(
            {
                "action": "drop",
                "reason": "unresolved_value_contract_request",
                "request_id": record["request_id"],
                "error": record.get("error"),
            }
            for record in value_records
            if record.get("status") != "completed"
        )
        if any(entry["action"] not in {"extract", "derive"} for entry in registry):
            raise RuntimeError("Canonical registry contains a non-executable final state")
        _write_json(
            harmonization_dir / "manifest.json",
            {
                "schema_version": CANONICAL_REGISTRY_SCHEMA_VERSION,
                "fit_row_fingerprint": row["fit_row_fingerprint"],
                "n_registry_features": len(registry),
                "n_dropped_or_aliased": len(dropped),
                "registry": registry,
                "dropped": dropped,
                "fold_local_state": True,
            },
        )
        return registry, dropped

    def _revise_value_contracts_from_training(
        self,
        *,
        row: Dict[str, Any],
        context_dir: Path,
        registry: Sequence[Dict[str, Any]],
        fit_df: pd.DataFrame,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        """Audit/revise contracts from extracted fit values before held-out extraction."""
        summaries: List[Dict[str, Any]] = []
        issue_names: set = set()
        for entry in registry:
            value_column = f"explicit_feat_{entry['name']}"
            missing_column = f"{value_column}_missing"
            if value_column not in fit_df.columns:
                summary = {
                    "name": entry["name"],
                    "issue": "missing_extracted_column",
                    "coverage": 0.0,
                    "sample_values": [],
                }
                issue_names.add(entry["name"])
                summaries.append(summary)
                continue
            missing = (
                fit_df[missing_column].astype(bool)
                if missing_column in fit_df.columns
                else fit_df[value_column].isna()
            )
            observed = fit_df.loc[~missing, value_column]
            sample_values = [
                value.item() if isinstance(value, np.generic) else value
                for value in observed.drop_duplicates().head(20).tolist()
            ]
            issues: List[str] = []
            if entry["type"] == "continuous":
                numeric = pd.to_numeric(observed, errors="coerce")
                if int(numeric.isna().sum()) > 0:
                    issues.append("non_numeric_values_under_continuous_contract")
            else:
                permitted = set(map(str, entry.get("categories") or []))
                unexpected = sorted({str(value) for value in observed.tolist()} - permitted)
                if unexpected:
                    issues.append("out_of_contract_categories")
                else:
                    unexpected = []
            if issues:
                issue_names.add(entry["name"])
            summaries.append(
                {
                    "name": entry["name"],
                    "coverage": float(1.0 - missing.mean()),
                    "n_unique_observed": int(observed.nunique(dropna=True)),
                    "sample_values": sample_values,
                    "issues": issues,
                    "unexpected_values": unexpected if entry["type"] == "categorical" else [],
                }
            )
        audit: Dict[str, Any] = {
            "fit_row_fingerprint": row["fit_row_fingerprint"],
            "uses_training_values_only": True,
            "summaries": summaries,
            "n_contracts_with_issues": len(issue_names),
            "changed_contracts": [],
        }
        if not issue_names:
            _write_json(context_dir / "harmonization" / "training_value_audit.json", audit)
            return list(registry), audit, []

        by_name = {entry["name"]: entry for entry in registry}
        summary_by_name = {summary["name"]: summary for summary in summaries}
        issue_entries = [by_name[name] for name in sorted(issue_names)]
        contexts: List[Dict[str, Any]] = []
        for batch_index, start in enumerate(
            range(0, len(issue_entries), _VALUE_HARMONIZATION_BATCH_SIZE), start=1
        ):
            batch = issue_entries[start : start + _VALUE_HARMONIZATION_BATCH_SIZE]
            contexts.append(
                {
                    **self._harmonization_context_base(
                        row,
                        prompt_version=TOPIC_VALUE_REPAIR_PROMPT_VERSION,
                        stage="training_value_contract_repair",
                    ),
                    "request_id": f"training_value_repair_{batch_index:04d}",
                    "candidates": [
                        {
                            **_compact_harmonization_candidate(entry),
                            "current_value_contract": entry.get("value_contract"),
                            "training_value_summary": summary_by_name[entry["name"]],
                        }
                        for entry in batch
                    ],
                }
            )
        records = self._run_harmonization_requests(
            contexts=contexts,
            checkpoint=(context_dir / "harmonization" / "training_value_repairs.jsonl"),
        )
        responses = [
            record["response"] for record in records if record.get("status") == "completed"
        ]
        repaired, repair_drops = apply_topic_value_harmonization(issue_entries, responses)
        repaired_by_name = {entry["name"]: entry for entry in repaired}
        combined: List[Dict[str, Any]] = []
        for entry in registry:
            if entry["name"] not in issue_names:
                combined.append(entry)
            elif entry["name"] in repaired_by_name:
                combined.append(repaired_by_name[entry["name"]])
        combined, merge_drops = _merge_executable_registry_entries(combined)
        repair_drops.extend(merge_drops)
        old_hashes = {entry["name"]: entry["contract_hash"] for entry in registry}
        changed = [
            entry["name"]
            for entry in combined
            if old_hashes.get(entry["name"]) != entry["contract_hash"]
        ]
        audit["changed_contracts"] = changed
        audit["repair_requests"] = records
        audit["dropped"] = repair_drops
        _write_json(context_dir / "harmonization" / "training_value_audit.json", audit)
        return combined, audit, repair_drops

    def _extract_scope(
        self, row_ids: Sequence[int], registry: Sequence[Dict[str, Any]]
    ) -> pd.DataFrame:
        positions = list(map(int, row_ids))
        source = self.dataset.set_index("_oci_row_id").loc[positions].reset_index()
        label_free = source[["_oci_row_id", self.config.text_column]].copy()
        all_specs = registry_specs(registry)
        action_by_name = {entry["name"]: entry.get("action", "extract") for entry in registry}
        extraction_specs = [
            spec for spec in all_specs if action_by_name.get(spec.name) == "extract"
        ]
        extracted = (
            self.extraction_provider.ensure_features(label_free, extraction_specs)
            if extraction_specs
            else label_free
        )
        extracted = apply_registry_derivations(extracted, registry)
        columns = [
            column
            for column in extracted.columns
            if column == "_oci_row_id" or column.startswith("explicit_feat_")
        ]
        return source.merge(extracted[columns], on="_oci_row_id", how="left", validate="one_to_one")

    def _run_inner_context(self, row: Dict[str, Any], outer_dir: Path) -> Dict[str, Any]:
        inner_fold = int(row["inner_fold"])
        context_dir = outer_dir / f"inner_{inner_fold:03d}"
        labels = self._label_context_topics(row, context_dir)
        initially_labeled_topic_ids = {str(label["topic_id"]) for label in labels}
        recovery_topic_attempts: Counter = Counter()
        attempted_recovery_raw_terms: set[str] = set()
        recovery_topic_evidence: List[Dict[str, Any]] = []
        candidates = [
            {
                **candidate,
                "source_topic_general_topic": ((label.get("response") or {}).get("general_topic")),
                "source_topic_quality": ((label.get("response") or {}).get("topic_quality")),
            }
            for label in labels
            for candidate in label.get("candidates", [])
        ]
        candidate_pool, dropped = self._harmonize_context_candidates(
            row=row,
            context_dir=context_dir,
            candidates=candidates,
        )
        registry, deferred_registry, initial_selection = select_initial_topic_evidence_registry(
            candidate_pool,
            row["discovery"],
            coverage_target=float(self.nn_config.tfidf_topic.initial_effect_coverage_target),
        )
        _write_json(context_dir / "initial_review_selection.json", initial_selection)
        fit_df = self._extract_scope(row["fit_row_ids"], registry)
        registry, value_audit, value_drops = self._revise_value_contracts_from_training(
            row=row,
            context_dir=context_dir,
            registry=registry,
            fit_df=fit_df,
        )
        dropped.extend(value_drops)
        candidate_pool, pool_merge_drops = _merge_executable_registry_entries(
            [
                *deferred_registry,
                *registry,
            ]
        )
        dropped.extend(pool_merge_drops)
        deferred_registry = [
            entry
            for entry in candidate_pool
            if entry["name"] not in {row["name"] for row in registry}
        ]
        if value_audit.get("changed_contracts"):
            fit_df = self._extract_scope(row["fit_row_ids"], registry)
        heldout_df = self._extract_scope(row["heldout_row_ids"], registry)
        diagnostic = structured_heldout_diagnostic(
            fit_df=fit_df,
            heldout_df=heldout_df,
            registry=registry,
            metadata=row["discovery"],
            config=self.config,
            candidate_evidence_universe=candidate_pool,
        )
        gate = structured_review_gate(diagnostic, self.nn_config)
        rounds = [
            {
                "round": 0,
                "gate": gate,
                "diagnostic": diagnostic,
                "registry_size": len(registry),
                "deferred_valid_contracts": len(deferred_registry),
            }
        ]
        # Recovery is additive. It never removes a valid existing definition.
        previous_score = int(gate["n_failed_criteria"])
        consecutive_non_improving = 0
        for round_index in range(1, int(self.nn_config.extracted_feature_review_max_rounds) + 1):
            if gate["passed"]:
                break
            addition_registry, deferred_addition_audit = select_deferred_review_additions(
                registry,
                deferred_registry,
                gate,
                diagnostic,
                row["discovery"],
                max_additions=20,
            )
            newly_dropped: List[Dict[str, Any]] = []
            recovery_source = "deferred_canonical_contracts"
            recovery_prompt_context: Optional[Dict[str, Any]] = None
            recovery_topic_summary: Optional[Dict[str, Any]] = None
            if not addition_registry:
                # Only when the fold-local valid pool has no relevant contract
                # do we revisit one uncovered source topic with a bounded prompt.
                uncovered = diagnostic["effect_coverage"].get("uncovered_topic_ids", [])
                failed_targets = [
                    str(criterion.get("target"))
                    for criterion in gate.get("criteria", [])
                    if not criterion.get("passed", False)
                    and criterion.get("family") == "nuisance"
                    and criterion.get("target") in {"treatment", "outcome"}
                ]
                relevant_banks = set(failed_targets)
                if any(
                    not criterion.get("passed", False) and criterion.get("family") == "effect"
                    for criterion in gate.get("criteria", [])
                ):
                    relevant_banks.add("effect")
                ranked_jobs = [
                    *self._inner_ranked_topic_jobs(row),
                    *self._inner_orphan_jobs(row),
                ]
                if not relevant_banks:
                    relevant_banks = {bank for bank, _topic in ranked_jobs}
                by_id = {str(topic["topic_id"]): (bank, topic) for bank, topic in ranked_jobs}
                topic_choice: Optional[Tuple[str, Dict[str, Any]]] = next(
                    (
                        by_id[str(topic_id)]
                        for topic_id in uncovered
                        if str(topic_id) in by_id and recovery_topic_attempts[str(topic_id)] == 0
                    ),
                    None,
                )
                if topic_choice is None:
                    # Expand beyond the initial shortlist in fixed held-out
                    # evidence order when the structured model still misses a
                    # diagnostic family. This is bounded by the global recovery
                    # round limit and never consults a new row set.
                    topic_choice = next(
                        (
                            (bank, topic)
                            for bank, topic in ranked_jobs
                            if bank in relevant_banks
                            and (
                                bank != "effect"
                                or not bool(self.nn_config.tfidf_topic.orphan_ngram_enabled)
                                or str(topic["topic_id"]) in initially_labeled_topic_ids
                            )
                            and str(topic["topic_id"]) not in initially_labeled_topic_ids
                            and recovery_topic_attempts[str(topic["topic_id"])] == 0
                        ),
                        None,
                    )
                if topic_choice is None:
                    topic_choice = next(
                        (
                            (bank, topic)
                            for bank, topic in ranked_jobs
                            if bank in relevant_banks
                            and (
                                bank != "effect"
                                or not bool(self.nn_config.tfidf_topic.orphan_ngram_enabled)
                                or str(topic["topic_id"]) in initially_labeled_topic_ids
                            )
                            and recovery_topic_attempts[str(topic["topic_id"])] == 0
                        ),
                        None,
                    )
                if topic_choice is None:
                    rounds.append(
                        {
                            "round": round_index,
                            "stop": "no_relevant_deferred_contract_or_source_topic",
                        }
                    )
                    break
                bank, topic = topic_choice
                topic_id = str(topic["topic_id"])
                recovery_topic_attempts[topic_id] += 1
                score_path = row["discovery"]["artifacts"]["ngram_scores"][bank]
                raw_scores = pd.read_parquet(score_path)
                covered_raw_terms = {
                    str(term.get("term") if isinstance(term, dict) else term)
                    for entry in registry
                    for provenance in entry.get("provenance", [])
                    for term in provenance.get("supporting_terms", [])
                }
                covered_raw_terms.update(attempted_recovery_raw_terms)
                if bank == "effect":
                    preserved = set(
                        diagnostic["effect_coverage"].get("preserved_highest_ranked_raw_ngrams", [])
                    )
                    covered_raw_terms.update(map(str, preserved))
                is_orphan_cluster = topic.get("evidence_kind") == "orphan_raw_ngram_cluster"
                raw_terms = (
                    []
                    if is_orphan_cluster
                    else select_topic_recovery_raw_ngrams(
                        raw_scores,
                        topic,
                        excluded_terms=covered_raw_terms,
                        limit=20,
                    )
                )
                attempted_recovery_raw_terms.update(raw_terms)
                recovery_prompt_context = build_topic_label_context(
                    outer_fold=int(row["outer_fold"]),
                    scope=str(row["scope"]),
                    inner_fold=inner_fold,
                    bank=bank,
                    topic=topic,
                    prompt_version=(
                        ORPHAN_NGRAM_LABEL_PROMPT_VERSION
                        if is_orphan_cluster
                        else TOPIC_RECOVERY_PROMPT_VERSION
                    ),
                    uncovered_raw_ngrams=raw_terms,
                    current_definitions=[
                        {
                            key: entry.get(key)
                            for key in ("name", "type", "categories", "description")
                        }
                        for entry in registry
                        if any(
                            provenance.get("topic_id") == topic_id
                            or bool(
                                {
                                    str(term.get("term"))
                                    for term in provenance.get("supporting_terms", [])
                                }
                                & set(raw_terms)
                            )
                            for provenance in entry.get("provenance", [])
                        )
                    ],
                    extraction_failures=[
                        summary
                        for summary in diagnostic.get("extraction_summary", [])
                        if float(summary.get("coverage", 0.0))
                        < float(self.search_config.min_feature_coverage)
                    ],
                    score_test_evidence=topic.get("_selection_evidence"),
                )
                recovery_topic_summary = {
                    "round": round_index,
                    "bank": bank,
                    "topic_id": topic_id,
                    "was_initially_selected": topic_id in initially_labeled_topic_ids,
                    "terms": [str(term.get("term")) for term in topic.get("terms", [])],
                    "score_test_evidence": dict(topic.get("_selection_evidence") or {}),
                }
                recovery_topic_evidence.append(recovery_topic_summary)
                response = self.proposal_agent.propose(recovery_prompt_context)
                additions = [
                    candidate
                    for proposal in _proposal_rows(response)[:20]
                    if (
                        candidate := _candidate_from_response(
                            proposal, context=recovery_prompt_context, topic=topic
                        )
                    )
                    is not None
                ]
                addition_registry, newly_dropped = self._harmonize_context_candidates(
                    row=row,
                    context_dir=context_dir / f"recovery_round_{round_index:02d}",
                    candidates=additions,
                    include_prespecified=False,
                )
                recovery_source = "targeted_source_topic_prompt"
            old_registry_state = {
                entry["name"]: _registry_entry_state_hash(entry) for entry in registry
            }
            old_contract_hashes = {entry["name"]: entry["contract_hash"] for entry in registry}
            updated, merge_drops = _merge_executable_registry_entries(
                [*registry, *addition_registry]
            )
            newly_dropped.extend(merge_drops)
            changed = [
                entry
                for entry in updated
                if old_registry_state.get(entry["name"]) != _registry_entry_state_hash(entry)
            ]
            if not changed:
                rounds.append(
                    {
                        "round": round_index,
                        "stop": "no_registry_specification_change",
                        "recovery_source": recovery_source,
                        "deferred_addition_audit": deferred_addition_audit,
                        "recovery_topic": recovery_topic_summary,
                    }
                )
                break
            registry = updated
            added_names = {entry["name"] for entry in changed}
            deferred_registry = [
                entry for entry in deferred_registry if entry["name"] not in added_names
            ]
            dropped.extend(newly_dropped)
            fit_df = self._extract_scope(row["fit_row_ids"], registry)
            registry, recovery_value_audit, recovery_value_drops = (
                self._revise_value_contracts_from_training(
                    row=row,
                    context_dir=context_dir / f"recovery_round_{round_index:02d}",
                    registry=registry,
                    fit_df=fit_df,
                )
            )
            dropped.extend(recovery_value_drops)
            candidate_pool, recovery_pool_drops = _merge_executable_registry_entries(
                [*deferred_registry, *registry]
            )
            dropped.extend(recovery_pool_drops)
            active_names = {entry["name"] for entry in registry}
            deferred_registry = [
                entry for entry in candidate_pool if entry["name"] not in active_names
            ]
            if recovery_value_audit.get("changed_contracts"):
                fit_df = self._extract_scope(row["fit_row_ids"], registry)
            heldout_df = self._extract_scope(row["heldout_row_ids"], registry)
            diagnostic = structured_heldout_diagnostic(
                fit_df=fit_df,
                heldout_df=heldout_df,
                registry=registry,
                metadata=row["discovery"],
                config=self.config,
                candidate_evidence_universe=candidate_pool,
            )
            gate = structured_review_gate(diagnostic, self.nn_config)
            score = int(gate["n_failed_criteria"])
            consecutive_non_improving = (
                consecutive_non_improving + 1 if score >= previous_score else 0
            )
            previous_score = score
            rounds.append(
                {
                    "round": round_index,
                    "gate": gate,
                    "diagnostic": diagnostic,
                    "changed_registry_entries": [entry["name"] for entry in changed],
                    "materially_changed_extraction_contracts": [
                        entry["name"]
                        for entry in changed
                        if old_contract_hashes.get(entry["name"]) != entry["contract_hash"]
                    ],
                    "registry_size": len(registry),
                    "deferred_valid_contracts": len(deferred_registry),
                    "recovery_source": recovery_source,
                    "deferred_addition_audit": deferred_addition_audit,
                    "targeted_prompt_context_hash": (
                        stable_hash(recovery_prompt_context)
                        if recovery_prompt_context is not None
                        else None
                    ),
                    "recovery_topic": recovery_topic_summary,
                }
            )
            if consecutive_non_improving >= 2:
                rounds[-1]["stop"] = "two_consecutive_non_improving_rounds"
                break

        pre_parsimony_registry = [dict(entry) for entry in registry]
        registry, parsimony = self._parsimony_review(
            outer_fold=int(row["outer_fold"]),
            inner_fold=inner_fold,
            train_df=fit_df,
            heldout_df=heldout_df,
            discovery_metadata=row["discovery"],
            base_diagnostic=diagnostic,
            registry=registry,
            outer_dir=context_dir,
            candidate_evidence_universe=candidate_pool,
        )
        if int(parsimony.get("accepted_replacements", 0)) > 0:
            fit_df = self._extract_scope(row["fit_row_ids"], registry)
            heldout_df = self._extract_scope(row["heldout_row_ids"], registry)
            diagnostic = structured_heldout_diagnostic(
                fit_df=fit_df,
                heldout_df=heldout_df,
                registry=registry,
                metadata=row["discovery"],
                config=self.config,
                candidate_evidence_universe=candidate_pool,
            )
            gate = structured_review_gate(diagnostic, self.nn_config)

        audit: Dict[str, Any]
        try:
            evaluation: SplitEvaluation = self.evaluator.evaluate_split(
                train_df=self._without_oracle_columns(fit_df),
                test_df=self._without_oracle_columns(heldout_df),
                specs=registry_specs(registry),
                fold_id=1000 * int(row["outer_fold"]) + inner_fold,
            )
            audit = {
                "status": "completed",
                "used_for_pruning": False,
                "metrics": evaluation.metrics,
            }
        except Exception as exc:
            audit = {"status": "failed", "used_for_pruning": False, "error": repr(exc)}
        result = {
            "outer_fold": int(row["outer_fold"]),
            "inner_fold": inner_fold,
            "registry": registry,
            "initial_review_selection": initial_selection,
            "candidate_pool_size": len(candidate_pool),
            "deferred_valid_contracts": [
                {
                    "name": entry["name"],
                    "contract_hash": entry["contract_hash"],
                    "roles": entry.get("roles", []),
                    "reason": (
                        "heldout_review_gates_passed_without_contract"
                        if gate.get("passed")
                        else "not_added_before_configured_review_stopping_rule"
                    ),
                }
                for entry in deferred_registry
            ],
            "dropped": dropped,
            "training_value_audit": value_audit,
            "review_rounds": rounds,
            "final_gate": gate,
            "final_diagnostic": diagnostic,
            "parsimony": parsimony,
            "pre_parsimony_feature_names": [entry["name"] for entry in pre_parsimony_registry],
            "inner_forest_audit": audit,
            "topic_score_selection": self._topic_score_selection_snapshot(row),
            "recovery_topic_evidence": recovery_topic_evidence,
        }
        _write_json(
            context_dir / "canonical_registry.json",
            {
                "schema_version": CANONICAL_REGISTRY_SCHEMA_VERSION,
                **result,
            },
        )
        return result

    def _aggregate_policy(self, inner_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        name_counts = Counter(
            entry["name"] for result in inner_results for entry in result["registry"]
        )
        topic_counts = Counter(
            provenance["topic_id"]
            for result in inner_results
            for entry in result["registry"]
            for provenance in entry.get("provenance", [])
        )
        domain_parent_counts = Counter(
            f"{entry.get('clinical_domain', '')}::{entry.get('parent_object', '')}"
            for result in inner_results
            for entry in result["registry"]
        )
        term_fold_sets: Dict[str, set] = defaultdict(set)
        feature_evidence: List[Dict[str, Any]] = []
        entries_by_name: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
        for result in inner_results:
            inner_fold = int(result["inner_fold"])
            for entry in result["registry"]:
                entries_by_name[entry["name"]].append((inner_fold, entry))
                terms = sorted(
                    {
                        str(term.get("term") if isinstance(term, dict) else term)
                        for provenance in entry.get("provenance", [])
                        for term in provenance.get("supporting_terms", [])
                    }
                )
                for term in terms:
                    term_fold_sets[term].add(inner_fold)
                feature_evidence.append(
                    {
                        "inner_fold": inner_fold,
                        "name": entry["name"],
                        "description": entry.get("description"),
                        "clinical_domain": entry.get("clinical_domain"),
                        "parent_object": entry.get("parent_object"),
                        "roles": entry.get("roles"),
                        "supporting_terms": terms,
                        "source_banks": sorted(
                            {
                                str(provenance.get("bank"))
                                for provenance in entry.get("provenance", [])
                            }
                        ),
                        "gate_passed": bool(result["final_gate"]["passed"]),
                    }
                )
        replacements: List[Dict[str, Any]] = []
        for result in inner_results:
            before = set(result.get("pre_parsimony_feature_names") or [])
            after = {entry["name"] for entry in result["registry"]}
            if before != after:
                replacements.append(
                    {
                        "inner_fold": int(result["inner_fold"]),
                        "removed_names": sorted(before - after),
                        "added_entries": [
                            entry for entry in result["registry"] if entry["name"] not in before
                        ],
                        "accepted_cluster_ids": result["parsimony"].get("accepted_cluster_ids", []),
                    }
                )
        completed_rounds = [
            max(
                [int(round_row.get("round", 0)) for round_row in result.get("review_rounds", [])],
                default=0,
            )
            for result in inner_results
        ]
        stable_recovery_candidates = []
        recurrence_minimum = min(
            max(2, int(self.nn_config.candidate_consistency_min_folds)),
            max(2, len(inner_results)),
        )
        for name, occurrences in sorted(entries_by_name.items()):
            folds = sorted({fold for fold, _entry in occurrences})
            if len(folds) < recurrence_minimum:
                continue
            template = max(
                (entry for _fold, entry in occurrences),
                key=lambda entry: len(str(entry.get("description") or "")),
            )
            stable_recovery_candidates.append(
                {
                    "name": name,
                    "supporting_inner_folds": folds,
                    "entry": template,
                }
            )
        score_policy_banks: Dict[str, Any] = {}
        for bank in ("treatment", "outcome", "effect"):
            signatures: List[Dict[str, Any]] = []
            selected_term_folds: Dict[str, set] = defaultdict(set)
            selected_ngram_folds: Dict[str, set] = defaultdict(set)
            for result in inner_results:
                inner_fold = int(result["inner_fold"])
                seen_inner_topics: set = set()
                bank_selection = (
                    (result.get("topic_score_selection") or {}).get("banks", {}).get(bank, {})
                )
                for ngram in bank_selection.get("selected_ngrams", []):
                    term = str(ngram.get("term") or "")
                    if term:
                        selected_ngram_folds[term].add(inner_fold)
                for topic in bank_selection.get("selected_topics", []):
                    seen_inner_topics.add(str(topic.get("topic_id")))
                    term_scores = list(topic.get("term_scores") or [])
                    terms = [
                        str(term.get("term"))
                        for term in term_scores
                        if str(term.get("term") or "").strip()
                    ]
                    for term in terms:
                        selected_term_folds[term].add(inner_fold)
                    signatures.append(
                        {
                            "inner_fold": inner_fold,
                            "topic_id": str(topic.get("topic_id")),
                            "terms": terms,
                            "primary_p": topic.get("primary_p"),
                            "primary_p_source": topic.get("primary_p_source"),
                            "fdr_q": topic.get("fdr_q"),
                            "familywise_p": topic.get("familywise_p"),
                            "topic_standardized_score": topic.get("topic_standardized_score"),
                            "topic_score_moment": topic.get("topic_score_moment"),
                            "topic_familywise_p": topic.get("topic_familywise_p"),
                            "term_group_primary_p": topic.get("term_group_primary_p"),
                            "term_group_fdr_q": topic.get("term_group_fdr_q"),
                            "selection_reason": topic.get("selection_reason"),
                            "evidence_rank": topic.get("evidence_rank"),
                            "quadratic_statistic_per_rank": topic.get(
                                "quadratic_statistic_per_rank"
                            ),
                            "term_scores": term_scores,
                            "selected_ngram_terms": [
                                str(term.get("term"))
                                for term in term_scores
                                if bool(term.get("selected_for_agent_evidence"))
                            ],
                            "policy_source": "initial_score_test_shortlist",
                        }
                    )
                for recovered in result.get("recovery_topic_evidence", []):
                    if str(recovered.get("bank")) != bank:
                        continue
                    topic_id = str(recovered.get("topic_id"))
                    if topic_id in seen_inner_topics:
                        continue
                    evidence = dict(recovered.get("score_test_evidence") or {})
                    term_scores = list(evidence.get("term_scores") or [])
                    terms = [
                        str(term) for term in (recovered.get("terms") or []) if str(term).strip()
                    ]
                    if not terms:
                        terms = [
                            str(term.get("term"))
                            for term in term_scores
                            if str(term.get("term") or "").strip()
                        ]
                    for term in terms:
                        selected_term_folds[term].add(inner_fold)
                    signatures.append(
                        {
                            "inner_fold": inner_fold,
                            "topic_id": topic_id,
                            "terms": terms,
                            "primary_p": evidence.get("primary_p"),
                            "primary_p_source": evidence.get("primary_p_source"),
                            "fdr_q": evidence.get("fdr_q"),
                            "familywise_p": evidence.get("familywise_p"),
                            "topic_standardized_score": evidence.get("topic_standardized_score"),
                            "topic_score_moment": evidence.get("topic_score_moment"),
                            "topic_familywise_p": evidence.get("topic_familywise_p"),
                            "term_group_primary_p": evidence.get("term_group_primary_p"),
                            "term_group_fdr_q": evidence.get("term_group_fdr_q"),
                            "selection_reason": "additive_review_expansion",
                            "evidence_rank": evidence.get("evidence_rank"),
                            "quadratic_statistic_per_rank": evidence.get(
                                "quadratic_statistic_per_rank"
                            ),
                            "term_scores": term_scores,
                            "selected_ngram_terms": [
                                str(term.get("term"))
                                for term in term_scores
                                if bool(term.get("selected_for_agent_evidence"))
                            ],
                            "policy_source": "additive_review_expansion",
                            "recovery_round": recovered.get("round"),
                        }
                    )
                    seen_inner_topics.add(topic_id)
            score_policy_banks[bank] = {
                "signatures": signatures,
                "selected_topic_count": len(signatures),
                "selected_term_fold_recurrence": {
                    term: len(folds) for term, folds in sorted(selected_term_folds.items())
                },
                "selected_ngram_fold_recurrence": {
                    term: len(folds) for term, folds in sorted(selected_ngram_folds.items())
                },
                "selected_ngram_count": len(selected_ngram_folds),
                "inner_folds_with_selected_topics": sorted(
                    {int(signature["inner_fold"]) for signature in signatures}
                ),
            }
        orphan_signatures: List[Dict[str, Any]] = []
        orphan_term_folds: Dict[str, set] = defaultdict(set)
        for result in inner_results:
            inner_fold = int(result["inner_fold"])
            orphan_selection = (result.get("topic_score_selection") or {}).get(
                "effect_orphan_ngram_branch"
            ) or {}
            for cluster in orphan_selection.get("selected_clusters") or []:
                term_scores = list(cluster.get("term_scores") or [])
                terms = [
                    str(term.get("term"))
                    for term in term_scores
                    if str(term.get("term") or "").strip()
                ]
                for term in terms:
                    orphan_term_folds[term].add(inner_fold)
                orphan_signatures.append(
                    {
                        "inner_fold": inner_fold,
                        "cluster_id": str(cluster.get("cluster_id")),
                        "terms": terms,
                        "primary_p": cluster.get("primary_p"),
                        "primary_p_source": cluster.get("primary_p_source"),
                        "fdr_q": cluster.get("fdr_q"),
                        "familywise_p": cluster.get("familywise_p"),
                        "quadratic_statistic_per_rank": cluster.get("quadratic_statistic_per_rank"),
                        "maximum_absolute_standardized_score": cluster.get(
                            "maximum_absolute_standardized_score"
                        ),
                        "evidence_rank": cluster.get("evidence_rank"),
                        "term_scores": term_scores,
                    }
                )
        orphan_policy = {
            "enabled": bool(self.nn_config.tfidf_topic.orphan_ngram_enabled),
            "signatures": orphan_signatures,
            "selected_cluster_count": len(orphan_signatures),
            "selected_term_fold_recurrence": {
                term: len(folds) for term, folds in sorted(orphan_term_folds.items())
            },
            "inner_folds_with_selected_clusters": sorted(
                {int(signature["inner_fold"]) for signature in orphan_signatures}
            ),
            "full_context_min_inner_folds": int(
                self.nn_config.tfidf_topic.orphan_ngram_full_min_inner_folds
            ),
        }
        return {
            "policy_version": "tfidf_topic_inner_policy_v6",
            "inner_fold_count": len(inner_results),
            "exact_name_recurrence": dict(name_counts),
            "domain_parent_recurrence": dict(domain_parent_counts),
            "supporting_topic_recurrence": dict(topic_counts),
            "supporting_term_fold_recurrence": {
                term: len(folds) for term, folds in term_fold_sets.items()
            },
            "inner_feature_evidence": feature_evidence,
            "stable_recovery_candidates": stable_recovery_candidates,
            "passing_inner_folds": sum(result["final_gate"]["passed"] for result in inner_results),
            "nuisance_auc_tolerance": self.nn_config.extracted_feature_review_auc_margin,
            "nuisance_loss_relative_tolerance": (
                self.nn_config.extracted_feature_review_loss_relative_margin
            ),
            "effect_coverage_target": self.nn_config.tfidf_topic.initial_effect_coverage_target,
            "fixed_recovery_round_limit": (
                int(np.median(completed_rounds)) if completed_rounds else 0
            ),
            "inner_accepted_parsimony_replacements": replacements,
            "strong_full_outer_evidence_not_vetoed_by_name_recurrence": True,
            "agent_selector_can_veto_stable_evidence": False,
            "topic_score_selection": {
                "schema_version": "tfidf_topic_inner_score_policy_v5",
                "uses_outer_heldout_labels": False,
                "banks": score_policy_banks,
                "effect_orphan_ngram_branch": orphan_policy,
            },
        }

    def _recover_from_inner_policy(
        self, registry: Sequence[Dict[str, Any]], policy: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Add recurrent inner candidates absent from full-outer naming, without vetoes."""
        current = list(registry)
        existing_names = {entry["name"] for entry in current}

        def term_set(entry: Dict[str, Any]) -> set:
            return {
                str(term.get("term") if isinstance(term, dict) else term)
                for provenance in entry.get("provenance", [])
                for term in provenance.get("supporting_terms", [])
            }

        existing_terms = [term_set(entry) for entry in current]
        recovered: List[str] = []
        semantically_present: List[str] = []
        for candidate in policy.get("stable_recovery_candidates") or []:
            template = dict(candidate.get("entry") or {})
            name = str(template.get("name") or "")
            if not name or name in existing_names:
                continue
            template_terms = term_set(template)
            domain_parent = (
                template.get("clinical_domain"),
                template.get("parent_object"),
            )
            equivalent_present = any(
                (
                    entry.get("clinical_domain"),
                    entry.get("parent_object"),
                )
                == domain_parent
                or bool(template_terms & terms)
                for entry, terms in zip(current, existing_terms)
            )
            if equivalent_present:
                semantically_present.append(name)
                continue
            template["recovered_from_inner_policy"] = True
            template["supporting_inner_folds"] = candidate.get("supporting_inner_folds", [])
            current.append(_refresh_registry_entry(template))
            existing_terms.append(template_terms)
            existing_names.add(name)
            recovered.append(name)
        merged, merge_drops = _merge_executable_registry_entries(current)
        return merged, {
            "policy_fixed_before_full_outer_recovery": True,
            "recovered_names": recovered,
            "semantically_present_under_other_names": semantically_present,
            "merge_drops": merge_drops,
            "full_outer_candidates_vetoed_for_inner_nonrecurrence": 0,
        }

    def _annotate_registry_with_inner_policy(
        self, registry: Sequence[Dict[str, Any]], policy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Attach recurrence priority without using it as a hard full-outer veto."""
        evidence = list(policy.get("inner_feature_evidence") or [])

        def tokens(value: Any) -> set:
            return {
                token
                for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
                if len(token) > 1
            }

        output: List[Dict[str, Any]] = []
        for raw in registry:
            entry = dict(raw)
            entry_terms = {
                str(term.get("term") if isinstance(term, dict) else term)
                for provenance in entry.get("provenance", [])
                for term in provenance.get("supporting_terms", [])
            }
            entry_tokens = tokens(entry["name"]) | tokens(entry.get("description"))
            matched_folds: set = set()
            semantic_folds: set = set()
            topic_term_folds: set = set()
            best_similarity = 0.0
            for item in evidence:
                fold = int(item["inner_fold"])
                if item.get("name") == entry["name"]:
                    matched_folds.add(fold)
                other_tokens = tokens(item.get("name")) | tokens(item.get("description"))
                union = entry_tokens | other_tokens
                similarity = 0.0 if not union else len(entry_tokens & other_tokens) / len(union)
                best_similarity = max(best_similarity, similarity)
                if similarity >= 0.50 or (
                    item.get("clinical_domain") == entry.get("clinical_domain")
                    and item.get("parent_object") == entry.get("parent_object")
                ):
                    semantic_folds.add(fold)
                if entry_terms & set(item.get("supporting_terms") or []):
                    topic_term_folds.add(fold)
            recurrence = {
                "exact_name_inner_folds": sorted(matched_folds),
                "semantic_inner_folds": sorted(semantic_folds),
                "supporting_term_inner_folds": sorted(topic_term_folds),
                "best_contract_token_similarity": float(best_similarity),
                "priority_fold_count": len(matched_folds | semantic_folds | topic_term_folds),
                "hard_requirement": False,
            }
            entry["inner_recurrence_evidence"] = recurrence
            entry["retained_despite_name_nonrecurrence_allowed"] = True
            output.append(entry)
        return output

    def _parsimony_review(
        self,
        *,
        outer_fold: int,
        inner_fold: Optional[int] = None,
        train_df: pd.DataFrame,
        heldout_df: Optional[pd.DataFrame] = None,
        discovery_metadata: Optional[Dict[str, Any]] = None,
        base_diagnostic: Optional[Dict[str, Any]] = None,
        registry: List[Dict[str, Any]],
        outer_dir: Path,
        fixed_policy: Optional[Dict[str, Any]] = None,
        candidate_evidence_universe: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.nn_config.parsimony_review_enabled or len(registry) < 2:
            return registry, {
                "schema_version": PARSIMONY_SCHEMA_VERSION,
                "enabled": bool(self.nn_config.parsimony_review_enabled),
                "stop_reason": "disabled_or_fewer_than_two_features",
            }
        specs = registry_specs(registry)
        documents = [_parsimony_feature_contract_document(spec) for spec in specs]
        semantic = _parsimony_tfidf_semantic_vectors(documents)
        cluster_result = _build_value_driven_feature_clusters(
            train_df=train_df,
            specs=specs,
            semantic_vectors=semantic,
            nn_config=self.nn_config,
            random_state=97_000 + outer_fold,
        )
        # A factor is never accepted merely because an agent proposed it. The
        # exact inner contexts in this implementation have already fixed the
        # admissible policy; absent a dimension-reducing replacement that has
        # every diagnostic family attached, retain the original cluster.
        clusters = list(cluster_result.get("clusters", []))
        protected_names = {spec.name for spec in self.prespecified_specs}

        if heldout_df is None:
            # Full-outer decisions cannot consult the outer test. Apply only an
            # identical replacement that independently passed in enough inner
            # contexts; otherwise the unpruned full-outer registry wins.
            replacements = list(
                (fixed_policy or {}).get("inner_accepted_parsimony_replacements") or []
            )
            signatures: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for replacement in replacements:
                signature = stable_hash(
                    {
                        "removed_names": replacement.get("removed_names") or [],
                        "added_names": sorted(
                            entry["name"] for entry in replacement.get("added_entries") or []
                        ),
                    }
                )
                signatures[signature].append(replacement)
            required_recurrence = min(
                max(2, int(self.nn_config.candidate_consistency_min_folds)),
                int((fixed_policy or {}).get("inner_fold_count") or 2),
            )

            def encoded_dimension(entries: Sequence[Dict[str, Any]]) -> int:
                total = 0
                for entry in entries:
                    base = (
                        1
                        if entry.get("type") == "continuous"
                        else max(1, len(entry.get("categories") or []))
                    )
                    role_count = int("confounder" in entry.get("roles", [])) + int(
                        "effect_modifier" in entry.get("roles", [])
                    )
                    total += base * max(1, role_count)
                return int(total)

            selected = list(registry)
            rows: List[Dict[str, Any]] = []
            accepted_signatures: List[str] = []
            for signature, occurrences in sorted(signatures.items()):
                template = occurrences[0]
                removed_names = set(template.get("removed_names") or [])
                added_templates = list(template.get("added_entries") or [])
                reasons: List[str] = []
                if len(occurrences) < required_recurrence:
                    reasons.append("replacement_did_not_recur_across_inner_folds")
                selected_by_name = {entry["name"]: entry for entry in selected}
                if not removed_names.issubset(selected_by_name):
                    reasons.append("full_outer_source_members_not_present")
                if protected_names & removed_names:
                    reasons.append("required_feature_would_be_removed")
                inherited_provenance = [
                    provenance
                    for name in removed_names
                    for provenance in selected_by_name.get(name, {}).get("provenance", [])
                ]
                inherited_roles = sorted(
                    {
                        role
                        for name in removed_names
                        for role in selected_by_name.get(name, {}).get("roles", [])
                    }
                )
                additions = []
                for template_entry in added_templates:
                    addition = dict(template_entry)
                    addition["provenance"] = inherited_provenance
                    addition["roles"] = inherited_roles
                    additions.append(_refresh_registry_entry(addition))
                trial = [
                    entry for entry in selected if entry["name"] not in removed_names
                ] + additions
                if encoded_dimension(trial) >= encoded_dimension(selected):
                    reasons.append("encoded_dimension_not_reduced")
                before_roles = {role for entry in selected for role in entry.get("roles", [])}
                after_roles = {role for entry in trial for role in entry.get("roles", [])}
                if before_roles != after_roles:
                    reasons.append("role_union_changed")
                allowed = not reasons
                rows.append(
                    {
                        "schema_version": PARSIMONY_SCHEMA_VERSION,
                        "outer_fold": outer_fold,
                        "inner_fold": None,
                        "phase": "fixed_inner_policy_application",
                        "replacement_signature": signature,
                        "inner_recurrence": len(occurrences),
                        "required_recurrence": required_recurrence,
                        "removed_names": sorted(removed_names),
                        "added_names": [entry["name"] for entry in additions],
                        "allowed": allowed,
                        "reasons": reasons,
                    }
                )
                if allowed:
                    selected = trial
                    accepted_signatures.append(signature)
            _write_jsonl(outer_dir / "parsimony_clusters.jsonl", rows)
            return selected, {
                "schema_version": PARSIMONY_SCHEMA_VERSION,
                "enabled": True,
                "uses_actual_training_values": True,
                "semantic_vectorizer": "word_character_contract_tfidf",
                "cluster_maximum": self.nn_config.parsimony_cluster_max_size,
                "n_value_driven_clusters": len(clusters),
                "fixed_from_inner_heldout_diagnostics": True,
                "accepted_replacements": len(accepted_signatures),
                "accepted_replacement_signatures": accepted_signatures,
                "stop_reason": (
                    "recurrent_inner_passing_replacements_applied"
                    if accepted_signatures
                    else "unpruned_set_preferred_without_recurrent_inner_passing_replacement"
                ),
            }

        def request_factor(cluster: Dict[str, Any]) -> Dict[str, Any]:
            members = [spec for spec in specs if spec.name in cluster.get("member_names", [])]
            context = {
                "prompt_version": "multi_model_agentic_parsimony_factor_v1",
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "cluster_id": cluster["cluster_id"],
                "replaceable_members": [
                    spec.name for spec in members if spec.name not in protected_names
                ],
                "protected_members": [
                    spec.name for spec in members if spec.name in protected_names
                ],
                "required_role_union": sorted({role for spec in members for role in spec.roles}),
                "max_factors": min(2, int(self.nn_config.parsimony_max_factors_per_cluster)),
                "members": [
                    {
                        "name": spec.name,
                        "type": spec.type,
                        "categories": spec.categories,
                        "roles": spec.roles,
                        "description": spec.description,
                    }
                    for spec in members
                ],
                "value_summaries": cluster.get("value_summaries", []),
                "pair_associations": cluster.get("pair_associations", []),
            }
            try:
                response = self.proposal_agent.propose(context)
                return {"context_hash": stable_hash(context), "response": response}
            except Exception as exc:
                return {"context_hash": stable_hash(context), "error": repr(exc)}

        factor_results: Dict[str, Dict[str, Any]] = {}
        if clusters:
            workers = min(len(clusters), int(self.nn_config.tfidf_topic.topic_label_parallelism))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                future_map = {
                    executor.submit(request_factor, cluster): cluster for cluster in clusters
                }
                for future in as_completed(future_map):
                    cluster = future_map[future]
                    factor_results[str(cluster["cluster_id"])] = future.result()
        rows: List[Dict[str, Any]] = []
        passing: List[Dict[str, Any]] = []
        registry_by_name = {entry["name"]: entry for entry in registry}

        def diagnostic_losses(trial: Dict[str, Any]) -> Tuple[float, float]:
            assert base_diagnostic is not None
            base_coverage = float(base_diagnostic["effect_coverage"].get("coverage_fraction", 0.0))
            trial_coverage = float(trial["effect_coverage"].get("coverage_fraction", 0.0))
            coverage_loss = max(0.0, base_coverage - trial_coverage)
            base_reconstruction = base_diagnostic["effect_topic_reconstruction"].get(
                "mean_correlation"
            )
            trial_reconstruction = trial["effect_topic_reconstruction"].get("mean_correlation")
            if base_reconstruction is None:
                reconstruction_loss = 0.0
            elif trial_reconstruction is None:
                reconstruction_loss = 1.0
            else:
                reconstruction_loss = max(
                    0.0, float(base_reconstruction) - float(trial_reconstruction)
                )
            return coverage_loss, reconstruction_loss

        parsimony_row = {
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "scope": "inner_parsimony_evaluation",
            "fit_row_fingerprint": row_set_fingerprint(
                train_df["_oci_row_id"].astype(int).tolist()
            ),
        }

        def evaluate_registry(
            trial_registry: List[Dict[str, Any]],
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int]:
            assert heldout_df is not None and discovery_metadata is not None
            evaluation_dir = (
                outer_dir
                / "parsimony_value_evaluations"
                / stable_hash(
                    [(entry["name"], entry["contract_hash"]) for entry in trial_registry]
                )[:16]
            )
            trial_fit = self._extract_scope(
                train_df["_oci_row_id"].astype(int).tolist(), trial_registry
            )
            trial_registry, _audit, _drops = self._revise_value_contracts_from_training(
                row=parsimony_row,
                context_dir=evaluation_dir,
                registry=trial_registry,
                fit_df=trial_fit,
            )
            if _audit.get("changed_contracts"):
                trial_fit = self._extract_scope(
                    train_df["_oci_row_id"].astype(int).tolist(), trial_registry
                )
            trial_heldout = self._extract_scope(
                heldout_df["_oci_row_id"].astype(int).tolist(), trial_registry
            )
            trial_diagnostic = structured_heldout_diagnostic(
                fit_df=trial_fit,
                heldout_df=trial_heldout,
                registry=trial_registry,
                metadata=discovery_metadata,
                config=self.config,
                candidate_evidence_universe=candidate_evidence_universe,
            )
            dimension = int(trial_diagnostic["n_w_encoded"]) + int(trial_diagnostic["n_x_encoded"])
            return trial_registry, trial_diagnostic, dimension

        for cluster in clusters:
            proposal = factor_results.get(str(cluster["cluster_id"]), {})
            row = {
                "schema_version": PARSIMONY_SCHEMA_VERSION,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                **cluster,
                "factor_proposal": proposal,
                "decision": "retain_original_cluster",
                "reason": "no_fully_evaluated_dimension_reducing_replacement",
            }
            response = proposal.get("response")
            context = {
                "cluster_id": cluster["cluster_id"],
                "max_factors": min(2, int(self.nn_config.parsimony_max_factors_per_cluster)),
            }
            candidate, validation = _validate_parsimony_factor_candidate(
                response=response,
                context=context,
                cluster=cluster,
                current_specs=specs,
                required_names=protected_names,
            )
            row["factor_validation"] = validation
            if (
                candidate is not None
                and heldout_df is not None
                and discovery_metadata is not None
                and base_diagnostic is not None
            ):
                replaces = list(candidate["replaces"])
                inherited_provenance = [
                    provenance
                    for name in replaces
                    for provenance in registry_by_name[name].get("provenance", [])
                ]
                factor_inputs = [
                    {
                        "name": factor.name,
                        "type": factor.type,
                        "categories": factor.categories,
                        "roles": factor.roles,
                        "description": factor.description,
                        "provenance": inherited_provenance,
                    }
                    for factor in candidate["factor_specs"]
                ]
                factor_registry, factor_drops = self._harmonize_context_candidates(
                    row=parsimony_row,
                    context_dir=(
                        outer_dir / "parsimony_factor_contracts" / str(cluster["cluster_id"])
                    ),
                    candidates=factor_inputs,
                    include_prespecified=False,
                )
                trial_registry = [
                    entry for entry in registry if entry["name"] not in set(replaces)
                ] + factor_registry
                trial_registry, trial_diagnostic, trial_dimension = evaluate_registry(
                    trial_registry
                )
                base_dimension = int(base_diagnostic["n_w_encoded"]) + int(
                    base_diagnostic["n_x_encoded"]
                )
                coverage_loss, reconstruction_loss = diagnostic_losses(trial_diagnostic)
                allowed, reasons = parsimony_replacement_passes(
                    base=base_diagnostic,
                    trial=trial_diagnostic,
                    base_dimension=base_dimension,
                    trial_dimension=trial_dimension,
                    source_topic_coverage_loss=coverage_loss,
                    topic_reconstruction_loss=reconstruction_loss,
                    required_features_preserved=protected_names.issubset(
                        {entry["name"] for entry in trial_registry}
                    ),
                    role_union_preserved=(
                        {role for entry in registry for role in entry["roles"]}
                        == {role for entry in trial_registry for role in entry["roles"]}
                    ),
                )
                factor_names = {entry["name"] for entry in factor_registry}
                factor_summaries = [
                    summary
                    for summary in trial_diagnostic.get("extraction_summary", [])
                    if summary.get("name") in factor_names
                ]
                extraction_quality_passed = bool(factor_summaries) and all(
                    float(summary.get("coverage", 0.0))
                    >= float(self.nn_config.parsimony_factor_min_coverage)
                    and int(summary.get("n_unique_observed", 0)) >= 2
                    for summary in factor_summaries
                )
                if not extraction_quality_passed:
                    allowed = False
                    reasons = [*reasons, "factor_not_operationally_extractable"]
                evaluation = {
                    "phase": "independent",
                    "allowed": allowed,
                    "reasons": reasons,
                    "base_dimension": base_dimension,
                    "trial_dimension": trial_dimension,
                    "source_topic_coverage_loss": coverage_loss,
                    "topic_reconstruction_loss": reconstruction_loss,
                    "factor_harmonization_drops": factor_drops,
                    "factor_extraction_quality": factor_summaries,
                }
                row["replacement_evaluation"] = evaluation
                if allowed:
                    passing.append(
                        {
                            "cluster_id": cluster["cluster_id"],
                            "replaces": set(replaces),
                            "factors": factor_registry,
                            "registry": trial_registry,
                            "diagnostic": trial_diagnostic,
                            "dimension": trial_dimension,
                            "row": row,
                        }
                    )
            rows.append(row)

        selected_registry = registry
        accepted_ids: List[str] = []
        if passing and base_diagnostic is not None:
            passing.sort(key=lambda item: (item["dimension"], item["cluster_id"]))
            nonoverlapping: List[Dict[str, Any]] = []
            removed: set = set()
            for item in passing:
                if item["replaces"] & removed:
                    continue
                nonoverlapping.append(item)
                removed.update(item["replaces"])
            joint_registry = [entry for entry in registry if entry["name"] not in removed] + [
                factor for item in nonoverlapping for factor in item["factors"]
            ]
            joint_registry, joint_diagnostic, joint_dimension = evaluate_registry(joint_registry)
            base_dimension = int(base_diagnostic["n_w_encoded"]) + int(
                base_diagnostic["n_x_encoded"]
            )
            coverage_loss, reconstruction_loss = diagnostic_losses(joint_diagnostic)
            joint_allowed, joint_reasons = parsimony_replacement_passes(
                base=base_diagnostic,
                trial=joint_diagnostic,
                base_dimension=base_dimension,
                trial_dimension=joint_dimension,
                source_topic_coverage_loss=coverage_loss,
                topic_reconstruction_loss=reconstruction_loss,
                required_features_preserved=protected_names.issubset(
                    {entry["name"] for entry in joint_registry}
                ),
                role_union_preserved=(
                    {role for entry in registry for role in entry["roles"]}
                    == {role for entry in joint_registry for role in entry["roles"]}
                ),
            )
            rows.append(
                {
                    "schema_version": PARSIMONY_SCHEMA_VERSION,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "phase": "joint_greedy",
                    "cluster_ids": [item["cluster_id"] for item in nonoverlapping],
                    "allowed": joint_allowed,
                    "reasons": joint_reasons,
                    "base_dimension": base_dimension,
                    "trial_dimension": joint_dimension,
                    "source_topic_coverage_loss": coverage_loss,
                    "topic_reconstruction_loss": reconstruction_loss,
                }
            )
            if joint_allowed:
                selected_registry = joint_registry
                accepted_ids = [item["cluster_id"] for item in nonoverlapping]
            else:
                selected_registry = passing[0]["registry"]
                accepted_ids = [passing[0]["cluster_id"]]
            for row in rows:
                if row.get("cluster_id") in accepted_ids:
                    row["decision"] = "replace_cluster"
                    row["reason"] = "all_required_heldout_diagnostics_passed"
        _write_jsonl(outer_dir / "parsimony_clusters.jsonl", rows)
        return selected_registry, {
            "schema_version": PARSIMONY_SCHEMA_VERSION,
            "enabled": True,
            "uses_actual_training_values": True,
            "semantic_vectorizer": "word_character_contract_tfidf",
            "cluster_maximum": self.nn_config.parsimony_cluster_max_size,
            "n_clusters": len(rows),
            "n_factor_proposals": sum(
                "response" in (row.get("factor_proposal") or {}) for row in rows
            ),
            "accepted_replacements": len(accepted_ids),
            "accepted_cluster_ids": accepted_ids,
            "stop_reason": (
                "smallest_fully_passing_dimension_selected"
                if accepted_ids
                else "unpruned_set_preferred_without_fully_passing_replacement"
            ),
        }

    def _run_outer(self, outer_fold: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        full_row, inner_rows = self._rows_for_outer(outer_fold)
        outer_dir = self.artifact_dir / f"outer_fold_{outer_fold:03d}"
        inner_results = [self._run_inner_context(row, outer_dir) for row in inner_rows]
        policy = self._aggregate_policy(inner_results)
        _write_json(outer_dir / "inner_policy.json", policy)

        full_context_dir = outer_dir / "full_outer_train"
        labels = self._label_context_topics(
            full_row,
            full_context_dir,
            selection_policy=policy,
        )
        candidates = [
            {
                **candidate,
                "source_topic_general_topic": ((label.get("response") or {}).get("general_topic")),
                "source_topic_quality": ((label.get("response") or {}).get("topic_quality")),
            }
            for label in labels
            for candidate in label.get("candidates", [])
        ]
        candidate_pool, dropped = self._harmonize_context_candidates(
            row=full_row,
            context_dir=full_context_dir,
            candidates=candidates,
        )
        candidate_pool, recovery_audit = self._recover_from_inner_policy(candidate_pool, policy)
        dropped.extend(recovery_audit.get("merge_drops") or [])
        _write_json(full_context_dir / "fixed_inner_recovery_policy.json", recovery_audit)
        candidate_pool = self._annotate_registry_with_inner_policy(candidate_pool, policy)
        recurrence_minimum = min(
            max(2, int(self.nn_config.candidate_consistency_min_folds)),
            max(2, int(policy.get("inner_fold_count") or 2)),
        )
        fixed_priority_names = [
            entry["name"]
            for entry in candidate_pool
            if int((entry.get("inner_recurrence_evidence") or {}).get("priority_fold_count", 0))
            >= recurrence_minimum
        ]
        registry, deferred_registry, initial_selection = select_initial_topic_evidence_registry(
            candidate_pool,
            full_row["discovery"],
            coverage_target=float(self.nn_config.tfidf_topic.initial_effect_coverage_target),
            fixed_policy_priority_names=fixed_priority_names,
        )
        _write_json(full_context_dir / "initial_review_selection.json", initial_selection)
        train_df = self._extract_scope(full_row["fit_row_ids"], registry)
        registry, value_audit, value_drops = self._revise_value_contracts_from_training(
            row=full_row,
            context_dir=full_context_dir,
            registry=registry,
            fit_df=train_df,
        )
        dropped.extend(value_drops)
        candidate_pool, pool_merge_drops = _merge_executable_registry_entries(
            [*deferred_registry, *registry]
        )
        dropped.extend(pool_merge_drops)
        active_names = {entry["name"] for entry in registry}
        deferred_registry = [entry for entry in candidate_pool if entry["name"] not in active_names]
        if value_audit.get("changed_contracts"):
            train_df = self._extract_scope(full_row["fit_row_ids"], registry)
        registry, parsimony = self._parsimony_review(
            outer_fold=outer_fold,
            inner_fold=None,
            train_df=train_df,
            heldout_df=None,
            discovery_metadata=None,
            base_diagnostic=None,
            registry=registry,
            outer_dir=outer_dir,
            fixed_policy=policy,
        )
        if int(parsimony.get("accepted_replacements", 0)) > 0:
            train_df = self._extract_scope(full_row["fit_row_ids"], registry)
        specs = registry_specs(registry)
        # Outer-test text is first touched by Stage 2 only after the registry and
        # parsimony decision are frozen. No test labels are sent to extraction.
        test_df = self._extract_scope(full_row["heldout_row_ids"], registry)
        evaluation: SplitEvaluation = self.evaluator.evaluate_split(
            train_df=self._without_oracle_columns(train_df),
            test_df=self._without_oracle_columns(test_df),
            specs=specs,
            fold_id=outer_fold,
        )
        predictions = evaluation.predictions.copy()
        predictions["outer_fold"] = outer_fold
        predictions["honest_outer_holdout"] = True
        predictions["estimation_provenance"] = "outer_train_structured_features_only"
        predictions["forest_fit_row_ids"] = [
            list(map(int, full_row["fit_row_ids"])) for _ in range(len(predictions))
        ]
        predictions["selected_feature_names"] = ",".join(spec.name for spec in specs)
        predictions["selected_feature_roles"] = json.dumps(
            {spec.name: spec.roles for spec in specs}, sort_keys=True
        )
        predictions["selected_confounder_names"] = ",".join(
            spec.name for spec in specs if "confounder" in spec.roles
        )
        predictions["selected_effect_modifier_names"] = ",".join(
            spec.name for spec in specs if "effect_modifier" in spec.roles
        )
        stage1_nuisance = pd.read_parquet(
            full_row["discovery"]["artifacts"]["nuisance_predictions"]
        )
        stage1_nuisance = stage1_nuisance[
            stage1_nuisance["prediction_scope"] == "external_heldout"
        ][["_oci_row_id", "treatment_stacked", "outcome_stacked", "fit_row_ids"]].rename(
            columns={
                "treatment_stacked": "stage1_bow_propensity_prediction",
                "outcome_stacked": "stage1_bow_outcome_prediction",
                "fit_row_ids": "stage1_nuisance_fit_row_ids",
            }
        )
        predictions = predictions.merge(
            stage1_nuisance,
            on="_oci_row_id",
            how="left",
            validate="one_to_one",
        )
        registry_payload = {
            "schema_version": CANONICAL_REGISTRY_SCHEMA_VERSION,
            "outer_fold": outer_fold,
            "policy": policy,
            "registry": registry,
            "initial_review_selection": initial_selection,
            "candidate_pool_size": len(candidate_pool),
            "deferred_valid_contracts": [
                {
                    "name": entry["name"],
                    "contract_hash": entry["contract_hash"],
                    "roles": entry.get("roles", []),
                    "reason": "not_selected_by_fixed_inner_policy_and_training_evidence_cover",
                }
                for entry in deferred_registry
            ],
            "dropped": dropped,
            "training_value_audit": value_audit,
            "fixed_inner_recovery": recovery_audit,
            "parsimony": parsimony,
            "role_mapping": {entry["name"]: entry["roles"] for entry in registry},
        }
        _write_json(outer_dir / "canonical_registry.json", registry_payload)
        predictions.to_parquet(outer_dir / "predictions.parquet", index=False)
        return predictions, {
            "outer_fold": outer_fold,
            "n_selected_features": len(registry),
            "metrics": evaluation.metrics,
            "parsimony": parsimony,
        }

    def run(self) -> None:
        outer_folds = sorted({int(row["outer_fold"]) for row in self.rows})
        predictions: List[pd.DataFrame] = []
        metrics: List[Dict[str, Any]] = []
        for outer_fold in outer_folds:
            prediction, metric = self._run_outer(outer_fold)
            predictions.append(prediction)
            metrics.append(metric)
        combined = pd.concat(predictions, ignore_index=True).sort_values("_oci_row_id")
        if any(str(column).startswith("true_") for column in combined.columns):
            raise RuntimeError("An oracle column entered structured forest predictions")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.output_path, index=False)
        frozen_prediction_sha256 = _sha256_file(self.output_path)
        posthoc_oracle_metrics: Optional[Dict[str, Any]] = None
        if "true_ite_prob" in self.dataset.columns:
            posthoc_oracle_metrics = evaluate_frozen_structured_predictions(
                prediction_path=self.output_path,
                oracle_frame=self.dataset[["_oci_row_id", "true_ite_prob"]],
                output_dir=self.artifact_dir,
                oracle_ite_column="true_ite_prob",
            )
        _write_jsonl(self.artifact_dir / "outer_metrics.jsonl", metrics)
        _write_json(
            self.artifact_dir / "manifest.json",
            {
                "schema_version": "tfidf_topic_agentic_forest_v7",
                "handoff_schema_version": HANDOFF_SCHEMA_VERSION,
                "topic_score_test_schema_version": (TOPIC_SCORE_TEST_SCHEMA_VERSION),
                "n_outer_folds": len(outer_folds),
                "final_ite_source": "structured_causal_forest_only",
                "forbidden_artifacts_present": False,
                "oracle_columns_consumed_by_modeling": False,
                "all_outer_predictions_frozen_before_oracle_join": True,
                "prediction_sha256_before_oracle_join": frozen_prediction_sha256,
                "posthoc_oracle_metrics_path": (
                    str(self.artifact_dir / "posthoc_oracle_metrics.json")
                    if posthoc_oracle_metrics is not None
                    else None
                ),
            },
        )


def _read_unversioned_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_tfidf_topic_agentic_forest(
    *,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    handoff_path: Path,
    proposal_agent: Optional[Any] = None,
    extraction_provider: Optional[Any] = None,
    evaluator: Optional[Any] = None,
    resume: bool = True,
) -> None:
    TfidfTopicAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        handoff_path=handoff_path,
        proposal_agent=proposal_agent,
        extraction_provider=extraction_provider,
        evaluator=evaluator,
        resume=resume,
    ).run()
