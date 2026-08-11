"""Legacy TF-IDF topic helpers retained for shared prompt compatibility.

The runnable Stage 2 workflow formerly implemented here was retired with
``MultiModelForestRunner``. Research runs use ``plain_handoff_stage2``.
"""

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
    NuisanceCalibrationScientificConfig,
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
from .tfidf_safe_artifacts import load_named_array_bank
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
                    "the retired TF-IDF-topic Stage 2 accepts only its exact-scope handoffs. "
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
        configured_terms_per_topic = int(
            config.architecture.multi_model_forest.tfidf_topic.terms_per_topic
        )
        fit_archive = load_named_array_bank(
            fit_topics_path,
            expected_row_count=len(fit_ids),
        )
        heldout_archive = load_named_array_bank(
            heldout_topics_path,
            expected_row_count=len(heldout_ids),
        )
        for bank in ("treatment", "outcome", "effect"):
            if bank not in bank_metadata:
                raise RuntimeError(f"Missing {bank} topic bank in fold {row.get('fold_key')}")
            topics = list((bank_metadata.get(bank) or {}).get("topics") or [])
            topic_counts[bank].append(len(topics))
            if (
                int(
                    (bank_metadata.get(bank) or {}).get(
                        "terms_per_topic",
                        configured_terms_per_topic,
                    )
                )
                != configured_terms_per_topic
                or any(
                    int(
                        topic.get(
                            "terms_per_topic",
                            configured_terms_per_topic,
                        )
                    )
                    != configured_terms_per_topic
                    or len(topic.get("terms") or [])
                    != configured_terms_per_topic
                    for topic in topics
                )
            ):
                raise RuntimeError(
                    f"A {bank} topic does not contain exactly the configured "
                    f"{configured_terms_per_topic} terms"
                )
            fit_values = np.asarray(
                fit_archive.get(bank, np.zeros((len(fit_ids), 0)))
            )
            heldout_values = np.asarray(
                heldout_archive.get(bank, np.zeros((len(heldout_ids), 0)))
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
                or int(
                    score_tests.get(
                        "terms_per_topic",
                        configured_terms_per_topic,
                    )
                )
                != configured_terms_per_topic
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
                    len(test.get("term_scores") or [])
                    != configured_terms_per_topic
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
    terms_per_topic: Optional[int] = None,
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
        configured_terms_per_topic: Optional[int] = None
    else:
        if terms_per_topic is None:
            raise ValueError(
                "Every topic-label prompt requires configured terms_per_topic"
            )
        configured_terms_per_topic = int(terms_per_topic)
        if configured_terms_per_topic < 1:
            raise ValueError("terms_per_topic must be positive")
        if (
            int(
                topic.get(
                    "terms_per_topic",
                    configured_terms_per_topic,
                )
            )
            != configured_terms_per_topic
            or len(terms) != configured_terms_per_topic
        ):
            raise ValueError(
                "Every topic-label prompt must receive exactly the configured "
                f"{configured_terms_per_topic} supplied terms"
            )
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
        "terms_per_topic": configured_terms_per_topic,
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
    else:
        configured_terms_per_topic = context.get("terms_per_topic")
        if (
            not isinstance(configured_terms_per_topic, int)
            or isinstance(configured_terms_per_topic, bool)
            or configured_terms_per_topic < 1
            or len(terms) != configured_terms_per_topic
        ):
            raise ValueError(
                "Topic prompt rendering requires the complete configured term set"
            )
    payload = json.dumps(context, indent=2, default=str)
    evidence_description = (
        "These raw text phrases formed a stable, held-out-supported evidence "
        "group that was not represented in the fitted topic summaries. Review "
        f"this one group and its {len(terms)} supplied phrases."
        if is_orphan
        else "Topic modeling was used only to organize candidate text signals. "
        f"Review this one topic and its {len(terms)} highest-loading supplied terms."
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
    calibration_config: NuisanceCalibrationScientificConfig,
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
                    calibration_config=calibration_config,
                )
            },
            "outcome": {
                "stacked_metrics": nuisance_metrics(
                    outcome,
                    predictions["outcome_stacked"].to_numpy(dtype=float),
                    binary=outcome_binary,
                    calibration_config=calibration_config,
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
    calibration_config = (
        config.architecture.multi_model_forest.tfidf_topic
        .nuisance_stack_scientific.calibration
    )
    structured_e = _fit_nuisance_from_structured(w_fit, w_heldout, t_fit, binary=True)
    structured_m = _fit_nuisance_from_structured(w_fit, w_heldout, y_fit, binary=outcome_binary)
    treatment_metrics = nuisance_metrics(
        t_heldout,
        structured_e,
        binary=True,
        calibration_config=calibration_config,
    )
    outcome_metrics = nuisance_metrics(
        y_heldout,
        structured_m,
        binary=outcome_binary,
        calibration_config=calibration_config,
    )
    benchmark = _nuisance_benchmark(
        metadata,
        heldout_df["_oci_row_id"].astype(int).tolist(),
        t_heldout,
        y_heldout,
        outcome_binary=outcome_binary,
        calibration_config=calibration_config,
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
    fit_topics = load_named_array_bank(
        metadata["artifacts"]["fit_topic_values"],
        expected_row_count=len(fit_df),
    )
    heldout_topics = load_named_array_bank(
        metadata["artifacts"]["heldout_topic_values"],
        expected_row_count=len(heldout_df),
    )
    if "effect" in fit_topics and x_fit.shape[1] > 0:
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
