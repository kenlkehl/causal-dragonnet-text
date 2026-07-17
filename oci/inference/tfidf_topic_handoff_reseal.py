"""Non-mutating resealing of legacy exact-context TF-IDF handoffs."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import AppliedInferenceConfig
from .tfidf_topic_discovery import (
    HANDOFF_SCHEMA_VERSION,
    row_set_fingerprint,
    stable_hash,
)
from .tfidf_topic_score_selection import TOPIC_SCORE_TEST_SCHEMA_VERSION
from .tfidf_topic_split_registry import (
    TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
    load_tfidf_topic_split_registry,
    validate_handoff_rows_against_split_registry,
)
from .tfidf_topic_stage1 import tfidf_topic_stage1_identity

RESEALED_HANDOFF_MANIFEST_SCHEMA_VERSION = "tfidf_topic_registry_reseal_v1"
PRE_ORPHAN_TOPIC_SCORE_TEST_SCHEMA_VERSION = "tfidf_topic_and_ngram_score_test_v4"


def derive_tfidf_topic_split_registry_from_handoff(
    *,
    source_handoff_path: Path,
    output_registry_path: Path,
    dataset_row_count: int,
    outer_fold_count: int,
    inner_fold_count: int,
    source_manifest_path: Path | None = None,
) -> Dict[str, Any]:
    """Derive and validate an explicit split registry from an exact-context handoff.

    The handoff is treated as the membership source of record: no fold is
    regenerated from a seed.  Before emitting the registry, this function
    authenticates the handoff's Stage 1 hash against its source manifest and
    validates the complete outer/inner partition, exact row order, and the row
    fingerprints duplicated in each discovery payload.
    """
    source_handoff_path = Path(source_handoff_path).expanduser().resolve()
    output_registry_path = Path(output_registry_path).expanduser().resolve()
    source_manifest_path = (
        Path(source_manifest_path).expanduser().resolve()
        if source_manifest_path is not None
        else (source_handoff_path.parent / "manifest.json").resolve()
    )
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Source Stage 1 manifest not found: {source_manifest_path}")

    rows = _read_handoff(source_handoff_path)
    source_hashes = {str(row.get("stage1_config_hash")) for row in rows}
    if len(source_hashes) != 1 or source_hashes == {"None"}:
        raise RuntimeError("Source handoff contains inconsistent Stage 1 hashes")
    source_stage1_hash = next(iter(source_hashes))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("schema_version") != HANDOFF_SCHEMA_VERSION
        or str(source_manifest.get("stage1_config_hash")) != source_stage1_hash
    ):
        raise RuntimeError("Source manifest does not authenticate the source handoff")

    grouped: Dict[int, list[Dict[str, Any]]] = {}
    for row in rows:
        outer_fold = int(row.get("outer_fold"))
        grouped.setdefault(outer_fold, []).append(row)
    expected_outer_ids = set(range(1, int(outer_fold_count) + 1))
    if set(grouped) != expected_outer_ids:
        raise RuntimeError("Source handoff outer folds do not match the requested registry")

    outer_folds: list[Dict[str, Any]] = []
    for outer_fold in sorted(grouped):
        fold_rows = grouped[outer_fold]
        full_rows = [row for row in fold_rows if row.get("scope") == "full_outer_train"]
        inner_rows = {
            int(row.get("inner_fold")): row
            for row in fold_rows
            if row.get("scope") == "candidate_selection_inner_fit"
        }
        expected_inner_ids = set(range(1, int(inner_fold_count) + 1))
        if (
            len(full_rows) != 1
            or set(inner_rows) != expected_inner_ids
            or len(fold_rows) != 1 + int(inner_fold_count)
        ):
            raise RuntimeError(
                f"Source handoff exact-context set is incomplete for outer_fold={outer_fold}"
            )
        full = full_rows[0]
        outer_folds.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": list(map(int, full.get("fit_row_ids") or [])),
                "heldout_row_ids": list(map(int, full.get("heldout_row_ids") or [])),
                "inner_folds": [
                    {
                        "inner_fold": inner_fold,
                        "fit_row_ids": list(
                            map(int, inner_rows[inner_fold].get("fit_row_ids") or [])
                        ),
                        "heldout_row_ids": list(
                            map(int, inner_rows[inner_fold].get("heldout_row_ids") or [])
                        ),
                    }
                    for inner_fold in sorted(inner_rows)
                ],
            }
        )

    payload = {
        "schema_version": TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
        "dataset_row_count": int(dataset_row_count),
        "outer_folds": outer_folds,
        "provenance": {
            "derivation": "exact_context_handoff_membership_v1",
            "source_handoff_path": str(source_handoff_path),
            "source_handoff_sha256": _sha256_file(source_handoff_path),
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": _sha256_file(source_manifest_path),
            "source_stage1_config_hash": source_stage1_hash,
            "folds_regenerated_from_seed": False,
        },
    }

    output_registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_registry_path.with_suffix(output_registry_path.suffix + ".tmp")
    _write_json_atomic(temporary_path, payload)
    try:
        validated = load_tfidf_topic_split_registry(
            temporary_path,
            dataset_row_count=int(dataset_row_count),
            outer_fold_count=int(outer_fold_count),
            inner_fold_count=int(inner_fold_count),
        )
        validate_handoff_rows_against_split_registry(rows, validated)
        temporary_path.replace(output_registry_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return load_tfidf_topic_split_registry(
        output_registry_path,
        dataset_row_count=int(dataset_row_count),
        outer_fold_count=int(outer_fold_count),
        inner_fold_count=int(inner_fold_count),
    )


def _legacy_tfidf_topic_stage1_config_hash(
    config: AppliedInferenceConfig,
    *,
    omit_topic_prefixes: Sequence[str] = (),
    topic_score_test_schema: str = TOPIC_SCORE_TEST_SCHEMA_VERSION,
) -> str:
    nn_config = config.architecture.multi_model_forest
    topic = {
        key: value
        for key, value in asdict(nn_config.tfidf_topic).items()
        if not any(key.startswith(prefix) for prefix in omit_topic_prefixes)
    }
    return stable_hash(
        {
            "schema": HANDOFF_SCHEMA_VERSION,
            "topic_score_test_schema": str(topic_score_test_schema),
            "views": [asdict(view) for view in nn_config.bow_views],
            "nuisance_folds": nn_config.nuisance_folds,
            "topic": topic,
            "text_column": config.text_column,
            "treatment_column": config.treatment_column,
            "outcome_column": config.outcome_column,
            "outcome_type": config.outcome_type,
        }
    )


def legacy_tfidf_topic_stage1_config_hash(config: AppliedInferenceConfig) -> str:
    """Return the pre-dataset-fingerprint v2 Stage 1 configuration hash."""
    return _legacy_tfidf_topic_stage1_config_hash(config)


def pre_orphan_tfidf_topic_stage1_config_hash(config: AppliedInferenceConfig) -> str:
    """Return the legacy hash used before orphan n-gram controls existed.

    This compatibility identity is only used to authenticate historical
    handoffs.  A migrated config must separately disable orphan n-gram evidence
    so the resealed identity does not claim those absent artifacts exist.
    """
    return _legacy_tfidf_topic_stage1_config_hash(
        config,
        omit_topic_prefixes=("orphan_ngram_",),
        topic_score_test_schema=PRE_ORPHAN_TOPIC_SCORE_TEST_SCHEMA_VERSION,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_handoff(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != HANDOFF_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported source handoff schema on line {line_number}: "
                    f"{row.get('schema_version')!r}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Source handoff is empty: {path}")
    return rows


def _resolve_artifact(value: Any, source_handoff_path: Path, *, label: str) -> Path:
    requested = Path(str(value or "")).expanduser()
    candidates = [requested, source_handoff_path.parent / requested]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"Missing {label} artifact referenced by source handoff: {value!r}")


def _ids(value: Any) -> list[int]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise RuntimeError("Nuisance fit_row_ids must be a list-like value")
    return [int(item) for item in value]


def _expected_stack_hashes(config: AppliedInferenceConfig) -> Dict[str, str]:
    nn_config = config.architecture.multi_model_forest
    common = {
        "views": [asdict(view) for view in nn_config.bow_views],
        "folds": int(nn_config.nuisance_folds),
        "joint_label_free_vectorization": True,
        "random_state": int(nn_config.tfidf_topic.random_state) + 101,
    }
    return {
        target: stable_hash({**common, "target": target}) for target in ("treatment", "outcome")
    }


def _verify_and_absolutize_artifacts(
    row: Dict[str, Any],
    *,
    source_handoff_path: Path,
    config: AppliedInferenceConfig,
) -> None:
    """Verify artifact split alignment and make copied references relocatable."""
    discovery = row["discovery"]
    fit_ids = list(map(int, row["fit_row_ids"]))
    heldout_ids = list(map(int, row["heldout_row_ids"]))
    expected_stack_hashes = _expected_stack_hashes(config)
    for target, expected_hash in expected_stack_hashes.items():
        actual_hash = ((discovery.get("nuisance") or {}).get(target) or {}).get("stack_config_hash")
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{target} nuisance configuration mismatch in source "
                f"fold_key={row.get('fold_key')}"
            )

    artifacts = dict(discovery.get("artifacts") or {})
    fitted = _resolve_artifact(
        artifacts.get("fitted_context"),
        source_handoff_path,
        label="fitted context",
    )
    fit_topics = _resolve_artifact(
        artifacts.get("fit_topic_values"),
        source_handoff_path,
        label="fit topic values",
    )
    heldout_topics = _resolve_artifact(
        artifacts.get("heldout_topic_values"),
        source_handoff_path,
        label="held-out topic values",
    )
    nuisance_path = _resolve_artifact(
        artifacts.get("nuisance_predictions"),
        source_handoff_path,
        label="nuisance predictions",
    )
    ngram_paths = dict(artifacts.get("ngram_scores") or {})
    if set(ngram_paths) != {"treatment", "outcome", "effect"}:
        raise RuntimeError(f"Incomplete n-gram artifact bank in fold_key={row.get('fold_key')}")
    resolved_ngrams = {
        bank: str(
            _resolve_artifact(
                ngram_paths[bank],
                source_handoff_path,
                label=f"{bank} n-gram scores",
            )
        )
        for bank in ("treatment", "outcome", "effect")
    }

    topic_banks = discovery.get("topic_banks") or {}
    with np.load(fit_topics) as fit_archive, np.load(heldout_topics) as heldout_archive:
        for bank in ("treatment", "outcome", "effect"):
            topics = list((topic_banks.get(bank) or {}).get("topics") or [])
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
                    f"Fit topic matrix split mismatch for {bank} in "
                    f"fold_key={row.get('fold_key')}"
                )
            if heldout_values.shape != (len(heldout_ids), len(topics)):
                raise RuntimeError(
                    f"Held-out topic matrix split mismatch for {bank} in "
                    f"fold_key={row.get('fold_key')}"
                )

    nuisance = pd.read_parquet(nuisance_path)
    required = {"_oci_row_id", "prediction_scope", "fit_row_ids"}
    if not required <= set(nuisance.columns):
        raise RuntimeError(
            f"Nuisance split provenance is incomplete in fold_key={row.get('fold_key')}"
        )
    fit_rows = nuisance[nuisance["prediction_scope"] == "fit_oof"]
    external_rows = nuisance[nuisance["prediction_scope"] == "external_heldout"]
    if set(map(int, fit_rows["_oci_row_id"])) != set(fit_ids):
        raise RuntimeError(
            f"Nuisance OOF rows do not match fit rows in fold_key={row.get('fold_key')}"
        )
    if set(map(int, external_rows["_oci_row_id"])) != set(heldout_ids):
        raise RuntimeError(
            f"Nuisance external rows do not match held-out rows in "
            f"fold_key={row.get('fold_key')}"
        )
    fit_set = set(fit_ids)
    for record in fit_rows[["_oci_row_id", "fit_row_ids"]].to_dict(orient="records"):
        training_ids = set(_ids(record["fit_row_ids"]))
        if int(record["_oci_row_id"]) in training_ids or not training_ids <= fit_set:
            raise RuntimeError(
                f"Invalid OOF nuisance fit provenance in fold_key={row.get('fold_key')}"
            )
    for record in external_rows[["_oci_row_id", "fit_row_ids"]].to_dict(orient="records"):
        if set(_ids(record["fit_row_ids"])) != fit_set:
            raise RuntimeError(
                f"Invalid external nuisance fit provenance in fold_key={row.get('fold_key')}"
            )

    score_value = artifacts.get("topic_score_tests")
    if row.get("scope") == "candidate_selection_inner_fit":
        score_path = _resolve_artifact(
            score_value,
            source_handoff_path,
            label="inner topic score tests",
        )
        score = json.loads(score_path.read_text(encoding="utf-8"))
        source_score_schema = score.get("schema_version")
        compatible_score_schemas = {TOPIC_SCORE_TEST_SCHEMA_VERSION}
        if not bool(config.architecture.multi_model_forest.tfidf_topic.orphan_ngram_enabled):
            compatible_score_schemas.add(PRE_ORPHAN_TOPIC_SCORE_TEST_SCHEMA_VERSION)
        if (
            source_score_schema not in compatible_score_schemas
            or score.get("status") != "completed"
            or not bool(score.get("uses_heldout_treatment_and_outcome"))
            or int(score.get("fit_n", -1)) != len(fit_ids)
            or int(score.get("heldout_n", -1)) != len(heldout_ids)
        ):
            raise RuntimeError(f"Invalid inner score-test scope in fold_key={row.get('fold_key')}")
        resolved_score: str | None = str(score_path)
    elif row.get("scope") == "full_outer_train":
        compact = discovery.get("topic_score_tests") or {}
        if (
            score_value is not None
            or compact.get("status") != "not_run"
            or bool(compact.get("uses_heldout_treatment_and_outcome"))
        ):
            raise RuntimeError(
                f"Outer-held-out labels appear in score artifacts for "
                f"fold_key={row.get('fold_key')}"
            )
        resolved_score = None
    else:
        raise RuntimeError(f"Unknown exact-context scope: {row.get('scope')!r}")

    artifacts.update(
        {
            "fitted_context": str(fitted),
            "fit_topic_values": str(fit_topics),
            "heldout_topic_values": str(heldout_topics),
            "nuisance_predictions": str(nuisance_path),
            "ngram_scores": resolved_ngrams,
            "topic_score_tests": resolved_score,
        }
    )
    discovery["artifacts"] = artifacts


def _migrate_pre_orphan_score_artifact(
    row: Dict[str, Any],
    *,
    output_handoff_path: Path,
    config: AppliedInferenceConfig,
) -> Dict[str, Any] | None:
    """Add the disabled v5 orphan branch without recomputing any statistic."""
    if row.get("scope") != "candidate_selection_inner_fit":
        return None
    discovery = row["discovery"]
    artifacts = discovery["artifacts"]
    source_path = Path(str(artifacts["topic_score_tests"])).resolve()
    score = json.loads(source_path.read_text(encoding="utf-8"))
    source_schema = score.get("schema_version")
    if source_schema == TOPIC_SCORE_TEST_SCHEMA_VERSION:
        return None
    if source_schema != PRE_ORPHAN_TOPIC_SCORE_TEST_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported legacy score-test schema: {source_schema!r}")
    if bool(config.architecture.multi_model_forest.tfidf_topic.orphan_ngram_enabled):
        raise RuntimeError(
            "A pre-orphan score-test artifact can only be migrated with "
            "orphan_ngram_enabled=False"
        )

    migrated = copy.deepcopy(score)
    migrated["schema_version"] = TOPIC_SCORE_TEST_SCHEMA_VERSION
    migrated["effect_orphan_ngram_branch"] = {
        "status": "disabled",
        "uses_heldout_treatment_and_outcome": False,
        "clusters": [],
        "selected_clusters": [],
        "selected_cluster_ids": [],
        "selection_count": 0,
    }
    migrated["schema_migration"] = {
        "from_schema": PRE_ORPHAN_TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "to_schema": TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "source_path": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "orphan_ngram_enabled": False,
        "statistics_recomputed": False,
        "labels_read": False,
    }
    migrated_path = (
        output_handoff_path.parent
        / "migrated_score_tests"
        / f"outer_{int(row['outer_fold']):03d}_inner_{int(row['inner_fold']):03d}.json"
    ).resolve()
    _write_json_atomic(migrated_path, migrated)
    artifacts["topic_score_tests"] = str(migrated_path)
    compact = discovery.get("topic_score_tests") or {}
    compact["schema_version"] = TOPIC_SCORE_TEST_SCHEMA_VERSION
    compact["effect_orphan_ngram_branch"] = {
        "status": "disabled",
        "selection_count": 0,
    }
    discovery["topic_score_tests"] = compact
    return {
        "fold_key": int(row["fold_key"]),
        "source_path": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "output_path": str(migrated_path),
        "output_sha256": _sha256_file(migrated_path),
        "from_schema": PRE_ORPHAN_TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "to_schema": TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "statistics_recomputed": False,
        "labels_read": False,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    temporary.replace(path)


def reseal_tfidf_topic_handoff(
    *,
    source_handoff_path: Path,
    output_handoff_path: Path,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    source_manifest_path: Path | None = None,
    output_manifest_path: Path | None = None,
) -> Dict[str, Any]:
    """Audit and reseal a legacy v2 handoff under an explicit split registry.

    The source handoff, its manifest, and every referenced artifact remain
    untouched.  Artifact references in the emitted copy are absolute so that a
    new destination directory cannot silently change their interpretation.
    """
    source_handoff_path = Path(source_handoff_path).expanduser().resolve()
    output_handoff_path = Path(output_handoff_path).expanduser().resolve()
    source_manifest_path = (
        Path(source_manifest_path).expanduser().resolve()
        if source_manifest_path is not None
        else (source_handoff_path.parent / "manifest.json").resolve()
    )
    output_manifest_path = (
        Path(output_manifest_path).expanduser().resolve()
        if output_manifest_path is not None
        else (output_handoff_path.parent / "manifest.json").resolve()
    )
    if output_handoff_path == source_handoff_path:
        raise ValueError("Resealing requires a different output handoff path")
    if output_manifest_path == source_manifest_path:
        raise ValueError("Resealing requires a different output manifest path")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Source Stage 1 manifest not found: {source_manifest_path}")

    data = dataset.reset_index(drop=True)
    nn_config = config.architecture.multi_model_forest
    registry_path = getattr(nn_config, "split_registry_path", None)
    if not registry_path:
        raise ValueError("multi_model_forest.split_registry_path is required for resealing")
    registry = load_tfidf_topic_split_registry(
        registry_path,
        dataset_row_count=len(data),
        outer_fold_count=int(config.cv_folds),
        inner_fold_count=int(nn_config.candidate_consistency_inner_folds),
    )
    source_rows = _read_handoff(source_handoff_path)
    source_hashes = {str(row.get("stage1_config_hash")) for row in source_rows}
    if len(source_hashes) != 1:
        raise RuntimeError("Source handoff contains inconsistent Stage 1 hashes")
    source_hash = next(iter(source_hashes))
    discovery_config_hashes = {
        str((row.get("discovery") or {}).get("config_hash")) for row in source_rows
    }
    if len(discovery_config_hashes) != 1 or discovery_config_hashes == {"None"}:
        raise RuntimeError("Source handoff contains inconsistent fitted topic configuration hashes")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("schema_version") != HANDOFF_SCHEMA_VERSION
        or str(source_manifest.get("stage1_config_hash")) != source_hash
    ):
        raise RuntimeError("Source manifest does not authenticate the source handoff")

    compatible_hashes = {
        legacy_tfidf_topic_stage1_config_hash(config),
        pre_orphan_tfidf_topic_stage1_config_hash(config),
    }
    config_without_registry = copy.deepcopy(config)
    config_without_registry.architecture.multi_model_forest.split_registry_path = None
    compatible_hashes.add(stable_hash(tfidf_topic_stage1_identity(config_without_registry, data)))
    if source_hash not in compatible_hashes:
        raise RuntimeError(
            "Source Stage 1 hash is incompatible with the current TF-IDF configuration"
        )

    rows = copy.deepcopy(source_rows)
    validate_handoff_rows_against_split_registry(rows, registry)
    for row in rows:
        _verify_and_absolutize_artifacts(
            row,
            source_handoff_path=source_handoff_path,
            config=config,
        )
    score_schema_migrations = [
        migration
        for row in rows
        if (
            migration := _migrate_pre_orphan_score_artifact(
                row,
                output_handoff_path=output_handoff_path,
                config=config,
            )
        )
        is not None
    ]

    identity = tfidf_topic_stage1_identity(config, data)
    stage1_hash = stable_hash(identity)
    dataset_identity = identity["dataset"]
    for row in rows:
        fit_fingerprint = row_set_fingerprint(row["fit_row_ids"])
        heldout_fingerprint = row_set_fingerprint(row["heldout_row_ids"])
        row.update(
            {
                "stage1_config_hash": stage1_hash,
                "fit_row_fingerprint": fit_fingerprint,
                "heldout_row_fingerprint": heldout_fingerprint,
                "dataset_content_fingerprint": dataset_identity["content_fingerprint"],
                "dataset_ordered_row_fingerprint": dataset_identity["ordered_row_fingerprint"],
                "split_semantics_hash": identity["split_semantics_hash"],
                "split_registry_content_hash": registry["content_hash"],
            }
        )
        discovery = row["discovery"]
        discovery.update(
            {
                "stage1_config_hash": stage1_hash,
                "fit_row_fingerprint": fit_fingerprint,
                "heldout_row_fingerprint": heldout_fingerprint,
                "dataset_content_fingerprint": dataset_identity["content_fingerprint"],
                "dataset_ordered_row_fingerprint": dataset_identity["ordered_row_fingerprint"],
                "split_semantics_hash": identity["split_semantics_hash"],
                "split_schema_version": TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
            }
        )
    validate_handoff_rows_against_split_registry(rows, registry)

    manifest = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "reseal_schema_version": RESEALED_HANDOFF_MANIFEST_SCHEMA_VERSION,
        "stage1_config_hash": stage1_hash,
        "dataset_content_fingerprint": dataset_identity["content_fingerprint"],
        "dataset_ordered_row_fingerprint": dataset_identity["ordered_row_fingerprint"],
        "split_semantics_hash": identity["split_semantics_hash"],
        "split_schema_version": TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
        "split_registry_content_hash": registry["content_hash"],
        "path": str(output_handoff_path),
        "n_rows": len(rows),
        "n_outer_folds": int(config.cv_folds),
        "inner_contexts_per_outer": int(nn_config.candidate_consistency_inner_folds),
        "exact_inner_contexts": True,
        "stage1_raw_text_forest_prediction": False,
        "stage2_raw_text_modeling_required": False,
        "inner_topic_group_score_tests": bool(nn_config.tfidf_topic.score_test_enabled),
        "inner_topic_and_ngram_score_test_schema": TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "outer_test_labels_used_for_topic_score_tests": False,
        "feature_discovery_methods": ["bow", "tfidf_topic_contrast"],
        "migration": {
            "source_handoff_path": str(source_handoff_path),
            "source_handoff_sha256": _sha256_file(source_handoff_path),
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": _sha256_file(source_manifest_path),
            "source_stage1_config_hash": source_hash,
            "source_artifacts_mutated": False,
            "artifact_references_absolutized": True,
            "score_test_schema_migrations": score_schema_migrations,
        },
    }
    _write_jsonl_atomic(output_handoff_path, rows)
    _write_json_atomic(output_manifest_path, manifest)
    return manifest
