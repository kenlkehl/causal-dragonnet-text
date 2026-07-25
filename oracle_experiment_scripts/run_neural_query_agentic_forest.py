#!/usr/bin/env python
"""Honest one-outer-fold neural-query -> structured causal-forest experiment.

Three independent banks of frozen-embedding queries are learned on outer-training
rows: direct treatment contrast, direct outcome contrast, and an orthogonalized
cohort effect contrast.  No query is gated.  Every final query is shown to an
agent, which creates a bounded structured registry for RAG-style extraction.
Outer-held-out labels are projected but not consumed until evaluation; oracle
columns are first read only after predictions freeze. Query-evidence-only mode
never projects any non-model dataset column.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import logging
import re
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.config import (  # noqa: E402
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ModelArchitectureConfig,
    TfidfNuisanceStackScientificConfig,
)
from oci.inference.agentic_explicit_feature_forest import (  # noqa: E402
    make_explicit_feature_extraction_provider,
    make_feature_search_agent,
)
from oci.inference.applied_explicit_feature_forest import (  # noqa: E402
    _build_features,
)
from oci.inference.neural_cohort_witness import (  # noqa: E402
    NeuralCohortWitnessConfig,
    build_ungated_consensus_query_bank,
    cohort_contribution,
    fit_soft_contrast_queries,
    fit_soft_target_queries,
    soft_retrieval_activations,
    standardized_cohort_moments,
    standardized_direct_target_contrasts,
)
from oci.inference.neural_query_agentic_forest import (  # noqa: E402
    QUERY_REVIEW_PROMPT_VERSION,
    NeuralQueryAgenticForestConfig,
    apply_review_candidates_to_registry,
    build_query_evidence,
    build_query_feature_context,
    build_query_rag_documents,
    build_query_registry_context,
    extraction_request_groups,
    query_candidates_from_response,
    registry_from_response,
    registry_specs,
    review_candidates_from_response,
)
from oci.inference.review_spent_evidence_provider import (  # noqa: E402
    SpentOnlyFrozenChunkEmbeddingCache,
)
from oci.inference.query_moment_evidence_adapter import (  # noqa: E402
    NEURAL_QUERY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
)
from oci.inference.all_evidence_fusion import FoldEvidenceProvenance  # noqa: E402
from oci.inference.neural_query_signal_artifact import (  # noqa: E402
    WrittenFoldHonestQuerySignalArtifact,
    build_fold_honest_query_signals,
    write_fold_honest_query_signal_artifact,
)
from oci.inference.neural_query_signal_fusion_adapter import (  # noqa: E402
    load_authenticated_neural_query_feature_banks,
)
from oci.inference.tfidf_topic_discovery import (  # noqa: E402
    _strata,
    fit_joint_cross_fitted_nuisance_stacks,
    legacy_tfidf_nuisance_stack_v1,
    row_set_fingerprint,
)
from oci.models.causal_forest_head import CausalForestHead  # noqa: E402

LOGGER = logging.getLogger("neural_query_agentic_forest")
BANKS = ("treatment", "outcome", "effect")
AUTHENTICATED_QUERY_SIGNAL_SUBDIRECTORY = "authenticated_query_signals"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=_json_default, allow_nan=True),
        encoding="utf-8",
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_exact_bytes(path: Path | str, *, label: str) -> tuple[bytes, str]:
    """Return one immutable file snapshot and its own SHA-256 attribution."""

    source = Path(path)
    try:
        snapshot = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {source}") from exc
    return snapshot, hashlib.sha256(snapshot).hexdigest()


def _load_joblib_exact_bytes(path: Path | str) -> tuple[Any, str]:
    """Deserialize exactly the checkpoint bytes attributed by the returned digest.

    Standalone query-evidence runs may reuse executable checkpoints.  Reading a
    path once for a checksum and asking ``joblib`` to reopen it leaves a file
    replacement window in which the object used for evidence differs from the
    bytes named by the audit.  Snapshot once, hash that immutable value, and
    deserialize only from the same in-memory stream.
    """

    checkpoint_path = Path(path)
    snapshot, digest = _read_exact_bytes(checkpoint_path, label="query checkpoint")
    try:
        loaded = joblib.load(io.BytesIO(snapshot))
    except Exception as exc:
        raise ValueError(f"query checkpoint is malformed: {checkpoint_path}") from exc
    return loaded, digest


def _authenticated_query_signal_directory(output_dir: Path) -> Path:
    return output_dir / AUTHENTICATED_QUERY_SIGNAL_SUBDIRECTORY


def _write_authenticated_query_signals_from_checkpoints(
    output_dir: Path,
    *,
    outer_fold: int,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    fit_chunk_matrices: Sequence[np.ndarray],
    heldout_chunk_matrices: Sequence[np.ndarray],
    query_discovery_checkpoint_path: Path,
    subfold_checkpoint_paths: Sequence[Path],
    temperature: float,
    patient_batch_size: int,
    devices_by_bank: Mapping[str, str],
    expected_parent_input_binding_sha256: str,
    expected_query_discovery_identity: str,
) -> WrittenFoldHonestQuerySignalArtifact:
    """Seal signal banks from persisted discovery state, without refitting queries."""

    bundle = build_fold_honest_query_signals(
        outer_fold=outer_fold,
        fit_row_ids=fit_row_ids,
        heldout_row_ids=heldout_row_ids,
        fit_chunk_matrices=fit_chunk_matrices,
        heldout_chunk_matrices=heldout_chunk_matrices,
        query_discovery_checkpoint_path=query_discovery_checkpoint_path,
        subfold_checkpoint_paths=subfold_checkpoint_paths,
        temperature=temperature,
        patient_batch_size=patient_batch_size,
        devices_by_bank=devices_by_bank,
        expected_parent_input_binding_sha256=expected_parent_input_binding_sha256,
        expected_query_discovery_identity=expected_query_discovery_identity,
    )
    return write_fold_honest_query_signal_artifact(output_dir, bundle=bundle)


def _authenticated_query_signal_summary_fields(
    artifact: WrittenFoldHonestQuerySignalArtifact,
) -> Dict[str, Any]:
    """Return the trusted identities that a downstream loader must receive."""

    return {
        "query_signal_artifact_directory": str(artifact.manifest_path.parent.resolve()),
        "query_signal_artifact_path": str(artifact.signal_parquet_path.resolve()),
        "query_signal_artifact_sha256": artifact.signal_parquet_sha256,
        "query_signal_manifest_path": str(artifact.manifest_path.resolve()),
        "query_signal_manifest_sha256": artifact.manifest_sha256,
        "query_signal_consumer_contract": "role_aware_features_not_tau_predictions",
        "query_effect_features_calibrated_tau": False,
    }


def _reuse_authenticated_query_signal_artifact(
    output_dir: Path,
    *,
    expected_outer_fold: int,
    expected_split_fingerprint: str,
    expected_outer_train_row_ids: Sequence[int],
    expected_outer_heldout_row_ids: Sequence[int],
    expected_parent_input_binding_sha256: str,
    expected_query_discovery_identity: str,
) -> WrittenFoldHonestQuerySignalArtifact | None:
    """Reuse only an artifact authenticated by the prior frozen stage summary."""

    artifact_dir = _authenticated_query_signal_directory(output_dir)
    manifest_path = artifact_dir / "query_signal_manifest.json"
    signal_path = artifact_dir / "query_signals.parquet"
    if not manifest_path.exists():
        if signal_path.exists():
            raise FileExistsError(
                "Authenticated query signal parquet exists without its immutable manifest: "
                f"{signal_path}"
            )
        return None
    summary_path = output_dir / "query_evidence_stage_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            "Authenticated query signal manifest exists without a trusted stage summary"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("query evidence stage summary is not valid JSON") from exc
    if not isinstance(summary, Mapping):
        raise TypeError("query evidence stage summary must be an object")
    if (
        summary.get("query_signal_consumer_contract") != "role_aware_features_not_tau_predictions"
        or summary.get("query_effect_features_calibrated_tau") is not False
    ):
        raise ValueError("stage summary does not declare the role-aware query feature contract")
    recorded_manifest_path = Path(str(summary.get("query_signal_manifest_path") or "")).resolve()
    recorded_signal_path = Path(str(summary.get("query_signal_artifact_path") or "")).resolve()
    if (
        recorded_manifest_path != manifest_path.resolve()
        or recorded_signal_path != signal_path.resolve()
    ):
        raise ValueError(
            "Existing authenticated query artifact is not the artifact frozen in the "
            "stage summary"
        )
    expected_manifest_sha256 = str(summary.get("query_signal_manifest_sha256") or "").strip()
    loaded = load_authenticated_neural_query_feature_banks(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_outer_fold=expected_outer_fold,
        expected_split_fingerprint=expected_split_fingerprint,
        expected_outer_train_row_ids=expected_outer_train_row_ids,
        expected_outer_heldout_row_ids=expected_outer_heldout_row_ids,
        expected_parent_input_binding_sha256=expected_parent_input_binding_sha256,
        expected_query_discovery_identity=expected_query_discovery_identity,
    )
    recorded_signal_sha256 = str(summary.get("query_signal_artifact_sha256") or "").strip()
    if recorded_signal_sha256 != loaded.signal_parquet_sha256:
        raise ValueError("stage summary query signal parquet SHA-256 mismatch")
    return WrittenFoldHonestQuerySignalArtifact(
        manifest_path=manifest_path.resolve(),
        manifest_sha256=loaded.manifest_sha256,
        signal_parquet_path=signal_path.resolve(),
        signal_parquet_sha256=loaded.signal_parquet_sha256,
    )


def _load_exact_model_projection(
    path: Path | str,
    *,
    patient_id_column: str,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
) -> pd.DataFrame:
    columns = tuple(
        str(value).strip()
        for value in (
            patient_id_column,
            text_column,
            treatment_column,
            outcome_column,
        )
    )
    if any(not value for value in columns):
        raise ValueError("model projection columns must be non-empty")
    if len(set(columns)) != len(columns):
        raise ValueError("patient, text, treatment, and outcome columns must be distinct")
    snapshot, digest = _read_exact_bytes(Path(path), label="model dataset")
    try:
        frame = pd.read_parquet(io.BytesIO(snapshot), columns=list(columns)).reset_index(drop=True)
    except Exception as exc:
        raise ValueError("model dataset snapshot is not a readable parquet projection") from exc
    frame.attrs["source_snapshot_sha256"] = digest
    return frame


def _fold_scoped_query_evidence_bundle(
    *,
    outer_fold: int,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    query_evidence: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Seal learned query evidence to one complete outer-fold partition."""

    fit_ids = [int(value) for value in fit_row_ids]
    heldout_ids = [int(value) for value in heldout_row_ids]
    if int(outer_fold) < 1:
        raise ValueError("outer_fold must be positive")
    if not fit_ids or not heldout_ids:
        raise ValueError("fold-scoped query evidence requires non-empty row partitions")
    if len(fit_ids) != len(set(fit_ids)) or len(heldout_ids) != len(set(heldout_ids)):
        raise ValueError("fold-scoped query evidence contains duplicate row IDs")
    if set(fit_ids) & set(heldout_ids):
        raise ValueError("fold-scoped query evidence partitions overlap")
    evidence = [dict(value) for value in query_evidence]
    if not evidence:
        raise ValueError("fold-scoped query evidence cannot be empty")
    return {
        "schema_version": NEURAL_QUERY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "source_kind": "neural_query_moments",
        "source_family": "neural_query_moments",
        "outer_fold": int(outer_fold),
        "scope": "outer_train",
        "fit_row_ids": fit_ids,
        "heldout_row_ids": heldout_ids,
        "fit_row_fingerprint": row_set_fingerprint(fit_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
        "query_evidence": evidence,
    }


def _load_outer_context(path: Path, outer_fold: int) -> Dict[str, Any]:
    matches: List[Dict[str, Any]] = []
    snapshot, digest = _read_exact_bytes(path, label="Stage-1 handoff")
    try:
        lines = snapshot.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("Stage-1 handoff snapshot is not UTF-8") from exc
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("scope") == "full_outer_train"
            and int(row.get("outer_fold", -1)) == int(outer_fold)
            and row.get("inner_fold") is None
        ):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"expected one full outer-training context for fold {outer_fold}; "
            f"found {len(matches)}"
        )
    if matches[0].get("schema_version") != "multi_model_forest_handoff_v2":
        raise ValueError("the neural-query path requires a v2 Stage-1 handoff")
    result = matches[0]
    result["_source_snapshot_sha256"] = digest
    return result


def _ordered_nuisance(
    frame: pd.DataFrame,
    row_ids: Sequence[int],
    prediction_scope: str,
) -> pd.DataFrame:
    selected = frame.loc[frame["prediction_scope"] == prediction_scope].copy()
    selected["_oci_row_id"] = selected["_oci_row_id"].astype(int)
    selected = selected.set_index("_oci_row_id", drop=False)
    ordered = selected.loc[[int(value) for value in row_ids]].reset_index(drop=True)
    if len(ordered) != len(row_ids):
        raise ValueError(f"missing {prediction_scope} nuisance rows")
    return ordered


def _witness_config(
    config: NeuralQueryAgenticForestConfig,
    bank: str,
    *,
    final_refit: bool,
) -> NeuralCohortWitnessConfig:
    return NeuralCohortWitnessConfig(
        n_prototypes=config.query_count(bank),
        initial_pool_size=int(config.initial_pool_size),
        temperature=float(config.temperature),
        learning_rate=float(config.learning_rate),
        epochs=(int(config.final_refit_epochs) if final_refit else int(config.query_epochs)),
        max_query_drift=(
            float(config.final_refit_max_query_drift)
            if final_refit
            else float(config.max_query_drift)
        ),
        query_diversity_weight=float(config.query_diversity_weight),
        activation_diversity_weight=float(config.activation_diversity_weight),
        anchor_weight=float(config.anchor_weight),
        min_activation_sd=float(config.min_activation_sd),
        activation_scale_weight=float(config.activation_scale_weight),
        kmeans_iterations=int(config.kmeans_iterations),
        kmeans_sample_chunks=int(config.kmeans_sample_chunks),
        initialization_max_cosine=float(config.initialization_max_cosine),
        consensus_min_prototypes=config.query_count(bank),
        consensus_max_prototypes=config.query_count(bank),
        epsilon=float(config.witness_epsilon),
        optimizer_beta1=float(config.optimizer_beta1),
        optimizer_beta2=float(config.optimizer_beta2),
        optimizer_epsilon=float(config.optimizer_epsilon),
        optimizer_weight_decay=float(config.optimizer_weight_decay),
        optimizer_amsgrad=bool(config.optimizer_amsgrad),
        optimizer_maximize=bool(config.optimizer_maximize),
        optimizer_foreach=bool(config.optimizer_foreach),
        optimizer_capturable=bool(config.optimizer_capturable),
        optimizer_differentiable=bool(config.optimizer_differentiable),
        optimizer_fused=bool(config.optimizer_fused),
        gradient_clip_norm=float(config.gradient_clip_norm),
        consensus_kmeans_init=str(config.consensus_kmeans_init),
        consensus_kmeans_n_init=int(config.consensus_kmeans_n_init),
        consensus_kmeans_max_iter=int(config.consensus_kmeans_max_iter),
        consensus_kmeans_tolerance=float(config.consensus_kmeans_tolerance),
        consensus_kmeans_copy_x=bool(config.consensus_kmeans_copy_x),
        consensus_kmeans_algorithm=str(config.consensus_kmeans_algorithm),
    )


def _fit_subfold(
    *,
    fold: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    row_ids: Sequence[int],
    chunks: Sequence[np.ndarray],
    texts: Sequence[str],
    treatment: np.ndarray,
    outcome: np.ndarray,
    outcome_binary: bool,
    nuisance_views: Sequence[Any],
    nuisance_folds: int,
    nuisance_stack_config: TfidfNuisanceStackScientificConfig,
    config: NeuralQueryAgenticForestConfig,
    seed: int,
    device: str,
    checkpoint_path: Path,
    parent_input_binding_sha256: str | None = None,
    use_executable_checkpoints: bool = True,
) -> Dict[str, Any]:
    parent_binding = str(parent_input_binding_sha256 or "").strip().lower()
    if use_executable_checkpoints and not re.fullmatch(r"[0-9a-f]{64}", parent_binding):
        raise ValueError("executable query subfold reuse requires a complete parent input binding")
    identity_payload = {
        "schema": "neural_query_subfold_v2",
        "fold": int(fold),
        "train_row_ids": [int(row_ids[index]) for index in train_indices],
        "validation_row_ids": [int(row_ids[index]) for index in validation_indices],
        "train_treatment": treatment[train_indices].tolist(),
        "train_outcome": outcome[train_indices].tolist(),
        "train_text_hash": _stable_hash([texts[index] for index in train_indices]),
        "outcome_binary": bool(outcome_binary),
        "parent_input_binding_sha256": parent_binding or None,
        "nuisance_folds": int(nuisance_folds),
        "nuisance_stack_scientific": asdict(nuisance_stack_config),
        "nuisance_views_sha256": _stable_hash(list(nuisance_views)),
        "query_config": config.to_dict(),
        "seed": int(seed),
    }
    identity = _stable_hash(identity_payload)
    if use_executable_checkpoints and checkpoint_path.exists():
        cached, checkpoint_snapshot_sha256 = _load_joblib_exact_bytes(checkpoint_path)
        if not isinstance(cached, Mapping):
            raise ValueError(f"query subfold checkpoint is malformed: {checkpoint_path}")
        if cached.get("identity") == identity:
            LOGGER.info(
                "fold=%s reusing complete query checkpoint sha256=%s",
                fold,
                checkpoint_snapshot_sha256,
            )
            return cached

    LOGGER.info(
        "fold=%s device=%s fitting strict nuisances on %s rows; audit=%s rows",
        fold,
        device,
        len(train_indices),
        len(validation_indices),
    )
    train_texts = [texts[index] for index in train_indices]
    validation_texts = [texts[index] for index in validation_indices]
    train_t = treatment[train_indices]
    train_y = outcome[train_indices]
    validation_t = treatment[validation_indices]
    validation_y = outcome[validation_indices]
    nuisance_checkpoint = checkpoint_path.with_name(f"{checkpoint_path.stem}_nuisance.joblib")
    nuisance_cached: Dict[str, Any] = {}
    if use_executable_checkpoints and nuisance_checkpoint.exists():
        candidate_nuisance, nuisance_snapshot_sha256 = _load_joblib_exact_bytes(nuisance_checkpoint)
        if not isinstance(candidate_nuisance, Mapping):
            raise ValueError(f"query nuisance checkpoint is malformed: {nuisance_checkpoint}")
        if candidate_nuisance.get("identity") == identity:
            nuisance_cached = candidate_nuisance
            LOGGER.info(
                "fold=%s reusing strict nuisance checkpoint sha256=%s",
                fold,
                nuisance_snapshot_sha256,
            )
    if not nuisance_cached:
        nuisance = fit_joint_cross_fitted_nuisance_stacks(
            texts=train_texts,
            treatment=train_t,
            outcome=train_y,
            outcome_binary=bool(outcome_binary),
            strata=_strata(train_t, train_y, outcome_binary=bool(outcome_binary)),
            views=nuisance_views,
            folds=int(nuisance_folds),
            random_state=int(seed + 10_000),
            nuisance_stack_config=nuisance_stack_config,
        )
        validation_e, _ = nuisance["treatment"]["fitted"].predict(validation_texts)
        validation_m, _ = nuisance["outcome"]["fitted"].predict(validation_texts)
        nuisance_cached = {
            "identity": identity,
            "train_e": np.asarray(nuisance["treatment"]["stacked_oof"], dtype=float),
            "train_m": np.asarray(nuisance["outcome"]["stacked_oof"], dtype=float),
            "validation_e": np.asarray(validation_e, dtype=float),
            "validation_m": np.asarray(validation_m, dtype=float),
            "metrics": {
                "treatment": nuisance["treatment"]["metrics"],
                "outcome": nuisance["outcome"]["metrics"],
            },
        }
        if use_executable_checkpoints:
            nuisance_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(nuisance_cached, nuisance_checkpoint)
        # The fitted sparse/tree stacks are not needed after their external
        # validation predictions have frozen; release them before GPU padding.
        del nuisance
    train_e = np.asarray(nuisance_cached["train_e"], dtype=float)
    train_m = np.asarray(nuisance_cached["train_m"], dtype=float)
    validation_e = np.asarray(nuisance_cached["validation_e"], dtype=float)
    validation_m = np.asarray(nuisance_cached["validation_m"], dtype=float)
    train_u, train_v = train_t - train_e, train_y - train_m
    validation_u = validation_t - np.asarray(validation_e, dtype=float)
    validation_v = validation_y - np.asarray(validation_m, dtype=float)
    train_chunks = [chunks[index] for index in train_indices]
    validation_chunks = [chunks[index] for index in validation_indices]

    fitted: Dict[str, Dict[str, Any]] = {}
    for bank_index, bank in enumerate(BANKS):
        LOGGER.info("fold=%s device=%s optimizing %s query bank", fold, device, bank)
        bank_config = _witness_config(config, bank, final_refit=False)
        bank_seed = int(seed + 100 * bank_index)
        if bank == "treatment":
            result = fit_soft_target_queries(
                train_chunks,
                train_t,
                binary=True,
                config=bank_config,
                seed=bank_seed,
                device=device,
                target_name="treatment",
            )
        elif bank == "outcome":
            result = fit_soft_target_queries(
                train_chunks,
                train_y,
                binary=bool(outcome_binary),
                config=bank_config,
                seed=bank_seed,
                device=device,
                target_name="outcome",
            )
        else:
            contribution, constant_effect = cohort_contribution(train_u, train_v)
            result = fit_soft_contrast_queries(
                train_chunks,
                contribution,
                center_weights=np.square(train_u),
                config=bank_config,
                seed=bank_seed,
                device=device,
                objective_name="constant_effect_orthogonalized_cohort_contrast",
            )
            result["constant_effect"] = float(constant_effect)

        validation_activations = soft_retrieval_activations(
            validation_chunks,
            result["queries"],
            temperature=float(config.temperature),
            device=device,
            patient_batch_size=int(config.retrieval_patient_batch_size),
        )
        if bank == "treatment":
            audit = standardized_direct_target_contrasts(
                validation_activations, validation_t, binary=True
            )
        elif bank == "outcome":
            audit = standardized_direct_target_contrasts(
                validation_activations,
                validation_y,
                binary=bool(outcome_binary),
            )
        else:
            audit = standardized_cohort_moments(
                validation_activations,
                validation_u,
                validation_v,
                constant_effect=float(result["constant_effect"]),
            )
        candidates: List[Dict[str, Any]] = []
        for query_index, query in enumerate(result["queries"]):
            candidates.append(
                {
                    "candidate_id": (f"{bank}_fold_{fold:02d}_query_{query_index + 1:03d}"),
                    "bank": bank,
                    "subfold": int(fold),
                    "query": np.asarray(query, dtype=np.float32),
                    "train_standardized_score": float(
                        result["train_standardized_scores"][query_index]
                    ),
                    "validation_audit_standardized_score": float(
                        audit["standardized_scores"][query_index]
                    ),
                    "validation_audit_only_not_used_for_gating": True,
                    "query_drift": float(result["query_drift"][query_index]),
                }
            )
        fitted[bank] = {
            "candidates": candidates,
            "loss_history": result["loss_history"],
            "objective": result["objective"],
        }
        LOGGER.info("fold=%s completed %s query bank", fold, bank)

    output = {
        "identity": identity,
        "identity_payload": identity_payload,
        "fold": int(fold),
        "device": str(device),
        "training_prediction_scope": "strict_subsubfold_oof",
        "validation_prediction_scope": "subfold_train_to_external_validation",
        "validation_audit_does_not_gate_queries": True,
        "nuisance_metrics": {
            "treatment": nuisance_cached["metrics"]["treatment"],
            "outcome": nuisance_cached["metrics"]["outcome"],
        },
        "banks": fitted,
    }
    if use_executable_checkpoints:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(output, checkpoint_path)
    return output


def _run_device_tasks(device: str, tasks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for task in tasks:
        output.append(_fit_subfold(device=device, **task))
    return output


def _fit_query_discovery(
    *,
    fit_ids: Sequence[int],
    fit_chunks: Sequence[np.ndarray],
    fit_texts: Sequence[str],
    treatment: np.ndarray,
    outcome: np.ndarray,
    outcome_binary: bool,
    fit_e: np.ndarray,
    fit_m: np.ndarray,
    nuisance_views: Sequence[Any],
    nuisance_stack_config: TfidfNuisanceStackScientificConfig,
    config: NeuralQueryAgenticForestConfig,
    nuisance_folds: int,
    devices: Sequence[str],
    seed: int,
    checkpoint_dir: Path,
    parent_input_binding_sha256: str | None = None,
    use_executable_checkpoints: bool = True,
) -> Dict[str, Any]:
    """Fit query banks, optionally keeping every nested subfold in memory.

    The standalone artifact runner retains checkpoint reuse by default. Callers
    operating across a security boundary can disable executable checkpoint I/O
    so a concurrently replaced joblib path is never deserialized.
    """

    if not isinstance(use_executable_checkpoints, bool):
        raise TypeError("use_executable_checkpoints must be a boolean")
    parent_binding = str(parent_input_binding_sha256 or "").strip().lower()
    if use_executable_checkpoints and not re.fullmatch(r"[0-9a-f]{64}", parent_binding):
        raise ValueError(
            "standalone executable query reuse requires a complete parent input binding"
        )
    if not parent_binding:
        parent_binding = _stable_hash(
            {
                "scope": "in_memory_no_executable_checkpoint_reuse",
                "row_ids": list(map(int, fit_ids)),
                "texts_sha256": _stable_hash(list(fit_texts)),
                "treatment": np.asarray(treatment, dtype=float).tolist(),
                "outcome": np.asarray(outcome, dtype=float).tolist(),
                "outcome_binary": bool(outcome_binary),
                "nuisance_views_sha256": _stable_hash(list(nuisance_views)),
                "nuisance_stack_scientific": asdict(nuisance_stack_config),
                "query_config": config.to_dict(),
            }
        )
    strata = _strata(treatment, outcome, outcome_binary=bool(outcome_binary))
    splitter = StratifiedKFold(
        n_splits=int(config.query_inner_folds),
        shuffle=True,
        random_state=int(seed),
    )
    tasks_by_device: Dict[str, List[Dict[str, Any]]] = {str(device): [] for device in devices}
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(treatment)), strata), start=1
    ):
        device = str(devices[(fold - 1) % len(devices)])
        tasks_by_device[device].append(
            {
                "fold": int(fold),
                "train_indices": np.asarray(train_indices, dtype=int),
                "validation_indices": np.asarray(validation_indices, dtype=int),
                "row_ids": fit_ids,
                "chunks": fit_chunks,
                "texts": fit_texts,
                "treatment": treatment,
                "outcome": outcome,
                "outcome_binary": bool(outcome_binary),
                "nuisance_views": nuisance_views,
                "nuisance_folds": int(nuisance_folds),
                "nuisance_stack_config": nuisance_stack_config,
                "config": config,
                "seed": int(seed + fold),
                "checkpoint_path": checkpoint_dir / f"subfold_{fold:02d}.joblib",
                "parent_input_binding_sha256": parent_binding,
                "use_executable_checkpoints": use_executable_checkpoints,
            }
        )
    subfolds: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(devices), int(config.query_inner_folds))
    ) as executor:
        futures = [
            executor.submit(_run_device_tasks, device, tasks)
            for device, tasks in tasks_by_device.items()
            if tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            subfolds.extend(future.result())
    subfolds.sort(key=lambda row: int(row["fold"]))

    final_banks: Dict[str, Dict[str, Any]] = {}
    fit_u, fit_v = treatment - fit_e, outcome - fit_m
    for bank_index, bank in enumerate(BANKS):
        LOGGER.info("Consolidating and refitting full outer-train %s query bank", bank)
        candidates = [
            candidate for subfold in subfolds for candidate in subfold["banks"][bank]["candidates"]
        ]
        candidate_queries = np.vstack([row["query"] for row in candidates])
        candidate_activations = soft_retrieval_activations(
            fit_chunks,
            candidate_queries,
            temperature=float(config.temperature),
            device=str(devices[bank_index % len(devices)]),
            patient_batch_size=int(config.retrieval_patient_batch_size),
        )
        consensus_config = _witness_config(
            config,
            bank,
            final_refit=False,
        )
        consensus = build_ungated_consensus_query_bank(
            candidates,
            candidate_activations=candidate_activations,
            n_queries=config.query_count(bank),
            bank=bank,
            seed=int(seed + 1000 + bank_index),
            config=consensus_config,
        )
        initial_queries = np.asarray(consensus.pop("queries"), dtype=np.float32)
        refit_config = _witness_config(config, bank, final_refit=True)
        refit_seed = int(seed + 2000 + bank_index)
        device = str(devices[bank_index % len(devices)])
        if bank == "treatment":
            refit = fit_soft_target_queries(
                fit_chunks,
                treatment,
                binary=True,
                config=refit_config,
                seed=refit_seed,
                device=device,
                initial_queries=initial_queries,
                target_name="treatment",
            )
        elif bank == "outcome":
            refit = fit_soft_target_queries(
                fit_chunks,
                outcome,
                binary=bool(outcome_binary),
                config=refit_config,
                seed=refit_seed,
                device=device,
                initial_queries=initial_queries,
                target_name="outcome",
            )
        else:
            contribution, constant_effect = cohort_contribution(fit_u, fit_v)
            refit = fit_soft_contrast_queries(
                fit_chunks,
                contribution,
                center_weights=np.square(fit_u),
                config=refit_config,
                seed=refit_seed,
                device=device,
                initial_queries=initial_queries,
                objective_name="constant_effect_orthogonalized_cohort_contrast",
            )
            refit["constant_effect"] = float(constant_effect)
        for query_index, record in enumerate(consensus["records"]):
            record["fit_standardized_score"] = float(
                refit["train_standardized_scores"][query_index]
            )
            record["final_refit_query_drift"] = float(refit["query_drift"][query_index])
        final_banks[bank] = {
            "queries": np.asarray(refit["queries"], dtype=np.float32),
            "train_activations": np.asarray(refit["train_activations"], dtype=np.float32),
            "records": consensus["records"],
            "consensus": {key: value for key, value in consensus.items() if key != "records"},
            "objective": refit["objective"],
            "all_queries_retained": True,
            "statistical_gate_applied": False,
        }
    return {
        "banks": final_banks,
        "subfold_audit": [
            {key: value for key, value in subfold.items() if key not in {"banks"}}
            | {
                "bank_candidates": {
                    bank: [
                        {key: value for key, value in candidate.items() if key != "query"}
                        for candidate in subfold["banks"][bank]["candidates"]
                    ]
                    for bank in BANKS
                }
            }
            for subfold in subfolds
        ],
        "all_queries_retained": True,
        "validation_audits_used_for_selection": False,
    }


def _agent_call_cached(
    agent: Any,
    context: Mapping[str, Any],
    *,
    cache_path: Path,
) -> Dict[str, Any]:
    model_identity = str(agent._resolve_agent_model_name())  # audited pool identity
    context_hash = _stable_hash(context)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            payload.get("context_hash") == context_hash
            and payload.get("model_identity") == model_identity
        ):
            LOGGER.info("Reusing agent checkpoint %s", cache_path.name)
            return dict(payload["response"])
    response = agent.propose(dict(context))
    payload = {
        "context_hash": context_hash,
        "model_identity": model_identity,
        "response": response,
        "response_trace": getattr(agent, "last_response_trace", None),
    }
    _write_json(cache_path, payload)
    return dict(response)


def _feature_diagnostics(
    frame: pd.DataFrame,
    registry: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output = []
    for row in registry:
        name = str(row["name"])
        value_column = f"explicit_feat_{name}"
        missing_column = f"{value_column}_missing"
        missing = (
            frame[missing_column].fillna(True).astype(bool)
            if missing_column in frame
            else pd.Series(True, index=frame.index)
        )
        observed = (
            frame.loc[~missing, value_column] if value_column in frame else pd.Series(dtype=object)
        )
        output.append(
            {
                "name": name,
                "missing_fraction": float(np.mean(missing)),
                "observed_unique_values": int(observed.nunique(dropna=True)),
                "problem": bool(np.mean(missing) > 0.50 or observed.nunique(dropna=True) < 2),
                "source_query_ids": list(row.get("source_query_ids") or []),
            }
        )
    return output


def _forest_fit_predict(
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    w_train: np.ndarray | None,
    treatment: np.ndarray,
    outcome: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[CausalForestHead, Dict[str, np.ndarray]]:
    forest = CausalForestHead(
        n_estimators=int(args.cf_n_estimators),
        max_depth=args.cf_max_depth,
        min_samples_leaf=int(args.cf_min_samples_leaf),
        max_features=str(args.cf_max_features),
        honest=True,
        inference=True,
        random_state=int(seed),
        tune_model=False,
    )
    forest.fit(x_train, treatment, outcome, W=w_train)
    return forest, forest.predict(x_test, return_ci=True)


def _loss_metrics(
    constant_loss: np.ndarray,
    model_loss: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    differences = np.asarray(constant_loss - model_loss, dtype=float)
    reduction = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
    z_value = reduction / standard_error if standard_error > 0 else float("nan")
    rng = np.random.default_rng(int(seed))
    bootstrap = np.asarray(
        [
            np.mean(differences[rng.integers(0, len(differences), len(differences))])
            for _ in range(int(repeats))
        ]
    )
    constant_mean = float(np.mean(constant_loss))
    return {
        "constant_r_loss": constant_mean,
        "model_r_loss": float(np.mean(model_loss)),
        "absolute_r_loss_reduction": reduction,
        "relative_r_loss_reduction": (
            float(reduction / constant_mean) if constant_mean > 0 else float("nan")
        ),
        "paired_standard_error": standard_error,
        "one_sided_wald_p": float(norm.sf(z_value)) if np.isfinite(z_value) else None,
        "bootstrap_95_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _load_outer_context(Path(args.stage1_handoff), args.outer_fold)
    fit_ids = [int(value) for value in context["fit_row_ids"]]
    heldout_ids = [int(value) for value in context["heldout_row_ids"]]
    discovery_artifacts = context["discovery"]["artifacts"]

    # Stage-only execution never materializes any non-model dataset column.
    complete_dataset = _load_exact_model_projection(
        args.dataset,
        patient_id_column=args.patient_id_column,
        text_column=args.text_column,
        treatment_column=args.treatment_column,
        outcome_column=args.outcome_column,
    )
    dataset_snapshot_sha256 = str(complete_dataset.attrs["source_snapshot_sha256"])
    complete_dataset["_oci_row_id"] = np.arange(len(complete_dataset), dtype=int)
    indexed = complete_dataset.set_index("_oci_row_id", drop=False)
    # Only outer-training labels enter discovery.  Held-out rows expose text/id only
    # until all feature contracts and fitted forests have frozen predictions.
    fit_data = indexed.loc[
        fit_ids,
        [
            "_oci_row_id",
            args.patient_id_column,
            args.text_column,
            args.treatment_column,
            args.outcome_column,
        ],
    ].reset_index(drop=True)
    heldout_text_data = indexed.loc[
        heldout_ids,
        ["_oci_row_id", args.patient_id_column, args.text_column],
    ].reset_index(drop=True)
    embedding_cache = SpentOnlyFrozenChunkEmbeddingCache(args.embedding_cache)
    all_row_ids = tuple(int(value) for value in indexed.index)
    all_dataset_texts = tuple(
        indexed.loc[list(all_row_ids), args.text_column].fillna("").astype(str)
    )
    embedding_cache_identity = embedding_cache.identity()
    cache = embedding_cache.bind_spent(all_row_ids, all_dataset_texts)
    if embedding_cache.identity() != embedding_cache_identity:
        raise RuntimeError("frozen embedding cache changed while binding dataset text")
    all_chunk_texts = [list(row) for row in cache.chunk_texts(all_row_ids)]
    fit_chunks = list(cache.chunk_matrices(fit_ids))
    fit_texts = fit_data[args.text_column].fillna("").astype(str).tolist()
    treatment = fit_data[args.treatment_column].to_numpy(dtype=float)
    outcome = fit_data[args.outcome_column].to_numpy(dtype=float)
    outcome_binary = str(args.outcome_type) == "binary"

    nuisance_snapshot, nuisance_snapshot_sha256 = _read_exact_bytes(
        discovery_artifacts["nuisance_predictions"],
        label="Stage-1 nuisance predictions",
    )
    try:
        nuisance_frame = pd.read_parquet(io.BytesIO(nuisance_snapshot))
    except Exception as exc:
        raise ValueError("Stage-1 nuisance snapshot is not readable parquet") from exc
    fit_nuisance = _ordered_nuisance(nuisance_frame, fit_ids, "fit_oof")
    heldout_nuisance = _ordered_nuisance(nuisance_frame, heldout_ids, "external_heldout")
    fit_e = fit_nuisance["treatment_stacked"].to_numpy(dtype=float)
    fit_m = fit_nuisance["outcome_stacked"].to_numpy(dtype=float)
    fitted_context, fitted_context_snapshot_sha256 = _load_joblib_exact_bytes(
        discovery_artifacts["fitted_context"]
    )
    LOGGER.info(
        "loaded fitted TF-IDF context from exact checkpoint bytes sha256=%s",
        fitted_context_snapshot_sha256,
    )
    nuisance_views = list(fitted_context.treatment_stack.views)
    nuisance_views_sha256 = _stable_hash(nuisance_views)
    # This standalone historical runner has no typed scientific-spec input.
    # Bind its compatibility profile explicitly and include the complete
    # profile in every discovery/checkpoint identity.
    nuisance_stack_config = legacy_tfidf_nuisance_stack_v1()

    query_config = NeuralQueryAgenticForestConfig(
        treatment_query_count=int(args.treatment_queries),
        outcome_query_count=int(args.outcome_queries),
        effect_query_count=int(args.effect_queries),
        query_inner_folds=int(args.query_inner_folds),
        initial_pool_size=int(args.initial_pool_size),
        query_epochs=int(args.query_epochs),
        final_refit_epochs=int(args.final_refit_epochs),
        learning_rate=float(args.query_learning_rate),
        temperature=float(args.query_temperature),
        max_query_drift=float(args.max_query_drift),
        final_refit_max_query_drift=float(args.final_refit_max_query_drift),
        max_features_per_query=int(args.max_features_per_query),
        max_raw_feature_candidates=(
            int(args.treatment_queries + args.outcome_queries + args.effect_queries)
            * int(args.max_features_per_query)
        ),
        max_canonical_features=int(args.max_canonical_features),
        max_review_rounds=int(args.max_review_rounds),
        max_review_additions_per_round=int(args.max_review_additions_per_round),
        max_variables_per_extraction_request=int(args.max_variables_per_extraction_request),
    )
    query_config.validate()
    devices = [str(value) for value in args.devices]
    if not devices:
        raise ValueError("at least one query device is required")
    parent_input_binding_sha256 = _stable_hash(
        {
            "schema": "neural_query_parent_input_binding_v1",
            "fit_fingerprint": context["fit_row_fingerprint"],
            "heldout_fingerprint_recorded_but_not_consumed": context["heldout_row_fingerprint"],
            "embedding_cache_identity_sha256": _stable_hash(embedding_cache_identity),
            "dataset_snapshot_sha256": dataset_snapshot_sha256,
            "stage1_handoff_snapshot_sha256": context["_source_snapshot_sha256"],
            "nuisance_snapshot_sha256": nuisance_snapshot_sha256,
            "fitted_context_snapshot_sha256": fitted_context_snapshot_sha256,
            "nuisance_views_sha256": nuisance_views_sha256,
            "nuisance_stack_scientific": asdict(nuisance_stack_config),
            "stage1_config_hash": context["stage1_config_hash"],
            "query_config": query_config.to_dict(),
            "nuisance_folds": int(args.subfold_nuisance_folds),
            "outcome_binary": bool(outcome_binary),
            "outcome_type": str(args.outcome_type),
            "seed": int(args.seed),
        }
    )
    discovery_identity = _stable_hash(
        {
            "schema": "neural_query_discovery_v2",
            "parent_input_binding_sha256": parent_input_binding_sha256,
        }
    )
    query_checkpoint = output_dir / "query_discovery.joblib"
    query_discovery: Dict[str, Any]
    if query_checkpoint.exists():
        candidate, query_checkpoint_snapshot_sha256 = _load_joblib_exact_bytes(query_checkpoint)
        if not isinstance(candidate, Mapping):
            raise ValueError(f"query discovery checkpoint is malformed: {query_checkpoint}")
        if (
            candidate.get("identity") == discovery_identity
            and candidate.get("parent_input_binding_sha256") == parent_input_binding_sha256
        ):
            LOGGER.info(
                "Reusing complete three-bank query discovery sha256=%s",
                query_checkpoint_snapshot_sha256,
            )
            query_discovery = candidate
        else:
            query_discovery = {}
    else:
        query_discovery = {}
    if not query_discovery:
        query_discovery = _fit_query_discovery(
            fit_ids=fit_ids,
            fit_chunks=fit_chunks,
            fit_texts=fit_texts,
            treatment=treatment,
            outcome=outcome,
            outcome_binary=outcome_binary,
            fit_e=fit_e,
            fit_m=fit_m,
            nuisance_views=nuisance_views,
            nuisance_stack_config=nuisance_stack_config,
            config=query_config,
            nuisance_folds=int(args.subfold_nuisance_folds),
            devices=devices,
            seed=int(args.seed),
            checkpoint_dir=output_dir / "query_subfolds",
            parent_input_binding_sha256=parent_input_binding_sha256,
        )
        query_discovery["identity"] = discovery_identity
        query_discovery["parent_input_binding_sha256"] = parent_input_binding_sha256
        joblib.dump(query_discovery, query_checkpoint)
    _write_json(output_dir / "query_subfold_audit.json", query_discovery["subfold_audit"])

    evidence: List[Dict[str, Any]] = []
    for bank_index, bank in enumerate(BANKS):
        bank_result = query_discovery["banks"][bank]
        evidence.extend(
            build_query_evidence(
                bank=bank,
                queries=bank_result["queries"],
                query_records=bank_result["records"],
                row_ids=fit_ids,
                chunk_matrices=fit_chunks,
                all_chunk_texts=all_chunk_texts,
                config=query_config,
                device=devices[bank_index % len(devices)],
                seed=int(args.seed + 3000 + bank_index),
            )
        )
    if len(evidence) != sum(query_config.query_count(bank) for bank in BANKS):
        raise RuntimeError("every final query must produce one agent evidence record")
    legacy_evidence_path = output_dir / "query_evidence.json"
    _write_json(legacy_evidence_path, evidence)
    bundle_path = output_dir / "query_evidence.fold_scoped.json"
    _write_json(
        bundle_path,
        _fold_scoped_query_evidence_bundle(
            outer_fold=int(args.outer_fold),
            fit_row_ids=fit_ids,
            heldout_row_ids=heldout_ids,
            query_evidence=evidence,
        ),
    )
    subfold_checkpoint_paths = [
        output_dir / "query_subfolds" / f"subfold_{fold:02d}.joblib"
        for fold in range(1, int(query_config.query_inner_folds) + 1)
    ]
    query_signal_split_fingerprint = FoldEvidenceProvenance(
        outer_fold=int(args.outer_fold),
        train_row_ids=tuple(fit_ids),
        heldout_row_ids=tuple(heldout_ids),
        scope="outer_train",
        artifact_id=f"neural-query-signals-{int(args.outer_fold)}",
    ).split_fingerprint
    heldout_chunks: Sequence[np.ndarray] | None = None
    authenticated_query_signals = _reuse_authenticated_query_signal_artifact(
        output_dir,
        expected_outer_fold=int(args.outer_fold),
        expected_split_fingerprint=query_signal_split_fingerprint,
        expected_outer_train_row_ids=fit_ids,
        expected_outer_heldout_row_ids=heldout_ids,
        expected_parent_input_binding_sha256=parent_input_binding_sha256,
        expected_query_discovery_identity=discovery_identity,
    )
    if authenticated_query_signals is None:
        heldout_chunks = list(cache.chunk_matrices(heldout_ids))
        authenticated_query_signals = _write_authenticated_query_signals_from_checkpoints(
            _authenticated_query_signal_directory(output_dir),
            outer_fold=int(args.outer_fold),
            fit_row_ids=fit_ids,
            heldout_row_ids=heldout_ids,
            fit_chunk_matrices=fit_chunks,
            heldout_chunk_matrices=heldout_chunks,
            query_discovery_checkpoint_path=query_checkpoint,
            subfold_checkpoint_paths=subfold_checkpoint_paths,
            temperature=float(query_config.temperature),
            patient_batch_size=int(query_config.retrieval_patient_batch_size),
            devices_by_bank={
                bank: devices[index % len(devices)] for index, bank in enumerate(BANKS)
            },
            expected_parent_input_binding_sha256=parent_input_binding_sha256,
            expected_query_discovery_identity=discovery_identity,
        )
    else:
        LOGGER.info(
            "Reusing authenticated neural-query feature artifact %s",
            authenticated_query_signals.manifest_path,
        )
    evidence_stage_summary = {
        "schema_version": NEURAL_QUERY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "status": "neural_query_moment_evidence_frozen",
        "outer_fold": int(args.outer_fold),
        "scope": "outer_train",
        "source_kind": "neural_query_moments",
        "source_family": "neural_query_moments",
        "fit_row_count": len(fit_ids),
        "heldout_row_count": len(heldout_ids),
        "fit_row_fingerprint": row_set_fingerprint(fit_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
        "dataset_snapshot_sha256": dataset_snapshot_sha256,
        "stage1_handoff_snapshot_sha256": context["_source_snapshot_sha256"],
        "nuisance_snapshot_sha256": nuisance_snapshot_sha256,
        "fitted_context_snapshot_sha256": fitted_context_snapshot_sha256,
        "nuisance_views_sha256": nuisance_views_sha256,
        "embedding_cache_identity_sha256": _stable_hash(embedding_cache_identity),
        "query_parent_input_binding_sha256": parent_input_binding_sha256,
        "query_discovery_identity": discovery_identity,
        "query_count": len(evidence),
        "query_count_by_bank": {
            bank: sum(str(row.get("bank")) == bank for row in evidence) for bank in BANKS
        },
        "artifact_path": str(bundle_path.resolve()),
        "artifact_sha256": _file_sha256(bundle_path),
        "legacy_bare_artifact_path": str(legacy_evidence_path.resolve()),
        "legacy_bare_artifact_sha256": _file_sha256(legacy_evidence_path),
        **_authenticated_query_signal_summary_fields(authenticated_query_signals),
        "outer_train_query_signals_are_strict_inner_oof": True,
        "outer_heldout_query_signals_use_final_refit_text_only": True,
        "full_refit_train_activations_used": False,
        "remote_agent_called": False,
        "feature_extraction_performed": False,
        "causal_forest_fitted": False,
        "heldout_labels_consumed": False,
        "heldout_label_columns_projected_but_positions_not_selected_for_discovery": True,
    }
    _write_json(output_dir / "query_evidence_stage_summary.json", evidence_stage_summary)
    if bool(args.query_evidence_only):
        LOGGER.info(
            "outer_fold=%s query evidence frozen; stopping before all agent/model stages",
            int(args.outer_fold),
        )
        return evidence_stage_summary
    if heldout_chunks is None:
        heldout_chunks = list(cache.chunk_matrices(heldout_ids))

    agent_config = AgenticFeatureSearchConfig(
        outer_folds=2,
        inner_folds=2,
        max_iterations=1,
        agent_server_url=str(args.agent_server_url),
        agent_model_name=str(args.agent_model_name),
        agent_api_key=str(args.api_key),
        agent_temperature=0.0,
        agent_max_tokens=int(args.agent_max_tokens),
        agent_enable_thinking=False,
        agent_schema_repair_attempts=2,
        agent_request_timeout=float(args.request_timeout),
        agent_request_max_retries=3,
    )
    agent = make_feature_search_agent(agent_config)
    agent_dir = output_dir / "agent_checkpoints"
    raw_candidates: List[Dict[str, Any]] = []
    LOGGER.info("Sending all %s ungated queries for bounded feature interpretation", len(evidence))
    for item in evidence:
        feature_context = build_query_feature_context(item, config=query_config)
        response = _agent_call_cached(
            agent,
            feature_context,
            cache_path=agent_dir / f"{item['query_id']}.json",
        )
        raw_candidates.extend(query_candidates_from_response(response, feature_context))
    if len(raw_candidates) > int(query_config.max_raw_feature_candidates):
        raise RuntimeError("agent exceeded the predeclared raw feature bound")
    if not raw_candidates:
        raise RuntimeError("all fifteen query prompts returned zero executable features")
    registry_context = build_query_registry_context(raw_candidates, config=query_config)
    registry_response = _agent_call_cached(
        agent,
        registry_context,
        cache_path=agent_dir / "canonical_registry.json",
    )
    registry, dropped_candidates = registry_from_response(registry_response, registry_context)
    if not registry:
        raise RuntimeError("the canonical registry is empty")
    LOGGER.info(
        "Initial feature registry: raw=%s canonical=%s",
        len(raw_candidates),
        len(registry),
    )
    _write_json(
        output_dir / "initial_registry.json",
        {
            "registry": registry,
            "dropped_candidates": dropped_candidates,
            "raw_candidate_count": len(raw_candidates),
        },
    )

    all_queries = np.vstack([query_discovery["banks"][bank]["queries"] for bank in BANKS])
    query_ids = [
        record["query_id"] for bank in BANKS for record in query_discovery["banks"][bank]["records"]
    ]
    query_banks = [bank for bank in BANKS for _ in query_discovery["banks"][bank]["records"]]
    fit_rag_documents = build_query_rag_documents(
        row_ids=fit_ids,
        chunk_matrices=fit_chunks,
        all_chunk_texts=all_chunk_texts,
        queries=all_queries,
        query_ids=query_ids,
        query_banks=query_banks,
        config=query_config,
        device=devices[0],
    )
    fit_extract = fit_data[["_oci_row_id", args.patient_id_column]].copy()
    fit_extract[args.text_column] = fit_rag_documents
    # This frame intentionally has no treatment, outcome, or oracle columns.
    extraction_config = AppliedInferenceConfig(
        outcome_type=str(args.outcome_type),
        dataset_path=str(Path(args.dataset).resolve()),
        text_column=str(args.text_column),
        outcome_column=str(args.outcome_column),
        treatment_column=str(args.treatment_column),
        architecture=ModelArchitectureConfig(
            model_type="explicit_feature_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=int(args.cf_n_estimators),
                max_depth=args.cf_max_depth,
                min_samples_leaf=int(args.cf_min_samples_leaf),
                max_features=str(args.cf_max_features),
                honest=True,
                inference=True,
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=[],
            vllm_mode="server",
            vllm_server_url=str(args.extraction_server_url),
            vllm_model_name=str(args.extraction_model_name),
            vllm_api_key=str(args.api_key),
            vllm_reasoning_parser=str(args.extraction_reasoning_parser),
            vllm_enable_thinking=False,
            extraction_batch_size=int(args.extraction_batch_size),
            max_variables_per_extraction_request=int(args.max_variables_per_extraction_request),
            extraction_max_retries=3,
            extraction_request_timeout=float(args.request_timeout),
            extraction_max_tokens=int(args.extraction_max_tokens),
            extraction_max_text_length=int(args.extraction_max_text_length),
            cache_enabled=True,
            cache_dir=str(output_dir / "extraction_cache"),
        ),
    )
    provider = make_explicit_feature_extraction_provider(
        config=extraction_config,
        output_dir=output_dir / "extraction",
    )
    LOGGER.info("Extracting %s initial contracts on outer-training RAG documents", len(registry))
    fit_extracted = provider.ensure_features(fit_extract, registry_specs(registry))
    review_history: List[Dict[str, Any]] = []
    for round_index in range(1, int(query_config.max_review_rounds) + 1):
        diagnostics = _feature_diagnostics(fit_extracted, registry)
        problematic = [row for row in diagnostics if row["problem"]]
        if not problematic:
            break
        relevant_query_ids = list(
            dict.fromkeys(
                query_id for row in problematic for query_id in row.get("source_query_ids") or []
            )
        )
        relevant_evidence = [row for row in evidence if row["query_id"] in set(relevant_query_ids)][
            :6
        ]
        if not relevant_evidence:
            break
        review_context = {
            "prompt_version": QUERY_REVIEW_PROMPT_VERSION,
            "round_index": int(round_index),
            "max_additions": int(query_config.max_review_additions_per_round),
            "current_registry": registry,
            "feature_diagnostics": problematic,
            "query_evidence": relevant_evidence,
            "outer_heldout_labels_or_oracles_available": False,
        }
        review_response = _agent_call_cached(
            agent,
            review_context,
            cache_path=agent_dir / f"review_round_{round_index:02d}.json",
        )
        review_candidates = review_candidates_from_response(
            review_response,
            review_context,
            round_index=round_index,
        )
        revised_registry, decisions = apply_review_candidates_to_registry(
            registry,
            review_candidates,
            maximum=int(query_config.max_canonical_features),
        )
        review_history.append(
            {
                "round": int(round_index),
                "diagnostics": diagnostics,
                "agent_response": review_response,
                "decisions": decisions,
            }
        )
        if _stable_hash(revised_registry) == _stable_hash(registry):
            break
        registry = revised_registry
        fit_extracted = provider.ensure_features(fit_extracted, registry_specs(registry))

    if len(registry) > int(query_config.max_canonical_features):
        raise RuntimeError("final registry exceeded its hard feature cap")
    specs = registry_specs(registry)
    extraction_groups = extraction_request_groups(
        registry, maximum=int(query_config.max_variables_per_extraction_request)
    )
    _write_json(
        output_dir / "final_registry.json",
        {
            "registry": registry,
            "review_history": review_history,
            "training_diagnostics": _feature_diagnostics(fit_extracted, registry),
            "extraction_request_groups": extraction_groups,
            "hard_feature_cap": int(query_config.max_canonical_features),
        },
    )

    # Only now, after the registry freezes, expose outer-test retrieval documents
    # to extraction.  Earlier held-out query activations were text-only, frozen
    # to the signal artifact, and never entered proposal or registry review.
    heldout_rag_documents = build_query_rag_documents(
        row_ids=heldout_ids,
        chunk_matrices=heldout_chunks,
        all_chunk_texts=all_chunk_texts,
        queries=all_queries,
        query_ids=query_ids,
        query_banks=query_banks,
        config=query_config,
        device=devices[0],
    )
    heldout_extract = heldout_text_data[["_oci_row_id", args.patient_id_column]].copy()
    heldout_extract[args.text_column] = heldout_rag_documents
    heldout_extracted = provider.ensure_features(heldout_extract, specs)

    # Build label-free test matrices before any held-out treatment/outcome is read.
    fit_model = fit_extracted.copy()
    fit_model[args.treatment_column] = treatment
    fit_model[args.outcome_column] = outcome
    structured_x_fit, structured_w_fit, x_names, w_names, means, stds = _build_features(
        fit_model, specs
    )
    structured_x_test, structured_w_test, _, _, _, _ = _build_features(
        heldout_extracted, specs, means, stds
    )
    if structured_x_fit is None:
        structured_x_fit = np.zeros((len(fit_model), 1), dtype=np.float32)
        structured_x_test = np.zeros((len(heldout_extracted), 1), dtype=np.float32)
        x_names = ["intercept_effect"]

    train_query_values = np.hstack(
        [query_discovery["banks"][bank]["train_activations"] for bank in BANKS]
    ).astype(np.float32)
    test_query_values = soft_retrieval_activations(
        heldout_chunks,
        all_queries,
        temperature=float(query_config.temperature),
        device=devices[0],
        patient_batch_size=int(query_config.retrieval_patient_batch_size),
    ).astype(np.float32)
    query_means = np.mean(train_query_values, axis=0)
    query_scales = np.std(train_query_values, axis=0)
    query_scales = np.where(query_scales > 1e-7, query_scales, 1.0)
    train_query_values = (train_query_values - query_means) / query_scales
    test_query_values = (test_query_values - query_means) / query_scales
    nuisance_query_count = int(
        query_config.treatment_query_count + query_config.outcome_query_count
    )
    query_w_fit = train_query_values[:, :nuisance_query_count]
    query_x_fit = train_query_values[:, nuisance_query_count:]
    query_x_test = test_query_values[:, nuisance_query_count:]

    forests: Dict[str, CausalForestHead] = {}
    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    forests["query"], predictions["query"] = _forest_fit_predict(
        x_train=query_x_fit,
        x_test=query_x_test,
        w_train=query_w_fit,
        treatment=treatment,
        outcome=outcome,
        args=args,
        seed=int(args.seed + 5000),
    )
    forests["structured"], predictions["structured"] = _forest_fit_predict(
        x_train=np.asarray(structured_x_fit, dtype=np.float32),
        x_test=np.asarray(structured_x_test, dtype=np.float32),
        w_train=(
            None if structured_w_fit is None else np.asarray(structured_w_fit, dtype=np.float32)
        ),
        treatment=treatment,
        outcome=outcome,
        args=args,
        seed=int(args.seed + 5001),
    )
    hybrid_x_fit = np.hstack([np.asarray(structured_x_fit, dtype=np.float32), query_x_fit])
    hybrid_x_test = np.hstack([np.asarray(structured_x_test, dtype=np.float32), query_x_test])
    hybrid_w_fit = (
        query_w_fit
        if structured_w_fit is None
        else np.hstack([np.asarray(structured_w_fit, dtype=np.float32), query_w_fit])
    )
    forests["hybrid"], predictions["hybrid"] = _forest_fit_predict(
        x_train=hybrid_x_fit,
        x_test=hybrid_x_test,
        w_train=hybrid_w_fit,
        treatment=treatment,
        outcome=outcome,
        args=args,
        seed=int(args.seed + 5002),
    )
    for name, forest in forests.items():
        joblib.dump(forest, output_dir / f"causal_forest_{name}.joblib")

    frozen = pd.DataFrame(
        {
            "_oci_row_id": heldout_ids,
            args.patient_id_column: heldout_text_data[args.patient_id_column].values,
            "outer_fold": int(args.outer_fold),
            "query_cate": np.asarray(predictions["query"]["tau_pred"], dtype=float),
            "structured_cate": np.asarray(predictions["structured"]["tau_pred"], dtype=float),
            "hybrid_cate": np.asarray(predictions["hybrid"]["tau_pred"], dtype=float),
            "prediction_fitting_set_excludes_row_labels": True,
            "registry_frozen_before_test_extraction": True,
        }
    )
    for name, result in predictions.items():
        if "tau_lower" in result:
            frozen[f"{name}_cate_lower"] = result["tau_lower"]
            frozen[f"{name}_cate_upper"] = result["tau_upper"]
    prediction_path = output_dir / "frozen_outer_predictions.parquet"
    frozen.to_parquet(prediction_path, index=False)
    frozen_hash = _file_sha256(prediction_path)

    # Evaluation starts after the immutable prediction artifact exists.
    heldout_e = heldout_nuisance["treatment_stacked"].to_numpy(dtype=float)
    heldout_m = heldout_nuisance["outcome_stacked"].to_numpy(dtype=float)
    evaluation_columns = [args.treatment_column, args.outcome_column]
    try:
        import pyarrow.parquet as pq

        dataset_columns = set(pq.ParquetFile(args.dataset).schema.names)
    except Exception:
        dataset_columns = set(evaluation_columns)
    if "true_ite_prob" in dataset_columns:
        evaluation_columns.append("true_ite_prob")
    heldout_evaluation = pd.read_parquet(
        args.dataset,
        columns=evaluation_columns,
    ).iloc[heldout_ids]
    heldout_t = heldout_evaluation[args.treatment_column].to_numpy(dtype=float)
    heldout_y = heldout_evaluation[args.outcome_column].to_numpy(dtype=float)
    fit_u, fit_v = treatment - fit_e, outcome - fit_m
    _, constant_effect = cohort_contribution(fit_u, fit_v)
    heldout_u, heldout_v = heldout_t - heldout_e, heldout_y - heldout_m
    constant_loss = np.square(heldout_v - heldout_u * constant_effect)
    evaluation: Dict[str, Any] = {
        "frozen_prediction_sha256": frozen_hash,
        "constant_effect_from_outer_training": float(constant_effect),
        "models": {},
    }
    true_ite = (
        heldout_evaluation["true_ite_prob"].to_numpy(dtype=float)
        if "true_ite_prob" in heldout_evaluation
        else None
    )
    for name in ("query", "structured", "hybrid"):
        cate = frozen[f"{name}_cate"].to_numpy(dtype=float)
        model_loss = np.square(heldout_v - heldout_u * cate)
        row: Dict[str, Any] = {
            "r_loss": _loss_metrics(
                constant_loss,
                model_loss,
                repeats=int(args.r_loss_bootstrap_repeats),
                seed=int(args.seed + 6000 + len(evaluation["models"])),
            ),
            "predicted_cate_sd": float(np.std(cate)),
        }
        if true_ite is not None:
            pearson = np.corrcoef(cate, true_ite)[0, 1]
            spearman = spearmanr(cate, true_ite).statistic
            row["posthoc_oracle"] = {
                "pearson_ite_correlation": float(pearson),
                "spearman_ite_correlation": float(spearman),
                "mean_absolute_ite_error": float(np.mean(np.abs(cate - true_ite))),
            }
        evaluation["models"][name] = row
    evaluation["oracle_join_occurred_only_after_predictions_froze"] = True
    _write_json(output_dir / "evaluation.json", evaluation)

    bank_artifact = {
        "queries": all_queries,
        "query_ids": np.asarray(query_ids),
        "query_banks": np.asarray(query_banks),
        "fit_means": query_means,
        "fit_scales": query_scales,
    }
    np.savez_compressed(output_dir / "query_bank.npz", **bank_artifact)
    summary = {
        "method": "ungated_three_bank_neural_query_agentic_forest_v1",
        "scope": {
            "outer_fold": int(args.outer_fold),
            "fit_n": len(fit_ids),
            "heldout_n": len(heldout_ids),
            "fit_row_fingerprint": context["fit_row_fingerprint"],
            "heldout_row_fingerprint": context["heldout_row_fingerprint"],
        },
        "query_config": query_config.to_dict(),
        "query_counts": {
            bank: int(len(query_discovery["banks"][bank]["queries"])) for bank in BANKS
        },
        "all_queries_sent_to_agent": True,
        "statistical_query_gate_applied": False,
        "full_cached_baseline_history_used_without_timeline_truncation": True,
        "embedding_cache": {
            "path": str(cache.path),
            "model": cache.metadata.get("sentence_model_name"),
            "cache_hash": cache.metadata.get("cache_hash"),
            "encoder_loaded_by_this_run": False,
        },
        "features": {
            "raw_candidate_count": len(raw_candidates),
            "canonical_feature_count": len(registry),
            "canonical_feature_names": [row["name"] for row in registry],
            "roles": {row["name"]: row["roles"] for row in registry},
            "encoded_x_names": x_names,
            "encoded_w_names": w_names,
            "extraction_group_count": len(extraction_groups),
            "maximum_variables_per_request": int(query_config.max_variables_per_extraction_request),
        },
        "evaluation": evaluation,
        "honesty": {
            "subfold_nuisances_strictly_cross_fitted": True,
            "subfold_validation_scores_audit_only": True,
            "outer_test_labels_unavailable_to_discovery_agent_and_extraction": True,
            "oracle_columns_used_only_posthoc_after_prediction_hash": frozen_hash,
        },
        "artifacts": {
            "query_discovery": str(query_checkpoint),
            "query_evidence": str(output_dir / "query_evidence.json"),
            "initial_registry": str(output_dir / "initial_registry.json"),
            "final_registry": str(output_dir / "final_registry.json"),
            "frozen_predictions": str(prediction_path),
            "evaluation": str(output_dir / "evaluation.json"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage1-handoff", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--outer-fold", type=int, default=1)
    parser.add_argument(
        "--query-evidence-only",
        action="store_true",
        help=(
            "Fit the three learned neural query banks and write a fold-scoped, "
            "SHA-addressable query-evidence bundle, then stop before agent calls, "
            "feature extraction, causal forests, held-out labels, or evaluation."
        ),
    )
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--patient-id-column", default="patient_id")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--outcome-type", choices=["binary", "continuous"], default="binary")
    parser.add_argument("--treatment-queries", type=int, default=5)
    parser.add_argument("--outcome-queries", type=int, default=5)
    parser.add_argument("--effect-queries", type=int, default=5)
    parser.add_argument("--query-inner-folds", type=int, default=5)
    parser.add_argument("--subfold-nuisance-folds", type=int, default=3)
    parser.add_argument("--initial-pool-size", type=int, default=24)
    parser.add_argument("--query-epochs", type=int, default=120)
    parser.add_argument("--final-refit-epochs", type=int, default=80)
    parser.add_argument("--query-learning-rate", type=float, default=0.025)
    parser.add_argument("--query-temperature", type=float, default=0.05)
    parser.add_argument("--max-query-drift", type=float, default=0.35)
    parser.add_argument("--final-refit-max-query-drift", type=float, default=0.20)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--max-features-per-query", type=int, default=3)
    parser.add_argument("--max-canonical-features", type=int, default=20)
    parser.add_argument("--max-review-rounds", type=int, default=2)
    parser.add_argument("--max-review-additions-per-round", type=int, default=4)
    parser.add_argument("--max-variables-per-extraction-request", type=int, default=10)
    endpoint_default = "http://camus.dfci.harvard.edu:8010/v1," "http://localhost:2345/v1"
    parser.add_argument("--agent-server-url", default=endpoint_default)
    parser.add_argument("--agent-model-name", default="auto")
    parser.add_argument("--agent-max-tokens", type=int, default=20000)
    parser.add_argument("--extraction-server-url", default=endpoint_default)
    parser.add_argument("--extraction-model-name", default="auto")
    parser.add_argument("--extraction-reasoning-parser", default="auto")
    parser.add_argument("--extraction-batch-size", type=int, default=16)
    parser.add_argument("--extraction-max-tokens", type=int, default=5000)
    parser.add_argument("--extraction-max-text-length", type=int, default=100000)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--cf-n-estimators", type=int, default=400)
    parser.add_argument("--cf-max-depth", type=int, default=None)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=10)
    parser.add_argument("--cf-max-features", default="sqrt")
    parser.add_argument("--r-loss-bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = run(args)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
