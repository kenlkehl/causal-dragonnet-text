"""Production in-memory runtime for nested neural-query discovery.

This module owns the adaptive context path's three ungated query banks:
direct treatment contrast, direct outcome contrast, and orthogonalized effect
contrast.  The statistical procedure mirrors the standalone neural-query
experiment, but its API deliberately has no checkpoint path, checkpoint-reuse
flag, or executable deserialization dependency.  Every nested nuisance fit and
query fit therefore exists only in memory for the duration of one call.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ..config import TfidfNuisanceStackScientificConfig
from .neural_cohort_witness import (
    NeuralCohortWitnessConfig,
    build_ungated_consensus_query_bank,
    cohort_contribution,
    fit_soft_contrast_queries,
    fit_soft_target_queries,
    soft_retrieval_activations,
    standardized_cohort_moments,
    standardized_direct_target_contrasts,
)
from .neural_query_agentic_forest import NeuralQueryAgenticForestConfig
from .tfidf_topic_discovery import (
    _strata,
    fit_joint_cross_fitted_nuisance_stacks,
)

LOGGER = logging.getLogger(__name__)

NEURAL_QUERY_DISCOVERY_RUNTIME_ID = "neural_query_in_memory_discovery_runtime_v2"
NEURAL_QUERY_DISCOVERY_SUBFOLD_SCHEMA = "neural_query_in_memory_subfold_v2"

BANKS = ("treatment", "outcome", "effect")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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
    parent_input_binding_sha256: str,
) -> dict[str, Any]:
    """Fit one nested subfold entirely in memory.

    Validation labels are used only to create audit scores.  Those scores are
    retained for diagnostics but are never passed to the ungated consensus
    constructor or used to select queries.
    """

    identity_payload = {
        "schema": NEURAL_QUERY_DISCOVERY_SUBFOLD_SCHEMA,
        "runtime": NEURAL_QUERY_DISCOVERY_RUNTIME_ID,
        "fold": int(fold),
        "train_row_ids": [int(row_ids[index]) for index in train_indices],
        "validation_row_ids": [int(row_ids[index]) for index in validation_indices],
        "train_treatment": np.asarray(treatment[train_indices], dtype=float).tolist(),
        "train_outcome": np.asarray(outcome[train_indices], dtype=float).tolist(),
        "train_text_hash": _stable_hash([texts[index] for index in train_indices]),
        "outcome_binary": bool(outcome_binary),
        "parent_input_binding_sha256": str(parent_input_binding_sha256),
        "nuisance_folds": int(nuisance_folds),
        "nuisance_stack_scientific": asdict(nuisance_stack_config),
        "nuisance_views_sha256": _stable_hash(list(nuisance_views)),
        "query_config": config.to_dict(),
        "seed": int(seed),
        "executable_checkpoint_io": False,
    }
    identity = _stable_hash(identity_payload)

    LOGGER.info(
        "fold=%s device=%s fitting strict nuisances on %s rows; audit=%s rows",
        fold,
        device,
        len(train_indices),
        len(validation_indices),
    )
    train_texts = [texts[index] for index in train_indices]
    validation_texts = [texts[index] for index in validation_indices]
    train_t = np.asarray(treatment[train_indices], dtype=float)
    train_y = np.asarray(outcome[train_indices], dtype=float)
    validation_t = np.asarray(treatment[validation_indices], dtype=float)
    validation_y = np.asarray(outcome[validation_indices], dtype=float)

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
    train_e = np.asarray(nuisance["treatment"]["stacked_oof"], dtype=float)
    train_m = np.asarray(nuisance["outcome"]["stacked_oof"], dtype=float)
    validation_e = np.asarray(validation_e, dtype=float)
    validation_m = np.asarray(validation_m, dtype=float)
    nuisance_metrics = {
        "treatment": nuisance["treatment"]["metrics"],
        "outcome": nuisance["outcome"]["metrics"],
    }
    # The fitted sparse/tree stacks are not needed after their external
    # validation predictions freeze; release them before GPU query fitting.
    del nuisance

    train_u, train_v = train_t - train_e, train_y - train_m
    validation_u = validation_t - validation_e
    validation_v = validation_y - validation_m
    train_chunks = [chunks[index] for index in train_indices]
    validation_chunks = [chunks[index] for index in validation_indices]

    fitted: dict[str, dict[str, Any]] = {}
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
                validation_activations,
                validation_t,
                binary=True,
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

        candidates: list[dict[str, Any]] = []
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

    return {
        "identity": identity,
        "identity_payload": identity_payload,
        "fold": int(fold),
        "device": str(device),
        "training_prediction_scope": "strict_subsubfold_oof",
        "validation_prediction_scope": "subfold_train_to_external_validation",
        "validation_audit_does_not_gate_queries": True,
        "nuisance_metrics": nuisance_metrics,
        "banks": fitted,
    }


def _run_device_tasks(
    device: str,
    tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [_fit_subfold(device=device, **dict(task)) for task in tasks]


def fit_in_memory_query_discovery(
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
) -> dict[str, Any]:
    """Fit all three nested query banks without executable checkpoint I/O."""

    device_names = tuple(str(device) for device in devices)
    if not device_names:
        raise ValueError("neural-query discovery requires at least one device")
    if type(nuisance_stack_config) is not TfidfNuisanceStackScientificConfig:
        raise TypeError(
            "neural-query discovery requires an explicit "
            "TfidfNuisanceStackScientificConfig"
        )
    row_ids = tuple(int(row_id) for row_id in fit_ids)
    texts = tuple(fit_texts)
    chunks = tuple(fit_chunks)
    treatment_values = np.asarray(treatment, dtype=float)
    outcome_values = np.asarray(outcome, dtype=float)
    fit_e_values = np.asarray(fit_e, dtype=float)
    fit_m_values = np.asarray(fit_m, dtype=float)
    row_count = len(row_ids)
    if not (
        len(texts)
        == len(chunks)
        == len(treatment_values)
        == len(outcome_values)
        == len(fit_e_values)
        == len(fit_m_values)
        == row_count
    ):
        raise ValueError("neural-query discovery inputs must have identical row counts")

    parent_binding = _stable_hash(
        {
            "scope": "production_in_memory_no_executable_checkpoint_io",
            "runtime": NEURAL_QUERY_DISCOVERY_RUNTIME_ID,
            "row_ids": list(row_ids),
            "texts_sha256": _stable_hash(list(texts)),
            "treatment": treatment_values.tolist(),
            "outcome": outcome_values.tolist(),
            "fit_e": fit_e_values.tolist(),
            "fit_m": fit_m_values.tolist(),
            "outcome_binary": bool(outcome_binary),
            "nuisance_views_sha256": _stable_hash(list(nuisance_views)),
            "nuisance_stack_scientific": asdict(nuisance_stack_config),
            "query_config": config.to_dict(),
        }
    )
    strata = _strata(
        treatment_values,
        outcome_values,
        outcome_binary=bool(outcome_binary),
    )
    splitter = StratifiedKFold(
        n_splits=int(config.query_inner_folds),
        shuffle=True,
        random_state=int(seed),
    )
    tasks_by_device: dict[str, list[dict[str, Any]]] = {device: [] for device in device_names}
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(row_count), strata),
        start=1,
    ):
        device = device_names[(fold - 1) % len(device_names)]
        tasks_by_device[device].append(
            {
                "fold": int(fold),
                "train_indices": np.asarray(train_indices, dtype=int),
                "validation_indices": np.asarray(validation_indices, dtype=int),
                "row_ids": row_ids,
                "chunks": chunks,
                "texts": texts,
                "treatment": treatment_values,
                "outcome": outcome_values,
                "outcome_binary": bool(outcome_binary),
                "nuisance_views": nuisance_views,
                "nuisance_folds": int(nuisance_folds),
                "nuisance_stack_config": nuisance_stack_config,
                "config": config,
                "seed": int(seed + fold),
                "parent_input_binding_sha256": parent_binding,
            }
        )

    subfolds: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(device_names), int(config.query_inner_folds))
    ) as executor:
        futures = [
            executor.submit(_run_device_tasks, device, tasks)
            for device, tasks in tasks_by_device.items()
            if tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            subfolds.extend(future.result())
    subfolds.sort(key=lambda row: int(row["fold"]))

    final_banks: dict[str, dict[str, Any]] = {}
    fit_u = treatment_values - fit_e_values
    fit_v = outcome_values - fit_m_values
    for bank_index, bank in enumerate(BANKS):
        LOGGER.info("Consolidating and refitting full-context %s query bank", bank)
        candidates = [
            candidate for subfold in subfolds for candidate in subfold["banks"][bank]["candidates"]
        ]
        candidate_queries = np.vstack([row["query"] for row in candidates])
        candidate_activations = soft_retrieval_activations(
            chunks,
            candidate_queries,
            temperature=float(config.temperature),
            device=device_names[bank_index % len(device_names)],
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
        device = device_names[bank_index % len(device_names)]
        if bank == "treatment":
            refit = fit_soft_target_queries(
                chunks,
                treatment_values,
                binary=True,
                config=refit_config,
                seed=refit_seed,
                device=device,
                initial_queries=initial_queries,
                target_name="treatment",
            )
        elif bank == "outcome":
            refit = fit_soft_target_queries(
                chunks,
                outcome_values,
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
                chunks,
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
            "train_activations": np.asarray(
                refit["train_activations"],
                dtype=np.float32,
            ),
            "records": consensus["records"],
            "consensus": {key: value for key, value in consensus.items() if key != "records"},
            "objective": refit["objective"],
            "all_queries_retained": True,
            "statistical_gate_applied": False,
        }

    return {
        "runtime": NEURAL_QUERY_DISCOVERY_RUNTIME_ID,
        "fit_input_binding_sha256": parent_binding,
        "fit_nuisance_output_binding": {
            "schema_version": "context_fit_neural_query_nuisance_output_binding_v1",
            "fit_row_ids": list(row_ids),
            "fit_e_sha256": _stable_hash(fit_e_values.tolist()),
            "fit_m_sha256": _stable_hash(fit_m_values.tolist()),
            "heldout_labels_accessed": False,
        },
        "banks": final_banks,
        "subfold_audit": [
            {key: value for key, value in subfold.items() if key != "banks"}
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
        "executable_checkpoint_io": False,
    }


__all__ = [
    "BANKS",
    "NEURAL_QUERY_DISCOVERY_RUNTIME_ID",
    "fit_in_memory_query_discovery",
]
