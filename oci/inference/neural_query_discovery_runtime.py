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
import copy
import hashlib
import json
import logging
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Mapping, NamedTuple, Sequence

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
from .neural_query_operational_controls import (
    RoleNeutralNeuralQueryTaskResourcePlan,
)
from .neural_query_task_execution import (
    NeuralQueryAuthenticatedCacheReference,
    execute_bounded_neural_query_tasks,
)
from .tfidf_topic_discovery import (
    _strata,
    fit_joint_cross_fitted_nuisance_stacks,
)

LOGGER = logging.getLogger(__name__)

NEURAL_QUERY_DISCOVERY_RUNTIME_ID = "neural_query_in_memory_discovery_runtime_v2"
NEURAL_QUERY_DISCOVERY_SUBFOLD_SCHEMA = "neural_query_in_memory_subfold_v2"

BANKS = ("treatment", "outcome", "effect")


class _NeuralQueryInnerFoldTask(NamedTuple):
    """Spawn-pickleable arguments for one complete inner-fold fit."""

    arguments: Mapping[str, Any]


class _NeuralQueryFinalBankTask(NamedTuple):
    """Spawn-pickleable arguments for one consensus/final-refit bank."""

    arguments: Mapping[str, Any]


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


def _resolve_task_chunks(
    *,
    row_ids: Sequence[int],
    texts: Sequence[str],
    chunks: Sequence[np.ndarray] | None,
    embedding_task_reference: (
        NeuralQueryAuthenticatedCacheReference | None
    ),
) -> tuple[np.ndarray, ...]:
    """Resolve exact owner chunks locally, reopening mmap state in processes."""

    rows = tuple(map(int, row_ids))
    exact_texts = tuple(texts)
    if embedding_task_reference is not None:
        if chunks is not None:
            raise ValueError(
                "neural-query task cannot receive both cache-backed and "
                "materialized chunks"
            )
        bound = embedding_task_reference.open_bound(
            row_ids=rows,
            texts=exact_texts,
        )
        resolved = tuple(bound.chunk_matrices(rows))
    else:
        if chunks is None:
            raise ValueError(
                "neural-query task requires chunks or an authenticated "
                "mmap cache reference"
            )
        resolved = tuple(chunks)
    if (
        len(resolved) != len(rows)
        or len(exact_texts) != len(rows)
        or any(
            not isinstance(value, np.ndarray)
            or value.ndim != 2
            or value.dtype != np.dtype(np.float32)
            or not np.isfinite(value).all()
            for value in resolved
        )
    ):
        raise ValueError(
            "neural-query task chunk matrices changed shape, dtype, or "
            "row coverage"
        )
    return resolved


def _fit_subfold(
    *,
    fold: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    row_ids: Sequence[int],
    chunks: Sequence[np.ndarray] | None,
    embedding_task_reference: (
        NeuralQueryAuthenticatedCacheReference | None
    ) = None,
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

    resolved_chunks = _resolve_task_chunks(
        row_ids=row_ids,
        texts=texts,
        chunks=chunks,
        embedding_task_reference=embedding_task_reference,
    )
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
        tfidf_workers=1,
        tfidf_parallel_backend="threads",
        owner_cpu_budget=1,
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
    train_chunks = [resolved_chunks[index] for index in train_indices]
    validation_chunks = [
        resolved_chunks[index] for index in validation_indices
    ]

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


def _run_inner_fold_task(
    task: _NeuralQueryInnerFoldTask,
    device: str,
) -> dict[str, Any]:
    if not isinstance(task, _NeuralQueryInnerFoldTask):
        raise TypeError(
            "neural-query inner-fold worker received another task type"
        )
    return _fit_subfold(
        device=str(device),
        **dict(task.arguments),
    )


def _fit_final_bank(
    *,
    bank: str,
    bank_index: int,
    candidates: Sequence[Mapping[str, Any]],
    row_ids: Sequence[int],
    chunks: Sequence[np.ndarray] | None,
    embedding_task_reference: (
        NeuralQueryAuthenticatedCacheReference | None
    ),
    texts: Sequence[str],
    treatment: np.ndarray,
    outcome: np.ndarray,
    fit_e: np.ndarray,
    fit_m: np.ndarray,
    outcome_binary: bool,
    config: NeuralQueryAgenticForestConfig,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Build consensus and final-refit exactly one canonical query bank."""

    if (
        bank not in BANKS
        or int(bank_index) != BANKS.index(bank)
    ):
        raise ValueError(
            "neural-query final-bank task changed canonical bank identity"
        )
    resolved_chunks = _resolve_task_chunks(
        row_ids=row_ids,
        texts=texts,
        chunks=chunks,
        embedding_task_reference=embedding_task_reference,
    )
    treatment_values = np.asarray(treatment, dtype=float)
    outcome_values = np.asarray(outcome, dtype=float)
    fit_e_values = np.asarray(fit_e, dtype=float)
    fit_m_values = np.asarray(fit_m, dtype=float)
    row_count = len(tuple(row_ids))
    if any(
        value.shape != (row_count,) or not np.isfinite(value).all()
        for value in (
            treatment_values,
            outcome_values,
            fit_e_values,
            fit_m_values,
        )
    ):
        raise ValueError(
            "neural-query final-bank vectors changed shape or finiteness"
        )
    canonical_candidates = [
        copy.deepcopy(dict(candidate)) for candidate in candidates
    ]
    if not canonical_candidates:
        raise ValueError(
            "neural-query final-bank task received no candidates"
        )
    LOGGER.info(
        "Consolidating and refitting full-context %s query bank",
        bank,
    )
    candidate_queries = np.vstack(
        [
            np.asarray(row["query"], dtype=np.float32)
            for row in canonical_candidates
        ]
    )
    candidate_activations = soft_retrieval_activations(
        resolved_chunks,
        candidate_queries,
        temperature=float(config.temperature),
        device=str(device),
        patient_batch_size=int(config.retrieval_patient_batch_size),
    )
    consensus_config = _witness_config(
        config,
        bank,
        final_refit=False,
    )
    consensus_seed = int(seed + 1000 + bank_index)
    consensus = build_ungated_consensus_query_bank(
        canonical_candidates,
        candidate_activations=candidate_activations,
        n_queries=config.query_count(bank),
        bank=bank,
        seed=consensus_seed,
        config=consensus_config,
    )
    initial_queries = np.asarray(
        consensus.pop("queries"),
        dtype=np.float32,
    )
    refit_config = _witness_config(config, bank, final_refit=True)
    refit_seed = int(seed + 2000 + bank_index)
    fit_u = treatment_values - fit_e_values
    fit_v = outcome_values - fit_m_values
    if bank == "treatment":
        refit = fit_soft_target_queries(
            resolved_chunks,
            treatment_values,
            binary=True,
            config=refit_config,
            seed=refit_seed,
            device=str(device),
            initial_queries=initial_queries,
            target_name="treatment",
        )
    elif bank == "outcome":
        refit = fit_soft_target_queries(
            resolved_chunks,
            outcome_values,
            binary=bool(outcome_binary),
            config=refit_config,
            seed=refit_seed,
            device=str(device),
            initial_queries=initial_queries,
            target_name="outcome",
        )
    else:
        contribution, constant_effect = cohort_contribution(
            fit_u,
            fit_v,
        )
        refit = fit_soft_contrast_queries(
            resolved_chunks,
            contribution,
            center_weights=np.square(fit_u),
            config=refit_config,
            seed=refit_seed,
            device=str(device),
            initial_queries=initial_queries,
            objective_name=(
                "constant_effect_orthogonalized_cohort_contrast"
            ),
        )
        refit["constant_effect"] = float(constant_effect)

    for query_index, record in enumerate(consensus["records"]):
        record["fit_standardized_score"] = float(
            refit["train_standardized_scores"][query_index]
        )
        record["final_refit_query_drift"] = float(
            refit["query_drift"][query_index]
        )
    return {
        "bank": bank,
        "bank_index": int(bank_index),
        "consensus_seed": consensus_seed,
        "final_refit_seed": refit_seed,
        "result": {
            "queries": np.asarray(
                refit["queries"],
                dtype=np.float32,
            ),
            "train_activations": np.asarray(
                refit["train_activations"],
                dtype=np.float32,
            ),
            "records": consensus["records"],
            "consensus": {
                key: value
                for key, value in consensus.items()
                if key != "records"
            },
            "objective": refit["objective"],
            "all_queries_retained": True,
            "statistical_gate_applied": False,
        },
    }


def _run_final_bank_task(
    task: _NeuralQueryFinalBankTask,
    device: str,
) -> dict[str, Any]:
    if not isinstance(task, _NeuralQueryFinalBankTask):
        raise TypeError(
            "neural-query final-bank worker received another task type"
        )
    return _fit_final_bank(
        device=str(device),
        **dict(task.arguments),
    )


def fit_in_memory_query_discovery(
    *,
    fit_ids: Sequence[int],
    fit_chunks: Sequence[np.ndarray] | None,
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
    task_resource_plan: (
        RoleNeutralNeuralQueryTaskResourcePlan | None
    ) = None,
    embedding_task_reference: (
        NeuralQueryAuthenticatedCacheReference | None
    ) = None,
    execution_attestation_sink: (
        Callable[[Mapping[str, Any]], None] | None
    ) = None,
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
    chunks = None if fit_chunks is None else tuple(fit_chunks)
    treatment_values = np.asarray(treatment, dtype=float)
    outcome_values = np.asarray(outcome, dtype=float)
    fit_e_values = np.asarray(fit_e, dtype=float)
    fit_m_values = np.asarray(fit_m, dtype=float)
    row_count = len(row_ids)
    if not (
        len(texts)
        == len(treatment_values)
        == len(outcome_values)
        == len(fit_e_values)
        == len(fit_m_values)
        == row_count
    ) or (chunks is not None and len(chunks) != row_count):
        raise ValueError("neural-query discovery inputs must have identical row counts")
    if (
        len(row_ids) != len(set(row_ids))
        or any(not isinstance(text, str) for text in texts)
        or any(
            value.shape != (row_count,) or not np.isfinite(value).all()
            for value in (
                treatment_values,
                outcome_values,
                fit_e_values,
                fit_m_values,
            )
        )
    ):
        raise ValueError(
            "neural-query discovery requires unique rows, exact texts, and "
            "finite aligned vectors"
        )
    if task_resource_plan is None:
        if chunks is None:
            raise ValueError(
                "legacy neural-query execution requires materialized chunks"
            )
        if embedding_task_reference is not None:
            raise ValueError(
                "an authenticated task cache reference requires a bounded "
                "resource plan"
            )
        if execution_attestation_sink is not None:
            raise ValueError(
                "neural-query execution attestation requires bounded task "
                "controls"
            )
    else:
        if not isinstance(
            task_resource_plan,
            RoleNeutralNeuralQueryTaskResourcePlan,
        ):
            raise TypeError(
                "neural-query execution requires a typed task resource plan"
            )
        if tuple(task_resource_plan.devices) != device_names:
            raise ValueError(
                "neural-query task resources differ from selected devices"
            )
        if (
            task_resource_plan.fold_parallel_backend == "processes"
            and embedding_task_reference is None
        ):
            raise ValueError(
                "process neural-query tasks require an authenticated mmap "
                "cache reference"
            )
        if embedding_task_reference is not None and (
            tuple(embedding_task_reference.allowed_row_ids) != row_ids
        ):
            raise ValueError(
                "neural-query task cache authority changed owner row order"
            )
        if (
            execution_attestation_sink is not None
            and not callable(execution_attestation_sink)
        ):
            raise TypeError(
                "neural-query execution attestation sink must be callable"
            )

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
    canonical_tasks: list[dict[str, Any]] = []
    expected_splits: dict[
        int,
        tuple[np.ndarray, np.ndarray],
    ] = {}
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(row_count), strata),
        start=1,
    ):
        train = np.asarray(train_indices, dtype=int)
        validation = np.asarray(validation_indices, dtype=int)
        expected_splits[int(fold)] = (train, validation)
        canonical_tasks.append(
            {
                "fold": int(fold),
                "train_indices": train,
                "validation_indices": validation,
                "row_ids": row_ids,
                "chunks": (
                    None
                    if embedding_task_reference is not None
                    else chunks
                ),
                "embedding_task_reference": (
                    embedding_task_reference
                ),
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

    inner_attestation: Mapping[str, Any] | None = None
    if task_resource_plan is None:
        tasks_by_device: dict[str, list[dict[str, Any]]] = {
            device: [] for device in device_names
        }
        for index, task in enumerate(canonical_tasks):
            device = device_names[index % len(device_names)]
            tasks_by_device[device].append(task)
        unordered_subfolds: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(
                len(device_names),
                int(config.query_inner_folds),
            )
        ) as executor:
            futures = [
                executor.submit(_run_device_tasks, device, tasks)
                for device, tasks in tasks_by_device.items()
                if tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                unordered_subfolds.extend(future.result())
        subfolds = sorted(
            unordered_subfolds,
            key=lambda row: int(row["fold"]),
        )
    else:
        values, inner_attestation = execute_bounded_neural_query_tasks(
            tuple(
                _NeuralQueryInnerFoldTask(arguments=task)
                for task in canonical_tasks
            ),
            task_names=tuple(
                f"inner_fold_{fold:03d}"
                for fold in range(
                    1,
                    len(canonical_tasks) + 1,
                )
            ),
            resource_plan=task_resource_plan,
            worker=_run_inner_fold_task,
            parallelism=task_resource_plan.inner_fold_parallelism,
            phase="inner_folds",
        )
        subfolds = list(values)

    if len(subfolds) != len(canonical_tasks):
        raise RuntimeError(
            "neural-query inner-fold execution omitted a canonical fold"
        )
    for expected_fold, subfold in enumerate(subfolds, start=1):
        train, validation = expected_splits[expected_fold]
        identity = subfold.get("identity_payload")
        if (
            int(subfold.get("fold", -1)) != expected_fold
            or not isinstance(identity, Mapping)
            or int(identity.get("fold", -1)) != expected_fold
            or int(identity.get("seed", -1))
            != int(seed + expected_fold)
            or identity.get("train_row_ids")
            != [int(row_ids[index]) for index in train]
            or identity.get("validation_row_ids")
            != [int(row_ids[index]) for index in validation]
            or set(subfold.get("banks") or {}) != set(BANKS)
        ):
            raise RuntimeError(
                "neural-query inner-fold result changed rows, seed, "
                "identity, or banks"
            )

    # No final bank task is even constructed until every inner-fold result
    # has returned and its canonical rows/seeds/banks have been verified.
    inner_fold_barrier_ns = time.monotonic_ns()
    final_tasks: list[_NeuralQueryFinalBankTask] = []
    for bank_index, bank in enumerate(BANKS):
        candidates = [
            candidate
            for subfold in subfolds
            for candidate in subfold["banks"][bank]["candidates"]
        ]
        final_tasks.append(
            _NeuralQueryFinalBankTask(
                arguments={
                    "bank": bank,
                    "bank_index": bank_index,
                    "candidates": candidates,
                    "row_ids": row_ids,
                    "chunks": (
                        None
                        if embedding_task_reference is not None
                        else chunks
                    ),
                    "embedding_task_reference": (
                        embedding_task_reference
                    ),
                    "texts": texts,
                    "treatment": treatment_values,
                    "outcome": outcome_values,
                    "fit_e": fit_e_values,
                    "fit_m": fit_m_values,
                    "outcome_binary": bool(outcome_binary),
                    "config": config,
                    "seed": int(seed),
                }
            )
        )

    bank_attestation: Mapping[str, Any] | None = None
    if task_resource_plan is None:
        final_rows = tuple(
            _run_final_bank_task(
                task,
                device_names[index % len(device_names)],
            )
            for index, task in enumerate(final_tasks)
        )
    else:
        final_rows, bank_attestation = execute_bounded_neural_query_tasks(
            tuple(final_tasks),
            task_names=tuple(
                f"final_{bank}_bank" for bank in BANKS
            ),
            resource_plan=task_resource_plan,
            worker=_run_final_bank_task,
            parallelism=task_resource_plan.bank_parallelism,
            phase="consensus_and_final_refit_banks",
        )
    final_banks: dict[str, dict[str, Any]] = {}
    for bank_index, (bank, row) in enumerate(
        zip(BANKS, final_rows, strict=True)
    ):
        if (
            row.get("bank") != bank
            or int(row.get("bank_index", -1)) != bank_index
            or int(row.get("consensus_seed", -1))
            != int(seed + 1000 + bank_index)
            or int(row.get("final_refit_seed", -1))
            != int(seed + 2000 + bank_index)
            or not isinstance(row.get("result"), Mapping)
        ):
            raise RuntimeError(
                "neural-query final-bank result changed canonical order "
                "or seed formulas"
            )
        final_banks[bank] = dict(row["result"])

    if task_resource_plan is not None:
        if inner_attestation is None or bank_attestation is None:
            raise RuntimeError(
                "neural-query bounded execution omitted phase telemetry"
            )
        inner_finishes = [
            int(row["finished_monotonic_ns"])
            for row in inner_attestation["task_intervals"]
        ]
        bank_starts = [
            int(row["started_monotonic_ns"])
            for row in bank_attestation["task_intervals"]
        ]
        if (
            not inner_finishes
            or not bank_starts
            or max(inner_finishes) >= inner_fold_barrier_ns
            or min(bank_starts) <= inner_fold_barrier_ns
        ):
            raise RuntimeError(
                "neural-query final refits crossed the inner-fold barrier"
            )
        attestation_body = {
            "schema_version": (
                "production_neural_query_discovery_execution_attestation_v1"
            ),
            "resource_plan": task_resource_plan.as_dict(),
            "inner_fold_phase": copy.deepcopy(dict(inner_attestation)),
            "inner_fold_barrier_monotonic_ns": inner_fold_barrier_ns,
            "inner_fold_barrier_enforced": True,
            "all_inner_results_verified_before_final_task_construction": True,
            "final_bank_phase": copy.deepcopy(dict(bank_attestation)),
            "canonical_fold_order": list(
                range(1, len(subfolds) + 1)
            ),
            "canonical_bank_order": list(BANKS),
            "scientific_payload_contains_device_metadata": False,
        }
        attestation = {
            **attestation_body,
            "content_sha256": _stable_hash(attestation_body),
        }
        if execution_attestation_sink is not None:
            execution_attestation_sink(attestation)

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
