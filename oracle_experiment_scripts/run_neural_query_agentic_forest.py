#!/usr/bin/env python
"""Honest one-outer-fold neural-query -> structured causal-forest experiment.

Three independent banks of frozen-embedding queries are learned on outer-training
rows: direct treatment contrast, direct outcome contrast, and an orthogonalized
cohort effect contrast.  No query is gated.  Every final query is shown to an
agent, which creates a bounded structured registry for RAG-style extraction.
Outer-held-out labels and oracle columns are first read after predictions freeze.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
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
    FrozenChunkEmbeddingCache,
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
from oci.inference.tfidf_topic_discovery import (  # noqa: E402
    _strata,
    fit_joint_cross_fitted_nuisance_stacks,
)
from oci.models.causal_forest_head import CausalForestHead  # noqa: E402


LOGGER = logging.getLogger("neural_query_agentic_forest")
BANKS = ("treatment", "outcome", "effect")


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


def _load_outer_context(path: Path, outer_fold: int) -> Dict[str, Any]:
    matches: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
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
    return matches[0]


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
        epochs=(
            int(config.final_refit_epochs)
            if final_refit
            else int(config.query_epochs)
        ),
        max_query_drift=(
            float(config.final_refit_max_query_drift)
            if final_refit
            else float(config.max_query_drift)
        ),
        kmeans_iterations=int(config.kmeans_iterations),
        kmeans_sample_chunks=int(config.kmeans_sample_chunks),
        consensus_min_prototypes=config.query_count(bank),
        consensus_max_prototypes=config.query_count(bank),
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
    config: NeuralQueryAgenticForestConfig,
    seed: int,
    device: str,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    identity_payload = {
        "schema": "neural_query_subfold_v1",
        "fold": int(fold),
        "train_row_ids": [int(row_ids[index]) for index in train_indices],
        "validation_row_ids": [int(row_ids[index]) for index in validation_indices],
        "train_treatment": treatment[train_indices].tolist(),
        "train_outcome": outcome[train_indices].tolist(),
        "train_text_hash": _stable_hash([texts[index] for index in train_indices]),
        "nuisance_folds": int(nuisance_folds),
        "nuisance_views": [str(getattr(view, "name", repr(view))) for view in nuisance_views],
        "query_config": config.to_dict(),
        "seed": int(seed),
    }
    identity = _stable_hash(identity_payload)
    if checkpoint_path.exists():
        cached = joblib.load(checkpoint_path)
        if cached.get("identity") == identity:
            LOGGER.info("fold=%s reusing complete query checkpoint", fold)
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
    nuisance_checkpoint = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_nuisance.joblib"
    )
    nuisance_cached: Dict[str, Any] = {}
    if nuisance_checkpoint.exists():
        candidate_nuisance = joblib.load(nuisance_checkpoint)
        if candidate_nuisance.get("identity") == identity:
            nuisance_cached = candidate_nuisance
            LOGGER.info("fold=%s reusing strict nuisance checkpoint", fold)
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
        )
        validation_e, _ = nuisance["treatment"]["fitted"].predict(validation_texts)
        validation_m, _ = nuisance["outcome"]["fitted"].predict(validation_texts)
        nuisance_cached = {
            "identity": identity,
            "train_e": np.asarray(
                nuisance["treatment"]["stacked_oof"], dtype=float
            ),
            "train_m": np.asarray(nuisance["outcome"]["stacked_oof"], dtype=float),
            "validation_e": np.asarray(validation_e, dtype=float),
            "validation_m": np.asarray(validation_m, dtype=float),
            "metrics": {
                "treatment": nuisance["treatment"]["metrics"],
                "outcome": nuisance["outcome"]["metrics"],
            },
        }
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
                    "candidate_id": (
                        f"{bank}_fold_{fold:02d}_query_{query_index + 1:03d}"
                    ),
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
    config: NeuralQueryAgenticForestConfig,
    nuisance_folds: int,
    devices: Sequence[str],
    seed: int,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    strata = _strata(treatment, outcome, outcome_binary=bool(outcome_binary))
    splitter = StratifiedKFold(
        n_splits=int(config.query_inner_folds),
        shuffle=True,
        random_state=int(seed),
    )
    tasks_by_device: Dict[str, List[Dict[str, Any]]] = {
        str(device): [] for device in devices
    }
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
                "config": config,
                "seed": int(seed + fold),
                "checkpoint_path": checkpoint_dir / f"subfold_{fold:02d}.joblib",
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
            candidate
            for subfold in subfolds
            for candidate in subfold["banks"][bank]["candidates"]
        ]
        candidate_queries = np.vstack([row["query"] for row in candidates])
        candidate_activations = soft_retrieval_activations(
            fit_chunks,
            candidate_queries,
            temperature=float(config.temperature),
            device=str(devices[bank_index % len(devices)]),
        )
        consensus = build_ungated_consensus_query_bank(
            candidates,
            candidate_activations=candidate_activations,
            n_queries=config.query_count(bank),
            bank=bank,
            seed=int(seed + 1000 + bank_index),
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
            record["final_refit_query_drift"] = float(
                refit["query_drift"][query_index]
            )
        final_banks[bank] = {
            "queries": np.asarray(refit["queries"], dtype=np.float32),
            "train_activations": np.asarray(
                refit["train_activations"], dtype=np.float32
            ),
            "records": consensus["records"],
            "consensus": {
                key: value for key, value in consensus.items() if key != "records"
            },
            "objective": refit["objective"],
            "all_queries_retained": True,
            "statistical_gate_applied": False,
        }
    return {
        "banks": final_banks,
        "subfold_audit": [
            {
                key: value
                for key, value in subfold.items()
                if key not in {"banks"}
            }
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
        observed = frame.loc[~missing, value_column] if value_column in frame else pd.Series(dtype=object)
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

    complete_dataset = pd.read_parquet(args.dataset).reset_index(drop=True)
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
    cache = FrozenChunkEmbeddingCache(args.embedding_cache, expected_rows=len(indexed))
    all_chunk_texts = cache.chunk_texts()
    fit_chunks = cache.matrices(fit_ids)
    fit_texts = fit_data[args.text_column].fillna("").astype(str).tolist()
    treatment = fit_data[args.treatment_column].to_numpy(dtype=float)
    outcome = fit_data[args.outcome_column].to_numpy(dtype=float)
    outcome_binary = str(args.outcome_type) == "binary"

    nuisance_frame = pd.read_parquet(discovery_artifacts["nuisance_predictions"])
    fit_nuisance = _ordered_nuisance(nuisance_frame, fit_ids, "fit_oof")
    heldout_nuisance = _ordered_nuisance(
        nuisance_frame, heldout_ids, "external_heldout"
    )
    fit_e = fit_nuisance["treatment_stacked"].to_numpy(dtype=float)
    fit_m = fit_nuisance["outcome_stacked"].to_numpy(dtype=float)
    fitted_context = joblib.load(discovery_artifacts["fitted_context"])
    nuisance_views = list(fitted_context.treatment_stack.views)

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
        max_variables_per_extraction_request=int(
            args.max_variables_per_extraction_request
        ),
    )
    query_config.validate()
    devices = [str(value) for value in args.devices]
    if not devices:
        raise ValueError("at least one query device is required")
    discovery_identity = _stable_hash(
        {
            "schema": "neural_query_discovery_v1",
            "fit_fingerprint": context["fit_row_fingerprint"],
            "heldout_fingerprint_recorded_but_not_consumed": context[
                "heldout_row_fingerprint"
            ],
            "embedding_cache_hash": cache.metadata.get("cache_hash"),
            "stage1_config_hash": context["stage1_config_hash"],
            "query_config": query_config.to_dict(),
            "nuisance_folds": int(args.subfold_nuisance_folds),
            "seed": int(args.seed),
        }
    )
    query_checkpoint = output_dir / "query_discovery.joblib"
    query_discovery: Dict[str, Any]
    if query_checkpoint.exists():
        candidate = joblib.load(query_checkpoint)
        if candidate.get("identity") == discovery_identity:
            LOGGER.info("Reusing complete three-bank query discovery")
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
            config=query_config,
            nuisance_folds=int(args.subfold_nuisance_folds),
            devices=devices,
            seed=int(args.seed),
            checkpoint_dir=output_dir / "query_subfolds",
        )
        query_discovery["identity"] = discovery_identity
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
    _write_json(output_dir / "query_evidence.json", evidence)

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
        raw_candidates.extend(
            query_candidates_from_response(response, feature_context)
        )
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
    registry, dropped_candidates = registry_from_response(
        registry_response, registry_context
    )
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

    all_queries = np.vstack(
        [query_discovery["banks"][bank]["queries"] for bank in BANKS]
    )
    query_ids = [
        record["query_id"]
        for bank in BANKS
        for record in query_discovery["banks"][bank]["records"]
    ]
    query_banks = [
        bank
        for bank in BANKS
        for _ in query_discovery["banks"][bank]["records"]
    ]
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
            max_variables_per_extraction_request=int(
                args.max_variables_per_extraction_request
            ),
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
                query_id
                for row in problematic
                for query_id in row.get("source_query_ids") or []
            )
        )
        relevant_evidence = [
            row for row in evidence if row["query_id"] in set(relevant_query_ids)
        ][:6]
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
        fit_extracted = provider.ensure_features(
            fit_extracted, registry_specs(registry)
        )

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

    # Only now, after the registry freezes, transform/retrieve/extract outer test text.
    heldout_chunks = cache.matrices(heldout_ids)
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
            None
            if structured_w_fit is None
            else np.asarray(structured_w_fit, dtype=np.float32)
        ),
        treatment=treatment,
        outcome=outcome,
        args=args,
        seed=int(args.seed + 5001),
    )
    hybrid_x_fit = np.hstack(
        [np.asarray(structured_x_fit, dtype=np.float32), query_x_fit]
    )
    hybrid_x_test = np.hstack(
        [np.asarray(structured_x_test, dtype=np.float32), query_x_test]
    )
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
            "structured_cate": np.asarray(
                predictions["structured"]["tau_pred"], dtype=float
            ),
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
    heldout_evaluation = indexed.loc[heldout_ids]
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
            bank: int(len(query_discovery["banks"][bank]["queries"]))
            for bank in BANKS
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
            "maximum_variables_per_request": int(
                query_config.max_variables_per_extraction_request
            ),
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
    endpoint_default = (
        "http://camus.dfci.harvard.edu:8010/v1,"
        "http://localhost:2345/v1"
    )
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
