#!/usr/bin/env python
"""One-fold neural soft-witness cohort-contrast experiment.

This diagnostic reuses frozen chunk embeddings and exact Stage-1 nuisance
predictions.  Semantic queries are learned only within sub-inner training rows,
validated on their corresponding sub-inner held-out rows, harmonized by
recurrence, and evaluated once on the untouched exact inner-held-out split.
Oracle columns are joined only after predictions and score tests are frozen.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.inference.neural_cohort_witness import (  # noqa: E402
    NeuralCohortWitnessConfig,
    benjamini_hochberg,
    build_consensus_witness_bank,
    fit_constant_residual_effect,
    fit_soft_witness_queries,
    multiplier_group_score_test,
    soft_retrieval_activations,
    standardized_cohort_moments,
)
from oci.inference.tfidf_topic_discovery import (  # noqa: E402
    _strata,
    fit_joint_cross_fitted_nuisance_stacks,
)
from oci.models.causal_forest_head import CausalForestHead  # noqa: E402
from oracle_experiment_scripts.run_topic_ngram_inner_fold_forest import (  # noqa: E402
    _context,
    _ordered_nuisance,
    _paired_loss_metrics,
)


LOGGER = logging.getLogger("neural_cohort_witness")


class FrozenChunkCache:
    """Read-only view of a complete, row-ordered chunk-embedding cache."""

    def __init__(self, path: Path, *, expected_rows: int) -> None:
        self.path = Path(path)
        metadata_path = self.path / "metadata.json"
        embeddings_path = self.path / "chunk_embeddings.npy"
        offsets_path = self.path / "offsets.npy"
        chunks_path = self.path / "chunk_texts.jsonl"
        for required in (metadata_path, embeddings_path, offsets_path, chunks_path):
            if not required.exists():
                raise FileNotFoundError(f"Incomplete embedding cache: missing {required}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        self.offsets = np.load(offsets_path)
        if int(self.metadata.get("num_samples", -1)) != int(expected_rows):
            raise ValueError(
                "Embedding cache row count does not match the dataset: "
                f"{self.metadata.get('num_samples')} != {expected_rows}"
            )
        if len(self.offsets) != expected_rows + 1:
            raise ValueError("Embedding cache offsets do not match dataset rows")
        if int(self.offsets[-1]) != int(self.embeddings.shape[0]):
            raise ValueError("Embedding cache offsets do not span the embedding matrix")
        if int(self.metadata.get("hidden_size", -1)) != int(self.embeddings.shape[1]):
            raise ValueError("Embedding cache metadata has the wrong hidden size")
        self._chunks_path = chunks_path
        self._chunk_texts: List[List[str]] | None = None

    def matrices(self, row_ids: Sequence[int]) -> List[np.ndarray]:
        matrices = []
        for raw_id in row_ids:
            row_id = int(raw_id)
            start = int(self.offsets[row_id])
            stop = int(self.offsets[row_id + 1])
            matrices.append(np.asarray(self.embeddings[start:stop], dtype=np.float32))
        return matrices

    def chunk_texts(self) -> List[List[str]]:
        if self._chunk_texts is None:
            rows: List[List[str]] = []
            with self._chunks_path.open(encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    rows.append([str(value) for value in payload.get("chunks", [])])
            if len(rows) != int(self.metadata["num_samples"]):
                raise ValueError("Chunk-text cache row count does not match metadata")
            self._chunk_texts = rows
        return self._chunk_texts


def _subfold_fit(
    *,
    subfold: int,
    seed: int,
    device: str,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    fit_chunks: Sequence[np.ndarray],
    fit_texts: Sequence[str],
    fit_treatment: np.ndarray,
    fit_outcome: np.ndarray,
    nuisance_views: Sequence[Any],
    nuisance_folds: int,
    nuisance_cache_path: Path,
    config: NeuralCohortWitnessConfig,
) -> Dict[str, Any]:
    LOGGER.info(
        "subfold=%d device=%s train=%d validation=%d",
        subfold,
        device,
        len(train_indices),
        len(validation_indices),
    )
    train_treatment = fit_treatment[train_indices]
    train_outcome = fit_outcome[train_indices]
    validation_treatment = fit_treatment[validation_indices]
    validation_outcome = fit_outcome[validation_indices]
    train_texts = [fit_texts[int(index)] for index in train_indices]
    validation_texts = [fit_texts[int(index)] for index in validation_indices]
    cache_identity_payload = {
        "schema": "neural_cohort_witness_subfold_nuisance_v1",
        "subfold": int(subfold),
        "seed": int(seed),
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "nuisance_folds": int(nuisance_folds),
        "view_names": [view.name for view in nuisance_views],
        "train_text_sha256": hashlib.sha256(
            "\n\x1e\n".join(train_texts).encode("utf-8")
        ).hexdigest(),
        "validation_text_sha256": hashlib.sha256(
            "\n\x1e\n".join(validation_texts).encode("utf-8")
        ).hexdigest(),
        "train_treatment": train_treatment.tolist(),
        "train_outcome": train_outcome.tolist(),
    }
    cache_identity = hashlib.sha256(
        json.dumps(cache_identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    nuisance_cache_path = Path(nuisance_cache_path)
    nuisance_cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = None
    if nuisance_cache_path.exists():
        candidate_cache = joblib.load(nuisance_cache_path)
        if candidate_cache.get("cache_identity") == cache_identity:
            cached = candidate_cache
            LOGGER.info("subfold=%d reusing strict nuisance cache", subfold)
    if cached is None:
        nuisance = fit_joint_cross_fitted_nuisance_stacks(
            texts=train_texts,
            treatment=train_treatment,
            outcome=train_outcome,
            outcome_binary=True,
            strata=_strata(train_treatment, train_outcome, outcome_binary=True),
            views=nuisance_views,
            folds=int(nuisance_folds),
            random_state=int(seed + 10_000),
        )
        train_e = np.asarray(nuisance["treatment"]["stacked_oof"], dtype=float)
        train_m = np.asarray(nuisance["outcome"]["stacked_oof"], dtype=float)
        validation_e, validation_e_views = nuisance["treatment"]["fitted"].predict(
            validation_texts
        )
        validation_m, validation_m_views = nuisance["outcome"]["fitted"].predict(
            validation_texts
        )
        cached = {
            "cache_identity": cache_identity,
            "cache_identity_payload": cache_identity_payload,
            "train_e": np.asarray(train_e, dtype=float),
            "train_m": np.asarray(train_m, dtype=float),
            "validation_e": np.asarray(validation_e, dtype=float),
            "validation_m": np.asarray(validation_m, dtype=float),
            "validation_e_view_names": sorted(validation_e_views),
            "validation_m_view_names": sorted(validation_m_views),
            "treatment_train_metrics": nuisance["treatment"]["metrics"],
            "outcome_train_metrics": nuisance["outcome"]["metrics"],
        }
        joblib.dump(cached, nuisance_cache_path)
    else:
        train_e = np.asarray(cached["train_e"], dtype=float)
        train_m = np.asarray(cached["train_m"], dtype=float)
        validation_e = np.asarray(cached["validation_e"], dtype=float)
        validation_m = np.asarray(cached["validation_m"], dtype=float)
    train_u = train_treatment - train_e
    train_v = train_outcome - train_m
    validation_u = validation_treatment - validation_e
    validation_v = validation_outcome - validation_m

    train_chunks = [fit_chunks[int(index)] for index in train_indices]
    result = fit_soft_witness_queries(
        train_chunks,
        train_u,
        train_v,
        config=config,
        seed=seed,
        device=device,
    )
    validation_chunks = [fit_chunks[int(index)] for index in validation_indices]
    validation_activations = soft_retrieval_activations(
        validation_chunks,
        result["queries"],
        temperature=config.temperature,
        device=device,
    )
    validation_scores = standardized_cohort_moments(
        validation_activations,
        validation_u,
        validation_v,
        constant_effect=float(result["constant_effect"]),
    )
    candidates = []
    for query_index, query in enumerate(result["queries"]):
        train_z = float(result["train_standardized_scores"][query_index])
        validation_z = float(validation_scores["standardized_scores"][query_index])
        candidates.append(
            {
                "candidate_id": f"subfold_{subfold:02d}_query_{query_index + 1:03d}",
                "subfold": int(subfold),
                "seed": int(seed),
                "query_index": int(query_index),
                "query": query,
                "train_standardized_score": train_z,
                "validation_standardized_score": validation_z,
                "validation_two_sided_p": float(
                    validation_scores["two_sided_p_values"][query_index]
                ),
                "train_validation_sign_agreement": bool(
                    np.sign(train_z) == np.sign(validation_z)
                ),
                "query_drift": float(result["query_drift"][query_index]),
                "train_n": int(len(train_indices)),
                "validation_n": int(len(validation_indices)),
            }
        )
    return {
        "subfold": int(subfold),
        "seed": int(seed),
        "device": str(device),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "constant_effect": float(result["constant_effect"]),
        "nuisance": {
            "nuisance_folds": int(nuisance_folds),
            "training_prediction_scope": "strict_subsubfold_oof",
            "validation_prediction_scope": "complete_subfold_train_to_external_validation",
            "cache_path": str(nuisance_cache_path),
            "cache_identity": cache_identity,
            "treatment_train_metrics": cached["treatment_train_metrics"],
            "outcome_train_metrics": cached["outcome_train_metrics"],
            "validation_view_names": {
                "treatment": cached["validation_e_view_names"],
                "outcome": cached["validation_m_view_names"],
            },
            "validation_propensity_summary": {
                "mean": float(np.mean(validation_e)),
                "minimum": float(np.min(validation_e)),
                "maximum": float(np.max(validation_e)),
            },
            "validation_outcome_prediction_summary": {
                "mean": float(np.mean(validation_m)),
                "minimum": float(np.min(validation_m)),
                "maximum": float(np.max(validation_m)),
            },
        },
        "loss_history": result["loss_history"],
        "candidates": candidates,
    }


def _run_subfolds_by_device(
    device: str,
    tasks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    outputs = []
    for task in tasks:
        outputs.append(_subfold_fit(device=device, **task))
    return outputs


def _json_safe_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in candidate.items() if key != "query"}


def _retrieved_witness_evidence(
    *,
    query: np.ndarray,
    witness_id: str,
    row_ids: Sequence[int],
    cache: FrozenChunkCache,
    top_chunks: int,
    top_terms: int,
    seed: int,
) -> Dict[str, Any]:
    query = np.asarray(query, dtype=float)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    chunk_texts = cache.chunk_texts()
    patient_best = []
    for row_id in row_ids:
        matrix = cache.matrices([int(row_id)])[0]
        if not len(matrix):
            continue
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        scores = matrix @ query
        chunk_index = int(np.argmax(scores))
        texts = chunk_texts[int(row_id)]
        text = texts[chunk_index] if chunk_index < len(texts) else ""
        patient_best.append(
            {
                "_oci_row_id": int(row_id),
                "chunk_index": chunk_index,
                "similarity": float(scores[chunk_index]),
                "text": text,
            }
        )
    patient_best.sort(key=lambda item: item["similarity"], reverse=True)
    selected = patient_best[: int(top_chunks)]

    rng = np.random.default_rng(int(seed))
    background_pool = patient_best[int(top_chunks) :]
    background_count = min(max(4 * len(selected), 20), len(background_pool))
    background = (
        list(rng.choice(background_pool, size=background_count, replace=False))
        if background_count
        else []
    )
    selected_texts = [str(item["text"]) for item in selected]
    background_texts = [str(item["text"]) for item in background]
    terms: List[Dict[str, Any]] = []
    documents = selected_texts + background_texts
    if selected_texts and documents:
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                min_df=1,
                max_df=1.0,
                max_features=20_000,
                sublinear_tf=True,
                stop_words="english",
            )
            matrix = vectorizer.fit_transform(documents)
            foreground_mean = np.asarray(matrix[: len(selected_texts)].mean(axis=0)).ravel()
            if background_texts:
                background_mean = np.asarray(matrix[len(selected_texts) :].mean(axis=0)).ravel()
            else:
                background_mean = np.zeros_like(foreground_mean)
            contrast = foreground_mean - background_mean
            names = vectorizer.get_feature_names_out()
            for index in np.argsort(contrast)[::-1][: int(top_terms)]:
                terms.append(
                    {
                        "term": str(names[int(index)]),
                        "tfidf_contrast": float(contrast[int(index)]),
                    }
                )
        except ValueError:
            terms = []
    return {
        "witness_id": witness_id,
        "retrieval_role": "interpretation_only_after_witness_selection",
        "top_fit_chunks": [
            {
                **{key: value for key, value in item.items() if key != "text"},
                "text": str(item["text"])[:1200],
            }
            for item in selected
        ],
        "top_contrastive_ngrams": terms,
    }


def _oracle_association(values: np.ndarray, target: pd.Series) -> Dict[str, Any]:
    values = np.asarray(values, dtype=float)
    usable = target.notna().to_numpy() & np.isfinite(values)
    target = target.loc[usable]
    values = values[usable]
    if len(values) < 3 or target.nunique() < 2:
        return {"kind": "unavailable", "strength": None}
    if pd.api.types.is_numeric_dtype(target) and target.nunique() > 10:
        statistic = float(spearmanr(values, target.to_numpy(dtype=float)).statistic)
        return {
            "kind": "absolute_spearman",
            "strength": abs(statistic) if np.isfinite(statistic) else None,
            "signed_statistic": statistic if np.isfinite(statistic) else None,
        }
    categories = target.astype(str)
    results = []
    for category in sorted(categories.unique()):
        binary = categories.eq(category).astype(int).to_numpy()
        if len(np.unique(binary)) != 2:
            continue
        auc = float(roc_auc_score(binary, values))
        results.append(
            {
                "category": category,
                "orientation_free_auc": max(auc, 1.0 - auc),
                "signed_auc": auc,
            }
        )
    if not results:
        return {"kind": "unavailable", "strength": None}
    best = max(results, key=lambda item: item["orientation_free_auc"])
    return {
        "kind": "best_one_vs_rest_orientation_free_auc",
        "strength": float(best["orientation_free_auc"]),
        "best_category": best["category"],
        "signed_auc": float(best["signed_auc"]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _context(Path(args.stage1_handoff), args.outer_fold, args.inner_fold)
    discovery = context["discovery"]
    artifacts = discovery["artifacts"]
    fit_ids = [int(value) for value in context["fit_row_ids"]]
    heldout_ids = [int(value) for value in context["heldout_row_ids"]]

    dataset = pd.read_parquet(args.dataset).reset_index(drop=True)
    dataset["_oci_row_id"] = np.arange(len(dataset), dtype=int)
    indexed = dataset.set_index("_oci_row_id", drop=False)
    fit_data = indexed.loc[fit_ids].reset_index(drop=True)
    heldout_data = indexed.loc[heldout_ids].reset_index(drop=True)
    cache = FrozenChunkCache(Path(args.embedding_cache), expected_rows=len(dataset))
    fit_chunks = cache.matrices(fit_ids)
    heldout_chunks = cache.matrices(heldout_ids)
    fit_texts = fit_data[args.text_column].fillna("").astype(str).tolist()

    nuisance = pd.read_parquet(artifacts["nuisance_predictions"])
    fit_nuisance = _ordered_nuisance(nuisance, fit_ids, "fit_oof")
    heldout_nuisance = _ordered_nuisance(nuisance, heldout_ids, "external_heldout")
    fit_treatment = fit_data[args.treatment_column].to_numpy(dtype=float)
    fit_outcome = fit_data[args.outcome_column].to_numpy(dtype=float)
    heldout_treatment = heldout_data[args.treatment_column].to_numpy(dtype=float)
    heldout_outcome = heldout_data[args.outcome_column].to_numpy(dtype=float)
    fit_e = fit_nuisance["treatment_stacked"].to_numpy(dtype=float)
    fit_m = fit_nuisance["outcome_stacked"].to_numpy(dtype=float)
    heldout_e = heldout_nuisance["treatment_stacked"].to_numpy(dtype=float)
    heldout_m = heldout_nuisance["outcome_stacked"].to_numpy(dtype=float)
    fit_u = fit_treatment - fit_e
    fit_v = fit_outcome - fit_m
    heldout_u = heldout_treatment - heldout_e
    heldout_v = heldout_outcome - heldout_m
    fitted_context = joblib.load(artifacts["fitted_context"])
    nuisance_views = list(fitted_context.treatment_stack.views)

    config = NeuralCohortWitnessConfig(
        n_prototypes=args.n_prototypes,
        initial_pool_size=args.initial_pool_size,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_query_drift=args.max_query_drift,
        validation_min_abs_z=args.validation_min_abs_z,
        consensus_cosine_threshold=args.consensus_cosine_threshold,
        consensus_activation_correlation_threshold=(
            args.consensus_activation_correlation_threshold
        ),
        consensus_min_subfold_recurrence=args.consensus_min_subfold_recurrence,
        consensus_max_prototypes=args.consensus_max_prototypes,
        consensus_min_prototypes=args.consensus_min_prototypes,
        kmeans_iterations=args.kmeans_iterations,
        kmeans_sample_chunks=args.kmeans_sample_chunks,
    )
    config.validate()
    devices = list(args.devices)
    if not devices:
        raise ValueError("At least one torch device is required")

    strata = (
        fit_treatment.astype(int) * 2 + fit_outcome.astype(int)
    ).astype(str)
    splitter = StratifiedKFold(
        n_splits=args.subfolds,
        shuffle=True,
        random_state=args.seed,
    )
    tasks_by_device: Dict[str, List[Dict[str, Any]]] = {device: [] for device in devices}
    nuisance_cache_dir = output_dir / "subfold_nuisance_cache"
    for subfold, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(fit_data)), strata), start=1
    ):
        device = devices[(subfold - 1) % len(devices)]
        tasks_by_device[device].append(
            {
                "subfold": subfold,
                "seed": int(args.seed + subfold - 1),
                "train_indices": np.asarray(train_indices, dtype=int),
                "validation_indices": np.asarray(validation_indices, dtype=int),
                "fit_chunks": fit_chunks,
                "fit_texts": fit_texts,
                "fit_treatment": fit_treatment,
                "fit_outcome": fit_outcome,
                "nuisance_views": nuisance_views,
                "nuisance_folds": int(args.subfold_nuisance_folds),
                "nuisance_cache_path": nuisance_cache_dir
                / f"subfold_{subfold:03d}.joblib",
                "config": config,
            }
        )
    subfold_results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(devices), args.subfolds)
    ) as executor:
        futures = [
            executor.submit(_run_subfolds_by_device, device, tasks)
            for device, tasks in tasks_by_device.items()
            if tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            subfold_results.extend(future.result())
    subfold_results.sort(key=lambda item: item["subfold"])
    candidates = [
        candidate
        for subfold_result in subfold_results
        for candidate in subfold_result["candidates"]
    ]
    all_candidate_queries = np.vstack([candidate["query"] for candidate in candidates])
    candidate_activations = soft_retrieval_activations(
        fit_chunks,
        all_candidate_queries,
        temperature=config.temperature,
        device=devices[0],
    )
    consensus = build_consensus_witness_bank(
        candidates,
        config=config,
        candidate_activations=candidate_activations,
    )
    queries = np.asarray(consensus.pop("queries"), dtype=np.float32)

    fit_activations = soft_retrieval_activations(
        fit_chunks,
        queries,
        temperature=config.temperature,
        device=devices[0],
    )
    heldout_activations = soft_retrieval_activations(
        heldout_chunks,
        queries,
        temperature=config.temperature,
        device=devices[0],
    )
    fit_means = np.mean(fit_activations, axis=0)
    fit_scales = np.std(fit_activations, axis=0)
    retained = fit_scales > 1e-7
    if not np.any(retained):
        raise RuntimeError("Every selected neural witness is constant on inner-fit rows")
    queries = queries[retained]
    fit_activations = fit_activations[:, retained]
    heldout_activations = heldout_activations[:, retained]
    fit_means = fit_means[retained]
    fit_scales = fit_scales[retained]
    consensus["records"] = [
        record for record, keep in zip(consensus["records"], retained) if keep
    ]
    consensus["selected_count_after_constant_filter"] = int(np.sum(retained))
    x_fit = (fit_activations - fit_means) / fit_scales
    x_heldout = (heldout_activations - fit_means) / fit_scales

    fit_constant_effect = fit_constant_residual_effect(fit_u, fit_v)
    heldout_moments = standardized_cohort_moments(
        heldout_activations,
        heldout_u,
        heldout_v,
        constant_effect=fit_constant_effect,
    )
    group_test = multiplier_group_score_test(
        heldout_moments["row_scores"],
        repeats=args.bootstrap_repeats,
        seed=args.seed + 1000,
        chunk_size=args.bootstrap_chunk_size,
    )
    group_test["retained_columns"] = np.asarray(
        group_test["retained_columns"], dtype=bool
    ).tolist()
    individual_q = benjamini_hochberg(heldout_moments["two_sided_p_values"])
    heldout_score_records = []
    for index, record in enumerate(consensus["records"]):
        heldout_score_records.append(
            {
                "witness_id": record["witness_id"],
                "heldout_cohort_moment": float(heldout_moments["moments"][index]),
                "heldout_standardized_score": float(
                    heldout_moments["standardized_scores"][index]
                ),
                "heldout_two_sided_p": float(
                    heldout_moments["two_sided_p_values"][index]
                ),
                "heldout_bh_q": float(individual_q[index]),
            }
        )

    w_fit = fit_nuisance[
        ["treatment_stacked", "outcome_stacked"]
    ].to_numpy(dtype=np.float32)
    forest = CausalForestHead(
        n_estimators=args.cf_n_estimators,
        max_depth=args.cf_max_depth,
        min_samples_leaf=args.cf_min_samples_leaf,
        max_features=args.cf_max_features,
        honest=True,
        inference=True,
        random_state=args.seed,
        tune_model=False,
    )
    forest.fit(x_fit, fit_treatment, fit_outcome, W=w_fit)
    forest_result = forest.predict(x_heldout, return_ci=True)
    cate = np.asarray(forest_result["tau_pred"], dtype=float)
    constant_loss = np.square(heldout_v - heldout_u * fit_constant_effect)
    forest_loss = np.square(heldout_v - heldout_u * cate)
    loss_metrics = _paired_loss_metrics(
        constant_loss,
        forest_loss,
        bootstrap_repeats=args.r_loss_bootstrap_repeats,
        seed=args.seed + 2000,
    )

    prediction_frame = pd.DataFrame(
        {
            "_oci_row_id": heldout_ids,
            "patient_id": heldout_data["patient_id"].to_numpy(),
            "outer_fold": int(args.outer_fold),
            "inner_fold": int(args.inner_fold),
            "treatment": heldout_treatment,
            "outcome": heldout_outcome,
            "propensity_oof": heldout_e,
            "outcome_prediction_oof": heldout_m,
            "treatment_residual": heldout_u,
            "outcome_residual": heldout_v,
            "fit_constant_effect": fit_constant_effect,
            "neural_witness_cate": cate,
            "constant_r_loss": constant_loss,
            "forest_r_loss": forest_loss,
            "paired_r_loss_reduction": constant_loss - forest_loss,
            "prediction_fitting_set_excludes_row_labels": True,
        }
    )
    if "tau_lower" in forest_result:
        prediction_frame["neural_witness_cate_lower"] = forest_result["tau_lower"]
        prediction_frame["neural_witness_cate_upper"] = forest_result["tau_upper"]
    prediction_path = output_dir / "heldout_predictions.parquet"
    prediction_frame.to_parquet(prediction_path, index=False)
    frozen_prediction_sha256 = _sha256(prediction_path)

    activation_rows = []
    for scope, row_ids, values in (
        ("fit", fit_ids, fit_activations),
        ("exact_inner_heldout", heldout_ids, heldout_activations),
    ):
        frame = pd.DataFrame(values, columns=[record["witness_id"] for record in consensus["records"]])
        frame.insert(0, "prediction_scope", scope)
        frame.insert(0, "_oci_row_id", row_ids)
        activation_rows.append(frame)
    pd.concat(activation_rows, ignore_index=True).to_parquet(
        output_dir / "witness_activations.parquet", index=False
    )
    np.savez_compressed(
        output_dir / "witness_bank.npz",
        queries=queries,
        fit_means=fit_means,
        fit_scales=fit_scales,
    )
    np.savez_compressed(
        output_dir / "subfold_candidate_queries.npz",
        queries=all_candidate_queries,
        candidate_ids=np.asarray([candidate["candidate_id"] for candidate in candidates]),
    )
    np.savez_compressed(
        output_dir / "subfold_candidate_activations.npz",
        activations=candidate_activations,
        row_ids=np.asarray(fit_ids, dtype=int),
    )
    joblib.dump(forest, output_dir / "causal_forest.joblib")

    retrieval_evidence = []
    for index, record in enumerate(consensus["records"]):
        retrieval_evidence.append(
            _retrieved_witness_evidence(
                query=queries[index],
                witness_id=record["witness_id"],
                row_ids=fit_ids,
                cache=cache,
                top_chunks=args.retrieval_top_chunks,
                top_terms=args.retrieval_top_terms,
                seed=args.seed + 3000 + index,
            )
        )
    (output_dir / "retrieved_witness_evidence.json").write_text(
        json.dumps(retrieval_evidence, indent=2), encoding="utf-8"
    )

    subfold_audit = []
    for result in subfold_results:
        subfold_audit.append(
            {
                "subfold": result["subfold"],
                "seed": result["seed"],
                "device": result["device"],
                "train_indices": result["train_indices"].tolist(),
                "validation_indices": result["validation_indices"].tolist(),
                "constant_effect": result["constant_effect"],
                "nuisance": result["nuisance"],
                "loss_history": result["loss_history"],
                "candidates": [
                    _json_safe_candidate(candidate) for candidate in result["candidates"]
                ],
            }
        )
    (output_dir / "subfold_audit.json").write_text(
        json.dumps(subfold_audit, indent=2), encoding="utf-8"
    )

    score_test = {
        "exact_inner_heldout_used_once_after_witness_bank_frozen": True,
        "patient_level_pseudo_target_constructed": False,
        "cate_model_used_for_witness_selection": False,
        "constant_effect": fit_constant_effect,
        "cohort_contribution_formula": (
            "treatment_residual * (outcome_residual - "
            "fit_constant_effect * treatment_residual)"
        ),
        "activation_formula": (
            "temperature * log(mean(exp(chunk_cosine_similarity / temperature)))"
        ),
        "group_test": group_test,
        "individual_witness_tests": heldout_score_records,
    }
    (output_dir / "heldout_cohort_score_test.json").write_text(
        json.dumps(score_test, indent=2), encoding="utf-8"
    )

    # Oracle evaluation begins only after the prediction file and score test exist.
    oracle_modifiers = [
        "true_histology_type",
        "true_egfr_mutation_status",
        "true_baseline_nlr",
        "true_brain_metastases_status",
        "true_baseline_hemoglobin",
    ]
    oracle_witness_associations: Dict[str, Any] = {}
    for index, record in enumerate(consensus["records"]):
        oracle_witness_associations[record["witness_id"]] = {
            column: _oracle_association(heldout_activations[:, index], heldout_data[column])
            for column in oracle_modifiers
            if column in heldout_data
        }
    oracle_metrics: Dict[str, Any] = {
        "evaluation_is_post_hoc": True,
        "frozen_prediction_sha256": frozen_prediction_sha256,
        "oracle_columns_did_not_influence_witness_learning_selection_or_forest": True,
        "witness_modifier_associations": oracle_witness_associations,
    }
    if "true_ite_prob" in heldout_data:
        truth = heldout_data["true_ite_prob"].to_numpy(dtype=float)
        oracle_metrics["cate"] = {
            "pearson_correlation": float(np.corrcoef(cate, truth)[0, 1]),
            "spearman_correlation": float(spearmanr(cate, truth).statistic),
            "oracle_ite_sd": float(np.std(truth)),
            "predicted_cate_sd": float(np.std(cate)),
        }
    (output_dir / "posthoc_oracle_metrics.json").write_text(
        json.dumps(oracle_metrics, indent=2), encoding="utf-8"
    )

    summary = {
        "scope": {
            "outer_fold": int(args.outer_fold),
            "inner_fold": int(args.inner_fold),
            "scope_id": discovery["scope_id"],
            "fit_n": len(fit_ids),
            "exact_inner_heldout_n": len(heldout_ids),
        },
        "method": {
            "name": "neural_soft_witness_cohort_contrast_v2",
            "encoder_frozen": True,
            "embedding_cache_reused": True,
            "embedding_cache": str(cache.path),
            "embedding_model": cache.metadata.get("sentence_model_name"),
            "embedding_dimension": int(cache.embeddings.shape[1]),
            "sub_inner_folds": int(args.subfolds),
            "subfold_nuisance_folds": int(args.subfold_nuisance_folds),
            "subfold_nuisance_views": [view.name for view in nuisance_views],
            "devices": devices,
            "config": config.to_dict(),
            "prior_embedding_contrast_difference": (
                "continuous constant-effect-orthogonalized cohort moments with "
                "nested validation; no R-pseudo-target tails or raw 2x2 cell contrast"
            ),
        },
        "consensus": consensus,
        "heldout_cohort_score_test": score_test,
        "causal_forest": {
            "heterogeneity_feature_count": int(x_fit.shape[1]),
            "n_estimators": int(args.cf_n_estimators),
            "min_samples_leaf": int(args.cf_min_samples_leaf),
            "adjustment_inputs": ["treatment_stacked", "outcome_stacked"],
            "r_loss": loss_metrics,
        },
        "posthoc_oracle": oracle_metrics,
        "artifacts": {
            "frozen_predictions": str(prediction_path),
            "witness_bank": str(output_dir / "witness_bank.npz"),
            "witness_activations": str(output_dir / "witness_activations.parquet"),
            "subfold_audit": str(output_dir / "subfold_audit.json"),
            "heldout_score_test": str(output_dir / "heldout_cohort_score_test.json"),
            "retrieved_evidence": str(output_dir / "retrieved_witness_evidence.json"),
            "posthoc_oracle": str(output_dir / "posthoc_oracle_metrics.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage1-handoff", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outer-fold", type=int, default=1)
    parser.add_argument("--inner-fold", type=int, default=1)
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--subfolds", type=int, default=3)
    parser.add_argument("--subfold-nuisance-folds", type=int, default=3)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--n-prototypes", type=int, default=16)
    parser.add_argument("--initial-pool-size", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--max-query-drift", type=float, default=0.35)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--kmeans-sample-chunks", type=int, default=6000)
    parser.add_argument("--validation-min-abs-z", type=float, default=1.0)
    parser.add_argument("--consensus-cosine-threshold", type=float, default=0.62)
    parser.add_argument(
        "--consensus-activation-correlation-threshold", type=float, default=0.85
    )
    parser.add_argument("--consensus-min-subfold-recurrence", type=int, default=2)
    parser.add_argument("--consensus-max-prototypes", type=int, default=16)
    parser.add_argument("--consensus-min-prototypes", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=50_000)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=2000)
    parser.add_argument("--r-loss-bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--retrieval-top-chunks", type=int, default=10)
    parser.add_argument("--retrieval-top-terms", type=int, default=20)
    parser.add_argument("--cf-n-estimators", type=int, default=400)
    parser.add_argument("--cf-max-depth", type=int, default=None)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=10)
    parser.add_argument("--cf-max-features", default="sqrt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(arguments.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    payload = run(arguments)
    print(json.dumps(payload, indent=2))
