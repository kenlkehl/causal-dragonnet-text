from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    TFIDF_SEMANTIC_RETRIEVAL,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    SEMANTIC_RETRIEVAL_DERIVATION,
)
from oci.inference.production_stage1_cluster_preflight_artifact import (
    PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_ARTIFACT_VERSION,
    PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_RESULT_SCHEMA,
    ProductionStage1ClusterPreflightArtifact,
    _artifact_code_sha256,
)
from oci.inference.production_stage1_legacy_scope_fragments import (
    build_role_neutral_fit_only_family_seal,
)
from oci.inference.production_stage1_scope_scheduler import (
    Stage1ScopePlan,
    build_canonical_stage1_scope_plan,
)
import oci.inference.production_stage1_scope_scheduler as scope_scheduler
from oci.inference.review_spent_evidence_provider import (
    SpentOnlyFrozenChunkEmbeddingCache,
)
from oci.inference.role_neutral_embedding_group_execution import (
    EmbeddingContrastSpec,
    ExactHeldoutEmbeddingBatch,
    RoleNeutralEmbeddingPhysicalGroupRequest,
    RoleNeutralEmbeddingScientificConfig,
    _array_identity,
    _canonical_json,
    _sha256_json,
    execute_role_neutral_embedding_physical_group,
    load_canonical_clustered_preflight_state_bundle,
    replay_role_neutral_embedding_exact_transform,
    seal_canonical_clustered_preflight_state_bundle,
    seal_canonical_clustered_preflight_scope_state,
    validate_role_neutral_embedding_group_execution,
)
from oci.inference.role_neutral_all_ten_binding import (
    authenticate_role_neutral_embedding_component,
)
from oci.models.concept_embedding_utils import chunk_text_words


def _registry() -> dict:
    row_count = 30
    all_rows = tuple(range(row_count))
    outer_rows = []
    for outer_fold in range(1, 3):
        start = (outer_fold - 1) * (row_count // 2)
        heldout = tuple(range(start, start + row_count // 2))
        fit = tuple(row for row in all_rows if row not in set(heldout))
        partitions = tuple(fit[index::5] for index in range(5))
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": list(fit),
                "heldout_row_ids": list(heldout),
                "inner_folds": [
                    {
                        "inner_fold": inner_fold,
                        "fit_row_ids": [
                            row for row in fit if row not in set(inner_heldout)
                        ],
                        "heldout_row_ids": list(inner_heldout),
                    }
                    for inner_fold, inner_heldout in enumerate(
                        partitions,
                        start=1,
                    )
                ],
            }
        )
    return {"dataset_row_count": row_count, "outer_folds": outer_rows}


def _plan(*, gpu_ids: tuple[int, ...] = ()):
    return build_canonical_stage1_scope_plan(
        registry=_registry(),
        registry_content_sha256="a" * 64,
        global_seed=42,
        gpu_ids=gpu_ids,
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=2,
        expected_inner_fold_count=5,
    )


def _request(plan=None) -> RoleNeutralEmbeddingPhysicalGroupRequest:
    plan = _plan() if plan is None else plan
    owner = next(
        owner
        for owner, members in plan.physical_scope_groups
        if len(members) > 1
    )
    return RoleNeutralEmbeddingPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
    )


def _one_physical_group_plan() -> Stage1ScopePlan:
    base = _plan()
    owner, members = next(
        (owner, members)
        for owner, members in base.physical_scope_groups
        if len(members) > 1
    )
    selected_ids = {member.scope_id for member in members}
    assignments = tuple(
        row for row in base.assignments if row.scope_id in selected_ids
    )
    body = scope_scheduler._stage1_scope_plan_body(
        registry_content_sha256=base.registry_content_sha256,
        global_seed=base.global_seed,
        review_rounds=base.review_rounds,
        initial_training_partitions=base.initial_training_partitions,
        gpu_ids=base.gpu_ids,
        scope_workers_per_gpu=base.scope_workers_per_gpu,
        scopes=members,
        assignments=assignments,
    )
    plan = Stage1ScopePlan(
        registry_content_sha256=base.registry_content_sha256,
        global_seed=base.global_seed,
        review_rounds=base.review_rounds,
        initial_training_partitions=base.initial_training_partitions,
        gpu_ids=base.gpu_ids,
        scope_workers_per_gpu=base.scope_workers_per_gpu,
        scopes=members,
        assignments=assignments,
        content_sha256=_sha256_json(body),
    )
    assert plan.physical_scopes == (owner,)
    plan.as_dict()
    return plan


def _singleton_request_for_kind(
    kind: str,
    plan=None,
) -> RoleNeutralEmbeddingPhysicalGroupRequest:
    plan = _plan() if plan is None else plan
    owner = next(
        owner
        for owner, members in plan.physical_scope_groups
        if owner.scope_kind == kind and len(members) == 1
    )
    return RoleNeutralEmbeddingPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
    )


def _texts() -> tuple[str, ...]:
    return tuple(
        " ".join(
            (
                f"patient_{row}",
                "treated regimen" if row % 2 else "control baseline",
                "durable response" if (row // 2) % 2 else "progressive symptoms",
                f"modifier_group_{row % 5}",
                f"complete_tail_{row}",
            )
        )
        for row in range(30)
    )


def _write_cache(
    root: Path,
    *,
    texts: tuple[str, ...],
    max_chunks_override: int | None = None,
) -> SpentOnlyFrozenChunkEmbeddingCache:
    chunk_size = 3
    overlap = 1
    full_chunks = [
        tuple(
            chunk_text_words(
                text,
                chunk_size,
                overlap,
                10_000,
                "first",
            )
        )
        for text in texts
    ]
    max_chunks = (
        max(len(value) for value in full_chunks)
        if max_chunks_override is None
        else int(max_chunks_override)
    )
    if max_chunks_override is None:
        chunks = list(full_chunks)
    else:
        # Deliberately forge the legacy clipped-cache shape so the production
        # validator can prove it rejects such bytes.  Production chunk helpers
        # themselves now fail closed and cannot create this fixture.
        chunks = [tuple(value[:max_chunks]) for value in full_chunks]
    offsets = np.zeros(len(texts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(value) for value in chunks])
    embeddings = []
    for row, row_chunks in enumerate(chunks):
        base = np.asarray(
            [
                1.0 if row % 2 else -1.0,
                1.0 if (row // 2) % 2 else -1.0,
                (row % 5) - 2.0,
                (row % 3) - 1.0,
            ],
            dtype=np.float32,
        )
        for chunk_index, _chunk in enumerate(row_chunks):
            value = base.copy()
            value[3] += 0.01 * chunk_index
            embeddings.append(value)
    root.mkdir(parents=True)
    np.save(root / "chunk_embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    np.save(root / "offsets.npy", offsets)
    chunk_counts = [len(value) for value in chunks]
    metadata = {
        "num_samples": len(texts),
        "hidden_size": 4,
        "chunk_size_words": chunk_size,
        "chunk_overlap_words": overlap,
        "max_chunks": max_chunks,
        "chunk_selection": "first",
        "chunk_counts": chunk_counts,
        "uncapped_chunk_counts_sha256": _sha256_json(chunk_counts),
        "chunk_cap_nonbinding": max_chunks_override is None,
        "semantic_truncation_allowed": False,
        "tokenizer_truncation_allowed": False,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    with (root / "chunk_texts.jsonl").open("w", encoding="utf-8") as handle:
        for row_chunks in chunks:
            handle.write(json.dumps({"chunks": list(row_chunks)}) + "\n")
    return SpentOnlyFrozenChunkEmbeddingCache(root)


def _config(**overrides) -> RoleNeutralEmbeddingScientificConfig:
    values = {
        "contrasts": (
            EmbeddingContrastSpec(
                name="treatment",
                contrast_family="marginal",
                target_name="treatment",
                sample_weight_target_name=None,
                split_rule="binary_zero_one",
            ),
            EmbeddingContrastSpec(
                name="outcome",
                contrast_family="marginal",
                target_name="outcome",
                sample_weight_target_name=None,
                split_rule="binary_zero_one",
            ),
            EmbeddingContrastSpec(
                name="effect_signal",
                contrast_family="r_pseudo_target",
                target_name="effect_signal",
                sample_weight_target_name=None,
                split_rule="stable_ordered_halves",
            ),
        ),
        "normalize_patient_embeddings": True,
        "patient_embedding_pooling": "arithmetic_mean",
        "numeric_compute_dtype": "float64",
        "vector_norm_order": "l2",
        "direction_norm_epsilon": 1e-10,
        "pseudo_target_quantile": 0.2,
        "pseudo_target_weighted": False,
        "quantile_method": "linear",
        "minimum_contrast_side_rows": 2,
        "lstsq_rcond": None,
        "lstsq_solution_policy": "numpy_minimum_norm_v1",
        "semantic_input": "content",
        "semantic_encoding": "utf-8",
        "semantic_decode_error": "strict",
        "semantic_preprocessor": None,
        "semantic_tokenizer": None,
        "semantic_analyzer": "word",
        "semantic_ngram_min": 1,
        "semantic_ngram_max": 2,
        "semantic_token_pattern": r"(?u)\b\w+\b",
        "semantic_lowercase": True,
        "semantic_strip_accents": "unicode",
        "semantic_min_df": 1,
        "semantic_max_df": 1.0,
        "semantic_sublinear_tf": True,
        "semantic_norm": "l2",
        "semantic_use_idf": True,
        "semantic_smooth_idf": True,
        "semantic_binary": False,
        "semantic_dtype": "float64",
        "semantic_stop_words": None,
        "semantic_vocabulary": None,
        "semantic_max_features": None,
        "semantic_member_batch_size": 3,
        "maximum_source_chunks_per_row": None,
        "maximum_retrieval_chunks_per_side": None,
        "maximum_semantic_terms": None,
        "overflow_policy": "fail_closed_no_selection",
    }
    values.update(overrides)
    return RoleNeutralEmbeddingScientificConfig(**values)


def _targets(rows: tuple[int, ...]) -> dict[str, np.ndarray]:
    return {
        "treatment": np.asarray([row % 2 for row in rows], dtype=float),
        "outcome": np.asarray([(row // 2) % 2 for row in rows], dtype=float),
        "effect_signal": np.asarray([(row % 5) - 2 for row in rows], dtype=float),
    }


def test_semantic_member_batch_size_is_typed_and_scientific_identity_bound():
    first = _config(semantic_member_batch_size=3)
    second = _config(semantic_member_batch_size=5)
    assert first.content_sha256 != second.content_sha256
    assert first.as_dict()["semantic_member_batch_size"] == 3
    assert second.as_dict()["semantic_member_batch_size"] == 5
    with pytest.raises(
        ValueError,
        match="semantic_member_batch_size",
    ):
        _config(semantic_member_batch_size=0)


def test_embedding_vectorizer_pooling_and_linalg_choices_are_explicit_and_bound():
    baseline = _config()
    changed_quantile = _config(quantile_method="median_unbiased")
    changed_side_support = _config(minimum_contrast_side_rows=3)
    changed_rcond = _config(lstsq_rcond=1e-12)
    assert len(
        {
            baseline.content_sha256,
            changed_quantile.content_sha256,
            changed_side_support.content_sha256,
            changed_rcond.content_sha256,
        }
    ) == 4
    configuration = baseline.as_dict()
    assert configuration["patient_embedding_pooling"] == "arithmetic_mean"
    assert configuration["numeric_compute_dtype"] == "float64"
    assert configuration["vector_norm_order"] == "l2"
    assert configuration["semantic_input"] == "content"
    assert configuration["semantic_preprocessor"] is None
    assert configuration["semantic_tokenizer"] is None
    assert configuration["semantic_vocabulary"] is None
    assert configuration["semantic_max_features"] is None


@pytest.mark.parametrize(
    "field_name",
    (
        "normalize_patient_embeddings",
        "pseudo_target_weighted",
        "semantic_lowercase",
        "semantic_sublinear_tf",
        "semantic_use_idf",
        "semantic_smooth_idf",
        "semantic_binary",
    ),
)
def test_embedding_boolean_scientific_fields_reject_truthy_substitutions(field_name):
    with pytest.raises(TypeError, match=field_name):
        _config(**{field_name: 1})


def _catalog_concepts():
    clustered = []
    semantic = []
    for index, family in enumerate(
        (
            "cluster_local_treatment_contrast_basis",
            "cluster_local_residualized_interaction_contrast_basis",
        )
    ):
        contrast = {
            "name": f"{family}_{index}",
            "contrast_family": family,
            "direction_source": f"canonical_preflight_svd_{index}",
            "cluster_component_index": index,
        }
        witnesses = [
            {"concept": f"cluster concept {index}", "score": 0.5 + index}
        ]
        common = {
            "contrast": contrast,
            "concept_witnesses": witnesses,
            "member_batch_index": 1,
            "member_batch_count": 1,
            "full_member_count": 1,
        }
        clustered.append(
            {
                "atom_kind": "embedding_contrast",
                "content": {
                    "architecture_view": "embedding_contrast",
                    **copy.deepcopy(common),
                },
            }
        )
        semantic.append(
            {
                "atom_kind": "tfidf_semantic_retrieval_contrast",
                "content": {
                    "architecture_view": SEMANTIC_RETRIEVAL_DERIVATION,
                    "source_passages_removed": True,
                    **copy.deepcopy(common),
                },
            }
        )
    return {
        EMBEDDING_CLUSTERED: clustered,
        TFIDF_SEMANTIC_RETRIEVAL: semantic,
    }


def _cluster_scientific_configuration() -> dict:
    return {
        "requested_cluster_count": 2,
        "cluster_count_policy": "require_exact_configured_count_v1",
        "maximum_components_per_family": 2,
        "loading_evidence_capacity": None,
        "loading_evidence_overflow_policy": "fail_closed_no_truncation_v1",
        "minimum_cluster_size": 2,
        "minimum_group_size": 2,
        "minimum_cell_size": 1,
        "minimum_distinct_local_clusters_per_family": 2,
        "minimum_numerical_rank_per_family": 2,
        "patient_pooling_policy": (
            "arithmetic_mean_all_authenticated_chunks_v1"
        ),
        "computation_dtype": "float64",
        "normalize_patient_embeddings": True,
        "normalization_epsilon": 1e-12,
        "zero_vector_policy": "reject",
        "local_direction_weighting_policy": (
            "sqrt_cluster_size_times_unit_direction_v1"
        ),
        "kmeans_init": "k-means++",
        "kmeans_max_iter": 10,
        "kmeans_batch_size_policy": (
            "clamp_usable_rows_to_configured_bounds_v1"
        ),
        "kmeans_batch_size_lower_bound": 1,
        "kmeans_batch_size_upper_bound": 1024,
        "kmeans_verbose": 0,
        "kmeans_compute_labels": True,
        "kmeans_seed_derivation_policy": (
            "canonical_ordered_fit_rows_group_seed_v1"
        ),
        "kmeans_tol": 0.0,
        "kmeans_max_no_improvement": 10,
        "kmeans_init_size": None,
        "kmeans_n_init": 1,
        "kmeans_reassignment_ratio": 0.01,
        "svd_full_matrices": False,
        "svd_compute_uv": True,
        "svd_hermitian": False,
        "svd_sign_canonicalization_policy": (
            "largest_absolute_coordinate_positive_v1"
        ),
        "svd_rank_tolerance_policy": (
            "dtype_epsilon_times_max_shape_times_largest_singular_v1"
        ),
        "svd_rank_tolerance_dtype": "float64",
        "svd_rank_tolerance_multiplier": 1.0,
        "replay_comparison_policy": (
            "allclose_and_exact_discrete_state_v1"
        ),
        "replay_relative_tolerance": 2e-6,
        "replay_absolute_tolerance": 2e-7,
        "exception_policy": "abort_scope_no_skip_or_fallback_v1",
    }


def _preflight_and_states(
    *,
    tmp_path: Path,
    request: RoleNeutralEmbeddingPhysicalGroupRequest,
    cache: SpentOnlyFrozenChunkEmbeddingCache,
):
    rows = request.physical_owner.fit_row_ids
    usable = np.ones(len(rows), dtype=np.bool_)
    labels = np.asarray([index % 2 for index in range(len(rows))], dtype=np.int64)
    centers = np.asarray(
        [[-0.5, -0.5, -0.5, 0.0], [0.5, 0.5, 0.5, 0.0]],
        dtype=np.float64,
    )
    counts = np.bincount(labels, minlength=2).astype(np.int64)
    components_0 = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    components_1 = np.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cluster_configuration = _cluster_scientific_configuration()
    svd_parameters = {
        "full_matrices": False,
        "compute_uv": True,
        "hermitian": False,
    }
    rank_tolerance = (
        np.finfo(np.float64).eps
        * max(components_0.shape)
        * 1.0
    )
    svd_states = [
        {
            "family_key": "treatment",
            "item_cluster_ids": [0, 1],
            "weighted_matrix": components_0.copy(),
            "singular_values": np.asarray([1.0, 0.5], dtype=np.float64),
            "components": components_0,
            "parameters": copy.deepcopy(svd_parameters),
            "sign_canonicalization_policy": (
                "largest_absolute_coordinate_positive_v1"
            ),
            "rank_tolerance_policy": (
                "dtype_epsilon_times_max_shape_times_largest_singular_v1"
            ),
            "rank_tolerance_dtype": "float64",
            "rank_tolerance_multiplier": 1.0,
            "rank_tolerance": rank_tolerance,
            "numerical_rank": 2,
            "replay_comparison_policy": (
                "allclose_and_exact_discrete_state_v1"
            ),
            "replay_relative_tolerance": 2e-6,
            "replay_absolute_tolerance": 2e-7,
        },
        {
            "family_key": "residualized_interaction",
            "item_cluster_ids": [0, 1],
            "weighted_matrix": components_1.copy(),
            "singular_values": np.asarray([0.9, 0.4], dtype=np.float64),
            "components": components_1,
            "parameters": copy.deepcopy(svd_parameters),
            "sign_canonicalization_policy": (
                "largest_absolute_coordinate_positive_v1"
            ),
            "rank_tolerance_policy": (
                "dtype_epsilon_times_max_shape_times_largest_singular_v1"
            ),
            "rank_tolerance_dtype": "float64",
            "rank_tolerance_multiplier": 1.0,
            "rank_tolerance": rank_tolerance,
            "numerical_rank": 2,
            "replay_comparison_policy": (
                "allclose_and_exact_discrete_state_v1"
            ),
            "replay_relative_tolerance": 2e-6,
            "replay_absolute_tolerance": 2e-7,
        },
    ]
    kmeans_parameters = {
        "n_clusters": 2,
        "init": "k-means++",
        "max_iter": 10,
        "batch_size": len(rows),
        "verbose": 0,
        "compute_labels": True,
        "random_state": request.physical_owner.scope_seed,
        "tol": 0.0,
        "max_no_improvement": 10,
        "init_size": None,
        "n_init": 1,
        "reassignment_ratio": 0.01,
    }
    kmeans_state = {
        "fit_row_ids": list(rows),
        "parameters": kmeans_parameters,
        "scientific_configuration": cluster_configuration,
        "canonical_group_seed": request.physical_owner.scope_seed,
        "ordered_fit_row_seed_policy": (
            "canonical_ordered_fit_rows_group_seed_v1"
        ),
        "usable_mask": usable,
        "cluster_labels": labels,
        "cluster_centers": centers,
        "cluster_counts": counts,
        "n_iter": 3,
        "inertia": 1.25,
    }
    fit_body = {
        "schema_version": "production_stage1_embedding_cluster_fit_identity_v2",
        "scope_id": request.physical_owner.scope_id,
        "fit_row_ids": list(rows),
        "fit_row_order_fingerprint": _sha256_json(list(rows)),
        "canonical_group_seed": request.physical_owner.scope_seed,
        "ordered_fit_row_seed_policy": (
            "canonical_ordered_fit_rows_group_seed_v1"
        ),
        "cluster_scientific_configuration": cluster_configuration,
        "cluster_scientific_configuration_sha256": _sha256_json(
            cluster_configuration
        ),
        "kmeans": {
            "parameters": copy.deepcopy(kmeans_state["parameters"]),
            "usable_mask": _array_identity(usable),
            "cluster_labels": _array_identity(labels),
            "cluster_centers": _array_identity(centers),
            "cluster_counts": _array_identity(counts),
            "n_iter": 3,
            "inertia_hex": float(1.25).hex(),
        },
        "svd_families": [
            {
                "family_key": row["family_key"],
                "item_cluster_ids": row["item_cluster_ids"],
                "weighted_matrix": _array_identity(row["weighted_matrix"]),
                "singular_values": _array_identity(row["singular_values"]),
                "components": _array_identity(row["components"]),
                "parameters": copy.deepcopy(row["parameters"]),
                "sign_canonicalization_policy": row[
                    "sign_canonicalization_policy"
                ],
                "rank_tolerance_policy": row["rank_tolerance_policy"],
                "rank_tolerance_dtype": row["rank_tolerance_dtype"],
                "rank_tolerance_multiplier_hex": float(
                    row["rank_tolerance_multiplier"]
                ).hex(),
                "rank_tolerance_hex": float(row["rank_tolerance"]).hex(),
                "numerical_rank": row["numerical_rank"],
                "replay_comparison_policy": row[
                    "replay_comparison_policy"
                ],
                "replay_relative_tolerance_hex": float(
                    row["replay_relative_tolerance"]
                ).hex(),
                "replay_absolute_tolerance_hex": float(
                    row["replay_absolute_tolerance"]
                ).hex(),
            }
            for row in svd_states
        ],
        "raw_cluster_concepts": [{"complete": True}],
        "raw_cluster_concepts_sha256": _sha256_json([{"complete": True}]),
        "semantic_cluster_concepts": [{"complete": True}],
        "semantic_cluster_concepts_sha256": _sha256_json([{"complete": True}]),
        "final_catalog_concepts": _catalog_concepts(),
        "final_catalog_concepts_sha256": _sha256_json(_catalog_concepts()),
    }
    fit_identity = {**fit_body, "content_sha256": _sha256_json(fit_body)}
    scope = {
        "scope_id": request.physical_owner.scope_id,
        "scope_kind": request.physical_owner.scope_kind,
        "outer_fold": request.physical_owner.outer_fold,
        "inner_fold": request.physical_owner.inner_fold,
        "context_epoch": request.physical_owner.context_epoch,
        "provider_inner_fold": request.physical_owner.provider_inner_fold,
        "fit_row_count": len(rows),
        "fit_row_order_fingerprint": _sha256_json(list(rows)),
        "canonical_group_seed": request.physical_owner.scope_seed,
        "heldout_row_count": len(request.physical_owner.heldout_row_ids),
        "heldout_row_order_fingerprint": _sha256_json(
            list(request.physical_owner.heldout_row_ids)
        ),
        "token_bounded_row_count": 0,
        "uncapped_semantic_projection": True,
        "cluster_fit_identity": fit_identity,
    }
    cache_identity = cache.identity()
    audit_body = {
        "schema_version": "test_cluster_audit_v1",
        "embedding_cache_identity_sha256": _sha256_json(cache_identity),
        "scope_order": [request.physical_owner.scope_id],
        "scopes": [scope],
    }
    audit = {**audit_body, "content_sha256": _sha256_json(audit_body)}
    artifact_identity_body = {
        "schema_version": PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_RESULT_SCHEMA,
        "artifact_version": PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_ARTIFACT_VERSION,
        "artifact_code_sha256": _artifact_code_sha256(),
        "root": str(tmp_path / "sealed_preflight"),
        "manifest_path": str(tmp_path / "sealed_preflight" / "manifest.json"),
        "audit_path": str(tmp_path / "sealed_preflight" / "audit.json"),
        "stage1_request_path": str(tmp_path / "sealed_preflight" / "request.json"),
        "manifest_sha256": "b" * 64,
        "audit_sha256": "c" * 64,
        "stage1_request_file_sha256": "d" * 64,
        "stage1_request_sha256": "e" * 64,
        "cluster_audit_content_sha256": audit["content_sha256"],
        "scope_count": 1,
        "scope_order": [request.physical_owner.scope_id],
        "scope_fit_identity_sha256": _sha256_json(
            [fit_identity["content_sha256"]]
        ),
    }
    artifact_identity = {
        **artifact_identity_body,
        "content_sha256": _sha256_json(artifact_identity_body),
    }
    artifact = ProductionStage1ClusterPreflightArtifact(
        root=Path(artifact_identity_body["root"]),
        manifest_path=Path(artifact_identity_body["manifest_path"]),
        audit_path=Path(artifact_identity_body["audit_path"]),
        stage1_request_path=Path(artifact_identity_body["stage1_request_path"]),
        audit=audit,
        stage1_request={"request_sha256": "e" * 64},
        _identity=artifact_identity,
    )
    state = seal_canonical_clustered_preflight_scope_state(
        output_root=(tmp_path / "cluster_state").resolve(),
        preflight=artifact,
        request=request,
        kmeans_state=kmeans_state,
        svd_states=svd_states,
    )
    return artifact, state, kmeans_state, svd_states


def _case(
    tmp_path: Path,
    *,
    request: RoleNeutralEmbeddingPhysicalGroupRequest | None = None,
):
    request = _request() if request is None else request
    texts = _texts()
    cache = _write_cache(tmp_path / "cache", texts=texts)
    fit_rows = request.physical_owner.fit_row_ids
    heldout_rows = request.physical_owner.heldout_row_ids
    fit_texts = tuple(texts[row] for row in fit_rows)
    heldout_texts = tuple(texts[row] for row in heldout_rows)
    fit_provider = cache.bind_spent(fit_rows, fit_texts)
    heldout_provider = cache.bind_spent(heldout_rows, heldout_texts)
    preflight, state, _kmeans, _svds = _preflight_and_states(
        tmp_path=tmp_path,
        request=request,
        cache=cache,
    )
    return {
        "request": request,
        "texts": texts,
        "fit_texts": fit_texts,
        "heldout_texts": heldout_texts,
        "fit_provider": fit_provider,
        "heldout_provider": heldout_provider,
        "preflight": preflight,
        "state": state,
        "config": _config(),
        "targets": _targets(fit_rows),
    }


def _execute(tmp_path: Path, case: dict, *, loader=None):
    request = case["request"]
    batch = ExactHeldoutEmbeddingBatch(
        row_ids=request.physical_owner.heldout_row_ids,
        texts=case["heldout_texts"],
        embedding_provider=case["heldout_provider"],
    )
    return execute_role_neutral_embedding_physical_group(
        request=request,
        output_root=(tmp_path / "execution").resolve(),
        fit_texts=case["fit_texts"],
        fit_targets=case["targets"],
        fit_embedding_provider=case["fit_provider"],
        scientific_config=case["config"],
        clustered_preflight=case["preflight"],
        clustered_preflight_state_manifest=case["state"].root
        / "cluster_state_manifest.json",
        exact_heldout_loader=(lambda _rows: batch) if loader is None else loader,
    )


def test_request_scientific_identity_is_device_neutral():
    cpu = _plan()
    heterogeneous = _plan(gpu_ids=(7, 2))
    cpu_request = _request(cpu)
    gpu_request = _request(heterogeneous)
    assert cpu.content_sha256 != heterogeneous.content_sha256
    assert cpu.scientific_content_sha256 == heterogeneous.scientific_content_sha256
    assert cpu_request.as_dict() == gpu_request.as_dict()


def test_cluster_state_scientific_manifest_is_preflight_location_neutral(
    tmp_path: Path,
):
    case = _case(tmp_path)
    first = case["preflight"]
    identity = first.identity()
    relocated_body = {
        key: copy.deepcopy(value)
        for key, value in identity.items()
        if key != "content_sha256"
    }
    relocated_root = tmp_path / "another_machine" / "sealed_preflight"
    relocated_body.update(
        {
            "root": str(relocated_root),
            "manifest_path": str(relocated_root / "manifest.json"),
            "audit_path": str(relocated_root / "audit.json"),
            "stage1_request_path": str(relocated_root / "request.json"),
            "manifest_sha256": "f" * 64,
        }
    )
    relocated_identity = {
        **relocated_body,
        "content_sha256": _sha256_json(relocated_body),
    }
    relocated = ProductionStage1ClusterPreflightArtifact(
        root=relocated_root,
        manifest_path=Path(relocated_body["manifest_path"]),
        audit_path=Path(relocated_body["audit_path"]),
        stage1_request_path=Path(relocated_body["stage1_request_path"]),
        audit=dict(first.audit),
        stage1_request=dict(first.stage1_request),
        _identity=relocated_identity,
    )
    scope = case["state"].scope_record
    kmeans = {
        "fit_row_ids": scope["cluster_fit_identity"]["fit_row_ids"],
        "parameters": case["state"].manifest["state_metadata"][
            "kmeans_parameters"
        ],
        "scientific_configuration": case["state"].manifest["state_metadata"][
            "cluster_scientific_configuration"
        ],
        "canonical_group_seed": case["state"].manifest["state_metadata"][
            "canonical_group_seed"
        ],
        "ordered_fit_row_seed_policy": case["state"].manifest[
            "state_metadata"
        ]["ordered_fit_row_seed_policy"],
        "usable_mask": case["state"].arrays["cluster_kmeans_usable_mask"],
        "cluster_labels": case["state"].arrays["cluster_kmeans_labels"],
        "cluster_centers": case["state"].arrays["cluster_kmeans_centers"],
        "cluster_counts": case["state"].arrays["cluster_kmeans_counts"],
        "n_iter": case["state"].manifest["state_metadata"]["kmeans_n_iter"],
        "inertia": float.fromhex(
            case["state"].manifest["state_metadata"]["kmeans_inertia_hex"]
        ),
    }
    svds = [
        {
            "family_key": row["family_key"],
            "item_cluster_ids": row["item_cluster_ids"],
            "weighted_matrix": case["state"].arrays[row["weighted_matrix"]],
            "singular_values": case["state"].arrays[row["singular_values"]],
            "components": case["state"].arrays[row["components"]],
            "parameters": row["parameters"],
            "sign_canonicalization_policy": row[
                "sign_canonicalization_policy"
            ],
            "rank_tolerance_policy": row["rank_tolerance_policy"],
            "rank_tolerance_dtype": row["rank_tolerance_dtype"],
            "rank_tolerance_multiplier": float.fromhex(
                row["rank_tolerance_multiplier_hex"]
            ),
            "rank_tolerance": float.fromhex(row["rank_tolerance_hex"]),
            "numerical_rank": row["numerical_rank"],
            "replay_comparison_policy": row[
                "replay_comparison_policy"
            ],
            "replay_relative_tolerance": float.fromhex(
                row["replay_relative_tolerance_hex"]
            ),
            "replay_absolute_tolerance": float.fromhex(
                row["replay_absolute_tolerance_hex"]
            ),
        }
        for row in case["state"].manifest["state_metadata"]["svd_states"]
    ]
    second = seal_canonical_clustered_preflight_scope_state(
        output_root=(tmp_path / "relocated_cluster_state").resolve(),
        preflight=relocated,
        request=case["request"],
        kmeans_state=kmeans,
        svd_states=svds,
    )
    assert second.manifest == case["state"].manifest


def test_cluster_state_bundle_seals_every_physical_owner_once_and_fails_closed(
    tmp_path: Path,
):
    plan = _one_physical_group_plan()
    request = RoleNeutralEmbeddingPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=plan.physical_scopes[0].scope_id,
    )
    texts = _texts()
    cache = _write_cache(tmp_path / "bundle_cache", texts=texts)
    preflight, _state, kmeans, svds = _preflight_and_states(
        tmp_path=tmp_path / "bundle_fixture",
        request=request,
        cache=cache,
    )
    fit_identity = preflight.audit["scopes"][0]["cluster_fit_identity"]
    captured = {
        request.physical_owner.scope_id: {
                "schema_version": (
                    "production_stage1_cluster_preflight_scope_state_capture_v2"
                ),
            "scope_id": request.physical_owner.scope_id,
            "cluster_fit_identity_content_sha256": (
                fit_identity["content_sha256"]
            ),
            "kmeans_state": kmeans,
            "svd_states": svds,
            "captured_from_canonical_preflight_fit": True,
            "refit_performed_for_state_capture": False,
        }
    }
    bundle = seal_canonical_clustered_preflight_state_bundle(
        output_root=(tmp_path / "state_bundle").resolve(),
        preflight=preflight,
        plan=plan,
        captured_scope_states=captured,
    )
    owner_id = request.physical_owner.scope_id
    assert bundle.manifest["physical_owner_count"] == 1
    assert bundle.manifest["logical_scope_count"] == 2
    assert bundle.manifest["deduplicated_logical_scope_count"] == 1
    assert bundle.manifest["cluster_refit_performed"] is False
    assert bundle.manifest["logical_alias_state_copies_published"] is False
    assert set(bundle.states) == {owner_id}
    assert (
        bundle.manifest_path_for_owner(owner_id)
        == bundle.root / "owners" / "000" / "cluster_state_manifest.json"
    )
    reopened = load_canonical_clustered_preflight_state_bundle(
        manifest_path=bundle.root / "cluster_state_bundle_manifest.json",
        preflight=preflight,
        plan=plan,
    )
    assert reopened.manifest == bundle.manifest

    with pytest.raises(ValueError, match="legacy audit-only"):
        load_canonical_clustered_preflight_state_bundle(
            manifest_path=preflight.manifest_path,
            preflight=preflight,
            plan=plan,
        )

    array_path = next((bundle.root / "owners" / "000" / "arrays").glob("*.npy"))
    payload = bytearray(array_path.read_bytes())
    payload[-1] ^= 1
    array_path.write_bytes(payload)
    with pytest.raises((ValueError, RuntimeError), match="cluster state array|array"):
        load_canonical_clustered_preflight_state_bundle(
            manifest_path=bundle.root / "cluster_state_bundle_manifest.json",
            preflight=preflight,
            plan=plan,
        )


def test_all_three_families_seal_before_exact_loader_and_replay(tmp_path: Path):
    case = _case(tmp_path)
    request = case["request"]
    root = (tmp_path / "execution").resolve()
    batch = ExactHeldoutEmbeddingBatch(
        row_ids=request.physical_owner.heldout_row_ids,
        texts=case["heldout_texts"],
        embedding_provider=case["heldout_provider"],
    )
    calls = []

    def loader(rows):
        for filename in (
            "fit_only_embedding_whole_cohort_seal.json",
            "fit_only_embedding_clustered_seal.json",
            "fit_only_tfidf_semantic_retrieval_seal.json",
        ):
            assert (root / filename).is_file()
        cumulative = request.logical_members[1]
        for family_slug in (
            "embedding_whole_cohort",
            "embedding_clustered",
            "tfidf_semantic_retrieval",
        ):
            path = root / "logical_views" / f"{cumulative.scope_id}.{family_slug}.json"
            assert path.is_file()
            assert json.loads(path.read_text())["registered_heldout_text_accessed"] is False
        calls.append(rows)
        return batch

    terminal = _execute(tmp_path, case, loader=loader)
    assert calls == [request.physical_owner.heldout_row_ids]
    assert terminal["families"] == [
        EMBEDDING_WHOLE_COHORT,
        EMBEDDING_CLUSTERED,
        TFIDF_SEMANTIC_RETRIEVAL,
    ]
    assert terminal["cluster_refit_performed"] is False
    assert terminal["text_truncation_applied"] is False
    assert terminal["semantic_term_truncation_applied"] is False

    for family, filename in (
        (EMBEDDING_WHOLE_COHORT, "fit_only_embedding_whole_cohort_seal.json"),
        (EMBEDDING_CLUSTERED, "fit_only_embedding_clustered_seal.json"),
        (
            TFIDF_SEMANTIC_RETRIEVAL,
            "fit_only_tfidf_semantic_retrieval_seal.json",
        ),
    ):
        seal = json.loads((root / filename).read_text())
        assert seal == build_role_neutral_fit_only_family_seal(
            plan=_plan(),
            physical_owner_scope_id=request.physical_owner.scope_id,
            family=family,
            evidence_payload=seal["evidence_payload"],
            producer_identity_sha256=seal["producer_identity_sha256"],
            configuration_identity_sha256=seal[
                "configuration_identity_sha256"
            ],
            fit_state_artifact_sha256=seal["fit_state_artifact_sha256"],
        )
        assert seal["evidence_payload"]["architecture_evidence"]
    receipt = authenticate_role_neutral_embedding_component(
        root=root,
        plan=_plan(),
        request=request,
        clustered_preflight=case["preflight"],
        clustered_preflight_state_manifest=case["state"].root
        / "cluster_state_manifest.json",
        expected_scientific_config=case["config"],
        expected_fit_texts=case["fit_texts"],
        expected_fit_targets=case["targets"],
        expected_exact_batch=batch,
    )
    assert set(receipt.family_fit_seals) == {
        EMBEDDING_WHOLE_COHORT,
        EMBEDDING_CLUSTERED,
        TFIDF_SEMANTIC_RETRIEVAL,
    }
    assert receipt.lossy_evidence_selection_applied is False

    replay = replay_role_neutral_embedding_exact_transform(
        root=root,
        request=request,
        clustered_preflight=case["preflight"],
        clustered_preflight_state_manifest=case["state"].root
        / "cluster_state_manifest.json",
        exact_heldout_batch=batch,
    )
    assert replay["cluster_refit_performed"] is False
    assert replay["pickle_joblib_or_npz_loaded"] is False
    assert "heldout_cluster_distances" in replay["arrays"]
    assert "heldout_whole_patient_scores" in replay["arrays"]
    assert "heldout_lexical_csr_data" in replay["arrays"]
    assert not tuple(root.rglob("*.npz"))
    assert not tuple(root.rglob("*.joblib"))
    assert not tuple(root.rglob("*.pickle"))


@pytest.mark.parametrize("scope_kind", ["full_outer", "cumulative_spent"])
def test_singleton_physical_owners_receive_complete_heldout_embedding_evidence(
    tmp_path: Path,
    scope_kind: str,
):
    request = _singleton_request_for_kind(scope_kind)
    case = _case(tmp_path, request=request)
    calls: list[tuple[int, ...]] = []
    batch = ExactHeldoutEmbeddingBatch(
        row_ids=request.physical_owner.heldout_row_ids,
        texts=case["heldout_texts"],
        embedding_provider=case["heldout_provider"],
    )

    def loader(rows):
        calls.append(tuple(rows))
        return batch

    terminal = _execute(tmp_path, case, loader=loader)
    assert calls == [request.physical_owner.heldout_row_ids]
    assert terminal["only_physical_owner_transformed_heldout"] is True
    assert terminal["registered_heldout_labels_accessed"] is False
    assert len(terminal["logical_views"]) == 3
    for registration in terminal["logical_views"]:
        view = json.loads(
            ((tmp_path / "execution") / registration["relative_path"]).read_text()
        )
        assert view["logical_scope_id"] == request.physical_owner.scope_id
        assert view["logical_purpose"] == scope_kind
        assert view["logical_transform_performed"] is True
        assert view["prediction_artifacts"]
        assert view["registered_heldout_text_accessed"] is True
        assert view["registered_heldout_labels_accessed"] is False
    assert all(
        event["registered_heldout_labels_accessed"] is False
        for event in terminal["event_order"]
    )


def test_nonbinding_limits_preserve_late_text_and_all_semantic_terms(tmp_path: Path):
    case = _case(tmp_path)
    request = case["request"]
    long_fit_texts = list(case["fit_texts"])
    sentinel = "sentinel_after_fourteen_thousand"
    long_fit_texts[0] = ("paddingterm " * 1300) + sentinel
    texts = list(case["texts"])
    texts[request.physical_owner.fit_row_ids[0]] = long_fit_texts[0]
    # Rebuild the complete cache and its preflight binding for the changed data.
    alternate = tmp_path / "long"
    alternate.mkdir()
    cache = _write_cache(alternate / "cache", texts=tuple(texts))
    case["fit_texts"] = tuple(
        texts[row] for row in request.physical_owner.fit_row_ids
    )
    case["heldout_texts"] = tuple(
        texts[row] for row in request.physical_owner.heldout_row_ids
    )
    case["fit_provider"] = cache.bind_spent(
        request.physical_owner.fit_row_ids,
        case["fit_texts"],
    )
    case["heldout_provider"] = cache.bind_spent(
        request.physical_owner.heldout_row_ids,
        case["heldout_texts"],
    )
    case["preflight"], case["state"], _k, _s = _preflight_and_states(
        tmp_path=alternate,
        request=request,
        cache=cache,
    )
    execute_role_neutral_embedding_physical_group(
        request=request,
        output_root=(alternate / "execution").resolve(),
        fit_texts=case["fit_texts"],
        fit_targets=case["targets"],
        fit_embedding_provider=case["fit_provider"],
        scientific_config=case["config"],
        clustered_preflight=case["preflight"],
        clustered_preflight_state_manifest=case["state"].root
        / "cluster_state_manifest.json",
        exact_heldout_loader=lambda _rows: ExactHeldoutEmbeddingBatch(
            row_ids=request.physical_owner.heldout_row_ids,
            texts=case["heldout_texts"],
            embedding_provider=case["heldout_provider"],
        ),
    )
    seal = json.loads(
        (
            alternate
            / "execution"
            / "fit_only_tfidf_semantic_retrieval_seal.json"
        ).read_text()
    )
    serialized = _canonical_json(seal)
    assert len(case["fit_texts"][0]) > 14_000
    assert sentinel in serialized
    native_rows = [
        item
        for item in seal["evidence_payload"]["architecture_evidence"]
        if str(
            item.get("content", {})
            .get("contrast", {})
            .get("direction_source", "")
        ).startswith("fit_target:")
    ]
    assert native_rows
    grouped = {}
    for item in native_rows:
        content = item["content"]
        grouped.setdefault(content["contrast"]["name"], []).append(content)
    assert any(
        rows[0]["full_member_count"] > 3
        for rows in grouped.values()
    )
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: row["member_batch_index"])
        assert [row["member_batch_index"] for row in ordered] == list(
            range(1, len(ordered) + 1)
        )
        assert all(
            row["member_batch_count"] == len(ordered)
            for row in ordered
        )
        assert all(
            len(row["concept_witnesses"])
            <= case["config"].semantic_member_batch_size
            for row in ordered
        )
        assert sum(
            len(row["concept_witnesses"])
            for row in ordered
        ) == ordered[0]["full_member_count"]


def test_capacity_overflow_and_truncated_cache_fail_closed(tmp_path: Path):
    case = _case(tmp_path)
    with pytest.raises(ValueError, match="semantic-term capacity"):
        execute_role_neutral_embedding_physical_group(
            request=case["request"],
            output_root=(tmp_path / "term_overflow").resolve(),
            fit_texts=case["fit_texts"],
            fit_targets=case["targets"],
            fit_embedding_provider=case["fit_provider"],
            scientific_config=_config(maximum_semantic_terms=1),
            clustered_preflight=case["preflight"],
            clustered_preflight_state_manifest=case["state"].root
            / "cluster_state_manifest.json",
            exact_heldout_loader=None,
        )

    truncated_root = tmp_path / "truncated"
    cache = _write_cache(
        truncated_root,
        texts=case["texts"],
        max_chunks_override=1,
    )
    with pytest.raises(ValueError, match="semantic truncation|truncate"):
        cache.bind_spent(
            case["request"].physical_owner.fit_row_ids,
            case["fit_texts"],
        )


@pytest.mark.parametrize("kind", ["tamper", "symlink", "hardlink", "reorder"])
def test_fresh_validation_rejects_tamper_links_and_reordering(
    tmp_path: Path,
    kind: str,
):
    case = _case(tmp_path)
    terminal = _execute(tmp_path, case)
    root = (tmp_path / "execution").resolve()
    if kind == "tamper":
        path = root / "fit_state" / "arrays" / "whole_direction_matrix.npy"
        value = np.load(path, allow_pickle=False)
        value = np.asarray(value).copy()
        value[0, 0] += 100
        with path.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
    elif kind == "symlink":
        path = root / "fit_state" / "arrays" / "semantic_idf.npy"
        backup = tmp_path / "semantic_idf.npy"
        os.replace(path, backup)
        path.symlink_to(backup)
    elif kind == "hardlink":
        path = root / "fit_state" / "arrays" / "fit_target_matrix.npy"
        os.link(path, tmp_path / "linked_target.npy")
    else:
        path = root / "execution_manifest.json"
        value = json.loads(path.read_text())
        value["logical_views"][0], value["logical_views"][1] = (
            value["logical_views"][1],
            value["logical_views"][0],
        )
        body = {
            key: child
            for key, child in value.items()
            if key != "content_sha256"
        }
        value["content_sha256"] = _sha256_json(body)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match="artifact|array|linked|view|manifest"):
        validate_role_neutral_embedding_group_execution(
            root=root,
            request=case["request"],
            clustered_preflight=case["preflight"],
            clustered_preflight_state_manifest=case["state"].root
            / "cluster_state_manifest.json",
            expected_scientific_config=case["config"],
        )


def test_family_independence_and_cluster_state_tamper_fail_closed(tmp_path: Path):
    case = _case(tmp_path)
    _execute(tmp_path, case)
    root = (tmp_path / "execution").resolve()
    payloads = {}
    for family, filename in (
        (EMBEDDING_WHOLE_COHORT, "fit_only_embedding_whole_cohort_seal.json"),
        (EMBEDDING_CLUSTERED, "fit_only_embedding_clustered_seal.json"),
        (
            TFIDF_SEMANTIC_RETRIEVAL,
            "fit_only_tfidf_semantic_retrieval_seal.json",
        ),
    ):
        payloads[family] = json.loads((root / filename).read_text())[
            "evidence_payload"
        ]
    assert len({_sha256_json(value) for value in payloads.values()}) == 3
    assert all(value["architecture_evidence"] for value in payloads.values())

    center_path = (
        case["state"].root
        / "arrays"
        / "cluster_kmeans_centers.npy"
    )
    centers = np.load(center_path, allow_pickle=False).copy()
    centers[0, 0] += 1
    with center_path.open("wb") as handle:
        np.save(handle, centers, allow_pickle=False)
    with pytest.raises(ValueError, match="cluster state array|canonical"):
        validate_role_neutral_embedding_group_execution(
            root=root,
            request=case["request"],
            clustered_preflight=case["preflight"],
            clustered_preflight_state_manifest=case["state"].root
            / "cluster_state_manifest.json",
        )
