"""Explicit closed cluster-local embedding scientific fixtures."""

from __future__ import annotations

from typing import Any

from oci.config import ClusterLocalEmbeddingScientificConfig


def cluster_local_embedding_mapping(**overrides: Any) -> dict[str, Any]:
    value = {
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
        "kmeans_max_iter": 50,
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
    value.update(overrides)
    return value


def cluster_local_embedding_config(
    **overrides: Any,
) -> ClusterLocalEmbeddingScientificConfig:
    return ClusterLocalEmbeddingScientificConfig.from_mapping(
        cluster_local_embedding_mapping(**overrides)
    )
