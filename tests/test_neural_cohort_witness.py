import inspect

import numpy as np

from oci.inference.neural_cohort_witness import (
    NeuralCohortWitnessConfig,
    build_ungated_consensus_query_bank,
    build_consensus_witness_bank,
    cohort_contribution,
    direct_target_contribution,
    fit_soft_target_queries,
    fit_soft_witness_queries,
    multiplier_group_score_test,
    soft_retrieval_activations,
    standardized_cohort_moments,
)


def test_binary_direct_target_contribution_recovers_group_mean_contrast():
    target = np.array([0, 0, 1, 1, 1], dtype=float)
    activation = np.array([0.1, 0.3, 0.7, 0.8, 0.9], dtype=float)
    contribution = direct_target_contribution(target, binary=True)
    observed = np.mean(activation * contribution)
    expected = np.mean(activation[target == 1]) - np.mean(activation[target == 0])
    assert abs(observed - expected) < 1e-12
    assert abs(np.mean(contribution)) < 1e-12


def test_small_soft_target_fit_finds_direct_semantic_signal():
    rng = np.random.default_rng(82)
    target = np.tile([0.0, 1.0], 20)
    chunks = []
    for value in target:
        matrix = rng.normal(size=(3, 8)).astype(np.float32)
        if value:
            matrix[0] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            matrix[0] += 0.03 * rng.normal(size=8)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        chunks.append(matrix)
    config = NeuralCohortWitnessConfig(
        n_prototypes=2,
        initial_pool_size=4,
        epochs=20,
        kmeans_iterations=4,
        kmeans_sample_chunks=200,
        max_query_drift=0.40,
        consensus_min_prototypes=1,
        consensus_max_prototypes=2,
    )
    fitted = fit_soft_target_queries(
        chunks,
        target,
        binary=True,
        config=config,
        seed=8,
        device="cpu",
        target_name="treatment",
    )
    assert fitted["queries"].shape == (2, 8)
    assert np.max(np.abs(fitted["train_standardized_scores"])) > 2.0
    assert fitted["no_validation_or_heldout_rows_consumed"] is True


def test_ungated_consensus_assigns_every_candidate_without_score_filtering():
    rng = np.random.default_rng(5)
    first = np.linspace(-1.0, 1.0, 40)
    second = np.sin(np.linspace(0.0, 3.0 * np.pi, 40))
    activations = np.column_stack(
        [
            first + 0.01 * rng.normal(size=40),
            first + 0.01 * rng.normal(size=40),
            second + 0.01 * rng.normal(size=40),
            second + 0.01 * rng.normal(size=40),
        ]
    )
    candidates = [
        {
            "candidate_id": f"candidate_{index}",
            "subfold": index + 1,
            "query": rng.normal(size=6),
            "train_standardized_score": score,
        }
        for index, score in enumerate([0.0, -0.2, 8.0, -9.0])
    ]
    consensus = build_ungated_consensus_query_bank(
        candidates,
        candidate_activations=activations,
        n_queries=2,
        bank="effect",
        seed=3,
        config=NeuralCohortWitnessConfig(
            n_prototypes=2,
            initial_pool_size=4,
            consensus_min_prototypes=2,
            consensus_max_prototypes=2,
        ),
    )
    assert consensus["selected_count"] == 2
    assert consensus["statistical_gate_applied"] is False
    assert consensus["all_candidates_assigned"] is True
    members = [
        member["candidate_id"]
        for record in consensus["records"]
        for member in record["members"]
    ]
    assert sorted(members) == [f"candidate_{index}" for index in range(4)]


def test_cohort_contribution_is_orthogonal_to_fitted_constant_effect():
    treatment_residual = np.array([-0.4, 0.7, -0.3, 0.8, -0.2])
    outcome_residual = np.array([0.2, 0.8, -0.1, 0.3, -0.6])
    contribution, constant = cohort_contribution(
        treatment_residual, outcome_residual
    )
    assert constant != 0.0
    assert abs(np.sum(contribution)) < 1e-12


def test_soft_retrieval_matches_log_mean_exp_over_all_chunks():
    chunks = [
        np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32),
        np.array([[0.9, np.sqrt(0.19)], [0.9, -np.sqrt(0.19)]], dtype=np.float32),
    ]
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    temperature = 0.2
    observed = soft_retrieval_activations(
        chunks,
        query,
        temperature=temperature,
        device="cpu",
        patient_batch_size=2,
    ).ravel()
    expected = []
    for similarities in ([1.0, 0.8], [0.9, 0.9]):
        values = np.asarray(similarities) / temperature
        maximum = np.max(values)
        expected.append(
            temperature
            * (maximum + np.log(np.mean(np.exp(values - maximum))))
        )
    np.testing.assert_allclose(observed, expected, atol=1e-6)
    assert np.all(observed < np.array([1.0, 0.9]) + 1e-7)


def test_witness_training_api_cannot_receive_validation_or_heldout_labels():
    parameters = inspect.signature(fit_soft_witness_queries).parameters
    assert "validation_outcome" not in parameters
    assert "heldout_outcome" not in parameters
    assert "validation_chunks" not in parameters


def test_small_soft_witness_fit_is_constrained_and_finds_cohort_signal():
    rng = np.random.default_rng(12)
    modifier = np.tile([0.0, 1.0], 18)
    chunks = []
    for value in modifier:
        matrix = rng.normal(size=(3, 8)).astype(np.float32)
        if value:
            matrix[0] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            matrix[0] += 0.05 * rng.normal(size=8)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        chunks.append(matrix)
    treatment_residual = rng.normal(size=len(modifier))
    outcome_residual = (
        0.2 * treatment_residual
        + 1.1 * modifier * treatment_residual
        + 0.15 * rng.normal(size=len(modifier))
    )
    config = NeuralCohortWitnessConfig(
        n_prototypes=2,
        initial_pool_size=4,
        epochs=20,
        kmeans_iterations=4,
        kmeans_sample_chunks=200,
        max_query_drift=0.40,
        consensus_min_prototypes=1,
        consensus_max_prototypes=2,
    )
    fitted = fit_soft_witness_queries(
        chunks,
        treatment_residual,
        outcome_residual,
        config=config,
        seed=4,
        device="cpu",
    )
    assert fitted["queries"].shape == (2, 8)
    assert np.max(fitted["query_drift"]) <= config.max_query_drift + 0.01
    assert np.max(np.abs(fitted["train_standardized_scores"])) > 2.0
    assert "patient-level effect target" in fitted["objective"]


def test_consensus_requires_recurrence_before_using_labeled_fallback():
    config = NeuralCohortWitnessConfig(
        n_prototypes=2,
        initial_pool_size=2,
        validation_min_abs_z=1.0,
        consensus_cosine_threshold=0.8,
        consensus_min_subfold_recurrence=2,
        consensus_min_prototypes=1,
        consensus_max_prototypes=2,
    )
    candidates = [
        {
            "candidate_id": "a",
            "subfold": 1,
            "query": np.array([1.0, 0.0, 0.0]),
            "train_standardized_score": 2.5,
            "validation_standardized_score": 2.0,
        },
        {
            "candidate_id": "b",
            "subfold": 2,
            "query": np.array([0.99, 0.05, 0.0]),
            "train_standardized_score": 2.0,
            "validation_standardized_score": 1.8,
        },
        {
            "candidate_id": "c",
            "subfold": 3,
            "query": np.array([0.0, 1.0, 0.0]),
            "train_standardized_score": -3.0,
            "validation_standardized_score": -2.0,
        },
    ]
    consensus = build_consensus_witness_bank(candidates, config=config)
    assert consensus["selected_count"] == 1
    assert consensus["recurrence_fallback_used"] is False
    assert consensus["records"][0]["subfold_recurrence"] == 2


def test_consensus_uses_patient_activations_in_anisotropic_query_space():
    config = NeuralCohortWitnessConfig(
        n_prototypes=2,
        initial_pool_size=2,
        validation_min_abs_z=1.0,
        consensus_activation_correlation_threshold=0.85,
        consensus_min_subfold_recurrence=2,
        consensus_min_prototypes=2,
        consensus_max_prototypes=2,
    )
    # Every query is highly cosine-aligned with the same dominant embedding
    # direction, while its patient activation identifies one of two patterns.
    candidates = [
        {
            "candidate_id": "a1",
            "subfold": 1,
            "query": np.array([1.0, 0.10, 0.00]),
            "train_standardized_score": 2.0,
            "validation_standardized_score": 1.8,
        },
        {
            "candidate_id": "a2",
            "subfold": 2,
            "query": np.array([1.0, 0.11, 0.00]),
            "train_standardized_score": 2.1,
            "validation_standardized_score": 1.7,
        },
        {
            "candidate_id": "b1",
            "subfold": 1,
            "query": np.array([1.0, 0.00, 0.10]),
            "train_standardized_score": 2.2,
            "validation_standardized_score": 1.6,
        },
        {
            "candidate_id": "b2",
            "subfold": 2,
            "query": np.array([1.0, 0.00, 0.11]),
            "train_standardized_score": 2.3,
            "validation_standardized_score": 1.5,
        },
    ]
    pattern_a = np.linspace(-1.0, 1.0, 30)
    pattern_b = np.sin(np.linspace(0.0, 3.0 * np.pi, 30))
    activations = np.column_stack(
        [
            pattern_a,
            pattern_a + 0.01 * np.cos(np.arange(30)),
            pattern_b,
            pattern_b + 0.01 * np.sin(np.arange(30)),
        ]
    )
    consensus = build_consensus_witness_bank(
        candidates,
        config=config,
        candidate_activations=activations,
    )
    assert consensus["selected_count"] == 2
    assert consensus["clustering_similarity"] == (
        "centered_patient_activation_correlation"
    )
    assert [record["strict_subfold_recurrence"] for record in consensus["records"]] == [
        2,
        2,
    ]


def test_group_score_test_is_row_order_invariant():
    rng = np.random.default_rng(7)
    activations = rng.normal(size=(80, 3))
    treatment_residual = rng.normal(size=80)
    outcome_residual = 0.3 * treatment_residual + rng.normal(size=80)
    moments = standardized_cohort_moments(
        activations,
        treatment_residual,
        outcome_residual,
        constant_effect=0.3,
    )
    first = multiplier_group_score_test(
        moments["row_scores"], repeats=500, seed=9, chunk_size=100
    )
    order = rng.permutation(80)
    second = multiplier_group_score_test(
        moments["row_scores"][order], repeats=500, seed=9, chunk_size=100
    )
    assert first["quadratic_statistic"] == second["quadratic_statistic"]
    assert first["maximum_absolute_standardized_score"] == second[
        "maximum_absolute_standardized_score"
    ]
