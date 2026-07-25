import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from oracle_experiment_scripts.audit_tfidf_topic_oracle_recovery import (
    topic_oracle_associations,
)
from oci.config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    ModelArchitectureConfig,
    MultiModelForestConfig,
    TfidfNuisanceStackScientificConfig,
    TfidfTopicDiscoveryConfig,
    BoWViewConfig,
    normalize_tfidf_topic_feature_discovery_methods,
)
from oci.inference.agentic_explicit_feature_forest import (
    SplitEvaluation,
    VLLMExplicitFeatureExtractionProvider,
)
from oci.inference.tfidf_topic_agentic_forest import (
    ORPHAN_NGRAM_LABEL_PROMPT_VERSION,
    TfidfTopicAgenticForestRunner,
    TOPIC_GLOBAL_DEDUP_PROMPT_VERSION,
    TOPIC_NAME_HARMONIZATION_PROMPT_VERSION,
    TOPIC_VALUE_HARMONIZATION_PROMPT_VERSION,
    _effect_evidence_coverage,
    _registry_entry_state_hash,
    apply_registry_derivations,
    apply_topic_global_dedup,
    apply_topic_name_harmonization,
    apply_topic_value_harmonization,
    build_topic_global_dedup_blocks,
    build_topic_label_context,
    harmonize_topic_candidates,
    parsimony_replacement_passes,
    render_topic_label_prompt,
    select_deferred_review_additions,
    select_initial_topic_evidence_registry,
    select_topic_recovery_raw_ngrams,
    structured_review_gate,
    topic_harmonization_response_issues,
    run_tfidf_topic_agentic_forest,
    validate_tfidf_topic_stage2_handoff,
)
from oci.inference.tfidf_topic_discovery import (
    ConsensusNMFTopicBank,
    add_effect_stability,
    align_topic_components,
    cohort_contrast_scores,
    fit_tfidf_topic_context,
    fit_cross_fitted_nuisance_stack,
    unsigned_linear_screen,
)
from oci.inference.tfidf_safe_artifacts import load_named_array_bank
from oci.inference.tfidf_topic_score_selection import (
    benjamini_hochberg,
    build_fit_side_orphan_ngram_clusters,
    reselect_persisted_topic_scores,
    score_effect_orphan_ngram_clusters,
    score_topic_banks,
)
from oci.inference.tfidf_topic_stage1 import run_tfidf_topic_stage1


def _dense_contrast_reference(x, t, y, e, m):
    u = t - e
    v = y - m
    constant = np.dot(u, v) / np.dot(u, u)
    contribution = u * (v - constant * u)
    z = x * contribution[:, None]
    moment = z.mean(axis=0)
    se = np.sqrt(np.maximum(0.0, np.mean(z**2, axis=0) - moment**2) / len(t))
    score = np.divide(moment, se, out=np.zeros_like(moment), where=se > 0)
    return moment, se, score


def test_posthoc_topic_oracle_associations_are_absolute_and_row_order_invariant():
    topics = np.asarray(
        [
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 4.0],
            [2.0, 1.0, 4.0],
            [3.0, 0.0, 4.0],
            [4.0, 1.0, 4.0],
            [5.0, 0.0, 4.0],
        ]
    )
    oracle = np.column_stack([topics[:, 0], 1.0 - topics[:, 1]])
    observed = topic_oracle_associations(topics, oracle)
    assert observed[0] == pytest.approx(1.0)
    assert observed[1] == pytest.approx(1.0)
    assert observed[2] == pytest.approx(0.0)
    order = np.asarray([4, 1, 5, 0, 3, 2])
    np.testing.assert_allclose(
        topic_oracle_associations(topics[order], oracle[order]), observed
    )


@pytest.mark.parametrize("continuous", [False, True])
def test_cohort_contrast_matches_dense_reference_and_is_order_invariant(continuous):
    rng = np.random.default_rng(4)
    x = (rng.random((60, 7)) > 0.65).astype(float) * rng.uniform(0.2, 1.0, (60, 7))
    t = np.tile([0.0, 1.0], 30)
    y = rng.normal(size=60) if continuous else (rng.random(60) > 0.5).astype(float)
    e = np.clip(rng.uniform(0.15, 0.85, 60), 0.01, 0.99)
    m = rng.normal(size=60) if continuous else rng.uniform(0.1, 0.9, 60)
    expected = _dense_contrast_reference(x, t, y, e, m)
    observed = cohort_contrast_scores(x, [f"f{i}" for i in range(7)], t, y, e, m)
    np.testing.assert_allclose(observed["moment"], expected[0])
    np.testing.assert_allclose(observed["robust_se"], expected[1])
    np.testing.assert_allclose(observed["signed_score"], expected[2])
    order = rng.permutation(len(t))
    reordered = cohort_contrast_scores(
        x[order], [f"f{i}" for i in range(7)], t[order], y[order], e[order], m[order]
    )
    np.testing.assert_allclose(reordered["signed_score"], observed["signed_score"])


def test_unsigned_screen_keeps_equal_positive_and_negative_coefficients():
    x = sparse.csr_matrix(
        np.column_stack(
            [np.tile([0.0, 1.0], 80), np.tile([1.0, 0.0], 80), np.zeros(160)]
        )
    )
    y = np.tile([0.0, 1.0], 80)
    result = unsigned_linear_screen(
        x,
        ["positive", "negative", "zero"],
        y,
        binary=True,
        view=BoWViewConfig(
            name="linear_1_3",
            min_df=1,
            max_df=1.0,
            ngram_range_min=1,
            ngram_range_max=3,
        ),
        random_state=42,
    )
    by_name = result.set_index("feature")
    assert by_name.loc["positive", "signed_score"] > 0
    assert by_name.loc["negative", "signed_score"] < 0
    assert by_name.loc["positive", "unsigned_score"] == pytest.approx(
        by_name.loc["negative", "unsigned_score"]
    )


def _score_test_topic(topic_id, terms):
    return {
        "topic_id": topic_id,
        "terms": [
            {
                "term": term,
                "loading": 1.0 / (index + 1),
                "screen_rank": index + 1,
                "signed_score": 1.0,
            }
            for index, term in enumerate(terms)
        ],
    }


def _synthetic_topic_score_inputs(effect_direction=1.0):
    rng = np.random.default_rng(718)
    n_fit = 500
    n_heldout = 400
    confounder_fit = rng.binomial(1, 0.5, n_fit)
    confounder_heldout = rng.binomial(1, 0.5, n_heldout)
    modifier_fit = rng.binomial(1, 0.5, n_fit)
    modifier_heldout = rng.binomial(1, 0.5, n_heldout)

    def feature_block(indicator, n_rows):
        present = rng.binomial(1, 0.88, (n_rows, 15)) * indicator[:, None]
        return present * rng.uniform(0.6, 1.4, (n_rows, 15))

    confounder_terms = [f"confounder_term_{index}" for index in range(15)]
    modifier_terms = [f"modifier_term_{index}" for index in range(15)]
    noise_terms = [f"noise_term_{index}" for index in range(15)]
    feature_names = [*confounder_terms, *modifier_terms, *noise_terms]
    fit_matrix = sparse.csr_matrix(
        np.column_stack(
            [
                feature_block(confounder_fit, n_fit),
                feature_block(modifier_fit, n_fit),
                rng.binomial(1, 0.20, (n_fit, 15))
                * rng.uniform(0.4, 1.0, (n_fit, 15)),
            ]
        )
    )
    heldout_matrix = sparse.csr_matrix(
        np.column_stack(
            [
                feature_block(confounder_heldout, n_heldout),
                feature_block(modifier_heldout, n_heldout),
                rng.binomial(1, 0.20, (n_heldout, 15))
                * rng.uniform(0.4, 1.0, (n_heldout, 15)),
            ]
        )
    )

    propensity_fit = 1.0 / (1.0 + np.exp(-(-1.0 + 2.2 * confounder_fit)))
    propensity_heldout = 1.0 / (
        1.0 + np.exp(-(-1.0 + 2.2 * confounder_heldout))
    )
    treatment_fit = rng.binomial(1, propensity_fit).astype(float)
    treatment_heldout = rng.binomial(1, propensity_heldout).astype(float)
    tau_fit = 0.1 + effect_direction * 1.5 * modifier_fit
    tau_heldout = 0.1 + effect_direction * 1.5 * modifier_heldout
    baseline_fit = 0.9 * confounder_fit
    baseline_heldout = 0.9 * confounder_heldout
    outcome_fit = (
        baseline_fit
        + tau_fit * treatment_fit
        + rng.normal(0.0, 0.25, n_fit)
    )
    outcome_heldout = (
        baseline_heldout
        + tau_heldout * treatment_heldout
        + rng.normal(0.0, 0.25, n_heldout)
    )
    outcome_prediction_fit = baseline_fit + propensity_fit * tau_fit
    outcome_prediction_heldout = baseline_heldout + propensity_heldout * tau_heldout

    topic_banks = {
        "treatment": {
            "topics": [
                _score_test_topic("treatment_signal", confounder_terms),
                _score_test_topic("treatment_noise", noise_terms),
            ]
        },
        "outcome": {
            "topics": [
                _score_test_topic("outcome_signal", confounder_terms),
                _score_test_topic("outcome_noise", noise_terms),
            ]
        },
        "effect": {
            "topics": [
                _score_test_topic("effect_signal", modifier_terms),
                _score_test_topic("effect_noise", noise_terms),
            ]
        },
    }
    fit_dense = fit_matrix.toarray()
    heldout_dense = heldout_matrix.toarray()
    fit_confounder_topic = np.mean(fit_dense[:, :15], axis=1)
    heldout_confounder_topic = np.mean(heldout_dense[:, :15], axis=1)
    fit_modifier_topic = np.mean(fit_dense[:, 15:30], axis=1)
    heldout_modifier_topic = np.mean(heldout_dense[:, 15:30], axis=1)
    fit_noise_topic = np.mean(fit_dense[:, 30:45], axis=1)
    heldout_noise_topic = np.mean(heldout_dense[:, 30:45], axis=1)
    return {
        "fit_matrix": fit_matrix,
        "heldout_matrix": heldout_matrix,
        "feature_names": feature_names,
        "topic_banks": topic_banks,
        "fit_topic_values": {
            "treatment": np.column_stack(
                [fit_confounder_topic, fit_noise_topic]
            ),
            "outcome": np.column_stack(
                [fit_confounder_topic, fit_noise_topic]
            ),
            "effect": np.column_stack([fit_modifier_topic, fit_noise_topic]),
        },
        "heldout_topic_values": {
            "treatment": np.column_stack(
                [heldout_confounder_topic, heldout_noise_topic]
            ),
            "outcome": np.column_stack(
                [heldout_confounder_topic, heldout_noise_topic]
            ),
            "effect": np.column_stack(
                [heldout_modifier_topic, heldout_noise_topic]
            ),
        },
        "fit_treatment": treatment_fit,
        "fit_outcome": outcome_fit,
        "heldout_treatment": treatment_heldout,
        "heldout_outcome": outcome_heldout,
        "fit_propensity": propensity_fit,
        "fit_outcome_prediction": outcome_prediction_fit,
        "heldout_propensity": propensity_heldout,
        "heldout_outcome_prediction": outcome_prediction_heldout,
    }


@pytest.mark.parametrize("effect_direction", [-1.0, 1.0])
def test_all_topic_group_score_tests_find_nuisance_and_effect_signal(effect_direction):
    inputs = _synthetic_topic_score_inputs(effect_direction)
    config = TfidfTopicDiscoveryConfig(
        min_df=1,
        topic_count=2,
        stability_repeats=0,
        score_test_bootstrap_repeats=199,
        score_test_bootstrap_top_topics=0,
        score_test_bootstrap_chunk_size=40,
        score_test_min_topics_per_bank=1,
        score_test_max_topics_per_bank=1,
    )
    result = score_topic_banks(
        **inputs,
        config=config,
        scope_id=f"synthetic_direction_{effect_direction}",
    )
    expected = {
        "treatment": "treatment_signal",
        "outcome": "outcome_signal",
        "effect": "effect_signal",
    }
    for bank, topic_id in expected.items():
        bank_result = result["banks"][bank]
        assert bank_result["bootstrap_calibration"]["complete_topic_family"]
        assert bank_result["bootstrap_calibration"]["complete_ngram_family"]
        assert bank_result["bootstrap_calibration"]["bootstrapped_topic_count"] == 2
        assert bank_result["unique_ngram_count"] == 30
        assert bank_result["testable_unique_ngram_count"] == 30
        assert bank_result["ngram_selection_count"] > 0
        by_id = {row["topic_id"]: row for row in bank_result["topic_tests"]}
        assert by_id[topic_id]["evidence_rank"] == 1
        assert by_id[topic_id]["selected_for_agent"]
        assert by_id[topic_id]["primary_p_source"] == (
            "complete_family_multiplier_bootstrap"
        )
        assert "topic_familywise_p" in by_id[topic_id]
        assert abs(by_id[topic_id]["topic_standardized_score"]) > 0
        assert "candidate_familywise_energy_p" in by_id[topic_id]
        assert by_id[topic_id]["selected_ngram_count"] > 0
        assert all(
            "ngram_fdr_q" in term and "ngram_evidence_rank" in term
            for term in by_id[topic_id]["term_scores"]
        )
        assert bank_result["selected_topic_ids"] == [topic_id]


def _synthetic_orphan_branch_inputs():
    inputs = _synthetic_topic_score_inputs(effect_direction=1.0)
    fit_base = inputs["fit_matrix"].toarray()
    heldout_base = inputs["heldout_matrix"].toarray()
    # These phrases deliberately encode the same patient signal as a modifier
    # topic while remaining outside every topic's supplied 15-term summary.
    fit_signal = fit_base[:, 15]
    heldout_signal = heldout_base[:, 15]
    rng = np.random.default_rng(922)
    fit_noise = rng.binomial(1, 0.25, len(fit_signal))
    heldout_noise = rng.binomial(1, 0.25, len(heldout_signal))
    orphan_terms = [
        "calculated nlr",
        "baseline calculated nlr",
        "low lymphocytes",
        "administrative date",
    ]
    inputs["fit_matrix"] = sparse.csr_matrix(
        np.column_stack(
            [fit_base, fit_signal, fit_signal, fit_signal, fit_noise]
        )
    )
    inputs["heldout_matrix"] = sparse.csr_matrix(
        np.column_stack(
            [
                heldout_base,
                heldout_signal,
                heldout_signal,
                heldout_signal,
                heldout_noise,
            ]
        )
    )
    inputs["feature_names"] = [*inputs["feature_names"], *orphan_terms]
    scores = pd.DataFrame(
        {
            "feature": inputs["feature_names"],
            "signed_score": [0.0] * 45 + [3.2, 3.1, -2.9, 2.1],
            "unsigned_score": [0.0] * 45 + [3.2, 3.1, 2.9, 2.1],
            "combined_importance": [0.0] * 45 + [3.1, 3.0, 2.8, 2.0],
            "eligible": [True] * 49,
            "support_control": [20] * 49,
            "support_treated": [20] * 49,
            "nuisance_source_agreement": [1.0] * 49,
            "subsample_selection_stability": [1.0] * 49,
            "subsample_sign_agreement": [1.0] * 49,
            "tail_contrast_sign_agreement": [1.0] * 49,
        }
    )
    return inputs, scores


def test_orphan_branch_deduplicates_nested_phrases_clusters_and_selects_signal():
    inputs, scores = _synthetic_orphan_branch_inputs()
    config = TfidfTopicDiscoveryConfig(
        min_df=1,
        topic_count=2,
        stability_repeats=0,
        score_test_bootstrap_repeats=199,
        score_test_bootstrap_top_topics=0,
        score_test_bootstrap_chunk_size=40,
        orphan_ngram_min_abs_fit_score=2.0,
        orphan_ngram_cluster_similarity_threshold=0.30,
        orphan_ngram_max_selected_clusters=5,
    )
    represented = [
        term["term"]
        for topic in inputs["topic_banks"]["effect"]["topics"]
        for term in topic["terms"]
    ]
    universe = build_fit_side_orphan_ngram_clusters(
        fit_matrix=inputs["fit_matrix"],
        feature_names=inputs["feature_names"],
        effect_scores=scores,
        represented_topic_terms=represented,
        config=config,
    )
    assert universe["candidate_count_before_nested_deduplication"] == 4
    assert universe["deduplicated_alias_count"] >= 1
    assert all(1 <= len(cluster["terms"]) <= 15 for cluster in universe["clusters"])

    result = score_effect_orphan_ngram_clusters(
        fit_matrix=inputs["fit_matrix"],
        heldout_matrix=inputs["heldout_matrix"],
        feature_names=inputs["feature_names"],
        effect_scores=scores,
        effect_topics=inputs["topic_banks"]["effect"]["topics"],
        fit_treatment=inputs["fit_treatment"],
        fit_outcome=inputs["fit_outcome"],
        heldout_treatment=inputs["heldout_treatment"],
        heldout_outcome=inputs["heldout_outcome"],
        fit_propensity=inputs["fit_propensity"],
        fit_outcome_prediction=inputs["fit_outcome_prediction"],
        heldout_propensity=inputs["heldout_propensity"],
        heldout_outcome_prediction=inputs["heldout_outcome_prediction"],
        config=config,
    )
    assert result["status"] == "completed"
    assert not result["cluster_construction_uses_heldout_rows_or_labels"]
    assert result["bootstrap_calibration"]["complete_term_group_family"]
    assert result["selection_count"] >= 1
    selected_terms = {
        term["term"]
        for cluster in result["selected_clusters"]
        for term in cluster["term_scores"]
    }
    assert selected_terms & {
        "calculated nlr",
        "baseline calculated nlr",
        "low lymphocytes",
    }
    assert not selected_terms & set(represented)


def test_orphan_cluster_observed_score_is_heldout_row_order_invariant():
    inputs, scores = _synthetic_orphan_branch_inputs()
    config = TfidfTopicDiscoveryConfig(
        min_df=1,
        topic_count=2,
        stability_repeats=0,
        score_test_bootstrap_repeats=0,
        orphan_ngram_max_selected_clusters=5,
    )

    def run(payload):
        return score_effect_orphan_ngram_clusters(
            fit_matrix=payload["fit_matrix"],
            heldout_matrix=payload["heldout_matrix"],
            feature_names=payload["feature_names"],
            effect_scores=scores,
            effect_topics=payload["topic_banks"]["effect"]["topics"],
            fit_treatment=payload["fit_treatment"],
            fit_outcome=payload["fit_outcome"],
            heldout_treatment=payload["heldout_treatment"],
            heldout_outcome=payload["heldout_outcome"],
            fit_propensity=payload["fit_propensity"],
            fit_outcome_prediction=payload["fit_outcome_prediction"],
            heldout_propensity=payload["heldout_propensity"],
            heldout_outcome_prediction=payload["heldout_outcome_prediction"],
            config=config,
        )

    first = run(inputs)
    order = np.random.default_rng(72).permutation(len(inputs["heldout_treatment"]))
    reordered = dict(inputs)
    reordered["heldout_matrix"] = inputs["heldout_matrix"][order]
    for key in (
        "heldout_treatment",
        "heldout_outcome",
        "heldout_propensity",
        "heldout_outcome_prediction",
    ):
        reordered[key] = np.asarray(inputs[key])[order]
    second = run(reordered)
    first_by_id = {row["cluster_id"]: row for row in first["clusters"]}
    second_by_id = {row["cluster_id"]: row for row in second["clusters"]}
    assert set(first_by_id) == set(second_by_id)
    for cluster_id in first_by_id:
        assert first_by_id[cluster_id]["quadratic_statistic"] == pytest.approx(
            second_by_id[cluster_id]["quadratic_statistic"]
        )


def test_ngram_score_family_deduplicates_terms_and_keeps_topic_provenance():
    inputs = _synthetic_topic_score_inputs()
    duplicate = inputs["topic_banks"]["treatment"]["topics"][0]["terms"][0][
        "term"
    ]
    inputs["topic_banks"]["treatment"]["topics"][1]["terms"][0]["term"] = (
        duplicate
    )
    result = score_topic_banks(
        **inputs,
        config=TfidfTopicDiscoveryConfig(
            min_df=1,
            topic_count=2,
            stability_repeats=0,
            score_test_bootstrap_repeats=49,
            score_test_bootstrap_top_topics=0,
            score_test_bootstrap_chunk_size=20,
            score_test_min_topics_per_bank=1,
            score_test_max_topics_per_bank=2,
        ),
        scope_id="duplicate_ngram_provenance",
    )
    treatment = result["banks"]["treatment"]
    matches = [
        row for row in treatment["ngram_tests"] if row["term"] == duplicate
    ]
    assert len(matches) == 1
    assert matches[0]["topic_ids"] == ["treatment_noise", "treatment_signal"]
    assert matches[0]["topic_occurrence_count"] == 2
    assert treatment["unique_ngram_count"] == 29
    occurrences = [
        term
        for topic in treatment["topic_tests"]
        for term in topic["term_scores"]
        if term["term"] == duplicate
    ]
    assert len(occurrences) == 2
    assert {term["ngram_fdr_q"] for term in occurrences} == {
        matches[0]["fdr_q"]
    }


def test_scalar_nmf_topic_route_survives_an_untestable_joint_term_group():
    inputs = _synthetic_topic_score_inputs()
    inputs["heldout_matrix"] = sparse.csr_matrix(
        inputs["heldout_matrix"].shape, dtype=float
    )
    result = score_topic_banks(
        **inputs,
        config=TfidfTopicDiscoveryConfig(
            min_df=1,
            topic_count=2,
            stability_repeats=0,
            score_test_bootstrap_repeats=199,
            score_test_bootstrap_top_topics=0,
            score_test_min_topics_per_bank=0,
            score_test_max_topics_per_bank=1,
        ),
        scope_id="scalar_topic_only",
    )
    expected = {
        "treatment": "treatment_signal",
        "outcome": "outcome_signal",
        "effect": "effect_signal",
    }
    for bank, topic_id in expected.items():
        by_id = {
            row["topic_id"]: row
            for row in result["banks"][bank]["topic_tests"]
        }
        assert by_id[topic_id]["topic_score_testable"]
        assert by_id[topic_id]["quadratic_covariance_rank"] == 0
        assert by_id[topic_id]["selected_for_agent"]
        assert by_id[topic_id]["selection_reason"] == "nmf_topic_score_evidence"


def test_joint_term_group_route_survives_an_untestable_scalar_topic_score():
    inputs = _synthetic_topic_score_inputs()
    inputs["heldout_topic_values"] = {
        bank: np.zeros_like(values)
        for bank, values in inputs["heldout_topic_values"].items()
    }
    result = score_topic_banks(
        **inputs,
        config=TfidfTopicDiscoveryConfig(
            min_df=1,
            topic_count=2,
            stability_repeats=0,
            score_test_bootstrap_repeats=199,
            score_test_bootstrap_top_topics=0,
            score_test_min_topics_per_bank=0,
            score_test_max_topics_per_bank=1,
        ),
        scope_id="joint_term_group_only",
    )
    expected = {
        "treatment": "treatment_signal",
        "outcome": "outcome_signal",
        "effect": "effect_signal",
    }
    for bank, topic_id in expected.items():
        by_id = {
            row["topic_id"]: row
            for row in result["banks"][bank]["topic_tests"]
        }
        assert not by_id[topic_id]["topic_score_testable"]
        assert by_id[topic_id]["quadratic_covariance_rank"] > 0
        assert by_id[topic_id]["selected_for_agent"]
        assert by_id[topic_id]["term_group_primary_p_source"] == (
            "complete_family_multiplier_bootstrap"
        )


def test_persisted_score_reselection_changes_policy_without_recomputing_statistics():
    inputs = _synthetic_topic_score_inputs()
    config = TfidfTopicDiscoveryConfig(
        min_df=1,
        topic_count=2,
        stability_repeats=0,
        score_test_bootstrap_repeats=49,
        score_test_bootstrap_top_topics=0,
        score_test_min_topics_per_bank=1,
        score_test_max_topics_per_bank=2,
    )
    original = score_topic_banks(
        **inputs, config=config, scope_id="persisted_reselection"
    )
    original_statistic = original["banks"]["effect"]["topic_tests"][0][
        "quadratic_statistic"
    ]
    migrated = reselect_persisted_topic_scores(original, config)

    assert migrated["selection_recomputed_without_labels"]
    assert not migrated["score_statistics_recomputed"]
    assert migrated["banks"]["effect"]["topic_tests"][0][
        "quadratic_statistic"
    ] == pytest.approx(original_statistic)
    assert "term_group_fdr_q" in migrated["banks"]["effect"]["topic_tests"][0]


def test_persisted_orphan_reselection_changes_only_the_bounded_shortlist():
    inputs, scores = _synthetic_orphan_branch_inputs()
    original_config = TfidfTopicDiscoveryConfig(
        min_df=1,
        topic_count=2,
        stability_repeats=0,
        score_test_bootstrap_repeats=0,
        score_test_min_topics_per_bank=1,
        score_test_max_topics_per_bank=2,
        orphan_ngram_min_abs_fit_score=2.0,
        orphan_ngram_cluster_similarity_threshold=0.30,
        orphan_ngram_min_selected_clusters=1,
        orphan_ngram_max_selected_clusters=1,
    )
    original = score_topic_banks(
        **inputs,
        config=original_config,
        scope_id="orphan_reselection",
        raw_ngram_scores={"effect": scores},
    )
    orphan = original["effect_orphan_ngram_branch"]
    assert orphan["cluster_count"] >= 2
    original_statistics = {
        row["cluster_id"]: (
            row["quadratic_statistic"],
            row["primary_p"],
            row["fdr_q"],
        )
        for row in orphan["clusters"]
    }

    migrated = reselect_persisted_topic_scores(
        original,
        TfidfTopicDiscoveryConfig(
            min_df=1,
            topic_count=2,
            stability_repeats=0,
            score_test_bootstrap_repeats=0,
            score_test_min_topics_per_bank=1,
            score_test_max_topics_per_bank=2,
            orphan_ngram_min_abs_fit_score=2.0,
            orphan_ngram_cluster_similarity_threshold=0.30,
            orphan_ngram_min_selected_clusters=2,
            orphan_ngram_max_selected_clusters=2,
        ),
    )
    migrated_orphan = migrated["effect_orphan_ngram_branch"]

    assert migrated_orphan["selection_count"] == 2
    assert len(migrated_orphan["selected_cluster_ids"]) == 2
    assert migrated_orphan["selection_recomputed_without_labels"]
    assert not migrated_orphan["score_statistics_recomputed"]
    assert {
        row["cluster_id"]: (
            row["quadratic_statistic"],
            row["primary_p"],
            row["fdr_q"],
        )
        for row in migrated_orphan["clusters"]
    } == original_statistics


def test_topic_group_score_observed_statistics_are_row_order_invariant():
    inputs = _synthetic_topic_score_inputs()
    config = TfidfTopicDiscoveryConfig(
        min_df=1,
        topic_count=2,
        stability_repeats=0,
        score_test_bootstrap_repeats=0,
        score_test_min_topics_per_bank=1,
        score_test_max_topics_per_bank=1,
    )
    first = score_topic_banks(**inputs, config=config, scope_id="first")
    order = np.random.default_rng(55).permutation(len(inputs["heldout_treatment"]))
    reordered = dict(inputs)
    reordered["heldout_matrix"] = inputs["heldout_matrix"][order]
    for key in (
        "heldout_treatment",
        "heldout_outcome",
        "heldout_propensity",
        "heldout_outcome_prediction",
    ):
        reordered[key] = inputs[key][order]
    reordered["heldout_topic_values"] = {
        bank: values[order]
        for bank, values in inputs["heldout_topic_values"].items()
    }
    second = score_topic_banks(**reordered, config=config, scope_id="second")
    for bank in ("treatment", "outcome", "effect"):
        first_by_id = {
            row["topic_id"]: row for row in first["banks"][bank]["topic_tests"]
        }
        second_by_id = {
            row["topic_id"]: row for row in second["banks"][bank]["topic_tests"]
        }
        for topic_id in first_by_id:
            assert first_by_id[topic_id]["topic_standardized_score"] == pytest.approx(
                second_by_id[topic_id]["topic_standardized_score"], rel=1e-10
            )
            assert first_by_id[topic_id]["quadratic_statistic"] == pytest.approx(
                second_by_id[topic_id]["quadratic_statistic"], rel=1e-10
            )
        first_ngrams = {
            row["term"]: row for row in first["banks"][bank]["ngram_tests"]
        }
        second_ngrams = {
            row["term"]: row for row in second["banks"][bank]["ngram_tests"]
        }
        assert first_ngrams.keys() == second_ngrams.keys()
        for term in first_ngrams:
            assert first_ngrams[term]["heldout_standardized_score"] == pytest.approx(
                second_ngrams[term]["heldout_standardized_score"], rel=1e-10
            )


def test_limited_topic_bootstrap_cannot_drive_post_selected_familywise_decision():
    inputs = _synthetic_topic_score_inputs()
    config = TfidfTopicDiscoveryConfig(
        min_df=1,
        topic_count=2,
        stability_repeats=0,
        score_test_bootstrap_repeats=99,
        score_test_bootstrap_top_topics=1,
        score_test_bootstrap_chunk_size=25,
        score_test_fdr_level=0.0,
        score_test_p_threshold=0.10,
        score_test_min_topics_per_bank=0,
        score_test_max_topics_per_bank=1,
    )
    result = score_topic_banks(
        **inputs,
        config=config,
        scope_id="limited_bootstrap_audit",
    )

    for bank in ("treatment", "outcome", "effect"):
        bank_result = result["banks"][bank]
        assert not bank_result["bootstrap_calibration"]["complete_topic_family"]
        assert not bank_result["bootstrap_calibration"]["complete_ngram_family"]
        assert {
            row["primary_p_source"] for row in bank_result["topic_tests"]
        } == {"asymptotic_nmf_topic_score"}
        assert {
            row["primary_p_source"] for row in bank_result["ngram_tests"]
        } == {"asymptotic_complete_unique_ngram_family"}
        assert all(
            row["selection_reason"] != "ngram_familywise_p"
            for row in bank_result["ngram_tests"]
        )
        assert all(
            row["selection_reason"] != "topic_familywise_p"
            for row in bank_result["topic_tests"]
        )


def test_minimum_topic_fallback_never_selects_untestable_topic():
    inputs = _synthetic_topic_score_inputs()
    inputs["heldout_matrix"] = sparse.csr_matrix(
        inputs["heldout_matrix"].shape, dtype=float
    )
    inputs["heldout_topic_values"] = {
        bank: np.zeros_like(values)
        for bank, values in inputs["heldout_topic_values"].items()
    }
    result = score_topic_banks(
        **inputs,
        config=TfidfTopicDiscoveryConfig(
            min_df=1,
            topic_count=2,
            stability_repeats=0,
            score_test_bootstrap_repeats=19,
            score_test_min_topics_per_bank=2,
            score_test_max_topics_per_bank=2,
        ),
        scope_id="untestable_topics",
    )

    for bank in ("treatment", "outcome", "effect"):
        assert result["banks"][bank]["selected_topic_ids"] == []
        assert result["banks"][bank]["selection_count"] == 0
        assert all(
            not row["selected_for_agent"]
            for row in result["banks"][bank]["topic_tests"]
        )


def test_benjamini_hochberg_is_monotone_in_ranked_p_values():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    np.testing.assert_allclose(adjusted, [0.04, 0.0533333333, 0.0533333333, 0.20])


def test_effect_support_and_stability_filters_are_applied():
    rng = np.random.default_rng(7)
    x = sparse.csr_matrix(
        np.column_stack(
            [
                np.ones(80),
                np.tile([1.0, 0.0], 40),
                rng.integers(0, 2, 80),
            ]
        )
    )
    t = np.tile([0.0, 1.0], 40)
    y = np.where(x[:, 2].toarray().ravel() > 0, t, 1 - t)
    e = np.full(80, 0.5)
    m = np.full(80, 0.5)
    base = cohort_contrast_scores(x, ["common", "one_arm", "signal"], t, y, e, m)
    config = TfidfTopicDiscoveryConfig(
        min_df=1,
        topic_count=2,
        stability_repeats=4,
        minimum_arm_document_support=3,
        minimum_subsample_selection_fraction=0.0,
        minimum_nuisance_source_agreement=0.0,
        minimum_tail_sign_agreement=0.0,
    )
    result = add_effect_stability(
        base,
        x,
        t,
        y,
        e,
        m,
        [(e, m)],
        strata=(2 * t + y).astype(int),
        config=config,
        random_state=5,
    ).set_index("feature")
    assert not bool(result.loc["one_arm", "eligible"])
    assert "nuisance_source_agreement" in result
    assert "subsample_selection_stability" in result
    assert "tail_contrast_sign_agreement" in result


def test_consensus_nmf_reproducible_aligns_and_transforms_without_refit(monkeypatch):
    rng = np.random.default_rng(11)
    matrix = sparse.csr_matrix(rng.gamma(2.0, 1.0, size=(30, 20)))
    names = [f"term_{index}" for index in range(20)]
    scores = pd.DataFrame(
        {
            "feature": names,
            "signed_score": np.linspace(-2, 2, 20),
            "unsigned_score": np.linspace(2, 1, 20),
            "combined_importance": np.linspace(2, 1, 20),
            "eligible": True,
        }
    )
    config = TfidfTopicDiscoveryConfig(
        min_df=1,
        topic_count=4,
        topic_seeds=[42, 43, 44],
        stability_repeats=0,
        nmf_max_iter=100,
    )
    first, first_values = ConsensusNMFTopicBank.fit(
        bank_name="effect", matrix=matrix, feature_names=names, scores=scores, config=config
    )
    second, second_values = ConsensusNMFTopicBank.fit(
        bank_name="effect", matrix=matrix, feature_names=names, scores=scores, config=config
    )
    np.testing.assert_allclose(first_values, second_values)
    np.testing.assert_allclose(first.consensus_loadings, second.consensus_loadings)
    reference = np.eye(4)
    candidate = reference[[2, 0, 3, 1]]
    np.testing.assert_array_equal(align_topic_components(reference, candidate), [1, 3, 0, 2])
    for model in first.models:
        monkeypatch.setattr(model, "fit_transform", lambda *_args, **_kwargs: pytest.fail("refit"))
    assert first.transform(matrix[:3]).shape == (3, 4)
    assert all(len(topic) == 15 for topic in first.topic_terms)


def test_consensus_nmf_reduces_components_for_small_context():
    matrix = sparse.csr_matrix(np.arange(24, dtype=float).reshape(4, 6) + 1)
    names = [f"t{i}" for i in range(6)]
    scores = pd.DataFrame(
        {
            "feature": names,
            "signed_score": 1.0,
            "unsigned_score": 1.0,
            "combined_importance": 1.0,
            "eligible": True,
        }
    )
    bank, _ = ConsensusNMFTopicBank.fit(
        bank_name="treatment",
        matrix=matrix,
        feature_names=names,
        scores=scores,
        config=TfidfTopicDiscoveryConfig(
            min_df=1,
            topic_count=100,
            terms_per_topic=6,
            stability_repeats=0,
        ),
    )
    assert bank.actual_components == 3
    assert bank.reduction_reason


def test_topic_prompt_has_exact_terms_traceability_and_no_forbidden_language():
    topic = {
        "topic_id": "effect_topic_001",
        "terms_per_topic": 15,
        "terms": [
            {"term": f"term {index}", "loading": 1 / (index + 1), "screen_rank": index + 1,
             "signed_score": (-1) ** index}
            for index in range(15)
        ],
    }
    context = build_topic_label_context(
        outer_fold=1,
        scope="full_outer_train",
        inner_fold=None,
        bank="effect",
        topic=topic,
        terms_per_topic=15,
        score_test_evidence={
            "source": "fixed_inner_score_test_policy_mapping",
            "uses_outer_heldout_labels": False,
            "matched_inner_evidence": [
                {
                    "inner_fold": 1,
                    "primary_p": 0.02,
                    "strongest_term_scores": [
                        {
                            "term": "term 0",
                            "heldout_standardized_score": 2.4,
                        }
                    ],
                }
            ],
        },
    )
    prompt = render_topic_label_prompt(context)
    assert len(context["topic_terms"]) == 15
    assert "supporting_terms" in prompt
    assert "heldout_relevance_evidence" in prompt
    assert "fixed_inner_score_test_policy_mapping" in prompt
    assert "causal" not in prompt.lower()
    assert "administrative" in prompt.lower()


def test_topic_term_evidence_capacity_is_configured_and_not_sliced():
    configured_capacity = 7
    topic = {
        "topic_id": "effect_topic_configured_capacity",
        "terms_per_topic": configured_capacity,
        "terms": [
            {
                "term": f"configured term {index}",
                "loading": 1.0 / (index + 1),
                "screen_rank": index + 1,
                "signed_score": float(index),
            }
            for index in range(configured_capacity)
        ],
    }
    context = build_topic_label_context(
        outer_fold=1,
        scope="candidate_selection_inner_fit",
        inner_fold=1,
        bank="effect",
        topic=topic,
        terms_per_topic=configured_capacity,
    )
    prompt = render_topic_label_prompt(context)

    assert context["terms_per_topic"] == configured_capacity
    assert len(context["topic_terms"]) == configured_capacity
    assert "configured term 6" in prompt
    with pytest.raises(ValueError, match="complete configured term set"):
        render_topic_label_prompt(
            {
                **context,
                "topic_terms": context["topic_terms"][:-1],
            }
        )


def test_orphan_ngram_prompt_is_bounded_traceable_and_not_a_topic_padding_hack():
    topic = {
        "topic_id": "effect_orphan_cluster_001",
        "evidence_kind": "orphan_raw_ngram_cluster",
        "terms": [
            {
                "term": "calculated nlr",
                "fit_rank": 623,
                "fit_signed_score": 2.14,
            },
            {
                "term": "low lymphocytes",
                "fit_rank": 575,
                "fit_signed_score": -2.15,
            },
        ],
    }
    context = build_topic_label_context(
        outer_fold=1,
        scope="candidate_selection_inner_fit",
        inner_fold=1,
        bank="effect",
        topic=topic,
        prompt_version=ORPHAN_NGRAM_LABEL_PROMPT_VERSION,
        score_test_evidence={"primary_p": 0.03},
    )
    prompt = render_topic_label_prompt(context)
    assert len(context["topic_terms"]) == 2
    assert "calculated nlr" in prompt
    assert "low lymphocytes" in prompt
    assert "not represented in the fitted topic summaries" in prompt
    assert "causal" not in prompt.lower()


def test_full_outer_orphan_jobs_are_mapped_from_fixed_inner_policy_without_labels(
    tmp_path,
):
    raw_path = tmp_path / "effect_scores.parquet"
    pd.DataFrame(
        {
            "feature": [
                "topic represented phrase",
                "squamous nsclc",
                "frequent mitoses",
            ],
            "signed_score": [3.5, 3.1, 2.8],
            "unsigned_score": [3.5, 3.1, 2.8],
            "combined_importance": [3.5, 3.0, 2.7],
            "eligible": [True, True, True],
        }
    ).to_parquet(raw_path, index=False)
    runner = object.__new__(TfidfTopicAgenticForestRunner)
    runner.nn_config = MultiModelForestConfig(
        candidate_consistency_inner_folds=2,
        tfidf_topic=TfidfTopicDiscoveryConfig(
            min_df=1,
            topic_count=2,
            stability_repeats=0,
            orphan_ngram_cluster_similarity_threshold=0.25,
            orphan_ngram_full_min_inner_folds=1,
        ),
    )
    row = {
        "outer_fold": 1,
        "discovery": {
            "artifacts": {"ngram_scores": {"effect": str(raw_path)}},
            "topic_banks": {
                "effect": {
                    "topics": [
                        {
                            "topic_id": "effect_topic_001",
                            "terms": [{"term": "topic represented phrase"}],
                        }
                    ]
                }
            },
        },
    }
    policy = {
        "inner_fold_count": 2,
        "topic_score_selection": {
            "effect_orphan_ngram_branch": {
                "signatures": [
                    {
                        "inner_fold": 1,
                        "cluster_id": "effect_orphan_cluster_004",
                        "terms": ["squamous nsclc -"],
                        "primary_p": 0.02,
                    }
                ]
            }
        },
    }
    jobs, audit = runner._full_orphan_jobs_from_policy(row, policy)
    assert audit["uses_outer_heldout_labels"] is False
    assert audit["mapped_job_count"] == 1
    assert len(jobs) == 1
    bank, topic = jobs[0]
    assert bank == "effect"
    assert topic["_prompt_version"] == ORPHAN_NGRAM_LABEL_PROMPT_VERSION
    assert {term["term"] for term in topic["terms"]} == {"squamous nsclc"}
    assert topic["_selection_evidence"]["uses_outer_heldout_labels"] is False


def test_recovery_raw_ngrams_prioritize_topic_ties_and_exclude_covered_terms():
    topic = {
        "topic_id": "effect_topic_001",
        "terms": [
            {"term": f"brain metastasis signal {index}"}
            for index in range(15)
        ],
    }
    scores = pd.DataFrame(
        {
            "feature": [
                "global high rank noise",
                "brain lesions present",
                "metastasis surrounding edema",
                "brain imaging finding",
                "unrelated lower rank",
            ],
            "eligible": [True, True, True, True, True],
        }
    )
    selected = select_topic_recovery_raw_ngrams(
        scores,
        topic,
        excluded_terms=["brain lesions present"],
        limit=3,
    )
    assert selected[:2] == [
        "metastasis surrounding edema",
        "brain imaging finding",
    ]
    assert "brain lesions present" not in selected
    assert selected[2] == "global high rank noise"


def test_recovery_raw_ngrams_reserve_global_slots_and_can_avoid_retries():
    topic = {
        "topic_id": "effect_topic_001",
        "terms": [
            {"term": f"brain metastasis signal {index}"}
            for index in range(15)
        ],
    }
    scores = pd.DataFrame(
        {
            "feature": [
                "globally strongest unrelated contrast",
                "brain focused first",
                "brain focused second",
                "brain focused third",
                "metastasis focused fourth",
                "second unrelated contrast",
                "third unrelated contrast",
                "fourth unrelated contrast",
            ],
            "eligible": [True] * 8,
        }
    )
    first = select_topic_recovery_raw_ngrams(scores, topic, limit=4)
    assert first[:2] == ["brain focused first", "brain focused second"]
    assert "globally strongest unrelated contrast" in first
    second = select_topic_recovery_raw_ngrams(
        scores,
        topic,
        excluded_terms=first,
        limit=4,
    )
    assert set(first).isdisjoint(second)


def test_initial_review_registry_uses_evidence_mass_without_feature_count_cap():
    def provenance(bank, topic_id, score):
        return {
            "bank": bank,
            "topic_id": topic_id,
            "supporting_terms": [
                {"term": f"{topic_id}_term", "loading": 1.0, "signed_score": score}
            ],
        }

    candidates = [
        {
            "name": "shared_strong_signal",
            "type": "categorical",
            "categories": ["absent", "present"],
            "roles": ["confounder", "effect_modifier"],
            "provenance": [
                provenance("treatment", "t1", 10.0),
                provenance("outcome", "o1", 10.0),
                provenance("effect", "e1", 10.0),
            ],
        },
        {
            "name": "alternative_treatment_interpretation",
            "type": "categorical",
            "categories": ["absent", "present"],
            "roles": ["confounder"],
            "provenance": [provenance("treatment", "t1", 10.0)],
        },
        {
            "name": "weak_tail_signal",
            "type": "categorical",
            "categories": ["absent", "present"],
            "roles": ["confounder", "effect_modifier"],
            "provenance": [
                provenance("treatment", "t2", 1.0),
                provenance("outcome", "o2", 1.0),
                provenance("effect", "e2", 1.0),
            ],
        },
        {
            "name": "required_baseline_variable",
            "type": "continuous",
            "roles": ["confounder"],
            "required_or_prespecified": True,
            "provenance": [{"bank": "prespecified", "topic_id": "prespecified"}],
        },
    ]
    registry, _ = harmonize_topic_candidates(candidates)
    metadata = {
        "topic_banks": {
            bank: {
                "topics": [
                    {
                        "topic_id": f"{bank[0]}1" if bank != "effect" else "e1",
                        "terms": [{"loading": 1.0, "signed_score": 10.0}],
                    },
                    {
                        "topic_id": f"{bank[0]}2" if bank != "effect" else "e2",
                        "terms": [{"loading": 1.0, "signed_score": 1.0}],
                    },
                ]
            }
            for bank in ("treatment", "outcome", "effect")
        },
        "artifacts": {"ngram_scores": {}},
    }
    # Outcome ids use o1/o2; treatment ids use t1/t2 from the compact expression.
    active, deferred, audit = select_initial_topic_evidence_registry(
        registry, metadata, coverage_target=0.80
    )

    active_names = {entry["name"] for entry in active}
    assert active_names == {"shared_strong_signal", "required_baseline_variable"}
    assert {entry["name"] for entry in deferred} == {
        "alternative_treatment_interpretation",
        "weak_tail_signal",
    }
    assert audit["has_global_feature_count_cap"] is False
    assert audit["deferred_contracts_remain_eligible_for_additive_review"] is True
    assert all(audit["banks"][bank]["coverage_fraction"] >= 0.80 for bank in audit["banks"])


def test_deferred_review_additions_follow_failed_diagnostic_family_and_limit():
    def candidate(name, bank, topic_id, score):
        return {
            "name": name,
            "type": "categorical",
            "categories": ["absent", "present"],
            "roles": ["effect_modifier" if bank == "effect" else "confounder"],
            "provenance": [
                {
                    "bank": bank,
                    "topic_id": topic_id,
                    "supporting_terms": [
                        {
                            "term": f"{topic_id}_term",
                            "loading": 1.0,
                            "signed_score": score,
                        }
                    ],
                }
            ],
        }

    registry, _ = harmonize_topic_candidates(
        [
            candidate("active_treatment", "treatment", "t1", 10.0),
            candidate("deferred_treatment", "treatment", "t2", 5.0),
            candidate("deferred_effect", "effect", "e2", 9.0),
        ]
    )
    by_name = {entry["name"]: entry for entry in registry}
    metadata = {
        "topic_banks": {
            "treatment": {
                "topics": [
                    {"topic_id": "t1", "terms": [{"loading": 1.0, "signed_score": 10.0}]},
                    {"topic_id": "t2", "terms": [{"loading": 1.0, "signed_score": 5.0}]},
                ]
            },
            "outcome": {"topics": []},
            "effect": {
                "topics": [
                    {"topic_id": "e2", "terms": [{"loading": 1.0, "signed_score": 9.0}]}
                ]
            },
        }
    }
    gate = {
        "criteria": [
            {"family": "nuisance", "target": "treatment", "passed": False},
            {"family": "effect", "metric": "contrast_mass_coverage", "passed": True},
        ]
    }
    additions, audit = select_deferred_review_additions(
        [by_name["active_treatment"]],
        [by_name["deferred_treatment"], by_name["deferred_effect"]],
        gate,
        {"effect_coverage": {}},
        metadata,
        max_additions=1,
    )

    assert [entry["name"] for entry in additions] == ["deferred_treatment"]
    assert audit["relevant_banks"] == ["treatment"]
    assert audit["maximum_new_contracts"] == 1


def test_structured_review_gate_uses_reconstruction_and_cohort_contrast():
    diagnostic = {
        "treatment": {"auroc": 0.80, "brier": 0.20, "log_loss": 0.50},
        "outcome": {"auroc": 0.75, "brier": 0.22, "log_loss": 0.60},
        "benchmark": {
            "treatment": {
                "stacked_metrics": {"auroc": 0.80, "brier": 0.20, "log_loss": 0.50}
            },
            "outcome": {
                "stacked_metrics": {"auroc": 0.75, "brier": 0.22, "log_loss": 0.60}
            },
        },
        "effect_coverage": {
            "raw_effect_evidence_weak": False,
            "coverage_fraction": 0.85,
            "highest_ranked_raw_ngram_preservation": 0.85,
        },
        "effect_topic_reconstruction": {"mean_correlation": 0.10},
        "structured_contrast": {"mean_sign_agreement": 0.60},
    }
    config = MultiModelForestConfig(
        tfidf_topic=TfidfTopicDiscoveryConfig(
            minimum_tail_sign_agreement=0.50,
            initial_effect_coverage_target=0.80,
        )
    )

    passing = structured_review_gate(diagnostic, config)
    assert passing["passed"]
    effect_metrics = {
        row["metric"] for row in passing["criteria"] if row["family"] == "effect"
    }
    assert "heldout_topic_reconstruction_mean_correlation" in effect_metrics
    assert "structured_cohort_contrast_sign_agreement" in effect_metrics

    diagnostic["effect_topic_reconstruction"]["mean_correlation"] = -0.05
    diagnostic["structured_contrast"]["mean_sign_agreement"] = 0.40
    failing = structured_review_gate(diagnostic, config)
    assert not failing["passed"]
    assert sum(
        not row["passed"]
        for row in failing["criteria"]
        if row["family"] == "effect"
    ) == 2


def test_effect_raw_ngram_gate_does_not_hide_unmapped_stable_evidence(tmp_path):
    score_path = tmp_path / "effect_scores.parquet"
    pd.DataFrame(
        {
            "feature": ["raw_marker_a", "raw_marker_b"],
            "eligible": [True, True],
            "unsigned_score": [5.0, 4.0],
        }
    ).to_parquet(score_path, index=False)
    topic = {
        "topic_id": "effect_topic_000",
        "terms": [
            {"term": "raw_marker_a", "loading": 1.0, "signed_score": 2.0}
        ],
    }
    metadata = {
        "topic_banks": {
            "effect": {
                "topics": [topic],
                "weak_or_unstable_raw_evidence": False,
            }
        },
        "artifacts": {"ngram_scores": {"effect": str(score_path)}},
    }
    registry = [
        {
            "name": "marker_a",
            "provenance": [
                {
                    "bank": "effect",
                    "topic_id": "effect_topic_000",
                    "supporting_terms": [{"term": "raw_marker_a"}],
                }
            ],
        }
    ]

    coverage = _effect_evidence_coverage(
        registry,
        metadata,
        candidate_evidence_universe=registry,
    )

    assert coverage["mappable_highest_ranked_raw_ngrams"] == ["raw_marker_a"]
    assert coverage["unmapped_or_nonoperational_highest_ranked_raw_ngrams"] == [
        "raw_marker_b"
    ]
    assert coverage["preserved_highest_ranked_raw_ngrams"] == ["raw_marker_a"]
    assert coverage["highest_ranked_raw_ngram_preservation"] == 0.5


def test_effect_coverage_cannot_drop_a_shortlisted_topic_without_a_candidate():
    topics = [
        {
            "topic_id": topic_id,
            "terms": [
                {"term": f"term_{index}", "loading": 1.0, "signed_score": 1.0}
            ],
        }
        for index, topic_id in enumerate(["effect_topic_000", "effect_topic_001"])
    ]
    metadata = {
        "topic_banks": {"effect": {"topics": topics}},
        "topic_score_tests": {
            "banks": {
                "effect": {
                    "selected_topic_ids": [
                        "effect_topic_000",
                        "effect_topic_001",
                    ]
                }
            }
        },
        "artifacts": {"ngram_scores": {}},
    }
    first = {
        "name": "first_marker",
        "provenance": [
            {
                "bank": "effect",
                "topic_id": "effect_topic_000",
                "supporting_terms": [{"term": "term_0"}],
            }
        ],
    }

    coverage = _effect_evidence_coverage(
        [first], metadata, candidate_evidence_universe=[first]
    )

    assert coverage["covered_topic_ids"] == ["effect_topic_000"]
    assert coverage["uncovered_topic_ids"] == ["effect_topic_001"]
    assert coverage["coverage_fraction"] == pytest.approx(0.5)
    assert coverage["shortlist_to_all_fitted_mass_fraction"] == pytest.approx(1.0)


def test_name_and_value_harmonization_is_global_and_has_no_review_state():
    candidates = [
        {
            "name": "ECOG-status",
            "type": "categorical",
            "categories": ["0", "1", "2"],
            "roles": ["confounder"],
            "description": "ECOG status",
            "provenance": [{"bank": "treatment", "topic_id": "t1"}],
        },
        {
            "name": "ecog status",
            "type": "categorical",
            "categories": ["0", "1", "2", "3"],
            "roles": ["effect_modifier"],
            "description": "Baseline ECOG status",
            "provenance": [{"bank": "effect", "topic_id": "e1"}],
        },
    ]
    registry, dropped = harmonize_topic_candidates(candidates)
    assert not dropped
    assert len(registry) == 1
    assert registry[0]["action"] == "extract"
    assert set(registry[0]["roles"]) == {"confounder", "effect_modifier"}
    semantics = registry[0]["value_contract"]["missing_semantics"]
    assert len({semantics[key] for key in ("missing", "unknown", "absent", "not_documented")}) == 4
    assert "review" not in {entry["action"] for entry in registry}


def test_registry_state_tracks_new_topic_provenance_and_role_without_contract_change():
    registry, _ = harmonize_topic_candidates(
        [
            {
                "name": "baseline_marker",
                "type": "categorical",
                "categories": ["low", "high"],
                "roles": ["confounder"],
                "description": "Baseline marker status.",
                "provenance": [{"bank": "treatment", "topic_id": "t1"}],
            }
        ]
    )
    original = registry[0]
    expanded = {
        **original,
        "roles": ["confounder", "effect_modifier"],
        "provenance": [
            *original["provenance"],
            {"bank": "effect", "topic_id": "e2"},
        ],
    }

    assert expanded["contract_hash"] == original["contract_hash"]
    assert _registry_entry_state_hash(expanded) != _registry_entry_state_hash(
        original
    )


def test_agent_name_harmonization_merges_true_aliases_and_preserves_role_union():
    registry, _ = harmonize_topic_candidates(
        [
            {
                "name": "ecog_status",
                "type": "categorical",
                "categories": ["0", "1", "2", "3", "4"],
                "roles": ["confounder"],
                "description": "Baseline ECOG performance status",
                "provenance": [{"bank": "treatment", "topic_id": "t1"}],
            },
            {
                "name": "performance_status_ecog",
                "type": "categorical",
                "categories": ["0", "1", "2", "3", "4"],
                "roles": ["effect_modifier"],
                "description": "Baseline performance status on the ECOG scale",
                "provenance": [{"bank": "effect", "topic_id": "e1"}],
            },
        ]
    )
    by_name = {entry["name"]: entry for entry in registry}
    context = {
        "prompt_version": TOPIC_NAME_HARMONIZATION_PROMPT_VERSION,
        "candidates": [
            {
                "candidate_id": entry["candidate_id"],
                "name": entry["name"],
            }
            for entry in registry
        ],
    }
    response = {
        "decisions": [
            {
                "candidate_id": by_name["ecog_status"]["candidate_id"],
                "action": "extract",
                "canonical_name": "ecog_performance_status",
                "clinical_domain": "functional_status",
                "parent_object": "ecog_performance_status",
                "alias_of": None,
                "source_names": [],
                "derivation": None,
                "reason": "canonical owner",
            },
            {
                "candidate_id": by_name["performance_status_ecog"]["candidate_id"],
                "action": "alias/drop",
                "canonical_name": "ecog_performance_status",
                "clinical_domain": "functional_status",
                "parent_object": "ecog_performance_status",
                "alias_of": "ecog_status",
                "source_names": [],
                "derivation": None,
                "reason": "same baseline ECOG target",
            },
        ]
    }
    assert not topic_harmonization_response_issues(response, context)
    merged, dropped = apply_topic_name_harmonization(registry, [response])
    assert [entry["name"] for entry in merged] == ["ecog_performance_status"]
    assert set(merged[0]["roles"]) == {"confounder", "effect_modifier"}
    assert {item["topic_id"] for item in merged[0]["provenance"]} == {"t1", "e1"}
    assert any(item.get("action") == "alias/drop" for item in dropped)


def test_global_dedup_blocks_are_bounded_and_alias_application_is_sparse():
    candidates = [
        {
            "name": "ecog_status",
            "type": "categorical",
            "categories": ["0", "1", "2", "3", "4"],
            "roles": ["confounder"],
            "description": "Baseline ECOG performance status",
            "clinical_domain": "functional_status",
            "parent_object": "ecog",
            "provenance": [{"bank": "treatment", "topic_id": "t1"}],
        },
        {
            "name": "performance_status_ecog",
            "type": "categorical",
            "categories": ["0", "1", "2", "3", "4"],
            "roles": ["effect_modifier"],
            "description": "Baseline performance status measured on the ECOG scale",
            "clinical_domain": "oncology_assessment",
            "parent_object": "performance_score",
            "provenance": [{"bank": "effect", "topic_id": "e1"}],
        },
        *[
            {
                "name": f"distinct_measure_{index}",
                "type": "continuous",
                "roles": ["confounder"],
                "description": f"Distinct pretreatment laboratory measure number {index}",
                "clinical_domain": f"domain_{index}",
                "parent_object": f"object_{index}",
                "provenance": [{"bank": "outcome", "topic_id": f"o{index}"}],
            }
            for index in range(12)
        ],
    ]
    registry, _ = harmonize_topic_candidates(candidates)
    blocks = build_topic_global_dedup_blocks(
        registry, max_block_size=8, min_similarity=0.15, max_neighbors=2
    )
    assert blocks
    assert max(map(len, blocks)) <= 8
    assert any(
        {entry["name"] for entry in block}
        >= {"ecog_status", "performance_status_ecog"}
        for block in blocks
    )

    context = {
        "prompt_version": TOPIC_GLOBAL_DEDUP_PROMPT_VERSION,
        "candidates": [
            {"name": "ecog_status"},
            {"name": "performance_status_ecog"},
        ],
    }
    response = {
        "resolutions": [
            {
                "action": "alias/drop",
                "member_names": ["ecog_status", "performance_status_ecog"],
                "canonical_name": "ecog_status",
                "source_names": [],
                "derivation": None,
                "reason": "same baseline ECOG construct",
            }
        ]
    }
    assert not topic_harmonization_response_issues(response, context)
    merged, dropped = apply_topic_global_dedup(registry, response)
    ecog = next(entry for entry in merged if entry["name"] == "ecog_status")
    assert set(ecog["roles"]) == {"confounder", "effect_modifier"}
    assert "performance_status_ecog" not in {entry["name"] for entry in merged}
    assert any(item.get("action") == "alias/drop" for item in dropped)


def test_name_harmonization_cannot_alias_a_base_variable_to_distinct_subfield():
    registry, _ = harmonize_topic_candidates(
        [
            {
                "name": "prior_treatment_history",
                "type": "categorical",
                "categories": ["absent", "present"],
                "roles": ["confounder"],
                "description": "Any prior treatment history",
                "provenance": [{"bank": "treatment", "topic_id": "t1"}],
            },
            {
                "name": "prior_treatment_cycles_completed",
                "type": "categorical",
                "categories": ["0", "1_to_3", "4_or_more"],
                "roles": ["effect_modifier"],
                "description": "Number of prior treatment cycles completed",
                "provenance": [{"bank": "effect", "topic_id": "e1"}],
            },
        ]
    )
    by_name = {entry["name"]: entry for entry in registry}
    response = {
        "decisions": [
            {
                "candidate_id": by_name["prior_treatment_history"]["candidate_id"],
                "action": "alias/drop",
                "canonical_name": "prior_treatment_cycles_completed",
                "clinical_domain": "oncology_treatment",
                "parent_object": "prior_treatment",
                "alias_of": "prior_treatment_cycles_completed",
                "source_names": [],
                "derivation": None,
                "reason": "incorrectly proposed alias",
            },
            {
                "candidate_id": by_name["prior_treatment_cycles_completed"][
                    "candidate_id"
                ],
                "action": "extract",
                "canonical_name": "prior_treatment_cycles_completed",
                "clinical_domain": "oncology_treatment",
                "parent_object": "prior_treatment",
                "alias_of": None,
                "source_names": [],
                "derivation": None,
                "reason": "canonical owner",
            },
        ]
    }

    retained, dropped = apply_topic_name_harmonization(registry, [response])

    assert {entry["name"] for entry in retained} == {
        "prior_treatment_history",
        "prior_treatment_cycles_completed",
    }
    history = next(
        entry for entry in retained if entry["name"] == "prior_treatment_history"
    )
    assert history["harmonization_audit"][0]["action"] == "extract"
    assert not any(item.get("action") == "alias/drop" for item in dropped)


def test_value_contract_validation_and_application_preserve_distinct_missing_semantics():
    registry, _ = harmonize_topic_candidates(
        [
            {
                "name": "serum_creatinine",
                "type": "continuous",
                "roles": ["confounder"],
                "description": "Pretreatment serum creatinine",
                "provenance": [{"bank": "outcome", "topic_id": "o1"}],
            }
        ]
    )
    semantics = {
        "missing": "request failed or no usable value",
        "unknown": "explicitly indeterminate",
        "absent": "explicitly absent",
        "not_documented": "not stated before treatment",
    }
    context = {
        "prompt_version": TOPIC_VALUE_HARMONIZATION_PROMPT_VERSION,
        "candidates": [{"name": "serum_creatinine"}],
    }
    response = {
        "features": [
            {
                "name": "serum_creatinine",
                "data_type": "continuous",
                "permitted_categories": None,
                "canonical_unit": "mg/dL",
                "unit_conversions": {
                    "umol/L": {"multiply": 0.011312, "add": 0.0}
                },
                "category_synonyms": {},
                "ordinal_order": None,
                "missing_semantics": semantics,
                "deterministic_derivation": None,
                "temporal_cutoff": "use only information documented before the treatment decision",
                "description": "Return pretreatment serum creatinine in mg/dL.",
                "reason": "canonical laboratory unit",
            }
        ]
    }
    assert not topic_harmonization_response_issues(response, context)
    harmonized, dropped = apply_topic_value_harmonization(registry, [response])
    assert not dropped
    contract = harmonized[0]["value_contract"]
    assert contract["canonical_unit"] == "mg/dL"
    assert contract["unit_conversions"]["umol/L"]["multiply"] == pytest.approx(0.011312)
    assert len(set(contract["missing_semantics"].values())) == 4


def test_categorical_value_contract_always_encodes_unknown_and_not_documented():
    registry, _ = harmonize_topic_candidates(
        [
            {
                "name": "radiation_referral_status",
                "type": "categorical",
                "categories": ["yes", "no"],
                "roles": ["effect_modifier"],
                "provenance": [{"bank": "effect", "topic_id": "e1"}],
            }
        ]
    )
    response = {
        "features": [
            {
                "name": "radiation_referral_status",
                "data_type": "categorical",
                "permitted_categories": ["yes", "no"],
                "canonical_unit": None,
                "unit_conversions": {},
                "category_synonyms": {},
                "ordinal_order": None,
                "missing_semantics": {
                    "missing": "request failed",
                    "unknown": "explicitly indeterminate",
                    "absent": "explicitly absent",
                    "not_documented": "not stated before treatment",
                },
                "deterministic_derivation": None,
                "temporal_cutoff": "use only information documented before the treatment decision",
                "description": "Pretreatment radiation referral status.",
            }
        ]
    }

    harmonized, dropped = apply_topic_value_harmonization(registry, [response])

    assert not dropped
    assert harmonized[0]["categories"] == [
        "yes",
        "no",
        "unknown",
        "not_documented",
    ]
    aliases = harmonized[0]["value_contract"]["category_synonyms"]
    assert "indeterminate" in aliases["unknown"]
    assert "not stated" in aliases["not_documented"]


def test_categorical_value_contract_adds_common_treatment_line_aliases():
    registry, _ = harmonize_topic_candidates(
        [
            {
                "name": "treatment_line",
                "type": "categorical",
                "categories": [
                    "first-line",
                    "second-line",
                    "third-line",
                    "subsequent-line",
                ],
                "roles": ["confounder"],
                "provenance": [{"bank": "treatment", "topic_id": "t1"}],
            }
        ]
    )
    response = {
        "features": [
            {
                "name": "treatment_line",
                "data_type": "categorical",
                "permitted_categories": [
                    "first-line",
                    "second-line",
                    "third-line",
                    "subsequent-line",
                ],
                "canonical_unit": None,
                "unit_conversions": {},
                "category_synonyms": {},
                "ordinal_order": [
                    "first-line",
                    "second-line",
                    "third-line",
                    "subsequent-line",
                ],
                "missing_semantics": {
                    "missing": "request failed",
                    "unknown": "explicitly indeterminate",
                    "absent": "explicitly absent",
                    "not_documented": "not stated before treatment",
                },
                "deterministic_derivation": None,
                "temporal_cutoff": "before treatment",
                "description": "Pretreatment systemic therapy line.",
            }
        ]
    }

    harmonized, dropped = apply_topic_value_harmonization(registry, [response])

    assert not dropped
    aliases = harmonized[0]["value_contract"]["category_synonyms"]
    assert "fourth-line" in aliases["subsequent-line"]
    assert "4th line or later" in aliases["subsequent-line"]


def test_deterministic_derivation_is_materialized_without_llm_extraction():
    registry, _ = harmonize_topic_candidates(
        [
            {
                "name": "weight_kg",
                "type": "continuous",
                "roles": ["confounder"],
                "description": "Weight in kg",
                "provenance": [{"bank": "treatment", "topic_id": "t1"}],
            },
            {
                "name": "height_m",
                "type": "continuous",
                "roles": ["confounder"],
                "description": "Height in meters",
                "provenance": [{"bank": "treatment", "topic_id": "t2"}],
            },
            {
                "name": "body_mass_index",
                "type": "continuous",
                "roles": ["confounder"],
                "description": "BMI",
                "provenance": [{"bank": "outcome", "topic_id": "o1"}],
                "action": "derive",
                "derivation": {
                    "operation": "ratio",
                    "source_names": ["weight_kg", "height_m"],
                    "parameters": {"denominator_power": 2},
                },
            },
        ]
    )
    frame = pd.DataFrame(
        {
            "_oci_row_id": [0, 1],
            "explicit_feat_weight_kg": [80.0, 70.0],
            "explicit_feat_weight_kg_missing": [False, False],
            "explicit_feat_height_m": [2.0, np.nan],
            "explicit_feat_height_m_missing": [False, True],
        }
    )
    derived = apply_registry_derivations(frame, registry)
    assert derived.loc[0, "explicit_feat_body_mass_index"] == pytest.approx(20.0)
    assert bool(derived.loc[1, "explicit_feat_body_mass_index_missing"])


def test_extraction_group_cap_and_text_hash_invalidate_cache_identity(tmp_path):
    config = AppliedInferenceConfig(
        dataset_path="in_memory",
        text_column="clinical_text",
        architecture=ModelArchitectureConfig(),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            vllm_model_name="fixed-model",
            cache_dir=str(tmp_path),
            max_variables_per_extraction_request=3,
        ),
    )
    provider = VLLMExplicitFeatureExtractionProvider(config, tmp_path)
    specs = [
        ExplicitFeatureSpec(
            name=f"lab_{index}", type="continuous", roles=["confounder"],
            description=f"lab: value {index}",
        )
        for index in range(11)
    ]
    groups = provider._extraction_spec_groups(specs)
    assert max(map(len, groups)) == 3
    provider._active_text_hash = "first"
    first = provider._cache_config([specs[0]])
    provider._active_text_hash = "second"
    second = provider._cache_config([specs[0]])
    assert provider.cache._get_cache_path("in_memory", first) != provider.cache._get_cache_path(
        "in_memory", second
    )
    assert provider.cache._get_patient_cache_path(
        "in_memory", first
    ) == provider.cache._get_patient_cache_path("in_memory", second)
    value_col = "explicit_feat_lab_0"
    missing_col = f"{value_col}_missing"
    provider.cache.save_patient_values(
        "in_memory",
        first,
        ["patient_a_text", "patient_b_text"],
        pd.DataFrame(
            {
                value_col: [1.0, 2.0],
                missing_col: [False, False],
            }
        ),
    )
    reused = provider.cache.load_patient_values(
        "in_memory",
        second,
        ["patient_b_text", "unseen_text", "patient_a_text"],
    )
    assert reused["__oci_cache_row_index"].tolist() == [0, 2]
    assert reused[value_col].tolist() == [2.0, 1.0]

    structured = [
        ExplicitFeatureSpec(
            name=f"oncology_{index}",
            type="categorical",
            categories=["absent", "present"],
            roles=["confounder"],
            description=(
                "clinical_domain=oncology; "
                f"parent_object=parent_{index}: distinct feature"
            ),
        )
        for index in range(3)
    ]
    assert len(provider._extraction_spec_groups(structured)) == 1

    synonymous_domains = [
        ExplicitFeatureSpec(
            name="brain_mri_findings",
            type="categorical",
            categories=["absent", "present"],
            roles=["effect_modifier"],
            description=(
                "clinical_domain=neuroimaging; parent_object=brain_mri: findings"
            ),
        ),
        ExplicitFeatureSpec(
            name="pulmonary_nodule_suvmax",
            type="continuous",
            roles=["effect_modifier"],
            description=(
                "clinical_domain=radiology; parent_object=pet_ct: nodule SUVmax"
            ),
        ),
        ExplicitFeatureSpec(
            name="surveillance_imaging_history",
            type="categorical",
            categories=["absent", "present"],
            roles=["confounder"],
            description=(
                "clinical_domain=imaging; parent_object=surveillance: prior imaging"
            ),
        ),
    ]
    packed_synonyms = provider._extraction_spec_groups(synonymous_domains)
    assert len(packed_synonyms) == 1
    assert {spec.name for spec in packed_synonyms[0]} == {
        "brain_mri_findings",
        "pulmonary_nodule_suvmax",
        "surveillance_imaging_history",
    }


def test_parsimony_requires_dimension_reduction_and_all_diagnostic_families():
    base = {
        "treatment": {"auroc": 0.8, "brier": 0.20, "log_loss": 0.5},
        "outcome": {"auroc": 0.75, "brier": 0.22, "log_loss": 0.6},
        "structured_contrast": {"mean_sign_agreement": 0.8},
    }
    trial = {
        "treatment": {"auroc": 0.79, "brier": 0.20, "log_loss": 0.51},
        "outcome": {"auroc": 0.74, "brier": 0.22, "log_loss": 0.61},
        "structured_contrast": {"mean_sign_agreement": 0.8},
    }
    passed, _ = parsimony_replacement_passes(
        base=base,
        trial=trial,
        base_dimension=10,
        trial_dimension=10,
        source_topic_coverage_loss=0.0,
        topic_reconstruction_loss=0.0,
        required_features_preserved=True,
        role_union_preserved=True,
    )
    assert not passed
    passed, reasons = parsimony_replacement_passes(
        base=base,
        trial=trial,
        base_dimension=10,
        trial_dimension=8,
        source_topic_coverage_loss=0.04,
        topic_reconstruction_loss=0.02,
        required_features_preserved=True,
        role_union_preserved=True,
    )
    assert passed, reasons


def test_v2_config_rejects_legacy_methods():
    with pytest.raises(ValueError, match="legacy discovery method"):
        normalize_tfidf_topic_feature_discovery_methods(["htr"])


def test_nested_nuisance_prediction_provenance_excludes_each_row():
    texts = [f"patient token_{index % 4} age {40 + index}" for index in range(36)]
    labels = np.tile([0.0, 1.0], 18)
    result = fit_cross_fitted_nuisance_stack(
        texts=texts,
        values=labels,
        views=[
            BoWViewConfig(
                name="linear_1_3",
                min_df=1,
                max_features=100,
                ngram_range_min=1,
                ngram_range_max=3,
                bow_model="linear",
            )
        ],
        folds=3,
        binary=True,
        random_state=9,
        nuisance_stack_config=TfidfNuisanceStackScientificConfig(),
    )
    assert np.isfinite(result["stacked_oof"]).all()
    for row_index, fit_positions in enumerate(result["fit_positions_by_row"]):
        assert row_index not in fit_positions
        assert fit_positions


def test_fake_agent_nested_topic_workflow_is_fold_local_and_structured_only(tmp_path):
    rows = []
    for index in range(64):
        treatment = (index // 2) % 2
        outcome = index % 2
        marker = "high" if treatment else "low"
        rows.append(
            {
                "clinical_text": (
                    f"baseline marker {marker} age {45 + index % 30} ecog {index % 3} "
                    f"stage lung biomarker smoking history laboratory imaging symptom "
                    f"performance pretreatment token{index % 8}"
                ),
                "treatment_indicator": treatment,
                "outcome_indicator": outcome,
                "true_ite_prob": 0.2 if marker == "high" else -0.1,
            }
        )
    dataset = pd.DataFrame(rows)
    topic_config = TfidfTopicDiscoveryConfig(
        max_features=200,
        min_df=1,
        max_df=1.0,
        top_fraction=0.25,
        topic_count=2,
        topic_seeds=[42],
        nmf_max_iter=80,
        stability_repeats=1,
        stability_fraction=0.75,
        minimum_arm_document_support=1,
        minimum_nuisance_source_agreement=0.0,
        minimum_subsample_selection_fraction=0.0,
        minimum_tail_sign_agreement=0.0,
        topic_label_parallelism=2,
        score_test_bootstrap_repeats=49,
        score_test_bootstrap_top_topics=0,
        score_test_bootstrap_chunk_size=20,
        score_test_min_topics_per_bank=1,
        score_test_max_topics_per_bank=1,
    )
    config = AppliedInferenceConfig(
        dataset_path="in_memory_nested_topic_test",
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                agent_model_name="fake-topic-agent",
                min_feature_coverage=0.0,
                agent_schema_repair_attempts=1,
            ),
            multi_model_forest=MultiModelForestConfig(
                nuisance_folds=2,
                feature_discovery_methods=["bow", "tfidf_topic_contrast"],
                bow_views=[
                    BoWViewConfig(
                        name="linear_1_3",
                        max_features=200,
                        min_df=1,
                        max_df=1.0,
                        ngram_range_min=1,
                        ngram_range_max=3,
                        bow_model="linear",
                    )
                ],
                candidate_consistency_inner_folds=2,
                extracted_feature_review_max_rounds=1,
                parsimony_review_enabled=False,
                cpus_total=1,
                tfidf_topic=topic_config,
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )

    class FakeTopicAgent:
        def __init__(self):
            self.contexts = []

        def _resolve_agent_model_name(self):
            return "fake-topic-agent"

        def propose(self, context):
            self.contexts.append(context)
            version = context["prompt_version"]
            if version in {
                "tfidf_topic_label_v2",
                "tfidf_topic_recovery_v2",
                ORPHAN_NGRAM_LABEL_PROMPT_VERSION,
            }:
                supporting = context["topic_terms"][0]["term"]
                return {
                    "general_topic": "baseline marker",
                    "topic_quality": "coherent",
                    "proposals": [
                        {
                            "action": "add",
                            "name": "baseline_marker_status",
                            "type": "categorical",
                            "categories": ["low", "high"],
                            "roles": [context["mechanical_role"]],
                            "description": "Marker status documented before treatment",
                            "supporting_terms": [supporting],
                            "rationale": "The supplied term records the baseline marker.",
                            "expected_signal": "topic-organized evidence",
                        }
                    ],
                }
            if version == TOPIC_NAME_HARMONIZATION_PROMPT_VERSION:
                return {
                    "decisions": [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "action": "extract",
                            "canonical_name": candidate["name"],
                            "clinical_domain": "biomarker",
                            "parent_object": "baseline_marker",
                            "alias_of": None,
                            "source_names": [],
                            "derivation": None,
                            "reason": "operational distinct target",
                        }
                        for candidate in context["candidates"]
                    ]
                }
            if version == "tfidf_topic_global_dedup_v2":
                return {"resolutions": []}
            if version in {
                TOPIC_VALUE_HARMONIZATION_PROMPT_VERSION,
                "tfidf_topic_value_repair_v2",
            }:
                semantics = {
                    "missing": "request failed or unusable",
                    "unknown": "explicitly unknown",
                    "absent": "explicitly absent",
                    "not_documented": "not stated before treatment",
                }
                return {
                    "features": [
                        {
                            "name": candidate["name"],
                            "data_type": "categorical",
                            "permitted_categories": ["low", "high"],
                            "canonical_unit": None,
                            "unit_conversions": {},
                            "category_synonyms": {},
                            "ordinal_order": ["low", "high"],
                            "missing_semantics": semantics,
                            "deterministic_derivation": None,
                            "temporal_cutoff": "use only information documented before the treatment decision",
                            "description": "Extract low or high baseline marker status.",
                            "reason": "fixed categorical contract",
                        }
                        for candidate in context["candidates"]
                    ]
                }
            raise AssertionError(f"unexpected fake-agent prompt: {version}")

    class FakeTopicExtraction:
        def ensure_features(self, frame, specs):
            result = frame.copy()
            for spec in specs:
                value_column = f"explicit_feat_{spec.name}"
                result[value_column] = np.where(
                    result["clinical_text"].str.contains("marker high"), "high", "low"
                )
                result[f"{value_column}_missing"] = False
            return result

    class FakeTopicEvaluator:
        def evaluate_split(self, train_df, test_df, specs, fold_id):
            assert not any(
                str(column).startswith("true_")
                for column in [*train_df.columns, *test_df.columns]
            )
            predictions = test_df[["_oci_row_id"]].copy()
            predictions["pred_ite_prob"] = 0.1
            predictions["pred_y0_prob"] = 0.4
            predictions["pred_y1_prob"] = 0.5
            predictions["pred_propensity_prob"] = 0.5
            predictions["pred_outcome_prob"] = 0.5
            predictions["cv_fold"] = fold_id
            return SplitEvaluation(
                predictions=predictions,
                metrics={"fold": fold_id, "n_explicit_features": len(specs)},
            )

    primary_path = tmp_path / "stage1_nuisance_predictions.parquet"
    handoff_path = tmp_path / "handoff" / "discovery_contexts.jsonl"
    run_tfidf_topic_stage1(
        dataset=dataset,
        config=config,
        output_path=primary_path,
        artifact_dir=tmp_path,
        handoff_path=handoff_path,
    )
    handoff_rows = [
        json.loads(line)
        for line in handoff_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for handoff_row in handoff_rows:
        discovery = handoff_row["discovery"]
        if handoff_row["scope"] == "candidate_selection_inner_fit":
            score_path = discovery["artifacts"]["topic_score_tests"]
            assert score_path
            scores = json.loads(Path(score_path).read_text(encoding="utf-8"))
            assert scores["status"] == "completed"
            assert scores["uses_heldout_treatment_and_outcome"]
            for bank in ("treatment", "outcome", "effect"):
                bank_scores = scores["banks"][bank]
                assert len(bank_scores["topic_tests"]) == 2
                assert bank_scores["selection_count"] == 1
                assert bank_scores["bootstrap_calibration"][
                    "complete_topic_family"
                ]
        else:
            assert discovery["artifacts"]["topic_score_tests"] is None
            assert discovery["topic_score_tests"]["status"] == "not_run"
            assert not discovery["topic_score_tests"][
                "uses_heldout_treatment_and_outcome"
            ]
    preflight = validate_tfidf_topic_stage2_handoff(
        dataset=dataset,
        config=config,
        handoff_path=handoff_path,
    )
    assert preflight["status"] == "passed"
    assert preflight["exact_context_count"] == 6
    assert preflight["inner_score_context_count"] == 4
    assert preflight["full_outer_context_count"] == 2
    assert preflight["outer_test_score_artifact_count"] == 0
    assert preflight["llm_or_extraction_client_constructed"] is False
    agent = FakeTopicAgent()
    stage2_path = tmp_path / "stage2" / "agentic_predictions.parquet"
    run_tfidf_topic_agentic_forest(
        dataset=dataset,
        config=config,
        output_path=stage2_path,
        handoff_path=handoff_path,
        proposal_agent=agent,
        extraction_provider=FakeTopicExtraction(),
        evaluator=FakeTopicEvaluator(),
    )

    predictions = pd.read_parquet(stage2_path)
    assert len(predictions) == len(dataset)
    assert predictions["_oci_row_id"].is_unique
    assert set(predictions["honest_outer_holdout"]) == {True}
    assert not any(str(column).startswith("true_") for column in predictions.columns)
    for _, prediction_row in predictions.iterrows():
        row_id = int(prediction_row["_oci_row_id"])
        assert row_id not in set(map(int, prediction_row["forest_fit_row_ids"]))
        assert row_id not in set(map(int, prediction_row["stage1_nuisance_fit_row_ids"]))
    assert all("true_" not in json.dumps(context) for context in agent.contexts)
    label_contexts = [
        context
        for context in agent.contexts
        if context.get("prompt_version") == "tfidf_topic_label_v2"
    ]
    assert label_contexts
    label_checkpoints = list(
        (stage2_path.parent / "tfidf_topic_agentic_forest").rglob(
            "topic_labels.jsonl"
        )
    )
    assert label_checkpoints
    for checkpoint in label_checkpoints:
        for record in (
            json.loads(line)
            for line in checkpoint.read_text().splitlines()
            if line.strip()
        ):
            assert record["model_identity"] == "fake-topic-agent"
            assert record["request_settings_hash"]
    label_groups = {}
    for context in label_contexts:
        key = (
            context["outer_fold"],
            context["scope"],
            context["inner_fold"],
            context["bank"],
        )
        label_groups.setdefault(key, []).append(context)
        evidence = context["heldout_relevance_evidence"]
        assert evidence
        if context["scope"] == "full_outer_train":
            assert evidence["source"] == "fixed_inner_score_test_policy_mapping"
            assert not evidence["uses_outer_heldout_labels"]
            assert evidence["matched_inner_evidence"]
        else:
            assert evidence["source"] == (
                "exact_inner_heldout_topic_and_ngram_score_test"
            )
            assert evidence["uses_heldout_treatment_and_outcome"]
    assert all(len(contexts) == 1 for contexts in label_groups.values())
    recovery_contexts = [
        context
        for context in agent.contexts
        if context.get("prompt_version") == "tfidf_topic_recovery_v2"
    ]
    assert recovery_contexts
    assert any(
        not context["heldout_relevance_evidence"].get(
            "selected_for_agent", True
        )
        for context in recovery_contexts
    )
    assert all(len(context["topic_terms"]) == 15 for context in recovery_contexts)
    artifact_root = stage2_path.parent / "tfidf_topic_agentic_forest"
    manifest = json.loads((artifact_root / "manifest.json").read_text())
    frozen_hash = hashlib.sha256(stage2_path.read_bytes()).hexdigest()
    assert manifest["schema_version"] == "tfidf_topic_agentic_forest_v7"
    assert manifest["prediction_sha256_before_oracle_join"] == frozen_hash
    assert not manifest["oracle_columns_consumed_by_modeling"]
    posthoc = json.loads((artifact_root / "posthoc_oracle_metrics.json").read_text())
    assert posthoc["all_outer_predictions_frozen_before_oracle_join"]
    assert posthoc["frozen_prediction_sha256"] == frozen_hash
    assert posthoc["overall"]["n"] == len(dataset)
    posthoc_predictions = pd.read_parquet(
        artifact_root / "posthoc_predictions_with_oracle.parquet"
    )
    assert "true_ite_prob" in posthoc_predictions.columns
    recovered_registry_evidence = []
    for outer_fold in (1, 2):
        outer_dir = artifact_root / f"outer_fold_{outer_fold:03d}"
        assert (outer_dir / "inner_001" / "canonical_registry.json").exists()
        assert (outer_dir / "inner_002" / "canonical_registry.json").exists()
        for inner_fold in (1, 2):
            inner_registry = json.loads(
                (
                    outer_dir
                    / f"inner_{inner_fold:03d}"
                    / "canonical_registry.json"
                ).read_text()
            )
            recovered_registry_evidence.extend(
                round_row.get("changed_registry_entries", [])
                for round_row in inner_registry.get("review_rounds", [])
                if round_row.get("recovery_source")
                == "targeted_source_topic_prompt"
            )
        assert (outer_dir / "full_outer_train" / "harmonization" / "manifest.json").exists()
        registry = json.loads((outer_dir / "canonical_registry.json").read_text())
        assert registry["registry"]
        assert {entry["action"] for entry in registry["registry"]} <= {
            "extract",
            "derive",
        }
    assert any(recovered_registry_evidence)
    assert not list(tmp_path.rglob("*htr*"))
    assert not list(tmp_path.rglob("*sentence_transformer*"))
    assert not list(tmp_path.rglob("*raw_text_forest*"))


def test_heldout_phrase_and_labels_cannot_change_fitted_topic_artifacts(tmp_path):
    fit_rows = []
    for index in range(32):
        fit_rows.append(
            {
                "_oci_row_id": index,
                "clinical_text": (
                    f"baseline lung marker status age ecog stage smoking laboratory imaging "
                    f"symptom history performance biomarker treatment decision token{index % 4}"
                ),
                "treatment_indicator": (index // 2) % 2,
                "outcome_indicator": index % 2,
            }
        )
    heldout = pd.DataFrame(
        [
            {
                "_oci_row_id": 32 + index,
                "clinical_text": (
                    "outersecretphrase outersecretbigram baseline heldout document "
                    f"marker status token{index % 4} age ecog {index % 3}"
                ),
                "treatment_indicator": index % 2,
                "outcome_indicator": (index // 2) % 2,
            }
            for index in range(8)
        ]
    )
    mutated = heldout.copy()
    mutated["treatment_indicator"] = 1 - mutated["treatment_indicator"]
    mutated["outcome_indicator"] = 1 - mutated["outcome_indicator"]
    topic_config = TfidfTopicDiscoveryConfig(
        max_features=200,
        min_df=1,
        max_df=1.0,
        top_fraction=0.25,
        topic_count=2,
        topic_seeds=[42],
        nmf_max_iter=80,
        stability_repeats=0,
        minimum_arm_document_support=1,
        minimum_nuisance_source_agreement=0.0,
        minimum_subsample_selection_fraction=0.0,
        minimum_tail_sign_agreement=0.0,
        score_test_bootstrap_repeats=0,
        score_test_min_topics_per_bank=1,
        score_test_max_topics_per_bank=2,
    )
    view = BoWViewConfig(
        name="linear_1_3",
        max_features=200,
        min_df=1,
        max_df=1.0,
        ngram_range_min=1,
        ngram_range_max=3,
        bow_model="linear",
    )
    first = fit_tfidf_topic_context(
        fit_df=pd.DataFrame(fit_rows),
        heldout_df=heldout,
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        views=[view],
        nuisance_folds=2,
        config=topic_config,
        artifact_dir=tmp_path / "first",
        scope_id="leakage_first",
    )
    second = fit_tfidf_topic_context(
        fit_df=pd.DataFrame(fit_rows),
        heldout_df=mutated,
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        views=[view],
        nuisance_folds=2,
        config=topic_config,
        artifact_dir=tmp_path / "second",
        scope_id="leakage_second",
    )
    assert "outersecretphrase" not in set(first["common_vocabulary"])
    assert first["common_vocabulary"] == second["common_vocabulary"]
    assert first["topic_banks"] == second["topic_banks"]
    first_topics = load_named_array_bank(
        first["artifacts"]["fit_topic_values"],
        expected_row_count=len(fit_rows),
    )
    second_topics = load_named_array_bank(
        second["artifacts"]["fit_topic_values"],
        expected_row_count=len(fit_rows),
    )
    assert set(first_topics) == set(second_topics)
    for bank in first_topics:
        np.testing.assert_allclose(first_topics[bank], second_topics[bank])
    prompt_text = json.dumps(first["topic_banks"])
    assert "outersecret" not in prompt_text

    scored_first = fit_tfidf_topic_context(
        fit_df=pd.DataFrame(fit_rows),
        heldout_df=heldout,
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        views=[view],
        nuisance_folds=2,
        config=topic_config,
        artifact_dir=tmp_path / "scored_first",
        scope_id="scored_first",
        enable_heldout_score_tests=True,
    )
    scored_second = fit_tfidf_topic_context(
        fit_df=pd.DataFrame(fit_rows),
        heldout_df=mutated,
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        views=[view],
        nuisance_folds=2,
        config=topic_config,
        artifact_dir=tmp_path / "scored_second",
        scope_id="scored_second",
        enable_heldout_score_tests=True,
    )
    assert scored_first["common_vocabulary"] == scored_second["common_vocabulary"]
    assert scored_first["topic_banks"] == scored_second["topic_banks"]
    first_scores = json.loads(
        Path(scored_first["artifacts"]["topic_score_tests"]).read_text(
            encoding="utf-8"
        )
    )
    second_scores = json.loads(
        Path(scored_second["artifacts"]["topic_score_tests"]).read_text(
            encoding="utf-8"
        )
    )
    assert first_scores["uses_heldout_treatment_and_outcome"]
    assert second_scores["uses_heldout_treatment_and_outcome"]
    first_moments = [
        term["heldout_score_moment"]
        for bank in first_scores["banks"].values()
        for topic in bank["topic_tests"]
        for term in topic["term_scores"]
    ]
    second_moments = [
        term["heldout_score_moment"]
        for bank in second_scores["banks"].values()
        for topic in bank["topic_tests"]
        for term in topic["term_scores"]
    ]
    assert not np.allclose(first_moments, second_moments)


def test_v1_handoff_is_rejected(tmp_path):
    from oci.inference.tfidf_topic_agentic_forest import _read_jsonl

    path = tmp_path / "handoff.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "multi_model_forest_handoff_v1",
                "outer_fold": 1,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="rejects legacy handoffs"):
        _read_jsonl(path)
