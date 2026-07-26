import json
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import oci.inference.multi_model_agentic_forest as multi_model_agentic_module

from oci.config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    BoWViewConfig,
    ClusterLocalEmbeddingScientificConfig,
    EmbeddingContrastDiscoveryConfig,
    ExperimentConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    ModelArchitectureConfig,
    MultiModelAgenticForestConfig,
    MultiModelForestConfig,
)
from oci.inference.agentic_explicit_feature_forest import (
    AgenticFeatureProposal,
    SplitEvaluation,
    build_agent_prompt,
)
from oci.inference.embedding_contrast_discovery import (
    EmbeddingContrastEvidenceGenerator,
    _default_embedding_cache_dir,
    _informative_chunk_text,
    redact_embedding_contrast_evidence,
)
from oci.inference.multi_model_agentic_forest import (
    MultiModelAgenticForestRunner,
    PrecomputedDiscoveryMultiModelAgenticForestRunner,
    _candidate_consistency_threshold,
    _build_value_driven_feature_clusters,
    _compact_multi_model_agent_context,
    _evaluate_extracted_feature_set_diagnostic,
    _extracted_feature_review_gate,
    _feature_redundancy_review,
    _parsimony_mutual_neighbor_pairs,
    _strict_parsimony_replacement_decision,
    _fallback_consistency_proposals,
    _fit_binary_bow_fold,
    run_multi_model_agentic_forest,
)
from oci.inference.multi_model_forest import (
    MultiModelForestRunner,
    _config_for_primary_runner,
    resolve_htr_sentence_model_snapshot,
    resolve_multi_model_forest_parallel_plan,
)
from oci.inference.multi_model_forest_stage1 import MultiModelForestStage1Runner
from oci.inference.production_stage1_scope_scheduler import derive_stage1_group_seed
from oci.models.concept_embedding_utils import chunk_text_words


_CONCEPT_CLUSTER_LABEL_PROMPT_VERSION = "multi_model_agentic_cluster_labeling_v2"
_CONCEPT_INVENTORY_SCHEMA_VERSION = "multi_model_agentic_clustered_concept_inventory_v2"
_EVIDENCE_DIGEST_ROLE_PROMPT_VERSION = "multi_model_agentic_evidence_digest_role_v1"
_TEST_PHYSICAL_GLOBAL_SEED = 1701


def _cluster_local_test_scientific(
    *,
    requested_cluster_count: int,
    maximum_components_per_family: int,
    minimum_cluster_size: int,
    minimum_group_size: int,
    minimum_cell_size: int,
) -> ClusterLocalEmbeddingScientificConfig:
    return ClusterLocalEmbeddingScientificConfig(
        requested_cluster_count=requested_cluster_count,
        cluster_count_policy="require_exact_configured_count_v1",
        maximum_components_per_family=maximum_components_per_family,
        loading_evidence_capacity=None,
        loading_evidence_overflow_policy="fail_closed_no_truncation_v1",
        minimum_cluster_size=minimum_cluster_size,
        minimum_group_size=minimum_group_size,
        minimum_cell_size=minimum_cell_size,
        minimum_distinct_local_clusters_per_family=2,
        minimum_numerical_rank_per_family=2,
        patient_pooling_policy="arithmetic_mean_all_authenticated_chunks_v1",
        computation_dtype="float32",
        normalize_patient_embeddings=True,
        normalization_epsilon=1e-12,
        zero_vector_policy="reject",
        local_direction_weighting_policy="sqrt_cluster_size_times_unit_direction_v1",
        kmeans_init="k-means++",
        kmeans_max_iter=300,
        kmeans_batch_size_policy="clamp_usable_rows_to_configured_bounds_v1",
        kmeans_batch_size_lower_bound=1,
        kmeans_batch_size_upper_bound=1024,
        kmeans_verbose=0,
        kmeans_compute_labels=True,
        kmeans_seed_derivation_policy="canonical_ordered_fit_rows_group_seed_v1",
        kmeans_tol=0.0,
        kmeans_max_no_improvement=10,
        kmeans_init_size=None,
        kmeans_n_init=20,
        kmeans_reassignment_ratio=0.01,
        svd_full_matrices=False,
        svd_compute_uv=True,
        svd_hermitian=False,
        svd_sign_canonicalization_policy="largest_absolute_coordinate_positive_v1",
        svd_rank_tolerance_policy=(
            "dtype_epsilon_times_max_shape_times_largest_singular_v1"
        ),
        svd_rank_tolerance_dtype="float32",
        svd_rank_tolerance_multiplier=1.0,
        replay_comparison_policy="allclose_and_exact_discrete_state_v1",
        replay_relative_tolerance=2e-6,
        replay_absolute_tolerance=2e-7,
        exception_policy="abort_scope_no_skip_or_fallback_v1",
    )


def _linear_test_bow_views() -> list[BoWViewConfig]:
    return [
        BoWViewConfig(
            name="linear_test",
            max_features=1000,
            min_df=1,
            max_df=1.0,
            ngram_range_min=1,
            ngram_range_max=3,
        )
    ]


def _two_linear_test_bow_views() -> list[BoWViewConfig]:
    return [
        BoWViewConfig(
            name="linear_test_unigram",
            max_features=1000,
            min_df=1,
            max_df=1.0,
            ngram_range_min=1,
            ngram_range_max=1,
        ),
        BoWViewConfig(
            name="linear_test_bigram",
            max_features=1000,
            min_df=1,
            max_df=1.0,
            ngram_range_min=1,
            ngram_range_max=2,
        ),
    ]


def _disable_htr_test_kwargs() -> dict:
    return {
        "htr_evidence_enabled": False,
        "htr_evidence_disable_reason": "unit test without HTR training",
    }


def _disable_required_evidence_test_kwargs() -> dict:
    return {
        "embedding_contrast": EmbeddingContrastDiscoveryConfig(
            enabled=False,
            disable_reason="unit test without embedding contrast",
        ),
        **_disable_htr_test_kwargs(),
    }


class FakeProposalAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        if context.get("prompt_version") == _CONCEPT_CLUSTER_LABEL_PROMPT_VERSION:
            return {
                "concepts": [
                    {
                        "name": "age",
                        "label": "Age",
                        "value_kind": "continuous",
                        "source_families": ["bow"],
                        "source_overlap": 1,
                        "supporting_phrases": ["age"],
                        "extractability": "high",
                        "cluster_ids": ["cluster_001"],
                    },
                    {
                        "name": "pd_l1_expression",
                        "label": "PD-L1 expression",
                        "value_kind": "categorical",
                        "source_families": ["bow"],
                        "source_overlap": 1,
                        "supporting_phrases": ["pd-l1"],
                        "extractability": "high",
                        "cluster_ids": ["cluster_002"],
                    },
                ]
            }
        if context.get("prompt_version") == _EVIDENCE_DIGEST_ROLE_PROMPT_VERSION:
            if context.get("target_role") == "confounder":
                return [
                    {
                        "action": "add",
                        "name": "age",
                        "type": "continuous",
                        "roles": ["confounder"],
                        "description": "Patient age at treatment initiation in years.",
                        "rationale": "Age-bearing terms appear in treatment and outcome models.",
                        "expected_signal": "treatment and outcome",
                    }
                ]
            return [
                {
                    "action": "add",
                    "name": "pd_l1_expression",
                    "type": "categorical",
                    "categories": ["low", "high", "unknown"],
                    "roles": ["effect_modifier"],
                    "description": "Pretreatment tumor PD-L1 expression category.",
                    "rationale": "PD-L1 threshold terms appear in the modifier evidence digest.",
                    "expected_signal": "R-stage or matched-pair uplift",
                },
                {
                    "action": "add",
                    "name": "pd_l1_expression_level",
                    "type": "categorical",
                    "categories": ["<1%", "1-49%", ">=50%"],
                    "roles": ["effect_modifier"],
                    "description": "Pretreatment tumor PD-L1 expression category.",
                    "rationale": "PD-L1 threshold terms appear in the modifier evidence digest.",
                    "expected_signal": "R-stage or matched-pair uplift",
                },
            ]
        if context.get("prompt_version") == "multi_model_agentic_alias_resolution_v1":
            return {
                "groups": [
                    {
                        "canonical_name": "pd_l1_expression",
                        "member_names": [
                            "pd_l1_expression",
                            "pd_l1_expression_level",
                        ],
                        "type": "categorical",
                        "categories": ["<1%", "1-49%", ">=50%"],
                        "description": "Pretreatment tumor PD-L1 expression category.",
                        "roles": ["effect_modifier"],
                        "rationale": "The two names refer to the same extraction target.",
                    }
                ],
                "unmerged": [{"name": "age", "reason": "No alias proposed."}],
            }
        if context.get("prompt_version") == "multi_model_agentic_value_harmonization_v1":
            return {
                "features": [
                    {
                        "name": "age",
                        "type": "continuous",
                        "categories": None,
                        "description": "Patient age at treatment initiation in years.",
                        "missing_values": ["unknown", "not_reported", "high", "low"],
                        "rationale": "Age should remain numeric; qualitative labels are missing.",
                    },
                    {
                        "name": "pd_l1_expression",
                        "type": "categorical",
                        "categories": ["<1%", "1-49%", ">=50%", "unknown"],
                        "description": "Pretreatment tumor PD-L1 expression category.",
                        "value_aliases": {
                            "<1%": ["low negative"],
                            ">=50%": ["high", "50% or greater"],
                        },
                        "missing_values": ["unknown", "not_reported"],
                        "rationale": "Collapse high/low aliases into threshold categories.",
                    },
                ]
            }
        return [
            {
                "action": "add",
                "name": "age",
                "type": "continuous",
                "roles": ["confounder"],
                "description": "Patient age at treatment initiation in years.",
                "rationale": "Age-bearing terms appear in treatment and outcome models.",
                "expected_signal": "treatment and outcome",
            },
            {
                "action": "add",
                "name": "pd_l1_expression",
                "type": "categorical",
                "categories": ["low", "high", "unknown"],
                "roles": ["effect_modifier"],
                "description": "Pretreatment tumor PD-L1 expression category.",
                "rationale": "PD-L1 threshold terms appear in the pseudo-target model.",
                "expected_signal": "pseudo-target",
            },
            {
                "action": "add",
                "name": "pd_l1_expression_level",
                "type": "categorical",
                "categories": ["<1%", "1-49%", ">=50%"],
                "roles": ["effect_modifier"],
                "description": "Pretreatment tumor PD-L1 expression category.",
                "rationale": "PD-L1 threshold terms appear in the pseudo-target model.",
                "expected_signal": "pseudo-target",
            },
        ]


def _agent_contexts(agent, prompt_version: str) -> list[dict]:
    return [
        context
        for context in agent.contexts
        if context.get("prompt_version") == prompt_version
    ]


def _first_agent_context(agent, prompt_version: str) -> dict:
    contexts = _agent_contexts(agent, prompt_version)
    assert contexts
    return contexts[0]


def _role_agent_context(agent, role: str) -> dict:
    contexts = [
        context
        for context in _agent_contexts(agent, _EVIDENCE_DIGEST_ROLE_PROMPT_VERSION)
        if context.get("target_role") == role
    ]
    assert contexts
    return contexts[0]


def _digest_source_label_in_blurb(blurb: str) -> bool:
    labels = [
        "BoW top features:",
        "HTR attended tokens:",
        "Retrieved text blurb:",
        "Model phrase cluster:",
    ]
    return any(label in str(blurb) for label in labels)


class ReviewRevisionAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        prompt_version = context.get("prompt_version")
        if prompt_version == "multi_model_agentic_value_harmonization_v1":
            return {"features": context.get("selected_features", [])}
        if prompt_version == "multi_model_agentic_extracted_feature_review_v1":
            return [
                {
                    "action": "add",
                    "name": "signal_marker",
                    "type": "categorical",
                    "categories": ["negative", "positive"],
                    "roles": ["confounder", "effect_modifier"],
                    "description": "Pretreatment signal marker status.",
                    "rationale": "BoW diagnostics show signal marker text captures the missed treatment and outcome signal.",
                    "expected_signal": "treatment, outcome, and pseudo-target",
                }
            ]
        return [
            {
                "action": "add",
                "name": "noise_marker",
                "type": "categorical",
                "categories": ["absent", "present"],
                "roles": ["confounder"],
                "description": "Pretreatment noise marker status.",
                "rationale": "Initial weak candidate.",
                "expected_signal": "treatment and outcome",
            }
        ]


class LowCoverageReviewAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        prompt_version = context.get("prompt_version")
        if prompt_version == "multi_model_agentic_value_harmonization_v1":
            return {"features": context.get("selected_features", [])}
        if prompt_version == "multi_model_agentic_extracted_feature_review_v1":
            low_coverage = context.get("low_coverage_features_needing_broader_targets", [])
            assert low_coverage
            assert low_coverage[0]["name"] == "rare_signal_phrase"
            assert low_coverage[0]["coverage"] < low_coverage[0]["required_min_coverage"]
            assert "broader" in context["review_policy"]["low_coverage_feature_policy"]
            return [
                {
                    "action": "add",
                    "name": "signal_marker",
                    "type": "categorical",
                    "categories": ["negative", "positive"],
                    "roles": ["confounder", "effect_modifier"],
                    "description": "Broader pretreatment signal marker status.",
                    "rationale": (
                        "The rare phrasing had low extraction coverage; broader "
                        "signal text is present across notes."
                    ),
                    "expected_signal": "treatment, outcome, and pseudo-target",
                }
            ]
        return [
            {
                "action": "add",
                "name": "rare_signal_phrase",
                "type": "categorical",
                "categories": ["absent", "present"],
                "roles": ["confounder", "effect_modifier"],
                "description": "Very narrow rare wording for signal marker status.",
                "rationale": "Initial narrow extraction target.",
                "expected_signal": "treatment and outcome",
            }
        ]


class ParsimonyFactorAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        prompt_version = context.get("prompt_version")
        if prompt_version == "multi_model_agentic_parsimony_factor_v1":
            return {
                "cluster_id": context["cluster_id"],
                "decision": "replace_cluster",
                "replaces": list(context["replaceable_members"]),
                "factors": [
                    {
                        "name": "latent_functional_burden",
                        "inference_kind": "implicit",
                        "type": "categorical",
                        "categories": ["low", "high"],
                        "roles": list(context["required_role_union"]),
                        "description": "Overall pretreatment functional burden.",
                        "supporting_indicators": [
                            "multiple concordant functional limitations"
                        ],
                        "contrary_indicators": ["documented normal function"],
                        "minimum_evidence": "at least two concordant indicators",
                        "null_policy": "return null with fewer than two indicators",
                        "rationale": "The member values are empirically redundant measures of function.",
                    }
                ],
                "rationale": "One operational burden factor represents the coherent cluster.",
            }
        if prompt_version == "multi_model_agentic_value_harmonization_v1":
            return {"features": context.get("selected_features", [])}
        if prompt_version == "multi_model_agentic_alias_resolution_v1":
            return {"groups": [], "unmerged": []}
        return []


class ParsimonyFactorExtractionProvider:
    reads_complete_documents = True

    def ensure_features(self, dataset, specs):
        dataset = dataset.copy()
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if spec.name == "latent_functional_burden":
                dataset[value_col] = np.where(
                    dataset["explicit_feat_function_a"].to_numpy(dtype=float) >= 0.5,
                    "high",
                    "low",
                )
                dataset[missing_col] = False
        return dataset


class FakeExtractionProvider:
    def ensure_features(self, dataset, specs):
        dataset = dataset.copy()
        text = dataset["clinical_text"].astype(str)
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if spec.name == "age":
                dataset[value_col] = text.str.extract(r"age (\d+)").astype(float)
            elif spec.name == "pd_l1_expression":
                dataset[value_col] = np.where(
                    text.str.contains(">=50%"),
                    ">=50%",
                    np.where(text.str.contains("1-49%"), "1-49%", "<1%"),
                )
            else:
                dataset[value_col] = np.nan
            dataset[missing_col] = dataset[value_col].isna()
        return dataset


class ReviewExtractionProvider:
    def __init__(self):
        self.calls = []

    def ensure_features(self, dataset, specs):
        self.calls.append([spec.name for spec in specs])
        dataset = dataset.copy()
        text = dataset["clinical_text"].astype(str)
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if spec.name == "noise_marker":
                dataset[value_col] = "present"
            elif spec.name == "signal_marker":
                dataset[value_col] = np.where(
                    text.str.contains("signal positive"),
                    "positive",
                    "negative",
                )
            else:
                dataset[value_col] = np.nan
            dataset[missing_col] = dataset[value_col].isna()
        return dataset


class LowCoverageExtractionProvider:
    def __init__(self):
        self.calls = []

    def ensure_features(self, dataset, specs):
        self.calls.append([spec.name for spec in specs])
        dataset = dataset.copy()
        text = dataset["clinical_text"].astype(str)
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if spec.name == "rare_signal_phrase":
                values = pd.Series(np.nan, index=dataset.index, dtype=object)
                values.loc[text.str.contains("rare signal phrase")] = "present"
                dataset[value_col] = values
            elif spec.name == "signal_marker":
                values = pd.Series(np.nan, index=dataset.index, dtype=object)
                values.loc[text.str.contains("signal positive")] = "positive"
                values.loc[text.str.contains("signal negative")] = "negative"
                dataset[value_col] = values
            else:
                dataset[value_col] = np.nan
            dataset[missing_col] = dataset[value_col].isna()
        return dataset


class FakeEvaluator:
    def __init__(self):
        self.seen_specs = []

    def evaluate_split(self, train_df, test_df, specs, fold_id):
        self.seen_specs.append(specs)
        predictions = test_df.copy()
        predictions["pred_ite_prob"] = 0.1
        predictions["pred_y0_prob"] = 0.4
        predictions["pred_y1_prob"] = 0.5
        predictions["pred_propensity_prob"] = 0.5
        predictions["pred_outcome_prob"] = 0.5
        predictions["cv_fold"] = fold_id
        metrics = {
            "fold": fold_id,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_explicit_features": len(specs),
        }
        return SplitEvaluation(predictions=predictions, metrics=metrics)


class FakeHTREvidenceProvider:
    def __init__(self):
        self.seen_effect_nuisance_predictions = []

    def fit_nuisance(self, discovery_df, outer_fold):
        y = discovery_df["outcome_indicator"].to_numpy(dtype=float)
        t = discovery_df["treatment_indicator"].to_numpy(dtype=float)
        e_hat = np.clip(0.25 + 0.45 * t, 0.05, 0.95)
        m_hat = np.clip(0.20 + 0.50 * y, 0.05, 0.95)
        y_resid = y - m_hat
        t_resid = t - e_hat
        predictions = pd.DataFrame(
            {
                "_oci_row_id": discovery_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "e_hat": e_hat,
                "m_hat": m_hat,
                "y_residual": y_resid,
                "t_residual": t_resid,
                "r_pseudo_outcome": y_resid / np.where(np.abs(t_resid) < 1e-6, 1e-6, t_resid),
                "r_loss_at_zero_tau": y_resid**2,
                "nuisance_fold": 1,
            }
        )
        return {
            "predictions": predictions,
            "attention": self._attention_rows(
                discovery_df, outer_fold, "nuisance", e_hat=e_hat, m_hat=m_hat
            ),
        }

    def fit_effect(self, discovery_df, nuisance_predictions, outer_fold):
        self.seen_effect_nuisance_predictions.append(nuisance_predictions.copy())
        predictions = nuisance_predictions.copy()
        tau_hat = np.linspace(0.10, 0.35, len(predictions))
        y_resid = predictions["y_residual"].to_numpy(dtype=float)
        t_resid = predictions["t_residual"].to_numpy(dtype=float)
        predictions["tau_hat_r_stage"] = tau_hat
        predictions["tau_logit_modifier"] = np.nan
        predictions["r_loss"] = (y_resid - tau_hat * t_resid) ** 2
        predictions["effect_loss"] = predictions["r_loss"]
        predictions["effect_loss_at_zero_tau"] = y_resid**2
        predictions["effect_fold"] = 1
        predictions["effect_objective"] = "squared_r_loss"
        predictions["r_stage_train_eligible"] = True
        return {
            "predictions": predictions,
            "attention": self._attention_rows(
                discovery_df, outer_fold, "effect_modifier", tau_hat_r_stage=tau_hat
            ),
        }

    def _attention_rows(self, discovery_df, outer_fold, stage, **extra):
        rows = []
        for offset, row in discovery_df.head(4).reset_index(drop=True).iterrows():
            text = str(row["clinical_text"])
            token = text.split()[0] if text.split() else "note"
            start = max(0, text.find(token))
            evidence = {
                "row_id": int(row["_oci_row_id"]),
                "outer_fold": int(outer_fold),
                "fold": 1,
                "stage": stage,
                "chunk_index": 0,
                "chunk_text": text,
                "attended_token_summary": token,
                "top_token_spans_json": json.dumps(
                    [
                        {
                            "text": token,
                            "focus_token": token,
                            "char_start": start,
                            "char_end": start + len(token),
                            "salience": 0.9,
                        }
                    ]
                ),
            }
            for key, values in extra.items():
                evidence[key] = values[offset]
            rows.append(evidence)
        return rows


class EmptyProposalAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        if context.get("prompt_version") == "multi_model_agentic_value_harmonization_v1":
            return {"features": context.get("selected_features", [])}
        return []


class FailingConsistencyAgent:
    def propose(self, context):
        if context.get("prompt_version") == "multi_model_agentic_consistency_v1":
            raise RuntimeError("simulated consistency agent failure")
        return []


class SelectingConsistencyAgent:
    def __init__(self, names):
        self.names = list(names)
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        if context.get("prompt_version") != "multi_model_agentic_consistency_v1":
            return []
        return {
            "proposals": [
                {
                    "action": "add",
                    "name": name,
                    "rationale": "Selected from candidate_summaries.",
                }
                for name in self.names
            ]
        }


class RecordingExtractionProvider:
    def __init__(self):
        self.calls = []

    def ensure_features(self, dataset, specs):
        self.calls.append([(spec.name, tuple(spec.roles)) for spec in specs])
        dataset = dataset.copy()
        text = dataset["clinical_text"].astype(str)
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if spec.name == "age":
                dataset[value_col] = text.str.extract(r"age (\d+)").astype(float)
            elif spec.name == "biomarker":
                dataset[value_col] = np.where(
                    text.str.contains("biomarker positive"),
                    "positive",
                    "negative",
                )
            else:
                dataset[value_col] = np.nan
            dataset[missing_col] = dataset[value_col].isna()
        return dataset


class KeywordEmbeddingProvider:
    def encode_chunks(self, texts):
        rows = []
        for text in texts:
            lower = str(text).lower()
            row = np.zeros(8, dtype=np.float32)
            for token, index in {
                "brain": 0,
                "liver": 1,
                "cachexia": 2,
                "pd-l1": 3,
                "pdl1": 3,
                "high": 4,
                "low": 5,
                "age": 6,
            }.items():
                if token in lower:
                    row[index] += max(1, lower.count(token))
            row[7] = 0.1
            rows.append(row)
        return np.vstack(rows)


class KeywordSentenceTransformer:
    class _WhitespaceTokenizer:
        def __init__(self):
            self._token_to_id = {}
            self._id_to_token = {}

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            token_ids = []
            for token in str(text).split():
                token_id = self._token_to_id.get(token)
                if token_id is None:
                    token_id = len(self._token_to_id) + 1
                    self._token_to_id[token] = token_id
                    self._id_to_token[token_id] = token
                token_ids.append(token_id)
            return token_ids

        def decode(
            self,
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ):
            del skip_special_tokens, clean_up_tokenization_spaces
            return " ".join(self._id_to_token[int(token_id)] for token_id in token_ids)

        @staticmethod
        def num_special_tokens_to_add(pair=False):
            del pair
            return 0

    def __init__(self):
        # The production cache must be able to audit and, if needed, split
        # every model-bounded input.  Keep this test double subject to the
        # same contract instead of relying on an implicit truncating encoder.
        self.tokenizer = self._WhitespaceTokenizer()

    def encode(
        self,
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        del batch_size, convert_to_numpy, show_progress_bar
        embeddings = KeywordEmbeddingProvider().encode_chunks(texts)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings.astype(np.float32, copy=False)


def test_concept_cluster_sentence_transformer_is_reused_and_released(
    tmp_path: Path,
    monkeypatch,
):
    instances = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, device=None, cache_folder=None):
            self.model_name = model_name
            self.device = device
            self.cache_folder = cache_folder
            self.to_calls = []
            instances.append(self)

        def encode(
            self,
            texts,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ):
            del batch_size, convert_to_numpy, normalize_embeddings, show_progress_bar
            return np.ones((len(texts), 4), dtype=np.float32)

        def to(self, device):
            self.to_calls.append(device)
            return self

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    config = AppliedInferenceConfig(
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                feature_discovery_methods=["embedding_contrast"],
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=True,
                    model_name="fake-concept-encoder",
                    cache_dir=str(tmp_path / "embedding_cache"),
                    device="cuda:7",
                    batch_size=2,
                ),
                **_disable_htr_test_kwargs(),
            ),
        ),
    )
    runner = MultiModelAgenticForestRunner(
        dataset=pd.DataFrame(
            {
                "clinical_text": ["age 70", "age 71"],
                "treatment_indicator": [0, 1],
                "outcome_indicator": [0, 1],
            }
        ),
        config=config,
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=object(),
        extraction_provider=object(),
        evaluator=object(),
    )

    first, first_error = runner._encode_concept_cluster_texts(["age 70", "ecog 1"])
    second, second_error = runner._encode_concept_cluster_texts(["albumin 3.1"])

    assert first_error is None
    assert second_error is None
    assert first.shape == (2, 4)
    assert second.shape == (1, 4)
    assert len(instances) == 1
    assert instances[0].model_name == "fake-concept-encoder"
    assert instances[0].device == "cuda:7"

    runner._release_concept_cluster_embedding_encoder()

    assert runner._concept_cluster_embedding_encoder is None
    assert instances[0].to_calls == ["cpu"]


def _embedding_contrast_cache_test_config(tmp_path: Path) -> AppliedInferenceConfig:
    return AppliedInferenceConfig(
        dataset_path=str(tmp_path / "cohort.parquet"),
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=True,
                    model_name="fake-keyword",
                    cache_dir=str(tmp_path / "embedding_cache"),
                    chunk_size_words=4,
                    chunk_overlap_words=0,
                    max_chunks=2,
                    min_probe_auc=0.0,
                    top_k_chunks_per_tail=3,
                    max_chunks_per_patient=1,
                    include_cluster_contrast_vectors=False,
                    concept_phrases=["brain metastases", "liver lesion"],
                    include_bow_phrases_as_concepts=False,
                ),
                **_disable_htr_test_kwargs(),
            ),
        ),
    )


def _embedding_contrast_cache_test_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_oci_row_id": np.arange(8),
            "clinical_text": [
                "brain metastases pd-l1 high",
                "brain metastases pd-l1 high",
                "liver lesion pd-l1 low",
                "liver lesion pd-l1 low",
                "brain metastases cachexia pd-l1 high",
                "liver lesion stable disease",
                "brain metastases cachexia",
                "liver lesion stable disease",
            ],
        }
    )


def _build_embedding_contrast_cache_test_evidence(
    generator: EmbeddingContrastEvidenceGenerator,
    dataset: pd.DataFrame,
):
    _bind_embedding_test_physical_fit_authority(generator, dataset)
    return generator.build_evidence(
        discovery_df=dataset,
        y=np.asarray([0, 1, 0, 0, 1, 0, 1, 0], dtype=float),
        t=np.asarray([1, 1, 0, 0, 1, 0, 1, 0], dtype=float),
        pseudo_target=np.asarray([1, 1, -1, -1, 1, -1, 1, -1], dtype=float),
        t_resid=np.ones(8, dtype=float),
        importance={},
    )


def _bind_embedding_test_physical_fit_authority(
    generator: EmbeddingContrastEvidenceGenerator,
    dataset: pd.DataFrame,
) -> None:
    rows = tuple(dataset["_oci_row_id"].astype(int).tolist())
    generator.bind_cluster_physical_fit_authority(
        ordered_fit_row_ids=rows,
        canonical_group_seed=derive_stage1_group_seed(
            _TEST_PHYSICAL_GLOBAL_SEED,
            rows,
        ),
    )


def test_embedding_contrast_retrieval_filters_low_content_chunks():
    assert not _informative_chunk_text("")
    assert not _informative_chunk_text("--- ### <new_note> ---")
    assert _informative_chunk_text("Brain MRI shows enhancing metastases.")


def test_embedding_contrast_chunk_capacity_fails_closed():
    from oci.models.lossless_tokenization import SemanticTruncationError

    text = " ".join(f"w{i}" for i in range(1, 11))
    for selection in ("first", "last"):
        with pytest.raises(
            SemanticTruncationError,
            match=r"requires 4 chunks.*max_chunks=2",
        ):
            chunk_text_words(
                text,
                chunk_size_words=3,
                chunk_overlap_words=0,
                max_chunks=2,
                chunk_selection=selection,
            )


def test_embedding_contrast_default_cache_dir_is_dataset_scoped(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    parquet_path = dataset_dir / "cohort.parquet"
    assert (
        _default_embedding_cache_dir(
            str(parquet_path),
            tmp_path / "run",
        )
        == dataset_dir / ".oci_cache" / "embedding_contrast"
    )
    assert (
        _default_embedding_cache_dir(
            str(dataset_dir),
            tmp_path / "run",
        )
        == dataset_dir / ".oci_cache" / "embedding_contrast"
    )


def test_embedding_contrast_reuses_warm_chunk_and_concept_phrase_caches_without_model_load(
    tmp_path: Path,
    monkeypatch,
):
    dataset = _embedding_contrast_cache_test_dataset()
    config = _embedding_contrast_cache_test_config(tmp_path)
    load_calls = []

    def fake_load_sentence_transformer(model_name, device=None, max_seq_length=None):
        del max_seq_length
        load_calls.append((model_name, str(device)))
        return KeywordSentenceTransformer()

    monkeypatch.setattr(
        "oci.models.concept_embedding_cache.load_sentence_transformer",
        fake_load_sentence_transformer,
    )
    monkeypatch.setattr(
        "oci.inference.embedding_contrast_discovery.load_sentence_transformer",
        fake_load_sentence_transformer,
    )

    generator = EmbeddingContrastEvidenceGenerator(config=config, output_dir=tmp_path)
    generator.prepare(dataset)
    evidence = _build_embedding_contrast_cache_test_evidence(generator, dataset)
    assert "concept_probe_skipped" not in evidence
    treatment = next(item for item in evidence["contrasts"] if item["name"] == "treatment")
    assert treatment["concept_probe_scores"]
    assert load_calls

    def fail_load_sentence_transformer(model_name, device=None, max_seq_length=None):
        del model_name, device, max_seq_length
        raise AssertionError("sentence-transformer should not load on warm caches")

    monkeypatch.setattr(
        "oci.models.concept_embedding_cache.load_sentence_transformer",
        fail_load_sentence_transformer,
    )
    monkeypatch.setattr(
        "oci.inference.embedding_contrast_discovery.load_sentence_transformer",
        fail_load_sentence_transformer,
    )

    warm_generator = EmbeddingContrastEvidenceGenerator(
        config=config,
        output_dir=tmp_path,
    )
    warm_generator.prepare(dataset)
    warm_evidence = _build_embedding_contrast_cache_test_evidence(
        warm_generator,
        dataset,
    )
    assert "concept_probe_skipped" not in warm_evidence
    warm_treatment = next(
        item for item in warm_evidence["contrasts"] if item["name"] == "treatment"
    )
    assert warm_treatment["concept_probe_scores"]


def test_embedding_contrast_skips_concept_probe_load_when_only_chunk_cache_is_warm(
    tmp_path: Path,
    monkeypatch,
):
    dataset = _embedding_contrast_cache_test_dataset()
    config = _embedding_contrast_cache_test_config(tmp_path)
    load_calls = []

    def fake_load_sentence_transformer(model_name, device=None, max_seq_length=None):
        del max_seq_length
        load_calls.append((model_name, str(device)))
        return KeywordSentenceTransformer()

    monkeypatch.setattr(
        "oci.models.concept_embedding_cache.load_sentence_transformer",
        fake_load_sentence_transformer,
    )
    generator = EmbeddingContrastEvidenceGenerator(config=config, output_dir=tmp_path)
    generator.prepare(dataset)
    assert load_calls

    def fail_load_sentence_transformer(model_name, device=None, max_seq_length=None):
        del model_name, device, max_seq_length
        raise AssertionError("sentence-transformer should not load for optional probes")

    monkeypatch.setattr(
        "oci.models.concept_embedding_cache.load_sentence_transformer",
        fail_load_sentence_transformer,
    )
    monkeypatch.setattr(
        "oci.inference.embedding_contrast_discovery.load_sentence_transformer",
        fail_load_sentence_transformer,
    )

    warm_generator = EmbeddingContrastEvidenceGenerator(
        config=config,
        output_dir=tmp_path,
    )
    warm_generator.prepare(dataset)
    evidence = _build_embedding_contrast_cache_test_evidence(warm_generator, dataset)
    assert evidence["concept_probe_skipped"] == "concept_phrase_cache_miss_on_warm_chunk_cache"
    treatment = next(item for item in evidence["contrasts"] if item["name"] == "treatment")
    assert treatment["concept_probe_scores"] == []


def test_embedding_contrast_evidence_retrieves_aligned_chunks(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "_oci_row_id": np.arange(8),
            "clinical_text": [
                "brain metastases pd-l1 high",
                "brain metastases pd-l1 high",
                "liver lesion pd-l1 low",
                "liver lesion pd-l1 low",
                "brain metastases cachexia pd-l1 high",
                "liver lesion stable disease",
                "brain metastases cachexia",
                "liver lesion stable disease",
            ],
            "site": ["a", "a", "b", "b", "a", "b", "a", "b"],
        }
    )
    config = AppliedInferenceConfig(
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=True,
                    model_name="fake-keyword",
                    chunk_size_words=4,
                    chunk_overlap_words=0,
                    max_chunks=2,
                    min_probe_auc=0.0,
                    top_k_chunks_per_tail=3,
                    max_chunks_per_patient=1,
                    include_cluster_contrast_vectors=False,
                    concept_phrases=["brain metastases", "liver lesion"],
                ),
                **_disable_htr_test_kwargs(),
            ),
        ),
    )
    generator = EmbeddingContrastEvidenceGenerator(
        config=config,
        output_dir=tmp_path,
        embedding_provider=KeywordEmbeddingProvider(),
    )
    generator.prepare(dataset)
    _bind_embedding_test_physical_fit_authority(generator, dataset)
    evidence = generator.build_evidence(
        discovery_df=dataset,
        y=np.asarray([0, 1, 0, 0, 1, 0, 1, 0], dtype=float),
        t=np.asarray([1, 1, 0, 0, 1, 0, 1, 0], dtype=float),
        pseudo_target=np.asarray([1, 1, -1, -1, 1, -1, 1, -1], dtype=float),
        t_resid=np.ones(8, dtype=float),
        importance={"phrase_features": [{"feature": "pd-l1 high"}]},
    )

    treatment = next(item for item in evidence["contrasts"] if item["name"] == "treatment")
    assert treatment["direction_source"] == "mean_difference"
    assert treatment["probe_auc_role"] == "diagnostic_only"
    assert any("brain" in row["text"] for row in treatment["positive_aligned_chunks"])
    assert any("liver" in row["text"] for row in treatment["negative_aligned_chunks"])
    assert any(row["concept"] == "brain metastases" for row in treatment["concept_probe_scores"])

    redacted = redact_embedding_contrast_evidence(evidence)
    redacted_treatment = next(item for item in redacted["contrasts"] if item["name"] == "treatment")
    assert all(row["text"] is None for row in redacted_treatment["positive_aligned_chunks"])
    assert all(
        row["text_redacted"] is True for row in redacted_treatment["positive_aligned_chunks"]
    )


def test_embedding_contrast_adds_cell_and_orthogonal_r_score_contrasts(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "_oci_row_id": np.arange(8),
            "clinical_text": [
                "brain metastases pd-l1 high",
                "brain metastases pd-l1 high",
                "liver lesion pd-l1 low",
                "liver lesion pd-l1 low",
                "liver lesion pd-l1 low",
                "liver lesion pd-l1 low",
                "brain metastases pd-l1 high",
                "brain metastases pd-l1 high",
            ],
        }
    )
    config = AppliedInferenceConfig(
        seed=_TEST_PHYSICAL_GLOBAL_SEED,
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=True,
                    model_name="fake-keyword",
                    chunk_size_words=4,
                    chunk_overlap_words=0,
                    max_chunks=1,
                    min_probe_auc=0.0,
                    top_k_chunks_per_tail=3,
                    max_chunks_per_patient=1,
                    include_cluster_contrast_vectors=False,
                    concept_phrases=["brain metastases", "liver lesion"],
                ),
                **_disable_htr_test_kwargs(),
            ),
        ),
    )
    generator = EmbeddingContrastEvidenceGenerator(
        config=config,
        output_dir=tmp_path,
        embedding_provider=KeywordEmbeddingProvider(),
    )
    generator.prepare(dataset)
    _bind_embedding_test_physical_fit_authority(generator, dataset)
    evidence = generator.build_evidence(
        discovery_df=dataset,
        y=np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=float),
        t=np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=float),
        pseudo_target=np.asarray([2, 2, -2, -2, -2, -2, 2, 2], dtype=float),
        t_resid=np.ones(8, dtype=float),
        importance={},
    )

    contrast_names = {item["name"] for item in evidence["contrasts"]}
    assert {
        "treated_outcome",
        "untreated_outcome",
        "treatment_outcome_interaction",
        "residualized_treatment_outcome_interaction",
        "orthogonal_r_score",
    }.issubset(contrast_names)

    interaction = next(
        item for item in evidence["contrasts"] if item["name"] == "treatment_outcome_interaction"
    )
    assert interaction["contrast_family"] == "treatment_outcome_cell_interaction"
    assert interaction["direction_source"] == "cell_mean_difference_in_differences"
    assert {row["n"] for row in interaction["component_counts"]} == {2}
    assert any("brain" in row["text"] for row in interaction["positive_aligned_chunks"])
    assert any("liver" in row["text"] for row in interaction["negative_aligned_chunks"])

    orthogonal = next(
        item for item in evidence["contrasts"] if item["name"] == "orthogonal_r_score"
    )
    assert orthogonal["contrast_family"] == "orthogonal_r_score"
    assert "(Y - m_hat) * (T - e_hat)" in orthogonal["score_formula"]
    assert any("brain" in row["text"] for row in orthogonal["positive_aligned_chunks"])

    residualized = next(
        item
        for item in evidence["contrasts"]
        if item["name"] == "residualized_treatment_outcome_interaction"
    )
    assert residualized["contrast_family"] == ("residualized_treatment_outcome_cell_interaction")
    assert residualized["projection_basis"] == ["treatment", "outcome"]
    assert any("brain" in row["text"] for row in residualized["positive_aligned_chunks"])
    assert any("liver" in row["text"] for row in residualized["negative_aligned_chunks"])


def test_embedding_contrast_residualized_interaction_does_not_require_r_targets(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "_oci_row_id": np.arange(8),
            "clinical_text": [
                "brain metastases pd-l1 high",
                "brain metastases pd-l1 high",
                "liver lesion pd-l1 low",
                "liver lesion pd-l1 low",
                "liver lesion pd-l1 low",
                "liver lesion pd-l1 low",
                "brain metastases pd-l1 high",
                "brain metastases pd-l1 high",
            ],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=True,
                    model_name="fake-keyword",
                    chunk_size_words=4,
                    chunk_overlap_words=0,
                    max_chunks=1,
                    min_probe_auc=0.0,
                    top_k_chunks_per_tail=3,
                    max_chunks_per_patient=1,
                    include_cluster_contrast_vectors=False,
                ),
                **_disable_htr_test_kwargs(),
            ),
        ),
    )
    generator = EmbeddingContrastEvidenceGenerator(
        config=config,
        output_dir=tmp_path,
        embedding_provider=KeywordEmbeddingProvider(),
    )
    generator.prepare(dataset)
    _bind_embedding_test_physical_fit_authority(generator, dataset)
    evidence = generator.build_evidence(
        discovery_df=dataset,
        y=np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=float),
        t=np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=float),
        pseudo_target=None,
        t_resid=None,
        importance={},
    )
    contrast_names = {item["name"] for item in evidence["contrasts"]}
    assert "r_pseudo_target" not in contrast_names
    residualized = next(
        item
        for item in evidence["contrasts"]
        if item["name"] == "residualized_treatment_outcome_interaction"
    )
    assert "retrieval_skipped" not in residualized
    assert any("brain" in row["text"] for row in residualized["positive_aligned_chunks"])


def test_embedding_contrast_adds_cluster_local_contrast_components(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "_oci_row_id": np.arange(16),
            "clinical_text": [
                "brain brain brain pd-l1 high",
                "brain brain brain pd-l1 high",
                "brain brain brain pd-l1 low",
                "brain brain brain pd-l1 low",
                "brain brain brain cachexia high",
                "brain brain brain cachexia high",
                "brain brain brain stable low",
                "brain brain brain stable low",
                "liver liver liver pd-l1 high",
                "liver liver liver pd-l1 high",
                "liver liver liver pd-l1 low",
                "liver liver liver pd-l1 low",
                "liver liver liver age high",
                "liver liver liver age high",
                "liver liver liver stable low",
                "liver liver liver stable low",
            ],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=True,
                    model_name="fake-keyword",
                    chunk_size_words=6,
                    chunk_overlap_words=0,
                    max_chunks=1,
                    min_probe_auc=0.0,
                    top_k_chunks_per_tail=4,
                    max_chunks_per_patient=1,
                    include_cluster_contrast_vectors=True,
                    cluster_contrast_n_clusters=2,
                    cluster_contrast_max_components=2,
                    cluster_contrast_min_cluster_size=4,
                    cluster_contrast_min_group_size=2,
                    cluster_contrast_min_cell_size=1,
                    cluster_contrast_top_loadings=2,
                    cluster_contrast_random_state=7,
                    cluster_local_scientific=_cluster_local_test_scientific(
                        requested_cluster_count=2,
                        maximum_components_per_family=2,
                        minimum_cluster_size=4,
                        minimum_group_size=2,
                        minimum_cell_size=1,
                    ),
                ),
                **_disable_htr_test_kwargs(),
            ),
        ),
    )
    generator = EmbeddingContrastEvidenceGenerator(
        config=config,
        output_dir=tmp_path,
        embedding_provider=KeywordEmbeddingProvider(),
    )
    generator.prepare(dataset)
    _bind_embedding_test_physical_fit_authority(generator, dataset)
    evidence = generator.build_evidence(
        discovery_df=dataset,
        y=np.asarray([1, 1, 0, 0, 1, 1, 0, 0] * 2, dtype=float),
        t=np.asarray([1, 1, 1, 1, 0, 0, 0, 0] * 2, dtype=float),
        pseudo_target=None,
        t_resid=None,
        importance={},
    )

    summary = evidence["cluster_contrast_vectors"]
    assert summary["n_clusters"] == 2
    assert summary["usable_treatment_local_contrasts"] >= 2
    cluster_treatment = [
        item
        for item in evidence["contrasts"]
        if item["contrast_family"] == "cluster_local_treatment_contrast_basis"
    ]
    assert cluster_treatment
    first = cluster_treatment[0]
    assert first["direction_source"] == "svd_of_cluster_local_treatment_mean_differences"
    assert first["local_contrast_count"] >= 2
    assert first["cluster_component_loadings"]
    assert "retrieval_skipped" not in first
    assert first["positive_aligned_chunks"] or first["negative_aligned_chunks"]


def test_embedding_contrast_retrieves_external_chunks(tmp_path: Path):
    external_cache = tmp_path / "external_cache"
    external_cache.mkdir()
    embeddings = KeywordEmbeddingProvider().encode_chunks(
        ["external brain metastases review", "external liver lesion review"]
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    np.save(external_cache / "chunk_embeddings.npy", embeddings.astype(np.float32))
    np.save(external_cache / "offsets.npy", np.asarray([0, 1, 2], dtype=np.int64))
    (external_cache / "chunk_texts.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"chunks": ["external brain metastases review"]}),
                json.dumps({"chunks": ["external liver lesion review"]}),
            ]
        )
        + "\n"
    )
    (external_cache / "metadata.json").write_text(
        json.dumps({"corpus_name": "pubmed_smoke", "hidden_size": 8})
    )
    (external_cache / "row_metadata.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"row_index": 0, "metadata": {"title": "Brain review"}}),
                json.dumps({"row_index": 1, "metadata": {"title": "Liver review"}}),
            ]
        )
        + "\n"
    )
    dataset = _embedding_contrast_cache_test_dataset()
    config = _embedding_contrast_cache_test_config(tmp_path)
    config.architecture.multi_model_agentic_forest.embedding_contrast.external_corpus_cache_dirs = [
        str(external_cache)
    ]
    config.architecture.multi_model_agentic_forest.embedding_contrast.external_top_k_chunks_per_tail = (
        2
    )
    generator = EmbeddingContrastEvidenceGenerator(
        config=config,
        output_dir=tmp_path,
        embedding_provider=KeywordEmbeddingProvider(),
    )
    generator.prepare(dataset)
    evidence = _build_embedding_contrast_cache_test_evidence(generator, dataset)
    treatment = next(item for item in evidence["contrasts"] if item["name"] == "treatment")
    assert any(row["corpus"] == "pubmed_smoke" for row in treatment["positive_external_chunks"])
    assert any("external brain" in row["text"] for row in treatment["positive_external_chunks"])


def test_multi_model_agentic_forest_runs_with_fake_agent_and_extractor(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 baseline note pd-l1 >=50% high marker",
                "age 78 baseline note pd-l1 <1% low marker",
                "age 57 baseline note pd-l1 >=50% high marker",
                "age 76 baseline note pd-l1 <1% low marker",
                "age 61 baseline note pd-l1 1-49% intermediate marker",
                "age 81 baseline note pd-l1 <1% low marker",
                "age 54 baseline note pd-l1 >=50% high marker",
                "age 70 baseline note pd-l1 1-49% intermediate marker",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 0, 0, 1, 0],
            "true_ite_prob": [0.3, 0.0, 0.3, 0.0, -0.1, 0.0, 0.3, -0.1],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                max_iterations=1,
                max_additions_per_iter=4,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_linear_test_bow_views(),
                top_n_features=5,
                candidate_consistency_enabled=False,
                fold_parallelism="2",
                **_disable_required_evidence_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = FakeProposalAgent()
    evaluator = FakeEvaluator()
    output_path = tmp_path / "predictions.parquet"

    run_multi_model_agentic_forest(
        dataset,
        config,
        output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=evaluator,
    )

    predictions = pd.read_parquet(output_path)
    assert len(predictions) == len(dataset)
    assert "selected_feature_names" in predictions.columns
    assert "selected_feature_roles" in predictions.columns
    assert "selected_confounder_names" in predictions.columns
    assert "selected_effect_modifier_names" in predictions.columns
    assert set(predictions["honest_outer_holdout"]) == {True}
    assert set(predictions["estimation_provenance"]) == {"honest_outer_fold"}
    assert set(predictions["selected_feature_roles"]) == {
        "age[confounder],pd_l1_expression[effect_modifier]"
    }
    assert set(predictions["selected_confounder_names"]) == {"age"}
    assert set(predictions["selected_effect_modifier_names"]) == {"pd_l1_expression"}
    assert agent.contexts
    first_context = _role_agent_context(agent, "confounder")
    modifier_context = _role_agent_context(agent, "effect_modifier")
    assert not _agent_contexts(agent, _CONCEPT_CLUSTER_LABEL_PROMPT_VERSION)
    assert _agent_contexts(agent, "multi_model_agentic_alias_resolution_v1")
    assert _agent_contexts(agent, "multi_model_agentic_value_harmonization_v1")
    assert first_context["prompt_version"] == _EVIDENCE_DIGEST_ROLE_PROMPT_VERSION
    assert first_context["target_role"] == "confounder"
    assert modifier_context["target_role"] == "effect_modifier"
    assert "feature_importance" not in first_context
    assert "concept_inventory" not in first_context
    assert "evidence" not in first_context
    assert first_context["text_blurbs"]
    assert any("ecog" in blurb.lower() or "age" in blurb.lower() for blurb in first_context["text_blurbs"])
    assert not any(_digest_source_label_in_blurb(blurb) for blurb in first_context["text_blurbs"])
    assert modifier_context["text_blurbs"]
    rendered_prompt = build_agent_prompt(first_context, config.architecture.agentic_feature_search)
    assert "Text blurbs:" in rendered_prompt
    assert "Current nested-CV context" not in rendered_prompt
    assert '"target_role"' not in rendered_prompt
    assert '"roles"' not in rendered_prompt
    assert "canonical_feature_name_guidance" not in first_context
    assert "true_" not in json.dumps(first_context)
    seen_names = [[spec.name for spec in specs] for specs in evaluator.seen_specs]
    assert all({"age", "pd_l1_expression"}.issubset(set(names)) for names in seen_names)
    assert all(names.count("pd_l1_expression") == 1 for names in seen_names)
    pdl1_specs = [
        spec for specs in evaluator.seen_specs for spec in specs if spec.name == "pd_l1_expression"
    ]
    assert pdl1_specs
    assert all(spec.categories == ["<1%", "1-49%", ">=50%"] for spec in pdl1_specs)
    assert all("unknown" not in spec.categories for spec in pdl1_specs)
    assert all(spec.value_aliases[">=50%"] == ["high", "50% or greater"] for spec in pdl1_specs)
    age_specs = [spec for specs in evaluator.seen_specs for spec in specs if spec.name == "age"]
    assert age_specs
    assert all(spec.type == "continuous" and spec.categories is None for spec in age_specs)
    assert all("numeric value only" in (spec.description or "") for spec in age_specs)
    artifact_dir = output_path.parent / "multi_model_agentic_forest"
    assert (artifact_dir / "bow_view_oof_predictions.parquet").exists()
    assert (artifact_dir / "bow_view_feature_importance_by_fold.jsonl").exists()
    assert (artifact_dir / "agent_candidate_proposals.jsonl").exists()
    assert (artifact_dir / "report.txt").exists()
    assert (artifact_dir / "dataset_summary.json").exists()
    assert (artifact_dir / "split_provenance.jsonl").exists()
    assert (artifact_dir / "text_evidence.bow.jsonl").exists()
    assert (artifact_dir / "candidate_features.parquet").exists()
    assert (artifact_dir / "candidate_signal_review.jsonl").exists()
    assert (artifact_dir / "ite_estimates.parquet").exists()
    for fold in (1, 2):
        fold_dir = artifact_dir / f"outer_fold_{fold:03d}"
        assert (fold_dir / "predictions.parquet").exists()
        assert (fold_dir / "selected_feature_set.json").exists()
        assert (fold_dir / "selected_feature_sets.json").exists()
        assert (fold_dir / "agent_candidate_proposals.jsonl").exists()
        assert (fold_dir / "extracted_feature_diagnostics_by_fold.jsonl").exists()
        assert (fold_dir / "candidate_signal_review.jsonl").exists()
        assert (fold_dir / "parsimony_review_by_fold.jsonl").exists()
        assert (fold_dir / "outer_cv_metrics.csv").exists()
        assert (fold_dir / "checkpoint_summary.json").exists()

        fold_predictions = pd.read_parquet(fold_dir / "predictions.parquet")
        assert set(fold_predictions["outer_fold"]) == {fold}
        selected = json.loads((fold_dir / "selected_feature_set.json").read_text())
        assert selected["outer_fold"] == fold
        assert {item["name"] for item in selected["selected_features"]} == {
            "age",
            "pd_l1_expression",
        }
        fold_agent_rows = [
            json.loads(line)
            for line in (fold_dir / "agent_candidate_proposals.jsonl").read_text().splitlines()
        ]
        assert fold_agent_rows
        assert all(row.get("outer_fold") == fold for row in fold_agent_rows)
        fold_parsimony_rows = [
            json.loads(line)
            for line in (fold_dir / "parsimony_review_by_fold.jsonl").read_text().splitlines()
        ]
        assert fold_parsimony_rows
        assert all(row["outer_fold"] == fold for row in fold_parsimony_rows)
        fold_metrics = pd.read_csv(fold_dir / "outer_cv_metrics.csv")
        assert set(fold_metrics["outer_fold"]) == {fold}
    split_rows = [
        json.loads(line)
        for line in (artifact_dir / "split_provenance.jsonl").read_text().splitlines()
    ]
    assert split_rows
    assert all(row["honest_outer_holdout"] for row in split_rows)


def test_multi_model_agentic_forest_strict_honest_split_requires_holdout(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": ["age 55", "age 78", "age 57", "age 76"],
            "treatment_indicator": [1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_linear_test_bow_views(),
                candidate_consistency_enabled=False,
                require_honest_outer_split=True,
                **_disable_required_evidence_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )

    with pytest.raises(ValueError, match="require_honest_outer_split"):
        run_multi_model_agentic_forest(
            dataset,
            config,
            tmp_path / "predictions.parquet",
            proposal_agent=EmptyProposalAgent(),
            extraction_provider=FakeExtractionProvider(),
            evaluator=FakeEvaluator(),
        )


def test_multi_model_agentic_forest_builtin_extraction_rejects_truncation(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 baseline marker long note",
                "age 78 baseline marker long note",
                "age 57 baseline marker long note",
                "age 76 baseline marker long note",
            ],
            "treatment_indicator": [1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                clinical_text_examples_per_prompt=0,
                min_feature_coverage=0.0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_linear_test_bow_views(),
                candidate_consistency_enabled=False,
                extracted_feature_review_enabled=False,
                **_disable_required_evidence_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=[],
            extraction_max_text_length=10,
        ),
    )

    with pytest.raises(ValueError, match="complete-document reading"):
        run_multi_model_agentic_forest(
            dataset,
            config,
            tmp_path / "predictions.parquet",
            proposal_agent=FakeProposalAgent(),
            evaluator=FakeEvaluator(),
        )


def test_multi_model_agentic_forest_adds_embedding_contrast_context(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 baseline note brain metastases pd-l1 high",
                "age 78 baseline note liver lesion pd-l1 low",
                "age 57 baseline note brain metastases pd-l1 high",
                "age 76 baseline note liver lesion pd-l1 low",
                "age 61 baseline note brain metastases cachexia",
                "age 81 baseline note liver lesion stable",
                "age 54 baseline note brain metastases cachexia",
                "age 70 baseline note liver lesion stable",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        seed=_TEST_PHYSICAL_GLOBAL_SEED,
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                max_iterations=1,
                max_additions_per_iter=4,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_linear_test_bow_views(),
                top_n_features=5,
                candidate_consistency_enabled=False,
                fold_parallelism="1",
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=True,
                    model_name="fake-keyword",
                    chunk_size_words=8,
                    chunk_overlap_words=0,
                    max_chunks=2,
                    min_probe_auc=0.0,
                    top_k_chunks_per_tail=2,
                    max_chunks_per_patient=1,
                    include_cluster_contrast_vectors=False,
                    concept_phrases=["brain metastases", "liver lesion"],
                ),
                **_disable_htr_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = FakeProposalAgent()
    output_path = tmp_path / "predictions.parquet"

    run_multi_model_agentic_forest(
        dataset,
        config,
        output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
        embedding_provider=KeywordEmbeddingProvider(),
    )

    first_context = _role_agent_context(agent, "confounder")
    assert "embedding_contrast_evidence" not in first_context
    assert first_context["text_blurbs"]
    assert any("brain" in blurb for blurb in first_context["text_blurbs"])
    assert not any(_digest_source_label_in_blurb(blurb) for blurb in first_context["text_blurbs"])

    artifact_dir = output_path.parent / "multi_model_agentic_forest"
    evidence_rows = [
        json.loads(line)
        for line in (artifact_dir / "embedding_contrast_evidence_by_fold.jsonl")
        .read_text()
        .splitlines()
    ]
    assert evidence_rows
    artifact_treatment = next(
        item
        for item in evidence_rows[0]["embedding_contrast_evidence"]["contrasts"]
        if item["name"] == "treatment"
    )
    assert all(row["text"] is None for row in artifact_treatment["positive_aligned_chunks"])
    assert all(
        row["text_redacted"] is True for row in artifact_treatment["positive_aligned_chunks"]
    )


def test_multi_model_agentic_forest_adds_htr_to_ensemble_and_agent_context(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 baseline note brain metastases pd-l1 high",
                "age 78 baseline note liver lesion pd-l1 low",
                "age 57 baseline note brain metastases pd-l1 high",
                "age 76 baseline note liver lesion pd-l1 low",
                "age 61 baseline note brain metastases cachexia",
                "age 81 baseline note liver lesion stable",
                "age 54 baseline note brain metastases cachexia",
                "age 70 baseline note liver lesion stable",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                max_iterations=1,
                max_additions_per_iter=4,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_linear_test_bow_views(),
                top_n_features=5,
                candidate_consistency_enabled=False,
                fold_parallelism="1",
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=False,
                    disable_reason="unit test focuses on HTR evidence",
                ),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = FakeProposalAgent()
    htr_provider = FakeHTREvidenceProvider()
    output_path = tmp_path / "predictions.parquet"

    run_multi_model_agentic_forest(
        dataset,
        config,
        output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
        htr_evidence_provider=htr_provider,
    )

    assert htr_provider.seen_effect_nuisance_predictions
    effect_nuisance = htr_provider.seen_effect_nuisance_predictions[0]
    assert set(effect_nuisance["target_source"]) == {"ensemble_mean_nuisance_with_htr"}
    confounder_context = _role_agent_context(agent, "confounder")
    modifier_context = _role_agent_context(agent, "effect_modifier")
    assert any("age" in blurb.lower() for blurb in confounder_context["text_blurbs"])
    assert any("age" in blurb.lower() for blurb in modifier_context["text_blurbs"])
    assert not any(_digest_source_label_in_blurb(blurb) for blurb in confounder_context["text_blurbs"])
    assert not any(_digest_source_label_in_blurb(blurb) for blurb in modifier_context["text_blurbs"])

    artifact_dir = output_path.parent / "multi_model_agentic_forest"
    assert (artifact_dir / "htr_nuisance_oof_predictions.parquet").exists()
    assert (artifact_dir / "htr_effect_oof_predictions.parquet").exists()
    assert (artifact_dir / "htr_attention_evidence.parquet").exists()
    text_predictions = pd.read_parquet(artifact_dir / "text_model_oof_predictions.parquet")
    assert {"htr_nuisance", "htr_effect"}.issubset(set(text_predictions["view_name"].dropna()))
    assert int((text_predictions["view_name"] == "htr_nuisance").sum()) == len(dataset)
    assert int((text_predictions["view_name"] == "htr_effect").sum()) == len(dataset)
    ensemble_nuisance = pd.read_parquet(artifact_dir / "ensemble_nuisance_predictions.parquet")
    assert {"source", "ensemble_mean"}.issubset(set(ensemble_nuisance["nuisance_record_type"]))
    assert "htr_effect" not in set(ensemble_nuisance["source_name"].dropna())
    importance_rows = [
        json.loads(line)
        for line in (artifact_dir / "bow_view_feature_importance_by_fold.jsonl")
        .read_text()
        .splitlines()
    ]
    consensus_row = next(row for row in importance_rows if row["record_type"] == "consensus")
    artifact_htr_group = consensus_row["context"]["evidence_digest"]["confounders"]["htr_blurbs"][0]
    artifact_htr_row = artifact_htr_group["rows"][0]
    assert artifact_htr_row["text_redacted"] is True
    assert "evidence_snippet" not in artifact_htr_row
    assert "top_token_spans" not in artifact_htr_row


def test_multi_model_agentic_forest_runs_htr_only_when_bow_disabled(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 baseline note brain metastases pd-l1 high",
                "age 78 baseline note liver lesion pd-l1 low",
                "age 57 baseline note brain metastases pd-l1 high",
                "age 76 baseline note liver lesion pd-l1 low",
                "age 61 baseline note brain metastases cachexia",
                "age 81 baseline note liver lesion stable",
                "age 54 baseline note brain metastases cachexia",
                "age 70 baseline note liver lesion stable",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                max_iterations=1,
                max_additions_per_iter=4,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                feature_discovery_methods=["htr"],
                candidate_consistency_enabled=False,
                extracted_feature_review_enabled=False,
                fold_parallelism="1",
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = FakeProposalAgent()
    htr_provider = FakeHTREvidenceProvider()
    output_path = tmp_path / "predictions.parquet"

    run_multi_model_agentic_forest(
        dataset,
        config,
        output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
        htr_evidence_provider=htr_provider,
    )

    first_context = _role_agent_context(agent, "confounder")
    assert "feature_discovery_methods" not in first_context
    assert first_context["text_blurbs"]
    assert any("age" in blurb.lower() for blurb in first_context["text_blurbs"])
    assert not any(_digest_source_label_in_blurb(blurb) for blurb in first_context["text_blurbs"])
    assert "htr_attention_evidence" not in first_context
    assert "embedding_contrast_evidence" not in first_context
    assert htr_provider.seen_effect_nuisance_predictions
    effect_nuisance = htr_provider.seen_effect_nuisance_predictions[0]
    assert set(effect_nuisance["target_source"]) == {"htr_nuisance"}

    artifact_dir = output_path.parent / "multi_model_agentic_forest"
    assert not (artifact_dir / "bow_view_oof_predictions.parquet").exists()
    assert (artifact_dir / "htr_nuisance_oof_predictions.parquet").exists()
    assert (artifact_dir / "htr_effect_oof_predictions.parquet").exists()


def test_multi_model_agentic_forest_adds_ensemble_r_target_context(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 baseline note pd-l1 high brain metastases",
                "age 78 baseline note pd-l1 low liver lesion",
                "age 57 baseline note pd-l1 high brain metastases",
                "age 76 baseline note pd-l1 low liver lesion",
                "age 61 baseline note pd-l1 high cachexia",
                "age 81 baseline note pd-l1 low stable",
                "age 54 baseline note pd-l1 high cachexia",
                "age 70 baseline note pd-l1 low stable",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                max_iterations=1,
                max_additions_per_iter=4,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_two_linear_test_bow_views(),
                top_n_features=5,
                candidate_consistency_enabled=False,
                fold_parallelism="1",
                **_disable_required_evidence_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = FakeProposalAgent()
    output_path = tmp_path / "predictions.parquet"

    run_multi_model_agentic_forest(
        dataset,
        config,
        output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
    )

    modifier_context = _role_agent_context(agent, "effect_modifier")
    assert modifier_context["text_blurbs"]
    assert any("pd" in blurb.lower() or "brain" in blurb.lower() for blurb in modifier_context["text_blurbs"])
    assert not any(_digest_source_label_in_blurb(blurb) for blurb in modifier_context["text_blurbs"])

    artifact_dir = output_path.parent / "multi_model_agentic_forest"
    bow_oof = pd.read_parquet(artifact_dir / "bow_view_oof_predictions.parquet")
    assert "target_source" in bow_oof.columns
    assert "ensemble_mean_nuisance" in set(bow_oof["target_source"].dropna())
    importance_rows = [
        json.loads(line)
        for line in (artifact_dir / "bow_view_feature_importance_by_fold.jsonl")
        .read_text()
        .splitlines()
    ]
    assert any(row["record_type"] == "ensemble_r_view" for row in importance_rows)
    assert any(row["record_type"] == "ensemble_r_consensus" for row in importance_rows)


def test_multi_model_agent_context_compacts_large_evidence_payload():
    long_rows = [
        {
            "feature": f"feature phrase {idx}",
            "score": 0.123456789,
            "combined_score": 0.987654321,
            "confounder_overlap_score": 0.25,
            "treatment_score": 0.5,
            "outcome_score": 0.5,
            "pseudo_target_score": 0.1,
            "abs_pseudo_target_score": 0.1,
            "supporting_views": ["v1"],
        }
        for idx in range(100)
    ]
    long_text = "brain metastases " * 200
    context = {
        "prompt_version": "multi_model_agentic_forest_v1",
        "feature_importance": {
            "n_views": 1,
            "phrase_consensus": long_rows,
            "views": [
                {
                    "view_name": "v1",
                    "view_index": 0,
                    "view_config": {"name": "v1"},
                    "metrics": {},
                    "n_features": 1000,
                    "n_bow_features": 1000,
                    "n_prespecified_features": 0,
                    "phrase_features": long_rows,
                    "confounder_overlap": long_rows,
                    "treatment_positive": long_rows,
                    "treatment_negative": long_rows,
                    "outcome_positive": long_rows,
                    "outcome_negative": long_rows,
                    "pseudo_target_positive": long_rows,
                    "pseudo_target_negative": long_rows,
                }
            ],
        },
        "embedding_contrast_evidence": {
            "enabled": True,
            "contrasts": [
                {
                    "name": "treatment",
                    "positive_label": "treated",
                    "negative_label": "untreated",
                    "role_hint": "confounder",
                    "positive_aligned_chunks": [
                        {"row_id": idx, "chunk_index": idx, "score": 0.123456, "text": long_text}
                        for idx in range(10)
                    ],
                    "negative_aligned_chunks": [
                        {"row_id": idx, "chunk_index": idx, "score": -0.123456, "text": long_text}
                        for idx in range(10)
                    ],
                    "concept_probe_scores": [
                        {"concept": f"concept {idx}", "score": 0.111111} for idx in range(20)
                    ],
                }
            ],
        },
    }
    compact = _compact_multi_model_agent_context(context)
    assert len(compact["feature_importance"]["phrase_consensus"]) == 40
    compact_view = compact["feature_importance"]["views"][0]
    assert len(compact_view["treatment_positive"]) == 12
    contrast = compact["embedding_contrast_evidence"]["contrasts"][0]
    assert len(contrast["positive_aligned_chunks"]) == 3
    assert len(contrast["concept_probe_scores"]) == 8
    assert len(contrast["positive_aligned_chunks"][0]["text"]) <= 600


def test_multi_model_bow_fold_uses_prespecified_explicit_features():
    texts = ["same baseline note"] * 6
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
    view = BoWViewConfig(
        name="closed_fixture",
        bow_model="linear",
        ngram_range_min=1,
        ngram_range_max=1,
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        max_features=100,
        logistic_c=1.0,
        logistic_max_iter=1000,
        ridge_alpha=1.0,
    )
    explicit_values = [
        {"age": 30.0, "age_missing": False},
        {"age": 31.0, "age_missing": False},
        {"age": 29.0, "age_missing": False},
        {"age": 80.0, "age_missing": False},
        {"age": 81.0, "age_missing": False},
        {"age": 79.0, "age_missing": False},
    ]
    heldout, pred = _fit_binary_bow_fold(
        texts,
        labels,
        fit_pos=np.asarray([0, 1, 3, 4]),
        heldout_pos=np.asarray([2, 5]),
        vectorizer_params=multi_model_agentic_module._bow_vectorizer_params(view),
        model_params=multi_model_agentic_module._bow_model_params(view),
        explicit_feature_dicts=explicit_values,
        explicit_specs=[
            ExplicitFeatureSpec(
                name="age",
                type="continuous",
                roles=["confounder"],
            )
        ],
        random_state=7,
    )
    assert heldout.tolist() == [2, 5]
    assert pred[1] > pred[0]


def test_extracted_feature_review_gate_flags_underperforming_features():
    dataset = pd.DataFrame(
        {
            "clinical_text": ["signal positive", "signal negative"] * 6,
            "treatment_indicator": [1, 0] * 6,
            "outcome_indicator": [1, 0] * 6,
            "explicit_feat_noise_marker": ["present"] * 12,
            "explicit_feat_noise_marker_missing": [False] * 12,
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
    )
    nn_config = MultiModelAgenticForestConfig(
        nuisance_folds=2,
        effect_folds=2,
        bow_views=_linear_test_bow_views(),
        extracted_feature_review_auc_margin=0.01,
        extracted_feature_review_loss_relative_margin=0.01,
        extracted_feature_review_min_benchmark_auc=0.55,
        **_disable_required_evidence_test_kwargs(),
    )
    diagnostic = _evaluate_extracted_feature_set_diagnostic(
        train_df=dataset,
        specs=[
            ExplicitFeatureSpec(
                name="noise_marker",
                type="categorical",
                categories=["absent", "present"],
                roles=["confounder"],
            )
        ],
        config=config,
        nn_config=nn_config,
        bow_metrics={
            "views": [
                {
                    "metrics": {
                        "treatment_auroc": 1.0,
                        "outcome_auroc": 1.0,
                        "r_loss_mean": 0.01,
                    }
                }
            ]
        },
        embedding_evidence={},
        random_state=123,
    )
    gate = _extracted_feature_review_gate(
        diagnostic=diagnostic,
        nn_config=nn_config,
    )
    failed_metrics = {item["metric"] for item in gate["failed_criteria"]}
    assert gate["passed"] is False
    assert {"treatment_auroc", "outcome_auroc"}.issubset(failed_metrics)


def test_multi_model_parsimony_redundancy_review_records_required_summaries():
    df = pd.DataFrame(
        {
            "explicit_feat_age": [50.0, 60.0, 70.0, 80.0],
            "explicit_feat_age_copy": [51.0, 61.0, 71.0, 81.0],
            "explicit_feat_marker": ["low", "low", "high", "high"],
            "explicit_feat_marker_copy": ["low", "low", "high", "high"],
            "explicit_feat_age_missing": [False, False, False, False],
            "explicit_feat_age_copy_missing": [False, False, False, False],
            "explicit_feat_marker_missing": [False, False, False, False],
            "explicit_feat_marker_copy_missing": [False, False, False, False],
        }
    )
    specs = [
        ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ExplicitFeatureSpec(name="age_copy", type="continuous", roles=["confounder"]),
        ExplicitFeatureSpec(
            name="marker",
            type="categorical",
            categories=["low", "high"],
            roles=["effect_modifier"],
        ),
        ExplicitFeatureSpec(
            name="marker_copy",
            type="categorical",
            categories=["low", "high"],
            roles=["effect_modifier"],
        ),
    ]

    review = _feature_redundancy_review(
        train_df=df,
        specs=specs,
        corr_threshold=0.75,
    )

    assert review["continuous_correlations_abs_ge_threshold"]
    assert review["categorical_contingency"]
    assert len(review["missingness_overlap"]) == 6


def test_value_driven_parsimony_clusters_use_values_and_ignore_task_labels():
    df = pd.DataFrame(
        {
            "explicit_feat_function_a": np.arange(12, dtype=float),
            "explicit_feat_function_b": np.arange(12, dtype=float) * 2.0 + 1.0,
            "explicit_feat_unrelated": [0, 5, 2, 9, 1, 8, 4, 11, 3, 10, 6, 7],
            "explicit_feat_function_a_missing": [False] * 12,
            "explicit_feat_function_b_missing": [False] * 12,
            "explicit_feat_unrelated_missing": [False] * 12,
            "treatment_indicator": [0, 1] * 6,
            "outcome_indicator": [1, 0, 0, 1] * 3,
        }
    )
    specs = [
        ExplicitFeatureSpec(
            name="function_a",
            type="continuous",
            roles=["confounder"],
        ),
        ExplicitFeatureSpec(
            name="function_b",
            type="continuous",
            roles=["confounder"],
        ),
        ExplicitFeatureSpec(
            name="unrelated",
            type="continuous",
            roles=["effect_modifier"],
        ),
    ]
    config = MultiModelAgenticForestConfig(
        parsimony_cluster_semantic_weight=0.0,
        parsimony_cluster_neighbors=2,
        parsimony_cluster_empirical_min_similarity=0.5,
        parsimony_cluster_strong_empirical_threshold=0.8,
        parsimony_cluster_combined_threshold=0.5,
        **_disable_required_evidence_test_kwargs(),
    )
    semantic = np.eye(len(specs), dtype=float)

    first = _build_value_driven_feature_clusters(
        train_df=df,
        specs=specs,
        semantic_vectors=semantic,
        nn_config=config,
        random_state=17,
    )
    assert first["generation"]["uses_actual_extracted_values"] is True
    assert first["generation"]["uses_treatment_or_outcome_labels"] is False
    assert any(
        set(cluster["member_names"]) == {"function_a", "function_b"}
        for cluster in first["clusters"]
    )

    relabeled = df.copy()
    relabeled["treatment_indicator"] = relabeled["treatment_indicator"].sample(
        frac=1.0,
        random_state=3,
    ).to_numpy()
    relabeled["outcome_indicator"] = relabeled["outcome_indicator"].sample(
        frac=1.0,
        random_state=4,
    ).to_numpy()
    second = _build_value_driven_feature_clusters(
        train_df=relabeled,
        specs=specs,
        semantic_vectors=semantic,
        nn_config=config,
        random_state=17,
    )
    assert second["clusters"] == first["clusters"]

    changed_values = df.copy()
    changed_values["explicit_feat_function_b"] = [
        9,
        2,
        7,
        4,
        5,
        11,
        0,
        3,
        6,
        10,
        8,
        1,
    ]
    third = _build_value_driven_feature_clusters(
        train_df=changed_values,
        specs=specs,
        semantic_vectors=semantic,
        nn_config=config,
        random_state=17,
    )
    assert not any(
        set(cluster["member_names"]) == {"function_a", "function_b"}
        for cluster in third["clusters"]
    )


def test_value_cluster_neighbor_graph_stays_sparse_at_one_thousand_features():
    vectors = np.random.default_rng(7).normal(size=(1000, 12))
    pairs = _parsimony_mutual_neighbor_pairs(vectors, neighbors=20)

    assert pairs
    assert len(pairs) <= 1000 * 20 // 2


def test_strict_parsimony_replacement_rejects_any_task_degradation():
    base_metrics = {
        "status": "ok",
        "treatment_auroc": 0.8,
        "treatment_brier": 0.2,
        "treatment_log_loss": 0.5,
        "outcome_auroc": 0.75,
        "outcome_brier": 0.21,
        "outcome_log_loss": 0.55,
        "r_loss_mean": 0.18,
    }
    trial_metrics = dict(base_metrics)
    trial_metrics["outcome_log_loss"] += 1e-3
    allowed, reasons, _ = _strict_parsimony_replacement_decision(
        base_diagnostic={"metrics": base_metrics},
        trial_diagnostic={"metrics": trial_metrics},
        base_gate={"n_failed_criteria": 0},
        trial_gate={"n_failed_criteria": 0},
        epsilon=1e-6,
    )

    assert allowed is False
    assert "outcome_log_loss_degraded" in reasons


def test_parsimony_factor_prompt_allows_operational_implicit_concepts():
    context = {
        "prompt_version": "multi_model_agentic_parsimony_factor_v1",
        "cluster_id": "value_cluster_001",
        "replaceable_members": ["fatigue", "weight_loss", "ecog"],
        "protected_members": [],
        "required_role_union": ["confounder", "effect_modifier"],
        "max_factors": 2,
    }
    prompt = build_agent_prompt(context, AgenticFeatureSearchConfig())

    assert "actual extracted patient-level values" in prompt
    assert "implicit rather than literally named" in prompt
    assert "minimum evidence" in prompt
    assert "return null" in prompt
    assert "before the treatment decision" in prompt
    assert "response, prognosis, survival, or toxicity" in prompt


def test_cluster_factor_parsimony_replaces_group_when_all_tasks_are_preserved(
    tmp_path: Path,
    monkeypatch,
):
    values = np.asarray([0.0, 0.0, 1.0, 1.0] * 4)
    dataset = pd.DataFrame(
        {
            "clinical_text": ["pretreatment functional assessment"] * len(values),
            "treatment_indicator": [0, 1] * (len(values) // 2),
            "outcome_indicator": [1, 0, 0, 1] * (len(values) // 4),
            "explicit_feat_function_a": values,
            "explicit_feat_function_b": values,
            "explicit_feat_function_c": values,
            "explicit_feat_function_a_missing": [False] * len(values),
            "explicit_feat_function_b_missing": [False] * len(values),
            "explicit_feat_function_c_missing": [False] * len(values),
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            agentic_feature_search=AgenticFeatureSearchConfig(min_feature_coverage=0.1),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                parsimony_review_enabled=True,
                parsimony_cluster_semantic_weight=0.0,
                parsimony_parallelism="1",
                **_disable_required_evidence_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = ParsimonyFactorAgent()
    runner = MultiModelAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=agent,
        extraction_provider=ParsimonyFactorExtractionProvider(),
        evaluator=FakeEvaluator(),
    )
    specs = [
        ExplicitFeatureSpec(
            name=name,
            type="continuous",
            roles=["confounder", "effect_modifier"],
        )
        for name in ["function_a", "function_b", "function_c"]
    ]

    def preserved_diagnostic(*, specs, **kwargs):
        del kwargs
        role_dimensions = sum(len(spec.roles) for spec in specs)
        return {
            "metrics": {
                "status": "ok",
                "n_selected_features": len(specs),
                "n_w_features": sum("confounder" in spec.roles for spec in specs),
                "n_x_features": sum("effect_modifier" in spec.roles for spec in specs),
                "treatment_auroc": 0.8,
                "treatment_brier": 0.2,
                "treatment_log_loss": 0.5,
                "outcome_auroc": 0.75,
                "outcome_brier": 0.21,
                "outcome_log_loss": 0.55,
                "r_loss_mean": 0.18,
                "role_dimensions": role_dimensions,
            },
            "benchmark": {},
            "extraction_summary": [],
        }

    monkeypatch.setattr(
        multi_model_agentic_module,
        "_evaluate_extracted_feature_set_diagnostic",
        preserved_diagnostic,
    )
    result = runner._run_mandatory_parsimony_review(
        outer_fold=1,
        train_idx=np.arange(len(dataset)),
        selected_specs=specs,
        bow_result={"metrics": {}, "context": {}, "htr_evidence": {}},
        embedding_evidence={},
    )

    assert [spec.name for spec in result["selected_specs"]] == [
        "latent_functional_burden"
    ]
    assert result["summary"]["decision"] == "replace_clusters"
    assert result["summary"]["n_removed"] == 3
    assert result["summary"]["added_factors"] == ["latent_functional_burden"]
    assert runner.parsimony_cluster_rows
    assert runner.parsimony_factor_rows[0]["extraction_quality"]["passed"] is True
    assert any(row["allowed"] for row in runner.parsimony_evaluation_rows)
    fold_dir = tmp_path / "multi_model_agentic_forest" / "outer_fold_001"
    assert (fold_dir / "parsimony_clusters_by_fold.jsonl").exists()
    assert (fold_dir / "parsimony_factor_proposals_by_fold.jsonl").exists()
    assert (fold_dir / "parsimony_replacement_evaluations_by_fold.jsonl").exists()


def test_extracted_feature_review_context_includes_htr_snippets_and_defers_temporal_filter(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": ["PD-L1 TPS 70%", "PD-L1 TPS below 1%"],
            "treatment_indicator": [1, 0],
            "outcome_indicator": [1, 0],
        }
    )
    config = AppliedInferenceConfig(
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            agentic_feature_search=AgenticFeatureSearchConfig(),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                **_disable_required_evidence_test_kwargs(),
            ),
        ),
    )
    runner = MultiModelAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=EmptyProposalAgent(),
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
    )
    context = runner._build_extracted_feature_review_context(
        outer_fold=1,
        round_index=0,
        current_specs=[],
        diagnostic={"extraction_summary": [], "metrics": {}},
        gate={"failed_criteria": []},
        benchmark={},
        bow_context={"model_diagnostics": {}, "feature_importance": {}},
        embedding_evidence={},
        htr_evidence={
            "effect": {
                "metrics": {"r_loss_mean": 0.2},
                "attention": [
                    {
                        "row_id": 7,
                        "stage": "effect_modifier",
                        "chunk_text": "PD-L1 TPS 70% documented before the decision",
                        "attended_token_summary": "PD-L1 TPS 70%",
                        "attention": 0.91,
                    }
                ],
            }
        },
        required_names=set(),
    )

    htr_rows = context["htr_attention_evidence"]["effect"]["attention"]
    assert len(htr_rows) == 1
    assert "PD-L1 TPS 70%" in htr_rows[0]["evidence_snippet"]
    assert htr_rows[0]["attended_token_summary"] == "PD-L1 TPS 70%"

    prompt = build_agent_prompt(context, runner.search_config)
    assert "Inspect htr_attention_evidence snippets" in prompt
    assert "Temporal eligibility is enforced upstream" in prompt
    assert (
        "Do not use treatment choice, post-treatment response, toxicity after "
        "treatment, survival, or outcome-derived variables."
        not in prompt
    )


def test_multi_model_extracted_feature_review_revises_underperforming_specs(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
            ],
            "treatment_indicator": [1, 0] * 6,
            "outcome_indicator": [1, 0] * 6,
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                max_removals_per_iter=2,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_linear_test_bow_views(),
                top_n_features=5,
                candidate_consistency_enabled=False,
                extracted_feature_review_enabled=True,
                extracted_feature_review_max_rounds=1,
                extracted_feature_review_auc_margin=0.0,
                extracted_feature_review_loss_relative_margin=0.0,
                extracted_feature_review_min_benchmark_auc=0.55,
                parsimony_review_enabled=True,
                fold_parallelism="1",
                **_disable_required_evidence_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = ReviewRevisionAgent()
    extractor = ReviewExtractionProvider()
    evaluator = FakeEvaluator()
    output_path = tmp_path / "predictions.parquet"

    run_multi_model_agentic_forest(
        dataset,
        config,
        output_path,
        proposal_agent=agent,
        extraction_provider=extractor,
        evaluator=evaluator,
    )

    assert any(
        context.get("prompt_version") == "multi_model_agentic_extracted_feature_review_v1"
        for context in agent.contexts
    )
    assert ["noise_marker"] in extractor.calls
    assert any("signal_marker" in call for call in extractor.calls)
    final_names = [spec.name for spec in evaluator.seen_specs[-1]]
    assert "signal_marker" in final_names
    artifact_dir = output_path.parent / "multi_model_agentic_forest"
    diagnostics = [
        json.loads(line)
        for line in (artifact_dir / "extracted_feature_diagnostics_by_fold.jsonl")
        .read_text()
        .splitlines()
    ]
    assert diagnostics
    selected_sets = json.loads((artifact_dir / "selected_feature_sets.json").read_text())
    assert selected_sets[0]["extracted_feature_review"]["review_rounds"] >= 1
    assert selected_sets[0]["parsimony_review"]["mandatory"] is True
    assert selected_sets[0]["parsimony_review"]["decision"] in {"retain_all", "prune"}
    parsimony_rows = [
        json.loads(line)
        for line in (artifact_dir / "parsimony_review_by_fold.jsonl").read_text().splitlines()
    ]
    assert parsimony_rows
    assert parsimony_rows[0]["event"] == "mandatory_parsimony_review"
    assert "redundancy_review" in parsimony_rows[0]
    assert "ablations" in parsimony_rows[0]


def test_low_coverage_features_are_reviewed_as_broader_targets(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "signal positive rare signal phrase baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
            ],
            "treatment_indicator": [1, 0] * 6,
            "outcome_indicator": [1, 0] * 6,
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                max_removals_per_iter=2,
                min_feature_coverage=0.50,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_linear_test_bow_views(),
                top_n_features=5,
                candidate_consistency_enabled=False,
                extracted_feature_review_enabled=True,
                extracted_feature_review_max_rounds=1,
                extracted_feature_review_auc_margin=0.0,
                extracted_feature_review_loss_relative_margin=0.0,
                extracted_feature_review_min_benchmark_auc=0.55,
                fold_parallelism="1",
                **_disable_required_evidence_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = LowCoverageReviewAgent()
    extractor = LowCoverageExtractionProvider()
    evaluator = FakeEvaluator()
    output_path = tmp_path / "predictions.parquet"

    run_multi_model_agentic_forest(
        dataset,
        config,
        output_path,
        proposal_agent=agent,
        extraction_provider=extractor,
        evaluator=evaluator,
    )

    review_contexts = [
        context
        for context in agent.contexts
        if context.get("prompt_version") == "multi_model_agentic_extracted_feature_review_v1"
    ]
    assert review_contexts
    low_coverage = review_contexts[0]["low_coverage_features_needing_broader_targets"]
    assert low_coverage[0]["feature"]["name"] == "rare_signal_phrase"
    assert "rare_signal_phrase" in extractor.calls[0]
    assert any("signal_marker" in call for call in extractor.calls)
    final_names = [spec.name for spec in evaluator.seen_specs[-1]]
    assert "rare_signal_phrase" not in final_names
    assert "signal_marker" in final_names


def test_multi_model_prespecified_features_extract_before_bow_and_merge_roles(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 biomarker positive baseline note",
                "age 78 biomarker negative baseline note",
                "age 57 biomarker positive baseline note",
                "age 76 biomarker negative baseline note",
                "age 61 biomarker positive baseline note",
                "age 81 biomarker negative baseline note",
                "age 54 biomarker positive baseline note",
                "age 70 biomarker negative baseline note",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                max_iterations=1,
                max_additions_per_iter=4,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_linear_test_bow_views(),
                top_n_features=5,
                candidate_consistency_enabled=False,
                fold_parallelism="1",
                prespecified_confounders=[
                    ExplicitFeatureSpec(
                        name="age",
                        type="continuous",
                        roles=["confounder"],
                    ),
                    {
                        "name": "biomarker",
                        "type": "categorical",
                        "categories": ["negative", "positive"],
                    },
                ],
                prespecified_effect_modifiers=[
                    {
                        "name": "biomarker",
                        "type": "categorical",
                        "categories": ["negative", "positive"],
                    },
                ],
                **_disable_required_evidence_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = EmptyProposalAgent()
    extractor = RecordingExtractionProvider()
    evaluator = FakeEvaluator()

    run_multi_model_agentic_forest(
        dataset,
        config,
        tmp_path / "predictions.parquet",
        proposal_agent=agent,
        extraction_provider=extractor,
        evaluator=evaluator,
    )

    assert extractor.calls
    assert extractor.calls[0] == [
        ("age", ("confounder",)),
        ("biomarker", ("confounder", "effect_modifier")),
    ]
    first_context = _role_agent_context(agent, "confounder")
    assert first_context["prompt_version"] == _EVIDENCE_DIGEST_ROLE_PROMPT_VERSION
    assert "current_features" not in first_context
    assert first_context["text_blurbs"]
    assert all("feature_importance" not in context for context in agent.contexts)
    seen_roles = {spec.name: spec.roles for specs in evaluator.seen_specs for spec in specs}
    assert seen_roles == {
        "age": ["confounder"],
        "biomarker": ["confounder", "effect_modifier"],
    }


def test_multi_model_agentic_forest_parses_bow_views_and_embedding_option():
    cfg = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": (
                    "synthetic_data/example_synthetic_datasets/"
                    "one_confounder_one_effect_modifier_nsclc_with_structured/"
                    "dataset.parquet"
                ),
                "architecture": {
                    "model_type": "multi_model_agentic_forest",
                    "multi_model_agentic_forest": {
                        "bow_views": [
                            {
                                "name": "trees",
                                "bow_model": "extratrees",
                                "ngram_range_min": 1,
                                "ngram_range_max": 3,
                                "min_df": 1,
                            }
                        ],
                        "nuisance_folds": 2,
                        "effect_folds": 2,
                        "candidate_consistency_enabled": True,
                        "candidate_consistency_inner_folds": 4,
                        "candidate_consistency_min_folds": 2,
                        "candidate_consistency_min_fold_fraction": 0.5,
                        "candidate_consistency_parallelism": "2",
                        "outer_parallelism": "3",
                        "bow_parallel_backend": "processes",
                        "extracted_feature_review_enabled": False,
                        "extracted_feature_review_max_rounds": 2,
                        "extracted_feature_review_auc_margin": 0.03,
                        "extracted_feature_review_loss_relative_margin": 0.07,
                        "extracted_feature_review_min_benchmark_auc": 0.6,
                        "parsimony_review_auc_tolerance": 0.02,
                        "parsimony_review_loss_relative_tolerance": 0.04,
                        "parsimony_review_corr_threshold": 0.8,
                        "parsimony_review_max_single_feature_ablations": 7,
                        "parsimony_cluster_semantic_weight": 0.4,
                        "parsimony_cluster_neighbors": 14,
                        "parsimony_cluster_combined_threshold": 0.62,
                        "parsimony_cluster_empirical_min_similarity": 0.35,
                        "parsimony_cluster_strong_empirical_threshold": 0.85,
                        "parsimony_cluster_missingness_weight": 0.1,
                        "parsimony_cluster_min_size": 3,
                        "parsimony_cluster_max_size": 10,
                        "parsimony_cluster_sketch_dim": 24,
                        "parsimony_max_factors_per_cluster": 2,
                        "parsimony_factor_min_coverage": 0.2,
                        "parsimony_parallelism": "2",
                        "parsimony_metric_epsilon": 1e-7,
                        "require_honest_outer_split": True,
                        "fail_on_extraction_truncation": False,
                        "embedding_contrast": {
                            "enabled": True,
                            "model_name": "Qwen/Qwen3-Embedding-8B",
                            "max_seq_length": 768,
                            "chunk_size_words": 128,
                            "chunk_overlap_words": 32,
                            "max_chunks": 16,
                            "chunk_selection": "last",
                            "min_probe_auc": 0.55,
                            "include_cell_contrasts": False,
                            "include_confounder_vector_contrast": False,
                            "include_residualized_interaction_contrast": False,
                            "include_orthogonal_r_score_contrasts": False,
                            "include_cluster_contrast_vectors": True,
                            "cluster_contrast_n_clusters": 12,
                            "cluster_contrast_max_components": 4,
                            "cluster_contrast_min_cluster_size": 20,
                            "cluster_contrast_min_group_size": 6,
                            "cluster_contrast_min_cell_size": 3,
                            "cluster_contrast_top_loadings": 4,
                            "cluster_contrast_random_state": 17,
                            "external_corpus_cache_dirs": ["/tmp/pubmed_cache"],
                            "external_top_k_chunks_per_tail": 5,
                            "concept_phrases": ["brain metastases"],
                        },
                    },
                },
                "explicit_features": {"enabled": True, "features": []},
            }
        }
    )
    nn_cfg = cfg.applied_inference.architecture.multi_model_agentic_forest
    assert len(nn_cfg.bow_views) == 1
    assert nn_cfg.bow_views[0].name == "trees"
    assert nn_cfg.bow_views[0].bow_model == "extratrees"
    assert nn_cfg.bow_views[0].ngram_range_max == 3
    assert nn_cfg.candidate_consistency_enabled is True
    assert nn_cfg.candidate_consistency_inner_folds == 4
    assert nn_cfg.candidate_consistency_min_folds == 2
    assert nn_cfg.candidate_consistency_min_fold_fraction == 0.5
    assert nn_cfg.candidate_consistency_parallelism == "2"
    assert nn_cfg.outer_parallelism == "3"
    assert nn_cfg.bow_parallel_backend == "processes"
    assert nn_cfg.extracted_feature_review_enabled is False
    assert nn_cfg.extracted_feature_review_max_rounds == 2
    assert nn_cfg.extracted_feature_review_auc_margin == 0.03
    assert nn_cfg.extracted_feature_review_loss_relative_margin == 0.07
    assert nn_cfg.extracted_feature_review_min_benchmark_auc == 0.6
    assert nn_cfg.parsimony_review_enabled is False
    assert nn_cfg.parsimony_review_auc_tolerance == 0.02
    assert nn_cfg.parsimony_review_loss_relative_tolerance == 0.04
    assert nn_cfg.parsimony_review_corr_threshold == 0.8
    assert nn_cfg.parsimony_review_max_single_feature_ablations == 7
    assert nn_cfg.parsimony_cluster_semantic_weight == 0.4
    assert nn_cfg.parsimony_cluster_neighbors == 14
    assert nn_cfg.parsimony_cluster_combined_threshold == 0.62
    assert nn_cfg.parsimony_cluster_empirical_min_similarity == 0.35
    assert nn_cfg.parsimony_cluster_strong_empirical_threshold == 0.85
    assert nn_cfg.parsimony_cluster_missingness_weight == 0.1
    assert nn_cfg.parsimony_cluster_min_size == 3
    assert nn_cfg.parsimony_cluster_max_size == 10
    assert nn_cfg.parsimony_cluster_sketch_dim == 24
    assert nn_cfg.parsimony_max_factors_per_cluster == 2
    assert nn_cfg.parsimony_factor_min_coverage == 0.2
    assert nn_cfg.parsimony_parallelism == "2"
    assert nn_cfg.parsimony_metric_epsilon == 1e-7
    assert nn_cfg.require_honest_outer_split is True
    assert nn_cfg.fail_on_extraction_truncation is False
    assert nn_cfg.embedding_contrast.enabled is True
    assert nn_cfg.embedding_contrast.model_name == "Qwen/Qwen3-Embedding-8B"
    assert nn_cfg.embedding_contrast.max_seq_length == 768
    assert nn_cfg.embedding_contrast.chunk_size_words == 128
    assert nn_cfg.embedding_contrast.chunk_overlap_words == 32
    assert nn_cfg.embedding_contrast.max_chunks == 16
    assert nn_cfg.embedding_contrast.chunk_selection == "last"
    assert nn_cfg.embedding_contrast.min_probe_auc == 0.55
    assert nn_cfg.embedding_contrast.include_cell_contrasts is False
    assert nn_cfg.embedding_contrast.include_confounder_vector_contrast is False
    assert nn_cfg.embedding_contrast.include_residualized_interaction_contrast is False
    assert nn_cfg.embedding_contrast.include_orthogonal_r_score_contrasts is False
    assert nn_cfg.embedding_contrast.include_cluster_contrast_vectors is True
    assert nn_cfg.embedding_contrast.cluster_contrast_n_clusters == 12
    assert nn_cfg.embedding_contrast.cluster_contrast_max_components == 4
    assert nn_cfg.embedding_contrast.cluster_contrast_min_cluster_size == 20
    assert nn_cfg.embedding_contrast.cluster_contrast_min_group_size == 6
    assert nn_cfg.embedding_contrast.cluster_contrast_min_cell_size == 3
    assert nn_cfg.embedding_contrast.cluster_contrast_top_loadings == 4
    assert nn_cfg.embedding_contrast.cluster_contrast_random_state == 17
    assert nn_cfg.embedding_contrast.external_corpus_cache_dirs == ["/tmp/pubmed_cache"]
    assert nn_cfg.embedding_contrast.external_top_k_chunks_per_tail == 5
    assert nn_cfg.embedding_contrast.concept_phrases == ["brain metastases"]
    assert nn_cfg.feature_discovery_methods == ["bow", "htr", "embedding_contrast"]
    cfg.validate()


def test_multi_model_feature_discovery_methods_control_config_flags():
    cfg = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": (
                    "synthetic_data/example_synthetic_datasets/"
                    "one_confounder_one_effect_modifier_nsclc_with_structured/"
                    "dataset.parquet"
                ),
                "architecture": {
                    "model_type": "multi_model_agentic_forest",
                    "multi_model_agentic_forest": {
                        "feature_discovery_methods": ["bow", "embedding"],
                    },
                },
                "explicit_features": {"enabled": True, "features": []},
            }
        }
    )
    nn_cfg = cfg.applied_inference.architecture.multi_model_agentic_forest
    assert nn_cfg.feature_discovery_methods == ["bow", "embedding_contrast"]
    assert nn_cfg.bow_discovery_enabled is True
    assert nn_cfg.htr_evidence_enabled is False
    assert "feature_discovery_methods" in nn_cfg.htr_evidence_disable_reason
    assert nn_cfg.embedding_contrast.enabled is True
    cfg.validate()

    htr_only = MultiModelAgenticForestConfig(
        nuisance_folds=2,
        effect_folds=2,
        feature_discovery_methods=["htr"],
    )
    assert htr_only.feature_discovery_methods == ["htr"]
    assert htr_only.bow_discovery_enabled is False
    assert htr_only.htr_evidence_enabled is True
    assert htr_only.embedding_contrast.enabled is False
    assert "feature_discovery_methods" in htr_only.embedding_contrast.disable_reason


def test_multi_model_agentic_forest_default_bow_view_grid():
    cfg = MultiModelAgenticForestConfig(nuisance_folds=2, effect_folds=2)
    names = [view.name for view in cfg.bow_views]
    assert names == [
        "linear_unigram_c0p5",
        "linear_1_2",
        "linear_1_3",
        "linear_2_4_min_df3",
        "extratrees_1_3",
        "random_forest_1_2",
    ]
    assert {view.bow_model for view in cfg.bow_views} == {
        "linear",
        "extratrees",
        "random_forest",
    }


def test_old_non_neural_agentic_forest_model_type_rejected():
    cfg = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": (
                    "synthetic_data/example_synthetic_datasets/"
                    "one_confounder_one_effect_modifier_nsclc_with_structured/"
                    "dataset.parquet"
                ),
                "architecture": {"model_type": "non_neural_agentic_forest"},
            }
        }
    )
    try:
        cfg.validate()
    except ValueError as exc:
        assert "multi_model_agentic_forest" in str(exc)
    else:
        raise AssertionError("old non_neural_agentic_forest model_type was accepted")


def test_oracle_multi_model_script_builds_default_and_cli_bow_views():
    from oracle_experiment_scripts import run_oracle_multi_model_agentic_forest as script

    dataset_path = Path(
        "synthetic_data/example_synthetic_datasets/"
        "one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet"
    )
    extracted_dataset_path = dataset_path.parent / "dataset_with_extraction.parquet"
    assert script._resolve_oracle_parquet_file(str(dataset_path)) == dataset_path
    assert script._resolve_oracle_parquet_file(str(dataset_path.parent)) == extracted_dataset_path

    cfg = script.MultiModelAgenticOracleConfig(
        dataset_path=str(dataset_path),
        dataset_name="smoke",
    )
    applied = script._make_applied_config(cfg, dataset_path)
    mm_cfg = applied.architecture.multi_model_agentic_forest
    assert len(mm_cfg.bow_views) == 6
    assert mm_cfg.embedding_contrast.enabled is True
    assert mm_cfg.embedding_contrast.max_seq_length == 1024
    assert mm_cfg.require_honest_outer_split is True
    assert mm_cfg.fail_on_extraction_truncation is True
    assert applied.architecture.agentic_feature_search.agent_model_name == "auto"
    assert applied.explicit_features.vllm_model_name == "auto"

    cfg.bow_view_grid = "cli_single"
    cfg.bow_model = "extratrees"
    cfg.embedding_contrast_enabled = True
    cfg.embedding_concept_phrases = ["brain metastases"]
    cfg.embedding_external_cache_dirs = ["/tmp/pubmed_cache"]
    cfg.embedding_external_top_k_chunks_per_tail = 7
    cfg.extracted_feature_review_enabled = False
    cfg.extracted_feature_review_max_rounds = 1
    cfg.require_honest_outer_split = False
    cfg.fail_on_extraction_truncation = False
    applied = script._make_applied_config(cfg, dataset_path)
    mm_cfg = applied.architecture.multi_model_agentic_forest
    assert [view.name for view in mm_cfg.bow_views] == ["cli_view"]
    assert mm_cfg.bow_views[0].bow_model == "extratrees"
    assert mm_cfg.embedding_contrast.enabled is True
    assert mm_cfg.embedding_contrast.concept_phrases == ["brain metastases"]
    assert mm_cfg.embedding_contrast.external_corpus_cache_dirs == ["/tmp/pubmed_cache"]
    assert mm_cfg.embedding_contrast.external_top_k_chunks_per_tail == 7
    assert mm_cfg.extracted_feature_review_enabled is False
    assert mm_cfg.extracted_feature_review_max_rounds == 1
    assert mm_cfg.require_honest_outer_split is False
    assert mm_cfg.fail_on_extraction_truncation is False

    cfg.feature_discovery_methods = ["bow", "htr"]
    cfg.embedding_contrast_enabled = True
    applied = script._make_applied_config(cfg, dataset_path)
    mm_cfg = applied.architecture.multi_model_agentic_forest
    assert mm_cfg.feature_discovery_methods == ["bow", "htr"]
    assert mm_cfg.bow_discovery_enabled is True
    assert mm_cfg.htr_evidence_enabled is True
    assert mm_cfg.embedding_contrast.enabled is False
    assert "feature-discovery-methods" in mm_cfg.embedding_contrast.disable_reason


def test_multi_model_forest_parses_config():
    cfg = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": (
                    "synthetic_data/example_synthetic_datasets/"
                    "one_confounder_one_effect_modifier_nsclc_with_structured/"
                    "dataset.parquet"
                ),
                "architecture": {
                    "model_type": "multi_model_forest",
                    "multi_model_forest": {
                        "feature_discovery_methods": ["bow", "embedding_contrast"],
                        "bow_views": [
                            {
                                "name": "linear_test",
                                "ngram_range_min": 1,
                                "ngram_range_max": 2,
                                "min_df": 1,
                            },
                        ],
                        "nuisance_folds": 2,
                        "effect_folds": 2,
                        "cpus_total": 10,
                        "htr_jobs_per_gpu": 2,
                        "embedding_contrast": {"enabled": True},
                    },
                },
            }
        }
    )
    arch = cfg.applied_inference.architecture
    assert arch.model_type == "multi_model_forest"
    nn_cfg = arch.multi_model_forest
    assert isinstance(nn_cfg, MultiModelForestConfig)
    assert nn_cfg.feature_discovery_methods == ["bow", "embedding_contrast"]
    assert nn_cfg.cpus_total == 10
    assert nn_cfg.htr_jobs_per_gpu == 2
    assert nn_cfg.htr_evidence_enabled is False
    assert nn_cfg.embedding_contrast.enabled is True
    assert nn_cfg.matched_pair_uplift_enabled is True
    assert nn_cfg.matched_pair_propensity_caliper == pytest.approx(0.05)


def test_multi_model_forest_stage1_adds_bow_pair_uplift_features(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 brain metastases high nlr",
                "age 78 liver lesion low nlr",
                "age 56 brain metastases high nlr",
                "age 79 liver lesion low nlr",
                "age 57 brain metastases high nlr",
                "age 77 liver lesion low nlr",
                "age 58 brain metastases high nlr",
                "age 76 liver lesion low nlr",
                "age 59 brain metastases high nlr",
                "age 75 liver lesion low nlr",
                "age 60 brain metastases high nlr",
                "age 74 liver lesion low nlr",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        cv_folds=2,
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                feature_discovery_methods=["bow"],
                bow_views=_linear_test_bow_views(),
                nuisance_folds=2,
                effect_folds=2,
                fold_parallelism="1",
                matched_pair_htr_enabled=False,
                matched_pair_bow_max_iter=25,
                matched_pair_max_controls_per_candidate=2,
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=False,
                    disable_reason="unit test disables embedding evidence",
                ),
            ),
        ),
    )
    runner = MultiModelForestStage1Runner(
        dataset=dataset,
        config=config,
        output_path=tmp_path / "stage1.parquet",
        num_workers=1,
    )
    train_df = runner.dataset.iloc[:10].reset_index(drop=True)
    test_df = runner.dataset.iloc[10:].reset_index(drop=True)

    bundle = runner._build_feature_bundle(train_df=train_df, test_df=test_df, outer_fold=1)

    assert "bow__linear_test__matched_pair_uplift_delta_logit" in bundle.x_names
    assert "bow__linear_test__matched_pair_treated_outcome_prob" in bundle.x_names
    pair_rows = [
        row for row in bundle.feature_rows if row["source_family"] == "bow_pair_uplift"
    ]
    assert {row["objective"] for row in pair_rows} == {
        "matched_pair_uplift_delta_logit",
        "matched_pair_treated_outcome_probability",
    }
    assert any(
        row.get("source_family") == "bow_pair_uplift"
        and row.get("matched_pair_train_rows", 0) >= 0
        for row in bundle.inner_model_rows
    )
    assert "matched_pair_uplift" in bundle.handoff_evidence["importance"]
    pair_importance = bundle.handoff_evidence["importance"]["matched_pair_uplift"]
    assert "uplift_delta_logit_positive" in pair_importance["views"][0]
    compact = _compact_multi_model_agent_context(
        {
            "feature_importance": bundle.handoff_evidence["importance"],
            "htr_attention_evidence": {
                "pair_uplift": {
                    "metrics": {},
                    "attention": [
                        {
                            "row_id": 1,
                            "stage": "effect_modifier",
                            "pair_side": "treated_candidate",
                            "chunk_text": "brain metastases high nlr",
                            "attention": 0.9,
                            "pair_delta_logit": 0.3,
                        }
                    ],
                }
            },
        }
    )
    assert "matched_pair_uplift" in compact["feature_importance"]
    assert "pair_uplift" in compact["htr_attention_evidence"]


def test_stage1_bow_process_folds_do_not_pickle_existing_htr_provider(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "brain metastases pd-l1 high",
                "liver lesion pd-l1 low",
                "brain metastases pd-l1 high",
                "liver lesion pd-l1 low",
                "brain metastases prior radiation",
                "liver lesion stable disease",
                "brain metastases cachexia",
                "liver lesion low nlr",
                "brain metastases high nlr",
                "liver lesion adrenal mass",
                "brain metastases pd-l1 high response",
                "liver lesion pd-l1 low progression",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        cv_folds=2,
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                feature_discovery_methods=["bow"],
                bow_views=_linear_test_bow_views(),
                nuisance_folds=2,
                effect_folds=2,
                bow_fold_parallelism="2",
                bow_parallel_backend="processes",
                matched_pair_uplift_enabled=False,
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=False,
                    disable_reason="unit test disables embedding evidence",
                ),
                **_disable_htr_test_kwargs(),
            ),
        ),
    )
    runner = MultiModelForestStage1Runner(
        dataset=dataset,
        config=config,
        output_path=tmp_path / "stage1.parquet",
        num_workers=2,
    )
    runner._default_htr_provider = types.SimpleNamespace(
        _runner=types.SimpleNamespace(_thread_state=threading.local())
    )

    oof, test_pred, evidence_rows = runner._fit_bow_binary_train_test(
        runner.dataset["clinical_text"].iloc[:10].tolist(),
        runner.dataset["clinical_text"].iloc[10:].tolist(),
        runner.dataset["treatment_indicator"].iloc[:10].to_numpy(dtype=int),
        outer_fold=2,
        view=_linear_test_bow_views()[0],
        view_index=0,
        label_name="treatment",
    )

    assert np.isfinite(oof).all()
    assert np.isfinite(test_pred).all()
    assert len(evidence_rows) == 2


def test_multi_model_forest_parallel_plan_reserves_htr_slots():
    plan = resolve_multi_model_forest_parallel_plan(
        cpus_total=10,
        num_workers=1,
        gpu_ids=[0, 1],
        htr_jobs_per_gpu=2,
        htr_enabled=True,
        embedding_enabled=True,
    )
    assert plan.htr_slots == 4
    assert plan.cpu_loky_workers == 6
    assert plan.context_workers == 2
    assert plan.htr_inner_jobs_per_outer == 2
    assert plan.htr_device_slots == [0, 1, 0, 1]

    bow_only = resolve_multi_model_forest_parallel_plan(
        cpus_total=10,
        num_workers=1,
        gpu_ids=[0, 1],
        htr_jobs_per_gpu=2,
        htr_enabled=False,
        embedding_enabled=True,
    )
    assert bow_only.htr_slots == 0
    assert bow_only.cpu_loky_workers == 10
    assert bow_only.context_workers == 10
    assert bow_only.htr_inner_jobs_per_outer == 1


def test_multi_model_forest_primary_config_uses_nested_parallelism():
    config = AppliedInferenceConfig(
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                feature_discovery_methods=["bow", "htr"],
                bow_views=_linear_test_bow_views(),
            ),
        )
    )
    plan = resolve_multi_model_forest_parallel_plan(
        cpus_total=20,
        num_workers=1,
        gpu_ids=[0, 1],
        htr_jobs_per_gpu=3,
        htr_enabled=True,
        embedding_enabled=False,
    )

    primary = _config_for_primary_runner(config, plan)
    mm_cfg = primary.architecture.multi_model_forest

    assert mm_cfg.outer_parallel_backend == "processes"
    assert mm_cfg.outer_parallelism == "2"
    assert mm_cfg.fold_parallelism == "auto"
    assert mm_cfg.bow_fold_parallelism == "auto"
    assert mm_cfg.htr_fold_parallelism == "3"
    assert primary.architecture.agentic_attention_variable_forest.fold_parallelism == "3"


def test_htr_sentence_model_snapshot_resolved_once(monkeypatch, tmp_path):
    resolved_path = tmp_path / "models--prajjwal1--bert-tiny" / "snapshots" / "abc"
    resolved_path.mkdir(parents=True)
    calls = []

    def fake_snapshot_download(model_name, local_files_only=False):
        calls.append((model_name, local_files_only))
        return str(resolved_path)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert resolve_htr_sentence_model_snapshot("prajjwal1/bert-tiny") == str(resolved_path)
    assert calls == [("prajjwal1/bert-tiny", False)]
    assert resolve_htr_sentence_model_snapshot(str(resolved_path)) == str(resolved_path)
    assert resolve_htr_sentence_model_snapshot("hash") is None


def test_multi_model_forest_rejects_missing_exact_inner_handoff_path(tmp_path):
    dataset = pd.DataFrame(
        {
            "clinical_text": [f"note {idx}" for idx in range(12)],
            "treatment_indicator": [idx % 2 for idx in range(12)],
            "outcome_indicator": [(idx + 1) % 2 for idx in range(12)],
        }
    )
    config = AppliedInferenceConfig(
        cv_folds=2,
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            agentic_feature_search=AgenticFeatureSearchConfig(),
            multi_model_forest=MultiModelForestConfig(
                feature_discovery_methods=["bow"],
                bow_views=_linear_test_bow_views(),
                candidate_consistency_inner_folds=3,
            ),
        ),
    )
    with pytest.raises(ValueError, match="requires both deterministic BoW"):
        MultiModelForestRunner(
            dataset=dataset,
            config=config,
            output_path=tmp_path / "primary_predictions.parquet",
            num_workers=2,
        )


def test_multi_model_agentic_proposal_bundle_cache_reuses_llm_result(tmp_path):
    class CountingAgent:
        def __init__(self):
            self.calls = 0

        def propose(self, context):
            self.calls += 1
            if context.get("prompt_version") == _CONCEPT_CLUSTER_LABEL_PROMPT_VERSION:
                return {
                    "concepts": [
                        {
                            "name": "cache_feature",
                            "label": "Cache feature",
                            "source_families": ["bow"],
                            "source_overlap": 1,
                            "supporting_phrases": ["cache feature"],
                            "cluster_ids": ["cluster_001"],
                        }
                    ]
                }
            return [
                {
                    "action": "add",
                    "name": "cache_feature",
                    "type": "continuous",
                    "roles": ["confounder"],
                    "description": "Cached feature.",
                }
            ]

    class RaisingAgent:
        def propose(self, _context):
            raise AssertionError("proposal agent should not be called when cache exists")

    dataset = pd.DataFrame(
        {
            "clinical_text": ["alpha", "beta", "gamma", "delta"],
            "treatment_indicator": [0, 1, 0, 1],
            "outcome_indicator": [0, 0, 1, 1],
        }
    )
    config = AppliedInferenceConfig(
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                feature_discovery_methods=["bow"],
                candidate_proposals_per_fold=3,
                **_disable_htr_test_kwargs(),
            ),
        ),
    )
    output_path = tmp_path / "predictions.parquet"
    agent = CountingAgent()
    runner = MultiModelAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        proposal_agent=agent,
        extraction_provider=object(),
        evaluator=object(),
    )
    first = runner._propose_candidate_bundle(
        outer_fold=1,
        scope="full_outer_train",
        bow_context={
            "outer_fold": 1,
            "feature_importance": {
                "phrase_consensus": [
                    {
                        "feature": "cache feature",
                        "supporting_view_count": 4,
                        "mean_abs_confounder_score": 0.2,
                    }
                ]
            },
        },
        n_rows=4,
    )
    assert agent.calls == 2
    assert first["valid_proposals"][0].name == "cache_feature"
    assert set(first["valid_proposals"][0].roles) == {"confounder", "effect_modifier"}
    assert "concept_inventory" not in first
    assert set(first["raw_proposals_by_role"]) == {"confounder", "effect_modifier"}

    resumed = MultiModelAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        proposal_agent=RaisingAgent(),
        extraction_provider=object(),
        evaluator=object(),
    )
    second = resumed._propose_candidate_bundle(
        outer_fold=1,
        scope="full_outer_train",
        bow_context={"outer_fold": 1},
        n_rows=4,
    )

    assert second["valid_proposals"][0].name == "cache_feature"
    assert set(second["valid_proposals"][0].roles) == {"confounder", "effect_modifier"}
    assert "resumed_from_cache" in second


def test_multi_model_agentic_outer_fold_checkpoint_loads_completed_fold(tmp_path):
    dataset = pd.DataFrame(
        {
            "clinical_text": ["alpha", "beta", "gamma", "delta"],
            "treatment_indicator": [0, 1, 0, 1],
            "outcome_indicator": [0, 0, 1, 1],
        }
    )
    config = AppliedInferenceConfig(
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                feature_discovery_methods=["bow"],
                **_disable_htr_test_kwargs(),
            ),
        ),
    )
    runner = MultiModelAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=object(),
        extraction_provider=object(),
        evaluator=object(),
    )
    fold_dir = runner.artifact_dir / "outer_fold_001"
    fold_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "_oci_row_id": [0, 1],
            "outer_fold": [1, 1],
            "tau_hat": [0.1, 0.2],
        }
    ).to_parquet(fold_dir / "predictions.parquet", index=False)
    (fold_dir / "checkpoint_summary.json").write_text(
        json.dumps({"outer_fold": 1, "n_predictions": 2})
    )
    (fold_dir / "selected_feature_sets.json").write_text(
        json.dumps(
            [
                {
                    "outer_fold": 1,
                    "selected_features": [{"name": "cache_feature"}],
                }
            ]
        )
    )
    (fold_dir / "agent_candidate_proposals.jsonl").write_text(
        json.dumps(
            {
                "outer_fold": 1,
                "consistency_enabled": True,
                "proposal_bundles": [
                    {
                        "scope": "full_outer_train",
                        "concept_inventory": {
                            "schema_version": _CONCEPT_INVENTORY_SCHEMA_VERSION,
                            "concepts": [],
                        },
                    }
                ],
                "selected_features": [],
            }
        )
        + "\n"
    )
    pd.DataFrame([{"outer_fold": 1, "ite_mean": 0.15}]).to_csv(
        fold_dir / "outer_cv_metrics.csv",
        index=False,
    )

    cached = runner._load_outer_fold_checkpoint(
        outer_fold=1,
        expected_prediction_rows=2,
    )

    assert cached is not None
    assert len(cached["predictions"]) == 2
    assert cached["agent_rows"][0]["outer_fold"] == 1
    assert cached["feature_set_rows"][0]["outer_fold"] == 1
    assert cached["outer_metric_rows"][0]["ite_mean"] == 0.15


def test_oracle_multi_model_forest_script_builds_config():
    from oracle_experiment_scripts import run_oracle_multi_model_forest as script

    dataset_path = Path(
        "synthetic_data/example_synthetic_datasets/"
        "one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet"
    )
    cfg = script.MultiModelForestOracleConfig(
        dataset_path=str(dataset_path),
        dataset_name="smoke",
        bow_view_grid="cli_single",
        bow_model="random_forest",
        feature_discovery_methods=["bow", "tfidf_topic_contrast"],
        cpus_total=10,
        gpu_ids=[0, 1],
        htr_gpu_ids=[0, 1],
        htr_jobs_per_gpu=2,
        tfidf_topic_score_test_bootstrap_repeats=321,
        tfidf_topic_score_test_bootstrap_top_topics=7,
        tfidf_topic_score_test_fdr_level=0.15,
        tfidf_topic_score_test_min_topics_per_bank=3,
        tfidf_topic_score_test_max_topics_per_bank=11,
    )
    applied = script._make_applied_config(cfg, dataset_path)
    assert applied.architecture.model_type == "multi_model_forest"
    nn_cfg = applied.architecture.multi_model_forest
    assert isinstance(nn_cfg, MultiModelForestConfig)
    assert nn_cfg.cpus_total == 10
    assert nn_cfg.htr_jobs_per_gpu == 2
    assert nn_cfg.feature_discovery_methods == ["bow", "tfidf_topic_contrast"]
    assert nn_cfg.htr_evidence_enabled is False
    assert nn_cfg.embedding_contrast.enabled is False
    assert [view.name for view in nn_cfg.bow_views] == ["cli_view"]
    assert nn_cfg.bow_views[0].bow_model == "random_forest"
    assert nn_cfg.tfidf_topic.score_test_enabled is True
    assert nn_cfg.tfidf_topic.score_test_bootstrap_repeats == 321
    assert nn_cfg.tfidf_topic.score_test_bootstrap_top_topics == 7
    assert nn_cfg.tfidf_topic.score_test_fdr_level == pytest.approx(0.15)
    assert nn_cfg.tfidf_topic.score_test_min_topics_per_bank == 3
    assert nn_cfg.tfidf_topic.score_test_max_topics_per_bank == 11
    assert applied.architecture.multi_model_agentic_forest is nn_cfg


def test_oracle_multi_model_forest_agent_platform_config():
    from oracle_experiment_scripts import run_oracle_multi_model_forest as script

    dataset_path = Path(
        "synthetic_data/example_synthetic_datasets/"
        "one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet"
    )
    cfg = script.MultiModelForestOracleConfig(
        dataset_path=str(dataset_path),
        dataset_name="smoke",
        agent_provider="google",
        agent_platform_project="proposal-project",
        agent_platform_location="global",
        extraction_provider="vertex_ai",
        extraction_agent_platform_project="extraction-project",
        extraction_agent_platform_location="global",
    )

    applied = script._make_applied_config(cfg, dataset_path)

    agent_cfg = applied.architecture.agentic_feature_search
    assert agent_cfg.agent_server_url == (
        "https://aiplatform.googleapis.com/v1/projects/"
        "proposal-project/locations/global/endpoints/openapi"
    )
    assert agent_cfg.agent_model_name == "google/gemma-4-26b-a4b-it-maas"
    assert agent_cfg.agent_api_key == "GOOGLE_ADC"
    assert applied.explicit_features.vllm_server_url == (
        "https://aiplatform.googleapis.com/v1/projects/"
        "extraction-project/locations/global/endpoints/openapi"
    )
    assert applied.explicit_features.vllm_model_name == "google/gemma-4-26b-a4b-it-maas"
    assert applied.explicit_features.vllm_api_key == "GOOGLE_ADC"

    cfg.agent_model_name = "gemma-4-26b-a4b-it-maas"
    cfg.extraction_model_name = "gemma-4-26b-a4b-it-maas"
    applied = script._make_applied_config(cfg, dataset_path)
    assert (
        applied.architecture.agentic_feature_search.agent_model_name
        == "google/gemma-4-26b-a4b-it-maas"
    )
    assert applied.explicit_features.vllm_model_name == "google/gemma-4-26b-a4b-it-maas"


def test_oracle_multi_model_forest_codex_cli_config():
    from oracle_experiment_scripts import run_oracle_multi_model_forest as script

    dataset_path = Path(
        "synthetic_data/example_synthetic_datasets/"
        "one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet"
    )
    cfg = script.MultiModelForestOracleConfig(
        dataset_path=str(dataset_path),
        dataset_name="smoke",
        agent_provider="codex",
        extraction_provider="codex_cli",
        codex_executable="/tmp/codex",
        codex_model_name="profile",
        codex_reasoning_effort="medium",
        codex_extra_args=["--profile", "local-codex"],
        codex_parallelism=2,
    )

    applied = script._make_applied_config(cfg, dataset_path)

    agent_cfg = applied.architecture.agentic_feature_search
    assert agent_cfg.agent_provider == "codex_cli"
    assert agent_cfg.codex_cli_executable == "/tmp/codex"
    assert agent_cfg.codex_cli_model_name == "profile"
    assert agent_cfg.codex_cli_reasoning_effort == "medium"
    assert agent_cfg.codex_cli_extra_args == ["--profile", "local-codex"]
    assert applied.explicit_features.extraction_provider == "codex_cli"
    assert applied.explicit_features.codex_cli_executable == "/tmp/codex"
    assert applied.explicit_features.codex_cli_extra_args == ["--profile", "local-codex"]
    assert applied.explicit_features.codex_cli_parallelism == 2


def test_oracle_multi_model_forest_splits_primary_and_agentic_hashes():
    from dataclasses import replace

    from oracle_experiment_scripts import run_oracle_multi_model_forest as script

    cfg = script.MultiModelForestOracleConfig(
        dataset_path="dataset.parquet",
        dataset_name="smoke",
        feature_discovery_methods=["bow"],
        cpus_total=10,
        gpu_ids=[0, 1],
        htr_jobs_per_gpu=2,
    )
    base_primary = cfg.primary_hash()
    base_agentic = cfg.agentic_hash()

    assert replace(cfg, agent_server_url="http://localhost:4321/v1").primary_hash() == base_primary
    assert replace(cfg, extraction_server_url="http://localhost:9876/v1").primary_hash() == base_primary
    assert (
        replace(cfg, agent_provider="agent_platform", agent_platform_project="p").primary_hash()
        == base_primary
    )
    assert replace(cfg, cpus_total=2).primary_hash() == base_primary
    assert replace(cfg, htr_jobs_per_gpu=1).primary_hash() == base_primary
    assert replace(cfg, agent_server_url="http://localhost:4321/v1").agentic_hash() != base_agentic
    assert (
        replace(cfg, agent_provider="agent_platform", agent_platform_project="p").agentic_hash()
        != base_agentic
    )
    google_auto = replace(cfg, agent_provider="agent_platform", agent_platform_project="p")
    google_bare = replace(
        google_auto,
        agent_model_name="gemma-4-26b-a4b-it-maas",
    )
    google_publisher = replace(
        google_auto,
        agent_model_name="google/gemma-4-26b-a4b-it-maas",
    )
    assert google_auto.agentic_hash() == google_bare.agentic_hash()
    assert google_auto.agentic_hash() == google_publisher.agentic_hash()
    assert replace(cfg, n_folds=3).primary_hash() != base_primary
    assert (
        replace(cfg, tfidf_topic_score_test_fdr_level=0.10).primary_hash()
        != base_primary
    )
    assert (
        replace(cfg, tfidf_topic_score_test_bootstrap_repeats=1000).primary_hash()
        != base_primary
    )


def test_embedding_contrast_prepare_uses_multi_gpu_precompute(tmp_path, monkeypatch):
    import oci.inference.embedding_contrast_discovery as module

    class FakeHiddenStates:
        flat = np.zeros((2, 3), dtype=np.float32)
        offsets = np.asarray([0, 1, 2], dtype=np.int64)

    class FakeCache:
        instances = []

        def __init__(self, **kwargs):
            self.cache_path = tmp_path / "fake_cache"
            self.hidden_states_array = FakeHiddenStates()
            self.multi_gpu_devices = None
            FakeCache.instances.append(self)

        def is_valid(self, expected_num_samples):
            assert expected_num_samples == 2
            return False

        def precompute_multi_gpu(self, texts, devices, batch_size):
            self.multi_gpu_devices = [str(device) for device in devices]
            self.batch_size = batch_size

        def precompute(self, texts, device=None, batch_size=128):
            raise AssertionError("single-device precompute should not be used")

        def open(self):
            return None

        def load_chunks(self, expected_num_samples):
            assert expected_num_samples == 2
            return [["alpha"], ["beta"]]

    monkeypatch.setattr(module, "ConceptEmbeddingCache", FakeCache)
    monkeypatch.setattr(module, "_release_sentence_transformer_model", lambda model_name: None)

    config = AppliedInferenceConfig(
        text_column="clinical_text",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                feature_discovery_methods=["embedding_contrast"],
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=True,
                    cache_dir=str(tmp_path),
                    batch_size=7,
                ),
                **_disable_htr_test_kwargs(),
            ),
        ),
    )
    generator = EmbeddingContrastEvidenceGenerator(
        config=config,
        output_dir=tmp_path,
        precompute_devices=["cuda:0", "cuda:1"],
    )
    generator.prepare(pd.DataFrame({"clinical_text": ["alpha", "beta"]}))

    assert FakeCache.instances[0].multi_gpu_devices == ["cuda:0", "cuda:1"]
    assert FakeCache.instances[0].batch_size == 7


def test_oracle_multi_model_forest_prefers_cached_parquet(tmp_path):
    from oracle_experiment_scripts import run_oracle_multi_model_forest as script

    dataset_dir = tmp_path / "oracle_dataset"
    dataset_dir.mkdir()
    base_parquet = dataset_dir / "dataset.parquet"
    extracted_parquet = dataset_dir / "dataset_with_extraction.parquet"
    base_parquet.write_text("placeholder")
    extracted_parquet.write_text("placeholder")

    cfg = script.MultiModelForestOracleConfig(
        dataset_path=str(dataset_dir),
        dataset_name="smoke",
        feature_discovery_methods=["embedding_contrast"],
    )
    cache_hash = script._embedding_cache_hash_for_config(cfg, base_parquet)
    cache_path = (
        dataset_dir / ".oci_cache" / "embedding_contrast" / f"cecnn_chunk_embeddings_{cache_hash}"
    )
    cache_path.mkdir(parents=True)
    (cache_path / "metadata.json").write_text(
        json.dumps(
            {
                "cache_hash": cache_hash,
                "storage_format": "variable_length_chunks",
            }
        )
    )
    for filename in ("chunk_embeddings.npy", "offsets.npy", "chunk_texts.jsonl"):
        (cache_path / filename).write_text("")

    resolved = script._resolve_oracle_parquet_file_for_cache(cfg)
    assert resolved == base_parquet
    assert script._normalize_embedding_cache_dir_arg(str(cache_path)) == str(cache_path.parent)


def test_multi_model_agentic_forest_parses_prespecified_feature_sources(tmp_path):
    features_json = tmp_path / "features.json"
    features_json.write_text(
        json.dumps(
            {
                "confounders": [
                    {
                        "name": "baseline_risk",
                        "type": "continuous",
                        "description": "Baseline clinical risk score.",
                    }
                ],
                "effect_modifiers": [
                    {
                        "name": "baseline_risk",
                        "type": "continuous",
                        "description": "Baseline clinical risk score.",
                    }
                ],
            }
        )
    )
    cfg = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": (
                    "synthetic_data/example_synthetic_datasets/"
                    "one_confounder_one_effect_modifier_nsclc_with_structured/"
                    "dataset.parquet"
                ),
                "architecture": {
                    "model_type": "multi_model_agentic_forest",
                    "multi_model_agentic_forest": {
                        "prespecified_confounders": [
                            {
                                "name": "age",
                                "type": "continuous",
                                "description": "Age in years.",
                            }
                        ],
                        "prespecified_effect_modifiers": [
                            {
                                "name": "pd_l1_expression",
                                "type": "categorical",
                                "categories": ["low", "high"],
                                "description": "PD-L1 expression.",
                            }
                        ],
                        "prespecified_features_json": str(features_json),
                    },
                },
                "explicit_features": {"enabled": True, "features": []},
            }
        }
    )
    nn_cfg = cfg.applied_inference.architecture.multi_model_agentic_forest
    assert nn_cfg.prespecified_confounders[0].roles == ["confounder"]
    assert nn_cfg.prespecified_effect_modifiers[0].roles == ["effect_modifier"]
    assert nn_cfg.prespecified_features_json == str(features_json)


def test_multi_model_candidate_consistency_fallback_prefers_stable_candidates():
    assert (
        _candidate_consistency_threshold(
            3,
            min_folds=2,
            min_fold_fraction=0.5,
        )
        == 2
    )
    age = AgenticFeatureProposal(
        action="add",
        name="patient_age",
        type="continuous",
        roles=["confounder"],
    )
    noise = AgenticFeatureProposal(
        action="add",
        name="rare_noise",
        type="categorical",
        categories=["present", "absent"],
        roles=["effect_modifier"],
    )
    selected = _fallback_consistency_proposals(
        [
            {
                "name": "patient_age",
                "passes_consistency_gate": True,
                "proposed_on_full_outer_train": True,
            },
            {
                "name": "rare_noise",
                "passes_consistency_gate": False,
                "proposed_on_full_outer_train": True,
            },
        ],
        {"patient_age": age, "rare_noise": noise},
    )
    assert selected == [age]


def test_precomputed_discovery_runner_uses_handoff(tmp_path):
    handoff_path = tmp_path / "agentic_handoff.jsonl"
    handoff_row = {
        "schema_version": "multi_model_agentic_discovery_handoff_v1",
        "fold_key": 1,
        "outer_fold": 1,
        "scope": "full_outer_train",
        "n_rows": 2,
        "metrics": {"n_bow_views": 1},
        "importance": {"phrase_consensus": [{"feature": "ecog"}]},
        "embedding_contrast_evidence": {},
        "htr_evidence": {},
        "context": {"prompt_version": "multi_model_agentic_forest_v1", "outer_fold": 1},
    }
    handoff_path.write_text(json.dumps(handoff_row) + "\n")
    runner = PrecomputedDiscoveryMultiModelAgenticForestRunner(
        dataset=pd.DataFrame(
            {
                "clinical_text": ["ecog 0", "ecog 2"],
                "treatment_indicator": [1, 0],
                "outcome_indicator": [1, 0],
            }
        ),
        config=AppliedInferenceConfig(
            text_column="clinical_text",
            treatment_column="treatment_indicator",
            outcome_column="outcome_indicator",
            architecture=ModelArchitectureConfig(
                model_type="multi_model_agentic_forest",
                agentic_feature_search=AgenticFeatureSearchConfig(),
                multi_model_agentic_forest=MultiModelAgenticForestConfig(
                    **_disable_required_evidence_test_kwargs(),
                ),
            ),
        ),
        output_path=tmp_path / "predictions.parquet",
        handoff_path=handoff_path,
    )

    result = runner._fit_bow_discovery(pd.DataFrame(), outer_fold=1)

    assert result["metrics"] == {"n_bow_views": 1}
    assert result["importance"]["phrase_consensus"][0]["feature"] == "ecog"
    assert result["context"]["outer_fold"] == 1
    assert result["predictions"].empty
    with pytest.raises(RuntimeError, match="Missing precomputed agentic discovery handoff"):
        runner._fit_bow_discovery(pd.DataFrame(), outer_fold=1001)


def test_multi_model_consistency_selection_uses_agent_choice(tmp_path):
    agent = SelectingConsistencyAgent(["brain_metastases_present"])
    runner = MultiModelAgenticForestRunner(
        dataset=pd.DataFrame(
            {
                "clinical_text": ["ecog 0", "ecog 2"],
                "treatment_indicator": [1, 0],
                "outcome_indicator": [1, 0],
            }
        ),
        config=AppliedInferenceConfig(
            text_column="clinical_text",
            treatment_column="treatment_indicator",
            outcome_column="outcome_indicator",
            architecture=ModelArchitectureConfig(
                model_type="multi_model_agentic_forest",
                agentic_feature_search=AgenticFeatureSearchConfig(),
                multi_model_agentic_forest=MultiModelAgenticForestConfig(
                    candidate_proposals_per_fold=5,
                    **_disable_required_evidence_test_kwargs(),
                ),
            ),
        ),
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
    )
    ecog = AgenticFeatureProposal(
        action="add",
        name="ecog_performance_status",
        type="categorical",
        categories=[
            "0",
            "1",
            "2",
            "3",
            "4",
            "ECOG 0",
            "ECOG 1",
            "ECOG 2",
            "ECOG 3",
        ],
        roles=["confounder"],
        description="Baseline ECOG performance status.",
    )
    brain_mets = AgenticFeatureProposal(
        action="add",
        name="brain_metastases_present",
        type="categorical",
        categories=["yes", "no"],
        roles=["effect_modifier"],
        description="Presence of baseline brain metastases.",
    )

    selected, artifact = runner._select_consistent_proposals(
        context={
            "prompt_version": "multi_model_agentic_consistency_v1",
            "candidate_summaries": [
                {
                    "name": "ecog_performance_status",
                    "passes_consistency_gate": True,
                },
                {
                    "name": "brain_metastases_present",
                    "passes_consistency_gate": False,
                    "proposed_on_full_outer_train": True,
                },
            ],
        },
        candidate_summaries=[
            {
                "name": "ecog_performance_status",
                "passes_consistency_gate": True,
                "inner_support_count": 3,
                "proposed_on_full_outer_train": True,
            },
            {
                "name": "brain_metastases_present",
                "passes_consistency_gate": False,
                "inner_support_count": 1,
                "proposed_on_full_outer_train": True,
            },
        ],
        canonical_proposals={
            "ecog_performance_status": ecog,
            "brain_metastases_present": brain_mets,
        },
    )

    assert selected == [brain_mets]
    assert artifact["selection_method"] == "agentic_consistency_selection"
    assert artifact["agent_selection_attempted"] is True
    assert artifact["agent_selection_used"] is True
    assert artifact["used_fallback"] is False
    assert agent.contexts


def test_multi_model_consistency_prompt_requires_exhaustive_gate_passing_keep_list():
    prompt = build_agent_prompt(
        {
            "prompt_version": "multi_model_agentic_consistency_v1",
            "max_selected_candidates": 17,
            "candidate_summaries": [
                {
                    "name": "patient_age",
                    "passes_consistency_gate": True,
                    "inner_support_count": 3,
                },
                {
                    "name": "disease_progression_status",
                    "passes_consistency_gate": False,
                    "inner_support_count": 1,
                    "proposed_on_full_outer_train": True,
                },
            ],
        },
        AgenticFeatureSearchConfig(),
    )

    assert "complete, exhaustive keep-list" in prompt
    assert "any candidate you omit will be discarded" in prompt
    assert "Treat passes_consistency_gate=true as a keep decision" in prompt
    assert "Do not spend selection capacity on below-threshold recovery candidates" in prompt
    assert "Return at most 17 add proposals" in prompt
    assert "Temporal eligibility has already been enforced upstream" in prompt
    assert "Do not independently reject a supplied candidate" in prompt
    assert (
        "Do not select variables that are post-treatment, outcome-derived, "
        "treatment choice itself, response, survival, or toxicity."
        not in prompt
    )


def test_multi_model_consistency_selection_falls_back_on_agent_error(tmp_path):
    runner = MultiModelAgenticForestRunner(
        dataset=pd.DataFrame(
            {
                "clinical_text": ["ecog 0", "ecog 2"],
                "treatment_indicator": [1, 0],
                "outcome_indicator": [1, 0],
            }
        ),
        config=AppliedInferenceConfig(
            text_column="clinical_text",
            treatment_column="treatment_indicator",
            outcome_column="outcome_indicator",
            architecture=ModelArchitectureConfig(
                model_type="multi_model_agentic_forest",
                agentic_feature_search=AgenticFeatureSearchConfig(),
                multi_model_agentic_forest=MultiModelAgenticForestConfig(
                    candidate_proposals_per_fold=5,
                    **_disable_required_evidence_test_kwargs(),
                ),
            ),
        ),
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=FailingConsistencyAgent(),
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
    )
    ecog = AgenticFeatureProposal(
        action="add",
        name="ecog_performance_status",
        type="categorical",
        categories=["0", "1", "2", "3", "4"],
        roles=["confounder"],
        description="Baseline ECOG performance status.",
    )

    selected, artifact = runner._select_consistent_proposals(
        context={"prompt_version": "multi_model_agentic_consistency_v1"},
        candidate_summaries=[
            {
                "name": "ecog_performance_status",
                "passes_consistency_gate": True,
                "inner_support_count": 3,
                "proposed_on_full_outer_train": True,
            }
        ],
        canonical_proposals={"ecog_performance_status": ecog},
    )

    assert selected == [ecog]
    assert artifact["selection_method"] == "deterministic_consistency_gate_after_agent_error"
    assert artifact["agent_selection_attempted"] is True
    assert artifact["agent_selection_used"] is False
    assert artifact["used_fallback"] is True
