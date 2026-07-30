"""Audited outer-fold runner for compact all-evidence feature fusion.

The runner deliberately owns only orchestration and deterministic estimation.
Remote feature fusion and extraction are injected dependencies.  In
particular, importing or constructing this runner cannot start an LLM.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import secrets
import statistics
import threading
import unicodedata
import weakref
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.model_selection import KFold, StratifiedKFold

from ..config import ExplicitFeatureSpec
from ..models.structured_interaction_head import StructuredInteractionHead
from . import minimal_staged_selection_postprocessor as _minimal_selection_module
from .all_evidence_fusion import (
    ALL_SOURCE_FAMILIES,
    AllEvidenceFusionRequest,
    CandidateContract,
    EXACT_INNER_RECURRENCE_VERSION,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
    HTR_NEURAL,
    LEGACY_ALL_SOURCE,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPICS,
    TFIDF_TOPIC_SOURCE,
    _bow_group_family,
    _embedding_family,
    ground_evidence_to_extraction_contract,
    prepare_all_evidence_fusion,
    source_text_temporal_policy_audit,
    validate_all_evidence_fusion_response,
)
from .all_evidence_discovery_interfaces import (
    render_interpret_evidence_chunk_messages,
)
from .all_evidence_discovery_interfaces import ACTIVE_STAGE1_CONCEPT_FAMILIES
from .adaptive_hierarchical_stage1_reconsideration import (
    AdaptiveCurrentFeature,
    AdaptiveDiagnostic,
    AdaptiveHierarchicalStage1Reconsideration,
    ExactSpentCatalogAuthentication,
)
from .authenticated_semantic_retrieval_compatibility import (
    current_spent_projection_compatibility_identity,
    restore_current_spent_projection_semantic_retrieval_view,
)
from .approved_hierarchical_discovery_agent import (
    AuthenticatedReferenceOnlyDirectNumericalContract,
    ApprovedHierarchicalDiscoveryAgent,
    MetadataJsonDiscoveryJobRunner,
)
from .approved_hierarchical_discovery_batch import (
    ApprovedHierarchicalDiscoveryBatchCoordinator,
    ApprovedHierarchicalDiscoveryBatchResult,
    FrozenReviewEvidencePolicyBinding,
    OrderedFoldDiscoveryAgent,
)
from .all_evidence_post_extraction_review import (
    AppliedReviewOperations,
    CausalReviewConfig,
    CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY,
    GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
    GateAcceptanceDecision,
    GateFeatureBankView,
    GateSourceSignalView,
    ObservableCausalRows,
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    POST_EXTRACTION_REVIEW_FRESH_NORMALIZATION_VERSION,
    POST_EXTRACTION_REVIEW_GROUNDING_REPAIR_VERSION,
    POST_EXTRACTION_REVIEW_PROMPT_VERSION,
    POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
    PostExtractionReviewResponseExhausted,
    apply_post_extraction_review_operations,
    build_causal_review_diagnostics,
    build_extraction_quality_diagnostics,
    build_redundancy_diagnostics,
    collect_post_extraction_diagnostic_ids,
    collect_post_extraction_diagnostic_targets,
    evaluate_untouched_gate_acceptance,
    extraction_semantics_sha256,
    validate_post_extraction_review_response,
)
from .fold_honest_r_stack import FitRowProvenance
from .final_context_fit_upstream_bank import (
    AuthenticatedFinalContextFitUpstreamBank,
    FinalContextFitUpstreamProducer,
)
from .authenticated_stable_nuisance_bridge import (
    derive_exact_nuisance_from_runtime_stable_stage1,
)
from .authenticated_coordinate_preserving_nuisance_bridge import (
    coordinate_preserving_nuisance_contract_sha256,
    derive_exact_nuisance_from_coordinate_preserved_stage1,
    precommit_runtime_producer_identity_sha256,
)
from .final_context_fit_causal_forest_adapter import (
    FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID,
    FinalCausalForestBackend,
    FixedCausalForestHeadBackend,
    SealedFinalForestExplicitBlock,
    StrictOuterHonestFinalCausalForestAdapter,
)
from .final_context_fit_r_stack_adapter import (
    EXACT_OUTCOME_PREDICTION,
    EXACT_PROPENSITY_PREDICTION,
    SealedExactNuisanceBankExtension,
)
from .context_fit_upstream_cache_overlay import (
    AuthenticatedFinalContextFitCacheOverlay,
    FINAL_CONTEXT_FIT_CACHE_OVERLAY_ID,
)
from .extraction_grounding_diagnostics import (
    EXTRACTION_GROUNDING_DIAGNOSTIC_VERSION,
    build_extraction_grounding_diagnostics,
)
from .post_extraction_scientific_policy import (
    PostExtractionScientificPolicy,
)
from .frozen_extraction_cache_overlay import (
    CacheOverlayReport,
    FrozenExtractionCacheOverlay,
    expected_extraction_columns,
    extraction_contract_sha256,
    ordered_dataset_text_fingerprint,
    sha256_file,
)
from .frozen_hierarchical_review_evidence import (
    FrozenHierarchicalReviewEvidence,
    freeze_hierarchical_review_evidence,
)
from .first_untouched_gate_direct_numerical_preparation import (
    FirstUntouchedGatePreparationBounds,
    prepare_first_untouched_gate_direct_numerical,
)
from .first_gate_materialization_contract import (
    FirstGateMaterializationIntent,
    prepare_first_gate_materialization_intent,
)
from .hierarchical_all_architecture_discovery import HierarchicalDiscoveryConfig
from .hierarchical_discovery_job_cache import (
    AuthenticatedHierarchicalDiscoveryJobCache,
    HierarchicalDiscoveryJobCacheConfig,
)
from .lossless_stage1_evidence_catalog import (
    DEFAULT_MAX_ATOMS_PER_ARCHITECTURE_CHUNK,
    DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK,
    DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK,
    ArchitectureChunkPlan,
    RoleNeutralEvidenceCatalog,
    build_complete_architecture_chunks,
    build_role_neutral_evidence_catalog,
    validate_role_neutral_catalog,
)
from .query_moment_evidence_adapter import (
    QueryMomentEvidenceAdapterConfig,
    derive_sparse_query_moment_evidence,
    load_query_moment_evidence_artifact,
)
from .staged_all_evidence_fusion_agent import (
    STAGED_FUSION_AUDIT_SCHEMA_VERSION,
    STAGED_SAME_NAME_MERGE_VERSION,
    STAGED_SELECTION_BACKFILL_VERSION,
    STAGED_SELECTION_UNION_POSTPROCESSING_VERSION,
)
from .stage1_architecture_explanations import production_stage1_family_explanations
from .minimal_staged_selection_postprocessor import (
    MINIMAL_STAGED_SELECTION_OUTPUT_SCHEMA,
    MINIMAL_STAGED_SELECTION_POSTPROCESSOR_VERSION,
)
from .safe_staged_proposal_union import (
    SAFE_STAGED_PROPOSAL_UNION_HASH_DOMAIN_VERSION,
    SAFE_STAGED_PROPOSAL_UNION_INPUT_SCHEMA_VERSION,
    SAFE_STAGED_PROPOSAL_UNION_OUTPUT_SCHEMA_VERSION,
    SAFE_STAGED_PROPOSAL_UNION_POLICY_VERSION,
    safe_staged_proposal_union_identity,
)
from .tfidf_orphan_evidence_adapter import (
    OrphanNgramEvidenceAdapterConfig,
    adapt_full_outer_orphan_ngram_evidence,
)
from .tfidf_topic_discovery import HANDOFF_SCHEMA_VERSION, row_set_fingerprint

RUNNER_SCHEMA_VERSION = "all_evidence_fusion_outer_runner_v21"
FOLD_MANIFEST_SCHEMA_VERSION = "all_evidence_fusion_frozen_fold_v20"
FUSION_RESPONSE_CACHE_SCHEMA_VERSION = "all_evidence_fusion_response_cache_v4"
POST_EXTRACTION_REVIEW_RESPONSE_CACHE_SCHEMA_VERSION = (
    "all_evidence_post_extraction_review_response_cache_v7"
)
POST_EXTRACTION_REVIEW_REQUEST_SCHEMA_VERSION = "all_evidence_post_extraction_review_request_v8"
POST_EXTRACTION_REVIEW_ROUND_SCHEMA_VERSION = "all_evidence_post_extraction_review_round_audit_v13"
POST_EXTRACTION_REVIEW_FAILURE_SCHEMA_VERSION = (
    "all_evidence_post_extraction_review_response_failure_v3"
)
ADAPTIVE_HIERARCHICAL_REVIEW_EXECUTION_SCHEMA_VERSION = (
    "authenticated_adaptive_hierarchical_review_execution_v2"
)
ADAPTIVE_DIAGNOSTIC_ADAPTER_AUDIT_SCHEMA_VERSION = "adaptive_diagnostic_adapter_audit_v2"
ADAPTIVE_DIAGNOSTIC_METRIC_COVERAGE_SCHEMA_VERSION = (
    "adaptive_diagnostic_metric_coverage_v1"
)
ADAPTIVE_DIAGNOSTIC_METRIC_PATH_ENCODING_VERSION = (
    "lossless_utf8_hex_reserved_segment_encoding_v1"
)
ADAPTIVE_PRE_GATE_CANDIDATE_FREEZE_SCHEMA_VERSION = (
    "adaptive_pre_gate_executable_and_provenance_freeze_v1"
)
POST_EXTRACTION_REVIEW_PARTITION_SCHEMA_VERSION = "all_evidence_post_extraction_review_partition_v2"
POST_EXTRACTION_REVIEW_UNRESOLVED_ONTOLOGY_SCHEMA_VERSION = (
    "all_evidence_post_extraction_review_unresolved_ontology_v1"
)
POST_EXTRACTION_REVIEW_CANDIDATE_WORKSPACE_POLICY_VERSION = (
    "sealed_spent_only_monotone_ontology_workspace_v2"
)
ADAPTIVE_REVIEW_CONTRACT_LOCAL_EXTRACTION_VERSION = "adaptive_review_contract_local_extraction_v1"
ORDERED_EXTRACTION_PROJECTION_SHA256_VERSION = "ordered_extraction_projection_sha256_v1"
POST_EXTRACTION_REVIEW_OPERATION_APPLY_POLICY_VERSION = "post_extraction_review_atomic_apply_v1"
POST_EXTRACTION_REVIEW_RESPONSE_VALIDATION_RETRY_POLICY_VERSION = (
    "sealed_same_workspace_response_validation_retry_v1"
)
SPENT_EVIDENCE_CONTEXT_EPOCH_POLICY_VERSION = (
    "spent_context_epoch_equals_consumed_review_gate_count_v1"
)
FROZEN_PREDICTION_SCHEMA_VERSION = "all_evidence_fusion_predictions_v5"
POSTHOC_EVALUATION_SCHEMA_VERSION = "all_evidence_fusion_posthoc_oracle_v1"
HIERARCHICAL_DISCOVERY_PREPARATION_INPUT_SCHEMA_VERSION = (
    "hierarchical_all_evidence_runner_preparation_input_v3"
)
HIERARCHICAL_DISCOVERY_PREPARATION_FOLD_SCHEMA_VERSION = (
    "hierarchical_all_evidence_runner_fold_preparation_v2"
)
HIERARCHICAL_DISCOVERY_BATCH_PACKET_SCHEMA_VERSION = (
    "hierarchical_all_evidence_runner_batch_packet_v1"
)
HIERARCHICAL_DISCOVERY_BATCH_RESULT_SCHEMA_VERSION = (
    "hierarchical_all_evidence_runner_batch_result_v1"
)
LEGACY_HANDOFF_SCHEMA_VERSION = "multi_model_agentic_discovery_handoff_v1"
FINAL_ITE_ESTIMATOR_AUDIT_SCHEMA_VERSION = "all_evidence_final_ite_estimator_v1"
FINAL_FOREST_POTENTIAL_OUTCOME_POLICY_VERSION = (
    "exact_nuisance_mean_feasible_potential_outcome_projection_v2"
)
DEFAULT_POST_EXTRACTION_REVIEW_ROUNDS = 2

_FORBIDDEN_NAME = re.compile(r"(?:^|_)(?:true|oracle|ground_truth)(?:_|$)", flags=re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ADAPTIVE_METRIC_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.:-]*\Z")
_ADAPTIVE_METRIC_SAFE_PATH_SEGMENT = re.compile(r"[a-z][a-z0-9_:-]*\Z")
_ADAPTIVE_METRIC_ESCAPED_SEGMENT_PREFIX = "escaped_"
_STAGED_AUDIT_CACHE_STATUSES = {
    "captured_and_request_bound",
    "unavailable_not_exposed_by_agent",
    "unavailable_in_legacy_request_bound_cache",
}
_REASONING_TRACE_PRESENCE_FIELDS = frozenset(
    {
        "response_trace_available",
        "completion_attempt_count",
        "reasoning_content_present_count",
        "reasoning_present_count",
        "any_reasoning_present",
    }
)
_REVIEW_FAILURE_MESSAGE_BY_CODE = {
    "unrelated_evidence_citation": (
        "A contract-changing operation cited evidence that failed exact grounding."
    ),
    "unknown_or_unavailable_citation": (
        "The response cited an ID outside the sealed sanitized request."
    ),
    "invalid_operation_target": ("An operation did not target only known current contracts."),
    "malformed_or_non_object_json": (
        "The reviewer exhausted repair without returning one valid JSON object."
    ),
    "remote_repair_exhausted": ("The remote reviewer exhausted bounded response repair."),
    "runner_boundary_response_invalid": (
        "The returned review object failed the closed response boundary."
    ),
}
_STAGED_AUDIT_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "outer_fold",
        "split_fingerprint",
        "original_request_sha256",
        "configured_final_cap",
        "effective_final_cap",
        "selection_backfill_version",
        "selection_union_postprocessing_version",
        "role_specific_proposal_policy",
        "stages",
        "proposal_union",
        "remote_selected_count",
        "final_selected_count",
        "backfilled_candidate_ids",
        "returned_proposal_count",
        "returned_response_sha256",
    }
)
_STAGED_PROPOSAL_STAGE_FIELDS = frozenset(
    {
        "stage",
        "request_sha256",
        "response_sha256",
        "evidence_block_count",
        "source_families",
        "validated_proposal_count",
        "evidence_id_map_to_original",
        "mapped_grounding_evidence_ids",
        "reasoning_trace_presence",
    }
)
_STAGED_FINAL_SELECTION_FIELDS = frozenset(
    {
        "stage",
        "request_sha256",
        "response_sha256",
        "evidence_block_count",
        "candidate_pool_count",
        "selected_count",
        "selected_candidate_ids",
        "remote_selected_count",
        "remote_selected_candidate_ids",
        "final_selected_count",
        "backfilled_candidate_ids",
        "mandatory_coverage_candidate_ids",
        "high_confidence_reserve_candidate_ids",
        "selection_postprocessor",
        "selection_backfill_version",
        "selection_union_postprocessing_version",
        "reasoning_trace_presence",
    }
)
_STAGED_PROPOSAL_UNION_FIELDS = frozenset(
    {
        "validated_proposal_count",
        "unique_contract_count",
        "exact_duplicate_count",
        "same_name_merge",
        "safe_union",
    }
)
_STAGED_SAME_NAME_MERGE_FIELDS = frozenset(
    {
        "version",
        "merged_contract_count",
        "final_candidate_pool_count",
    }
)
_STAGED_PROPOSAL_STAGE_NAMES = (
    "full_evidence_proposal",
    "confounder_role_proposal",
    "modifier_role_proposal",
)
_STAGED_ROLE_POLICY_FIELDS = frozenset(
    {
        "version",
        "eligible_source_families",
        "neural_query_moments_eligible",
        "matched_pair_htr_embedding_and_tfidf_evidence_eligible",
    }
)
_STAGED_ROLE_POLICY_VERSION = "role_specific_all_evidence_families_v1"
_STAGED_SELECTION_POSTPROCESSOR_FIELDS = frozenset(
    {
        "schema_version",
        "postprocessor_version",
        "postprocessor_code_sha256",
        "input_sha256",
        "output_sha256",
        "remote_selected_candidate_ids",
        "mandatory_coverage_candidate_ids",
        "high_confidence_reserve_candidate_ids",
        "omitted_candidate_ids",
        "candidate_pool_target_source_families",
        "candidate_pool_source_family_counts",
        "original_request_source_families",
        "original_request_families_without_candidate",
        "target_roles",
        "covered_source_families",
        "covered_roles",
        "candidate_pool_coverage_complete",
        "original_request_candidate_coverage_complete",
        "high_confidence_reserve_complete",
        "cap_limited",
        "final_count",
    }
)
_STAGED_SAFE_UNION_FIELDS = frozenset(
    {
        "identity",
        "input_sha256",
        "output_sha256",
        "input_candidate_count",
        "representative_candidate_ids",
        "exact_duplicate_candidate_ids",
        "compatible_role_merge_candidate_ids",
        "omitted_conflict_candidate_ids",
        "dispositions",
        "conflicts",
        "selection_candidate_to_representative_id",
        "incompatible_variant_support_or_roles_propagated",
        "semantic_fields_used_for_conflict_ranking",
        "patient_rows_or_observed_labels_used",
    }
)
_STAGED_SAFE_UNION_IDENTITY_FIELDS = frozenset(
    {
        "policy_version",
        "input_schema_version",
        "output_schema_version",
        "hash_domain_version",
        "implementation_module",
        "implementation_sha256",
    }
)
_STAGED_SAFE_UNION_DISPOSITION_FIELDS = frozenset(
    {"candidate_id", "disposition", "retained_candidate_id"}
)
_STAGED_SAFE_UNION_CONFLICT_FIELDS = frozenset(
    {
        "conflict_id",
        "retained_candidate_id",
        "omitted_candidate_ids",
        "differing_non_role_fields",
        "retained_strength",
        "omitted_strength",
    }
)
_STAGED_SAFE_UNION_STRENGTH_FIELDS = frozenset(
    {
        "validated_occurrence_count",
        "independent_source_family_breadth",
        "evidence_breadth",
    }
)
_STAGED_DISPOSITIONS = frozenset(
    {
        "representative",
        "exact_duplicate",
        "compatible_role_merge",
        "omitted_conflict",
    }
)
_STAGED_NON_ROLE_FIELDS = ("type", "categories", "description", "value_aliases")
_STAGED_ROLE_ORDER = ("confounder", "effect_modifier")
_EVIDENCE_AUDIT_ID = re.compile(r"^evidence_[0-9]{4}$")
_CANDIDATE_AUDIT_ID = re.compile(r"^candidate_[0-9]{4}$")
_CONFLICT_AUDIT_ID = re.compile(r"^conflict_[0-9a-f]{64}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _build_htr_stage2_prompt_preflight(
    *,
    provider: Any,
    max_atoms_per_chunk: int,
    max_bytes_per_chunk: int,
    max_semantic_member_ids_per_chunk: int,
    family_explanation: str,
    wire_budget: Any,
) -> dict[str, Any] | None:
    """Compile every cumulative HTR map request without invoking a runner."""

    catalogs_method = getattr(
        provider,
        "get_htr_stage2_preflight_catalogs",
        None,
    )
    if not callable(catalogs_method):
        return None
    rows = tuple(catalogs_method())
    identity = provider.identity()
    source_report = identity.get("htr_stage2_call_plan_preflight")
    if (
        not rows
        or not isinstance(source_report, Mapping)
        or source_report.get("schema_version")
        != "production_htr_stage2_call_plan_preflight_v2"
    ):
        raise ValueError(
            "authenticated HTR provider lacks its aggregate call plan"
        )
    prompt_sizes: list[int] = []
    content_sizes: list[int] = []
    per_scope: list[dict[str, Any]] = []
    delivered_member_ids: set[str] = set()
    for raw in rows:
        if (
            not isinstance(raw, tuple)
            or len(raw) != 3
            or isinstance(raw[0], bool)
            or not isinstance(raw[0], int)
            or isinstance(raw[1], bool)
            or not isinstance(raw[1], int)
            or not isinstance(raw[2], RoleNeutralEvidenceCatalog)
        ):
            raise ValueError("HTR prompt-preflight catalog row is malformed")
        outer_fold, context_epoch, catalog = raw
        plan = build_complete_architecture_chunks(
            catalog,
            max_atoms_per_chunk=max_atoms_per_chunk,
            max_bytes_per_chunk=max_bytes_per_chunk,
            max_semantic_member_ids_per_chunk=(
                max_semantic_member_ids_per_chunk
            ),
        )
        evidence_by_id = {
            atom.evidence_id: atom.as_discovery_item()
            for atom in catalog.atoms
        }
        scope_calls = 0
        scope_members = 0
        for chunk in plan.chunks:
            if chunk.source_family != HTR_NEURAL:
                continue
            evidence = tuple(
                evidence_by_id[str(row["evidence_id"])]
                for row in chunk.evidence
            )
            messages = render_interpret_evidence_chunk_messages(
                family_explanation=family_explanation,
                evidence=evidence,
                wire_budget=wire_budget,
            )
            prompt_sizes.append(
                len(_canonical_json(list(messages)).encode("utf-8"))
            )
            content_sizes.append(
                sum(
                    len(str(message["content"]).encode("utf-8"))
                    for message in messages
                )
            )
            member_ids = tuple(
                member_id
                for item in evidence
                for member_id in item.member_ids
            )
            if (
                not member_ids
                or len(member_ids) > max_semantic_member_ids_per_chunk
                or any(
                    member_id in delivered_member_ids
                    for member_id in member_ids
                )
            ):
                raise ValueError(
                    "HTR prompt preflight found invalid aggregate-member "
                    "delivery"
                )
            delivered_member_ids.update(member_ids)
            scope_members += len(member_ids)
            scope_calls += 1
        per_scope.append(
            {
                "outer_fold": outer_fold,
                "context_epoch": context_epoch,
                "catalog_sha256": catalog.catalog_sha256,
                "planned_htr_interpretation_call_count": scope_calls,
                "semantic_aggregate_member_count": scope_members,
            }
        )
    planned_calls = len(prompt_sizes)
    aggregate_count = len(delivered_member_ids)
    if (
        planned_calls
        != int(source_report["planned_htr_interpretation_call_count"])
        or aggregate_count
        != int(source_report["cross_fold_aggregate_count"])
        or planned_calls < 1
        or aggregate_count < 1
    ):
        raise ValueError(
            "compiled HTR prompt plan differs from the authenticated "
            "aggregate inventory"
        )
    baseline = int(source_report["one_atom_per_chunk_baseline_call_count"])
    body = {
        "schema_version": "production_htr_stage2_prompt_preflight_v1",
        "source_aggregate_call_plan_content_sha256": _content_sha256(
            dict(source_report)
        ),
        "scope_count": len(rows),
        "scopes": per_scope,
        "raw_token_occurrence_count": int(
            source_report["raw_token_occurrence_count"]
        ),
        "raw_chunk_interpretation_count": int(
            source_report["raw_chunk_interpretation_count"]
        ),
        "semantic_aggregate_count": aggregate_count,
        "planned_htr_interpretation_call_count": planned_calls,
        "one_atom_per_chunk_baseline_call_count": baseline,
        "call_reduction_fraction": 1.0 - (planned_calls / baseline),
        "total_prompt_canonical_message_bytes": sum(prompt_sizes),
        "maximum_prompt_size_bytes": max(prompt_sizes),
        "median_prompt_size_bytes": statistics.median(prompt_sizes),
        "maximum_prompt_content_utf8_bytes": max(content_sizes),
        "median_prompt_content_utf8_bytes": statistics.median(
            content_sizes
        ),
        "prompt_size_definition": (
            "canonical_utf8_bytes_of_exact_system_user_message_array_v1"
        ),
        "every_semantic_aggregate_delivered_exactly_once": True,
        "aggregate_member_dispositions_are_exact": True,
        "raw_token_arrays_copied_into_prompts": False,
        "top_k_sampling_or_truncation_applied": False,
        "endpoint_or_runner_calls_during_preflight": 0,
        "call_plan_on_order_of_hundreds_of_thousands": (
            planned_calls >= 100_000
        ),
        "stage2_endpoint_launch_allowed": planned_calls < 100_000,
    }
    return {**body, "content_sha256": _content_sha256(body)}


def _numerical_array_sha256(value: Any) -> str:
    """Hash one finite float64 array with explicit shape and byte order."""

    array = np.asarray(value, dtype="<f8", order="C")
    if not np.isfinite(array).all():
        raise ValueError("authenticated numerical array contains non-finite values")
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "schema_version": "ordered_float64_array_sha256_v1",
                "shape": list(array.shape),
                "dtype": "<f8",
            }
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _encode_adaptive_metric_path_segment(value: str) -> str:
    """Encode one mapping key injectively into an adaptive metric identifier segment."""

    if not isinstance(value, str):
        raise TypeError("adaptive diagnostic mapping keys must be strings")
    if (
        _ADAPTIVE_METRIC_SAFE_PATH_SEGMENT.fullmatch(value) is not None
        and not value.startswith(_ADAPTIVE_METRIC_ESCAPED_SEGMENT_PREFIX)
    ):
        return value
    return f"{_ADAPTIVE_METRIC_ESCAPED_SEGMENT_PREFIX}{value.encode('utf-8').hex()}"


def _validate_adaptive_diagnostic_metric_coverage_proof(
    proof: Mapping[str, Any],
) -> None:
    """Fail closed unless one diagnostic's metric proof is exact and self-consistent."""

    expected_fields = {
        "schema_version",
        "diagnostic_id",
        "path_encoding_version",
        "aggregate_metrics",
        "ordered_metric_keys",
        "eligible_metric_count",
        "emitted_metric_count",
        "eligible_metric_inventory_sha256",
        "emitted_metrics_sha256",
        "metric_names_unique",
        "every_eligible_metric_emitted_once",
        "coverage_proof_sha256",
    }
    if not isinstance(proof, Mapping) or set(proof) != expected_fields:
        raise ValueError("adaptive diagnostic metric coverage proof violates its closed schema")
    if proof.get("schema_version") != ADAPTIVE_DIAGNOSTIC_METRIC_COVERAGE_SCHEMA_VERSION:
        raise ValueError("adaptive diagnostic metric coverage proof schema is invalid")
    diagnostic_id = proof.get("diagnostic_id")
    if not isinstance(diagnostic_id, str) or not diagnostic_id:
        raise ValueError("adaptive diagnostic metric coverage proof has an invalid diagnostic ID")
    if (
        proof.get("path_encoding_version")
        != ADAPTIVE_DIAGNOSTIC_METRIC_PATH_ENCODING_VERSION
    ):
        raise ValueError("adaptive diagnostic metric path encoding version is invalid")

    metrics = proof.get("aggregate_metrics")
    ordered_keys = proof.get("ordered_metric_keys")
    if not isinstance(metrics, Mapping) or not isinstance(ordered_keys, list):
        raise ValueError("adaptive diagnostic metric coverage inventory is malformed")
    if not all(
        isinstance(key, str) and _ADAPTIVE_METRIC_IDENTIFIER.fullmatch(key) is not None
        for key in ordered_keys
    ):
        raise ValueError("adaptive diagnostic metric coverage keys are invalid")
    if len(set(ordered_keys)) != len(ordered_keys):
        raise ValueError("adaptive diagnostic metric coverage keys contain duplicates")
    if set(metrics) != set(ordered_keys) or tuple(metrics) != tuple(ordered_keys):
        raise ValueError("adaptive diagnostic metric coverage key order is inconsistent")
    for key, value in metrics.items():
        if value is not None and not isinstance(value, (bool, int, float)):
            raise ValueError(f"adaptive diagnostic metric {key!r} is not a scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"adaptive diagnostic metric {key!r} is not finite")

    for count_name in ("eligible_metric_count", "emitted_metric_count"):
        count = proof.get(count_name)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"adaptive diagnostic {count_name} is invalid")
        if count != len(ordered_keys):
            raise ValueError(f"adaptive diagnostic {count_name} is inconsistent")
    if proof.get("metric_names_unique") is not True:
        raise ValueError("adaptive diagnostic metric uniqueness proof is false")
    if proof.get("every_eligible_metric_emitted_once") is not True:
        raise ValueError("adaptive diagnostic metric coverage proof is incomplete")

    inventory = [
        {"metric_key": key, "value": metrics[key]}
        for key in ordered_keys
    ]
    if proof.get("eligible_metric_inventory_sha256") != _content_sha256(inventory):
        raise ValueError("adaptive diagnostic eligible metric inventory hash is invalid")
    if proof.get("emitted_metrics_sha256") != _content_sha256(dict(metrics)):
        raise ValueError("adaptive diagnostic emitted metric hash is invalid")
    identity = {
        key: proof[key] for key in expected_fields if key != "coverage_proof_sha256"
    }
    if proof.get("coverage_proof_sha256") != _content_sha256(identity):
        raise ValueError("adaptive diagnostic metric coverage proof hash is invalid")


def _validate_adaptive_diagnostic_adapter_audit(audit: Mapping[str, Any]) -> None:
    """Authenticate lossless metric coverage for every adapted diagnostic."""

    expected_fields = {
        "schema_version",
        "input_diagnostic_count",
        "adapted_diagnostic_count",
        "every_diagnostic_id_represented_once",
        "excluded_historical_target_count",
        "excluded_historical_targets_by_diagnostic",
        "unknown_current_diagnostic_targets_fail_closed",
        "model_context_contains_excluded_historical_names",
        "metric_coverage_proof_count",
        "total_eligible_metric_count",
        "total_emitted_metric_count",
        "metric_names_unique_within_each_diagnostic",
        "every_eligible_metric_emitted_once",
        "metric_coverage_proofs",
        "audit_sha256",
    }
    if not isinstance(audit, Mapping) or set(audit) != expected_fields:
        raise ValueError("adaptive diagnostic adapter audit violates its closed schema")
    if audit.get("schema_version") != ADAPTIVE_DIAGNOSTIC_ADAPTER_AUDIT_SCHEMA_VERSION:
        raise ValueError("adaptive diagnostic adapter audit schema is invalid")
    proofs = audit.get("metric_coverage_proofs")
    if not isinstance(proofs, list):
        raise ValueError("adaptive diagnostic metric coverage proofs are malformed")
    for proof in proofs:
        _validate_adaptive_diagnostic_metric_coverage_proof(proof)
    diagnostic_ids = [str(proof["diagnostic_id"]) for proof in proofs]
    if len(set(diagnostic_ids)) != len(diagnostic_ids):
        raise ValueError("adaptive diagnostic adapter audit repeats a diagnostic ID")

    for count_name in (
        "input_diagnostic_count",
        "adapted_diagnostic_count",
        "metric_coverage_proof_count",
        "total_eligible_metric_count",
        "total_emitted_metric_count",
        "excluded_historical_target_count",
    ):
        count = audit.get(count_name)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"adaptive diagnostic adapter {count_name} is invalid")
    if not (
        audit["input_diagnostic_count"]
        == audit["adapted_diagnostic_count"]
        == audit["metric_coverage_proof_count"]
        == len(proofs)
    ):
        raise ValueError("adaptive diagnostic adapter diagnostic coverage is inconsistent")
    eligible_total = sum(int(proof["eligible_metric_count"]) for proof in proofs)
    emitted_total = sum(int(proof["emitted_metric_count"]) for proof in proofs)
    if audit["total_eligible_metric_count"] != eligible_total:
        raise ValueError("adaptive diagnostic adapter eligible metric total is inconsistent")
    if audit["total_emitted_metric_count"] != emitted_total:
        raise ValueError("adaptive diagnostic adapter emitted metric total is inconsistent")
    for flag_name, expected in (
        ("every_diagnostic_id_represented_once", True),
        ("unknown_current_diagnostic_targets_fail_closed", True),
        ("model_context_contains_excluded_historical_names", False),
        ("metric_names_unique_within_each_diagnostic", True),
        ("every_eligible_metric_emitted_once", True),
    ):
        if audit.get(flag_name) is not expected:
            raise ValueError(f"adaptive diagnostic adapter flag {flag_name} is invalid")

    exclusions = audit.get("excluded_historical_targets_by_diagnostic")
    if not isinstance(exclusions, Mapping):
        raise ValueError("adaptive diagnostic historical-target exclusions are malformed")
    exclusion_count = 0
    for diagnostic_id, names in exclusions.items():
        if diagnostic_id not in set(diagnostic_ids) or not isinstance(names, list):
            raise ValueError("adaptive diagnostic historical-target exclusion is invalid")
        if (
            not all(isinstance(name, str) and name for name in names)
            or names != sorted(set(names))
        ):
            raise ValueError("adaptive diagnostic historical-target names are invalid")
        exclusion_count += len(names)
    if exclusion_count != audit["excluded_historical_target_count"]:
        raise ValueError("adaptive diagnostic historical-target count is inconsistent")

    identity = {key: audit[key] for key in expected_fields if key != "audit_sha256"}
    if audit.get("audit_sha256") != _content_sha256(identity):
        raise ValueError("adaptive diagnostic adapter audit hash is invalid")


def _read_path_snapshot(path: Path) -> tuple[bytes, str]:
    """Return one immutable byte snapshot and the digest of those exact bytes."""

    with path.open("rb") as handle:
        snapshot = handle.read()
    return snapshot, hashlib.sha256(snapshot).hexdigest()


def _read_jsonl_snapshot(
    snapshot: bytes,
    *,
    source_path: Path,
    schema_version: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = snapshot.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"handoff is not valid UTF-8: {source_path}") from exc
    for line_number, line in enumerate(io.StringIO(text), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number} of {source_path}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"handoff line {line_number} must be a JSON object")
        if row.get("schema_version") != schema_version:
            raise ValueError(
                f"unsupported handoff schema on line {line_number}: "
                f"{row.get('schema_version')!r}"
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"handoff is empty: {source_path}")
    return rows


def _write_immutable_json(path: Path, body: Mapping[str, Any], *, schema: str) -> str:
    """Create a hash-wrapped manifest once, or verify an identical prior copy."""

    detached = json.loads(_canonical_json(dict(body)))
    digest = _content_sha256(detached)
    payload = {
        "schema_version": schema,
        "content_sha256": digest,
        "body": detached,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"immutable manifest is invalid JSON: {path}") from exc
        if existing != payload:
            raise RuntimeError(f"refusing to mutate immutable manifest: {path}")
    return digest


def _assert_semantic_compatibility_identity_current(bound: Mapping[str, Any]) -> None:
    """Fail if the preparation helper changes after its input-manifest bind."""

    if current_spent_projection_compatibility_identity() != dict(bound):
        raise RuntimeError("semantic-retrieval compatibility identity changed during preparation")


def _write_immutable_plain_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Create one closed flat JSON artifact, or verify byte-identical replay."""

    serialized = (
        json.dumps(
            json.loads(_canonical_json(dict(payload))),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"refusing to mutate immutable JSON artifact: {path}")
    return sha256_file(path)


def _load_request_bound_fusion_response(
    path: Path,
    *,
    request_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None, str] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"fusion response cache is invalid JSON: {path}") from exc
    if payload.get("schema_version") != FUSION_RESPONSE_CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"fusion response cache has an unsupported schema: {path}")
    body = payload.get("body")
    if not isinstance(body, Mapping) or payload.get("content_sha256") != _content_sha256(body):
        raise RuntimeError(f"fusion response cache content hash is invalid: {path}")
    if body.get("request_sha256") != request_sha256:
        raise RuntimeError(
            f"fusion response cache belongs to a different immutable request: {path}"
        )
    response = body.get("response")
    if not isinstance(response, Mapping):
        raise RuntimeError(f"fusion response cache does not contain one JSON object: {path}")
    raw_stage_audit = body.get("staged_fusion_audit")
    if raw_stage_audit is not None and not isinstance(raw_stage_audit, Mapping):
        raise RuntimeError(f"fusion response cache has a malformed staged audit: {path}")
    stage_audit = None if raw_stage_audit is None else json.loads(_canonical_json(raw_stage_audit))
    stage_audit_status = str(
        body.get("staged_fusion_audit_status")
        or (
            "captured_and_request_bound"
            if stage_audit is not None
            else "unavailable_in_legacy_request_bound_cache"
        )
    )
    if stage_audit_status not in _STAGED_AUDIT_CACHE_STATUSES:
        raise RuntimeError(f"fusion response cache has an invalid staged audit status: {path}")
    if stage_audit_status == "captured_and_request_bound" and stage_audit is None:
        raise RuntimeError(f"fusion response cache claims a missing staged audit: {path}")
    if stage_audit_status != "captured_and_request_bound" and stage_audit is not None:
        raise RuntimeError(f"fusion response cache has an unclassified staged audit: {path}")
    return json.loads(_canonical_json(response)), stage_audit, stage_audit_status


def _load_request_bound_review_response(
    path: Path,
    *,
    request_sha256: str,
    max_contracts: int,
    review_round: int,
    review_attempt: int,
) -> tuple[Mapping[str, Any], str] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"post-extraction review cache is invalid JSON: {path}") from exc
    if payload.get("schema_version") != POST_EXTRACTION_REVIEW_RESPONSE_CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"post-extraction review cache has an unsupported schema: {path}")
    body = payload.get("body")
    if not isinstance(body, Mapping) or payload.get("content_sha256") != _content_sha256(body):
        raise RuntimeError(f"post-extraction review cache content hash is invalid: {path}")
    if set(body) != {
        "review_round",
        "review_attempt",
        "request_sha256",
        "response",
        "response_sha256",
        "applied_specs_sha256",
        "apply_policy_version",
        "max_contracts",
        "raw_response_persisted",
        "raw_reasoning_persisted",
    }:
        raise RuntimeError(f"post-extraction review cache violates its closed schema: {path}")
    if body.get("request_sha256") != request_sha256:
        raise RuntimeError(
            "post-extraction review cache belongs to a different immutable request: " f"{path}"
        )
    for field_name, expected in (
        ("review_round", int(review_round)),
        ("review_attempt", int(review_attempt)),
    ):
        actual = body.get(field_name)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise RuntimeError(f"post-extraction review cache attempt binding is invalid: {path}")
    if (
        body.get("raw_response_persisted") is not False
        or body.get("raw_reasoning_persisted") is not False
    ):
        raise RuntimeError(f"post-extraction review cache has unsafe trace flags: {path}")
    response = body.get("response")
    if not isinstance(response, Mapping):
        raise RuntimeError(f"post-extraction review cache lacks one JSON response: {path}")
    detached = json.loads(_canonical_json(response))
    if body.get("response_sha256") != _content_sha256(detached):
        raise RuntimeError(f"post-extraction review response hash is invalid: {path}")
    applied_sha = str(body.get("applied_specs_sha256") or "")
    if not _SHA256.fullmatch(applied_sha):
        raise RuntimeError(f"post-extraction review applied-spec hash is invalid: {path}")
    if body.get("apply_policy_version") != POST_EXTRACTION_REVIEW_OPERATION_APPLY_POLICY_VERSION:
        raise RuntimeError(f"post-extraction review apply policy is invalid: {path}")
    maximum = body.get("max_contracts")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum != int(max_contracts):
        raise RuntimeError(f"post-extraction review max_contracts is invalid: {path}")
    return detached, applied_sha


def _load_request_bound_adaptive_execution(
    path: Path,
    *,
    outer_fold: int,
    request_sha256: str,
    review_round: int,
    review_attempt: int,
    expected_diagnostic_adapter_audit: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], AppliedReviewOperations, Mapping[str, Any]] | None:
    """Replay one locally validated, hash-wrapped adaptive execution without model work."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"adaptive execution cache is invalid JSON: {path}") from exc
    if payload.get("schema_version") != ADAPTIVE_HIERARCHICAL_REVIEW_EXECUTION_SCHEMA_VERSION:
        raise RuntimeError(f"adaptive execution cache has an unsupported schema: {path}")
    body = payload.get("body")
    if not isinstance(body, Mapping) or payload.get("content_sha256") != _content_sha256(body):
        raise RuntimeError(f"adaptive execution cache content hash is invalid: {path}")
    expected_fields = {
        "outer_fold",
        "review_round",
        "review_attempt",
        "request_sha256",
        "diagnostic_adapter_audit",
        "authenticated_execution",
        "proposal_frozen_before_executable_bridge",
        "executable_revision_frozen_before_gate",
        "complete_catalog_sent_to_legacy_review_agent",
        "raw_response_persisted",
        "raw_reasoning_persisted",
        "gate_accessed",
    }
    if set(body) != expected_fields:
        raise RuntimeError(f"adaptive execution cache violates its closed schema: {path}")
    if body.get("request_sha256") != request_sha256:
        raise RuntimeError(f"adaptive execution cache belongs to another request: {path}")
    for field_name, expected in (
        ("outer_fold", int(outer_fold)),
        ("review_round", int(review_round)),
        ("review_attempt", int(review_attempt)),
    ):
        value = body.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise RuntimeError(f"adaptive execution cache attempt binding is invalid: {path}")
    for field_name, expected in (
        ("proposal_frozen_before_executable_bridge", True),
        ("executable_revision_frozen_before_gate", True),
        ("complete_catalog_sent_to_legacy_review_agent", False),
        ("raw_response_persisted", False),
        ("raw_reasoning_persisted", False),
        ("gate_accessed", False),
    ):
        if body.get(field_name) is not expected:
            raise RuntimeError(f"adaptive execution cache has unsafe flag {field_name}: {path}")
    diagnostic_adapter_audit = body.get("diagnostic_adapter_audit")
    try:
        _validate_adaptive_diagnostic_adapter_audit(diagnostic_adapter_audit)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"adaptive diagnostic adapter audit is invalid: {path}") from exc
    if expected_diagnostic_adapter_audit is not None:
        try:
            _validate_adaptive_diagnostic_adapter_audit(expected_diagnostic_adapter_audit)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("expected adaptive diagnostic adapter audit is invalid") from exc
        if _canonical_json(diagnostic_adapter_audit) != _canonical_json(
            expected_diagnostic_adapter_audit
        ):
            raise RuntimeError(
                f"adaptive diagnostic adapter audit differs from fresh extraction: {path}"
            )
    execution = body.get("authenticated_execution")
    if not isinstance(execution, Mapping):
        raise RuntimeError(f"adaptive execution cache lacks an execution object: {path}")
    frozen = execution.get("frozen_round")
    executable = execution.get("executable_revision")
    audit = execution.get("audit")
    if not all(isinstance(value, Mapping) for value in (frozen, executable, audit)):
        raise RuntimeError(f"adaptive execution cache has malformed nested freezes: {path}")
    frozen_body = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if frozen.get("freeze_sha256") != _content_sha256(frozen_body):
        raise RuntimeError(f"adaptive proposal freeze hash is invalid: {path}")
    proposal = frozen.get("proposal")
    if not isinstance(proposal, Mapping) or frozen.get("proposal_sha256") != _content_sha256(
        proposal
    ):
        raise RuntimeError(f"adaptive proposal hash is invalid: {path}")
    executable_body = {
        key: value for key, value in executable.items() if key != "executable_freeze_sha256"
    }
    if executable.get("executable_freeze_sha256") != _content_sha256(executable_body):
        raise RuntimeError(f"adaptive executable freeze hash is invalid: {path}")
    if executable.get("proposal_freeze_sha256") != frozen.get("freeze_sha256"):
        raise RuntimeError(f"adaptive executable derives from another proposal: {path}")
    applied = executable.get("applied")
    expected_applied_fields = {
        "specs",
        "reextract_specs",
        "removed_names",
        "added_names",
        "extraction_changed_names",
        "role_only_changed_names",
        "operation_audit",
    }
    if not isinstance(applied, Mapping) or set(applied) != expected_applied_fields:
        raise RuntimeError(f"adaptive execution cache has malformed application: {path}")
    if executable.get("applied_specs_sha256") != _content_sha256(applied["specs"]):
        raise RuntimeError(f"adaptive applied-spec hash is invalid: {path}")
    execution_identity = {
        "schema_version": execution.get("schema_version"),
        "freeze_sha256": frozen.get("freeze_sha256"),
        "executable_freeze_sha256": executable.get("executable_freeze_sha256"),
        "dossier_sha256s": execution.get("dossier_sha256s"),
        "lookback_sha256": (
            execution.get("lookback", {}).get("lookback_sha256")
            if isinstance(execution.get("lookback"), Mapping)
            else None
        ),
        "runner_identity_sha256": execution.get("runner_identity_sha256"),
        "cache_identity_sha256": execution.get("cache_identity_sha256"),
        "audit": audit,
    }
    if execution.get("execution_sha256") != _content_sha256(execution_identity):
        raise RuntimeError(f"adaptive execution identity hash is invalid: {path}")
    normalized_applied = AppliedReviewOperations(
        specs=tuple(applied["specs"]),
        reextract_specs=tuple(applied["reextract_specs"]),
        removed_names=tuple(applied["removed_names"]),
        added_names=tuple(applied["added_names"]),
        extraction_changed_names=tuple(applied["extraction_changed_names"]),
        role_only_changed_names=tuple(applied["role_only_changed_names"]),
        operation_audit=tuple(applied["operation_audit"]),
    )
    return (
        json.loads(_canonical_json(proposal)),
        normalized_applied,
        json.loads(_canonical_json(execution)),
    )


def _sanitized_review_failure_completion_attempts(
    agent: Any,
    *,
    returned_response: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Hash remote failure traces without persisting content or reasoning."""

    trace = getattr(agent, "last_response_trace", None)
    if not isinstance(trace, Mapping):
        raw = getattr(agent, "last_raw_response", None)
        if raw is not None:
            trace = {"raw_content": str(raw)}
        elif returned_response is not None:
            trace = {"raw_content": _canonical_json(returned_response)}
        else:
            trace = {"raw_content": ""}
    raw_attempts = trace.get("repair_attempts")
    attempts = (
        [row for row in raw_attempts if isinstance(row, Mapping)]
        if isinstance(raw_attempts, list)
        else [trace]
    )
    sanitized: list[dict[str, Any]] = []
    for index, row in enumerate(attempts, start=1):
        raw_content = str(row.get("raw_content") or "")
        reasoning_value = row.get("reasoning_content", row.get("reasoning"))
        reasoning_text = "" if reasoning_value is None else _canonical_json(reasoning_value)
        parsed_object: Mapping[str, Any] | None = None
        try:
            parsed = json.loads(raw_content)
            if isinstance(parsed, Mapping):
                parsed_object = parsed
        except json.JSONDecodeError:
            parsed_object = None
        operations = parsed_object.get("operations") if parsed_object is not None else None
        normalization = row.get("fresh_response_normalization")
        sanitized.append(
            {
                "attempt": index,
                "finish_reason": (
                    None if row.get("finish_reason") is None else str(row.get("finish_reason"))
                ),
                "content_chars": len(raw_content),
                "content_sha256": hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
                "parsed_json_object": parsed_object is not None,
                "parsed_response_sha256": (
                    _content_sha256(parsed_object) if parsed_object is not None else None
                ),
                "operation_count": len(operations) if isinstance(operations, list) else None,
                "normalization_audit_sha256": (
                    _content_sha256(normalization) if isinstance(normalization, Mapping) else None
                ),
                "reasoning_present": reasoning_value is not None,
                "reasoning_chars": len(reasoning_text),
                "reasoning_sha256": (
                    hashlib.sha256(reasoning_text.encode("utf-8")).hexdigest()
                    if reasoning_value is not None
                    else None
                ),
            }
        )
    return sanitized


def _sanitized_review_failure_targets(
    agent: Any,
    *,
    returned_response: Mapping[str, Any] | None,
    current_names: Sequence[str],
) -> list[str]:
    parsed: Mapping[str, Any] | None = returned_response
    if parsed is None:
        trace = getattr(agent, "last_response_trace", None)
        if isinstance(trace, Mapping):
            raw_attempts = trace.get("repair_attempts")
            if isinstance(raw_attempts, list) and raw_attempts:
                trace = raw_attempts[-1] if isinstance(raw_attempts[-1], Mapping) else trace
            raw_content = str(trace.get("raw_content") or "")
        else:
            raw_content = str(getattr(agent, "last_raw_response", "") or "")
        try:
            candidate = json.loads(raw_content)
            parsed = candidate if isinstance(candidate, Mapping) else None
        except json.JSONDecodeError:
            parsed = None
    allowed = set(map(str, current_names))
    retained: set[str] = set()
    operations = parsed.get("operations") if isinstance(parsed, Mapping) else None
    if isinstance(operations, list):
        for operation in operations:
            targets = operation.get("target_names") if isinstance(operation, Mapping) else None
            if isinstance(targets, list):
                retained.update(str(value) for value in targets if str(value) in allowed)
    return sorted(retained)


def _sanitized_review_failure_identity(exc: BaseException) -> dict[str, str]:
    """Classify an untrusted validator exception without persisting its text."""

    messages: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}:{current}")
        current = current.__cause__ or current.__context__
    raw_issue = " | ".join(messages)
    lowered = raw_issue.casefold()
    if "cites evidence unrelated" in lowered:
        code = "unrelated_evidence_citation"
    elif "unknown evidence" in lowered or "unknown diagnostic" in lowered:
        code = "unknown_or_unavailable_citation"
    elif "targets unknown" in lowered or "target_names" in lowered:
        code = "invalid_operation_target"
    elif "malformed json" in lowered or "invalid json" in lowered:
        code = "malformed_or_non_object_json"
    elif isinstance(exc, PostExtractionReviewResponseExhausted):
        code = "remote_repair_exhausted"
    else:
        code = "runner_boundary_response_invalid"
    return {
        "failure_code": code,
        "failure_message": _REVIEW_FAILURE_MESSAGE_BY_CODE[code],
        "failure_issue_sha256": hashlib.sha256(raw_issue.encode("utf-8")).hexdigest(),
    }


def _load_request_bound_review_failure(
    path: Path,
    *,
    request_sha256: str,
    review_round: int,
    review_attempt: int,
    expected_current_names: Sequence[str],
) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"post-extraction review failure audit is invalid JSON: {path}") from exc
    if payload.get("schema_version") != POST_EXTRACTION_REVIEW_FAILURE_SCHEMA_VERSION:
        raise RuntimeError(
            f"post-extraction review failure audit has an unsupported schema: {path}"
        )
    body = payload.get("body")
    if not isinstance(body, Mapping) or payload.get("content_sha256") != _content_sha256(body):
        raise RuntimeError(f"post-extraction review failure audit content hash is invalid: {path}")
    expected_fields = {
        "review_round",
        "review_attempt",
        "request_sha256",
        "failure_type",
        "failure_code",
        "failure_message",
        "failure_issue_sha256",
        "failed_contract_names",
        "completion_attempts",
        "raw_response_persisted",
        "raw_reasoning_persisted",
        "row_level_values_persisted",
        "gate_accessed",
        "gate_consumed",
        "outer_heldout_labels_used",
    }
    if set(body) != expected_fields:
        raise RuntimeError(f"post-extraction review failure audit violates its schema: {path}")
    if body.get("request_sha256") != request_sha256:
        raise RuntimeError(
            f"post-extraction review failure audit belongs to another request: {path}"
        )
    for field_name, expected in (
        ("review_round", int(review_round)),
        ("review_attempt", int(review_attempt)),
    ):
        value = body.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise RuntimeError(f"post-extraction review failure attempt binding is invalid: {path}")
    failure_type = body.get("failure_type")
    if failure_type not in {
        "remote_reviewer_exhausted",
        "runner_boundary_validation",
        "adaptive_hierarchy_or_executable_validation",
    }:
        raise RuntimeError(f"post-extraction review failure type is invalid: {path}")
    failure_code = body.get("failure_code")
    if failure_code not in _REVIEW_FAILURE_MESSAGE_BY_CODE:
        raise RuntimeError(f"post-extraction review failure code is invalid: {path}")
    failure_message = body.get("failure_message")
    if (
        not isinstance(failure_message, str)
        or not failure_message
        or len(failure_message) > 240
        or any(ord(character) < 32 for character in failure_message)
        or failure_message != _REVIEW_FAILURE_MESSAGE_BY_CODE[failure_code]
    ):
        raise RuntimeError(f"post-extraction review failure message is invalid: {path}")
    if not _SHA256.fullmatch(str(body.get("failure_issue_sha256") or "")):
        raise RuntimeError(f"post-extraction review failure issue hash is invalid: {path}")
    failed_names = body.get("failed_contract_names")
    if (
        not isinstance(failed_names, list)
        or failed_names != sorted(set(map(str, failed_names)))
        or any(
            re.fullmatch(r"[a-z][a-z0-9_]*", name) is None or _FORBIDDEN_NAME.search(name)
            for name in failed_names
        )
        or not set(failed_names).issubset(set(map(str, expected_current_names)))
    ):
        raise RuntimeError(f"post-extraction review failed-contract names are invalid: {path}")
    if any(
        body.get(field) is not False
        for field in (
            "raw_response_persisted",
            "raw_reasoning_persisted",
            "row_level_values_persisted",
            "gate_accessed",
            "gate_consumed",
            "outer_heldout_labels_used",
        )
    ):
        raise RuntimeError(f"post-extraction review failure audit has unsafe flags: {path}")
    completion_attempts = body.get("completion_attempts")
    if not isinstance(completion_attempts, list) or len(completion_attempts) > 16:
        raise RuntimeError(f"post-extraction review failure audit lacks attempt metadata: {path}")
    if failure_type == "adaptive_hierarchy_or_executable_validation":
        if completion_attempts:
            raise RuntimeError(
                "adaptive post-extraction review failure audit contains legacy "
                f"completion attempts: {path}"
            )
    elif not completion_attempts:
        raise RuntimeError(f"post-extraction review failure audit lacks attempt metadata: {path}")
    attempt_fields = {
        "attempt",
        "finish_reason",
        "content_chars",
        "content_sha256",
        "parsed_json_object",
        "parsed_response_sha256",
        "operation_count",
        "normalization_audit_sha256",
        "reasoning_present",
        "reasoning_chars",
        "reasoning_sha256",
    }
    for expected_attempt, row in enumerate(completion_attempts, start=1):
        if not isinstance(row, Mapping) or set(row) != attempt_fields:
            raise RuntimeError(f"post-extraction review failure attempt schema is invalid: {path}")
        attempt_number = row.get("attempt")
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number != expected_attempt
        ):
            raise RuntimeError(f"post-extraction review failure attempt index is invalid: {path}")
        finish_reason = row.get("finish_reason")
        if finish_reason is not None and (
            not isinstance(finish_reason, str)
            or len(finish_reason) > 64
            or any(ord(character) < 32 for character in finish_reason)
        ):
            raise RuntimeError(f"post-extraction review finish reason is invalid: {path}")
        for count_field in ("content_chars", "reasoning_chars"):
            count = row.get(count_field)
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 <= count <= 10_000_000
            ):
                raise RuntimeError(f"post-extraction review failure count is invalid: {path}")
        operation_count = row.get("operation_count")
        if operation_count is not None and (
            isinstance(operation_count, bool)
            or not isinstance(operation_count, int)
            or not 0 <= operation_count <= 32
        ):
            raise RuntimeError(f"post-extraction review operation count is invalid: {path}")
        for sha_field in (
            "content_sha256",
            "parsed_response_sha256",
            "normalization_audit_sha256",
            "reasoning_sha256",
        ):
            sha = row.get(sha_field)
            if sha is not None and not _SHA256.fullmatch(str(sha)):
                raise RuntimeError(f"post-extraction review failure hash is invalid: {path}")
        if not _SHA256.fullmatch(str(row.get("content_sha256") or "")):
            raise RuntimeError(f"post-extraction review content hash is missing: {path}")
        parsed_json = row.get("parsed_json_object")
        reasoning_present = row.get("reasoning_present")
        if not isinstance(parsed_json, bool) or not isinstance(reasoning_present, bool):
            raise RuntimeError(f"post-extraction review failure flags are invalid: {path}")
        if parsed_json != (row.get("parsed_response_sha256") is not None):
            raise RuntimeError(
                f"post-extraction review parsed-response hash is inconsistent: {path}"
            )
        if reasoning_present != (row.get("reasoning_sha256") is not None):
            raise RuntimeError(f"post-extraction review reasoning hash is inconsistent: {path}")
        if not reasoning_present and row.get("reasoning_chars") != 0:
            raise RuntimeError(f"post-extraction review reasoning count is inconsistent: {path}")
    return json.loads(_canonical_json(body))


def _validate_request_bound_staged_fusion_audit(
    value: Any,
    *,
    request_sha256: str,
    response_sha256: str,
    outer_fold: int,
    split_fingerprint: str,
) -> Mapping[str, Any] | None:
    """Detach and authenticate the staged wrapper's audit, when exposed."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError("fusion agent exposed a malformed staged audit")
    audit = json.loads(_canonical_json(value))
    _validate_closed_staged_fusion_audit(audit)
    expected = {
        "schema_version": STAGED_FUSION_AUDIT_SCHEMA_VERSION,
        "outer_fold": int(outer_fold),
        "split_fingerprint": str(split_fingerprint),
        "original_request_sha256": str(request_sha256),
        "returned_response_sha256": str(response_sha256),
    }
    mismatched = [
        key for key, expected_value in expected.items() if audit.get(key) != expected_value
    ]
    if mismatched:
        raise RuntimeError(
            "fusion agent staged audit is not bound to the current request/response: "
            f"{sorted(mismatched)}"
        )
    return audit


def _staged_audit_schema_error(path: str, detail: str) -> None:
    raise RuntimeError(
        "fusion agent staged audit violates the closed "
        f"{STAGED_FUSION_AUDIT_SCHEMA_VERSION} schema at {path}: {detail}"
    )


def _closed_audit_mapping(
    value: Any,
    *,
    fields: frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _staged_audit_schema_error(path, "expected one JSON object")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        _staged_audit_schema_error(
            path,
            f"exact keys required; missing={missing}, unknown={unknown}",
        )
    return value


def _closed_audit_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _staged_audit_schema_error(path, f"expected integer >= {minimum}")
    return value


def _closed_audit_bool(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        _staged_audit_schema_error(path, "expected boolean")
    return value


def _closed_audit_exact_string(value: Any, *, expected: str, path: str) -> str:
    if value != expected:
        _staged_audit_schema_error(path, f"expected {expected!r}")
    return value


def _closed_audit_sha256(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _staged_audit_schema_error(path, "expected a lowercase SHA-256 digest")
    return value


def _closed_audit_id_list(
    value: Any,
    *,
    pattern: re.Pattern[str],
    path: str,
) -> list[str]:
    if not isinstance(value, list):
        _staged_audit_schema_error(path, "expected an array of opaque IDs")
    if any(not isinstance(item, str) or pattern.fullmatch(item) is None for item in value):
        _staged_audit_schema_error(path, "contains a malformed opaque ID")
    if len(set(value)) != len(value):
        _staged_audit_schema_error(path, "opaque IDs must be unique")
    return value


def _closed_audit_ordered_enum_list(
    value: Any,
    *,
    allowed: Sequence[str],
    path: str,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        _staged_audit_schema_error(path, "expected an ordered string array")
    if not allow_empty and not value:
        _staged_audit_schema_error(path, "array must not be empty")
    if len(set(value)) != len(value):
        _staged_audit_schema_error(path, "values must be unique")
    unknown = set(value) - set(allowed)
    if unknown:
        _staged_audit_schema_error(path, f"contains unsupported values: {sorted(unknown)}")
    canonical = [item for item in allowed if item in set(value)]
    if value != canonical:
        _staged_audit_schema_error(path, "values are not in canonical order")
    return value


def _closed_audit_candidate_universe(count: int) -> list[str]:
    return [f"candidate_{index:04d}" for index in range(1, count + 1)]


def _closed_audit_evidence_universe(count: int) -> list[str]:
    return [f"evidence_{index:04d}" for index in range(1, count + 1)]


def _validate_reasoning_trace_presence(value: Any, *, path: str) -> None:
    trace = _closed_audit_mapping(
        value,
        fields=_REASONING_TRACE_PRESENCE_FIELDS,
        path=path,
    )
    available = _closed_audit_bool(
        trace["response_trace_available"],
        path=f"{path}.response_trace_available",
    )
    attempt_count = _closed_audit_int(
        trace["completion_attempt_count"],
        path=f"{path}.completion_attempt_count",
    )
    reasoning_content_count = _closed_audit_int(
        trace["reasoning_content_present_count"],
        path=f"{path}.reasoning_content_present_count",
    )
    reasoning_count = _closed_audit_int(
        trace["reasoning_present_count"],
        path=f"{path}.reasoning_present_count",
    )
    any_reasoning = _closed_audit_bool(
        trace["any_reasoning_present"],
        path=f"{path}.any_reasoning_present",
    )
    if available != (attempt_count > 0):
        _staged_audit_schema_error(
            path,
            "response_trace_available must agree with completion_attempt_count",
        )
    if reasoning_content_count > attempt_count or reasoning_count > attempt_count:
        _staged_audit_schema_error(path, "reasoning counts exceed completion attempts")
    if any_reasoning != bool(reasoning_content_count or reasoning_count):
        _staged_audit_schema_error(path, "reasoning-presence boolean disagrees with counts")


def _validate_proposal_stage(value: Any, *, stage_name: str, path: str) -> None:
    stage = _closed_audit_mapping(
        value,
        fields=_STAGED_PROPOSAL_STAGE_FIELDS,
        path=path,
    )
    _closed_audit_exact_string(stage["stage"], expected=stage_name, path=f"{path}.stage")
    _closed_audit_sha256(stage["request_sha256"], path=f"{path}.request_sha256")
    _closed_audit_sha256(stage["response_sha256"], path=f"{path}.response_sha256")
    evidence_count = _closed_audit_int(
        stage["evidence_block_count"],
        path=f"{path}.evidence_block_count",
        minimum=1,
    )
    _closed_audit_ordered_enum_list(
        stage["source_families"],
        allowed=ALL_SOURCE_FAMILIES,
        path=f"{path}.source_families",
    )
    _closed_audit_int(
        stage["validated_proposal_count"],
        path=f"{path}.validated_proposal_count",
    )
    evidence_map = stage["evidence_id_map_to_original"]
    if not isinstance(evidence_map, Mapping):
        _staged_audit_schema_error(
            f"{path}.evidence_id_map_to_original",
            "expected an opaque evidence-ID mapping",
        )
    if list(evidence_map) != _closed_audit_evidence_universe(evidence_count):
        _staged_audit_schema_error(
            f"{path}.evidence_id_map_to_original",
            "keys must be the canonical stage evidence universe",
        )
    for source_id, original_id in evidence_map.items():
        if (
            not isinstance(source_id, str)
            or _EVIDENCE_AUDIT_ID.fullmatch(source_id) is None
            or not isinstance(original_id, str)
            or _EVIDENCE_AUDIT_ID.fullmatch(original_id) is None
        ):
            _staged_audit_schema_error(
                f"{path}.evidence_id_map_to_original",
                "contains a malformed opaque evidence ID",
            )
    if len(set(evidence_map.values())) != len(evidence_map):
        _staged_audit_schema_error(
            f"{path}.evidence_id_map_to_original",
            "mapped evidence IDs must be unique",
        )
    mapped_grounding_ids = _closed_audit_id_list(
        stage["mapped_grounding_evidence_ids"],
        pattern=_EVIDENCE_AUDIT_ID,
        path=f"{path}.mapped_grounding_evidence_ids",
    )
    if mapped_grounding_ids != sorted(mapped_grounding_ids):
        _staged_audit_schema_error(
            f"{path}.mapped_grounding_evidence_ids",
            "IDs must be in canonical sorted order",
        )
    if not set(mapped_grounding_ids) <= set(evidence_map.values()):
        _staged_audit_schema_error(
            f"{path}.mapped_grounding_evidence_ids",
            "contains IDs outside the stage evidence map",
        )
    proposal_count = stage["validated_proposal_count"]
    if bool(mapped_grounding_ids) != bool(proposal_count):
        _staged_audit_schema_error(
            path,
            "grounded evidence must be present exactly when proposals were validated",
        )
    _validate_reasoning_trace_presence(
        stage["reasoning_trace_presence"],
        path=f"{path}.reasoning_trace_presence",
    )


def _validate_selection_postprocessor(
    value: Any,
    *,
    candidate_pool_count: int,
    remote_ids: Sequence[str],
    mandatory_ids: Sequence[str],
    reserve_ids: Sequence[str],
    final_count: int,
    path: str,
) -> Mapping[str, Any]:
    audit = _closed_audit_mapping(
        value,
        fields=_STAGED_SELECTION_POSTPROCESSOR_FIELDS,
        path=path,
    )
    _closed_audit_exact_string(
        audit["schema_version"],
        expected=MINIMAL_STAGED_SELECTION_OUTPUT_SCHEMA,
        path=f"{path}.schema_version",
    )
    _closed_audit_exact_string(
        audit["postprocessor_version"],
        expected=MINIMAL_STAGED_SELECTION_POSTPROCESSOR_VERSION,
        path=f"{path}.postprocessor_version",
    )
    for field_name in ("postprocessor_code_sha256", "input_sha256", "output_sha256"):
        _closed_audit_sha256(audit[field_name], path=f"{path}.{field_name}")
    expected_code_sha256 = hashlib.sha256(
        Path(_minimal_selection_module.__file__).read_bytes()
    ).hexdigest()
    if audit["postprocessor_code_sha256"] != expected_code_sha256:
        _staged_audit_schema_error(path, "postprocessor code identity is stale or mismatched")

    mirrored_id_lists = {
        "remote_selected_candidate_ids": list(remote_ids),
        "mandatory_coverage_candidate_ids": list(mandatory_ids),
        "high_confidence_reserve_candidate_ids": list(reserve_ids),
    }
    for field_name, expected_ids in mirrored_id_lists.items():
        observed = _closed_audit_id_list(
            audit[field_name],
            pattern=_CANDIDATE_AUDIT_ID,
            path=f"{path}.{field_name}",
        )
        if observed != expected_ids:
            _staged_audit_schema_error(path, f"{field_name} disagrees with final stage")
    omitted_ids = _closed_audit_id_list(
        audit["omitted_candidate_ids"],
        pattern=_CANDIDATE_AUDIT_ID,
        path=f"{path}.omitted_candidate_ids",
    )

    target_families = _closed_audit_ordered_enum_list(
        audit["candidate_pool_target_source_families"],
        allowed=ALL_SOURCE_FAMILIES,
        path=f"{path}.candidate_pool_target_source_families",
    )
    family_counts = audit["candidate_pool_source_family_counts"]
    if not isinstance(family_counts, Mapping) or set(family_counts) != set(target_families):
        _staged_audit_schema_error(
            f"{path}.candidate_pool_source_family_counts",
            "keys must exactly match the candidate-pool families",
        )
    for family, count in family_counts.items():
        _closed_audit_int(
            count,
            path=f"{path}.candidate_pool_source_family_counts.{family}",
            minimum=1,
        )
    request_families = _closed_audit_ordered_enum_list(
        audit["original_request_source_families"],
        allowed=ALL_SOURCE_FAMILIES,
        path=f"{path}.original_request_source_families",
    )
    missing_request_families = _closed_audit_ordered_enum_list(
        audit["original_request_families_without_candidate"],
        allowed=ALL_SOURCE_FAMILIES,
        path=f"{path}.original_request_families_without_candidate",
        allow_empty=True,
    )
    expected_missing = [family for family in request_families if family not in target_families]
    if missing_request_families != expected_missing:
        _staged_audit_schema_error(
            path,
            "original-request families without a candidate are inconsistent",
        )
    target_roles = _closed_audit_ordered_enum_list(
        audit["target_roles"],
        allowed=_STAGED_ROLE_ORDER,
        path=f"{path}.target_roles",
    )
    covered_families = _closed_audit_ordered_enum_list(
        audit["covered_source_families"],
        allowed=ALL_SOURCE_FAMILIES,
        path=f"{path}.covered_source_families",
    )
    covered_roles = _closed_audit_ordered_enum_list(
        audit["covered_roles"],
        allowed=_STAGED_ROLE_ORDER,
        path=f"{path}.covered_roles",
    )
    if not set(covered_families) <= set(target_families):
        _staged_audit_schema_error(path, "covered families are outside the candidate pool")
    if not set(covered_roles) <= set(target_roles):
        _staged_audit_schema_error(path, "covered roles are outside the candidate pool")

    pool_complete = _closed_audit_bool(
        audit["candidate_pool_coverage_complete"],
        path=f"{path}.candidate_pool_coverage_complete",
    )
    expected_pool_complete = set(target_families) <= set(covered_families) and set(
        target_roles
    ) <= set(covered_roles)
    if pool_complete != expected_pool_complete:
        _staged_audit_schema_error(path, "candidate-pool coverage boolean is inconsistent")
    request_complete = _closed_audit_bool(
        audit["original_request_candidate_coverage_complete"],
        path=f"{path}.original_request_candidate_coverage_complete",
    )
    expected_request_complete = not missing_request_families and set(request_families) <= set(
        covered_families
    )
    if request_complete != expected_request_complete:
        _staged_audit_schema_error(path, "original-request coverage boolean is inconsistent")
    reserve_complete = _closed_audit_bool(
        audit["high_confidence_reserve_complete"],
        path=f"{path}.high_confidence_reserve_complete",
    )
    cap_limited = _closed_audit_bool(audit["cap_limited"], path=f"{path}.cap_limited")
    if cap_limited != bool(not pool_complete or not reserve_complete):
        _staged_audit_schema_error(path, "cap-limited boolean is inconsistent")
    postprocessor_final_count = _closed_audit_int(
        audit["final_count"],
        path=f"{path}.final_count",
        minimum=1,
    )
    if postprocessor_final_count != final_count:
        _staged_audit_schema_error(path, "final count disagrees with final stage")

    retained_ids = [*remote_ids, *mandatory_ids, *reserve_ids]
    if len(set(retained_ids)) != len(retained_ids):
        _staged_audit_schema_error(path, "selected/backfilled candidate IDs overlap")
    if set(retained_ids) & set(omitted_ids):
        _staged_audit_schema_error(path, "retained and omitted candidate IDs overlap")
    if set(retained_ids) | set(omitted_ids) != set(
        _closed_audit_candidate_universe(candidate_pool_count)
    ):
        _staged_audit_schema_error(path, "candidate-pool disposition is incomplete")
    return audit


def _validate_final_selection_stage(value: Any, *, path: str) -> Mapping[str, Any]:
    stage = _closed_audit_mapping(
        value,
        fields=_STAGED_FINAL_SELECTION_FIELDS,
        path=path,
    )
    _closed_audit_exact_string(
        stage["stage"],
        expected="final_contract_selection",
        path=f"{path}.stage",
    )
    _closed_audit_sha256(stage["request_sha256"], path=f"{path}.request_sha256")
    _closed_audit_sha256(stage["response_sha256"], path=f"{path}.response_sha256")
    _closed_audit_int(
        stage["evidence_block_count"],
        path=f"{path}.evidence_block_count",
        minimum=1,
    )
    candidate_pool_count = _closed_audit_int(
        stage["candidate_pool_count"],
        path=f"{path}.candidate_pool_count",
        minimum=1,
    )
    selected_ids = _closed_audit_id_list(
        stage["selected_candidate_ids"],
        pattern=_CANDIDATE_AUDIT_ID,
        path=f"{path}.selected_candidate_ids",
    )
    remote_ids = _closed_audit_id_list(
        stage["remote_selected_candidate_ids"],
        pattern=_CANDIDATE_AUDIT_ID,
        path=f"{path}.remote_selected_candidate_ids",
    )
    backfilled_ids = _closed_audit_id_list(
        stage["backfilled_candidate_ids"],
        pattern=_CANDIDATE_AUDIT_ID,
        path=f"{path}.backfilled_candidate_ids",
    )
    mandatory_ids = _closed_audit_id_list(
        stage["mandatory_coverage_candidate_ids"],
        pattern=_CANDIDATE_AUDIT_ID,
        path=f"{path}.mandatory_coverage_candidate_ids",
    )
    reserve_ids = _closed_audit_id_list(
        stage["high_confidence_reserve_candidate_ids"],
        pattern=_CANDIDATE_AUDIT_ID,
        path=f"{path}.high_confidence_reserve_candidate_ids",
    )
    selected_count = _closed_audit_int(stage["selected_count"], path=f"{path}.selected_count")
    remote_count = _closed_audit_int(
        stage["remote_selected_count"],
        path=f"{path}.remote_selected_count",
    )
    final_count = _closed_audit_int(
        stage["final_selected_count"],
        path=f"{path}.final_selected_count",
        minimum=1,
    )
    if selected_ids != remote_ids or selected_count != len(selected_ids):
        _staged_audit_schema_error(path, "historical selected fields must equal remote fields")
    if remote_count != len(remote_ids):
        _staged_audit_schema_error(path, "remote selected count disagrees with IDs")
    if backfilled_ids != [*mandatory_ids, *reserve_ids]:
        _staged_audit_schema_error(path, "backfill must be mandatory coverage plus reserve")
    if final_count != remote_count + len(mandatory_ids) + len(reserve_ids):
        _staged_audit_schema_error(path, "final selected count disagrees with backfill")
    if final_count > candidate_pool_count:
        _staged_audit_schema_error(path, "selected more contracts than the candidate pool")
    _closed_audit_exact_string(
        stage["selection_backfill_version"],
        expected=STAGED_SELECTION_BACKFILL_VERSION,
        path=f"{path}.selection_backfill_version",
    )
    _closed_audit_exact_string(
        stage["selection_union_postprocessing_version"],
        expected=STAGED_SELECTION_UNION_POSTPROCESSING_VERSION,
        path=f"{path}.selection_union_postprocessing_version",
    )
    _validate_selection_postprocessor(
        stage["selection_postprocessor"],
        candidate_pool_count=candidate_pool_count,
        remote_ids=remote_ids,
        mandatory_ids=mandatory_ids,
        reserve_ids=reserve_ids,
        final_count=final_count,
        path=f"{path}.selection_postprocessor",
    )
    _validate_reasoning_trace_presence(
        stage["reasoning_trace_presence"],
        path=f"{path}.reasoning_trace_presence",
    )
    return stage


def _validate_safe_union_strength(value: Any, *, path: str) -> None:
    strength = _closed_audit_mapping(
        value,
        fields=_STAGED_SAFE_UNION_STRENGTH_FIELDS,
        path=path,
    )
    for field_name in _STAGED_SAFE_UNION_STRENGTH_FIELDS:
        _closed_audit_int(strength[field_name], path=f"{path}.{field_name}", minimum=1)


def _validate_safe_union(
    value: Any,
    *,
    validated_count: int,
    path: str,
) -> Mapping[str, Any]:
    audit = _closed_audit_mapping(value, fields=_STAGED_SAFE_UNION_FIELDS, path=path)
    identity = _closed_audit_mapping(
        audit["identity"],
        fields=_STAGED_SAFE_UNION_IDENTITY_FIELDS,
        path=f"{path}.identity",
    )
    expected_identity = safe_staged_proposal_union_identity().as_dict()
    for field_name, expected in expected_identity.items():
        _closed_audit_exact_string(
            identity[field_name],
            expected=expected,
            path=f"{path}.identity.{field_name}",
        )
    _closed_audit_exact_string(
        identity["policy_version"],
        expected=SAFE_STAGED_PROPOSAL_UNION_POLICY_VERSION,
        path=f"{path}.identity.policy_version",
    )
    _closed_audit_exact_string(
        identity["input_schema_version"],
        expected=SAFE_STAGED_PROPOSAL_UNION_INPUT_SCHEMA_VERSION,
        path=f"{path}.identity.input_schema_version",
    )
    _closed_audit_exact_string(
        identity["output_schema_version"],
        expected=SAFE_STAGED_PROPOSAL_UNION_OUTPUT_SCHEMA_VERSION,
        path=f"{path}.identity.output_schema_version",
    )
    _closed_audit_exact_string(
        identity["hash_domain_version"],
        expected=SAFE_STAGED_PROPOSAL_UNION_HASH_DOMAIN_VERSION,
        path=f"{path}.identity.hash_domain_version",
    )
    _closed_audit_sha256(audit["input_sha256"], path=f"{path}.input_sha256")
    _closed_audit_sha256(audit["output_sha256"], path=f"{path}.output_sha256")
    input_count = _closed_audit_int(
        audit["input_candidate_count"],
        path=f"{path}.input_candidate_count",
        minimum=1,
    )
    if input_count != validated_count:
        _staged_audit_schema_error(path, "input count disagrees with validated proposals")

    disposition_lists: dict[str, list[str]] = {}
    for field_name, disposition in (
        ("representative_candidate_ids", "representative"),
        ("exact_duplicate_candidate_ids", "exact_duplicate"),
        ("compatible_role_merge_candidate_ids", "compatible_role_merge"),
        ("omitted_conflict_candidate_ids", "omitted_conflict"),
    ):
        disposition_lists[disposition] = _closed_audit_id_list(
            audit[field_name],
            pattern=_CANDIDATE_AUDIT_ID,
            path=f"{path}.{field_name}",
        )
    representatives = disposition_lists["representative"]
    if not representatives:
        _staged_audit_schema_error(path, "safe union must retain at least one representative")
    partition = [candidate_id for values in disposition_lists.values() for candidate_id in values]
    if len(set(partition)) != len(partition) or set(partition) != set(
        _closed_audit_candidate_universe(input_count)
    ):
        _staged_audit_schema_error(path, "candidate disposition lists are not a full partition")

    dispositions = audit["dispositions"]
    if not isinstance(dispositions, list) or len(dispositions) != input_count:
        _staged_audit_schema_error(f"{path}.dispositions", "expected one row per input")
    observed_disposition_ids: list[str] = []
    retained_by_candidate: dict[str, str] = {}
    representative_set = set(representatives)
    for index, raw in enumerate(dispositions):
        row_path = f"{path}.dispositions[{index}]"
        row = _closed_audit_mapping(
            raw,
            fields=_STAGED_SAFE_UNION_DISPOSITION_FIELDS,
            path=row_path,
        )
        candidate_id = row["candidate_id"]
        retained_id = row["retained_candidate_id"]
        disposition = row["disposition"]
        if (
            not isinstance(candidate_id, str)
            or _CANDIDATE_AUDIT_ID.fullmatch(candidate_id) is None
            or not isinstance(retained_id, str)
            or _CANDIDATE_AUDIT_ID.fullmatch(retained_id) is None
        ):
            _staged_audit_schema_error(row_path, "contains a malformed candidate ID")
        if disposition not in _STAGED_DISPOSITIONS:
            _staged_audit_schema_error(row_path, "contains an unsupported disposition")
        if candidate_id not in disposition_lists[disposition]:
            _staged_audit_schema_error(row_path, "disposition disagrees with accounting list")
        if retained_id not in representative_set:
            _staged_audit_schema_error(row_path, "retained ID is not a representative")
        if disposition == "representative" and retained_id != candidate_id:
            _staged_audit_schema_error(row_path, "representative must retain itself")
        observed_disposition_ids.append(candidate_id)
        retained_by_candidate[candidate_id] = retained_id
    if observed_disposition_ids != _closed_audit_candidate_universe(input_count):
        _staged_audit_schema_error(path, "dispositions are not in canonical input order")

    conflicts = audit["conflicts"]
    if not isinstance(conflicts, list):
        _staged_audit_schema_error(f"{path}.conflicts", "expected an array")
    conflict_ids: set[str] = set()
    conflict_omitted_ids: list[str] = []
    for index, raw in enumerate(conflicts):
        conflict_path = f"{path}.conflicts[{index}]"
        conflict = _closed_audit_mapping(
            raw,
            fields=_STAGED_SAFE_UNION_CONFLICT_FIELDS,
            path=conflict_path,
        )
        conflict_id = conflict["conflict_id"]
        if not isinstance(conflict_id, str) or _CONFLICT_AUDIT_ID.fullmatch(conflict_id) is None:
            _staged_audit_schema_error(conflict_path, "contains a malformed conflict ID")
        if conflict_id in conflict_ids:
            _staged_audit_schema_error(conflict_path, "conflict IDs must be unique")
        conflict_ids.add(conflict_id)
        retained_id = conflict["retained_candidate_id"]
        if retained_id not in representative_set:
            _staged_audit_schema_error(conflict_path, "retained ID is not a representative")
        omitted_ids = _closed_audit_id_list(
            conflict["omitted_candidate_ids"],
            pattern=_CANDIDATE_AUDIT_ID,
            path=f"{conflict_path}.omitted_candidate_ids",
        )
        if not omitted_ids or not set(omitted_ids) <= set(disposition_lists["omitted_conflict"]):
            _staged_audit_schema_error(conflict_path, "omitted IDs disagree with accounting")
        differing = _closed_audit_ordered_enum_list(
            conflict["differing_non_role_fields"],
            allowed=_STAGED_NON_ROLE_FIELDS,
            path=f"{conflict_path}.differing_non_role_fields",
        )
        if not differing:  # pragma: no cover - nonempty enforced by helper
            _staged_audit_schema_error(conflict_path, "conflict must identify a differing field")
        if any(retained_by_candidate[candidate_id] != retained_id for candidate_id in omitted_ids):
            _staged_audit_schema_error(
                conflict_path,
                "conflict retained ID disagrees with candidate dispositions",
            )
        expected_conflict_id = "conflict_" + _content_sha256(
            {
                "policy_version": SAFE_STAGED_PROPOSAL_UNION_POLICY_VERSION,
                "retained_candidate_id": retained_id,
                "omitted_candidate_ids": omitted_ids,
                "differing_non_role_fields": differing,
            }
        )
        if conflict_id != expected_conflict_id:
            _staged_audit_schema_error(conflict_path, "conflict ID hash is inconsistent")
        _validate_safe_union_strength(
            conflict["retained_strength"], path=f"{conflict_path}.retained_strength"
        )
        _validate_safe_union_strength(
            conflict["omitted_strength"], path=f"{conflict_path}.omitted_strength"
        )
        conflict_omitted_ids.extend(omitted_ids)
    if len(set(conflict_omitted_ids)) != len(conflict_omitted_ids) or set(
        conflict_omitted_ids
    ) != set(disposition_lists["omitted_conflict"]):
        _staged_audit_schema_error(path, "conflicts do not account for omitted variants")

    selection_map = audit["selection_candidate_to_representative_id"]
    expected_selection_ids = _closed_audit_candidate_universe(len(representatives))
    if not isinstance(selection_map, Mapping) or list(selection_map) != expected_selection_ids:
        _staged_audit_schema_error(
            f"{path}.selection_candidate_to_representative_id",
            "keys must be the canonical selector candidate universe",
        )
    if list(selection_map.values()) != representatives:
        _staged_audit_schema_error(
            f"{path}.selection_candidate_to_representative_id",
            "values must be the ordered representative IDs",
        )
    for field_name in (
        "incompatible_variant_support_or_roles_propagated",
        "semantic_fields_used_for_conflict_ranking",
        "patient_rows_or_observed_labels_used",
    ):
        if _closed_audit_bool(audit[field_name], path=f"{path}.{field_name}"):
            _staged_audit_schema_error(path, f"{field_name} must be false")
    return audit


def _validate_proposal_union(value: Any, *, path: str) -> Mapping[str, Any]:
    union = _closed_audit_mapping(
        value,
        fields=_STAGED_PROPOSAL_UNION_FIELDS,
        path=path,
    )
    validated_count = _closed_audit_int(
        union["validated_proposal_count"],
        path=f"{path}.validated_proposal_count",
        minimum=1,
    )
    unique_count = _closed_audit_int(
        union["unique_contract_count"],
        path=f"{path}.unique_contract_count",
        minimum=1,
    )
    exact_duplicates = _closed_audit_int(
        union["exact_duplicate_count"],
        path=f"{path}.exact_duplicate_count",
    )
    if validated_count != unique_count + exact_duplicates:
        _staged_audit_schema_error(path, "exact duplicate accounting is inconsistent")
    same_name = _closed_audit_mapping(
        union["same_name_merge"],
        fields=_STAGED_SAME_NAME_MERGE_FIELDS,
        path=f"{path}.same_name_merge",
    )
    _closed_audit_exact_string(
        same_name["version"],
        expected=STAGED_SAME_NAME_MERGE_VERSION,
        path=f"{path}.same_name_merge.version",
    )
    merged_count = _closed_audit_int(
        same_name["merged_contract_count"],
        path=f"{path}.same_name_merge.merged_contract_count",
    )
    final_pool_count = _closed_audit_int(
        same_name["final_candidate_pool_count"],
        path=f"{path}.same_name_merge.final_candidate_pool_count",
        minimum=1,
    )
    safe_union = _validate_safe_union(
        union["safe_union"],
        validated_count=validated_count,
        path=f"{path}.safe_union",
    )
    representatives = safe_union["representative_candidate_ids"]
    compatible_merges = safe_union["compatible_role_merge_candidate_ids"]
    omitted_conflicts = safe_union["omitted_conflict_candidate_ids"]
    if exact_duplicates != len(safe_union["exact_duplicate_candidate_ids"]):
        _staged_audit_schema_error(path, "exact duplicate count disagrees with safe union")
    if merged_count != len(compatible_merges):
        _staged_audit_schema_error(path, "compatible merge count disagrees with safe union")
    if final_pool_count != len(representatives):
        _staged_audit_schema_error(path, "candidate-pool count disagrees with safe union")
    if unique_count != len(representatives) + len(compatible_merges) + len(omitted_conflicts):
        _staged_audit_schema_error(path, "unique proposal accounting is inconsistent")
    return union


def _validate_closed_staged_fusion_audit(value: Any) -> None:
    """Accept only the presence-only, content-free staged v3 audit schema."""

    audit = _closed_audit_mapping(
        value,
        fields=_STAGED_AUDIT_TOP_LEVEL_FIELDS,
        path="$",
    )
    _closed_audit_exact_string(
        audit["schema_version"],
        expected=STAGED_FUSION_AUDIT_SCHEMA_VERSION,
        path="$.schema_version",
    )
    _closed_audit_int(audit["outer_fold"], path="$.outer_fold", minimum=1)
    _closed_audit_sha256(audit["split_fingerprint"], path="$.split_fingerprint")
    _closed_audit_sha256(
        audit["original_request_sha256"],
        path="$.original_request_sha256",
    )
    _closed_audit_sha256(
        audit["returned_response_sha256"],
        path="$.returned_response_sha256",
    )
    configured_cap = _closed_audit_int(
        audit["configured_final_cap"],
        path="$.configured_final_cap",
        minimum=1,
    )
    effective_cap = _closed_audit_int(
        audit["effective_final_cap"],
        path="$.effective_final_cap",
        minimum=1,
    )
    if configured_cap > 64 or effective_cap > configured_cap:
        _staged_audit_schema_error("$", "invalid configured/effective candidate cap")
    _closed_audit_exact_string(
        audit["selection_backfill_version"],
        expected=STAGED_SELECTION_BACKFILL_VERSION,
        path="$.selection_backfill_version",
    )
    _closed_audit_exact_string(
        audit["selection_union_postprocessing_version"],
        expected=STAGED_SELECTION_UNION_POSTPROCESSING_VERSION,
        path="$.selection_union_postprocessing_version",
    )
    role_policy = _closed_audit_mapping(
        audit["role_specific_proposal_policy"],
        fields=_STAGED_ROLE_POLICY_FIELDS,
        path="$.role_specific_proposal_policy",
    )
    _closed_audit_exact_string(
        role_policy["version"],
        expected=_STAGED_ROLE_POLICY_VERSION,
        path="$.role_specific_proposal_policy.version",
    )
    if role_policy["eligible_source_families"] != list(ALL_SOURCE_FAMILIES):
        _staged_audit_schema_error(
            "$.role_specific_proposal_policy.eligible_source_families",
            "all evidence families must be eligible in canonical order",
        )
    for field_name in (
        "neural_query_moments_eligible",
        "matched_pair_htr_embedding_and_tfidf_evidence_eligible",
    ):
        if not _closed_audit_bool(
            role_policy[field_name], path=f"$.role_specific_proposal_policy.{field_name}"
        ):
            _staged_audit_schema_error(
                "$.role_specific_proposal_policy",
                f"{field_name} must be true",
            )

    stages = audit["stages"]
    if not isinstance(stages, list) or len(stages) != 4:
        _staged_audit_schema_error("$.stages", "expected exactly four ordered stages")
    proposal_counts = []
    for index, stage_name in enumerate(_STAGED_PROPOSAL_STAGE_NAMES):
        _validate_proposal_stage(stages[index], stage_name=stage_name, path=f"$.stages[{index}]")
        proposal_counts.append(stages[index]["validated_proposal_count"])
    full_stage = stages[0]
    full_evidence_ids = _closed_audit_evidence_universe(full_stage["evidence_block_count"])
    if full_stage["evidence_id_map_to_original"] != {
        evidence_id: evidence_id for evidence_id in full_evidence_ids
    }:
        _staged_audit_schema_error(
            "$.stages[0].evidence_id_map_to_original",
            "the full-evidence stage map must be the canonical identity",
        )
    for index in (1, 2):
        role_stage = stages[index]
        if not set(role_stage["evidence_id_map_to_original"].values()) <= set(full_evidence_ids):
            _staged_audit_schema_error(
                f"$.stages[{index}].evidence_id_map_to_original",
                "role-stage evidence must map into the full-evidence request",
            )
        if not set(role_stage["source_families"]) <= set(full_stage["source_families"]):
            _staged_audit_schema_error(
                f"$.stages[{index}].source_families",
                "role-stage families must be a subset of the full-evidence stage",
            )
    final_stage = _validate_final_selection_stage(stages[3], path="$.stages[3]")
    if final_stage["evidence_block_count"] != full_stage["evidence_block_count"]:
        _staged_audit_schema_error(
            "$.stages[3].evidence_block_count",
            "selection must receive the complete original evidence request",
        )
    selection_postprocessor = final_stage["selection_postprocessor"]
    if selection_postprocessor["original_request_source_families"] != full_stage["source_families"]:
        _staged_audit_schema_error(
            "$.stages[3].selection_postprocessor.original_request_source_families",
            "original-request families disagree with the full-evidence stage",
        )
    if selection_postprocessor["output_sha256"] != audit["returned_response_sha256"]:
        _staged_audit_schema_error(
            "$.stages[3].selection_postprocessor.output_sha256",
            "postprocessor output hash disagrees with the returned response",
        )
    proposal_union = _validate_proposal_union(audit["proposal_union"], path="$.proposal_union")
    if sum(proposal_counts) != proposal_union["validated_proposal_count"]:
        _staged_audit_schema_error("$", "proposal stage counts disagree with proposal union")
    if (
        final_stage["candidate_pool_count"]
        != proposal_union["same_name_merge"]["final_candidate_pool_count"]
    ):
        _staged_audit_schema_error("$", "candidate pool count disagrees with proposal union")

    top_backfilled = _closed_audit_id_list(
        audit["backfilled_candidate_ids"],
        pattern=_CANDIDATE_AUDIT_ID,
        path="$.backfilled_candidate_ids",
    )
    top_remote_count = _closed_audit_int(
        audit["remote_selected_count"],
        path="$.remote_selected_count",
    )
    top_final_count = _closed_audit_int(
        audit["final_selected_count"],
        path="$.final_selected_count",
        minimum=1,
    )
    returned_count = _closed_audit_int(
        audit["returned_proposal_count"],
        path="$.returned_proposal_count",
        minimum=1,
    )
    if (
        top_backfilled != final_stage["backfilled_candidate_ids"]
        or top_remote_count != final_stage["remote_selected_count"]
        or top_final_count != final_stage["final_selected_count"]
        or returned_count != top_final_count
        or top_final_count > effective_cap
    ):
        _staged_audit_schema_error("$", "top-level selection accounting is inconsistent")
    if selection_postprocessor["cap_limited"] and top_final_count != effective_cap:
        _staged_audit_schema_error("$", "cap-limited selection did not exhaust the effective cap")


def _write_immutable_parquet(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        try:
            assert_frame_equal(existing, frame, check_like=False, check_dtype=True)
        except AssertionError as exc:
            raise RuntimeError(f"refusing to mutate frozen prediction artifact: {path}") from exc
    else:
        frame.to_parquet(path, index=False)
    return sha256_file(path)


def _reject_forbidden_columns(columns: Sequence[Any], *, source: str) -> None:
    forbidden = [str(column) for column in columns if _FORBIDDEN_NAME.search(str(column))]
    if forbidden:
        raise ValueError(f"{source} contains oracle/true columns: {forbidden}")


def load_sanitized_dataset(
    dataset_path: Path | str,
    *,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
) -> pd.DataFrame:
    """Load only the model columns and assign canonical positional row IDs."""

    sanitized, _ = _load_sanitized_dataset_snapshot(
        dataset_path,
        text_column=text_column,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
    )
    return sanitized


def _load_sanitized_dataset_snapshot(
    dataset_path: Path | str,
    *,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
) -> tuple[pd.DataFrame, str]:
    """Project model columns from the same whole-file snapshot that is hashed."""

    configured = [text_column, treatment_column, outcome_column]
    if len(set(configured)) != len(configured):
        raise ValueError("text, treatment, and outcome columns must be distinct")
    _reject_forbidden_columns(configured, source="configured model columns")
    # Column projection is the semantic safety boundary: patient prompts,
    # timelines, DGP fields, and oracle columns are never decoded into objects.
    # The complete file is first captured as opaque immutable bytes so the
    # whole-dataset digest identifies exactly the parquet snapshot projected.
    try:
        snapshot, artifact_sha256 = _read_path_snapshot(Path(dataset_path).resolve())
        sanitized = pd.read_parquet(io.BytesIO(snapshot), columns=configured)
    except Exception as exc:
        raise ValueError(
            "dataset could not be loaded with the exact sanitized column allowlist"
        ) from exc
    sanitized = sanitized.copy().reset_index(drop=True)
    sanitized.insert(0, "_oci_row_id", np.arange(len(sanitized), dtype=int))
    sanitized[text_column] = sanitized[text_column].fillna("").astype(str)
    treatment = pd.to_numeric(sanitized[treatment_column], errors="coerce")
    outcome = pd.to_numeric(sanitized[outcome_column], errors="coerce")
    if treatment.isna().any() or outcome.isna().any():
        raise ValueError("treatment and outcome must be finite numeric values")
    if not set(treatment.unique()).issubset({0, 1}):
        raise ValueError("treatment must be binary and encoded as 0/1")
    sanitized[treatment_column] = treatment.astype(int)
    sanitized[outcome_column] = outcome.astype(float)
    return sanitized, artifact_sha256


@dataclass(frozen=True)
class LegacyFullOuterEvidence:
    rows_by_outer_fold: Mapping[int, Mapping[str, Any]]
    artifact_path: str
    artifact_sha256: str
    ignored_non_full_context_count: int
    validated_inner_context_count: int
    inner_contexts_per_outer: int
    dropped_diagnostic_field_count: int
    exact_inner_recurrence_group_count: int
    exact_inner_recurrent_term_count: int


def _sanitize_retained_legacy_digest(value: Any) -> tuple[Any, int]:
    """Remove diagnostics that are never consumed by fusion compaction."""

    dropped = 0
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.lower() == "metrics" or _FORBIDDEN_NAME.search(key):
                dropped += 1
                continue
            sanitized, child_dropped = _sanitize_retained_legacy_digest(child)
            cleaned[key] = sanitized
            dropped += child_dropped
        return cleaned, dropped
    if isinstance(value, (list, tuple)):
        cleaned_items = []
        for child in value:
            sanitized, child_dropped = _sanitize_retained_legacy_digest(child)
            cleaned_items.append(sanitized)
            dropped += child_dropped
        return cleaned_items, dropped
    return value, 0


def _normalize_exact_inner_term(value: Any) -> str:
    """Normalize lexical evidence without doing semantic or topic-ID matching."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 2 or len(text) > 160:
        return ""
    return text


def _legacy_exact_inner_terms(
    digest: Mapping[str, Any],
) -> set[tuple[str, str, str]]:
    """Return unique (family, role, normalized term) support in one inner fold."""

    found: set[tuple[str, str, str]] = set()
    for role_key, role in (
        ("confounders", "confounder"),
        ("effect_modifiers", "effect_modifier"),
    ):
        section = digest.get(role_key)
        if not isinstance(section, Mapping):
            continue
        for group in section.get("bow_blurbs") or []:
            if not isinstance(group, Mapping):
                continue
            family = _bow_group_family(group, role)
            for row in group.get("rows") or []:
                if not isinstance(row, Mapping):
                    continue
                term = _normalize_exact_inner_term(
                    row.get("feature") or row.get("term") or row.get("phrase")
                )
                if term:
                    found.add((family, role, term))
        for contrast in section.get("embedding_chunks") or []:
            if not isinstance(contrast, Mapping):
                continue
            family = _embedding_family(contrast)
            for row in contrast.get("concept_probe_scores") or []:
                if not isinstance(row, Mapping):
                    continue
                term = _normalize_exact_inner_term(
                    row.get("concept") or row.get("phrase") or row.get("label")
                )
                if term:
                    found.add((family, role, term))
        for group in section.get("htr_blurbs") or []:
            if not isinstance(group, Mapping):
                continue
            stage = str(group.get("stage") or "").casefold()
            families = (
                (HTR_NEURAL, MATCHED_PAIR_UPLIFT)
                if "pair" in stage or "uplift" in stage
                else (HTR_NEURAL,)
            )
            for row in group.get("rows") or []:
                if not isinstance(row, Mapping):
                    continue
                spans = row.get("top_token_spans")
                if not isinstance(spans, (list, tuple)):
                    continue
                for span in spans:
                    if not isinstance(span, Mapping):
                        continue
                    term = _normalize_exact_inner_term(
                        span.get("text") or span.get("token") or span.get("span")
                    )
                    if term:
                        for family in families:
                            found.add((family, role, term))
    return found


def _tfidf_exact_inner_terms(
    discovery: Mapping[str, Any],
) -> set[tuple[str, str, str]]:
    """Collect recurrent terms while deliberately ignoring latent topic IDs."""

    found: set[tuple[str, str, str]] = set()
    banks = discovery.get("topic_banks")
    if isinstance(banks, Mapping):
        for bank in ("treatment", "outcome", "effect"):
            bank_payload = banks.get(bank)
            if not isinstance(bank_payload, Mapping):
                continue
            role = "effect_modifier" if bank == "effect" else "confounder"
            for topic in bank_payload.get("topics") or []:
                if not isinstance(topic, Mapping):
                    continue
                # Never consume topic_id here: inner-fold latent components are
                # not aligned. Only exact normalized terms can recur.
                for raw_term in topic.get("terms") or []:
                    row = raw_term if isinstance(raw_term, Mapping) else {"term": raw_term}
                    term = _normalize_exact_inner_term(
                        row.get("term") or row.get("feature") or row.get("ngram")
                    )
                    if term:
                        found.add((TFIDF_TOPICS, role, term))

    orphan = discovery.get("effect_orphan_ngram_branch")
    if not isinstance(orphan, Mapping):
        for key in ("topic_score_tests", "topic_score_selection", "score_tests"):
            nested = discovery.get(key)
            if isinstance(nested, Mapping) and isinstance(
                nested.get("effect_orphan_ngram_branch"), Mapping
            ):
                orphan = nested["effect_orphan_ngram_branch"]
                break
    if isinstance(orphan, Mapping):
        selected_ids = {str(value) for value in orphan.get("selected_cluster_ids") or []}
        for cluster in orphan.get("selected_clusters") or orphan.get("clusters") or []:
            if not isinstance(cluster, Mapping):
                continue
            # Cluster IDs are used only to honor selection inside this one
            # context; they are never compared between inner folds.
            cluster_id = str(cluster.get("cluster_id") or cluster.get("topic_id") or "")
            if selected_ids and cluster_id not in selected_ids:
                continue
            values = (
                cluster.get("terms")
                or cluster.get("member_terms")
                or cluster.get("supporting_terms")
                or []
            )
            for raw_term in values:
                row = raw_term if isinstance(raw_term, Mapping) else {"term": raw_term}
                term = _normalize_exact_inner_term(
                    row.get("term") or row.get("feature") or row.get("ngram")
                )
                if term:
                    found.add((TFIDF_ORPHAN_NGRAMS, "effect_modifier", term))
    return found


def _build_exact_inner_recurrence(
    terms_by_inner_fold: Mapping[int, set[tuple[str, str, str]]],
) -> dict[str, Any]:
    inner_fold_count = len(terms_by_inner_fold)
    if inner_fold_count < 2:
        raise ValueError("exact-inner recurrence requires at least two inner folds")
    support: dict[tuple[str, str, str], set[int]] = {}
    for inner_fold, terms in sorted(terms_by_inner_fold.items()):
        for family, role, term in terms:
            support.setdefault((family, role, term), set()).add(int(inner_fold))
    by_group: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for (family, role, term), folds in support.items():
        if len(folds) >= 2:
            by_group.setdefault((family, role), []).append((term, len(folds)))
    groups: list[dict[str, Any]] = []
    for (family, role), rows in sorted(by_group.items()):
        ranked = sorted(rows, key=lambda row: (-row[1], row[0]))
        # Recurrence is evidence production, not a ranking gate. Preserve
        # every recurrent term; downstream architecture paging is responsible
        # for bounded model requests with exact coverage.
        retained = ranked
        groups.append(
            {
                "source_family": family,
                "role": role,
                "discovered_recurrent_term_count": len(ranked),
                "retained_term_count": len(retained),
                "terms": [
                    {
                        "term": term,
                        "inner_fold_support_count": count,
                        # Occurrences are deliberately de-duplicated within an
                        # inner fold, so this is an independent-fold count.
                        "occurrence_count": count,
                    }
                    for term, count in retained
                ],
            }
        )
    return {
        "schema_version": EXACT_INNER_RECURRENCE_VERSION,
        "normalization": "unicode_nfkc_casefold_nonword_to_space_exact_match",
        "inner_fold_count": inner_fold_count,
        "latent_topic_ids_compared_across_folds": False,
        "minimum_inner_fold_support": 2,
        "groups": groups,
    }


def _iter_allowlisted_legacy_records(snapshot: bytes):
    """Stream only split scalars and the three approved Stage-1 evidence maps."""

    try:
        import ijson
        from ijson.common import ObjectBuilder
    except ImportError as exc:  # pragma: no cover - available in the project env
        raise RuntimeError(
            "ijson is required to stream legacy handoffs without loading rich context"
        ) from exc

    scalar_keys = {
        "schema_version",
        "fold_key",
        "outer_fold",
        "inner_fold",
        "scope",
        "n_rows",
        "heldout_rows",
    }
    object_keys = {"importance", "embedding_contrast_evidence", "htr_evidence"}
    record: dict[str, Any] | None = None
    depth = 0
    top_key: str | None = None
    builder: Any | None = None
    builder_key: str | None = None
    builder_depth = 0
    for _, event, value in ijson.parse(io.BytesIO(snapshot), multiple_values=True):
        if event == "start_map" and depth == 0:
            record = {}

        if builder is not None:
            builder.event(event, value)
            if event in {"start_map", "start_array"}:
                builder_depth += 1
            elif event in {"end_map", "end_array"}:
                builder_depth -= 1
                if builder_depth == 0:
                    assert record is not None and builder_key is not None
                    record[builder_key] = builder.value
                    builder = None
                    builder_key = None

        old_depth = depth
        if event in {"start_map", "start_array"}:
            depth += 1
        elif event in {"end_map", "end_array"}:
            depth -= 1

        if builder is None:
            if event == "map_key" and depth == 1:
                top_key = str(value)
            elif (
                old_depth == 1 and top_key in object_keys and event in {"start_map", "start_array"}
            ):
                builder = ObjectBuilder()
                builder_key = top_key
                builder_depth = 1
                builder.event(event, value)
            elif (
                depth == 1
                and top_key in scalar_keys
                and event in {"string", "number", "boolean", "null"}
            ):
                assert record is not None
                record[top_key] = value
                top_key = None

        if event == "end_map" and depth == 0:
            assert record is not None
            yield record
            record = None
            top_key = None


def load_legacy_full_outer_evidence(path: Path | str) -> LegacyFullOuterEvidence:
    """Load exactly one sanitized full-outer record per legacy fold.

    Exact-inner records are validated as distinct split-scoped artifacts and
    reduced to normalized term recurrence counts. Raw inner contexts and fold-
    heldout rows are never exposed to the outer selector.
    """

    requested = Path(path).resolve()
    snapshot, artifact_sha256 = _read_path_snapshot(requested)
    by_fold: dict[int, Mapping[str, Any]] = {}
    ignored = 0
    seen = 0
    inner_by_outer: dict[int, set[int]] = {}
    inner_sizes: dict[int, list[tuple[int, int]]] = {}
    inner_terms_by_outer: dict[int, dict[int, set[tuple[str, str, str]]]] = {}
    dropped_diagnostic_fields = 0
    # This helper is deterministic and consumes only the three allowlisted maps.
    # Rich context, metrics, clinical examples, and raw model predictions never
    # cross the streaming parser boundary.
    from .multi_model_agentic_forest import _build_role_grouped_evidence_digest

    for line_number, row in enumerate(_iter_allowlisted_legacy_records(snapshot), start=1):
        seen += 1
        if row.get("schema_version") != LEGACY_HANDOFF_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported legacy handoff schema in record {line_number}: "
                f"{row.get('schema_version')!r}"
            )
        scope = str(row.get("scope"))
        outer_fold = int(row.get("outer_fold", row.get("fold_key", 0)))
        if outer_fold < 1:
            raise ValueError("legacy evidence record has an invalid outer fold")
        if scope != "full_outer_train":
            ignored += 1
            if scope not in {
                "candidate_consistency_inner_train",
                "candidate_selection_inner_fit",
            }:
                raise ValueError(f"unsupported legacy inner evidence scope: {scope!r}")
            inner_fold = int(row.get("inner_fold", 0))
            if inner_fold < 1 or int(row.get("fold_key", 0)) != outer_fold * 1000 + inner_fold:
                raise ValueError("legacy inner evidence has invalid fold provenance")
            seen_inner = inner_by_outer.setdefault(outer_fold, set())
            if inner_fold in seen_inner:
                raise ValueError("legacy handoff has duplicate exact-inner evidence")
            seen_inner.add(inner_fold)
            inner_sizes.setdefault(outer_fold, []).append(
                (int(row.get("n_rows", 0)), int(row.get("heldout_rows", 0)))
            )
            digest = _build_role_grouped_evidence_digest(
                importance=row.get("importance") or {},
                embedding_evidence=row.get("embedding_contrast_evidence") or {},
                htr_evidence=row.get("htr_evidence") or {},
            )
            digest, dropped = _sanitize_retained_legacy_digest(digest)
            dropped_diagnostic_fields += dropped
            inner_terms_by_outer.setdefault(outer_fold, {})[inner_fold] = _legacy_exact_inner_terms(
                digest
            )
            continue
        if int(row.get("fold_key", outer_fold)) != outer_fold:
            raise ValueError("legacy full-outer fold_key/outer_fold mismatch")
        if outer_fold in by_fold:
            raise ValueError(f"duplicate legacy full-outer evidence for fold {outer_fold}")
        digest = _build_role_grouped_evidence_digest(
            importance=row.get("importance") or {},
            embedding_evidence=row.get("embedding_contrast_evidence") or {},
            htr_evidence=row.get("htr_evidence") or {},
        )
        digest, dropped = _sanitize_retained_legacy_digest(digest)
        dropped_diagnostic_fields += dropped
        by_fold[outer_fold] = {
            "outer_fold": outer_fold,
            "scope": "full_outer_train",
            "n_rows": int(row.get("n_rows", 0)),
            "context": {"evidence_digest": json.loads(_canonical_json(digest))},
        }
    if not seen:
        raise ValueError(f"legacy handoff is empty: {requested}")
    if not by_fold:
        raise ValueError("legacy handoff contains no full_outer_train records")
    if set(inner_by_outer) != set(by_fold):
        raise ValueError("legacy exact-inner/full-outer fold sets do not match")
    inner_counts = {len(value) for value in inner_by_outer.values()}
    if len(inner_counts) != 1 or next(iter(inner_counts)) < 2:
        raise ValueError("legacy folds have incomplete or inconsistent exact-inner contexts")
    for outer_fold, sizes in inner_sizes.items():
        full_size = int(by_fold[outer_fold]["n_rows"])
        if any(fit <= 0 or heldout <= 0 or fit + heldout != full_size for fit, heldout in sizes):
            raise ValueError("legacy inner fit/heldout sizes do not partition outer training")
    recurrence_group_count = 0
    recurrence_term_count = 0
    for outer_fold in sorted(by_fold):
        recurrence = _build_exact_inner_recurrence(inner_terms_by_outer[outer_fold])
        recurrence_group_count += len(recurrence["groups"])
        recurrence_term_count += sum(
            int(group["retained_term_count"]) for group in recurrence["groups"]
        )
        row = dict(by_fold[outer_fold])
        context = dict(row["context"])
        context["exact_inner_recurrence"] = recurrence
        row["context"] = context
        by_fold[outer_fold] = row
    return LegacyFullOuterEvidence(
        rows_by_outer_fold=by_fold,
        artifact_path=str(requested),
        artifact_sha256=artifact_sha256,
        ignored_non_full_context_count=ignored,
        validated_inner_context_count=ignored,
        inner_contexts_per_outer=next(iter(inner_counts)),
        dropped_diagnostic_field_count=dropped_diagnostic_fields,
        exact_inner_recurrence_group_count=recurrence_group_count,
        exact_inner_recurrent_term_count=recurrence_term_count,
    )


def load_outer_splits_from_primary_predictions(
    path: Path | str,
    *,
    dataset_row_count: int,
) -> Mapping[int, tuple[int, ...]]:
    """Read only row/fold columns from a historical Stage-1 prediction file."""

    splits, _ = _load_outer_splits_from_primary_predictions_snapshot(
        path,
        dataset_row_count=dataset_row_count,
    )
    return splits


def _load_outer_splits_from_primary_predictions_snapshot(
    path: Path | str,
    *,
    dataset_row_count: int,
) -> tuple[Mapping[int, tuple[int, ...]], str]:
    """Project split columns from the same whole-file snapshot that is hashed."""

    requested = Path(path).resolve()
    try:
        import pyarrow.parquet as pq

        snapshot, artifact_sha256 = _read_path_snapshot(requested)
        columns = set(pq.ParquetFile(io.BytesIO(snapshot)).schema.names)
    except Exception as exc:
        raise ValueError("could not inspect primary prediction parquet schema") from exc
    if "_oci_row_id" not in columns:
        raise ValueError("primary predictions lack canonical row IDs")
    fold_columns = [name for name in ("outer_fold", "cv_fold") if name in columns]
    if not fold_columns:
        raise ValueError("primary predictions lack outer_fold/cv_fold")
    split_rows = pd.read_parquet(
        io.BytesIO(snapshot),
        columns=["_oci_row_id", *fold_columns],
    )
    if len(fold_columns) == 2 and not np.array_equal(
        split_rows["outer_fold"].to_numpy(), split_rows["cv_fold"].to_numpy()
    ):
        raise ValueError("primary prediction outer_fold/cv_fold disagree")
    fold_column = fold_columns[0]
    if len(split_rows) != int(dataset_row_count):
        raise ValueError("primary prediction split registry has the wrong row count")
    if split_rows["_oci_row_id"].duplicated().any():
        raise ValueError("primary prediction split registry has duplicate row IDs")
    expected_ids = set(range(int(dataset_row_count)))
    if set(map(int, split_rows["_oci_row_id"])) != expected_ids:
        raise ValueError("primary prediction split registry has invalid row IDs")
    by_fold: dict[int, tuple[int, ...]] = {}
    for fold, frame in split_rows.groupby(fold_column, sort=True):
        fold_id = int(fold)
        if fold_id < 1:
            raise ValueError("primary prediction split registry has an invalid fold")
        by_fold[fold_id] = tuple(map(int, frame["_oci_row_id"].tolist()))
    return by_fold, artifact_sha256


@dataclass(frozen=True)
class ResealedTfidfHandoff:
    rows_by_outer_fold: Mapping[int, tuple[Mapping[str, Any], ...]]
    full_rows_by_outer_fold: Mapping[int, Mapping[str, Any]]
    artifact_path: str
    artifact_sha256: str
    split_registry_content_hash: str
    structural_validation: Mapping[str, Any]


def load_resealed_tfidf_handoff(
    path: Path | str,
    *,
    dataset_row_count: int,
    require_registry_seal: bool = True,
) -> ResealedTfidfHandoff:
    """Validate split structure and registry seals without constructing clients."""

    requested = Path(path).resolve()
    snapshot, artifact_sha256 = _read_path_snapshot(requested)
    rows = _read_jsonl_snapshot(
        snapshot,
        source_path=requested,
        schema_version=HANDOFF_SCHEMA_VERSION,
    )
    valid_ids = set(range(int(dataset_row_count)))
    by_fold: dict[int, list[Mapping[str, Any]]] = {}
    seal_hashes: set[str] = set()
    identity_fields = (
        "dataset_content_fingerprint",
        "dataset_ordered_row_fingerprint",
        "split_semantics_hash",
    )
    for row in rows:
        outer_fold = int(row.get("outer_fold", 0))
        if outer_fold < 1:
            raise ValueError("TF-IDF handoff contains an invalid outer fold")
        fit_ids = [int(value) for value in row.get("fit_row_ids", [])]
        heldout_ids = [int(value) for value in row.get("heldout_row_ids", [])]
        if not fit_ids or not heldout_ids:
            raise ValueError("TF-IDF context must have non-empty fit and heldout rows")
        if len(fit_ids) != len(set(fit_ids)) or len(heldout_ids) != len(set(heldout_ids)):
            raise ValueError("TF-IDF context contains duplicate row IDs")
        if set(fit_ids) & set(heldout_ids) or not (set(fit_ids) | set(heldout_ids)) <= valid_ids:
            raise ValueError("TF-IDF context has overlapping or out-of-range row IDs")
        for name, values in (("fit", fit_ids), ("heldout", heldout_ids)):
            expected = row_set_fingerprint(values)
            discovery = row.get("discovery") or {}
            if row.get(f"{name}_row_fingerprint") != expected:
                raise ValueError(f"TF-IDF {name} row fingerprint mismatch")
            if discovery.get(f"{name}_row_fingerprint") != expected:
                raise ValueError(f"TF-IDF discovery {name} row fingerprint mismatch")
            discovery_ids = [int(value) for value in discovery.get(f"{name}_row_ids", [])]
            if discovery_ids != values:
                raise ValueError(f"TF-IDF discovery {name} row ordering mismatch")
        if require_registry_seal:
            registry_hash = str(row.get("split_registry_content_hash") or "")
            if not _SHA256.fullmatch(registry_hash):
                raise ValueError("TF-IDF context is not sealed to a split registry")
            seal_hashes.add(registry_hash)
            discovery = row.get("discovery") or {}
            for field_name in identity_fields:
                value = str(row.get(field_name) or "")
                if not _SHA256.fullmatch(value) or discovery.get(field_name) != value:
                    raise ValueError(f"invalid resealed TF-IDF identity field {field_name}")
        by_fold.setdefault(outer_fold, []).append(row)
    if require_registry_seal and len(seal_hashes) != 1:
        raise ValueError("TF-IDF contexts do not share one split-registry seal")

    outer_test_counts: dict[int, int] = {}
    full_by_fold: dict[int, Mapping[str, Any]] = {}
    inner_counts: set[int] = set()
    recurrence_group_count = 0
    recurrence_term_count = 0
    for outer_fold, fold_rows in sorted(by_fold.items()):
        full = [row for row in fold_rows if row.get("scope") == "full_outer_train"]
        inner = [row for row in fold_rows if row.get("scope") == "candidate_selection_inner_fit"]
        if len(full) != 1 or len(inner) < 2:
            raise ValueError(
                f"TF-IDF fold {outer_fold} requires one full and at least two inner contexts"
            )
        inner_counts.add(len(inner))
        full_row = full[0]
        outer_fit = set(map(int, full_row["fit_row_ids"]))
        outer_heldout = set(map(int, full_row["heldout_row_ids"]))
        if outer_fit | outer_heldout != valid_ids:
            raise ValueError(f"TF-IDF outer fold {outer_fold} does not partition the dataset")
        for row_id in outer_heldout:
            outer_test_counts[row_id] = outer_test_counts.get(row_id, 0) + 1
        seen_inner: set[int] = set()
        inner_heldout_counts: dict[int, int] = {}
        inner_terms: dict[int, set[tuple[str, str, str]]] = {}
        for row in inner:
            inner_fold = int(row.get("inner_fold", 0))
            if inner_fold < 1 or inner_fold in seen_inner:
                raise ValueError(f"TF-IDF fold {outer_fold} has duplicate/invalid inner folds")
            seen_inner.add(inner_fold)
            inner_discovery = row.get("discovery")
            if not isinstance(inner_discovery, Mapping):
                raise ValueError("TF-IDF inner context is missing discovery evidence")
            inner_terms[inner_fold] = _tfidf_exact_inner_terms(inner_discovery)
            inner_fit = set(map(int, row["fit_row_ids"]))
            inner_heldout = set(map(int, row["heldout_row_ids"]))
            if inner_fit | inner_heldout != outer_fit:
                raise ValueError(f"TF-IDF inner context does not partition outer fold {outer_fold}")
            for row_id in inner_heldout:
                inner_heldout_counts[row_id] = inner_heldout_counts.get(row_id, 0) + 1
        if set(inner_heldout_counts) != outer_fit or set(inner_heldout_counts.values()) != {1}:
            raise ValueError(f"TF-IDF inner heldouts do not partition outer fold {outer_fold}")
        compact_score = (full_row.get("discovery") or {}).get("topic_score_tests") or {}
        score_artifact = ((full_row.get("discovery") or {}).get("artifacts") or {}).get(
            "topic_score_tests"
        )
        if (
            score_artifact is not None
            or compact_score.get("status") != "not_run"
            or bool(compact_score.get("uses_heldout_treatment_and_outcome"))
        ):
            raise ValueError("full-outer TF-IDF context exposes outer-heldout score labels")
        recurrence = _build_exact_inner_recurrence(inner_terms)
        recurrence_group_count += len(recurrence["groups"])
        recurrence_term_count += sum(
            int(group["retained_term_count"]) for group in recurrence["groups"]
        )
        enriched_full = dict(full_row)
        enriched_discovery = dict(full_row.get("discovery") or {})
        enriched_discovery["exact_inner_recurrence"] = recurrence
        enriched_full["discovery"] = enriched_discovery
        full_by_fold[outer_fold] = enriched_full
    if len(inner_counts) != 1:
        raise ValueError("TF-IDF folds have inconsistent inner-context counts")
    if set(outer_test_counts) != valid_ids or set(outer_test_counts.values()) != {1}:
        raise ValueError("TF-IDF outer heldouts do not form a once-only dataset partition")
    return ResealedTfidfHandoff(
        rows_by_outer_fold={key: tuple(value) for key, value in by_fold.items()},
        full_rows_by_outer_fold=full_by_fold,
        artifact_path=str(requested),
        artifact_sha256=artifact_sha256,
        split_registry_content_hash=next(iter(seal_hashes), "unsealed"),
        structural_validation={
            "status": "passed",
            "dataset_row_count": int(dataset_row_count),
            "outer_fold_count": len(by_fold),
            "inner_contexts_per_outer": next(iter(inner_counts)),
            "outer_test_rows_predicted_once": True,
            "registry_sealed": bool(require_registry_seal),
            "exact_inner_recurrence_schema_version": EXACT_INNER_RECURRENCE_VERSION,
            "exact_inner_recurrence_group_count": recurrence_group_count,
            "exact_inner_recurrent_term_count": recurrence_term_count,
            "latent_topic_ids_compared_across_folds": False,
        },
    )


@dataclass(frozen=True)
class QueryEvidenceArtifact:
    """Authenticated fold-scoped learned neural query-moment evidence."""

    path: Path | str
    outer_fold: int
    artifact_sha256: str
    fit_row_fingerprint: str
    heldout_row_fingerprint: str
    scope: str = "outer_train"


@dataclass(frozen=True)
class TfidfOrphanNgramArtifact:
    """Trusted per-fold override for a nonportable handoff reference."""

    path: Path | str
    artifact_sha256: str | None = None


def _load_query_evidence(
    source: QueryEvidenceArtifact,
    *,
    provenance: FoldEvidenceProvenance,
    config: QueryMomentEvidenceAdapterConfig,
):
    outer_fold = int(provenance.outer_fold)
    if int(source.outer_fold) != outer_fold:
        raise ValueError("neural query evidence outer fold does not match its registry key")
    if str(source.scope) != "outer_train":
        raise ValueError("neural query evidence registration must have outer_train scope")
    expected_fit_fingerprint = row_set_fingerprint(provenance.train_row_ids)
    expected_heldout_fingerprint = row_set_fingerprint(provenance.heldout_row_ids)
    if source.fit_row_fingerprint != expected_fit_fingerprint:
        raise ValueError(f"neural query evidence fit fingerprint mismatch for fold {outer_fold}")
    if source.heldout_row_fingerprint != expected_heldout_fingerprint:
        raise ValueError(
            f"neural query evidence heldout fingerprint mismatch for fold {outer_fold}"
        )
    adapted = load_query_moment_evidence_artifact(
        source.path,
        provenance=provenance,
        expected_sha256=source.artifact_sha256,
        registered_fit_row_ids=provenance.train_row_ids,
        registered_heldout_row_ids=provenance.heldout_row_ids,
        config=config,
    )
    audit = adapted.audit
    audit["registration_outer_fold"] = outer_fold
    audit["registration_scope"] = "outer_train"
    audit["registration_fit_row_fingerprint"] = source.fit_row_fingerprint
    audit["registration_heldout_row_fingerprint"] = source.heldout_row_fingerprint
    audit["registered_sha256_verified"] = True
    return adapted, audit


def _normalize_query_evidence_artifact_registration(
    value: QueryEvidenceArtifact | Mapping[str, Any],
) -> QueryEvidenceArtifact:
    if isinstance(value, QueryEvidenceArtifact):
        fields = asdict(value)
    elif isinstance(value, Mapping):
        fields = dict(value)
    else:
        raise TypeError("neural query evidence registration must be an artifact object")
    path = fields.get("path") or fields.get("artifact_path")
    if not str(path or "").strip():
        raise ValueError("neural query evidence registration requires a path")
    try:
        outer_fold = int(fields.get("outer_fold"))
    except (TypeError, ValueError) as exc:
        raise ValueError("neural query evidence outer_fold must be a positive integer") from exc
    if outer_fold < 1:
        raise ValueError("neural query evidence outer_fold must be a positive integer")
    digest = str(fields.get("artifact_sha256") or fields.get("sha256") or "").lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError("neural query evidence registration requires a valid SHA-256")
    fit_fingerprint = str(fields.get("fit_row_fingerprint") or "").lower()
    heldout_fingerprint = str(fields.get("heldout_row_fingerprint") or "").lower()
    if not _SHA256.fullmatch(fit_fingerprint) or not _SHA256.fullmatch(heldout_fingerprint):
        raise ValueError("neural query evidence registration requires valid row fingerprints")
    scope = str(fields.get("scope") or "outer_train").strip().lower()
    if scope != "outer_train":
        raise ValueError("neural query evidence registration scope must be outer_train")
    return QueryEvidenceArtifact(
        path=Path(str(path)),
        outer_fold=outer_fold,
        artifact_sha256=digest,
        fit_row_fingerprint=fit_fingerprint,
        heldout_row_fingerprint=heldout_fingerprint,
        scope=scope,
    )


def _normalize_orphan_artifact_registration(
    value: TfidfOrphanNgramArtifact | Mapping[str, Any] | Path | str,
) -> TfidfOrphanNgramArtifact:
    if isinstance(value, TfidfOrphanNgramArtifact):
        path = value.path
        digest = value.artifact_sha256
    elif isinstance(value, Mapping):
        path = value.get("path") or value.get("artifact_path")
        digest = value.get("artifact_sha256") or value.get("sha256") or value.get("content_sha256")
    else:
        path = value
        digest = None
    if not str(path or "").strip():
        raise ValueError("TF-IDF orphan artifact registration requires a path")
    normalized_digest = None if digest is None else str(digest).strip().lower()
    if normalized_digest is not None and not _SHA256.fullmatch(normalized_digest):
        raise ValueError("TF-IDF orphan artifact registration has an invalid SHA-256")
    return TfidfOrphanNgramArtifact(
        path=Path(str(path)),
        artifact_sha256=normalized_digest,
    )


def _effect_ngram_registration(full_row: Mapping[str, Any]) -> Any:
    discovery = full_row.get("discovery")
    if not isinstance(discovery, Mapping):
        return None
    artifacts = discovery.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    ngram_scores = artifacts.get("ngram_scores")
    if not isinstance(ngram_scores, Mapping):
        return None
    return ngram_scores.get("effect")


def _registered_reference_path_and_hash(value: Any) -> tuple[str | None, str | None]:
    if isinstance(value, Mapping):
        forbidden = [str(key) for key in value if _FORBIDDEN_NAME.search(str(key))]
        if forbidden:
            raise ValueError(
                f"TF-IDF handoff effect registration contains oracle/true fields: {forbidden}"
            )
        path = next(
            (
                value.get(key)
                for key in ("path", "artifact_path", "file", "uri")
                if value.get(key) not in (None, "")
            ),
            None,
        )
        digest = next(
            (
                value.get(key)
                for key in ("sha256", "artifact_sha256", "content_sha256")
                if value.get(key) not in (None, "")
            ),
            None,
        )
    else:
        path = value
        digest = None
    normalized_path = str(path).strip() if path not in (None, "") else None
    normalized_digest = str(digest).strip().lower() if digest not in (None, "") else None
    if normalized_digest is not None and not _SHA256.fullmatch(normalized_digest):
        raise ValueError("TF-IDF handoff effect artifact has an invalid SHA-256")
    return normalized_path, normalized_digest


def load_candidate_pool(
    path: Path | str,
    *,
    expected_outer_fold: int,
) -> tuple[list[CandidateContract], dict[str, Any]]:
    requested = Path(path).resolve()
    snapshot, artifact_sha256 = _read_path_snapshot(requested)
    try:
        payload = json.loads(snapshot.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("candidate pool is not valid UTF-8") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("candidate pool must be a JSON object")
    if int(payload.get("outer_fold", 0)) != int(expected_outer_fold):
        raise ValueError("candidate pool outer fold does not match its registry key")
    raw_candidates = payload.get("valid_proposals")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate pool has no valid_proposals")
    allowed = {"name", "type", "categories", "roles", "description", "value_aliases"}
    contracts: list[CandidateContract] = []
    seen_hashes: set[str] = set()
    for position, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise ValueError(f"candidate pool entry {position} is not an object")
        spec = {key: raw[key] for key in allowed if key in raw}
        families = raw.get("source_families") or []
        contract = CandidateContract(spec, source_families=families)
        digest = extraction_contract_sha256(contract.extraction_spec)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        contracts.append(contract)
    return contracts, {
        "path": str(requested),
        "sha256": artifact_sha256,
        "candidate_count": len(contracts),
    }


@dataclass
class FoldTrainExplicitEncoder:
    """Deterministic explicit encoding with no learned preprocessing.

    Continuous missing values use the fixed value zero and retain a separate
    missingness indicator. The production causal forest consumes this fixed
    encoding directly; the degraded structured-interaction fallback performs
    its own split-local centering and scaling. The train summaries retained
    here are audit diagnostics only and are never applied to model inputs.
    """

    feature_names_: list[str] = field(default_factory=list, init=False)
    means_: np.ndarray | None = field(default=None, init=False, repr=False)
    scales_: np.ndarray | None = field(default=None, init=False, repr=False)
    specs_: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    feature_spec_names_: list[str] = field(default_factory=list, init=False)

    def _raw_matrix(self, frame: pd.DataFrame, *, fitting: bool) -> np.ndarray:
        columns: list[np.ndarray] = []
        names: list[str] = []
        feature_spec_names: list[str] = []
        for spec in self.specs_:
            name = str(spec["name"])
            value_column, missing_column = expected_extraction_columns(spec)
            missing = {value_column, missing_column} - set(frame.columns)
            if missing:
                raise ValueError(f"explicit extraction is missing columns: {sorted(missing)}")
            declared_missing = frame[missing_column].fillna(True).astype(bool).to_numpy()
            if spec["type"] == "continuous":
                numeric = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
                actual_missing = declared_missing | ~np.isfinite(numeric)
                columns.extend(
                    [np.where(actual_missing, 0.0, numeric), actual_missing.astype(float)]
                )
                names.extend([f"{name}__value", f"{name}__missing"])
                feature_spec_names.extend([name, name])
            else:
                values = frame[value_column].fillna("").astype(str).to_numpy()
                categories = [str(value) for value in spec.get("categories") or []]
                for category in categories:
                    columns.append(((values == category) & ~declared_missing).astype(float))
                    names.append(f"{name}__category__{category}")
                    feature_spec_names.append(name)
                unknown = (~declared_missing) & ~np.isin(values, categories)
                columns.extend([unknown.astype(float), declared_missing.astype(float)])
                names.extend([f"{name}__unknown", f"{name}__missing"])
                feature_spec_names.extend([name, name])
        if not columns:
            raise ValueError("at least one explicit feature contract is required")
        if fitting:
            self.feature_names_ = names
            self.feature_spec_names_ = feature_spec_names
        elif names != self.feature_names_:
            raise RuntimeError("explicit encoding layout changed after fit")
        return np.column_stack(columns).astype(float)

    def fit(
        self, frame: pd.DataFrame, specs: Sequence[Mapping[str, Any]]
    ) -> "FoldTrainExplicitEncoder":
        self.specs_ = [json.loads(_canonical_json(dict(spec))) for spec in specs]
        raw = self._raw_matrix(frame, fitting=True)
        self.means_ = raw.mean(axis=0)
        scales = raw.std(axis=0, ddof=0)
        self.scales_ = np.where(scales > 1e-12, scales, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.means_ is None or self.scales_ is None:
            raise RuntimeError("FoldTrainExplicitEncoder must be fit before transform")
        raw = self._raw_matrix(frame, fitting=False)
        if not np.isfinite(raw).all():
            raise ValueError("explicit matrix contains non-finite values")
        return raw

    def state_dict(self) -> dict[str, Any]:
        if self.means_ is None or self.scales_ is None:
            raise RuntimeError("FoldTrainExplicitEncoder has not been fit")
        return {
            "feature_names": list(self.feature_names_),
            "feature_spec_names": list(self.feature_spec_names_),
            "train_means": self.means_.tolist(),
            "train_scales": self.scales_.tolist(),
            "model_input_preprocessing": "fixed_zero_imputation_only",
            "train_summaries_used_by_model": False,
            "continuous_missing_fill_value": 0.0,
        }


def _seal_final_forest_explicit_block(
    package: AuthenticatedFinalContextFitUpstreamBank,
    *,
    encoder: FoldTrainExplicitEncoder,
    specs: Sequence[Mapping[str, Any]],
    train_values: np.ndarray,
    heldout_values: np.ndarray,
) -> tuple[SealedFinalForestExplicitBlock, Mapping[str, Any]]:
    """Route encoded contracts to forest X/W without collapsing dual roles."""

    if len(encoder.feature_names_) != len(encoder.feature_spec_names_):
        raise RuntimeError("explicit encoder name-to-contract routing is incomplete")
    train_matrix = np.asarray(train_values, dtype=float)
    heldout_matrix = np.asarray(heldout_values, dtype=float)
    expected_width = len(encoder.feature_names_)
    expected_train_rows = len(package.calibrated_sources.train_row_ids)
    expected_heldout_rows = len(package.calibrated_sources.heldout_row_ids)
    if train_matrix.shape != (expected_train_rows, expected_width):
        raise ValueError("explicit outer-train matrix does not match the final package")
    if heldout_matrix.shape != (expected_heldout_rows, expected_width):
        raise ValueError("explicit outer-heldout matrix does not match the final package")

    roles_by_contract: dict[str, frozenset[str]] = {}
    for raw_spec in specs:
        spec = CandidateContract(raw_spec).extraction_spec
        name = str(spec["name"])
        if name in roles_by_contract:
            raise ValueError("final explicit registry contains duplicate contract names")
        roles_by_contract[name] = frozenset(str(role) for role in spec.get("roles") or ())
    unknown_contracts = set(encoder.feature_spec_names_) - set(roles_by_contract)
    if unknown_contracts:
        raise ValueError(
            "explicit encoder contains columns outside the frozen contract registry: "
            f"{sorted(unknown_contracts)}"
        )

    effect_indices = tuple(
        index
        for index, contract_name in enumerate(encoder.feature_spec_names_)
        if "effect_modifier" in roles_by_contract[contract_name]
    )
    control_indices = tuple(
        index
        for index, contract_name in enumerate(encoder.feature_spec_names_)
        if "confounder" in roles_by_contract[contract_name]
    )
    effect_names = tuple(encoder.feature_names_[index] for index in effect_indices)
    control_names = tuple(encoder.feature_names_[index] for index in control_indices)
    block = SealedFinalForestExplicitBlock.seal_for_package(
        package,
        effect_names=effect_names,
        control_names=control_names,
        effect_train_values=train_matrix[:, effect_indices],
        effect_heldout_values=heldout_matrix[:, effect_indices],
        control_train_values=train_matrix[:, control_indices],
        control_heldout_values=heldout_matrix[:, control_indices],
    )
    dual_role_contracts = sorted(
        name
        for name, roles in roles_by_contract.items()
        if {"confounder", "effect_modifier"}.issubset(roles)
    )
    audit = {
        "schema_version": "final_forest_explicit_role_routing_v1",
        "content_sha256": block.content_sha256,
        "encoded_column_count": expected_width,
        "effect_modifier_column_count": len(effect_indices),
        "confounder_control_column_count": len(control_indices),
        "dual_role_contract_count": len(dual_role_contracts),
        "dual_role_contract_names": dual_role_contracts,
        "dual_role_columns_copied_to_both_x_and_w": True,
        "effect_modifier_encoded_column_names": list(effect_names),
        "confounder_encoded_column_names": list(control_names),
        "role_source": "frozen_extraction_contract_registry",
        "outer_heldout_labels_used": False,
        "row_level_values_persisted": False,
    }
    return block, audit


def _reconstruct_forest_potential_outcomes(
    forest_tau: Sequence[float],
    *,
    exact_nuisance: SealedExactNuisanceBankExtension,
    outcome_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Recover auditable potential outcomes without fitting an S-learner."""

    exact_nuisance.verify_authenticated_content()
    tau = np.asarray(forest_tau, dtype=float)
    values = np.asarray(exact_nuisance.outer_heldout_values, dtype=float)
    if tau.ndim != 1 or len(tau) != len(values) or not np.isfinite(tau).all():
        raise ValueError("forest tau must be one finite value per outer-heldout row")
    raw_tau = np.array(tau, dtype=float, copy=True)
    raw_tau_sha256 = _content_sha256(
        {
            "dtype": "float64_hex",
            "length": len(raw_tau),
            "values": [float(value).hex() for value in raw_tau],
        }
    )
    propensity_indices = tuple(
        index
        for index, semantic in enumerate(exact_nuisance.prediction_semantics)
        if semantic == EXACT_PROPENSITY_PREDICTION
    )
    outcome_indices = tuple(
        index
        for index, semantic in enumerate(exact_nuisance.prediction_semantics)
        if semantic == EXACT_OUTCOME_PREDICTION
    )
    if not propensity_indices or not outcome_indices:
        raise ValueError("exact nuisance bridge supplied no propensity or outcome prediction")
    propensity = np.mean(values[:, propensity_indices], axis=1)
    outcome = np.mean(values[:, outcome_indices], axis=1)
    normalized_outcome_type = str(outcome_type).strip().lower()
    clipped_tau_count = 0
    projected_p0_count = 0
    if normalized_outcome_type == "binary":
        if (
            np.any(propensity < 0.0)
            or np.any(propensity > 1.0)
            or np.any(outcome < 0.0)
            or np.any(outcome > 1.0)
        ):
            raise ValueError("binary exact nuisance predictions must be probabilities")
        clipped_tau = np.clip(tau, -1.0, 1.0)
        clipped_tau_count = int(np.count_nonzero(clipped_tau != tau))
        tau = clipped_tau
        unprojected_p0 = outcome - propensity * tau
        lower = np.maximum(0.0, -tau)
        upper = np.minimum(1.0, 1.0 - tau)
        p0 = np.clip(unprojected_p0, lower, upper)
        projected_p0_count = int(np.count_nonzero(p0 != unprojected_p0))
        p1 = p0 + tau
        if np.any(p0 < -1e-12) or np.any(p0 > 1.0 + 1e-12):
            raise RuntimeError("binary p0 feasibility projection failed")
        if np.any(p1 < -1e-12) or np.any(p1 > 1.0 + 1e-12):
            raise RuntimeError("binary p1 feasibility projection failed")
        policy = (
            "tau_clipped_to_unit_interval_then_p0_projected_to_" "max_0_minus_tau_min_1_1_minus_tau"
        )
    elif normalized_outcome_type == "continuous":
        p0 = outcome - propensity * tau
        p1 = p0 + tau
        policy = "unclipped_algebra_p0_equals_m_minus_e_tau_p1_equals_p0_plus_tau"
    else:
        raise ValueError("outcome_type must be 'binary' or 'continuous'")
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise RuntimeError("reconstructed potential outcomes are non-finite")
    final_tau_sha256 = _content_sha256(
        {
            "dtype": "float64_hex",
            "length": len(tau),
            "values": [float(value).hex() for value in tau],
        }
    )
    audit = {
        "schema_version": FINAL_FOREST_POTENTIAL_OUTCOME_POLICY_VERSION,
        "outcome_type": normalized_outcome_type,
        "policy": policy,
        "propensity_prediction_column_count": len(propensity_indices),
        "outcome_prediction_column_count": len(outcome_indices),
        "propensity_ensemble_reduction": "equal_mean",
        "outcome_ensemble_reduction": "equal_mean",
        "raw_sealed_forest_tau_values_sha256": raw_tau_sha256,
        "final_prediction_estimand_values_sha256": final_tau_sha256,
        "forest_tau_clip_count": clipped_tau_count,
        "forest_tau_clipping_changed_estimand": bool(clipped_tau_count),
        "final_prediction_estimand": (
            "minus_one_to_one_clipped_forest_tau"
            if normalized_outcome_type == "binary"
            else "unmodified_forest_tau"
        ),
        "final_estimand_equals_unmodified_sealed_forest_tau": bool(clipped_tau_count == 0),
        "p0_feasibility_projection_count": projected_p0_count,
        "s_learner_fit": False,
        "outer_heldout_labels_used": False,
        "row_level_values_persisted": False,
    }
    return p0, p1, tau, audit


class ReviewGateSourceProvider(Protocol):
    """Return calibrated effect signals honest to one complete review gate."""

    def get_gate_source_view(
        self,
        *,
        outer_fold: int,
        exact_gate_row_ids: tuple[int, ...],
    ) -> GateSourceSignalView: ...

    def identity(self) -> Mapping[str, Any]: ...


class ReviewGateFeatureBankProvider(Protocol):
    """Return role-aware raw feature banks honest to one complete review gate."""

    def get_gate_feature_bank_view(
        self,
        *,
        outer_fold: int,
        exact_gate_row_ids: tuple[int, ...],
    ) -> GateFeatureBankView: ...

    def identity(self) -> Mapping[str, Any]: ...


class GateOnlyReviewNumericalProvider(Protocol):
    """Open prefit cumulative transforms for diagnostics on one exact gate."""

    def get_gate_only_view(
        self,
        *,
        outer_fold: int,
        context_epoch: int,
        exact_spent_row_ids: tuple[int, ...],
        exact_gate_row_ids: tuple[int, ...],
    ) -> Any: ...

    def identity(self) -> Mapping[str, Any]: ...


class BindableReviewGateSourceProvider(Protocol):
    """Fit calibrated sources on spent labels and a label-free exact gate."""

    def bind_fold(
        self,
        *,
        outer_fold: int,
        context: ObservableCausalRows,
        context_texts: Sequence[str],
        gate_texts: Sequence[str],
        exact_gate_row_ids: tuple[int, ...],
    ) -> ReviewGateSourceProvider: ...

    def identity(self) -> Mapping[str, Any]: ...


class BindableReviewGateFeatureBankProvider(Protocol):
    """Fit on spent labels while receiving only label-free gate IDs/text."""

    def bind_fold(
        self,
        *,
        outer_fold: int,
        context: ObservableCausalRows,
        context_texts: Sequence[str],
        gate_texts: Sequence[str],
        exact_gate_row_ids: tuple[int, ...],
    ) -> ReviewGateFeatureBankProvider: ...

    def identity(self) -> Mapping[str, Any]: ...


class ReviewPartitionProvider(Protocol):
    """Return authenticated exact outer-train assignments for review gates."""

    def get_review_partition_assignments(
        self,
        *,
        outer_fold: int,
        exact_outer_train_row_ids: tuple[int, ...],
    ) -> Mapping[int, Sequence[int]]: ...

    def identity(self) -> Mapping[str, Any]: ...


class ReviewSpentEvidenceProvider(Protocol):
    """Build discovery evidence using spent rows while sealed rows stay data-free.

    The runner supplies labels and text only for ``exact_spent_row_ids``.  The
    provider receives future row IDs solely so its returned provenance can
    prove that every still-adaptive review partition was held out.  The legacy
    provider keyword ``review_round`` is bound to the spent-context epoch: the
    number of review gates already consumed before this context fit.  It is
    deliberately distinct from the reasoning agent's consumer review round.
    """

    def get_spent_evidence_inputs(
        self,
        *,
        outer_fold: int,
        review_round: int,
        exact_spent_row_ids: tuple[int, ...],
        exact_sealed_row_ids: tuple[int, ...],
        spent_texts: tuple[str, ...],
        spent_treatment: np.ndarray,
        spent_outcome: np.ndarray,
    ) -> Sequence[FoldEvidenceInput]: ...

    def identity(self) -> Mapping[str, Any]: ...


class FinalUpstreamProducer(Protocol):
    """Post-registry producer with no outer-heldout label input channel."""

    def identity(self) -> Mapping[str, Any]: ...

    def produce(
        self,
        *,
        outer_fold: int,
        outer_train_row_ids: Sequence[int],
        outer_train_texts: Sequence[str],
        outer_train_treatment: Sequence[float],
        outer_train_outcome: Sequence[float],
        outer_heldout_row_ids: Sequence[int],
        outer_heldout_texts: Sequence[str],
        meta_inner_fold_ids: Sequence[int],
    ) -> AuthenticatedFinalContextFitUpstreamBank: ...


@dataclass(frozen=True)
class AllEvidenceFusionRunnerConfig:
    text_column: str = "text"
    treatment_column: str = "treatment"
    outcome_column: str = "outcome"
    outcome_type: str = "binary"
    max_candidates: int = 16
    regularization_grid: tuple[float, ...] = (
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
    )
    interaction_inner_folds: int = 3
    interact_all_features: bool = True
    random_state: int = 42
    fusion_model_identity: str = "unspecified_remote_model"
    fusion_enable_thinking: bool = True
    fusion_max_tokens: int = 25000
    fusion_thinking_token_budget: int | None = None
    extraction_model_identity: str = "unspecified_remote_model"
    remote_endpoint_pool_identity: str = "unspecified_remote_endpoint_pool"
    extraction_prompt_template_version: str = (
        "explicit_features_v5+source_text_temporally_valid_by_design_v1"
    )
    extraction_enable_thinking: bool = False
    extraction_grouping_strategy: str = "clinical_domain"
    extraction_grouping_version: str = "explicit_feature_request_grouping_v2"
    extraction_context_strategy: str = "tail"
    extraction_context_compactor_version: str = "contract_lexical_rag_v1"
    extraction_max_text_length: int | None = None
    extraction_batch_size: int = 32
    max_variables_per_extraction_request: int = 10
    post_extraction_review_rounds: int = DEFAULT_POST_EXTRACTION_REVIEW_ROUNDS
    post_extraction_review_max_operations: int = 4
    post_extraction_review_max_quality_retries: int = 2
    post_extraction_review_min_partition_rows: int = 8
    post_extraction_review_config: CausalReviewConfig = field(default_factory=CausalReviewConfig)
    post_extraction_scientific_policy: PostExtractionScientificPolicy | None = None
    upstream_review_policy: str = CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY
    require_review_source_signals: bool = False
    require_review_feature_banks: bool = False
    require_final_upstream_inputs: bool = False
    require_final_upstream_neural_query_inputs: bool = False
    require_final_causal_forest: bool = False
    allow_degraded_review_without_all_upstream: bool = False
    final_upstream_meta_inner_folds: int = 3
    final_upstream_head_regularization: float = 1.0
    require_registry_seal: bool = True
    include_tfidf_orphan_ngrams: bool = False
    require_tfidf_orphan_ngrams: bool = False
    orphan_ngram_adapter: OrphanNgramEvidenceAdapterConfig | None = None
    derive_sparse_query_moments_when_missing: bool = False
    require_neural_query_moments: bool = False
    neural_query_moment_artifacts_by_fold: Mapping[
        int,
        QueryEvidenceArtifact | Mapping[str, Any],
    ] = field(default_factory=dict)
    query_moment_adapter: QueryMomentEvidenceAdapterConfig = field(
        default_factory=QueryMomentEvidenceAdapterConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.include_tfidf_orphan_ngrams, bool):
            raise ValueError("include_tfidf_orphan_ngrams must be a boolean")
        if not isinstance(self.require_tfidf_orphan_ngrams, bool):
            raise ValueError("require_tfidf_orphan_ngrams must be a boolean")
        if self.require_tfidf_orphan_ngrams and not self.include_tfidf_orphan_ngrams:
            raise ValueError(
                "require_tfidf_orphan_ngrams requires include_tfidf_orphan_ngrams"
            )
        if self.include_tfidf_orphan_ngrams:
            if not isinstance(
                self.orphan_ngram_adapter,
                OrphanNgramEvidenceAdapterConfig,
            ):
                raise ValueError(
                    "enabled TF-IDF orphan adaptation requires an explicit "
                    "OrphanNgramEvidenceAdapterConfig"
                )
            self.orphan_ngram_adapter.validate()
        elif self.orphan_ngram_adapter is not None:
            raise ValueError(
                "orphan_ngram_adapter must be null when TF-IDF orphan adaptation "
                "is disabled"
            )
        if not isinstance(self.fusion_enable_thinking, bool):
            raise ValueError("fusion_enable_thinking must be a boolean")
        maximum = self.fusion_max_tokens
        if (
            isinstance(maximum, (bool, np.bool_))
            or not isinstance(maximum, (int, np.integer))
            or int(maximum) <= 0
        ):
            raise ValueError("fusion_max_tokens must be a positive integer")
        object.__setattr__(self, "fusion_max_tokens", int(maximum))
        budget = self.fusion_thinking_token_budget
        if budget is not None:
            if (
                isinstance(budget, (bool, np.bool_))
                or not isinstance(budget, (int, np.integer))
                or int(budget) <= 0
            ):
                raise ValueError("fusion_thinking_token_budget must be a positive integer or None")
            object.__setattr__(self, "fusion_thinking_token_budget", int(budget))
            if int(budget) >= int(maximum):
                raise ValueError(
                    "fusion_thinking_token_budget must be strictly less than fusion_max_tokens"
                )
        if not isinstance(self.derive_sparse_query_moments_when_missing, bool):
            raise ValueError("derive_sparse_query_moments_when_missing must be a boolean")
        review_integers = {
            "post_extraction_review_rounds": (
                self.post_extraction_review_rounds,
                0,
                8,
            ),
            "post_extraction_review_max_operations": (
                self.post_extraction_review_max_operations,
                1,
                32,
            ),
            "post_extraction_review_max_quality_retries": (
                self.post_extraction_review_max_quality_retries,
                0,
                8,
            ),
            "post_extraction_review_min_partition_rows": (
                self.post_extraction_review_min_partition_rows,
                2,
                None,
            ),
        }
        for name, (value, minimum, maximum) in review_integers.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            normalized = int(value)
            if normalized < minimum or (maximum is not None and normalized > maximum):
                rendered = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
                raise ValueError(f"{name} must be {rendered}")
            object.__setattr__(self, name, normalized)
        if not isinstance(self.post_extraction_review_config, CausalReviewConfig):
            raise TypeError("post_extraction_review_config must be CausalReviewConfig")
        if self.post_extraction_scientific_policy is not None:
            if not isinstance(
                self.post_extraction_scientific_policy,
                PostExtractionScientificPolicy,
            ):
                raise TypeError(
                    "post_extraction_scientific_policy must be "
                    "PostExtractionScientificPolicy"
                )
            if (
                self.post_extraction_review_config.estimator_policy
                != self.post_extraction_scientific_policy.review_estimator
            ):
                raise ValueError(
                    "post-extraction causal-review estimator policy differs "
                    "from the configured scientific policy"
                )
        if self.upstream_review_policy not in {
            CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY,
            GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
        }:
            raise ValueError("upstream_review_policy is not a registered review policy")
        for name in (
            "require_review_source_signals",
            "require_review_feature_banks",
            "require_final_upstream_inputs",
            "require_final_upstream_neural_query_inputs",
            "require_final_causal_forest",
            "allow_degraded_review_without_all_upstream",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        # Positive adaptive review is strict by default.  The only supported
        # non-strict path is the existing explicit degraded research/test mode;
        # the v24 benchmark CLI never enables it.
        if (
            self.post_extraction_review_rounds > 0
            and not self.allow_degraded_review_without_all_upstream
        ):
            for name in (
                "require_review_source_signals",
                "require_review_feature_banks",
                "require_final_upstream_inputs",
                "require_final_upstream_neural_query_inputs",
                "require_final_causal_forest",
            ):
                object.__setattr__(self, name, True)
        if self.post_extraction_review_rounds == 0 and (
            self.require_review_source_signals or self.require_review_feature_banks
        ):
            raise ValueError(
                "required review signal providers need post_extraction_review_rounds > 0"
            )
        if (
            self.post_extraction_review_rounds == 0
            and self.allow_degraded_review_without_all_upstream
        ):
            raise ValueError("allow_degraded_review_without_all_upstream requires review rounds")
        if self.post_extraction_review_rounds > 0 and not (
            self.allow_degraded_review_without_all_upstream
            or (
                self.require_review_source_signals
                and self.require_review_feature_banks
                and self.require_final_upstream_inputs
                and self.require_final_upstream_neural_query_inputs
            )
        ):
            raise ValueError(
                "adaptive all-evidence review requires calibrated sources, role-aware "
                "feature banks, final direct upstream inputs, and exact neural-query "
                "inputs; degraded research/test mode must be explicitly opted into"
            )
        meta_folds = self.final_upstream_meta_inner_folds
        if (
            isinstance(meta_folds, (bool, np.bool_))
            or not isinstance(meta_folds, (int, np.integer))
            or int(meta_folds) < 2
        ):
            raise ValueError("final_upstream_meta_inner_folds must be an integer >= 2")
        object.__setattr__(self, "final_upstream_meta_inner_folds", int(meta_folds))
        head_regularization = self.final_upstream_head_regularization
        if (
            isinstance(head_regularization, (bool, np.bool_))
            or not isinstance(head_regularization, (int, float, np.integer, np.floating))
            or not math.isfinite(float(head_regularization))
            or float(head_regularization) <= 0.0
        ):
            raise ValueError("final_upstream_head_regularization must be positive and finite")
        object.__setattr__(
            self,
            "final_upstream_head_regularization",
            float(head_regularization),
        )
        if not isinstance(self.require_neural_query_moments, bool):
            raise ValueError("require_neural_query_moments must be a boolean")
        if self.require_neural_query_moments and self.derive_sparse_query_moments_when_missing:
            raise ValueError(
                "required neural query moments cannot enable the sparse query fallback"
            )
        if (
            self.require_neural_query_moments
            and self.post_extraction_review_rounds > 0
            and (
                not self.require_review_feature_banks
                or not self.require_final_upstream_neural_query_inputs
            )
        ):
            raise ValueError(
                "adaptive required neural query moments need context-fit review feature "
                "banks and final neural-query inputs"
            )
        normalized_query_artifacts: dict[int, QueryEvidenceArtifact] = {}
        for raw_fold, raw_artifact in self.neural_query_moment_artifacts_by_fold.items():
            try:
                fold = int(raw_fold)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "neural query evidence registry keys must be positive integers"
                ) from exc
            if fold < 1:
                raise ValueError("neural query evidence registry keys must be positive")
            artifact = _normalize_query_evidence_artifact_registration(raw_artifact)
            if artifact.outer_fold != fold:
                raise ValueError(
                    "neural query evidence registration fold does not match its registry key"
                )
            normalized_query_artifacts[fold] = artifact
        object.__setattr__(
            self,
            "neural_query_moment_artifacts_by_fold",
            normalized_query_artifacts,
        )
        self.query_moment_adapter.validate()


@dataclass(frozen=True)
class _FinalUpstreamHeadInputs:
    train_values: np.ndarray = field(repr=False)
    heldout_values: np.ndarray = field(repr=False)
    model_input_names: tuple[str, ...]
    modifier_indices: tuple[int, ...]
    audit: Mapping[str, Any] = field(repr=False)


def _build_final_upstream_meta_inner_fold_ids(
    outer_train: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int,
    outer_fold: int,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
) -> tuple[tuple[int, ...], Mapping[str, Any]]:
    """Precommit a deterministic, outer-train-only stratified meta partition."""

    count = int(n_splits)
    if count < 2:
        raise ValueError("final upstream meta-inner fold count must be at least two")
    if len(outer_train) < count:
        raise ValueError("outer train is smaller than the final upstream meta-inner fold count")
    required = {"_oci_row_id", treatment_column, outcome_column}
    missing = required - set(outer_train.columns)
    if missing:
        raise ValueError(f"outer train is missing final upstream split columns: {sorted(missing)}")
    row_ids = tuple(int(value) for value in outer_train["_oci_row_id"].tolist())
    if len(row_ids) != len(set(row_ids)) or any(value < 0 for value in row_ids):
        raise ValueError("outer train row IDs must be unique canonical nonnegative integers")
    treatment = outer_train[treatment_column].to_numpy(dtype=float)
    outcome = outer_train[outcome_column].to_numpy(dtype=float)
    if not np.isfinite(treatment).all() or set(np.unique(treatment).tolist()) != {0.0, 1.0}:
        raise ValueError("final upstream meta-inner stratification requires both treatments")
    if not np.isfinite(outcome).all():
        raise ValueError("final upstream meta-inner stratification outcome must be finite")
    treatment_counts = np.unique(treatment, return_counts=True)[1]
    if int(treatment_counts.min()) < count:
        raise ValueError(
            "each treatment arm must have at least final_upstream_meta_inner_folds rows"
        )

    normalized_outcome_type = str(outcome_type).strip().lower()
    if normalized_outcome_type not in {"binary", "continuous"}:
        raise ValueError("outcome_type must be 'binary' or 'continuous'")
    strata = np.asarray([f"treatment_{int(value)}" for value in treatment], dtype=object)
    strategy = "treatment"
    if normalized_outcome_type == "binary":
        if not set(np.unique(outcome).tolist()).issubset({0.0, 1.0}):
            raise ValueError("binary outcome stratification requires outcomes encoded as 0/1")
        joint = np.asarray(
            [f"treatment_{int(t)}__outcome_{int(y)}" for t, y in zip(treatment, outcome)],
            dtype=object,
        )
        _, joint_counts = np.unique(joint, return_counts=True)
        if len(joint_counts) >= 2 and int(joint_counts.min()) >= count:
            strata = joint
            strategy = "joint_treatment_outcome"

    seed = int(random_state) + 104729 * int(outer_fold)
    splitter = StratifiedKFold(n_splits=count, shuffle=True, random_state=seed)
    assignments = np.zeros(len(outer_train), dtype=int)
    for fold_id, (_, heldout_positions) in enumerate(
        splitter.split(np.zeros(len(outer_train)), strata),
        start=1,
    ):
        assignments[heldout_positions] = int(fold_id)
    if set(assignments.tolist()) != set(range(1, count + 1)):
        raise RuntimeError("final upstream meta-inner partition is incomplete")

    fold_audit: list[dict[str, Any]] = []
    for fold_id in range(1, count + 1):
        mask = assignments == fold_id
        fold_treatment = treatment[mask]
        record: dict[str, Any] = {
            "fold_id": fold_id,
            "heldout_row_count": int(mask.sum()),
            "complementary_fit_row_count": int(len(assignments) - mask.sum()),
            "treatment_counts": {
                str(int(value)): int(np.sum(fold_treatment == value)) for value in (0.0, 1.0)
            },
        }
        if normalized_outcome_type == "binary":
            fold_outcome = outcome[mask]
            record["outcome_counts"] = {
                str(int(value)): int(np.sum(fold_outcome == value)) for value in (0.0, 1.0)
            }
        fold_audit.append(record)
    fold_ids = tuple(int(value) for value in assignments.tolist())
    audit = {
        "strategy": strategy,
        "n_splits": count,
        "seed": seed,
        "outer_train_row_count": len(row_ids),
        "assignment_sha256": _content_sha256(
            {"outer_train_row_ids": list(row_ids), "meta_inner_fold_ids": list(fold_ids)}
        ),
        "folds": fold_audit,
        "outer_train_labels_only": True,
        "outer_heldout_rows_used": False,
        "row_level_assignments_persisted_in_runner_audit": False,
    }
    return fold_ids, audit


def _final_upstream_column_aggregates(
    *,
    names: Sequence[str],
    kinds: Sequence[str],
    roles: Sequence[str] | None,
    train_values: np.ndarray,
) -> list[dict[str, Any]]:
    values = np.asarray(train_values, dtype=float)
    rows: list[dict[str, Any]] = []
    for index, (name, kind) in enumerate(zip(names, kinds)):
        column = values[:, index]
        record: dict[str, Any] = {
            "name": str(name),
            "source_kind": str(kind),
            "outer_train_oof_mean": float(np.mean(column)),
            "outer_train_oof_standard_deviation": float(np.std(column, ddof=0)),
            "outer_train_oof_minimum": float(np.min(column)),
            "outer_train_oof_maximum": float(np.max(column)),
        }
        if roles is not None:
            record["consumer_role"] = str(roles[index])
        rows.append(record)
    return rows


_REQUIRED_NEURAL_QUERY_RAW_FAMILIES = (
    ("neural_query_treatment_moments", PROPENSITY_NUISANCE_FEATURE_ROLE),
    ("neural_query_outcome_moments", OUTCOME_NUISANCE_FEATURE_ROLE),
    ("neural_query_effect_moments", UNCALIBRATED_EFFECT_MODIFIER_ROLE),
)


def _prepare_final_upstream_head_inputs(
    package: AuthenticatedFinalContextFitUpstreamBank,
    *,
    outer_fold: int,
    expected_train_row_ids: tuple[int, ...],
    expected_heldout_row_ids: tuple[int, ...],
    expected_meta_inner_fold_ids: tuple[int, ...],
    expected_producer_identity_sha256: str,
    require_neural_query_inputs: bool,
) -> _FinalUpstreamHeadInputs:
    """Authenticate, copy, and namespace one final upstream package."""

    if type(package) is not AuthenticatedFinalContextFitUpstreamBank:
        raise TypeError("final upstream producer returned an unauthenticated bank type")
    package.verify_authenticated_content()
    if package.outer_fold != int(outer_fold):
        raise ValueError("final upstream bank outer fold does not match the requested fold")
    if package.producer_identity_sha256 != str(expected_producer_identity_sha256):
        raise ValueError("final upstream bank is not bound to the injected producer identity")
    source = package.calibrated_sources
    raw = package.raw_features
    for label, bank in (("calibrated source", source), ("raw feature", raw)):
        if bank.train_row_ids != expected_train_row_ids:
            raise ValueError(f"final upstream {label} train row identity or order changed")
        if bank.heldout_row_ids != expected_heldout_row_ids:
            raise ValueError(f"final upstream {label} heldout row identity or order changed")
        if bank.meta_inner_fold_ids != expected_meta_inner_fold_ids:
            raise ValueError(f"final upstream {label} meta-inner assignments changed")

    source_train = np.array(source.train_oof_values, dtype=float, copy=True, order="C")
    source_heldout = np.array(source.outer_heldout_values, dtype=float, copy=True, order="C")
    raw_train = np.array(raw.train_oof_values, dtype=float, copy=True, order="C")
    raw_heldout = np.array(raw.outer_heldout_values, dtype=float, copy=True, order="C")
    # Authenticate persisted bytes and in-memory content again after the copy,
    # so the estimator never consumes a view that changed between verification
    # and materialization.
    package.verify_authenticated_content()
    if not all(
        np.isfinite(values).all()
        for values in (source_train, source_heldout, raw_train, raw_heldout)
    ):
        raise ValueError("final upstream model inputs contain non-finite values")

    source_count = len(source.source_names)
    source_input_names = tuple(
        f"final_upstream__calibrated_tau__{index:03d}" for index in range(1, source_count + 1)
    )
    raw_input_names = tuple(
        f"final_upstream__raw_feature__{index:03d}"
        for index in range(1, len(raw.feature_names) + 1)
    )
    model_input_names = (*source_input_names, *raw_input_names)
    modifier_indices = [*range(source_count)]
    modifier_indices.extend(
        source_count + index
        for index, role in enumerate(raw.consumer_roles)
        if role == UNCALIBRATED_EFFECT_MODIFIER_ROLE
    )
    required_query_pairs = frozenset(_REQUIRED_NEURAL_QUERY_RAW_FAMILIES)
    observed_raw_pairs = frozenset(zip(raw.feature_kinds, raw.consumer_roles))
    present_query_pairs = required_query_pairs & observed_raw_pairs
    missing_query_pairs = required_query_pairs - observed_raw_pairs
    query_raw_features = [
        str(name)
        for name, kind, role in zip(
            raw.feature_names,
            raw.feature_kinds,
            raw.consumer_roles,
        )
        if (str(kind), str(role)) in required_query_pairs
    ]
    if require_neural_query_inputs and missing_query_pairs:
        rendered_missing = ", ".join(
            f"{kind} ({role})" for kind, role in sorted(missing_query_pairs)
        )
        raise ValueError(
            "required final upstream neural-query raw families are absent or have "
            f"the wrong consumer role: {rendered_missing}"
        )

    meta_fold_counts = [
        {
            "fold_id": int(fold_id),
            "heldout_row_count": int(
                sum(value == fold_id for value in expected_meta_inner_fold_ids)
            ),
            "complementary_fit_row_count": int(
                len(expected_meta_inner_fold_ids)
                - sum(value == fold_id for value in expected_meta_inner_fold_ids)
            ),
        }
        for fold_id in dict.fromkeys(expected_meta_inner_fold_ids)
    ]
    audit = {
        "enabled": True,
        "outer_fold": int(outer_fold),
        "cache_key": package.cache_key,
        "manifest_path": str(package.manifest_path),
        "manifest_sha256": package.manifest_sha256,
        "producer_identity_sha256": package.producer_identity_sha256,
        "calibrated_sources": {
            "names": list(source.source_names),
            "source_kinds": list(source.source_kinds),
            "content_sha256": source.content_sha256,
            "column_count": source_count,
            "outer_train_oof_aggregates": _final_upstream_column_aggregates(
                names=source.source_names,
                kinds=source.source_kinds,
                roles=None,
                train_values=source_train,
            ),
            "all_columns_routed_as_treatment_modifiers": True,
        },
        "raw_features": {
            "names": list(raw.feature_names),
            "source_kinds": list(raw.feature_kinds),
            "consumer_roles": list(raw.consumer_roles),
            "content_sha256": raw.content_sha256,
            "column_count": len(raw.feature_names),
            "outer_train_oof_aggregates": _final_upstream_column_aggregates(
                names=raw.feature_names,
                kinds=raw.feature_kinds,
                roles=raw.consumer_roles,
                train_values=raw_train,
            ),
            "modifier_only_routing": {
                "interaction_role": UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                "interaction_column_count": len(modifier_indices) - source_count,
                "applies_when_interact_all_features_is_false": True,
                "non_modifier_raw_features_are_main_effects_only_in_modifier_only_mode": True,
            },
        },
        "neural_query_inputs": {
            "required": bool(require_neural_query_inputs),
            "calibrated_source_names": [],
            "raw_feature_names": query_raw_features,
            "required_raw_kind_role_pairs": [
                {"source_kind": kind, "consumer_role": role}
                for kind, role in _REQUIRED_NEURAL_QUERY_RAW_FAMILIES
            ],
            "recognized_raw_kinds": sorted(kind for kind, _role in present_query_pairs),
            "complete_required_raw_family_set_present": not missing_query_pairs,
            "used_as_final_model_inputs": bool(query_raw_features),
        },
        "lineage": {
            "meta_inner_fold_counts": meta_fold_counts,
            "train_oof_recursive_fit_rows_are_exact_complementary_meta_folds": True,
            "train_oof_target_rows_excluded_from_recursive_fit_rows": True,
            "outer_heldout_recursive_fit_rows_are_exact_complete_outer_train": True,
            "authenticated_bank_verified_before_and_after_matrix_copy": True,
        },
        "train_row_fingerprint": row_set_fingerprint(expected_train_row_ids),
        "heldout_row_fingerprint": row_set_fingerprint(expected_heldout_row_ids),
        "train_row_order_sha256": _content_sha256(list(expected_train_row_ids)),
        "heldout_row_order_sha256": _content_sha256(list(expected_heldout_row_ids)),
        "meta_inner_assignment_sha256": _content_sha256(
            {
                "outer_train_row_ids": list(expected_train_row_ids),
                "meta_inner_fold_ids": list(expected_meta_inner_fold_ids),
            }
        ),
        "direct_upstream_numerical_signals_used_as_final_model_inputs": True,
        "outer_heldout_labels_passed_to_producer": False,
        "row_level_numerical_vectors_persisted_in_runner_audit": False,
        "outer_heldout_numerical_aggregates_persisted_in_runner_audit": False,
    }
    return _FinalUpstreamHeadInputs(
        train_values=np.column_stack((source_train, raw_train)),
        heldout_values=np.column_stack((source_heldout, raw_heldout)),
        model_input_names=model_input_names,
        modifier_indices=tuple(modifier_indices),
        audit=audit,
    )


@dataclass(frozen=True)
class ReviewPartitionSchedule:
    """Fixed outer-train-only partitions for one sequential review loop."""

    outer_fold: int
    seed: int
    strategy: str
    attempt: int
    initial_spent_fold_ids: tuple[int, ...]
    gate_fold_ids: tuple[int, ...]
    outer_train_row_ids: tuple[int, ...] = field(repr=False)
    row_ids_by_fold: Mapping[int, tuple[int, ...]] = field(repr=False)
    audit: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        canonical_rows = tuple(map(int, self.outer_train_row_ids))
        if (
            not canonical_rows
            or len(canonical_rows) != len(set(canonical_rows))
            or any(row_id < 0 for row_id in canonical_rows)
        ):
            raise ValueError(
                "review schedule canonical outer-training rows must be "
                "nonempty unique nonnegative integers"
            )
        partition_ids = set(map(int, self.row_ids_by_fold))
        scheduled_fold_ids = tuple(
            map(int, (*self.initial_spent_fold_ids, *self.gate_fold_ids))
        )
        if (
            len(scheduled_fold_ids) != len(set(scheduled_fold_ids))
            or set(scheduled_fold_ids) != partition_ids
        ):
            raise ValueError(
                "review schedule fold sequence must cover each partition exactly once"
            )
        partition_rows = tuple(
            int(row_id)
            for rows in self.row_ids_by_fold.values()
            for row_id in rows
        )
        if (
            len(partition_rows) != len(set(partition_rows))
            or set(partition_rows) != set(canonical_rows)
        ):
            raise ValueError(
                "review schedule partitions must cover the canonical "
                "outer-training rows exactly once"
            )

    def row_ids(self, fold_ids: Sequence[int]) -> tuple[int, ...]:
        requested = tuple(int(value) for value in fold_ids)
        unknown = set(requested) - set(self.row_ids_by_fold)
        if unknown:
            raise ValueError(f"unknown review partition IDs: {sorted(unknown)}")
        selected_rows = {
            int(row_id)
            for fold_id in set(requested)
            for row_id in self.row_ids_by_fold[fold_id]
        }
        return tuple(
            row_id
            for row_id in self.outer_train_row_ids
            if row_id in selected_rows
        )


def _spent_evidence_context_epoch(
    schedule: ReviewPartitionSchedule,
    spent_fold_ids: Sequence[int],
) -> int:
    """Return the cache epoch implied by the exact consumed-gate prefix."""

    spent = tuple(map(int, spent_fold_ids))
    initial = tuple(map(int, schedule.initial_spent_fold_ids))
    gates = tuple(map(int, schedule.gate_fold_ids))
    if len(spent) < len(initial) or spent[: len(initial)] != initial:
        raise ValueError("spent evidence folds must begin with the exact initial-spent fold order")
    consumed = spent[len(initial) :]
    if consumed != gates[: len(consumed)]:
        raise ValueError("spent evidence folds must add only the exact consumed review-gate prefix")
    return len(consumed)


def _spent_evidence_context_epoch_policy_audit() -> dict[str, Any]:
    return {
        "policy_version": SPENT_EVIDENCE_CONTEXT_EPOCH_POLICY_VERSION,
        "epoch_definition": "number_of_review_gates_consumed_before_context_fit",
        "provider_review_round_argument_is_context_epoch": True,
        "consumer_review_round_is_separate": True,
        "initial_selector_context_epoch": 0,
        "first_review_reuses_initial_selector_context_epoch": True,
    }


def _review_partition_is_usable(
    positions_by_fold: Sequence[np.ndarray],
    *,
    treatment: np.ndarray,
    outcome: np.ndarray,
    outcome_type: str,
    minimum_rows: int,
) -> bool:
    for positions in positions_by_fold:
        if len(positions) < int(minimum_rows):
            return False
        if set(np.unique(treatment[positions])) != {0, 1}:
            return False
        if outcome_type == "binary" and set(np.unique(outcome[positions])) != {0.0, 1.0}:
            return False
    return True


def _build_review_partition_schedule(
    outer_train: pd.DataFrame,
    *,
    outer_fold: int,
    review_rounds: int,
    minimum_partition_rows: int,
    random_state: int,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
) -> ReviewPartitionSchedule:
    """Build a deterministic stratified schedule without touching outer-heldout rows."""

    rounds = int(review_rounds)
    if rounds < 1:
        raise ValueError("review_rounds must be positive when building a schedule")
    partition_count = rounds + 3
    minimum_rows = int(minimum_partition_rows)
    if len(outer_train) < partition_count * minimum_rows:
        raise ValueError(
            "post-extraction review cannot form the required minimum-size spent/gate "
            f"partitions: rows={len(outer_train)} partitions={partition_count} "
            f"minimum_rows={minimum_rows}"
        )
    required = {"_oci_row_id", treatment_column, outcome_column}
    missing = required - set(outer_train.columns)
    if missing:
        raise ValueError(f"outer train lacks review partition columns: {sorted(missing)}")
    frame = outer_train.reset_index(drop=True)
    row_ids = frame["_oci_row_id"].to_numpy(dtype=int)
    if len(row_ids) != len(set(map(int, row_ids))):
        raise ValueError("review partition rows contain duplicate canonical IDs")
    treatment = frame[treatment_column].to_numpy(dtype=int)
    outcome = frame[outcome_column].to_numpy(dtype=float)
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError("review partitions require both treatment classes in outer train")
    normalized_outcome_type = str(outcome_type).strip().lower()
    if normalized_outcome_type not in {"binary", "continuous"}:
        raise ValueError("outcome_type must be binary or continuous")
    if normalized_outcome_type == "binary" and set(np.unique(outcome)) != {0.0, 1.0}:
        raise ValueError("binary review partitions require outcomes encoded as 0/1")

    def eligible(labels: np.ndarray) -> bool:
        _values, counts = np.unique(labels, return_counts=True)
        return bool(len(counts) >= 2 and np.min(counts) >= partition_count)

    stratifiers: list[tuple[str, np.ndarray | None]] = []
    if normalized_outcome_type == "binary":
        joint = np.asarray(
            [f"a{int(a)}_y{int(y)}" for a, y in zip(treatment, outcome)],
            dtype=object,
        )
        if eligible(joint):
            stratifiers.append(("joint_treatment_binary_outcome", joint))
        if eligible(treatment):
            stratifiers.append(("treatment", treatment.astype(object)))
        if eligible(outcome):
            stratifiers.append(("binary_outcome", outcome.astype(object)))
    else:
        bin_count = min(4, max(2, len(frame) // (partition_count * minimum_rows)))
        ranked = pd.Series(outcome).rank(method="first")
        outcome_bins = pd.qcut(ranked, q=bin_count, labels=False, duplicates="drop")
        joint = np.asarray(
            [f"a{int(a)}_q{int(q)}" for a, q in zip(treatment, outcome_bins)],
            dtype=object,
        )
        if eligible(joint):
            stratifiers.append(("joint_treatment_outcome_quantile", joint))
        if eligible(treatment):
            stratifiers.append(("treatment", treatment.astype(object)))
    stratifiers.append(("deterministic_shuffled_kfold", None))

    base_seed = int(random_state) + int(outer_fold) * 1009
    selected: tuple[str, int, int, list[np.ndarray]] | None = None
    for strategy, labels in stratifiers:
        for attempt in range(64):
            effective_seed = base_seed + attempt
            if labels is None:
                splitter = KFold(
                    n_splits=partition_count,
                    shuffle=True,
                    random_state=effective_seed,
                )
                splits = splitter.split(np.zeros(len(frame), dtype=float))
            else:
                splitter = StratifiedKFold(
                    n_splits=partition_count,
                    shuffle=True,
                    random_state=effective_seed,
                )
                splits = splitter.split(np.zeros(len(frame), dtype=float), labels)
            positions = [np.asarray(test, dtype=int) for _fit, test in splits]
            if _review_partition_is_usable(
                positions,
                treatment=treatment,
                outcome=outcome,
                outcome_type=normalized_outcome_type,
                minimum_rows=minimum_rows,
            ):
                selected = (strategy, attempt, effective_seed, positions)
                break
        if selected is not None:
            break
    if selected is None:
        raise ValueError(
            "post-extraction review could not form deterministic partitions with "
            "the required row counts and treatment/outcome class support"
        )
    strategy, attempt, effective_seed, positions_by_fold = selected
    rows_by_fold = {
        fold_id: tuple(map(int, row_ids[positions]))
        for fold_id, positions in enumerate(positions_by_fold, start=1)
    }
    partition_rows: list[dict[str, Any]] = []
    for fold_id, positions in enumerate(positions_by_fold, start=1):
        ids = rows_by_fold[fold_id]
        row: dict[str, Any] = {
            "fold_id": fold_id,
            "row_ids": list(ids),
            "row_fingerprint": row_set_fingerprint(ids),
            "row_count": len(ids),
            "treatment_counts": {
                str(value): int(np.sum(treatment[positions] == value)) for value in (0, 1)
            },
        }
        if normalized_outcome_type == "binary":
            row["outcome_counts"] = {
                str(value): int(np.sum(outcome[positions] == value)) for value in (0.0, 1.0)
            }
        partition_rows.append(row)
    assignment_content = {
        "outer_fold": int(outer_fold),
        "outer_train_row_ids": list(map(int, row_ids)),
        "partitions": [
            {"fold_id": row["fold_id"], "row_ids": row["row_ids"]} for row in partition_rows
        ],
    }
    audit = {
        "schema_version": POST_EXTRACTION_REVIEW_PARTITION_SCHEMA_VERSION,
        "outer_fold": int(outer_fold),
        "partition_count": partition_count,
        "initial_spent_partition_count": 3,
        "initial_spent_fold_ids": [1, 2, 3],
        "gate_fold_ids": list(range(4, partition_count + 1)),
        "configured_random_state": int(random_state),
        "base_seed": base_seed,
        "effective_seed": effective_seed,
        "selection_attempt": attempt,
        "stratification_strategy": strategy,
        "minimum_partition_rows": minimum_rows,
        "outer_train_row_fingerprint": row_set_fingerprint(row_ids),
        "partition_assignment_sha256": _content_sha256(assignment_content),
        "partitions": partition_rows,
        "outer_heldout_rows_used": False,
    }
    return ReviewPartitionSchedule(
        outer_fold=int(outer_fold),
        seed=effective_seed,
        strategy=strategy,
        attempt=attempt,
        initial_spent_fold_ids=(1, 2, 3),
        gate_fold_ids=tuple(range(4, partition_count + 1)),
        outer_train_row_ids=tuple(map(int, row_ids)),
        row_ids_by_fold=rows_by_fold,
        audit=json.loads(_canonical_json(audit)),
    )


def _build_injected_review_partition_schedule(
    outer_train: pd.DataFrame,
    *,
    outer_fold: int,
    review_rounds: int,
    minimum_partition_rows: int,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    provider: ReviewPartitionProvider,
    provider_identity: Mapping[str, Any],
) -> ReviewPartitionSchedule:
    exact_train_ids = tuple(map(int, outer_train["_oci_row_id"].tolist()))
    raw = provider.get_review_partition_assignments(
        outer_fold=int(outer_fold),
        exact_outer_train_row_ids=exact_train_ids,
    )
    if not isinstance(raw, Mapping):
        raise TypeError("review partition provider must return one fold-to-row mapping")
    gate_count = int(review_rounds)
    partition_count = len(raw)
    initial_spent_count = partition_count - gate_count
    if initial_spent_count < 1:
        raise ValueError(
            "authenticated review assignments must contain at least one initial "
            "spent fold plus one gate per round; "
            f"rounds={gate_count} partitions={partition_count}"
        )
    normalized_raw: list[tuple[int, tuple[int, ...]]] = []
    for raw_fold, raw_rows in raw.items():
        if isinstance(raw_fold, (bool, np.bool_)) or not isinstance(raw_fold, (int, np.integer)):
            raise TypeError("review assignment fold IDs must be positive integers")
        fold_id = int(raw_fold)
        if fold_id < 1:
            raise ValueError("review assignment fold IDs must be positive")
        if isinstance(raw_rows, (str, bytes, Mapping)):
            raise TypeError("review assignment rows must be a sequence")
        rows = tuple(map(int, raw_rows))
        if not rows or len(rows) != len(set(rows)) or any(value < 0 for value in rows):
            raise ValueError("review assignment rows must be non-empty unique row IDs")
        normalized_raw.append((fold_id, rows))
    normalized_raw.sort(key=lambda item: item[0])
    if len({fold_id for fold_id, _rows in normalized_raw}) != partition_count:
        raise ValueError("review assignment fold IDs must be unique")
    flattened = [row_id for _fold_id, rows in normalized_raw for row_id in rows]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(exact_train_ids):
        raise ValueError(
            "authenticated review assignments must partition the exact outer-train row set"
        )
    indexed_positions = {int(row_id): position for position, row_id in enumerate(exact_train_ids)}
    positions_by_fold = [
        np.asarray([indexed_positions[row_id] for row_id in rows], dtype=int)
        for _fold_id, rows in normalized_raw
    ]
    treatment = outer_train[treatment_column].to_numpy(dtype=int)
    outcome = outer_train[outcome_column].to_numpy(dtype=float)
    normalized_outcome_type = str(outcome_type).strip().lower()
    if not _review_partition_is_usable(
        positions_by_fold,
        treatment=treatment,
        outcome=outcome,
        outcome_type=normalized_outcome_type,
        minimum_rows=int(minimum_partition_rows),
    ):
        raise ValueError(
            "authenticated review assignments fail minimum rows or treatment/outcome "
            "class support"
        )
    rows_by_fold = {fold_id: rows for fold_id, rows in normalized_raw}
    ordered_fold_ids = tuple(fold_id for fold_id, _rows in normalized_raw)
    partitions: list[dict[str, Any]] = []
    for (fold_id, rows), positions in zip(normalized_raw, positions_by_fold):
        summary: dict[str, Any] = {
            "fold_id": fold_id,
            "row_ids": list(rows),
            "row_fingerprint": row_set_fingerprint(rows),
            "row_count": len(rows),
            "treatment_counts": {
                str(value): int(np.sum(treatment[positions] == value)) for value in (0, 1)
            },
        }
        if normalized_outcome_type == "binary":
            summary["outcome_counts"] = {
                str(value): int(np.sum(outcome[positions] == value)) for value in (0.0, 1.0)
            }
        partitions.append(summary)
    assignment_content = {
        "outer_fold": int(outer_fold),
        "outer_train_row_ids": list(exact_train_ids),
        "partitions": [
            {"fold_id": row["fold_id"], "row_ids": row["row_ids"]} for row in partitions
        ],
    }
    provider_identity_sha = str(provider_identity.get("identity_sha256") or "")
    audit = {
        "schema_version": POST_EXTRACTION_REVIEW_PARTITION_SCHEMA_VERSION,
        "outer_fold": int(outer_fold),
        "partition_count": partition_count,
        "initial_spent_partition_count": initial_spent_count,
        "initial_spent_fold_ids": list(ordered_fold_ids[:initial_spent_count]),
        "gate_fold_ids": list(ordered_fold_ids[initial_spent_count:]),
        "stratification_strategy": "authenticated_injected_exact_assignments",
        "minimum_partition_rows": int(minimum_partition_rows),
        "outer_train_row_fingerprint": row_set_fingerprint(exact_train_ids),
        "partition_assignment_sha256": _content_sha256(assignment_content),
        "partition_provider_identity_sha256": provider_identity_sha,
        "partitions": partitions,
        "outer_heldout_rows_used": False,
    }
    return ReviewPartitionSchedule(
        outer_fold=int(outer_fold),
        seed=0,
        strategy="authenticated_injected_exact_assignments",
        attempt=0,
        initial_spent_fold_ids=ordered_fold_ids[:initial_spent_count],
        gate_fold_ids=ordered_fold_ids[initial_spent_count:],
        outer_train_row_ids=exact_train_ids,
        row_ids_by_fold=rows_by_fold,
        audit=json.loads(_canonical_json(audit)),
    )


_MISSING_FUSION_REASONING_SETTING = object()


def _effective_fusion_agent_enable_thinking(fusion_agent: Any) -> Any:
    """Read an injected proposal agent's effective chat-template setting.

    Staged fusion wraps the OpenAI-compatible agent in one or more objects that
    expose it as ``proposal_agent``.  Custom test/deterministic agents need not
    carry an LLM search config, so absence is distinct from an explicit
    ``None`` (which means leave endpoint behavior unspecified).
    """

    current = fusion_agent
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        search_config = getattr(current, "search_config", None)
        if search_config is not None and hasattr(search_config, "agent_enable_thinking"):
            return getattr(search_config, "agent_enable_thinking")
        current = getattr(current, "proposal_agent", None)
    return _MISSING_FUSION_REASONING_SETTING


def _effective_fusion_agent_thinking_token_budget(fusion_agent: Any) -> Any:
    """Read an injected proposal agent's configured reasoning-token budget."""

    current = fusion_agent
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        search_config = getattr(current, "search_config", None)
        if search_config is not None and hasattr(search_config, "agent_thinking_token_budget"):
            return getattr(search_config, "agent_thinking_token_budget")
        current = getattr(current, "proposal_agent", None)
    return _MISSING_FUSION_REASONING_SETTING


def _effective_fusion_agent_max_tokens(fusion_agent: Any) -> Any:
    """Read an injected proposal agent's total generation-token cap."""

    current = fusion_agent
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        search_config = getattr(current, "search_config", None)
        if search_config is not None and hasattr(search_config, "agent_max_tokens"):
            return getattr(search_config, "agent_max_tokens")
        current = getattr(current, "proposal_agent", None)
    return _MISSING_FUSION_REASONING_SETTING


def _validate_fusion_reasoning_configuration(
    fusion_agent: Any,
    config: AllEvidenceFusionRunnerConfig,
) -> None:
    """Reject a declared runner/remote-agent reasoning mismatch at startup."""

    effective = _effective_fusion_agent_enable_thinking(fusion_agent)
    if effective is _MISSING_FUSION_REASONING_SETTING:
        return
    expected = config.fusion_enable_thinking
    if not isinstance(effective, (bool, np.bool_)) or bool(effective) != expected:
        raise ValueError(
            "fusion reasoning configuration mismatch: "
            "AllEvidenceFusionRunnerConfig.fusion_enable_thinking="
            f"{expected!r}, but the injected fusion agent's effective "
            "search_config.agent_enable_thinking="
            f"{effective!r}"
        )
    expected_budget = config.fusion_thinking_token_budget
    effective_budget = _effective_fusion_agent_thinking_token_budget(fusion_agent)
    if effective_budget is _MISSING_FUSION_REASONING_SETTING:
        # Older deterministic test doubles may expose only the reasoning
        # switch.  They are equivalent to an unset budget, but cannot satisfy
        # an explicitly declared positive runner budget.
        if expected_budget is None:
            effective_budget = None
    elif effective_budget is not None:
        if (
            isinstance(effective_budget, (bool, np.bool_))
            or not isinstance(effective_budget, (int, np.integer))
            or int(effective_budget) <= 0
        ):
            raise ValueError(
                "fusion thinking token budget configuration mismatch: "
                "the injected fusion agent's effective "
                "search_config.agent_thinking_token_budget="
                f"{effective_budget!r} is not a positive integer or None"
            )
        effective_budget = int(effective_budget)
    if effective_budget != expected_budget:
        rendered_effective = (
            "missing"
            if effective_budget is _MISSING_FUSION_REASONING_SETTING
            else repr(effective_budget)
        )
        raise ValueError(
            "fusion thinking token budget configuration mismatch: "
            "AllEvidenceFusionRunnerConfig.fusion_thinking_token_budget="
            f"{expected_budget!r}, but the injected fusion agent's effective "
            "search_config.agent_thinking_token_budget="
            f"{rendered_effective}"
        )

    expected_max_tokens = config.fusion_max_tokens
    effective_max_tokens = _effective_fusion_agent_max_tokens(fusion_agent)
    if effective_max_tokens is _MISSING_FUSION_REASONING_SETTING:
        rendered_effective_max_tokens = "missing"
    elif (
        isinstance(effective_max_tokens, (bool, np.bool_))
        or not isinstance(effective_max_tokens, (int, np.integer))
        or int(effective_max_tokens) <= 0
    ):
        raise ValueError(
            "fusion max token configuration mismatch: the injected fusion "
            "agent's effective search_config.agent_max_tokens="
            f"{effective_max_tokens!r} is not a positive integer"
        )
    else:
        effective_max_tokens = int(effective_max_tokens)
        rendered_effective_max_tokens = repr(effective_max_tokens)
    if effective_max_tokens != expected_max_tokens:
        raise ValueError(
            "fusion max token configuration mismatch: "
            "AllEvidenceFusionRunnerConfig.fusion_max_tokens="
            f"{expected_max_tokens!r}, but the injected fusion agent's "
            "effective search_config.agent_max_tokens="
            f"{rendered_effective_max_tokens}"
        )


def _review_provider_identity(provider: Any, *, label: str) -> Mapping[str, Any] | None:
    if provider is None:
        return None
    identity_method = getattr(provider, "identity", None)
    if not callable(identity_method):
        raise TypeError(f"{label} must expose identity() for immutable audit binding")
    raw = identity_method()
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label}.identity() must return one mapping")

    def reject_forbidden(value: Any, *, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if _FORBIDDEN_NAME.search(str(key)):
                    raise ValueError(f"{label} identity contains forbidden field at {path}.{key}")
                reject_forbidden(child, path=f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                reject_forbidden(child, path=f"{path}[{index}]")

    reject_forbidden(raw, path="identity")
    detached = json.loads(_canonical_json(raw))
    return {
        "identity": detached,
        "identity_sha256": _content_sha256(detached),
    }


@dataclass(frozen=True)
class AllEvidenceFusionRunResult:
    prediction_path: Path
    run_manifest_path: Path
    fold_manifest_paths: tuple[Path, ...]
    prediction_sha256: str


@dataclass(frozen=True)
class PreparedHierarchicalDiscoveryFold:
    """Local, transport-free inputs needed after one batch is approved."""

    outer_fold: int
    schedule: ReviewPartitionSchedule
    evidence_inputs: tuple[FoldEvidenceInput, ...] = field(repr=False)
    initial_spent_evidence_audit: Mapping[str, Any] = field(repr=False)
    catalog: RoleNeutralEvidenceCatalog = field(repr=False)
    chunk_plan: ArchitectureChunkPlan = field(repr=False)
    first_gate_materialization_intent: FirstGateMaterializationIntent | None = field(
        repr=False
    )
    reference_only_direct_numerical_contract: (
        AuthenticatedReferenceOnlyDirectNumericalContract | None
    ) = field(repr=False)
    first_gate_materialization_intent_path: Path
    first_gate_materialization_intent_file_sha256: str
    agent: ApprovedHierarchicalDiscoveryAgent = field(repr=False)
    preparation_manifest_path: Path

    def __post_init__(self) -> None:
        if isinstance(self.outer_fold, bool) or not isinstance(self.outer_fold, int):
            raise TypeError("prepared hierarchical outer_fold must be an integer")
        if self.outer_fold < 1 or self.schedule.outer_fold != self.outer_fold:
            raise ValueError("prepared hierarchical fold label differs from its schedule")
        if self.catalog.outer_fold != self.outer_fold:
            raise ValueError("prepared hierarchical fold label differs from its catalog")
        if self.agent.catalog.catalog_sha256 != self.catalog.catalog_sha256:
            raise ValueError("prepared hierarchical agent cites a different catalog")
        if self.agent.chunk_plan.plan_sha256 != self.chunk_plan.plan_sha256:
            raise ValueError("prepared hierarchical agent cites a different chunk plan")
        if (
            self.first_gate_materialization_intent is None
        ) == (self.reference_only_direct_numerical_contract is None):
            raise ValueError(
                "prepared hierarchy requires exactly one numerical contract"
            )
        agent_intent = getattr(self.agent, "first_gate_materialization_intent", None)
        agent_reference = getattr(
            self.agent,
            "reference_only_direct_numerical_contract",
            None,
        )
        if self.first_gate_materialization_intent is not None:
            self.first_gate_materialization_intent.verify()
            if (
                not isinstance(agent_intent, FirstGateMaterializationIntent)
                or agent_reference is not None
                or agent_intent.content_sha256
                != self.first_gate_materialization_intent.content_sha256
            ):
                raise ValueError(
                    "prepared hierarchical agent cites a different materialization intent"
                )
        else:
            assert self.reference_only_direct_numerical_contract is not None
            self.reference_only_direct_numerical_contract.verify(
                catalog=self.catalog
            )
            if (
                agent_intent is not None
                or type(agent_reference)
                is not AuthenticatedReferenceOnlyDirectNumericalContract
                or agent_reference.content_sha256
                != self.reference_only_direct_numerical_contract.content_sha256
            ):
                raise ValueError(
                    "prepared hierarchical agent cites a different reference contract"
                )
        if (
            not self.first_gate_materialization_intent_path.is_file()
            or not _SHA256.fullmatch(self.first_gate_materialization_intent_file_sha256)
            or sha256_file(self.first_gate_materialization_intent_path)
            != self.first_gate_materialization_intent_file_sha256
        ):
            raise ValueError("prepared first-gate materialization intent is unauthenticated")
        if not self.preparation_manifest_path.is_file():
            raise ValueError("prepared hierarchical fold manifest is missing")


_PREPARED_HIERARCHY_CAPABILITY_LOCK = threading.Lock()
_PREPARED_HIERARCHY_CAPABILITIES: dict[
    int,
    tuple[weakref.ReferenceType[object], str],
] = {}

PRODUCTION_HIERARCHY_RUNTIME_BINDING_SCHEMA = "production_hierarchy_same_process_runner_binding_v1"


def _current_production_hierarchy_runtime_binding(
    runner: "AllEvidenceFusionRunner",
) -> tuple[dict[str, Any], tuple[tuple[str, object], ...]]:
    """Reauthenticate the exact same-process objects used after preparation.

    Production one-shot execution deliberately does not accept pathname-based
    replay registrations.  The configured cache overlays already retain their
    authenticated immutable source snapshots.  This binding instead pins the
    exact runner/provider objects and freshly recomputed identities represented
    by the generic preparation input manifest.
    """

    if type(runner) is not AllEvidenceFusionRunner:
        raise TypeError("production hierarchy runtime binding requires the exact runner")
    if not runner.hierarchical_discovery_enabled:
        raise RuntimeError("production hierarchy runtime binding requires hierarchy mode")
    if runner.review_spent_evidence_provider is not runner.review_partition_provider:
        raise ValueError("production hierarchy requires one exact spent-catalog/partition provider")
    if runner.review_gate_source_provider is not runner.review_gate_feature_bank_provider:
        raise ValueError("production hierarchy requires one exact shared gate provider")

    spent_identity = _review_provider_identity(
        runner.review_spent_evidence_provider,
        label="review_spent_evidence_provider",
    )
    partition_identity = _review_provider_identity(
        runner.review_partition_provider,
        label="review_partition_provider",
    )
    gate_source_identity = _review_provider_identity(
        runner.review_gate_source_provider,
        label="review_gate_source_provider",
    )
    gate_feature_identity = _review_provider_identity(
        runner.review_gate_feature_bank_provider,
        label="review_gate_feature_bank_provider",
    )
    if spent_identity != partition_identity:
        raise ValueError("spent-catalog and partition provider identities differ")
    if gate_source_identity != gate_feature_identity:
        raise ValueError("shared gate provider identities differ")

    def file_binding(path_value: Path | str) -> dict[str, Any]:
        path = Path(path_value).resolve()
        _snapshot, digest = _read_path_snapshot(path)
        return {"path": str(path), "sha256": digest}

    def registered_file_binding(
        path_value: Path | str,
        declared_sha256: str | None,
        *,
        label: str,
    ) -> dict[str, Any]:
        binding = file_binding(path_value)
        if declared_sha256 is not None and binding["sha256"] != declared_sha256:
            raise ValueError(f"{label} changed from its registered SHA-256")
        return {**binding, "declared_sha256": declared_sha256}

    candidate_pool_registry = {
        str(fold): file_binding(path) for fold, path in sorted(runner.candidate_pool_paths.items())
    }
    query_artifact_registry = {
        str(fold): {
            **registered_file_binding(
                artifact.path,
                artifact.artifact_sha256,
                label=f"query evidence fold {fold}",
            ),
            "outer_fold": artifact.outer_fold,
            "scope": artifact.scope,
            "fit_row_fingerprint": artifact.fit_row_fingerprint,
            "heldout_row_fingerprint": artifact.heldout_row_fingerprint,
        }
        for fold, artifact in sorted(runner.query_evidence_by_fold.items())
    }
    orphan_artifact_registry = {
        str(fold): {
            **registered_file_binding(
                artifact.path,
                artifact.artifact_sha256,
                label=f"orphan evidence fold {fold}",
            ),
        }
        for fold, artifact in sorted(runner.tfidf_orphan_artifacts_by_fold.items())
    }

    hierarchical_runner_identity_raw = runner.hierarchical_discovery_runner.identity()
    if not isinstance(hierarchical_runner_identity_raw, Mapping):
        raise TypeError("hierarchical discovery runner identity must be a mapping")
    hierarchical_runner_identity = json.loads(_canonical_json(hierarchical_runner_identity_raw))
    runner.hierarchical_review_evidence_policy.validate_authentication()
    body = {
        "spent_evidence_provider": spent_identity,
        "review_partition_provider": partition_identity,
        "shared_first_gate_provider": gate_source_identity,
        "final_upstream_producer": _review_provider_identity(
            runner.final_upstream_producer,
            label="final_upstream_producer",
        ),
        "raw_final_upstream_producer": _review_provider_identity(
            runner.raw_final_upstream_producer,
            label="raw_final_upstream_producer",
        ),
        "final_causal_forest_backend": _review_provider_identity(
            runner.final_causal_forest_backend,
            label="final_causal_forest_backend",
        ),
        "extraction_cache_overlay": _review_provider_identity(
            runner.cache_overlay,
            label="cache_overlay",
        ),
        "hierarchical_runner_identity": hierarchical_runner_identity,
        "hierarchical_discovery_config": runner.hierarchical_discovery_config.as_dict(),
        "frozen_review_evidence_policy": (runner.hierarchical_review_evidence_policy.as_dict()),
        "effective_runner_config": asdict(runner.config),
        "hierarchical_architecture_chunk_limits": {
            "max_atoms_per_chunk": runner.hierarchical_max_atoms_per_chunk,
            "max_bytes_per_chunk": runner.hierarchical_max_bytes_per_chunk,
            "max_semantic_member_ids_per_chunk": (
                runner.hierarchical_max_semantic_member_ids_per_chunk
            ),
        },
        "dataset_artifact": file_binding(runner.dataset_path),
        "legacy_handoff_artifact": file_binding(runner.legacy_handoff_path),
        "tfidf_handoff_artifact": file_binding(runner.tfidf_handoff_path),
        "legacy_primary_predictions_artifact": (
            None
            if runner.legacy_primary_predictions_path is None
            else file_binding(runner.legacy_primary_predictions_path)
        ),
        "candidate_pool_registry": candidate_pool_registry,
        "query_evidence_registry": query_artifact_registry,
        "tfidf_orphan_registry": orphan_artifact_registry,
        "coordinate_preserving_nuisance_view_names": (
            None
            if runner.coordinate_preserving_nuisance_view_names is None
            else list(runner.coordinate_preserving_nuisance_view_names)
        ),
        "output_dir": str(runner.output_dir),
        "hierarchical_preparation_dir": str(runner.hierarchical_preparation_dir),
        "hierarchical_job_cache_root": str(runner.hierarchical_discovery_job_cache_root),
        "caller_replay_registrations_accepted": False,
        "runtime_sources_reauthenticated_by_exact_provider_identities": True,
    }
    binding = {
        "schema_version": PRODUCTION_HIERARCHY_RUNTIME_BINDING_SCHEMA,
        "content_sha256": _content_sha256(body),
        "body": json.loads(_canonical_json(body)),
    }
    objects = tuple(
        (
            name,
            getattr(runner, name),
        )
        for name in (
            "review_spent_evidence_provider",
            "review_partition_provider",
            "review_gate_source_provider",
            "review_gate_feature_bank_provider",
            "final_upstream_producer",
            "raw_final_upstream_producer",
            "final_causal_forest_backend",
            "cache_overlay",
            "hierarchical_discovery_runner",
            "hierarchical_discovery_config",
            "hierarchical_review_evidence_policy",
            "config",
            "fusion_agent",
            "extraction_provider",
            "review_agent",
            "tfidf_validator",
            "candidate_pool_paths",
            "query_evidence_by_fold",
            "tfidf_orphan_artifacts_by_fold",
        )
    )
    return binding, objects


def _issue_prepared_hierarchy_capability(
    prepared: "PreparedHierarchicalDiscoveryBatch",
) -> None:
    token = secrets.token_hex(32)
    object.__setattr__(prepared, "_internal_capability_token", token)
    identifier = id(prepared)

    def discard(reference: weakref.ReferenceType[object]) -> None:
        with _PREPARED_HIERARCHY_CAPABILITY_LOCK:
            registered = _PREPARED_HIERARCHY_CAPABILITIES.get(identifier)
            if registered is not None and registered[0] is reference:
                _PREPARED_HIERARCHY_CAPABILITIES.pop(identifier, None)

    reference = weakref.ref(prepared, discard)
    with _PREPARED_HIERARCHY_CAPABILITY_LOCK:
        if identifier in _PREPARED_HIERARCHY_CAPABILITIES:
            raise RuntimeError("prepared hierarchy batch already has an internal capability")
        _PREPARED_HIERARCHY_CAPABILITIES[identifier] = (reference, token)


def _claim_prepared_hierarchy_capability(
    prepared: "PreparedHierarchicalDiscoveryBatch",
) -> str:
    if type(prepared) is not PreparedHierarchicalDiscoveryBatch:
        raise TypeError("prepared batch must be the concrete in-process prepared batch type")
    token = prepared._internal_capability_token
    with _PREPARED_HIERARCHY_CAPABILITY_LOCK:
        registered = _PREPARED_HIERARCHY_CAPABILITIES.get(id(prepared))
        if (
            not token
            or registered is None
            or registered[0]() is not prepared
            or registered[1] != token
        ):
            raise RuntimeError("prepared hierarchy batch has no fresh in-memory capability")
        _PREPARED_HIERARCHY_CAPABILITIES.pop(id(prepared), None)
    return token


def _mark_prepared_hierarchy_external_execution(
    prepared: "PreparedHierarchicalDiscoveryBatch",
) -> None:
    token = prepared._internal_capability_token
    with _PREPARED_HIERARCHY_CAPABILITY_LOCK:
        registered = _PREPARED_HIERARCHY_CAPABILITIES.get(id(prepared))
        if registered is None or registered[0]() is not prepared or registered[1] != token:
            raise RuntimeError("prepared hierarchy batch is not fresh for execution")
        _PREPARED_HIERARCHY_CAPABILITIES.pop(id(prepared), None)


@dataclass(frozen=True)
class PreparedHierarchicalDiscoveryBatch:
    """Inspectable all-fold batch assembled without a hierarchy runner call."""

    coordinator: ApprovedHierarchicalDiscoveryBatchCoordinator = field(repr=False)
    folds: tuple[PreparedHierarchicalDiscoveryFold, ...] = field(repr=False)
    input_manifest_sha256: str
    input_manifest_path: Path
    context_fit_overlay_companion_path: Path
    context_fit_overlay_companion_sha256: str
    first_gate_materialization_intent_index_path: Path
    first_gate_materialization_intent_index_sha256: str
    batch_packet_path: Path
    dataset_sha256: str
    _internal_capability_token: str = field(
        init=False,
        repr=False,
        compare=False,
        default="",
    )

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.input_manifest_sha256):
            raise ValueError("hierarchical preparation input manifest SHA-256 is malformed")
        if not _SHA256.fullmatch(self.dataset_sha256):
            raise ValueError("hierarchical preparation dataset SHA-256 is malformed")
        if self.coordinator.input_manifest_sha256 != self.input_manifest_sha256:
            raise ValueError("hierarchical coordinator cites a different input manifest")
        observed = tuple(row.outer_fold for row in self.folds)
        if observed != tuple(range(1, len(observed) + 1)):
            raise ValueError("hierarchical prepared folds must be complete and ordered")
        if not self.input_manifest_path.is_file() or not self.batch_packet_path.is_file():
            raise ValueError("hierarchical preparation artifacts are missing")
        if (
            not self.context_fit_overlay_companion_path.is_file()
            or not _SHA256.fullmatch(self.context_fit_overlay_companion_sha256)
            or sha256_file(self.context_fit_overlay_companion_path)
            != self.context_fit_overlay_companion_sha256
        ):
            raise ValueError("hierarchical context-fit overlay companion is unauthenticated")
        if (
            not self.first_gate_materialization_intent_index_path.is_file()
            or not _SHA256.fullmatch(self.first_gate_materialization_intent_index_sha256)
            or sha256_file(self.first_gate_materialization_intent_index_path)
            != self.first_gate_materialization_intent_index_sha256
        ):
            raise ValueError("hierarchical first-gate intent index is unauthenticated")

    @property
    def approval_sha256(self) -> str:
        return self.coordinator.precommit.approval_sha256

    def render_offline_precommit(self, *, indent: int = 2) -> str:
        return self.coordinator.render_offline_precommit(indent=indent)

    def execute(self, *, approved_batch_sha256: str) -> ApprovedHierarchicalDiscoveryBatchResult:
        _mark_prepared_hierarchy_external_execution(self)
        return self.coordinator.execute(approved_batch_sha256=approved_batch_sha256)

    def execute_with_internal_authorization(
        self,
        *,
        authorization: object,
        runner: "AllEvidenceFusionRunner",
    ) -> ApprovedHierarchicalDiscoveryBatchResult:
        """Execute once using provider-bound authority, without a user digest."""

        from .production_stage1_hierarchy_handoff import (
            AuthenticatedProductionStage1HierarchyExecutionAuthorization,
        )
        from .production_role_neutral_stage2_handoff import (
            AuthenticatedRoleNeutralHierarchyExecutionAuthorization,
        )

        if type(authorization) not in {
            AuthenticatedProductionStage1HierarchyExecutionAuthorization,
            AuthenticatedRoleNeutralHierarchyExecutionAuthorization,
        }:
            raise TypeError(
                "production hierarchy execution requires one exact typed "
                "authorization from a registered provider"
            )
        result = authorization._execute_for_prepared_batch(
            prepared_batch=self,
            runner=runner,
        )
        return result


class AllEvidenceFusionRunner:
    """Run fold-local remote fusion and deterministic structured estimation."""

    def __init__(
        self,
        *,
        dataset_path: Path | str,
        legacy_handoff_path: Path | str | None,
        tfidf_handoff_path: Path | str | None,
        output_dir: Path | str,
        fusion_agent: Callable[[Any], Mapping[str, Any]] | Any | None,
        extraction_provider: Any,
        review_agent: Callable[[Any], Mapping[str, Any]] | Any | None = None,
        review_spent_evidence_provider: ReviewSpentEvidenceProvider | None = None,
        review_partition_provider: ReviewPartitionProvider | None = None,
        review_gate_source_provider: (
            ReviewGateSourceProvider
            | BindableReviewGateSourceProvider
            | GateOnlyReviewNumericalProvider
            | None
        ) = None,
        review_gate_feature_bank_provider: (
            ReviewGateFeatureBankProvider
            | BindableReviewGateFeatureBankProvider
            | GateOnlyReviewNumericalProvider
            | None
        ) = None,
        final_upstream_producer: FinalUpstreamProducer | None = None,
        raw_final_upstream_producer: FinalContextFitUpstreamProducer | None = None,
        coordinate_preserving_nuisance_view_names: Sequence[str] | None = None,
        final_causal_forest_backend: FinalCausalForestBackend | None = None,
        config: AllEvidenceFusionRunnerConfig = AllEvidenceFusionRunnerConfig(),
        candidate_pool_paths: Mapping[int, Path | str] | None = None,
        query_evidence_by_fold: Mapping[int, QueryEvidenceArtifact] | None = None,
        tfidf_orphan_artifacts_by_fold: (
            Mapping[
                int,
                TfidfOrphanNgramArtifact | Mapping[str, Any] | Path | str,
            ]
            | None
        ) = None,
        legacy_primary_predictions_path: Path | str | None = None,
        cache_overlay: FrozenExtractionCacheOverlay | None = None,
        tfidf_validator: Callable[..., Mapping[str, Any]] | None = None,
        hierarchical_discovery_runner: MetadataJsonDiscoveryJobRunner | None = None,
        hierarchical_discovery_config: HierarchicalDiscoveryConfig | None = None,
        hierarchical_discovery_job_cache_root: Path | str | None = None,
        hierarchical_discovery_job_cache_config: (
            HierarchicalDiscoveryJobCacheConfig | None
        ) = None,
        first_untouched_gate_preparation_bounds: (
            FirstUntouchedGatePreparationBounds | None
        ) = None,
        hierarchical_discovery_approved_batch_sha256: str | None = None,
        hierarchical_review_evidence_policy: FrozenReviewEvidencePolicyBinding | None = None,
        hierarchical_preparation_dir: Path | str | None = None,
        hierarchical_max_atoms_per_chunk: int = (DEFAULT_MAX_ATOMS_PER_ARCHITECTURE_CHUNK),
        hierarchical_max_bytes_per_chunk: int = (DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK),
        hierarchical_max_semantic_member_ids_per_chunk: int = (
            DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK
        ),
        reference_only_stage1_provider: Any | None = None,
        reference_only_stage1_runtime_binding: Any | None = None,
        reference_only_numerical_bank: Any | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path).resolve()
        self.reference_only_stage1_provider = reference_only_stage1_provider
        self.reference_only_stage1_runtime_binding = (
            reference_only_stage1_runtime_binding
        )
        self.reference_only_numerical_bank = reference_only_numerical_bank
        direct_values = (
            self.reference_only_stage1_provider,
            self.reference_only_stage1_runtime_binding,
            self.reference_only_numerical_bank,
        )
        self.reference_only_stage1_mode = any(
            value is not None for value in direct_values
        )
        if self.reference_only_stage1_mode and not all(
            value is not None for value in direct_values
        ):
            raise ValueError(
                "reference-only Stage 1 provider, runtime binding, and numerical "
                "bank must be supplied together"
            )
        if self.reference_only_stage1_mode:
            if legacy_handoff_path is not None or tfidf_handoff_path is not None:
                raise ValueError(
                    "reference-only Stage 2 forbids legacy and TF-IDF handoff paths"
                )
            self.legacy_handoff_path = None
            self.tfidf_handoff_path = None
        else:
            if legacy_handoff_path is None or tfidf_handoff_path is None:
                raise ValueError(
                    "historical Stage 2 requires both legacy and TF-IDF handoff paths"
                )
            self.legacy_handoff_path = Path(legacy_handoff_path).resolve()
            self.tfidf_handoff_path = Path(tfidf_handoff_path).resolve()
        self.output_dir = Path(output_dir)
        self.fusion_agent = fusion_agent
        self.extraction_provider = extraction_provider
        self.review_agent = review_agent
        self.review_spent_evidence_provider = review_spent_evidence_provider
        self.review_partition_provider = review_partition_provider
        self.review_gate_source_provider = review_gate_source_provider
        self.review_gate_feature_bank_provider = review_gate_feature_bank_provider
        self.final_upstream_producer = final_upstream_producer
        self.raw_final_upstream_producer = raw_final_upstream_producer
        self.coordinate_preserving_nuisance_view_names = (
            None
            if coordinate_preserving_nuisance_view_names is None
            else tuple(str(value).strip() for value in coordinate_preserving_nuisance_view_names)
        )
        self.config = config
        self.gate_only_reference_review = (
            self.config.upstream_review_policy
            == GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY
        )
        if self.reference_only_stage1_mode:
            from .direct_upstream_numerical_reference_bank import (
                AuthenticatedRoleNeutralDirectNumericalBank,
            )
            from .production_role_neutral_stage2_handoff import (
                AuthenticatedRoleNeutralStage2Provider,
                AuthenticatedRoleNeutralStage2RuntimeBinding,
                validate_authenticated_role_neutral_stage2_runtime_binding,
            )

            if type(self.reference_only_stage1_provider) is not (
                AuthenticatedRoleNeutralStage2Provider
            ):
                raise TypeError(
                    "reference-only Stage 2 requires the exact authenticated "
                    "role-neutral provider"
                )
            if type(self.reference_only_stage1_runtime_binding) is not (
                AuthenticatedRoleNeutralStage2RuntimeBinding
            ):
                raise TypeError(
                    "reference-only Stage 2 requires the exact provider-issued "
                    "runtime binding"
                )
            if type(self.reference_only_numerical_bank) is not (
                AuthenticatedRoleNeutralDirectNumericalBank
            ):
                raise TypeError(
                    "reference-only Stage 2 requires the exact direct numerical bank"
                )
            plan = self.reference_only_stage1_provider.authenticated_scope_plan()
            direct_runtime_payload = dict(
                validate_authenticated_role_neutral_stage2_runtime_binding(
                    self.reference_only_stage1_runtime_binding,
                    expected_plan_scientific_content_sha256=(
                        plan.scientific_content_sha256
                    ),
                    expected_source_execution_content_sha256=(
                        self.reference_only_numerical_bank.manifest[
                            "source_execution_content_sha256"
                        ]
                    ),
                )
            )
            if (
                self.reference_only_numerical_bank.plan is not plan
                or self.reference_only_numerical_bank.identity()[
                    "plan_scientific_content_sha256"
                ]
                != plan.scientific_content_sha256
                or direct_runtime_payload["provider_identity_sha256"]
                != self.reference_only_stage1_provider.identity()[
                    "identity_sha256"
                ]
            ):
                raise ValueError(
                    "reference-only Stage 2 inputs belong to different "
                    "authenticated Stage 1 graphs"
                )
            self.reference_only_stage1_runtime_payload = direct_runtime_payload
        else:
            self.reference_only_stage1_runtime_payload = None
        self.hierarchical_discovery_runner = hierarchical_discovery_runner
        self.hierarchical_discovery_config = hierarchical_discovery_config
        self.hierarchical_discovery_job_cache_root = (
            None
            if hierarchical_discovery_job_cache_root is None
            else Path(hierarchical_discovery_job_cache_root).resolve()
        )
        self.hierarchical_discovery_job_cache_config = (
            hierarchical_discovery_job_cache_config
        )
        self.first_untouched_gate_preparation_bounds = (
            first_untouched_gate_preparation_bounds
        )
        self.hierarchical_discovery_approved_batch_sha256 = (
            None
            if hierarchical_discovery_approved_batch_sha256 is None
            else str(hierarchical_discovery_approved_batch_sha256).strip().lower()
        )
        self.hierarchical_review_evidence_policy = hierarchical_review_evidence_policy
        self.hierarchical_preparation_dir = (
            None
            if hierarchical_preparation_dir is None
            else Path(hierarchical_preparation_dir).resolve()
        )
        self.hierarchical_max_atoms_per_chunk = hierarchical_max_atoms_per_chunk
        self.hierarchical_max_bytes_per_chunk = hierarchical_max_bytes_per_chunk
        self.hierarchical_max_semantic_member_ids_per_chunk = (
            hierarchical_max_semantic_member_ids_per_chunk
        )
        for label in (
            "hierarchical_max_atoms_per_chunk",
            "hierarchical_max_bytes_per_chunk",
            "hierarchical_max_semantic_member_ids_per_chunk",
        ):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.hierarchical_discovery_enabled = self.hierarchical_discovery_runner is not None
        hierarchical_values = {
            "hierarchical_discovery_config": self.hierarchical_discovery_config,
            "hierarchical_discovery_job_cache_root": (self.hierarchical_discovery_job_cache_root),
            "hierarchical_discovery_job_cache_config": (
                self.hierarchical_discovery_job_cache_config
            ),
            "first_untouched_gate_preparation_bounds": (
                self.first_untouched_gate_preparation_bounds
            ),
            "hierarchical_discovery_approved_batch_sha256": (
                self.hierarchical_discovery_approved_batch_sha256
            ),
            "hierarchical_review_evidence_policy": self.hierarchical_review_evidence_policy,
            "hierarchical_preparation_dir": self.hierarchical_preparation_dir,
        }
        if not self.hierarchical_discovery_enabled:
            unexpected = sorted(
                name for name, value in hierarchical_values.items() if value is not None
            )
            if unexpected:
                raise ValueError(
                    "hierarchical discovery options require hierarchical_discovery_runner: "
                    f"{unexpected}"
                )
            if self.fusion_agent is None:
                raise ValueError("legacy initial discovery requires fusion_agent")
        else:
            if self.fusion_agent is not None:
                raise ValueError(
                    "hierarchical and legacy initial discovery agents are mutually exclusive"
                )
            if not isinstance(self.hierarchical_discovery_runner, MetadataJsonDiscoveryJobRunner):
                raise TypeError(
                    "hierarchical_discovery_runner must expose identity, run_json, and "
                    "execution_metadata"
                )
            if not isinstance(self.hierarchical_discovery_config, HierarchicalDiscoveryConfig):
                raise TypeError("hierarchical_discovery_config must be HierarchicalDiscoveryConfig")
            if self.hierarchical_discovery_job_cache_root is None:
                raise ValueError(
                    "hierarchical discovery requires an explicit stable job cache root"
                )
            if not isinstance(
                self.hierarchical_discovery_job_cache_config,
                HierarchicalDiscoveryJobCacheConfig,
            ):
                raise TypeError(
                    "hierarchical discovery requires a typed "
                    "hierarchical_discovery_job_cache_config"
                )
            if not isinstance(
                self.first_untouched_gate_preparation_bounds,
                FirstUntouchedGatePreparationBounds,
            ):
                raise TypeError(
                    "hierarchical discovery requires typed "
                    "first_untouched_gate_preparation_bounds"
                )
            if not isinstance(
                self.hierarchical_review_evidence_policy,
                FrozenReviewEvidencePolicyBinding,
            ):
                raise TypeError(
                    "hierarchical_review_evidence_policy must be "
                    "FrozenReviewEvidencePolicyBinding"
                )
            self.hierarchical_review_evidence_policy.validate_authentication()
            if (
                self.hierarchical_review_evidence_policy.adaptive_config().max_operations
                != self.config.post_extraction_review_max_operations
            ):
                raise ValueError(
                    "hierarchical adaptive max_operations must equal the runner review bound"
                )
            adaptive_chunk_limits = self.hierarchical_review_evidence_policy.adaptive_config()
            expected_chunk_limits = {
                "max_atoms_per_chunk": self.hierarchical_max_atoms_per_chunk,
                "max_bytes_per_chunk": self.hierarchical_max_bytes_per_chunk,
                "max_semantic_member_ids_per_chunk": (
                    self.hierarchical_max_semantic_member_ids_per_chunk
                ),
            }
            observed_chunk_limits = {
                key: getattr(adaptive_chunk_limits, key) for key in expected_chunk_limits
            }
            if observed_chunk_limits != expected_chunk_limits:
                raise ValueError(
                    "initial and adaptive hierarchical architecture chunk limits must match; "
                    f"initial={expected_chunk_limits}, adaptive={observed_chunk_limits}"
                )
            if (
                self.hierarchical_discovery_config.max_semantic_member_ids_per_chunk
                != self.hierarchical_max_semantic_member_ids_per_chunk
            ):
                raise ValueError(
                    "hierarchy config and initial architecture chunk plan must use the same "
                    "max_semantic_member_ids_per_chunk"
                )
            if self.hierarchical_preparation_dir is None:
                raise ValueError(
                    "hierarchical discovery requires a separate hierarchical_preparation_dir"
                )
            output_resolved = self.output_dir.resolve()
            preparation = self.hierarchical_preparation_dir
            if (
                preparation == output_resolved
                or preparation.is_relative_to(output_resolved)
                or output_resolved.is_relative_to(preparation)
            ):
                raise ValueError(
                    "hierarchical_preparation_dir and final output_dir must be separate "
                    "non-nested directories"
                )
            cache_root = self.hierarchical_discovery_job_cache_root
            if not cache_root.is_relative_to(preparation):
                raise ValueError(
                    "hierarchical job cache root must be contained by the preparation directory"
                )
            if (
                self.hierarchical_discovery_approved_batch_sha256 is not None
                and _SHA256.fullmatch(self.hierarchical_discovery_approved_batch_sha256) is None
            ):
                raise ValueError(
                    "hierarchical_discovery_approved_batch_sha256 must be a lowercase SHA-256"
                )
            if self.config.post_extraction_review_rounds < 1:
                raise ValueError(
                    "hierarchical production discovery requires at least one untouched review gate"
                )
            shared_gate_provider = self.review_gate_source_provider
            if (
                shared_gate_provider is None
                or shared_gate_provider is not self.review_gate_feature_bank_provider
            ):
                raise ValueError(
                    "hierarchical production discovery requires one shared "
                    "source/feature gate provider"
                )
            if self.gate_only_reference_review:
                if not callable(getattr(shared_gate_provider, "get_gate_only_view", None)):
                    raise ValueError(
                        "gate-only hierarchical review requires a prefit cumulative "
                        "get_gate_only_view provider"
                    )
                if callable(getattr(shared_gate_provider, "bind_fold", None)):
                    raise ValueError(
                        "gate-only hierarchical review rejects bind_fold providers"
                    )
            elif not callable(getattr(shared_gate_provider, "bind_fold", None)):
                raise ValueError(
                    "conditional hierarchical review requires one shared bindable "
                    "source/feature gate provider"
                )
            if self.review_spent_evidence_provider is None:
                raise ValueError(
                    "hierarchical production discovery requires spent-only Stage-1 evidence"
                )
            if (
                self.hierarchical_discovery_config.max_integrated_features
                > self.config.max_candidates
            ):
                raise ValueError(
                    "hierarchical max_integrated_features cannot exceed runner max_candidates"
                )
        if (
            self.raw_final_upstream_producer is not None
            and type(self.raw_final_upstream_producer) is not FinalContextFitUpstreamProducer
        ):
            raise TypeError(
                "raw_final_upstream_producer must be the exact "
                "FinalContextFitUpstreamProducer runtime object"
            )
        if self.raw_final_upstream_producer is not None and self.final_upstream_producer is None:
            raise ValueError(
                "the exact raw final upstream runtime requires a final package producer"
            )
        if (
            self.config.require_final_causal_forest
            and self.raw_final_upstream_producer is None
            and not self.reference_only_stage1_mode
        ):
            raise ValueError(
                "the required final causal forest needs the exact raw "
                "FinalContextFitUpstreamProducer runtime"
            )
        if (
            final_causal_forest_backend is not None
            and self.raw_final_upstream_producer is None
            and not self.reference_only_stage1_mode
        ):
            raise ValueError(
                "a final causal-forest backend may be injected only with the exact raw runtime"
            )
        if (
            self.coordinate_preserving_nuisance_view_names is not None
            and self.raw_final_upstream_producer is None
        ):
            raise ValueError(
                "coordinate-preserving nuisance views require the exact raw final runtime"
            )
        self.final_causal_forest_backend_was_injected = (
            final_causal_forest_backend is not None
            and type(final_causal_forest_backend) is not FixedCausalForestHeadBackend
        )
        self.final_causal_forest_backend = (
            final_causal_forest_backend
            if final_causal_forest_backend is not None
            else (
                FixedCausalForestHeadBackend(random_state=int(self.config.random_state))
                if (
                    self.raw_final_upstream_producer is not None
                    or self.reference_only_stage1_mode
                )
                else None
            )
        )
        if not self.hierarchical_discovery_enabled:
            _validate_fusion_reasoning_configuration(self.fusion_agent, self.config)
        if self.config.post_extraction_review_rounds > 0:
            if self.review_agent is None:
                raise ValueError(
                    "post-extraction review is enabled but no base review_agent was injected"
                )
            if self.review_spent_evidence_provider is None:
                raise ValueError(
                    "post-extraction review requires a context-fit spent-only evidence "
                    "provider so initial selection and later proposals cannot inspect "
                    "future review gates"
                )
            _validate_fusion_reasoning_configuration(self.review_agent, self.config)
            contract_local = getattr(
                self.extraction_provider,
                "adaptive_review_contract_local_extraction",
                None,
            )
            if not callable(contract_local):
                raise ValueError(
                    "adaptive post-extraction review requires the extraction provider "
                    "to declare adaptive_review_contract_local_extraction()"
                )
            if not bool(contract_local()):
                raise ValueError(
                    "adaptive post-extraction review requires contract-local extraction "
                    "semantics; configure max_variables_per_extraction_request=1 so "
                    "selective re-extraction, gate extraction, and post-freeze extraction "
                    "use the same request group"
                )
        if self.config.require_review_source_signals and (self.review_gate_source_provider is None):
            raise ValueError(
                "review source signals are required but no gate-local provider was injected"
            )
        if self.config.require_review_feature_banks and (
            self.review_gate_feature_bank_provider is None
        ):
            raise ValueError(
                "review feature banks are required but no gate-local provider was injected"
            )
        if (
            self.gate_only_reference_review
            and self.config.post_extraction_review_rounds > 0
            and self.review_partition_provider is None
        ):
            raise ValueError(
                "gate-only review requires authenticated precommitted review partitions"
            )
        if (
            self.config.require_review_feature_banks
            and self.review_partition_provider is None
            and not self.gate_only_reference_review
            and not callable(getattr(self.review_gate_feature_bank_provider, "bind_fold", None))
        ):
            raise ValueError(
                "required precomputed review feature banks need authenticated exact review "
                "partitions; context-fit providers must expose bind_fold()"
            )
        if (
            self.config.post_extraction_review_rounds > 1
            and self.review_gate_feature_bank_provider is not None
            and not self.gate_only_reference_review
            and not callable(getattr(self.review_gate_feature_bank_provider, "bind_fold", None))
        ):
            raise ValueError(
                "multi-round adaptive review requires sequential context-fit feature banks; "
                "a precomputed OOF bank may be used only for the final gate of one round"
            )
        if (
            self.config.require_final_upstream_inputs
            or self.config.require_final_upstream_neural_query_inputs
        ) and self.final_upstream_producer is None and not self.reference_only_stage1_mode:
            raise ValueError(
                "final upstream model inputs are required but no post-registry producer "
                "was injected"
            )
        if self.final_upstream_producer is not None and not callable(
            getattr(self.final_upstream_producer, "produce", None)
        ):
            raise TypeError("final_upstream_producer must expose produce()")
        self.review_spent_evidence_provider_identity = _review_provider_identity(
            self.review_spent_evidence_provider,
            label="review_spent_evidence_provider",
        )
        if (
            self.config.require_neural_query_moments
            and self.config.post_extraction_review_rounds > 0
        ):
            assert self.review_spent_evidence_provider_identity is not None
            spent_identity = self.review_spent_evidence_provider_identity["identity"]
            declared_families = (
                spent_identity.get("required_source_families")
                if isinstance(spent_identity, Mapping)
                else None
            )
            if (
                isinstance(declared_families, (str, bytes, Mapping))
                or not isinstance(declared_families, Sequence)
                or NEURAL_QUERY_MOMENTS not in {str(value).strip() for value in declared_families}
            ):
                raise ValueError(
                    "adaptive required neural query moments need a spent discovery "
                    "provider identity that requires neural_query_moments"
                )
        self.review_partition_provider_identity = _review_provider_identity(
            self.review_partition_provider,
            label="review_partition_provider",
        )
        self.review_gate_source_provider_identity = _review_provider_identity(
            self.review_gate_source_provider,
            label="review_gate_source_provider",
        )
        self.review_gate_feature_bank_provider_identity = _review_provider_identity(
            self.review_gate_feature_bank_provider,
            label="review_gate_feature_bank_provider",
        )
        self.final_upstream_producer_identity = _review_provider_identity(
            self.final_upstream_producer,
            label="final_upstream_producer",
        )
        self.raw_final_upstream_producer_identity = _review_provider_identity(
            self.raw_final_upstream_producer,
            label="raw_final_upstream_producer",
        )
        self.final_causal_forest_backend_identity = _review_provider_identity(
            self.final_causal_forest_backend,
            label="final_causal_forest_backend",
        )
        self.coordinate_preserving_nuisance_contract_sha256: str | None = None
        self.coordinate_preserving_producer_precommit_sha256: str | None = None
        if self.raw_final_upstream_producer is not None:
            raw_digest = self._assert_raw_final_upstream_producer_identity()
            if self._assert_final_upstream_producer_identity() != raw_digest:
                raise ValueError(
                    "the final package producer is not identity-bound to the exact raw runtime"
                )
            if self.coordinate_preserving_nuisance_view_names is not None:
                self.coordinate_preserving_nuisance_contract_sha256 = (
                    coordinate_preserving_nuisance_contract_sha256(
                        self.coordinate_preserving_nuisance_view_names
                    )
                )
                self.coordinate_preserving_producer_precommit_sha256 = (
                    precommit_runtime_producer_identity_sha256(self.raw_final_upstream_producer)
                )
                if self.coordinate_preserving_producer_precommit_sha256 != raw_digest:
                    raise ValueError(
                        "coordinate-preserving producer precommit differs from the runner "
                        "runtime identity"
                    )
        self.candidate_pool_paths = {
            int(key): Path(value) for key, value in (candidate_pool_paths or {}).items()
        }
        config_query_artifacts = dict(config.neural_query_moment_artifacts_by_fold)
        injected_query_artifacts = {
            int(key): _normalize_query_evidence_artifact_registration(value)
            for key, value in (query_evidence_by_fold or {}).items()
        }
        for fold, artifact in injected_query_artifacts.items():
            if fold < 1 or artifact.outer_fold != fold:
                raise ValueError(
                    "injected neural query evidence fold does not match its registry key"
                )
        overlapping_query_folds = set(config_query_artifacts) & set(injected_query_artifacts)
        if overlapping_query_folds:
            raise ValueError(
                "neural query evidence was registered in both RunnerConfig and the "
                f"legacy constructor argument for folds {sorted(overlapping_query_folds)}"
            )
        self.query_evidence_by_fold = {
            **config_query_artifacts,
            **injected_query_artifacts,
        }
        self.tfidf_orphan_artifacts_by_fold = {
            int(key): _normalize_orphan_artifact_registration(value)
            for key, value in (tfidf_orphan_artifacts_by_fold or {}).items()
        }
        if self.config.require_tfidf_orphan_ngrams and not (
            self.config.include_tfidf_orphan_ngrams
        ):
            raise ValueError(
                "require_tfidf_orphan_ngrams cannot be enabled when the source is disabled"
            )
        self.legacy_primary_predictions_path = (
            None
            if legacy_primary_predictions_path is None
            else Path(legacy_primary_predictions_path).resolve()
        )
        self.cache_overlay = cache_overlay
        self.cache_overlay_identity = _review_provider_identity(
            self.cache_overlay,
            label="cache_overlay",
        )
        self.tfidf_validator = tfidf_validator
        if self.reference_only_stage1_mode:
            if (
                self.review_spent_evidence_provider
                is not self.reference_only_stage1_provider
                or self.review_partition_provider
                is not self.reference_only_stage1_provider
                or self.review_gate_source_provider
                is not self.reference_only_numerical_bank
                or self.review_gate_feature_bank_provider
                is not self.reference_only_numerical_bank
            ):
                raise ValueError(
                    "reference-only Stage 2 must use its authenticated provider "
                    "for spent evidence/partitions and its direct numerical bank "
                    "for both gate channels"
                )
            if (
                self.final_upstream_producer is not None
                or self.raw_final_upstream_producer is not None
                or self.legacy_primary_predictions_path is not None
                or self.tfidf_validator is not None
                or self.candidate_pool_paths
                or self.query_evidence_by_fold
                or self.tfidf_orphan_artifacts_by_fold
            ):
                raise ValueError(
                    "reference-only Stage 2 forbids historical final producers, "
                    "split artifacts, TF-IDF validators, and auxiliary legacy "
                    "evidence registrations"
                )
            if not self.gate_only_reference_review:
                raise ValueError(
                    "reference-only Stage 2 requires the explicit gate-only "
                    "upstream review policy"
                )

    @classmethod
    def from_reference_only_role_neutral_stage1(
        cls,
        *,
        handoff: Any,
        direct_inputs: Any,
        options: Any,
        endpoint: str,
    ) -> "AllEvidenceFusionRunner":
        """Construct the portable runtime through its production client factory."""

        from .production_stage1_hierarchy_one_shot import (
            _construct_reference_only_role_neutral_stage2_runner,
        )

        runner = _construct_reference_only_role_neutral_stage2_runner(
            runner_type=cls,
            handoff=handoff,
            direct_inputs=direct_inputs,
            options=options,
            endpoint=endpoint,
        )
        if type(runner) is not cls:
            raise TypeError(
                "reference-only Stage 2 factory returned a substituted runner"
            )
        return runner

    def _reference_only_outer_split_rows(
        self,
    ) -> tuple[tuple[int, ...], dict[int, dict[str, tuple[int, ...]]]]:
        """Return plan-derived direct folds after rechecking retained identities."""

        if not self.reference_only_stage1_mode:
            raise RuntimeError("runner is not in reference-only Stage 1 mode")
        provider = self.reference_only_stage1_provider
        current_provider_identity = provider.identity()
        if (
            self.reference_only_stage1_runtime_payload is None
            or current_provider_identity["identity_sha256"]
            != self.reference_only_stage1_runtime_payload[
                "provider_identity_sha256"
            ]
        ):
            raise RuntimeError(
                "reference-only Stage 1 provider identity changed after construction"
            )
        assignments = provider.get_outer_fold_assignments()
        folds = tuple(sorted(assignments))
        if not folds or folds != tuple(range(1, len(folds) + 1)):
            raise ValueError(
                "reference-only outer folds must be complete one-based contiguous folds"
            )
        rows: dict[int, dict[str, tuple[int, ...]]] = {}
        for outer_fold in folds:
            assignment = assignments[outer_fold]
            if set(assignment) != {"fit_row_ids", "heldout_row_ids"}:
                raise ValueError(
                    "reference-only outer-fold assignment schema changed"
                )
            fit_rows = tuple(map(int, assignment["fit_row_ids"]))
            heldout_rows = tuple(map(int, assignment["heldout_row_ids"]))
            if (
                not fit_rows
                or not heldout_rows
                or len(fit_rows) != len(set(fit_rows))
                or len(heldout_rows) != len(set(heldout_rows))
                or set(fit_rows) & set(heldout_rows)
            ):
                raise ValueError(
                    "reference-only outer-fold row assignments are malformed"
                )
            rows[outer_fold] = {
                "fit_row_ids": fit_rows,
                "heldout_row_ids": heldout_rows,
            }
        return folds, rows

    def _reference_only_source_identity(self) -> Mapping[str, Any]:
        if not self.reference_only_stage1_mode:
            raise RuntimeError("runner is not in reference-only Stage 1 mode")
        provider_identity = self.reference_only_stage1_provider.identity()
        numerical_identity = self.reference_only_numerical_bank.identity()
        runtime_payload = dict(self.reference_only_stage1_runtime_payload or {})
        return {
            "mode": "authenticated_role_neutral_all_ten_reference_only_v1",
            "provider_identity_sha256": provider_identity["identity_sha256"],
            "plan_scientific_content_sha256": runtime_payload[
                "plan_scientific_content_sha256"
            ],
            "source_execution_content_sha256": runtime_payload[
                "source_execution_content_sha256"
            ],
            "runtime_binding_content_sha256": runtime_payload["content_sha256"],
            "prepared_projection_binding_content_sha256": runtime_payload[
                "prepared_projection_binding_content_sha256"
            ],
            "prepared_cohort_artifact_sha256": runtime_payload[
                "runner_dataset_artifact_sha256"
            ],
            "row_map_sha256": runtime_payload["row_map_sha256"],
            "direct_numerical_bank_manifest_content_sha256": numerical_identity[
                "manifest_content_sha256"
            ],
            "legacy_stage1_loader_invoked": False,
            "tfidf_handoff_loader_invoked": False,
            "independent_stage1_refit_performed": False,
            "text_truncation_applied": False,
        }

    def run_reference_only_role_neutral_one_shot(
        self,
        *,
        handoff: Any,
    ) -> AllEvidenceFusionRunResult:
        """Prepare, authorize, and execute one direct run without a user digest."""

        from .production_role_neutral_stage2_handoff import (
            ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
            authorize_reference_only_role_neutral_hierarchy_execution,
        )

        if not self.reference_only_stage1_mode:
            raise RuntimeError(
                "runner was not constructed for reference-only Stage 1"
            )
        if (
            getattr(handoff, "handoff_kind", None)
            != ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
            or getattr(handoff, "stage2_provider", None)
            is not self.reference_only_stage1_provider
            or getattr(handoff, "legacy_bundle_build_invoked", None) is not False
        ):
            raise ValueError(
                "direct one-shot handoff differs from the retained Stage 1 provider"
            )
        if self.hierarchical_discovery_approved_batch_sha256 is not None:
            raise ValueError(
                "direct one-shot execution cannot accept a caller-supplied digest"
            )
        prepared = self.prepare_hierarchical_discovery_batch()
        authorization = (
            authorize_reference_only_role_neutral_hierarchy_execution(
                provider=self.reference_only_stage1_provider,
                runtime_binding=self.reference_only_stage1_runtime_binding,
                prepared_batch=prepared,
                runner=self,
            )
        )
        return self.run(
            prepared_hierarchical_batch=prepared,
            hierarchy_execution_authorization=authorization,
        )

    def _assert_raw_final_upstream_producer_identity(self) -> str:
        if (
            self.raw_final_upstream_producer is None
            or self.raw_final_upstream_producer_identity is None
            or type(self.raw_final_upstream_producer) is not FinalContextFitUpstreamProducer
        ):
            raise RuntimeError("no exact raw final upstream runtime is active")
        current = _review_provider_identity(
            self.raw_final_upstream_producer,
            label="raw_final_upstream_producer",
        )
        if current != self.raw_final_upstream_producer_identity:
            raise ValueError("raw final upstream runtime identity changed after construction")
        digest = str(current["identity_sha256"])
        if not _SHA256.fullmatch(digest):
            raise RuntimeError("raw final upstream runtime identity digest is invalid")
        return digest

    def _assert_final_upstream_producer_identity(self) -> str:
        if self.final_upstream_producer is None or self.final_upstream_producer_identity is None:
            raise RuntimeError("no final upstream producer is active")
        current = _review_provider_identity(
            self.final_upstream_producer,
            label="final_upstream_producer",
        )
        if current != self.final_upstream_producer_identity:
            raise ValueError("final upstream producer identity changed after runner construction")
        digest = str(current["identity_sha256"])
        if not _SHA256.fullmatch(digest):
            raise RuntimeError("final upstream producer identity digest is invalid")
        package_identity = getattr(
            self.final_upstream_producer,
            "authenticated_package_producer_identity_sha256",
            None,
        )
        if callable(package_identity):
            identity = current.get("identity")
            if (
                type(self.final_upstream_producer) is not AuthenticatedFinalContextFitCacheOverlay
                or not isinstance(identity, Mapping)
                or identity.get("producer") != FINAL_CONTEXT_FIT_CACHE_OVERLAY_ID
            ):
                raise RuntimeError(
                    "only the authenticated final cache overlay may delegate package identity"
                )
            delegated = str(package_identity())
            if (
                not _SHA256.fullmatch(delegated)
                or identity.get("package_producer_identity_sha256") != delegated
                or identity.get("delegate_producer_identity_sha256") != delegated
            ):
                raise RuntimeError(
                    "final cache overlay package identity is not audit-bound to its delegate"
                )
            return delegated
        return digest

    def prepare_hierarchical_discovery_batch(
        self,
    ) -> PreparedHierarchicalDiscoveryBatch:
        """Prepare every fold and one inspectable batch without running discovery.

        This method is the public cross-process approval seam.  It may perform
        local spent-context Stage-1 fitting and authenticate the first
        label-free review-gate cache, but it never consults the hierarchical
        JSON-job cache and never invokes ``hierarchical_discovery_runner``.
        Every artifact it creates lives below ``hierarchical_preparation_dir``.
        """

        if not self.hierarchical_discovery_enabled:
            raise RuntimeError("hierarchical discovery is not configured")
        assert self.hierarchical_discovery_runner is not None
        assert self.hierarchical_discovery_config is not None
        assert self.hierarchical_discovery_job_cache_root is not None
        assert self.hierarchical_review_evidence_policy is not None
        assert self.hierarchical_preparation_dir is not None

        runner_records_before = tuple(
            json.loads(_canonical_json(row))
            for row in self.hierarchical_discovery_runner.execution_metadata
        )
        data, dataset_sha256 = _load_sanitized_dataset_snapshot(
            self.dataset_path,
            text_column=self.config.text_column,
            treatment_column=self.config.treatment_column,
            outcome_column=self.config.outcome_column,
        )
        external_validation: Mapping[str, Any] | None = None
        legacy = None
        tfidf = None
        reference_source: Mapping[str, Any] | None = None
        if self.reference_only_stage1_mode:
            folds, split_rows = self._reference_only_outer_split_rows()
            reference_source = self._reference_only_source_identity()
            if dataset_sha256 != reference_source[
                "prepared_cohort_artifact_sha256"
            ]:
                raise ValueError(
                    "runner dataset differs from the provider-authenticated "
                    "prepared cohort artifact"
                )
        else:
            assert self.tfidf_handoff_path is not None
            assert self.legacy_handoff_path is not None
            if self.tfidf_validator is not None:
                external_validation = self.tfidf_validator(
                    dataset=data.drop(columns=["_oci_row_id"]),
                    handoff_path=self.tfidf_handoff_path,
                )
                if (
                    not isinstance(external_validation, Mapping)
                    or external_validation.get("status") != "passed"
                ):
                    raise RuntimeError(
                        "external TF-IDF handoff validation did not pass"
                    )
            legacy = load_legacy_full_outer_evidence(self.legacy_handoff_path)
            tfidf = load_resealed_tfidf_handoff(
                self.tfidf_handoff_path,
                dataset_row_count=len(data),
                require_registry_seal=self.config.require_registry_seal,
            )
            folds = tuple(sorted(tfidf.full_rows_by_outer_fold))
            if set(legacy.rows_by_outer_fold) != set(folds):
                raise ValueError(
                    "legacy and TF-IDF full-outer fold sets do not match exactly"
                )
            split_rows = {
                outer_fold: {
                    "fit_row_ids": tuple(
                        map(
                            int,
                            tfidf.full_rows_by_outer_fold[outer_fold][
                                "fit_row_ids"
                            ],
                        )
                    ),
                    "heldout_row_ids": tuple(
                        map(
                            int,
                            tfidf.full_rows_by_outer_fold[outer_fold][
                                "heldout_row_ids"
                            ],
                        )
                    ),
                }
                for outer_fold in folds
            }
        if folds != tuple(range(1, len(folds) + 1)):
            raise ValueError(
                "hierarchical batch approval requires complete one-based contiguous folds"
            )
        unexpected = (
            (set(self.candidate_pool_paths) | set(self.query_evidence_by_fold))
            | set(self.tfidf_orphan_artifacts_by_fold)
        ) - set(folds)
        if unexpected:
            raise ValueError(
                "candidate/query/orphan artifact registry contains an unknown outer fold"
            )
        legacy_split_audit: Mapping[str, Any] | None = None
        if self.legacy_primary_predictions_path is not None:
            assert tfidf is not None
            legacy_heldout, legacy_primary_sha256 = (
                _load_outer_splits_from_primary_predictions_snapshot(
                    self.legacy_primary_predictions_path,
                    dataset_row_count=len(data),
                )
            )
            if set(legacy_heldout) != set(folds):
                raise ValueError("legacy primary-prediction fold set does not match TF-IDF")
            for outer_fold in folds:
                tfidf_heldout = set(
                    map(
                        int,
                        tfidf.full_rows_by_outer_fold[outer_fold]["heldout_row_ids"],
                    )
                )
                if set(legacy_heldout[outer_fold]) != tfidf_heldout:
                    raise ValueError(
                        f"legacy and TF-IDF heldout splits differ for fold {outer_fold}"
                    )
            legacy_split_audit = {
                "path": str(self.legacy_primary_predictions_path),
                "sha256": legacy_primary_sha256,
                "matches_tfidf_outer_splits": True,
            }

        shared_provider = self.review_gate_source_provider
        if shared_provider is None or shared_provider is not self.review_gate_feature_bank_provider:
            raise RuntimeError("hierarchical shared gate provider changed after construction")
        current_shared_identity = _review_provider_identity(
            shared_provider,
            label="hierarchical_shared_gate_provider",
        )
        if (
            current_shared_identity != self.review_gate_source_provider_identity
            or current_shared_identity != self.review_gate_feature_bank_provider_identity
        ):
            raise RuntimeError("hierarchical shared gate provider identity changed")
        current_spent_identity = _review_provider_identity(
            self.review_spent_evidence_provider,
            label="review_spent_evidence_provider",
        )
        if current_spent_identity != self.review_spent_evidence_provider_identity:
            raise RuntimeError("spent-only evidence provider identity changed")
        semantic_compatibility_identity = current_spent_projection_compatibility_identity()
        family_explanations = production_stage1_family_explanations()
        htr_prompt_preflight = _build_htr_stage2_prompt_preflight(
            provider=self.review_spent_evidence_provider,
            max_atoms_per_chunk=self.hierarchical_max_atoms_per_chunk,
            max_bytes_per_chunk=self.hierarchical_max_bytes_per_chunk,
            max_semantic_member_ids_per_chunk=(
                self.hierarchical_max_semantic_member_ids_per_chunk
            ),
            family_explanation=family_explanations[HTR_NEURAL],
            wire_budget=self.hierarchical_discovery_config.wire_budget,
        )
        if (
            htr_prompt_preflight is not None
            and htr_prompt_preflight.get(
                "stage2_endpoint_launch_allowed"
            )
            is not True
        ):
            raise RuntimeError(
                "HTR Stage 2 prompt preflight still plans at least "
                "100,000 interpretation calls; report the remaining "
                "semantic redundancy before endpoint use"
            )

        preparation_dir = self.hierarchical_preparation_dir
        companion_body = {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "post_extraction_review_providers": {
                "calibrated_gate_sources": self.review_gate_source_provider_identity,
                "role_aware_gate_feature_banks": (self.review_gate_feature_bank_provider_identity),
            },
            "final_upstream_model_inputs": {
                "producer": self.final_upstream_producer_identity,
            },
        }
        companion_identity_sha256 = _content_sha256(companion_body)
        context_fit_overlay_companion_path = (
            preparation_dir / "context_fit_overlay_companions" / f"{companion_identity_sha256}.json"
        )
        _write_immutable_json(
            context_fit_overlay_companion_path,
            companion_body,
            schema=RUNNER_SCHEMA_VERSION,
        )
        context_fit_overlay_companion_sha256 = sha256_file(context_fit_overlay_companion_path)
        htr_prompt_preflight_path: Path | None = None
        htr_prompt_preflight_file_sha256: str | None = None
        if htr_prompt_preflight is not None:
            htr_prompt_preflight_path = (
                preparation_dir / "htr_stage2_prompt_preflight.json"
            )
            htr_prompt_preflight_file_sha256 = _write_immutable_json(
                htr_prompt_preflight_path,
                htr_prompt_preflight,
                schema="production_htr_stage2_prompt_preflight_envelope_v1",
            )
        input_manifest_body = {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "preparation_schema_version": (HIERARCHICAL_DISCOVERY_PREPARATION_INPUT_SCHEMA_VERSION),
            "runner_implementation_file_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "dataset": {
                "path": str(self.dataset_path),
                "sha256": dataset_sha256,
                "row_count": len(data),
                "text_fingerprint": ordered_dataset_text_fingerprint(
                    data,
                    text_column=self.config.text_column,
                ),
            },
            "stage1_reference_source": reference_source,
            "legacy_handoff": (
                None
                if legacy is None
                else {
                    "path": legacy.artifact_path,
                    "sha256": legacy.artifact_sha256,
                    "primary_split_audit": legacy_split_audit,
                }
            ),
            "tfidf_handoff": (
                None
                if tfidf is None
                else {
                    "path": tfidf.artifact_path,
                    "sha256": tfidf.artifact_sha256,
                    "split_registry_content_hash": (
                        tfidf.split_registry_content_hash
                    ),
                    "external_validation": external_validation,
                }
            ),
            "outer_folds": [
                {
                    "outer_fold": outer_fold,
                    "fit_row_fingerprint": row_set_fingerprint(
                        split_rows[outer_fold]["fit_row_ids"]
                    ),
                    "heldout_row_fingerprint": row_set_fingerprint(
                        split_rows[outer_fold]["heldout_row_ids"]
                    ),
                    "fit_row_count": len(
                        split_rows[outer_fold]["fit_row_ids"]
                    ),
                    "heldout_row_count": len(
                        split_rows[outer_fold]["heldout_row_ids"]
                    ),
                }
                for outer_fold in folds
            ],
            "effective_runner_config": asdict(self.config),
            "hierarchical_discovery_config": self.hierarchical_discovery_config.as_dict(),
            "hierarchical_architecture_chunk_limits": {
                "max_atoms_per_chunk": self.hierarchical_max_atoms_per_chunk,
                "max_bytes_per_chunk": self.hierarchical_max_bytes_per_chunk,
                "max_semantic_member_ids_per_chunk": (
                    self.hierarchical_max_semantic_member_ids_per_chunk
                ),
            },
            "hierarchical_runner_identity": json.loads(
                _canonical_json(self.hierarchical_discovery_runner.identity())
            ),
            "production_family_explanations": production_stage1_family_explanations(),
            "semantic_retrieval_compatibility": semantic_compatibility_identity,
            "spent_evidence_provider": self.review_spent_evidence_provider_identity,
            "htr_stage2_prompt_preflight": htr_prompt_preflight,
            "htr_stage2_prompt_preflight_path": (
                None
                if htr_prompt_preflight_path is None
                else str(htr_prompt_preflight_path)
            ),
            "htr_stage2_prompt_preflight_file_sha256": (
                htr_prompt_preflight_file_sha256
            ),
            "htr_stage2_endpoint_contacted_during_preflight": False,
            "shared_first_gate_provider": current_shared_identity,
            "frozen_review_evidence_policy": (self.hierarchical_review_evidence_policy.as_dict()),
            "final_upstream_producer": self.final_upstream_producer_identity,
            "raw_final_upstream_producer": self.raw_final_upstream_producer_identity,
            "final_causal_forest_backend": self.final_causal_forest_backend_identity,
            "extraction_cache_overlay": self.cache_overlay_identity,
            "hierarchical_preparation_dir": str(preparation_dir),
            "hierarchical_job_cache_root": str(self.hierarchical_discovery_job_cache_root),
            "context_fit_overlay_companion": {
                "path": str(context_fit_overlay_companion_path),
                "sha256": context_fit_overlay_companion_sha256,
                "overlay_compatible_closed_run_attestation": True,
            },
            "outer_heldout_labels_used": False,
            "hierarchy_runner_calls_during_preparation": 0,
        }
        input_manifest_path = preparation_dir / "immutable_hierarchical_input_manifest.json"
        input_manifest_sha256 = _write_immutable_json(
            input_manifest_path,
            input_manifest_body,
            schema=HIERARCHICAL_DISCOVERY_PREPARATION_INPUT_SCHEMA_VERSION,
        )

        label_free = data[["_oci_row_id", self.config.text_column]].copy()
        indexed = data.set_index("_oci_row_id", drop=False)
        prepared_folds: list[PreparedHierarchicalDiscoveryFold] = []
        ordered_agents: list[OrderedFoldDiscoveryAgent] = []
        first_gate_intent_index_entries: list[dict[str, Any]] = []
        for outer_fold in folds:
            fold_dir = preparation_dir / f"outer_fold_{outer_fold:03d}"
            full = split_rows[outer_fold]
            train_ids = tuple(map(int, full["fit_row_ids"]))
            outer_train = indexed.loc[list(train_ids)].reset_index(drop=True)
            schedule = self._review_schedule(
                outer_train=outer_train,
                outer_fold=outer_fold,
            )
            evidence_inputs, evidence_audit, prefit_catalog = self._spent_evidence_inputs(
                data=data,
                schedule=schedule,
                spent_fold_ids=schedule.initial_spent_fold_ids,
                outer_fold=outer_fold,
                review_round=0,
            )
            catalog = (
                prefit_catalog
                if prefit_catalog is not None
                else build_role_neutral_evidence_catalog(evidence_inputs)
            )
            chunk_plan = build_complete_architecture_chunks(
                catalog,
                max_atoms_per_chunk=self.hierarchical_max_atoms_per_chunk,
                max_bytes_per_chunk=self.hierarchical_max_bytes_per_chunk,
                max_semantic_member_ids_per_chunk=(
                    self.hierarchical_max_semantic_member_ids_per_chunk
                ),
            )

            spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
            first_gate_ids = schedule.row_ids((schedule.gate_fold_ids[0],))
            first_gate_intent: FirstGateMaterializationIntent | None = None
            reference_contract: (
                AuthenticatedReferenceOnlyDirectNumericalContract | None
            ) = None
            if self.gate_only_reference_review:
                prepare_reference_contract = getattr(
                    shared_provider,
                    "prepare_hierarchy_gate_contract",
                    None,
                )
                if not callable(prepare_reference_contract):
                    raise RuntimeError(
                        "gate-only numerical provider lacks its hierarchy "
                        "reference-contract boundary"
                    )
                reference_contract = prepare_reference_contract(
                    outer_fold=outer_fold,
                    context_epoch=0,
                    exact_spent_row_ids=spent_ids,
                    exact_gate_row_ids=first_gate_ids,
                    catalog=catalog,
                )
                if type(reference_contract) is not (
                    AuthenticatedReferenceOnlyDirectNumericalContract
                ):
                    raise TypeError(
                        "gate-only hierarchy returned a substituted numerical contract"
                    )
                reference_contract.verify(catalog=catalog)
                numerical_contract = reference_contract
                numerical_contract_kind = (
                    "authenticated_reference_only_direct_numerical_contract"
                )
                first_gate_intent_path = (
                    fold_dir / "reference_only_direct_numerical_contract.json"
                )
                numerical_source_cache_key = None
            else:
                spent_frame = indexed.loc[list(spent_ids)]
                fold_by_row = {
                    row_id: fold_id
                    for fold_id, rows in schedule.row_ids_by_fold.items()
                    for row_id in rows
                }
                spent_texts = tuple(
                    spent_frame[self.config.text_column].astype(str).tolist()
                )
                gate_texts = tuple(
                    label_free.set_index("_oci_row_id", drop=False)
                    .loc[list(first_gate_ids)][self.config.text_column]
                    .astype(str)
                    .tolist()
                )
                first_gate_intent = prepare_first_gate_materialization_intent(
                    outer_fold=outer_fold,
                    initial_spent_row_ids=spent_ids,
                    initial_spent_texts=spent_texts,
                    initial_spent_treatment=spent_frame[
                        self.config.treatment_column
                    ].to_numpy(dtype=float),
                    initial_spent_outcome=spent_frame[
                        self.config.outcome_column
                    ].to_numpy(dtype=float),
                    initial_spent_inner_fold_ids=tuple(
                        fold_by_row[row_id] for row_id in spent_ids
                    ),
                    first_gate_row_ids=first_gate_ids,
                    first_gate_texts=gate_texts,
                    catalog=catalog,
                    provider=shared_provider,
                )
                first_gate_intent.verify()
                numerical_contract = first_gate_intent
                numerical_contract_kind = "first_gate_materialization_intent"
                first_gate_intent_path = (
                    fold_dir / "first_gate_materialization_intent.json"
                )
                numerical_source_cache_key = first_gate_intent.body[
                    "source_cache_key"
                ]
            first_gate_intent_file_sha256 = _write_immutable_plain_json(
                first_gate_intent_path,
                numerical_contract.as_dict(),
            )
            first_gate_intent_index_entries.append(
                {
                    "outer_fold": outer_fold,
                    "contract_kind": numerical_contract_kind,
                    "contract_path": str(first_gate_intent_path),
                    "contract_file_sha256": first_gate_intent_file_sha256,
                    "contract_content_sha256": (
                        numerical_contract.content_sha256
                    ),
                    "source_cache_key": numerical_source_cache_key,
                    "materialization_deferred_until_after_exact_approval_and_proposal_freeze": (
                        not self.gate_only_reference_review
                    ),
                    "already_fit_stage1_reference_projection": (
                        self.gate_only_reference_review
                    ),
                    **(
                        {}
                        if self.gate_only_reference_review
                        else {
                            "intent_path": str(first_gate_intent_path),
                            "intent_file_sha256": (
                                first_gate_intent_file_sha256
                            ),
                            "intent_content_sha256": (
                                numerical_contract.content_sha256
                            ),
                        }
                    ),
                }
            )
            catalog_path = fold_dir / "role_neutral_evidence_catalog.json"
            catalog_file_sha256 = _write_immutable_json(
                catalog_path,
                catalog.as_dict(),
                schema="role_neutral_evidence_catalog_preparation_envelope_v1",
            )
            chunk_path = fold_dir / "architecture_chunk_plan.json"
            chunk_file_sha256 = _write_immutable_json(
                chunk_path,
                chunk_plan.as_dict(),
                schema="architecture_chunk_plan_preparation_envelope_v1",
            )
            job_cache = AuthenticatedHierarchicalDiscoveryJobCache(
                root=(self.hierarchical_discovery_job_cache_root / f"outer_fold_{outer_fold:03d}"),
                config=self.hierarchical_discovery_job_cache_config,
            )
            agent = ApprovedHierarchicalDiscoveryAgent(
                catalog=catalog,
                chunk_plan=chunk_plan,
                family_explanations=family_explanations,
                first_gate_materialization_intent=first_gate_intent,
                reference_only_direct_numerical_contract=reference_contract,
                runner=self.hierarchical_discovery_runner,
                config=self.hierarchical_discovery_config,
                job_cache=job_cache,
            )
            wrapper_path = fold_dir / "approved_hierarchical_wrapper_precommit.json"
            wrapper_file_sha256 = _write_immutable_json(
                wrapper_path,
                {
                    "approval_sha256": agent.precommit.approval_sha256,
                    "packet": agent.precommit.packet,
                },
                schema=HIERARCHICAL_DISCOVERY_BATCH_PACKET_SCHEMA_VERSION,
            )
            fold_manifest_path = fold_dir / "immutable_fold_preparation.json"
            _write_immutable_json(
                fold_manifest_path,
                {
                    "outer_fold": outer_fold,
                    "schedule_audit": schedule.audit,
                    "initial_spent_evidence_audit": evidence_audit,
                    "catalog_path": str(catalog_path),
                    "catalog_envelope_content_sha256": catalog_file_sha256,
                    "catalog_sha256": catalog.catalog_sha256,
                    "chunk_plan_path": str(chunk_path),
                    "chunk_plan_envelope_content_sha256": chunk_file_sha256,
                    "chunk_plan_sha256": chunk_plan.plan_sha256,
                    "first_gate_direct_numerical_contract_kind": (
                        numerical_contract_kind
                    ),
                    "first_gate_direct_numerical_contract_path": str(
                        first_gate_intent_path
                    ),
                    "first_gate_direct_numerical_contract_file_sha256": (
                        first_gate_intent_file_sha256
                    ),
                    "first_gate_direct_numerical_contract_content_sha256": (
                        numerical_contract.content_sha256
                    ),
                    **(
                        {}
                        if self.gate_only_reference_review
                        else {
                            "first_gate_materialization_intent_path": str(
                                first_gate_intent_path
                            ),
                            "first_gate_materialization_intent_file_sha256": (
                                first_gate_intent_file_sha256
                            ),
                            "first_gate_materialization_intent_content_sha256": (
                                numerical_contract.content_sha256
                            ),
                        }
                    ),
                    "first_gate_source_cache_key_precommitted": (
                        numerical_source_cache_key
                    ),
                    "first_gate_cache_materialized_before_discovery": (
                        self.gate_only_reference_review
                    ),
                    "first_gate_cache_materialization_deferred_until_after_exact_approval": (
                        not self.gate_only_reference_review
                    ),
                    "first_gate_cache_materialization_deferred_until_after_proposal_freeze": (
                        not self.gate_only_reference_review
                    ),
                    "first_gate_reference_projection_fit_or_refit_performed": False,
                    "first_gate_labels_supplied_to_provider": False,
                    "first_gate_views_exposed_to_discovery": False,
                    "wrapper_precommit_path": str(wrapper_path),
                    "wrapper_precommit_envelope_content_sha256": wrapper_file_sha256,
                    "wrapper_approval_sha256": agent.precommit.approval_sha256,
                    "hierarchy_runner_calls_during_preparation": 0,
                },
                schema=HIERARCHICAL_DISCOVERY_PREPARATION_FOLD_SCHEMA_VERSION,
            )
            prepared = PreparedHierarchicalDiscoveryFold(
                outer_fold=outer_fold,
                schedule=schedule,
                evidence_inputs=evidence_inputs,
                initial_spent_evidence_audit=evidence_audit,
                catalog=catalog,
                chunk_plan=chunk_plan,
                first_gate_materialization_intent=first_gate_intent,
                reference_only_direct_numerical_contract=reference_contract,
                first_gate_materialization_intent_path=first_gate_intent_path,
                first_gate_materialization_intent_file_sha256=(first_gate_intent_file_sha256),
                agent=agent,
                preparation_manifest_path=fold_manifest_path,
            )
            prepared_folds.append(prepared)
            ordered_agents.append(OrderedFoldDiscoveryAgent(outer_fold=outer_fold, agent=agent))

        _assert_semantic_compatibility_identity_current(semantic_compatibility_identity)
        first_gate_intent_index_content = {
            "schema_version": (
                "first_gate_direct_numerical_contract_index_v2"
                if self.gate_only_reference_review
                else "first_gate_materialization_intent_index_v1"
            ),
            "entries": first_gate_intent_index_entries,
            "all_first_gate_materializations_deferred": (
                not self.gate_only_reference_review
            ),
            "all_gate_only_references_already_fit": (
                self.gate_only_reference_review
            ),
            "gate_labels_in_intents": False,
        }
        first_gate_intent_index_payload = {
            **first_gate_intent_index_content,
            "content_sha256": _content_sha256(first_gate_intent_index_content),
        }
        first_gate_intent_index_identity_sha256 = _content_sha256(first_gate_intent_index_payload)
        first_gate_materialization_intent_index_path = (
            preparation_dir
            / "first_gate_materialization_intent_indexes"
            / f"{first_gate_intent_index_identity_sha256}.json"
        )
        first_gate_materialization_intent_index_sha256 = _write_immutable_plain_json(
            first_gate_materialization_intent_index_path,
            first_gate_intent_index_payload,
        )
        coordinator = ApprovedHierarchicalDiscoveryBatchCoordinator(
            input_manifest_sha256=input_manifest_sha256,
            fold_agents=tuple(ordered_agents),
            frozen_review_evidence_policy=self.hierarchical_review_evidence_policy,
        )
        batch_packet_path = preparation_dir / "approved_hierarchical_batch_precommit.json"
        _write_immutable_json(
            batch_packet_path,
            {
                "approval_sha256": coordinator.precommit.approval_sha256,
                "packet": coordinator.precommit.packet,
            },
            schema=HIERARCHICAL_DISCOVERY_BATCH_PACKET_SCHEMA_VERSION,
        )
        runner_records_after = tuple(
            json.loads(_canonical_json(row))
            for row in self.hierarchical_discovery_runner.execution_metadata
        )
        if runner_records_after != runner_records_before:
            raise RuntimeError(
                "hierarchical discovery runner metadata changed during local preparation"
            )
        prepared_batch = PreparedHierarchicalDiscoveryBatch(
            coordinator=coordinator,
            folds=tuple(prepared_folds),
            input_manifest_sha256=input_manifest_sha256,
            input_manifest_path=input_manifest_path,
            context_fit_overlay_companion_path=(context_fit_overlay_companion_path),
            context_fit_overlay_companion_sha256=(context_fit_overlay_companion_sha256),
            first_gate_materialization_intent_index_path=(
                first_gate_materialization_intent_index_path
            ),
            first_gate_materialization_intent_index_sha256=(
                first_gate_materialization_intent_index_sha256
            ),
            batch_packet_path=batch_packet_path,
            dataset_sha256=dataset_sha256,
        )
        _issue_prepared_hierarchy_capability(prepared_batch)
        return prepared_batch

    def _adapt_orphan_ngram_evidence(
        self,
        *,
        outer_fold: int,
        full_row: Mapping[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if not self.config.include_tfidf_orphan_ngrams:
            return None, {
                "status": "disabled_by_config",
                "outer_fold": int(outer_fold),
                "model_inference_performed": False,
            }
        adapter_config = self.config.orphan_ngram_adapter
        if not isinstance(adapter_config, OrphanNgramEvidenceAdapterConfig):
            raise RuntimeError("enabled TF-IDF orphan adapter lost its explicit config")

        original_reference = _effect_ngram_registration(full_row)
        registry_entry = self.tfidf_orphan_artifacts_by_fold.get(int(outer_fold))
        if original_reference in (None, "") and registry_entry is None:
            if self.config.require_tfidf_orphan_ngrams:
                raise ValueError(
                    f"TF-IDF orphan evidence is required but unregistered for fold {outer_fold}"
                )
            return None, {
                "status": "not_available_no_effect_ngram_registration",
                "outer_fold": int(outer_fold),
                "model_inference_performed": False,
            }

        adapter_row = full_row
        effect_path: Path | str | None = None
        expected_sha256: str | None = None
        original_path, original_inline_hash = _registered_reference_path_and_hash(
            original_reference
        )
        resolution = {
            "mode": "resealed_handoff_reference",
            "handoff_reference_present": original_path is not None,
            "handoff_declared_sha256_present": original_inline_hash is not None,
            "explicit_registry_used": False,
        }
        if registry_entry is not None:
            if (
                registry_entry.artifact_sha256 is not None
                and original_inline_hash is not None
                and registry_entry.artifact_sha256 != original_inline_hash
            ):
                raise ValueError(
                    f"TF-IDF orphan registry and handoff SHA-256 disagree for fold {outer_fold}"
                )
            expected_sha256 = registry_entry.artifact_sha256 or original_inline_hash
            effect_path = registry_entry.path
            # The explicit registry is a trusted, fold-keyed repair for a stale
            # absolute/relative path.  Patch a detached copy so the adapter can
            # still enforce its exact path/hash binding without mutating the
            # authenticated source handoff record.
            adapter_row = json.loads(_canonical_json(full_row))
            discovery = adapter_row.setdefault("discovery", {})
            artifacts = discovery.setdefault("artifacts", {})
            ngram_scores = artifacts.setdefault("ngram_scores", {})
            replacement: dict[str, Any] = {"path": str(registry_entry.path)}
            if expected_sha256 is not None:
                replacement["sha256"] = expected_sha256
            ngram_scores["effect"] = replacement
            artifacts.setdefault("topic_score_tests", None)
            resolution = {
                "mode": "explicit_per_fold_registry_override",
                "handoff_reference_present": original_path is not None,
                "handoff_declared_sha256_present": original_inline_hash is not None,
                "explicit_registry_used": True,
                "registry_path": str(registry_entry.path),
                "registry_declared_sha256": registry_entry.artifact_sha256,
            }

        adapted = adapt_full_outer_orphan_ngram_evidence(
            adapter_row,
            effect_path,
            artifact_base_dir=self.tfidf_handoff_path.parent,
            expected_sha256=expected_sha256,
            config=adapter_config,
        )
        audit = adapted.audit
        audit["artifact_resolution"] = resolution
        # Artifact paths/hashes belong in the immutable fold manifest, not in
        # the selector payload.  The compact fusion code ignores this field,
        # but removing it at the boundary also prevents source-directory names
        # from being mistaken for patient or oracle evidence.
        branch = adapted.branch
        branch.pop("source_artifact_audit", None)
        audit["source_artifact_audit_removed_before_fusion"] = True
        return branch, audit

    def _invoke_agent(self, request: Any) -> tuple[Mapping[str, Any], str]:
        if hasattr(self.fusion_agent, "propose"):
            raw = self.fusion_agent.propose(request.context())
        elif callable(self.fusion_agent):
            raw = self.fusion_agent(request)
        else:
            raise TypeError("fusion_agent must be callable or expose propose(context)")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("fusion agent returned invalid JSON") from exc
        if not isinstance(raw, Mapping):
            raise TypeError("fusion agent must return one JSON object")
        detached = json.loads(_canonical_json(raw))
        return detached, _content_sha256(detached)

    def _invoke_review_agent(
        self,
        context: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], str]:
        if self.review_agent is None:
            raise RuntimeError("post-extraction review agent is not configured")
        if hasattr(self.review_agent, "propose"):
            raw = self.review_agent.propose(json.loads(_canonical_json(context)))
        elif callable(self.review_agent):
            raw = self.review_agent(json.loads(_canonical_json(context)))
        else:
            raise TypeError("review_agent must be callable or expose propose(context)")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PostExtractionReviewResponseExhausted(
                    "post-extraction review agent returned invalid JSON"
                ) from exc
        if not isinstance(raw, Mapping):
            raise PostExtractionReviewResponseExhausted(
                "post-extraction review agent must return one JSON object"
            )
        detached = json.loads(_canonical_json(raw))
        return detached, _content_sha256(detached)

    def _validated_extraction_projection(
        self,
        extracted: pd.DataFrame,
        *,
        label_free: pd.DataFrame,
        specs: Sequence[Mapping[str, Any]],
        source: str,
    ) -> pd.DataFrame:
        if not isinstance(extracted, pd.DataFrame):
            raise TypeError(f"{source} must be a DataFrame")
        if len(extracted) != len(label_free):
            raise ValueError(f"{source} changed the dataset row count")
        if "_oci_row_id" not in extracted.columns or not np.array_equal(
            extracted["_oci_row_id"].to_numpy(),
            label_free["_oci_row_id"].to_numpy(),
        ):
            raise ValueError(f"{source} changed canonical row identity/order")
        _reject_forbidden_columns(extracted.columns, source=source)
        expected_feature_columns: list[str] = []
        for spec in specs:
            expected_feature_columns.extend(expected_extraction_columns(spec))
        missing = set(expected_feature_columns) - set(extracted.columns)
        if missing:
            raise ValueError(f"{source} is missing extraction columns: {sorted(missing)}")
        allowed = set(label_free.columns) | set(expected_feature_columns)
        unexpected = set(extracted.columns) - allowed
        if unexpected:
            raise ValueError(f"{source} returned unexpected columns: {sorted(unexpected)}")
        projected = label_free.copy()
        for column in expected_feature_columns:
            projected[column] = extracted[column].to_numpy(copy=True)
        return projected

    @staticmethod
    def _extraction_projection_sha256(
        extracted: pd.DataFrame,
        specs: Sequence[Mapping[str, Any]],
    ) -> str:
        """Hash exact ordered extracted values without persisting row-level content."""

        canonical_specs = [CandidateContract(spec).extraction_spec for spec in specs]
        columns = [
            "_oci_row_id",
            *(column for spec in canonical_specs for column in expected_extraction_columns(spec)),
        ]
        missing = set(columns) - set(extracted.columns)
        if missing:
            raise ValueError(
                f"cannot hash extraction projection with missing columns: {sorted(missing)}"
            )

        def scalar(value: Any) -> Any:
            if isinstance(value, np.generic):
                value = value.item()
            try:
                if pd.isna(value):
                    return None
            except (TypeError, ValueError):
                pass
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        digest = hashlib.sha256()
        digest.update(
            _canonical_json(
                {
                    "schema_version": ORDERED_EXTRACTION_PROJECTION_SHA256_VERSION,
                    "columns": columns,
                    "contract_sha256": [
                        extraction_contract_sha256(spec) for spec in canonical_specs
                    ],
                    "row_count": len(extracted),
                }
            ).encode("utf-8")
        )
        for row in extracted[columns].itertuples(index=False, name=None):
            digest.update(b"\n")
            digest.update(_canonical_json([scalar(value) for value in row]).encode("utf-8"))
        return digest.hexdigest()

    def _candidate_extraction_projection(
        self,
        *,
        label_free: pd.DataFrame,
        current_extracted: pd.DataFrame,
        current_specs: Sequence[Mapping[str, Any]],
        applied: Any,
        use_cache_overlay: bool = True,
    ) -> tuple[pd.DataFrame, Mapping[str, Any]]:
        candidate_specs = list(applied.specs)
        current_by_name = {str(spec["name"]): spec for spec in current_specs}
        reextract_specs = list(applied.reextract_specs)
        fresh: pd.DataFrame | None = None
        provider_audit: Mapping[str, Any] | None = None
        if reextract_specs:
            raw_fresh, provider_audit = self._extract(
                label_free,
                reextract_specs,
                use_cache_overlay=use_cache_overlay,
            )
            fresh = self._validated_extraction_projection(
                raw_fresh,
                label_free=label_free,
                specs=reextract_specs,
                source="selective post-extraction review output",
            )
        candidate = label_free.copy()
        reused_names: list[str] = []
        reextracted_names: list[str] = []
        for spec in candidate_specs:
            name = str(spec["name"])
            prior = current_by_name.get(name)
            reusable = bool(
                prior is not None
                and extraction_semantics_sha256(prior) == extraction_semantics_sha256(spec)
            )
            source_frame = current_extracted if reusable else fresh
            if source_frame is None:
                raise RuntimeError(
                    f"review candidate {name!r} changed extraction semantics without "
                    "a selective extraction result"
                )
            for column in expected_extraction_columns(spec):
                if column not in source_frame.columns:
                    raise ValueError(
                        f"review candidate source is missing required column {column!r}"
                    )
                candidate[column] = source_frame[column].to_numpy(copy=True)
            (reused_names if reusable else reextracted_names).append(name)
        audit = {
            "candidate_contract_count": len(candidate_specs),
            "selective_reextraction_spec_count": len(reextract_specs),
            "selective_reextraction_names": reextracted_names,
            "reused_extraction_names": reused_names,
            "role_only_changed_names": list(applied.role_only_changed_names),
            "removed_names": list(applied.removed_names),
            "added_names": list(applied.added_names),
            "provider_audit": provider_audit,
            "role_only_columns_reused_without_remote_extraction": bool(
                applied.role_only_changed_names
                and not set(applied.role_only_changed_names) & set(reextracted_names)
            ),
            "outer_heldout_labels_used": False,
            "cache_overlay_enabled_for_this_scope": bool(
                self.cache_overlay is not None and use_cache_overlay
            ),
        }
        return candidate, audit

    def _select_extraction_rows(
        self,
        extracted: pd.DataFrame,
        *,
        label_free: pd.DataFrame,
        specs: Sequence[Mapping[str, Any]],
        source: str,
    ) -> pd.DataFrame:
        """Project an accumulated extraction onto one exact sealed row scope."""

        if "_oci_row_id" not in extracted.columns:
            raise ValueError(f"{source} has no canonical row identity")
        if extracted["_oci_row_id"].duplicated().any():
            raise ValueError(f"{source} contains duplicate canonical row identities")
        exact_ids = tuple(map(int, label_free["_oci_row_id"].tolist()))
        indexed = extracted.set_index("_oci_row_id", drop=False)
        missing = set(exact_ids) - set(map(int, indexed.index))
        if missing:
            raise ValueError(f"{source} is missing requested extraction rows")
        selected = indexed.loc[list(exact_ids)].reset_index(drop=True)
        return self._validated_extraction_projection(
            selected,
            label_free=label_free,
            specs=specs,
            source=source,
        )

    def _combine_extraction_row_scopes(
        self,
        parts: Sequence[pd.DataFrame],
        *,
        label_free: pd.DataFrame,
        specs: Sequence[Mapping[str, Any]],
        source: str,
    ) -> pd.DataFrame:
        """Combine disjoint row-scoped projections in canonical target order."""

        if not parts:
            raise ValueError(f"{source} requires at least one extraction row scope")
        combined = pd.concat([part.copy() for part in parts], ignore_index=True)
        if "_oci_row_id" not in combined.columns or combined["_oci_row_id"].duplicated().any():
            raise ValueError(f"{source} row scopes overlap or lack canonical row identity")
        exact_ids = tuple(map(int, label_free["_oci_row_id"].tolist()))
        if set(map(int, combined["_oci_row_id"])) != set(exact_ids):
            raise ValueError(f"{source} does not cover the exact requested row scope")
        ordered = (
            combined.set_index("_oci_row_id", drop=False)
            .loc[list(exact_ids)]
            .reset_index(drop=True)
        )
        return self._validated_extraction_projection(
            ordered,
            label_free=label_free,
            specs=specs,
            source=source,
        )

    def _extract(
        self,
        label_free: pd.DataFrame,
        specs: Sequence[dict[str, Any]],
        *,
        use_cache_overlay: bool = True,
    ) -> tuple[pd.DataFrame, Mapping[str, Any]]:
        typed_specs = [ExplicitFeatureSpec(**spec) for spec in specs]
        if self.cache_overlay is not None and use_cache_overlay:
            extracted, report = self.cache_overlay.ensure_features(
                label_free,
                typed_specs,
                model_identity=self.config.extraction_model_identity,
                prompt_template_version=self.config.extraction_prompt_template_version,
                fallback_provider=self.extraction_provider,
            )
            return extracted, report.as_dict()
        provider = self.extraction_provider
        if hasattr(provider, "ensure_features"):
            extracted = provider.ensure_features(label_free.copy(), typed_specs)
        elif callable(provider):
            extracted = provider(label_free.copy(), typed_specs)
        else:
            raise TypeError("extraction_provider must be callable or expose ensure_features")
        if not isinstance(extracted, pd.DataFrame):
            raise TypeError("extraction provider must return a DataFrame")
        return (
            extracted,
            CacheOverlayReport(
                dataset_text_fingerprint=ordered_dataset_text_fingerprint(
                    label_free,
                    text_column=self.config.text_column,
                ),
                model_identity=self.config.extraction_model_identity,
                prompt_template_version=self.config.extraction_prompt_template_version,
                cache_hit_contract_hashes=(),
                cache_miss_contract_hashes=tuple(
                    extraction_contract_sha256(spec) for spec in specs
                ),
                authenticated_artifact_paths=(),
            ).as_dict(),
        )

    def _spent_evidence_inputs(
        self,
        *,
        data: pd.DataFrame,
        schedule: ReviewPartitionSchedule,
        spent_fold_ids: Sequence[int],
        outer_fold: int,
        review_round: int,
    ) -> tuple[
        tuple[FoldEvidenceInput, ...],
        Mapping[str, Any],
        RoleNeutralEvidenceCatalog | None,
    ]:
        """Materialize only context-fit evidence for one adaptive proposal."""

        provider = self.review_spent_evidence_provider
        if provider is None or self.review_spent_evidence_provider_identity is None:
            raise RuntimeError(
                "adaptive review cannot use full-outer discovery evidence; a spent-only "
                "evidence provider is required"
            )
        current_identity = _review_provider_identity(
            provider,
            label="review_spent_evidence_provider",
        )
        if current_identity != self.review_spent_evidence_provider_identity:
            raise RuntimeError("spent-only evidence provider identity changed during the run")

        consumer_review_round = int(review_round)
        context_epoch = _spent_evidence_context_epoch(schedule, spent_fold_ids)
        expected_context_epoch = 0 if consumer_review_round == 0 else consumer_review_round - 1
        if consumer_review_round < 0 or context_epoch != expected_context_epoch:
            raise ValueError(
                "spent-evidence context epoch must equal zero for the initial selector "
                "and consumer review_round - 1 for adaptive review"
            )

        spent_ids = schedule.row_ids(spent_fold_ids)
        spent_set = set(map(int, spent_fold_ids))
        sealed_fold_ids = tuple(
            fold_id for fold_id in schedule.row_ids_by_fold if int(fold_id) not in spent_set
        )
        sealed_ids = schedule.row_ids(sealed_fold_ids)
        if not sealed_ids:
            raise RuntimeError("adaptive review evidence requires at least one sealed gate")
        indexed = data.set_index("_oci_row_id", drop=False)
        spent_frame = indexed.loc[list(spent_ids)]
        spent_texts = tuple(spent_frame[self.config.text_column].astype(str).tolist())
        treatment = spent_frame[self.config.treatment_column].to_numpy(dtype=float).copy()
        outcome = spent_frame[self.config.outcome_column].to_numpy(dtype=float).copy()
        treatment.setflags(write=False)
        outcome.setflags(write=False)
        request_arguments = {
            "outer_fold": int(outer_fold),
            # Provider/cache APIs retain this legacy keyword, but its value is
            # the context epoch, not the reasoning-agent consumer round.
            "review_round": int(context_epoch),
            "exact_spent_row_ids": spent_ids,
            "exact_sealed_row_ids": sealed_ids,
            "spent_texts": spent_texts,
            "spent_treatment": treatment,
            "spent_outcome": outcome,
        }
        prefit_catalog_method = getattr(provider, "get_spent_evidence_catalog", None)
        if callable(prefit_catalog_method):
            catalog = prefit_catalog_method(**request_arguments)
            if not isinstance(catalog, RoleNeutralEvidenceCatalog):
                raise TypeError(
                    "prefit spent-evidence provider must return RoleNeutralEvidenceCatalog"
                )
            validate_role_neutral_catalog(catalog)
            expected_provenance = FoldEvidenceProvenance(
                outer_fold=int(outer_fold),
                train_row_ids=tuple(map(int, spent_ids)),
                heldout_row_ids=tuple(map(int, sealed_ids)),
                scope="inner_train",
                inner_fold=int(context_epoch) + 1,
                artifact_id=(f"production-stage1-hierarchy-{int(outer_fold)}-{int(context_epoch)}"),
            )
            if (
                catalog.outer_fold != int(outer_fold)
                or catalog.scope != "inner_train"
                or catalog.inner_fold != int(context_epoch) + 1
                or catalog.split_fingerprint != expected_provenance.split_fingerprint
            ):
                raise ValueError("prefit spent catalog changed its canonical row scope")
            family_counts = {
                family: len(catalog.family_atoms(family))
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            }
            if any(count < 1 for count in family_counts.values()):
                raise ValueError("prefit spent catalog must contain all ten architectures")
            audit = {
                "review_round": consumer_review_round,
                "consumer_review_round": consumer_review_round,
                "spent_evidence_context_epoch": int(context_epoch),
                "provider_review_round_argument": int(context_epoch),
                "consumed_gate_count_before_context_fit": int(context_epoch),
                "context_epoch_policy_version": (SPENT_EVIDENCE_CONTEXT_EPOCH_POLICY_VERSION),
                "spent_row_count": len(spent_ids),
                "sealed_row_count": len(sealed_ids),
                "spent_row_fingerprint": row_set_fingerprint(spent_ids),
                "sealed_row_fingerprint": row_set_fingerprint(sealed_ids),
                "provider_identity_sha256": self.review_spent_evidence_provider_identity[
                    "identity_sha256"
                ],
                "semantic_retrieval_compatibility": None,
                "source_kinds": list(catalog.audit["source_kinds"]),
                "family_atom_counts": family_counts,
                "prefit_cumulative_spent_catalog_used": True,
                "independent_runtime_stage1_refit_performed": False,
                "future_gate_text_or_labels_supplied_to_provider": False,
                "full_outer_discovery_evidence_used": False,
            }
            return (), json.loads(_canonical_json(audit)), catalog

        raw_inputs = provider.get_spent_evidence_inputs(
            **request_arguments,
        )
        if isinstance(raw_inputs, (str, bytes, Mapping)):
            raise TypeError("spent-only evidence provider must return a sequence of inputs")
        inputs = tuple(raw_inputs)
        if not inputs or not all(isinstance(item, FoldEvidenceInput) for item in inputs):
            raise TypeError(
                "spent-only evidence provider must return non-empty FoldEvidenceInput objects"
            )
        expected_inner_fold = int(context_epoch) + 1
        expected_spent = tuple(map(int, spent_ids))
        expected_sealed = tuple(map(int, sealed_ids))
        for item in inputs:
            provenance = item.provenance
            if provenance.outer_fold != int(outer_fold):
                raise ValueError("spent-only evidence changed the outer fold")
            if provenance.scope != "inner_train" or provenance.inner_fold != expected_inner_fold:
                raise ValueError(
                    "spent-only evidence must declare inner_train provenance bound to "
                    "spent_context_epoch + 1"
                )
            if tuple(map(int, provenance.train_row_ids)) != expected_spent:
                raise ValueError(
                    "spent-only evidence provenance must contain the exact spent row order"
                )
            if tuple(map(int, provenance.heldout_row_ids)) != expected_sealed:
                raise ValueError(
                    "spent-only evidence provenance must hold out every sealed row in order"
                )
        semantic_compatibility_audit: Mapping[str, Any] | None = None
        if self.hierarchical_discovery_enabled:
            compatibility = restore_current_spent_projection_semantic_retrieval_view(
                inputs,
                spent_evidence_provider=provider,
                outer_fold=int(outer_fold),
                review_round=int(context_epoch),
                exact_spent_row_ids=spent_ids,
                exact_sealed_row_ids=sealed_ids,
                spent_texts=spent_texts,
                spent_treatment=treatment,
                spent_outcome=outcome,
            )
            inputs = compatibility.evidence_inputs
            semantic_compatibility_audit = compatibility.audit
        audit = {
            "review_round": consumer_review_round,
            "consumer_review_round": consumer_review_round,
            "spent_evidence_context_epoch": int(context_epoch),
            "provider_review_round_argument": int(context_epoch),
            "consumed_gate_count_before_context_fit": int(context_epoch),
            "context_epoch_policy_version": (SPENT_EVIDENCE_CONTEXT_EPOCH_POLICY_VERSION),
            "spent_row_count": len(spent_ids),
            "sealed_row_count": len(sealed_ids),
            "spent_row_fingerprint": row_set_fingerprint(spent_ids),
            "sealed_row_fingerprint": row_set_fingerprint(sealed_ids),
            "provider_identity_sha256": self.review_spent_evidence_provider_identity[
                "identity_sha256"
            ],
            "semantic_retrieval_compatibility": semantic_compatibility_audit,
            "source_kinds": sorted(item.source_kind for item in inputs),
            "prefit_cumulative_spent_catalog_used": False,
            "independent_runtime_stage1_refit_performed": True,
            "future_gate_text_or_labels_supplied_to_provider": False,
            "full_outer_discovery_evidence_used": False,
        }
        return inputs, json.loads(_canonical_json(audit)), None

    def _spent_fusion_request(
        self,
        *,
        data: pd.DataFrame,
        schedule: ReviewPartitionSchedule,
        spent_fold_ids: Sequence[int],
        outer_fold: int,
        review_round: int,
        candidates: Sequence[CandidateContract] = (),
    ) -> tuple[AllEvidenceFusionRequest, Mapping[str, Any]]:
        inputs, audit, prefit_catalog = self._spent_evidence_inputs(
            data=data,
            schedule=schedule,
            spent_fold_ids=spent_fold_ids,
            outer_fold=outer_fold,
            review_round=review_round,
        )
        if prefit_catalog is not None:
            raise RuntimeError(
                "production prefit spent catalogs are consumed only by hierarchical discovery"
            )
        request = prepare_all_evidence_fusion(
            inputs,
            candidates=candidates,
            max_candidates=self.config.max_candidates,
        )
        return request, audit

    @staticmethod
    def _sanitize_spent_evidence_catalog(
        evidence_catalog: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep concept-level aggregates while removing row-level values and identities."""

        row_level = re.compile(
            r"(?:^|_)(?:record|patient|document|note|text|excerpt|chunk)s?(?:_|$)",
            flags=re.IGNORECASE,
        )
        raw_value = re.compile(
            r"(?:^|_)(?:raw|value|values|activation|activations|prediction|predictions)(?:_|$)",
            flags=re.IGNORECASE,
        )
        opaque_identity = re.compile(
            r"(?:^|_)(?:source|query|topic|cluster|view|model|artifact)_(?:id|ids|name|names)$",
            flags=re.IGNORECASE,
        )
        any_identifier = re.compile(r"(?:^|_)(?:id|ids)$", flags=re.IGNORECASE)
        safe_text_fields = {
            "kind",
            "term",
            "feature",
            "phrase",
            "ngram",
            "concept",
            "description",
            "role",
            "role_hint",
            "mechanical_role",
            "bank",
            "objective",
            "direction",
            "stage",
            "bow_model",
            "model_family",
            "contrast_family",
            "normalization_version",
        }
        safe_container_fields = {
            "rows",
            "terms",
            "topics",
            "clusters",
            "groups",
            "features",
            "ngrams",
            "summaries",
            "concept_scores",
            "top_terms",
            "top_ngrams",
            "top_contrastive_ngrams",
            "contrastive_ngrams",
            "topic_banks",
            "treatment",
            "outcome",
            "effect",
            "confounder",
            "confounders",
            "effect_modifier",
            "effect_modifiers",
            "metrics",
            "statistics",
            "support",
        }
        aggregate_metric = re.compile(
            r"(?:^|_)(?:score|contrast|count|fraction|rate|mean|median|std|range|loading|"
            r"correlation|loss|support|importance|rank|frequency|prevalence)(?:_|$)",
            flags=re.IGNORECASE,
        )
        identifier_like_phrase = re.compile(
            r"\b(?:patient|record|mrn|member|subject|row|id)\s*[:#=-]?\s*"
            r"(?:\d{4,}|[0-9a-f]{8,})\b",
            flags=re.IGNORECASE,
        )

        def safe_concept_phrase(value: Any) -> str | None:
            if not isinstance(value, str):
                return None
            phrase = " ".join(value.split())
            if (
                not phrase
                or identifier_like_phrase.search(phrase)
            ):
                return None
            return phrase

        def clean(value: Any, *, parent_key: str) -> Any:
            if isinstance(value, Mapping):
                result: dict[str, Any] = {}
                for raw_key, child in value.items():
                    key = str(raw_key)
                    lowered = key.strip().lower()
                    if (
                        row_level.search(key)
                        or raw_value.search(key)
                        or lowered in {"row_id", "row_ids"}
                    ):
                        continue
                    if opaque_identity.search(key):
                        result[f"{key}_sha256"] = _content_sha256(child)
                        continue
                    if any_identifier.search(key):
                        continue
                    if isinstance(child, str) and lowered not in safe_text_fields:
                        continue
                    if (
                        isinstance(child, (int, float, np.integer, np.floating))
                        and not isinstance(child, (bool, np.bool_))
                        and aggregate_metric.search(lowered) is None
                    ):
                        continue
                    if isinstance(child, (Mapping, list, tuple)) and (
                        lowered not in safe_container_fields
                    ):
                        continue
                    sanitized = clean(child, parent_key=key)
                    if sanitized not in (None, [], {}):
                        result[key] = sanitized
                return result
            if isinstance(value, (list, tuple)):
                # Numeric vectors can be row-level values in disguise. Scalar
                # aggregate statistics remain available through named fields.
                if value and all(
                    isinstance(item, (int, float, np.integer, np.floating))
                    and not isinstance(item, (bool, np.bool_))
                    for item in value
                ):
                    return None
                cleaned = [clean(item, parent_key=parent_key) for item in value]
                return [item for item in cleaned if item not in (None, [], {})]
            if isinstance(value, np.generic):
                return clean(value.item(), parent_key=parent_key)
            if value is None or isinstance(value, (bool, int)):
                return value
            if isinstance(value, float):
                return value if math.isfinite(value) else None
            if isinstance(value, str):
                if parent_key.strip().lower() in {
                    "term",
                    "feature",
                    "phrase",
                    "ngram",
                    "concept",
                    "summaries",
                }:
                    return safe_concept_phrase(value)
                return value
            return None

        sanitized: list[dict[str, Any]] = []
        for index, raw in enumerate(evidence_catalog, start=1):
            if not isinstance(raw, Mapping):
                raise TypeError("spent evidence catalog rows must be mappings")
            content = raw.get("content")
            if not isinstance(content, Mapping):
                raise ValueError("spent evidence catalog rows require mapping content")
            cleaned_content = clean(content, parent_key="content")
            if not isinstance(cleaned_content, Mapping):
                cleaned_content = {}
            sanitized.append(
                {
                    "evidence_id": f"evidence_{index:04d}",
                    "source_families": list(raw.get("source_families") or ()),
                    "role_hint": str(raw.get("role_hint") or ""),
                    "content": dict(cleaned_content),
                    "unsanitized_content_sha256": _content_sha256(content),
                }
            )
        return json.loads(_canonical_json(sanitized))

    def _build_sanitized_review_context(
        self,
        *,
        review_round: int,
        review_attempt: int,
        spent: ObservableCausalRows,
        spent_texts: Sequence[str],
        specs: Sequence[Mapping[str, Any]],
        evidence_catalog: Sequence[Mapping[str, Any]],
        spent_evidence_audit: Mapping[str, Any],
        feedback_diagnostics: Sequence[Mapping[str, Any]] = (),
        accepted_round_baseline_specs: Sequence[Mapping[str, Any]] = (),
        workspace_stage_history: Sequence[Mapping[str, Any]] = (),
        workspace_extraction_sha256: str | None = None,
        frozen_content_addressed_evidence: bool = False,
    ) -> Mapping[str, Any]:
        if int(spent_evidence_audit.get("consumer_review_round", -1)) != int(review_round):
            raise ValueError("spent-evidence consumer review round does not match review context")
        if (
            spent_evidence_audit.get("context_epoch_policy_version")
            != SPENT_EVIDENCE_CONTEXT_EPOCH_POLICY_VERSION
        ):
            raise ValueError("spent-evidence context epoch policy is missing or changed")
        expected_epoch = int(review_round) - 1
        if int(spent_evidence_audit.get("spent_evidence_context_epoch", -1)) != expected_epoch:
            raise ValueError(
                "spent-evidence context epoch does not match the consumer review round"
            )

        quality = build_extraction_quality_diagnostics(
            spent.extracted,
            specs,
            fold_ids=spent.inner_fold_ids or (),
            policy=(
                None
                if self.config.post_extraction_scientific_policy is None
                else self.config.post_extraction_scientific_policy.extraction_quality
            ),
        )
        quality_rows = list(quality["features"])
        grounding_start = len(quality_rows) + 1
        grounding = build_extraction_grounding_diagnostics(
            spent.extracted,
            spent_texts,
            specs,
            diagnostic_start=grounding_start,
            policy=(
                None
                if self.config.post_extraction_scientific_policy is None
                else self.config.post_extraction_scientific_policy.extraction_grounding
            ),
        )
        blocking_grounding_failures = {"alternative_category_only_value_support"}
        required_safety_remediation = [
            {
                "feature_name": str(row["feature_name"]),
                "diagnostic_id": str(row["diagnostic_id"]),
                "hard_failures": sorted(
                    blocking_grounding_failures.intersection(
                        set(map(str, row.get("hard_failures") or ()))
                    )
                ),
            }
            for row in grounding
            if blocking_grounding_failures.intersection(
                set(map(str, row.get("hard_failures") or ()))
            )
        ]
        redundancy_start = grounding_start + len(grounding)
        redundancy = build_redundancy_diagnostics(
            spent.extracted,
            specs,
            diagnostic_start=redundancy_start,
            policy=(
                None
                if self.config.post_extraction_scientific_policy is None
                else self.config.post_extraction_scientific_policy.extraction_redundancy
            ),
        )
        causal_start = redundancy_start + len(redundancy)
        causal = build_causal_review_diagnostics(
            spent,
            specs,
            config=self.config.post_extraction_review_config,
            diagnostic_start=causal_start,
        )
        diagnostics: list[Mapping[str, Any]] = [
            *quality_rows,
            *grounding,
            *redundancy,
            causal,
        ]
        existing_ids = collect_post_extraction_diagnostic_ids(diagnostics)
        next_diagnostic_number = (
            max(
                (int(value.rsplit("_", 1)[1]) for value in existing_ids),
                default=0,
            )
            + 1
        )
        for offset, raw_feedback in enumerate(feedback_diagnostics):
            if not isinstance(raw_feedback, Mapping) or "diagnostic_id" in raw_feedback:
                raise ValueError("review feedback must be an ID-free diagnostic mapping")
            feedback = json.loads(_canonical_json(dict(raw_feedback)))
            feedback["diagnostic_id"] = f"diagnostic_{next_diagnostic_number + offset:04d}"
            diagnostics.append(feedback)
        # Fail closed on collisions or malformed nested IDs before the context
        # is sent to the reasoning agent.
        collect_post_extraction_diagnostic_ids(diagnostics)
        if frozen_content_addressed_evidence:
            frozen_rows: list[dict[str, Any]] = []
            seen_frozen_ids: set[str] = set()
            for raw in evidence_catalog:
                if not isinstance(raw, Mapping) or set(raw) != {
                    "evidence_id",
                    "source_families",
                    "role_hint",
                    "content",
                }:
                    raise ValueError("frozen hierarchical review evidence has a wrong closed shape")
                evidence_id = str(raw["evidence_id"])
                if (
                    re.fullmatch(r"evidence_[0-9a-f]{64}", evidence_id) is None
                    or evidence_id in seen_frozen_ids
                ):
                    raise ValueError(
                        "frozen hierarchical evidence IDs must remain unique and "
                        "content-addressed"
                    )
                if not isinstance(raw["content"], Mapping):
                    raise TypeError("frozen hierarchical evidence content must be a mapping")
                seen_frozen_ids.add(evidence_id)
                frozen_rows.append(json.loads(_canonical_json(dict(raw))))
            sanitized_evidence = frozen_rows
        else:
            sanitized_evidence = self._sanitize_spent_evidence_catalog(evidence_catalog)
        if not sanitized_evidence:
            raise ValueError("spent-only evidence provider produced no review evidence")
        current_by_name = {
            str(spec["name"]): CandidateContract(spec).extraction_spec for spec in specs
        }
        for remediation in required_safety_remediation:
            feature_name = str(remediation["feature_name"])
            contract = current_by_name[feature_name]
            remediation["same_name_grounded_evidence_ids"] = sorted(
                str(evidence_row["evidence_id"])
                for evidence_row in sanitized_evidence
                if ground_evidence_to_extraction_contract(evidence_row, contract).supported
            )
            remediation["safe_fallback_action"] = "drop"
        canonical_baseline = [
            CandidateContract(spec).extraction_spec
            for spec in (accepted_round_baseline_specs or specs)
        ]
        if workspace_extraction_sha256 is None or not _SHA256.fullmatch(
            str(workspace_extraction_sha256)
        ):
            raise ValueError("review candidate workspace requires an exact extraction hash")
        context = {
            "prompt_version": POST_EXTRACTION_REVIEW_PROMPT_VERSION,
            "source_text_temporal_policy": source_text_temporal_policy_audit(),
            "review_round": int(review_round),
            "review_attempt": int(review_attempt),
            "maximum_quality_retries_per_gate": int(
                self.config.post_extraction_review_max_quality_retries
            ),
            "max_operations": int(self.config.post_extraction_review_max_operations),
            "max_contracts": int(self.config.max_candidates),
            "operation_apply_policy_version": (
                POST_EXTRACTION_REVIEW_OPERATION_APPLY_POLICY_VERSION
            ),
            "candidate_workspace_policy_version": (
                POST_EXTRACTION_REVIEW_CANDIDATE_WORKSPACE_POLICY_VERSION
            ),
            "current_contracts": [CandidateContract(spec).extraction_spec for spec in specs],
            "candidate_workspace": {
                "accepted_round_baseline_specs_sha256": _content_sha256(canonical_baseline),
                "workspace_specs_sha256": _content_sha256(
                    [CandidateContract(spec).extraction_spec for spec in specs]
                ),
                "workspace_extraction_sha256": str(workspace_extraction_sha256),
                "staged_attempt_count": len(workspace_stage_history),
                "staged_response_sha256s": [
                    str(row.get("response_sha256") or "") for row in workspace_stage_history
                ],
                "workspace_accepted": False,
                "same_gate_remains_sealed": True,
                "gate_rows_or_labels_available": False,
                "staging_requires_changed_contract_quality_pass": True,
                "staging_requires_strict_hard_failure_reduction": True,
                "final_gate_evaluation_is_atomic_against_accepted_baseline": True,
            },
            "required_safety_remediation": {
                "blocking_contract_count": len(required_safety_remediation),
                "blocking_contracts": required_safety_remediation,
                "hard_failure_policy": sorted(blocking_grounding_failures),
                "all_listed_contracts_must_be_resolved_before_gate": True,
                "partial_repairs_may_be_staged_without_gate_access": True,
                "computed_from_exact_spent_rows_only": True,
                "sealed_gate_used": False,
            },
            "diagnostics": diagnostics,
            "sanitized_evidence_catalog": sanitized_evidence,
            "evidence_sanitization": {
                "spent_only_source_blocks_retained": len(sanitized_evidence),
                "full_outer_discovery_blocks_available": False,
                "row_level_text_values_available": False,
                "row_level_numerical_values_available": False,
                "spent_note_text_used_only_for_local_aggregate_grounding": True,
                "grounding_diagnostic_version": EXTRACTION_GROUNDING_DIAGNOSTIC_VERSION,
                "raw_grounding_spans_or_matched_values_available": False,
                "source_and_query_identifiers_are_opaque_or_removed": True,
                "fresh_gate_derived_aggregates_available": False,
                "spent_aggregate_numerical_diagnostics_available": True,
                "hierarchical_frozen_accepted_support_active": bool(
                    frozen_content_addressed_evidence
                ),
                "original_content_addressed_evidence_ids_preserved": bool(
                    frozen_content_addressed_evidence
                ),
                "frozen_evidence_rows_changed_by_legacy_sanitizer": False,
                "sanitized_prior_gate_feedback_count": sum(
                    str(row.get("kind")) == "prior_gate_feedback" for row in feedback_diagnostics
                ),
                "sanitized_quality_retry_feedback_count": sum(
                    str(row.get("kind")) == "candidate_quality_retry_feedback"
                    for row in feedback_diagnostics
                ),
                "sanitized_ontology_retry_feedback_count": sum(
                    str(row.get("kind")) == "retained_registry_ontology_retry_feedback"
                    for row in feedback_diagnostics
                ),
            },
            "spent_evidence_provenance": {
                "provider_identity_sha256": spent_evidence_audit["provider_identity_sha256"],
                "consumer_review_round": spent_evidence_audit["consumer_review_round"],
                "spent_evidence_context_epoch": spent_evidence_audit[
                    "spent_evidence_context_epoch"
                ],
                "provider_review_round_argument": spent_evidence_audit[
                    "provider_review_round_argument"
                ],
                "consumed_gate_count_before_context_fit": spent_evidence_audit[
                    "consumed_gate_count_before_context_fit"
                ],
                "context_epoch_policy_version": spent_evidence_audit[
                    "context_epoch_policy_version"
                ],
                "spent_row_count": spent_evidence_audit["spent_row_count"],
                "sealed_row_count": spent_evidence_audit["sealed_row_count"],
                "source_kinds": spent_evidence_audit["source_kinds"],
                "future_gate_text_or_labels_supplied_to_provider": False,
                "full_outer_discovery_evidence_used": False,
            },
            "spent_scope": {
                "row_count": len(spent.row_ids),
                "fixed_inner_fold_count": len(set(spent.inner_fold_ids or ())),
            },
            "sealed_gate": {
                "row_ids_exposed": False,
                "text_exposed": False,
                "treatment_exposed": False,
                "outcome_exposed": False,
                "aggregates_exposed": False,
                "source_values_exposed": False,
                "feature_bank_values_exposed": False,
            },
            "outer_heldout": {
                "row_ids_exposed": False,
                "text_exposed": False,
                "labels_exposed": False,
                "values_exposed": False,
            },
            "persistence_disclosure": {
                "spent_aggregate_numerical_diagnostics_persisted_in_round_audit": True,
                "spent_raw_note_text_persisted_in_round_audit": False,
                "grounding_row_identifiers_or_value_spans_persisted": False,
                "row_level_numerical_vectors_persisted_in_round_audit": False,
                "provider_cache_persistence_is_provider_defined": True,
            },
        }
        return json.loads(_canonical_json(context))

    def _observable_review_rows(
        self,
        *,
        row_ids: Sequence[int],
        extracted: pd.DataFrame,
        data: pd.DataFrame,
        fold_by_row: Mapping[int, int] | None,
    ) -> ObservableCausalRows:
        exact_ids = tuple(map(int, row_ids))
        data_by_id = data.set_index("_oci_row_id", drop=False)
        extracted_by_id = extracted.set_index("_oci_row_id", drop=False)
        missing_data = set(exact_ids) - set(map(int, data_by_id.index))
        missing_extracted = set(exact_ids) - set(map(int, extracted_by_id.index))
        if missing_data or missing_extracted:
            raise ValueError("review rows are not present in data and extraction frames")
        selected_data = data_by_id.loc[list(exact_ids)]
        selected_extracted = extracted_by_id.loc[list(exact_ids)].reset_index(drop=True)
        folds = (
            None if fold_by_row is None else tuple(int(fold_by_row[row_id]) for row_id in exact_ids)
        )
        return ObservableCausalRows(
            row_ids=exact_ids,
            extracted=selected_extracted,
            treatment=selected_data[self.config.treatment_column].to_numpy(dtype=float),
            outcome=selected_data[self.config.outcome_column].to_numpy(dtype=float),
            inner_fold_ids=folds,
        )

    @staticmethod
    def _candidate_post_extraction_quality_guard(
        candidate_spent: ObservableCausalRows,
        candidate_specs: Sequence[Mapping[str, Any]],
        *,
        spent_texts: Sequence[str],
        extraction_changed_names: Sequence[str],
        scientific_policy: PostExtractionScientificPolicy | None,
    ) -> Mapping[str, Any]:
        changed = tuple(map(str, extraction_changed_names))
        quality = build_extraction_quality_diagnostics(
            candidate_spent.extracted,
            candidate_specs,
            fold_ids=candidate_spent.inner_fold_ids or (),
            policy=(
                None
                if scientific_policy is None
                else scientific_policy.extraction_quality
            ),
        )
        quality_rows = list(quality["features"])
        grounding_rows = build_extraction_grounding_diagnostics(
            candidate_spent.extracted,
            spent_texts,
            candidate_specs,
            diagnostic_start=len(quality_rows) + 1,
            policy=(
                None
                if scientific_policy is None
                else scientific_policy.extraction_grounding
            ),
        )
        retained_ontology = AllEvidenceFusionRunner._retained_registry_ontology_from_grounding(
            candidate_specs,
            grounding_rows,
        )
        if not changed:
            return json.loads(
                _canonical_json(
                    {
                        "applicable": False,
                        "passed": True,
                        "extraction_changed_names": [],
                        "failed_names": [],
                        "diagnostics": [],
                        "retained_registry_ontology_guard": retained_ontology,
                        "grounding_diagnostic_version": (EXTRACTION_GROUNDING_DIAGNOSTIC_VERSION),
                        "grounding_hard_failure_policy": [
                            "alternative_category_only_value_support"
                        ],
                        "source_text_temporal_policy": source_text_temporal_policy_audit(),
                        "row_level_values_persisted": False,
                        "raw_note_text_persisted": False,
                        "aggregate_numerical_diagnostics_persisted": False,
                    }
                )
            )
        changed_set = set(changed)
        changed_quality_rows = [
            row for row in quality_rows if str(row.get("feature_name")) in changed_set
        ]
        changed_grounding_rows = [
            row for row in grounding_rows if str(row.get("feature_name")) in changed_set
        ]
        if {str(row.get("feature_name")) for row in changed_quality_rows} != changed_set or {
            str(row.get("feature_name")) for row in changed_grounding_rows
        } != changed_set:
            raise RuntimeError("candidate quality guard did not evaluate every changed extraction")
        changed_rows = [*changed_quality_rows, *changed_grounding_rows]
        failed = sorted(
            {str(row["feature_name"]) for row in changed_rows if not bool(row["passed"])}
        )
        return json.loads(
            _canonical_json(
                {
                    "applicable": True,
                    "passed": not failed,
                    "extraction_changed_names": list(changed),
                    "failed_names": failed,
                    "diagnostics": changed_rows,
                    "retained_registry_ontology_guard": retained_ontology,
                    "grounding_diagnostic_version": EXTRACTION_GROUNDING_DIAGNOSTIC_VERSION,
                    "grounding_hard_failure_policy": ["alternative_category_only_value_support"],
                    "source_text_temporal_policy": source_text_temporal_policy_audit(),
                    "row_level_values_persisted": False,
                    "raw_note_text_persisted": False,
                    "grounding_row_identifiers_or_value_spans_persisted": False,
                    "aggregate_numerical_diagnostics_persisted": True,
                }
            )
        )

    @staticmethod
    def _retained_registry_ontology_guard(
        retained_spent: ObservableCausalRows,
        retained_specs: Sequence[Mapping[str, Any]],
        *,
        spent_texts: Sequence[str],
        scientific_policy: PostExtractionScientificPolicy | None,
    ) -> Mapping[str, Any]:
        """Fail only on locally grounded categorical-ontology hazards.

        This guard deliberately covers every retained contract, including
        contracts whose extraction semantics were unchanged by the proposal.
        It is separate from the general candidate quality guard so ordinary
        sparsity, constancy, weak lexical grounding, missing-value opportunities,
        and weak lexical grounding on an unchanged contract cannot become an
        automatic rejection. Source timing is trusted by design and is absent
        from this guard.
        """

        canonical_specs = [CandidateContract(spec).extraction_spec for spec in retained_specs]
        grounding_rows = build_extraction_grounding_diagnostics(
            retained_spent.extracted,
            spent_texts,
            canonical_specs,
            policy=(
                None
                if scientific_policy is None
                else scientific_policy.extraction_grounding
            ),
        )
        return AllEvidenceFusionRunner._retained_registry_ontology_from_grounding(
            canonical_specs,
            grounding_rows,
        )

    @staticmethod
    def _retained_registry_ontology_from_grounding(
        retained_specs: Sequence[Mapping[str, Any]],
        grounding_rows: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        canonical_specs = [CandidateContract(spec).extraction_spec for spec in retained_specs]
        expected_names = {str(spec["name"]) for spec in canonical_specs}
        evaluated_names = {str(row.get("feature_name")) for row in grounding_rows}
        if evaluated_names != expected_names:
            raise RuntimeError("retained registry ontology guard did not evaluate every contract")
        hard_failures = ("alternative_category_only_value_support",)
        failed_names_by_reason = {
            failure: sorted(
                {
                    str(row["feature_name"])
                    for row in grounding_rows
                    if failure in set(map(str, row.get("hard_failures") or ()))
                }
            )
            for failure in hard_failures
        }
        failed_names = sorted(
            set().union(*(set(names) for names in failed_names_by_reason.values()))
        )
        return json.loads(
            _canonical_json(
                {
                    "applicable": True,
                    "passed": not failed_names,
                    "evaluated_names": sorted(expected_names),
                    "failed_names": failed_names,
                    "failed_names_by_reason": failed_names_by_reason,
                    "diagnostics": grounding_rows,
                    "guard_scope": "all_retained_contracts_on_exact_spent_rows",
                    "grounding_diagnostic_version": (EXTRACTION_GROUNDING_DIAGNOSTIC_VERSION),
                    "hard_failure_policy": list(hard_failures),
                    "safety_dimensions": ["categorical_ontology_alignment"],
                    "source_text_temporal_policy": source_text_temporal_policy_audit(),
                    "ordinary_missingness_or_constancy_is_a_hard_failure": False,
                    "missing_value_opportunities_or_category_conflicts_are_hard_failures": False,
                    "row_level_values_persisted": False,
                    "raw_note_text_persisted": False,
                    "grounding_row_identifiers_or_value_spans_persisted": False,
                    "aggregate_numerical_diagnostics_persisted": True,
                }
            )
        )

    @staticmethod
    def _ontology_failure_keys(
        ontology_guard: Mapping[str, Any],
    ) -> frozenset[tuple[str, str]]:
        """Return exact spent-only ontology failures for monotone staging."""

        raw = ontology_guard.get("failed_names_by_reason")
        if not isinstance(raw, Mapping):
            raise ValueError("ontology guard lacks failed_names_by_reason")
        keys = frozenset(
            (str(name), str(reason)) for reason, names in raw.items() for name in (names or ())
        )
        declared_names = set(map(str, ontology_guard.get("failed_names") or ()))
        if {name for name, _reason in keys} != declared_names:
            raise ValueError("ontology failure names disagree with reason mapping")
        if bool(ontology_guard.get("passed")) != (not keys):
            raise ValueError("ontology pass flag disagrees with hard failures")
        return keys

    @staticmethod
    def _cumulative_extraction_changed_names(
        accepted_specs: Sequence[Mapping[str, Any]],
        candidate_specs: Sequence[Mapping[str, Any]],
    ) -> tuple[str, ...]:
        """Names whose extraction semantics differ from the accepted round baseline."""

        accepted = {
            str(spec["name"]): CandidateContract(spec).extraction_spec for spec in accepted_specs
        }
        changed: list[str] = []
        for raw in candidate_specs:
            spec = CandidateContract(raw).extraction_spec
            name = str(spec["name"])
            prior = accepted.get(name)
            if prior is None or extraction_semantics_sha256(prior) != extraction_semantics_sha256(
                spec
            ):
                changed.append(name)
        return tuple(changed)

    @staticmethod
    def _cumulative_review_projection_plan(
        accepted_specs: Sequence[Mapping[str, Any]],
        candidate_specs: Sequence[Mapping[str, Any]],
        *,
        operation_audit: Sequence[Mapping[str, Any]] = (),
    ) -> AppliedReviewOperations:
        """Build the exact accepted-base to cumulative-workspace extraction plan."""

        before = [CandidateContract(spec).extraction_spec for spec in accepted_specs]
        after = [CandidateContract(spec).extraction_spec for spec in candidate_specs]
        before_by_name = {str(spec["name"]): spec for spec in before}
        after_by_name = {str(spec["name"]): spec for spec in after}
        changed_names = AllEvidenceFusionRunner._cumulative_extraction_changed_names(
            before,
            after,
        )
        role_only = tuple(
            name
            for name, spec in after_by_name.items()
            if name in before_by_name
            and extraction_semantics_sha256(before_by_name[name])
            == extraction_semantics_sha256(spec)
            and tuple(before_by_name[name].get("roles") or ()) != tuple(spec.get("roles") or ())
        )
        return AppliedReviewOperations(
            specs=tuple(json.loads(_canonical_json(spec)) for spec in after),
            reextract_specs=tuple(
                json.loads(_canonical_json(after_by_name[name])) for name in changed_names
            ),
            removed_names=tuple(name for name in before_by_name if name not in after_by_name),
            added_names=tuple(name for name in after_by_name if name not in before_by_name),
            extraction_changed_names=changed_names,
            role_only_changed_names=role_only,
            operation_audit=tuple(
                json.loads(_canonical_json(dict(row))) for row in operation_audit
            ),
        )

    def _review_schedule(
        self,
        *,
        outer_train: pd.DataFrame,
        outer_fold: int,
    ) -> ReviewPartitionSchedule:
        if self.review_partition_provider is not None:
            if self.review_partition_provider_identity is None:
                raise RuntimeError("review partition provider lacks immutable identity")
            return _build_injected_review_partition_schedule(
                outer_train,
                outer_fold=outer_fold,
                review_rounds=self.config.post_extraction_review_rounds,
                minimum_partition_rows=self.config.post_extraction_review_min_partition_rows,
                treatment_column=self.config.treatment_column,
                outcome_column=self.config.outcome_column,
                outcome_type=self.config.outcome_type,
                provider=self.review_partition_provider,
                provider_identity=self.review_partition_provider_identity,
            )
        if self.config.require_review_feature_banks and not callable(
            getattr(self.review_gate_feature_bank_provider, "bind_fold", None)
        ):
            raise RuntimeError(
                "required feature-bank review cannot use independently generated partitions"
            )
        return _build_review_partition_schedule(
            outer_train,
            outer_fold=outer_fold,
            review_rounds=self.config.post_extraction_review_rounds,
            minimum_partition_rows=self.config.post_extraction_review_min_partition_rows,
            random_state=self.config.random_state,
            treatment_column=self.config.treatment_column,
            outcome_column=self.config.outcome_column,
            outcome_type=self.config.outcome_type,
        )

    def _gate_only_reference_views(
        self,
        *,
        outer_fold: int,
        context_epoch: int,
        gate_row_ids: tuple[int, ...],
        context: ObservableCausalRows,
    ) -> tuple[GateSourceSignalView, GateFeatureBankView, Mapping[str, Any]]:
        """Adapt one authenticated cumulative numerical view to diagnostics.

        No text, treatment, outcome, conditional-context matrix, or fit API is
        passed to the provider.  The returned Gate* views intentionally have
        empty context halves and exact complete-spent provenance.
        """

        if not self.gate_only_reference_review:
            raise RuntimeError("gate-only reference views require the gate-only policy")
        provider = self.review_gate_source_provider
        if (
            provider is None
            or provider is not self.review_gate_feature_bank_provider
            or not callable(getattr(provider, "get_gate_only_view", None))
        ):
            raise RuntimeError("gate-only review lost its shared cumulative numerical provider")
        if callable(getattr(provider, "bind_fold", None)):
            raise RuntimeError("gate-only review must never expose bind_fold")
        current = _review_provider_identity(
            provider,
            label="gate_only_shared_numerical_provider",
        )
        if (
            current != self.review_gate_source_provider_identity
            or current != self.review_gate_feature_bank_provider_identity
        ):
            raise RuntimeError("gate-only cumulative numerical provider identity changed")

        from .direct_upstream_numerical_reference_bank import (
            CALIBRATED_SOURCE_BANK,
            RAW_FEATURE_BANK,
            MaterializedRoleNeutralNumericalMatrix,
            RoleNeutralGateOnlyNumericalView,
        )

        opened = provider.get_gate_only_view(
            outer_fold=int(outer_fold),
            context_epoch=int(context_epoch),
            exact_spent_row_ids=tuple(map(int, context.row_ids)),
            exact_gate_row_ids=gate_row_ids,
        )
        if type(opened) is not RoleNeutralGateOnlyNumericalView:
            raise TypeError(
                "gate-only provider must return RoleNeutralGateOnlyNumericalView"
            )
        identity = opened.identity()
        if (
            tuple(opened.spent_row_ids) != tuple(context.row_ids)
            or tuple(opened.gate_row_ids) != gate_row_ids
            or identity.get("gate_fit_row_provenance") != list(context.row_ids)
            or identity.get("context_oof_available") is not False
            or identity.get("conditional_context_gate_view_claimed") is not False
            or identity.get("fit_or_refit_performed") is not False
            or identity.get("registered_gate_labels_accessed") is not False
        ):
            raise ValueError("gate-only numerical view identity or lineage changed")
        calibrated = opened.materialize(bank_kinds=(CALIBRATED_SOURCE_BANK,))
        raw = opened.materialize(bank_kinds=(RAW_FEATURE_BANK,))
        if type(calibrated) is not MaterializedRoleNeutralNumericalMatrix or type(
            raw
        ) is not MaterializedRoleNeutralNumericalMatrix:
            raise TypeError("gate-only numerical materialization returned an invalid matrix")
        if calibrated.row_ids != gate_row_ids or raw.row_ids != gate_row_ids:
            raise ValueError("gate-only numerical materialization changed gate row order")
        if set(calibrated.bank_kinds) != {CALIBRATED_SOURCE_BANK} or set(
            raw.bank_kinds
        ) != {RAW_FEATURE_BANK}:
            raise ValueError("gate-only numerical materialization mixed bank roles")

        lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
        source = GateSourceSignalView(
            row_ids=gate_row_ids,
            source_names=tuple(
                f"{family}__{coordinate_id}"
                for family, coordinate_id in zip(
                    calibrated.source_families,
                    calibrated.coordinate_ids,
                )
            ),
            source_kinds=tuple(
                f"{family}__calibrated_effect"
                for family in calibrated.source_families
            ),
            values=calibrated.values,
            fit_row_provenance=(lineage,) * len(calibrated.coordinate_ids),
        )
        features = GateFeatureBankView(
            row_ids=gate_row_ids,
            feature_names=tuple(
                f"{family}__{coordinate_id}"
                for family, coordinate_id in zip(
                    raw.source_families,
                    raw.coordinate_ids,
                )
            ),
            source_kinds=raw.source_kinds,
            consumer_roles=raw.consumer_roles,
            values=raw.values,
            fit_row_provenance=(lineage,) * len(raw.coordinate_ids),
        )
        for view in (source, features):
            view.aligned_values(gate_row_ids)
            self._gate_view_lineage_audit(view, context=context)
        return source, features, {
            "policy": GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
            "context_epoch": int(context_epoch),
            "view_identity": json.loads(_canonical_json(identity)),
            "source_coordinate_count": len(calibrated.coordinate_ids),
            "feature_coordinate_count": len(raw.coordinate_ids),
            "conditional_context_values_accessed": False,
            "bind_fold_called": False,
            "fit_or_refit_performed": False,
        }

    def _gate_source_view(
        self,
        *,
        outer_fold: int,
        gate_row_ids: tuple[int, ...],
        context: ObservableCausalRows,
        context_texts: Sequence[str],
        gate_texts: Sequence[str],
    ) -> GateSourceSignalView | None:
        provider = self.review_gate_source_provider
        if provider is None:
            if self.config.require_review_source_signals:
                raise RuntimeError("required gate-local source signals are unavailable")
            return None
        lookup: Any = provider
        bind_fold = getattr(provider, "bind_fold", None)
        if callable(bind_fold):
            lookup = bind_fold(
                outer_fold=int(outer_fold),
                context=context,
                context_texts=tuple(map(str, context_texts)),
                gate_texts=tuple(map(str, gate_texts)),
                exact_gate_row_ids=gate_row_ids,
            )
        get_view = getattr(lookup, "get_gate_source_view", None)
        if not callable(get_view):
            raise TypeError("review source provider lacks get_gate_source_view()")
        view = get_view(
            outer_fold=int(outer_fold),
            exact_gate_row_ids=gate_row_ids,
        )
        if not isinstance(view, GateSourceSignalView):
            raise TypeError("review source provider did not return GateSourceSignalView")
        if tuple(view.row_ids) != gate_row_ids:
            raise ValueError("review source view changed the exact gate row order/set")
        # Re-run the exact alignment guard at the orchestration boundary. The
        # dataclass constructor has already enforced recursive whole-gate exclusion.
        view.aligned_values(gate_row_ids)
        self._gate_view_lineage_audit(view, context=context)
        view.aligned_conditional_values(
            exact_context_row_ids=context.row_ids,
            exact_context_inner_fold_ids=context.inner_fold_ids or (),
            exact_gate_row_ids=gate_row_ids,
        )
        return view

    def _gate_feature_bank_view(
        self,
        *,
        outer_fold: int,
        gate_row_ids: tuple[int, ...],
        context: ObservableCausalRows,
        context_texts: Sequence[str],
        gate_texts: Sequence[str],
    ) -> GateFeatureBankView | None:
        provider = self.review_gate_feature_bank_provider
        if provider is None:
            if self.config.require_review_feature_banks:
                raise RuntimeError("required gate-local feature banks are unavailable")
            return None
        lookup: Any = provider
        bind_fold = getattr(provider, "bind_fold", None)
        if callable(bind_fold):
            lookup = bind_fold(
                outer_fold=int(outer_fold),
                context=context,
                context_texts=tuple(map(str, context_texts)),
                gate_texts=tuple(map(str, gate_texts)),
                exact_gate_row_ids=gate_row_ids,
            )
        get_view = getattr(lookup, "get_gate_feature_bank_view", None)
        if not callable(get_view):
            raise TypeError("review feature-bank provider lacks get_gate_feature_bank_view()")
        view = get_view(
            outer_fold=int(outer_fold),
            exact_gate_row_ids=gate_row_ids,
        )
        if not isinstance(view, GateFeatureBankView):
            raise TypeError("review feature-bank provider did not return GateFeatureBankView")
        if tuple(view.row_ids) != gate_row_ids:
            raise ValueError("review feature-bank view changed the exact gate row order/set")
        view.aligned_values(gate_row_ids)
        self._gate_view_lineage_audit(view, context=context)
        view.aligned_conditional_values(
            exact_context_row_ids=context.row_ids,
            exact_context_inner_fold_ids=context.inner_fold_ids or (),
            exact_gate_row_ids=gate_row_ids,
        )
        return view

    def _hierarchical_gate_views(
        self,
        *,
        outer_fold: int,
        gate_row_ids: tuple[int, ...],
        context: ObservableCausalRows,
        context_texts: Sequence[str],
        gate_texts: Sequence[str],
        prebound_provider: Any | None,
        context_epoch: int | None = None,
    ) -> tuple[GateSourceSignalView, GateFeatureBankView, bool]:
        """Bind after proposal freeze, or consume that boundary's verified provider.

        The returned boolean records whether the caller supplied an already
        bound provider from the just-verified first-gate realization.  It never
        means that numerical values existed during discovery preparation.
        """

        if getattr(self, "gate_only_reference_review", False):
            if prebound_provider is not None:
                raise ValueError("gate-only review rejects a prebound fit provider")
            if context_epoch is None:
                raise ValueError("gate-only review requires the cumulative context epoch")
            source, features, _audit = self._gate_only_reference_views(
                outer_fold=outer_fold,
                context_epoch=context_epoch,
                gate_row_ids=gate_row_ids,
                context=context,
            )
            return source, features, False

        provider = self.review_gate_source_provider
        if (
            provider is None
            or provider is not self.review_gate_feature_bank_provider
            or not callable(getattr(provider, "bind_fold", None))
        ):
            raise RuntimeError("hierarchical review lost its shared bindable gate provider")
        current = _review_provider_identity(
            provider,
            label="hierarchical_shared_gate_provider",
        )
        if (
            current != self.review_gate_source_provider_identity
            or current != self.review_gate_feature_bank_provider_identity
        ):
            raise RuntimeError("hierarchical shared gate provider identity changed")
        lookup = prebound_provider
        prebound_provider_used = lookup is not None
        if lookup is None:
            lookup = provider.bind_fold(
                outer_fold=int(outer_fold),
                context=context,
                context_texts=tuple(map(str, context_texts)),
                gate_texts=tuple(map(str, gate_texts)),
                exact_gate_row_ids=gate_row_ids,
            )
        get_source = getattr(lookup, "get_gate_source_view", None)
        get_features = getattr(lookup, "get_gate_feature_bank_view", None)
        if not callable(get_source) or not callable(get_features):
            raise TypeError("bound hierarchical gate provider lacks both view methods")
        source = get_source(
            outer_fold=int(outer_fold),
            exact_gate_row_ids=gate_row_ids,
        )
        features = get_features(
            outer_fold=int(outer_fold),
            exact_gate_row_ids=gate_row_ids,
        )
        if not isinstance(source, GateSourceSignalView) or not isinstance(
            features, GateFeatureBankView
        ):
            raise TypeError("bound hierarchical provider returned an invalid gate view")
        if tuple(source.row_ids) != gate_row_ids or tuple(features.row_ids) != gate_row_ids:
            raise ValueError("bound hierarchical provider changed exact gate row order")
        for view in (source, features):
            view.aligned_values(gate_row_ids)
            self._gate_view_lineage_audit(view, context=context)
            view.aligned_conditional_values(
                exact_context_row_ids=context.row_ids,
                exact_context_inner_fold_ids=context.inner_fold_ids or (),
                exact_gate_row_ids=gate_row_ids,
            )
        return source, features, prebound_provider_used

    def _gate_view_lineage_audit(
        self,
        view: GateSourceSignalView | GateFeatureBankView | None,
        *,
        context: ObservableCausalRows,
    ) -> Mapping[str, Any] | None:
        if view is None:
            return None
        allowed = frozenset(context.row_ids)
        gate = frozenset(view.row_ids)
        union: set[Any] = set()
        lineage_count = 0
        for source_lineages in view.fit_row_provenance:
            for lineage in source_lineages:
                fitted = lineage.recursive_fit_row_ids()
                lineage_count += 1
                if not fitted:
                    raise ValueError("gate source/feature lineage cannot be empty")
                if not fitted <= allowed:
                    raise ValueError(
                        "gate source/feature lineage includes an unspent future partition"
                    )
                if fitted & gate:
                    raise ValueError("gate source/feature lineage includes the current gate")
                union.update(fitted)
        if getattr(self, "gate_only_reference_review", False):
            if any(
                fitted.recursive_fit_row_ids() != allowed
                for source_lineages in view.fit_row_provenance
                for fitted in source_lineages
            ):
                raise ValueError(
                    "gate-only source/feature lineage must equal the exact complete spent context"
                )
            if (
                view.context_row_ids
                or view.context_inner_fold_ids
                or view.context_fit_row_provenance
                or np.asarray(view.context_values).size
            ):
                raise ValueError(
                    "gate-only source/feature views must not contain context-side values"
                )
            ordered = sorted(map(int, union))
            return {
                "lineage_count": lineage_count,
                "recursive_fit_row_count": len(ordered),
                "recursive_fit_row_fingerprint": row_set_fingerprint(ordered),
                "all_recursive_fit_rows_equal_complete_spent_context": True,
                "all_recursive_fit_rows_disjoint_from_current_gate": True,
                "context_oof_lineage_count": 0,
                "context_oof_recursive_fit_row_count": 0,
                "context_oof_recursive_fit_row_fingerprint": None,
                "context_values_cross_fitted_by_exact_inner_fold": False,
                "context_values_unavailable_by_design": True,
                "upstream_values_used_as_training_covariates": False,
            }
        context_lineage_count = 0
        context_union: set[Any] = set()
        context_ids = tuple(view.context_row_ids)
        if context_ids != tuple(context.row_ids):
            raise ValueError("gate source/feature context row order changed")
        for source_lineages in view.context_fit_row_provenance:
            for row_id, lineage in zip(context_ids, source_lineages):
                fitted = lineage.recursive_fit_row_ids()
                context_lineage_count += 1
                if not fitted or not fitted < allowed or row_id in fitted:
                    raise ValueError("gate source/feature context lineage is not exact out-of-fold")
                if fitted & gate:
                    raise ValueError(
                        "gate source/feature context lineage includes the current gate"
                    )
                context_union.update(fitted)
        ordered = sorted(map(int, union))
        context_ordered = sorted(map(int, context_union))
        return {
            "lineage_count": lineage_count,
            "recursive_fit_row_count": len(ordered),
            "recursive_fit_row_fingerprint": row_set_fingerprint(ordered),
            "all_recursive_fit_rows_within_spent_context": True,
            "all_recursive_fit_rows_disjoint_from_current_gate": True,
            "context_oof_lineage_count": context_lineage_count,
            "context_oof_recursive_fit_row_count": len(context_ordered),
            "context_oof_recursive_fit_row_fingerprint": row_set_fingerprint(context_ordered),
            "context_values_cross_fitted_by_exact_inner_fold": True,
        }

    @staticmethod
    def _opaque_gate_source_catalog(
        source_view: GateSourceSignalView | None,
        feature_bank_view: GateFeatureBankView | None,
    ) -> Mapping[str, Any]:
        calibrated: list[dict[str, Any]] = []
        if source_view is not None:
            calibrated = [
                {
                    "source_id": f"gate_source_{index:04d}",
                    "source_kind": kind,
                    "source_name_sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
                }
                for index, (name, kind) in enumerate(
                    zip(source_view.source_names, source_view.source_kinds),
                    start=1,
                )
            ]
        feature_banks: list[dict[str, Any]] = []
        if feature_bank_view is not None:
            feature_banks = [
                {
                    "feature_id": f"gate_feature_{index:04d}",
                    "source_kind": kind,
                    "consumer_role": role,
                    "feature_name_sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
                    "calibrated_tau": False,
                }
                for index, (name, kind, role) in enumerate(
                    zip(
                        feature_bank_view.feature_names,
                        feature_bank_view.source_kinds,
                        feature_bank_view.consumer_roles,
                    ),
                    start=1,
                )
            ]
        return {
            "calibrated_effect_sources": calibrated,
            "role_aware_uncalibrated_feature_banks": feature_banks,
            "row_level_numerical_vectors_persisted_in_review_round": False,
            "aggregate_numerical_acceptance_statistics_persisted_in_review_round": True,
            "provider_cache_numerical_persistence": "provider_defined_outside_round_audit",
            "raw_values_exposed_to_review_agent": False,
        }

    @staticmethod
    def _sanitized_gate_decision(decision: GateAcceptanceDecision) -> Mapping[str, Any]:
        payload = decision.as_dict()
        engine_sha256 = str(payload.pop("decision_sha256"))
        source_guard_ids: dict[str, str] = {}
        source_kind_by_id: dict[str, str] = {}
        for side in ("current", "candidate"):
            section = payload.get(side)
            if not isinstance(section, dict):
                continue
            source = section.get("source_signal_evaluation")
            if isinstance(source, dict) and isinstance(source.get("sources"), list):
                for row in source["sources"]:
                    if isinstance(row, dict):
                        name = str(row.pop("source_name", ""))
                        kind = str(row.get("source_kind") or "")
                        guard_key = f"{kind}::{name}"
                        source_id = source_guard_ids.setdefault(
                            guard_key,
                            f"gate_source_{len(source_guard_ids) + 1:04d}",
                        )
                        source_kind_by_id[source_id] = kind
                        row["source_id"] = source_id
            feature_bank = section.get("feature_bank_evaluation")
            if isinstance(feature_bank, dict) and isinstance(feature_bank.get("features"), list):
                for index, row in enumerate(feature_bank["features"], start=1):
                    if isinstance(row, dict):
                        row.pop("feature_name", None)
                        row["feature_id"] = f"gate_feature_{index:04d}"
        source_direction = (
            payload.get("guards", {}).get("source_direction_preservation")
            if isinstance(payload.get("guards"), Mapping)
            else None
        )
        if isinstance(source_direction, dict) and isinstance(
            source_direction.get("by_source"), Mapping
        ):
            opaque_by_source: dict[str, Any] = {}
            for raw_key, raw_row in source_direction["by_source"].items():
                source_id = source_guard_ids.get(str(raw_key))
                if source_id is None:
                    source_id = (
                        "gate_source_unmatched_"
                        + hashlib.sha256(str(raw_key).encode("utf-8")).hexdigest()[:16]
                    )
                row = dict(raw_row) if isinstance(raw_row, Mapping) else {"passed": False}
                row["source_kind"] = source_kind_by_id.get(source_id, "unmatched")
                opaque_by_source[source_id] = row
            source_direction["by_source"] = opaque_by_source
        forbidden_keys = {
            "row_id",
            "row_ids",
            "record_id",
            "record_ids",
            "patient_id",
            "patient_ids",
            "source_name",
            "source_names",
            "feature_name",
            "feature_names",
            "values",
            "raw_values",
            "activations",
            "predictions",
        }

        def validate_aggregate_only(value: Any, *, path: str) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    if str(key).strip().lower() in forbidden_keys:
                        raise ValueError(
                            f"gate acceptance payload contains row-level data at {path}.{key}"
                        )
                    validate_aggregate_only(child, path=f"{path}.{key}")
                return
            if isinstance(value, list):
                if value and all(
                    isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
                ):
                    raise ValueError(
                        f"gate acceptance payload contains an unnamed numerical vector at {path}"
                    )
                for index, child in enumerate(value):
                    validate_aggregate_only(child, path=f"{path}[{index}]")
                return
            if value is not None and not isinstance(value, (str, bool, int, float)):
                raise TypeError(f"gate acceptance payload is not closed JSON at {path}")

        validate_aggregate_only(payload, path="acceptance")
        payload["acceptance_engine_decision_sha256"] = engine_sha256
        payload["source_and_feature_names_replaced_by_opaque_ids"] = True
        payload["row_level_numerical_vectors_persisted"] = False
        payload["aggregate_numerical_statistics_persisted"] = True
        payload["sanitized_decision_sha256"] = _content_sha256(payload)
        return json.loads(_canonical_json(payload))

    @staticmethod
    def _prior_gate_feedback_diagnostic(
        decision: GateAcceptanceDecision,
        *,
        review_round: int,
        response_sha256: str,
        operation_audit: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Distill one consumed gate into a safe next-round diagnostic."""

        sanitized = AllEvidenceFusionRunner._sanitized_gate_decision(decision)

        def nested(mapping: Any, *keys: str) -> Any:
            current = mapping
            for key in keys:
                if not isinstance(current, Mapping):
                    return None
                current = current.get(key)
            return current

        def delta(candidate: Any, current: Any) -> float | None:
            if (
                isinstance(candidate, (bool, np.bool_))
                or isinstance(current, (bool, np.bool_))
                or not isinstance(candidate, (int, float, np.integer, np.floating))
                or not isinstance(current, (int, float, np.integer, np.floating))
                or not math.isfinite(float(candidate))
                or not math.isfinite(float(current))
            ):
                return None
            return float(candidate) - float(current)

        failed_guards: list[str] = []

        def collect_failed(value: Any, *, path: str) -> None:
            if isinstance(value, Mapping):
                if value.get("passed") is False:
                    failed_guards.append(path)
                for key, child in value.items():
                    if key != "passed":
                        collect_failed(child, path=f"{path}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    collect_failed(child, path=f"{path}[{index}]")

        collect_failed(sanitized.get("guards", {}), path="guards")
        current = sanitized.get("current") or {}
        candidate = sanitized.get("candidate") or {}
        current_r_ratio = nested(current, "metrics", "effect", "r_loss_ratio")
        candidate_r_ratio = nested(candidate, "metrics", "effect", "r_loss_ratio")
        current_weighted_r = nested(current, "metrics", "effect", "weighted_r_loss")
        candidate_weighted_r = nested(candidate, "metrics", "effect", "weighted_r_loss")
        current_score = nested(current, "penalized_relative_r_loss_score")
        candidate_score = nested(candidate, "penalized_relative_r_loss_score")

        current_complexity = nested(current, "complexity") or {}
        candidate_complexity = nested(candidate, "complexity") or {}
        complexity: dict[str, Any] = {}
        for key in ("contract_count", "encoded_column_count"):
            current_value = nested(current_complexity, key)
            candidate_value = nested(candidate_complexity, key)
            complexity[key] = {
                "current": current_value,
                "candidate": candidate_value,
                "candidate_minus_current": delta(candidate_value, current_value),
            }

        current_sources = {
            str(row.get("source_id")): row
            for row in (nested(current, "source_signal_evaluation", "sources") or [])
            if isinstance(row, Mapping) and row.get("source_id")
        }
        candidate_sources = {
            str(row.get("source_id")): row
            for row in (nested(candidate, "source_signal_evaluation", "sources") or [])
            if isinstance(row, Mapping) and row.get("source_id")
        }
        source_guard_rows = (
            nested(
                sanitized,
                "guards",
                "source_direction_preservation",
                "by_source",
            )
            or {}
        )
        calibrated_sources: list[dict[str, Any]] = []
        for source_id in sorted(set(current_sources) | set(candidate_sources)):
            current_row = current_sources.get(source_id, {})
            candidate_row = candidate_sources.get(source_id, {})
            current_corr = nested(current_row, "tau_correlation")
            candidate_corr = nested(candidate_row, "tau_correlation")
            current_context_ratio = nested(current_row, "contextual_r_loss_ratio")
            candidate_context_ratio = nested(candidate_row, "contextual_r_loss_ratio")
            guard_row = (
                source_guard_rows.get(source_id, {})
                if isinstance(source_guard_rows, Mapping)
                else {}
            )
            calibrated_sources.append(
                {
                    "source_id": source_id,
                    "source_kind": str(
                        current_row.get("source_kind")
                        or candidate_row.get("source_kind")
                        or guard_row.get("source_kind")
                        or ""
                    ),
                    "current_tau_correlation": current_corr,
                    "candidate_tau_correlation": candidate_corr,
                    "tau_correlation_delta": delta(candidate_corr, current_corr),
                    "current_contextual_r_loss_ratio": current_context_ratio,
                    "candidate_contextual_r_loss_ratio": candidate_context_ratio,
                    "contextual_r_loss_ratio_delta": delta(
                        candidate_context_ratio,
                        current_context_ratio,
                    ),
                    "direction_guard_passed": guard_row.get("passed"),
                }
            )

        current_roles = (
            nested(
                current,
                "feature_bank_evaluation",
                "preservation_score_by_consumer_role",
            )
            or {}
        )
        candidate_roles = (
            nested(
                candidate,
                "feature_bank_evaluation",
                "preservation_score_by_consumer_role",
            )
            or {}
        )
        role_guards = (
            nested(
                sanitized,
                "guards",
                "feature_bank_preservation",
                "by_consumer_role",
            )
            or {}
        )
        role_rows: list[dict[str, Any]] = []
        for role in sorted(set(current_roles) | set(candidate_roles) | set(role_guards)):
            current_value = current_roles.get(role)
            candidate_value = candidate_roles.get(role)
            guard_row = role_guards.get(role, {}) if isinstance(role_guards, Mapping) else {}
            role_rows.append(
                {
                    "consumer_role": role,
                    "current_preservation_score": current_value,
                    "candidate_preservation_score": candidate_value,
                    "candidate_minus_current": delta(candidate_value, current_value),
                    "guard_passed": guard_row.get("passed"),
                }
            )

        def family_rows_by_identity(value: Any) -> dict[tuple[str, str], Mapping[str, Any]]:
            rows: dict[tuple[str, str], Mapping[str, Any]] = {}
            if not isinstance(value, list):
                return rows
            for raw_row in value:
                if not isinstance(raw_row, Mapping):
                    continue
                source_kind = str(raw_row.get("source_kind") or "").strip()
                consumer_role = str(raw_row.get("consumer_role") or "").strip()
                if source_kind and consumer_role:
                    rows[(source_kind, consumer_role)] = raw_row
            return rows

        current_families = family_rows_by_identity(
            nested(
                current,
                "feature_bank_evaluation",
                "preservation_by_source_kind_and_consumer_role",
            )
        )
        candidate_families = family_rows_by_identity(
            nested(
                candidate,
                "feature_bank_evaluation",
                "preservation_by_source_kind_and_consumer_role",
            )
        )
        family_guards = family_rows_by_identity(
            nested(
                sanitized,
                "guards",
                "feature_bank_preservation",
                "by_source_kind_and_consumer_role",
            )
        )
        family_rows: list[dict[str, Any]] = []
        family_metric_names = (
            "feature_count",
            "finite_correlation_count",
            "mean_absolute_role_matched_prediction_correlation",
            "aggregate_absolute_role_matched_prediction_correlation",
            "aggregate_absolute_correlation_share",
            "leave_family_out_feature_mean_absolute_correlation",
            "feature_mean_absolute_correlation_delta_when_family_removed",
        )
        for source_kind, consumer_role in sorted(
            set(current_families) | set(candidate_families) | set(family_guards)
        ):
            current_row = current_families.get((source_kind, consumer_role), {})
            candidate_row = candidate_families.get((source_kind, consumer_role), {})
            guard_row = family_guards.get((source_kind, consumer_role), {})
            metrics: dict[str, Any] = {}
            for metric_name in family_metric_names:
                current_value = current_row.get(metric_name)
                candidate_value = candidate_row.get(metric_name)
                metrics[metric_name] = {
                    "current": current_value,
                    "candidate": candidate_value,
                    "candidate_minus_current": delta(candidate_value, current_value),
                }
            family_rows.append(
                {
                    "source_kind": source_kind,
                    "consumer_role": consumer_role,
                    "metrics": metrics,
                    "guard_passed": guard_row.get("passed"),
                    "feature_count_matches": guard_row.get("feature_count_matches"),
                    "minimum_candidate_score": guard_row.get("minimum_candidate_score"),
                }
            )

        def predictive_rows_by_identity(
            value: Any,
        ) -> dict[tuple[str, str, str], Mapping[str, Any]]:
            rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
            if not isinstance(value, list):
                return rows
            for raw_row in value:
                if not isinstance(raw_row, Mapping):
                    continue
                identity = (
                    str(raw_row.get("input_kind") or "").strip(),
                    str(raw_row.get("source_kind") or "").strip(),
                    str(raw_row.get("consumer_role") or "").strip(),
                )
                if all(identity):
                    rows[identity] = raw_row
            return rows

        current_predictive = predictive_rows_by_identity(
            nested(current, "upstream_predictive_family_ablations")
        )
        candidate_predictive = predictive_rows_by_identity(
            nested(candidate, "upstream_predictive_family_ablations")
        )
        predictive_guards = predictive_rows_by_identity(
            nested(
                sanitized,
                "guards",
                "upstream_predictive_family_ablations",
                "by_family",
            )
        )
        predictive_rows: list[dict[str, Any]] = []
        for input_kind, source_kind, consumer_role in sorted(
            set(current_predictive) | set(candidate_predictive) | set(predictive_guards)
        ):
            identity = (input_kind, source_kind, consumer_role)
            current_row = current_predictive.get(identity, {})
            candidate_row = candidate_predictive.get(identity, {})
            guard_row = predictive_guards.get(identity, {})
            current_delta = current_row.get("weighted_r_loss_delta_when_removed")
            candidate_delta = candidate_row.get("weighted_r_loss_delta_when_removed")
            predictive_rows.append(
                {
                    "input_kind": input_kind,
                    "source_kind": source_kind,
                    "consumer_role": consumer_role,
                    "current_weighted_r_loss_delta_when_removed": current_delta,
                    "candidate_weighted_r_loss_delta_when_removed": candidate_delta,
                    "candidate_minus_current_ablation_delta": delta(candidate_delta, current_delta),
                    "current_normalized_predictive_importance": guard_row.get(
                        "current_normalized_predictive_importance"
                    ),
                    "candidate_normalized_predictive_importance": guard_row.get(
                        "candidate_normalized_predictive_importance"
                    ),
                    "minimum_candidate_normalized_predictive_importance": guard_row.get(
                        "minimum_candidate_normalized_predictive_importance"
                    ),
                    "guard_passed": guard_row.get("passed"),
                    "predictive_refit_performed": True,
                }
            )

        accepted = bool(sanitized.get("accepted"))
        feedback: dict[str, Any] = {
            "kind": "prior_gate_feedback",
            "prior_review_round": int(review_round),
            "proposal_status": "accepted" if accepted else "rejected",
            "proposal_response_sha256": str(response_sha256),
            "prior_operations": [
                {
                    "action": str(row.get("action") or ""),
                    "target_names": list(map(str, row.get("target_names") or ())),
                }
                for row in operation_audit
            ],
            "decision_reasons": list(map(str, sanitized.get("reasons") or ())),
            "failed_guard_paths": list(dict.fromkeys(failed_guards)),
            "objective": {
                "current_r_loss_ratio": current_r_ratio,
                "candidate_r_loss_ratio": candidate_r_ratio,
                "r_loss_ratio_delta": delta(candidate_r_ratio, current_r_ratio),
                "current_weighted_r_loss": current_weighted_r,
                "candidate_weighted_r_loss": candidate_weighted_r,
                "weighted_r_loss_delta": delta(candidate_weighted_r, current_weighted_r),
                "current_penalized_score": current_score,
                "candidate_penalized_score": candidate_score,
                "penalized_score_delta": delta(candidate_score, current_score),
            },
            "complexity": complexity,
            "opaque_calibrated_source_preservation": calibrated_sources,
            "feature_bank_preservation_by_consumer_role": role_rows,
            "feature_bank_preservation_by_source_kind_and_consumer_role": family_rows,
            "upstream_predictive_family_ablations": predictive_rows,
            "non_repeat_guidance": (
                "Build on the accepted registry; change it again only for a newly "
                "diagnosed observable problem."
                if accepted
                else "Do not repeat the prior operations unchanged; address the failed "
                "guards or choose a materially different contract, category, or causal role."
            ),
            "source_and_feature_names_available": False,
            "row_level_values_available": False,
            "gate_rows_or_labels_available": False,
            "sanitized_gate_decision_sha256": sanitized.get("sanitized_decision_sha256"),
        }
        feedback["feedback_sha256"] = _content_sha256(feedback)
        return json.loads(_canonical_json(feedback))

    @staticmethod
    def _quality_retry_feedback_diagnostic(
        candidate_quality: Mapping[str, Any],
        *,
        review_round: int,
        failed_attempt: int,
        response_sha256: str,
        operation_audit: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        diagnostics: list[dict[str, Any]] = []
        for raw in candidate_quality.get("diagnostics") or ():
            if not isinstance(raw, Mapping):
                continue
            row = {key: value for key, value in raw.items() if key != "diagnostic_id"}
            diagnostics.append(json.loads(_canonical_json(row)))
        feedback: dict[str, Any] = {
            "kind": "candidate_quality_retry_feedback",
            "review_round": int(review_round),
            "failed_attempt": int(failed_attempt),
            "proposal_response_sha256": str(response_sha256),
            "failed_contract_names": list(map(str, candidate_quality.get("failed_names") or ())),
            "failed_candidate_quality": diagnostics,
            "prior_operations": [
                {
                    "action": str(row.get("action") or ""),
                    "target_names": list(map(str, row.get("target_names") or ())),
                }
                for row in operation_audit
            ],
            "non_repeat_guidance": (
                "The prior candidate failed spent-only extraction quality. Do not repeat "
                "it unchanged; repair its extraction contract/categories or "
                "choose a materially different operation."
            ),
            "same_gate_remains_sealed": True,
            "gate_rows_or_labels_available": False,
            "row_level_values_available": False,
        }
        feedback["feedback_sha256"] = _content_sha256(feedback)
        return json.loads(_canonical_json(feedback))

    @staticmethod
    def _ontology_retry_feedback_diagnostic(
        ontology_guard: Mapping[str, Any],
        *,
        review_round: int,
        failed_attempt: int,
        response_sha256: str,
        operation_audit: Sequence[Mapping[str, Any]],
        proposal_kind: str,
        workspace_advanced: bool = False,
    ) -> Mapping[str, Any]:
        diagnostics: list[dict[str, Any]] = []
        blocking_failures = set(
            map(
                str,
                ontology_guard.get("hard_failure_policy")
                or ("alternative_category_only_value_support",),
            )
        )
        for raw in ontology_guard.get("diagnostics") or ():
            if not isinstance(raw, Mapping):
                continue
            if not blocking_failures.intersection(set(map(str, raw.get("hard_failures") or ()))):
                continue
            row = {key: value for key, value in raw.items() if key != "diagnostic_id"}
            diagnostics.append(json.loads(_canonical_json(row)))
        feedback: dict[str, Any] = {
            "kind": "retained_registry_ontology_retry_feedback",
            "review_round": int(review_round),
            "failed_attempt": int(failed_attempt),
            "proposal_response_sha256": str(response_sha256),
            "proposal_kind": str(proposal_kind),
            "candidate_workspace_advanced": bool(workspace_advanced),
            "candidate_workspace_accepted": False,
            "ontology_mismatched_contract_names": list(
                map(str, ontology_guard.get("failed_names") or ())
            ),
            "failed_retained_registry_ontology": diagnostics,
            "blocking_failure_kinds": sorted(blocking_failures),
            "prior_operations": [
                {
                    "action": str(row.get("action") or ""),
                    "target_names": list(map(str, row.get("target_names") or ())),
                }
                for row in operation_audit
            ],
            "non_repeat_guidance": (
                "The prior proposal safely reduced the spent-only hard-failure set and "
                "was staged without gate access. Build on the supplied workspace; do "
                "not recreate already staged edits. Resolve every remaining listed "
                "contract before convergence."
                if workspace_advanced
                else "The proposed retained registry still has repeated locally grounded "
                "evidence for a different declared category on spent rows. The workspace did not "
                "advance. Do not stop, make a role-only edit, or leave those contracts "
                "unchanged; drop, replace, or revise their extraction semantics before "
                "convergence."
            ),
            "same_gate_remains_sealed": True,
            "gate_rows_or_labels_available": False,
            "row_level_values_available": False,
        }
        feedback["feedback_sha256"] = _content_sha256(feedback)
        return json.loads(_canonical_json(feedback))

    @staticmethod
    def _adaptive_feature_from_spec(
        spec: Mapping[str, Any],
        *,
        supporting_evidence_ids: Sequence[str],
        evidence_family_by_id: Mapping[str, str],
    ) -> AdaptiveCurrentFeature:
        canonical = CandidateContract(spec).extraction_spec
        support = tuple(dict.fromkeys(map(str, supporting_evidence_ids)))
        if not support:
            raise ValueError("adaptive registry features require authenticated support")
        missing = [
            evidence_id for evidence_id in support if evidence_id not in evidence_family_by_id
        ]
        if missing:
            raise ValueError(
                "adaptive registry provenance cites unknown evidence IDs: " f"{sorted(missing)}"
            )
        cited_families = {evidence_family_by_id[evidence_id] for evidence_id in support}
        families = tuple(
            family for family in ACTIVE_STAGE1_CONCEPT_FAMILIES if family in cited_families
        )
        if cited_families != set(families):
            raise ValueError("adaptive registry provenance cites an inactive architecture")
        description = str(canonical["description"])
        return AdaptiveCurrentFeature(
            feature_name=str(canonical["name"]),
            description=description,
            value_shape_hypothesis=(
                "continuous" if canonical["type"] == "continuous" else "categorical"
            ),
            source_families=families,
            supporting_evidence_ids=support,
            definition_summary=description,
        )

    @classmethod
    def _initial_adaptive_registry(
        cls,
        *,
        specs: Sequence[Mapping[str, Any]],
        frozen_review_evidence: FrozenHierarchicalReviewEvidence,
        initial_catalog: RoleNeutralEvidenceCatalog,
    ) -> tuple[
        tuple[AdaptiveCurrentFeature, ...],
        dict[str, str],
        dict[str, Any],
    ]:
        frozen_review_evidence.__post_init__()
        if frozen_review_evidence.catalog_sha256 != initial_catalog.catalog_sha256:
            raise ValueError("frozen review provenance cites another initial catalog")
        family_by_id = {atom.evidence_id: atom.source_family for atom in initial_catalog.atoms}
        raw_support = frozen_review_evidence.audit.get("accepted_feature_support")
        if not isinstance(raw_support, list):
            raise ValueError("frozen review evidence lost accepted feature provenance")
        support_by_name: dict[str, tuple[str, ...]] = {}
        for row in raw_support:
            if not isinstance(row, Mapping) or set(row) != {
                "canonical_name",
                "supporting_evidence_ids",
            }:
                raise ValueError("frozen accepted feature provenance has a wrong shape")
            name = str(row["canonical_name"])
            ids = tuple(map(str, row["supporting_evidence_ids"]))
            if not ids or name in support_by_name:
                raise ValueError("frozen accepted feature provenance is incomplete or repeated")
            support_by_name[name] = ids
        canonical_specs = [CandidateContract(spec).extraction_spec for spec in specs]
        names = [str(spec["name"]) for spec in canonical_specs]
        if len(names) != len(set(names)):
            raise ValueError("initial executable registry contains duplicate feature names")
        if not set(names) <= set(support_by_name):
            raise ValueError(
                "initial executable registry is not covered by frozen accepted provenance"
            )
        registry = tuple(
            cls._adaptive_feature_from_spec(
                spec,
                supporting_evidence_ids=support_by_name[str(spec["name"])],
                evidence_family_by_id=family_by_id,
            )
            for spec in canonical_specs
        )
        excluded = sorted(set(support_by_name) - set(names))
        audit = {
            "modeled_registry_names_sha256": _content_sha256(names),
            "frozen_accepted_feature_count": len(support_by_name),
            "modeled_feature_count": len(names),
            "excluded_nonmodeled_accepted_feature_count": len(excluded),
            "excluded_nonmodeled_accepted_feature_names_sha256": _content_sha256(excluded),
            "modeled_specs_are_unique_subset_of_frozen_accepted_support": True,
            "excluded_nonmodeled_features_treated_as_executable": False,
        }
        return registry, family_by_id, audit

    @classmethod
    def _transition_adaptive_registry(
        cls,
        *,
        before: Sequence[AdaptiveCurrentFeature],
        after_specs: Sequence[Mapping[str, Any]],
        operation_audit: Sequence[Mapping[str, Any]],
        evidence_family_by_id: Mapping[str, str],
    ) -> tuple[AdaptiveCurrentFeature, ...]:
        """Apply exact support-provenance algebra in candidate-workspace order."""

        support_by_name = {
            item.feature_name: tuple(item.supporting_evidence_ids) for item in before
        }

        def union_support(*groups: Sequence[str]) -> tuple[str, ...]:
            return tuple(dict.fromkeys(value for group in groups for value in group))

        for index, raw in enumerate(operation_audit):
            if not isinstance(raw, Mapping):
                raise TypeError("adaptive provenance operation audit must contain mappings")
            adaptive_kind = raw.get("adaptive_operation")
            legacy_kind = raw.get("action")
            if (adaptive_kind is None) == (legacy_kind is None):
                raise ValueError("provenance operation must identify exactly one operation kind")
            kind = str(adaptive_kind if adaptive_kind is not None else legacy_kind)
            targets = tuple(map(str, raw.get("target_names") or ()))
            citations = tuple(map(str, raw.get("supporting_evidence_ids") or ()))
            unknown_citations = set(citations) - set(evidence_family_by_id)
            if unknown_citations:
                raise ValueError(
                    f"operation {index} cites unavailable provenance: "
                    f"{sorted(unknown_citations)}"
                )
            if kind == "stop":
                if targets or citations:
                    raise ValueError("stop cannot alter adaptive provenance")
                continue
            if not targets:
                raise ValueError(f"operation {index} has no provenance target")
            missing_targets = set(targets) - set(support_by_name)
            if kind != "add" and missing_targets:
                raise ValueError(
                    f"operation {index} targets unavailable registry provenance: "
                    f"{sorted(missing_targets)}"
                )
            if kind == "drop":
                if adaptive_kind is not None and citations:
                    raise ValueError("drop cannot add adaptive support provenance")
                for target in targets:
                    support_by_name.pop(target)
                continue
            contract = raw.get("contract")
            if not isinstance(contract, Mapping):
                raise ValueError(f"operation {index} requires an executable contract")
            result_name = str(contract.get("name") or "")
            if not result_name:
                raise ValueError(f"operation {index} executable contract has no name")
            if kind == "re_role":
                if len(targets) != 1 or result_name != targets[0]:
                    raise ValueError("re_role must preserve exact historical provenance")
                continue
            if kind in {"add", "split", "replace"}:
                if not citations:
                    raise ValueError(f"{kind} requires current cited support provenance")
                if kind == "replace":
                    for target in targets:
                        support_by_name.pop(target)
                elif kind == "split":
                    if len(targets) != 1 or result_name in support_by_name:
                        raise ValueError("split provenance is malformed")
                elif result_name in support_by_name:
                    raise ValueError("add provenance collides with the current registry")
                support_by_name[result_name] = citations
                continue
            if kind in {"revise", "rename", "revise_definition", "merge"}:
                if not citations:
                    raise ValueError(f"{kind} requires current cited support provenance")
                historical = union_support(*(support_by_name[target] for target in targets))
                for target in targets:
                    support_by_name.pop(target)
                support_by_name[result_name] = union_support(historical, citations)
                continue
            raise ValueError(f"unsupported adaptive provenance operation: {kind}")

        canonical_after = [CandidateContract(spec).extraction_spec for spec in after_specs]
        after_names = [str(spec["name"]) for spec in canonical_after]
        if set(after_names) != set(support_by_name) or len(after_names) != len(support_by_name):
            raise ValueError("candidate specs and adaptive provenance registry diverged")
        return tuple(
            cls._adaptive_feature_from_spec(
                spec,
                supporting_evidence_ids=support_by_name[str(spec["name"])],
                evidence_family_by_id=evidence_family_by_id,
            )
            for spec in canonical_after
        )

    @staticmethod
    def _adaptive_diagnostics(
        diagnostics: Sequence[Mapping[str, Any]],
        *,
        current_registry: Sequence[AdaptiveCurrentFeature],
    ) -> tuple[tuple[AdaptiveDiagnostic, ...], dict[str, Any]]:
        """Convert every real legacy diagnostic to a lossless semantic scalar row."""

        kind_map = {
            "feature_quality": "extraction_missingness",
            "extraction_text_grounding": "extraction_validity",
            "redundancy": "redundancy",
            "nested_observable_causal_quality": "nuisance_residual",
            "contract_ablation": "heterogeneity",
            "prior_gate_feedback": "source_preservation",
            "candidate_quality_retry_feedback": "extraction_validity",
            "retained_registry_ontology_retry_feedback": "extraction_validity",
            "review_response_validation_retry_feedback": "extraction_validity",
        }
        summaries = {
            "extraction_missingness": "Observable extraction coverage and validity diagnostic.",
            "extraction_validity": "Observable extraction-grounding or retry diagnostic.",
            "redundancy": "Observable pairwise feature redundancy diagnostic.",
            "nuisance_residual": "Nested observable nuisance-fit diagnostic.",
            "heterogeneity": "Observable contract-ablation diagnostic.",
            "source_preservation": "Prior untouched-gate source-preservation diagnostic.",
        }
        targets_by_id = collect_post_extraction_diagnostic_targets(diagnostics)
        registry_names = {item.feature_name for item in current_registry}
        rows_by_id: dict[str, Mapping[str, Any]] = {}

        def collect(value: Any) -> None:
            if isinstance(value, Mapping):
                if "diagnostic_id" in value:
                    diagnostic_id = str(value["diagnostic_id"])
                    if diagnostic_id in rows_by_id:
                        raise ValueError("adaptive diagnostics contain a duplicate ID")
                    rows_by_id[diagnostic_id] = value
                for key, child in value.items():
                    if key != "diagnostic_id":
                        collect(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect(child)

        collect(diagnostics)
        if tuple(rows_by_id) != tuple(targets_by_id):
            raise RuntimeError("adaptive diagnostic traversal differs from target mapping")

        forbidden_path = re.compile(
            r"(?:^|_)(?:row|fold|gate|source|provider|identity|sha256|temporal|date|"
            r"patient|record|note|oracle|treatment|outcome)(?:_|$)",
            flags=re.IGNORECASE,
        )
        metric_key = re.compile(
            r"(?:^|_)(?:coverage|missingness|rate|count|delta|loss|ratio|score|"
            r"passed|applicable|agreement|association|complexity|warning|failure|"
            r"minimum|maximum|median|std|importance)(?:_|$)",
            flags=re.IGNORECASE,
        )

        def scalar_metrics(
            row: Mapping[str, Any],
            *,
            diagnostic_id: str,
        ) -> tuple[
            dict[str, int | float | bool | None],
            dict[str, Any],
        ]:
            eligible: list[tuple[str, int | float | bool | None]] = []
            emitted_names: set[str] = set()

            def visit(value: Any, path: tuple[str, ...]) -> None:
                if isinstance(value, Mapping):
                    items = list(value.items())
                    raw_keys = [key for key, _child in items]
                    if not all(isinstance(key, str) for key in raw_keys):
                        raise TypeError("adaptive diagnostic mapping keys must be strings")
                    if len(set(raw_keys)) != len(raw_keys):
                        raise ValueError("adaptive diagnostic mapping contains duplicate keys")
                    for key, child in sorted(items, key=lambda item: item[0].encode("utf-8")):
                        lowered = key.strip().lower()
                        if lowered in {"diagnostic_id", "kind"}:
                            continue
                        visit(child, (*path, key))
                    return
                if isinstance(value, (list, tuple)):
                    joined = "_".join(part.strip().lower() for part in path)
                    if (
                        joined
                        and not forbidden_path.search(joined)
                        and metric_key.search(joined)
                    ):
                        raise ValueError(
                            "adaptive diagnostic metric-like collections must be "
                            "pre-aggregated to scalar values"
                        )
                    return
                if value is not None and not isinstance(value, (bool, int, float, np.generic)):
                    return
                if isinstance(value, np.generic):
                    value = value.item()
                if value is not None and not isinstance(value, (bool, int, float)):
                    return
                if isinstance(value, float) and not math.isfinite(value):
                    return
                joined = "_".join(part.strip().lower() for part in path)
                if not joined or forbidden_path.search(joined) or not metric_key.search(joined):
                    return
                key = ".".join(_encode_adaptive_metric_path_segment(part) for part in path)
                if _ADAPTIVE_METRIC_IDENTIFIER.fullmatch(key) is None:
                    raise ValueError("adaptive diagnostic metric path encoding is invalid")
                if key in emitted_names:
                    raise ValueError("adaptive diagnostic metric paths collide")
                emitted_names.add(key)
                eligible.append((key, value))

            visit(row, ())
            eligible.sort(key=lambda item: item[0])
            metrics = {key: value for key, value in eligible}
            ordered_keys = list(metrics)
            inventory = [
                {"metric_key": key, "value": metrics[key]}
                for key in ordered_keys
            ]
            proof_identity = {
                "schema_version": ADAPTIVE_DIAGNOSTIC_METRIC_COVERAGE_SCHEMA_VERSION,
                "diagnostic_id": diagnostic_id,
                "path_encoding_version": ADAPTIVE_DIAGNOSTIC_METRIC_PATH_ENCODING_VERSION,
                "aggregate_metrics": metrics,
                "ordered_metric_keys": ordered_keys,
                "eligible_metric_count": len(eligible),
                "emitted_metric_count": len(metrics),
                "eligible_metric_inventory_sha256": _content_sha256(inventory),
                "emitted_metrics_sha256": _content_sha256(metrics),
                "metric_names_unique": len(metrics) == len(emitted_names),
                "every_eligible_metric_emitted_once": len(eligible) == len(metrics),
            }
            proof = {
                **proof_identity,
                "coverage_proof_sha256": _content_sha256(proof_identity),
            }
            _validate_adaptive_diagnostic_metric_coverage_proof(proof)
            return metrics, proof

        adapted: list[AdaptiveDiagnostic] = []
        metric_coverage_proofs: list[dict[str, Any]] = []
        historical_kinds = {
            "prior_gate_feedback",
            "candidate_quality_retry_feedback",
            "retained_registry_ontology_retry_feedback",
            "review_response_validation_retry_feedback",
        }
        excluded_historical_targets: dict[str, list[str]] = {}
        for diagnostic_id, row in rows_by_id.items():
            kind = str(row.get("kind") or "")
            if kind not in kind_map:
                raise ValueError(f"unsupported adaptive diagnostic kind: {kind or '<missing>'}")
            direct_targets = list(targets_by_id[diagnostic_id])
            for extra_field in ("ontology_mismatched_contract_names",):
                raw = row.get(extra_field)
                if isinstance(raw, (list, tuple)):
                    direct_targets.extend(str(value) for value in raw)
            targets = tuple(dict.fromkeys(direct_targets))
            unknown = set(targets) - registry_names
            if unknown:
                if kind not in historical_kinds:
                    raise ValueError(
                        "adaptive diagnostic targets absent registry features: "
                        f"{sorted(unknown)}"
                    )
                excluded_historical_targets[diagnostic_id] = sorted(unknown)
                targets = tuple(target for target in targets if target in registry_names)
            diagnostic_kind = kind_map[kind]
            aggregate_metrics, metric_coverage_proof = scalar_metrics(
                row,
                diagnostic_id=diagnostic_id,
            )
            adapted.append(
                AdaptiveDiagnostic(
                    diagnostic_id=diagnostic_id,
                    diagnostic_kind=diagnostic_kind,
                    affected_features=targets,
                    summary=summaries[diagnostic_kind],
                    aggregate_metrics=aggregate_metrics,
                )
            )
            metric_coverage_proofs.append(metric_coverage_proof)
        if not adapted:
            raise ValueError("adaptive reconsideration requires observable diagnostics")
        audit_identity = {
            "schema_version": ADAPTIVE_DIAGNOSTIC_ADAPTER_AUDIT_SCHEMA_VERSION,
            "input_diagnostic_count": len(rows_by_id),
            "adapted_diagnostic_count": len(adapted),
            "every_diagnostic_id_represented_once": len(rows_by_id) == len(adapted),
            "excluded_historical_target_count": sum(
                len(values) for values in excluded_historical_targets.values()
            ),
            "excluded_historical_targets_by_diagnostic": excluded_historical_targets,
            "unknown_current_diagnostic_targets_fail_closed": True,
            "model_context_contains_excluded_historical_names": False,
            "metric_coverage_proof_count": len(metric_coverage_proofs),
            "total_eligible_metric_count": sum(
                int(proof["eligible_metric_count"]) for proof in metric_coverage_proofs
            ),
            "total_emitted_metric_count": sum(
                int(proof["emitted_metric_count"]) for proof in metric_coverage_proofs
            ),
            "metric_names_unique_within_each_diagnostic": all(
                proof["metric_names_unique"] is True for proof in metric_coverage_proofs
            ),
            "every_eligible_metric_emitted_once": all(
                proof["every_eligible_metric_emitted_once"] is True
                for proof in metric_coverage_proofs
            ),
            "metric_coverage_proofs": metric_coverage_proofs,
        }
        audit = {**audit_identity, "audit_sha256": _content_sha256(audit_identity)}
        _validate_adaptive_diagnostic_adapter_audit(audit)
        return tuple(adapted), audit

    def _run_post_extraction_review(
        self,
        *,
        data: pd.DataFrame,
        label_free: pd.DataFrame,
        outer_fold: int,
        train_ids: tuple[int, ...],
        initial_specs: Sequence[Mapping[str, Any]],
        initial_extracted: pd.DataFrame,
        fold_dir: Path,
        review_schedule: ReviewPartitionSchedule | None = None,
        initial_selector_evidence_audit: Mapping[str, Any] | None = None,
        frozen_hierarchical_review_evidence: FrozenHierarchicalReviewEvidence | None = None,
        hierarchical_first_gate_materialization_intent: (
            FirstGateMaterializationIntent | None
        ) = None,
        hierarchical_first_gate_catalog: RoleNeutralEvidenceCatalog | None = None,
        hierarchical_approved_runner_identity: Mapping[str, Any] | None = None,
        hierarchical_approved_cache_identity: Mapping[str, Any] | None = None,
        hierarchical_family_explanations: Mapping[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], pd.DataFrame, Mapping[str, Any]]:
        rounds = int(self.config.post_extraction_review_rounds)
        max_quality_retries = int(self.config.post_extraction_review_max_quality_retries)
        canonical_initial = [CandidateContract(spec).extraction_spec for spec in initial_specs]
        hierarchical_review = frozen_hierarchical_review_evidence is not None
        if hierarchical_review:
            assert frozen_hierarchical_review_evidence is not None
            frozen_hierarchical_review_evidence.__post_init__()
            if self.gate_only_reference_review:
                if hierarchical_first_gate_materialization_intent is not None:
                    raise ValueError(
                        "gate-only review rejects a conditional first-gate "
                        "materialization intent"
                    )
            else:
                if not isinstance(
                    hierarchical_first_gate_materialization_intent,
                    FirstGateMaterializationIntent,
                ) or not isinstance(hierarchical_first_gate_catalog, RoleNeutralEvidenceCatalog):
                    raise ValueError(
                        "hierarchical review requires the approved first-gate "
                        "materialization intent and its exact semantic catalog"
                    )
                hierarchical_first_gate_materialization_intent.verify()
                if (
                    hierarchical_first_gate_materialization_intent.body["semantic_catalog"][
                        "catalog_sha256"
                    ]
                    != hierarchical_first_gate_catalog.catalog_sha256
                ):
                    raise ValueError("hierarchical first-gate intent cites a different catalog")
            if not isinstance(hierarchical_approved_runner_identity, Mapping) or not isinstance(
                hierarchical_approved_cache_identity, Mapping
            ):
                raise ValueError(
                    "hierarchical review requires the initially approved runner and cache "
                    "identities"
                )
            if not isinstance(hierarchical_family_explanations, Mapping) or set(
                hierarchical_family_explanations
            ) != set(ACTIVE_STAGE1_CONCEPT_FAMILIES):
                raise ValueError(
                    "hierarchical review requires the exact prepared ten-family explanations"
                )
            if (
                self.hierarchical_discovery_runner is None
                or self.hierarchical_discovery_job_cache_root is None
                or self.hierarchical_review_evidence_policy is None
            ):
                raise RuntimeError("hierarchical adaptive execution dependencies are missing")
            self.hierarchical_review_evidence_policy.validate_authentication()
            adaptive_config = self.hierarchical_review_evidence_policy.adaptive_config()
            if adaptive_config.max_operations != self.config.post_extraction_review_max_operations:
                raise ValueError(
                    "adaptive hierarchy operation bound differs from the review runner"
                )
        elif (
            hierarchical_first_gate_materialization_intent is not None
            or hierarchical_first_gate_catalog is not None
            or hierarchical_approved_runner_identity is not None
            or hierarchical_approved_cache_identity is not None
            or hierarchical_family_explanations is not None
        ):
            raise ValueError(
                "hierarchical first-gate intent/catalog are forbidden without frozen review "
                "evidence"
            )
        if rounds == 0:
            return (
                canonical_initial,
                initial_extracted,
                {
                    "enabled": False,
                    "configured_rounds": 0,
                    "rounds_completed": 0,
                    "stopped_by_agent": False,
                    "source_text_temporal_policy": source_text_temporal_policy_audit(),
                    "configured_max_quality_retries_per_gate": max_quality_retries,
                    "required_source_signals": self.config.require_review_source_signals,
                    "required_feature_banks": self.config.require_review_feature_banks,
                },
            )

        outer_train = (
            data.set_index("_oci_row_id", drop=False).loc[list(train_ids)].reset_index(drop=True)
        )
        schedule = review_schedule or self._review_schedule(
            outer_train=outer_train,
            outer_fold=outer_fold,
        )
        scheduled_rows = [row_id for rows in schedule.row_ids_by_fold.values() for row_id in rows]
        if (
            schedule.outer_fold != int(outer_fold)
            or len(scheduled_rows) != len(set(scheduled_rows))
            or set(scheduled_rows) != set(map(int, train_ids))
        ):
            raise ValueError("injected review schedule does not match the exact outer train")
        if len(schedule.gate_fold_ids) != rounds:
            raise RuntimeError("review schedule does not contain one fresh gate per round")
        fold_by_row = {
            row_id: fold_id for fold_id, rows in schedule.row_ids_by_fold.items() for row_id in rows
        }

        initial_spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
        supplied_initial_ids = tuple(map(int, initial_extracted["_oci_row_id"].tolist()))
        if supplied_initial_ids != initial_spent_ids:
            raise ValueError(
                "adaptive review initial extraction must contain exactly the ordered "
                "initial-spent rows; sealed gates and outer-heldout rows are forbidden"
            )
        initial_spent_label_free = (
            label_free.set_index("_oci_row_id", drop=False)
            .loc[list(initial_spent_ids)]
            .reset_index(drop=True)
        )
        initial_extracted = self._validated_extraction_projection(
            initial_extracted,
            label_free=initial_spent_label_free,
            specs=canonical_initial,
            source="initial-spent adaptive extraction",
        )

        spent_fold_ids = list(schedule.initial_spent_fold_ids)
        current_specs = canonical_initial
        current_extracted = initial_extracted
        current_adaptive_registry: tuple[AdaptiveCurrentFeature, ...] | None = None
        adaptive_evidence_family_by_id: dict[str, str] = {}
        initial_adaptive_registry_audit: Mapping[str, Any] | None = None
        if hierarchical_review:
            assert frozen_hierarchical_review_evidence is not None
            assert hierarchical_first_gate_catalog is not None
            (
                current_adaptive_registry,
                adaptive_evidence_family_by_id,
                initial_adaptive_registry_audit,
            ) = self._initial_adaptive_registry(
                specs=current_specs,
                frozen_review_evidence=frozen_hierarchical_review_evidence,
                initial_catalog=hierarchical_first_gate_catalog,
            )
        initial_hashes = [extraction_contract_sha256(spec) for spec in current_specs]
        round_records: list[dict[str, Any]] = []
        prior_gate_feedback: list[Mapping[str, Any]] = []
        stopped_by_agent = False
        quality_retry_exhausted = False
        valid_operation_proposals = 0
        gate_evaluated_proposals = 0
        candidate_quality_rejections = 0
        candidate_quality_retries = 0
        retained_ontology_rejections = 0
        retained_ontology_retries = 0
        unresolved_ontology_convergence_rejections = 0
        unresolved_ontology_convergence_retry_exhausted = False
        response_validation_rejections = 0
        response_validation_retries = 0
        response_validation_retry_exhausted = False
        candidate_workspace_stage_count = 0
        total_review_attempts = 0
        adaptive_execution_records: list[dict[str, Any]] = []

        def persist_attempt(
            *,
            attempt_dir: Path,
            body: Mapping[str, Any],
            status: str,
            attempt_index: int,
            gate_accessed: bool,
            gate_consumed: bool,
        ) -> dict[str, Any]:
            path = attempt_dir / "immutable_review_round.json"
            content_sha256 = _write_immutable_json(
                path,
                body,
                schema=POST_EXTRACTION_REVIEW_ROUND_SCHEMA_VERSION,
            )
            return {
                "attempt": int(attempt_index),
                "status": str(status),
                "path": str(path.resolve()),
                "content_sha256": content_sha256,
                "gate_accessed": bool(gate_accessed),
                "gate_consumed": bool(gate_consumed),
            }

        def finish_round(
            *,
            round_index: int,
            terminal_attempt: Mapping[str, Any],
            attempts: Sequence[Mapping[str, Any]],
            extra: Mapping[str, Any] | None = None,
        ) -> None:
            record = {
                "round": int(round_index),
                "status": str(terminal_attempt["status"]),
                "path": str(terminal_attempt["path"]),
                "content_sha256": str(terminal_attempt["content_sha256"]),
                "attempt_count": len(attempts),
                "attempt_audits": [dict(row) for row in attempts],
            }
            if extra:
                record.update(dict(extra))
            round_records.append(record)

        for round_index, gate_fold_id in enumerate(schedule.gate_fold_ids, start=1):
            spent_ids = schedule.row_ids(spent_fold_ids)
            spent_label_free = (
                label_free.set_index("_oci_row_id", drop=False)
                .loc[list(spent_ids)]
                .reset_index(drop=True)
            )
            spent_texts = tuple(spent_label_free[self.config.text_column].astype(str).tolist())
            current_spent_extracted = self._select_extraction_rows(
                current_extracted,
                label_free=spent_label_free,
                specs=current_specs,
                source="accumulated spent extraction",
            )
            spent = self._observable_review_rows(
                row_ids=spent_ids,
                extracted=current_spent_extracted,
                data=data,
                fold_by_row=fold_by_row,
            )
            accepted_round_specs = [
                CandidateContract(spec).extraction_spec for spec in current_specs
            ]
            accepted_round_spent_extracted = current_spent_extracted
            accepted_round_spent = spent
            accepted_round_spent_extraction_sha256 = self._extraction_projection_sha256(
                accepted_round_spent_extracted,
                accepted_round_specs,
            )
            accepted_round_adaptive_registry = current_adaptive_registry
            workspace_specs = [
                CandidateContract(spec).extraction_spec for spec in accepted_round_specs
            ]
            workspace_spent_extracted = accepted_round_spent_extracted
            workspace_adaptive_registry = accepted_round_adaptive_registry
            workspace_stage_history: list[dict[str, Any]] = []
            workspace_operation_audit_history: list[dict[str, Any]] = []
            adaptive_catalog: RoleNeutralEvidenceCatalog | None = None
            exact_spent_authentication: ExactSpentCatalogAuthentication | None = None
            if hierarchical_review:
                assert frozen_hierarchical_review_evidence is not None
                assert workspace_adaptive_registry is not None
                evidence_catalog = list(frozen_hierarchical_review_evidence.review_rows)
                sealed_row_count = len(train_ids) - len(spent_ids)
                if round_index == 1:
                    spent_evidence_audit = {
                        "review_round": int(round_index),
                        "consumer_review_round": int(round_index),
                        "spent_evidence_context_epoch": int(round_index - 1),
                        "provider_review_round_argument": int(round_index - 1),
                        "consumed_gate_count_before_context_fit": int(round_index - 1),
                        "context_epoch_policy_version": (
                            SPENT_EVIDENCE_CONTEXT_EPOCH_POLICY_VERSION
                        ),
                        "spent_row_count": len(spent_ids),
                        "sealed_row_count": sealed_row_count,
                        "spent_row_fingerprint": row_set_fingerprint(spent_ids),
                        "sealed_row_fingerprint": row_set_fingerprint(
                            tuple(row_id for row_id in train_ids if row_id not in set(spent_ids))
                        ),
                        "provider_identity_sha256": (
                            frozen_hierarchical_review_evidence.binding_sha256
                        ),
                        "source_kinds": sorted(
                            {
                                family
                                for row in evidence_catalog
                                for family in row["source_families"]
                            }
                        ),
                        "future_gate_text_or_labels_supplied_to_provider": False,
                        "full_outer_discovery_evidence_used": False,
                        "hierarchical_frozen_accepted_support": True,
                        "round_1_initial_frozen_support_only": True,
                        "later_round_fresh_exact_spent_catalog": False,
                        "frozen_review_evidence_binding_sha256": (
                            frozen_hierarchical_review_evidence.binding_sha256
                        ),
                        "frozen_review_evidence_sha256": (
                            frozen_hierarchical_review_evidence.review_evidence_sha256
                        ),
                        "dynamic_stage1_semantic_refit_performed": False,
                        "same_frozen_catalog_authorized_for_later_rounds": False,
                    }
                else:
                    spent_inputs, provider_audit, prefit_catalog = self._spent_evidence_inputs(
                        data=data,
                        schedule=schedule,
                        spent_fold_ids=spent_fold_ids,
                        outer_fold=outer_fold,
                        review_round=round_index,
                    )
                    adaptive_catalog = (
                        prefit_catalog
                        if prefit_catalog is not None
                        else build_role_neutral_evidence_catalog(spent_inputs)
                    )
                    family_counts = {
                        family: len(adaptive_catalog.family_atoms(family))
                        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
                    }
                    if any(count < 1 for count in family_counts.values()):
                        raise ValueError(
                            "later adaptive review requires evidence from all ten architectures"
                        )
                    for atom in adaptive_catalog.atoms:
                        prior_family = adaptive_evidence_family_by_id.get(atom.evidence_id)
                        if prior_family is not None and prior_family != atom.source_family:
                            raise ValueError(
                                "content-addressed adaptive evidence changed source architecture"
                            )
                        adaptive_evidence_family_by_id[atom.evidence_id] = atom.source_family
                    partition_by_fold = {
                        int(row["fold_id"]): row for row in schedule.audit["partitions"]
                    }
                    consumed_gate_fingerprints = tuple(
                        str(partition_by_fold[int(fold_id)]["row_fingerprint"])
                        for fold_id in schedule.gate_fold_ids[: round_index - 1]
                    )
                    if len(consumed_gate_fingerprints) != round_index - 1:
                        raise RuntimeError("consumed-gate authentication lost fold order")
                    current_gate_ids = schedule.row_ids((gate_fold_id,))
                    current_gate_fingerprint = row_set_fingerprint(current_gate_ids)
                    if (
                        current_gate_fingerprint
                        != partition_by_fold[int(gate_fold_id)]["row_fingerprint"]
                    ):
                        raise RuntimeError("still-sealed gate fingerprint differs from schedule")
                    spent_evidence_audit = {
                        **dict(provider_audit),
                        "hierarchical_frozen_accepted_support": False,
                        "round_1_initial_frozen_support_only": False,
                        "later_round_fresh_exact_spent_catalog": True,
                        "adaptive_catalog_sha256": adaptive_catalog.catalog_sha256,
                        "adaptive_catalog_split_fingerprint": (adaptive_catalog.split_fingerprint),
                        "adaptive_family_atom_counts": family_counts,
                        "prepared_family_explanations_sha256": _content_sha256(
                            dict(hierarchical_family_explanations or {})
                        ),
                        "all_ten_stage1_architectures_present": True,
                        "complete_catalog_sent_to_legacy_review_agent": False,
                        "legacy_diagnostic_evidence_scope": (
                            "round_1_frozen_accepted_support_local_grounding_only"
                        ),
                        "same_frozen_catalog_authorized_for_later_rounds": False,
                    }
                    upstream_authentication_sha256 = _content_sha256(
                        {
                            "complete_spent_provider_and_semantic_compatibility_audit": (
                                spent_evidence_audit
                            ),
                            "catalog_sha256": adaptive_catalog.catalog_sha256,
                            "catalog_split_fingerprint": adaptive_catalog.split_fingerprint,
                            "family_atom_counts": family_counts,
                        }
                    )
                    exact_spent_authentication = ExactSpentCatalogAuthentication.create(
                        catalog=adaptive_catalog,
                        accumulated_spent_scope_sha256=_content_sha256(
                            {"ordered_spent_row_ids": list(map(int, spent_ids))}
                        ),
                        accumulated_spent_row_count=len(spent_ids),
                        consumed_gate_fingerprints=consumed_gate_fingerprints,
                        still_sealed_gate_fingerprint=current_gate_fingerprint,
                        upstream_authentication_sha256=upstream_authentication_sha256,
                    )
                    spent_evidence_audit = {
                        **spent_evidence_audit,
                        "exact_spent_authentication_sha256": (
                            exact_spent_authentication.authentication_sha256
                        ),
                        "accumulated_spent_scope_sha256": (
                            exact_spent_authentication.accumulated_spent_scope_sha256
                        ),
                        "consumed_gate_fingerprints": list(consumed_gate_fingerprints),
                        "still_sealed_gate_fingerprint": current_gate_fingerprint,
                        "upstream_authentication_sha256": upstream_authentication_sha256,
                    }
            else:
                spent_request, spent_evidence_audit = self._spent_fusion_request(
                    data=data,
                    schedule=schedule,
                    spent_fold_ids=spent_fold_ids,
                    outer_fold=outer_fold,
                    review_round=round_index,
                )
                evidence_catalog = spent_request.context()["evidence"]
            round_dir = fold_dir / "post_extraction_review" / f"round_{round_index:03d}"
            attempt_records: list[dict[str, Any]] = []
            retry_feedback: list[Mapping[str, Any]] = []
            round_finished = False

            for attempt_index in range(1, max_quality_retries + 2):
                total_review_attempts += 1
                workspace_spent = self._observable_review_rows(
                    row_ids=spent_ids,
                    extracted=workspace_spent_extracted,
                    data=data,
                    fold_by_row=fold_by_row,
                )
                workspace_extraction_sha256 = self._extraction_projection_sha256(
                    workspace_spent_extracted,
                    workspace_specs,
                )
                context = self._build_sanitized_review_context(
                    review_round=round_index,
                    review_attempt=attempt_index,
                    spent=workspace_spent,
                    spent_texts=spent_texts,
                    specs=workspace_specs,
                    evidence_catalog=evidence_catalog,
                    spent_evidence_audit=spent_evidence_audit,
                    feedback_diagnostics=[*prior_gate_feedback, *retry_feedback],
                    accepted_round_baseline_specs=accepted_round_specs,
                    workspace_stage_history=workspace_stage_history,
                    workspace_extraction_sha256=workspace_extraction_sha256,
                    frozen_content_addressed_evidence=hierarchical_review,
                )
                workspace_ontology = self._retained_registry_ontology_from_grounding(
                    workspace_specs,
                    [
                        row
                        for row in context["diagnostics"]
                        if str(row.get("kind")) == "extraction_text_grounding"
                    ],
                )
                request_sha256 = _content_sha256(context)
                diagnostic_ids = collect_post_extraction_diagnostic_ids(context["diagnostics"])
                diagnostic_targets = collect_post_extraction_diagnostic_targets(
                    context["diagnostics"]
                )
                evidence_ids = [
                    str(row["evidence_id"])
                    for row in context["sanitized_evidence_catalog"]
                    if isinstance(row, Mapping) and row.get("evidence_id")
                ]

                attempt_dir = round_dir / f"attempt_{attempt_index:03d}"
                request_cache_path = attempt_dir / "immutable_review_request.json"
                request_manifest_sha256 = _write_immutable_json(
                    request_cache_path,
                    {
                        "outer_fold": int(outer_fold),
                        "review_round": int(round_index),
                        "review_attempt": int(attempt_index),
                        "spent_evidence_context_audit": spent_evidence_audit,
                        "request_sha256": request_sha256,
                        "sanitized_context": context,
                        "row_level_values_persisted": False,
                        "raw_reasoning_persisted": False,
                        "outer_heldout_labels_used": False,
                    },
                    schema=POST_EXTRACTION_REVIEW_REQUEST_SCHEMA_VERSION,
                )
                adaptive_attempt = bool(hierarchical_review and round_index >= 2)
                response_cache_path = (
                    attempt_dir / "authenticated_adaptive_hierarchy.json"
                    if adaptive_attempt
                    else attempt_dir / "immutable_review_response.json"
                )
                failure_cache_path = attempt_dir / "immutable_review_failure.json"
                adaptive_execution = None
                adaptive_execution_record: Mapping[str, Any] | None = None
                adaptive_execution_sha256: str | None = None
                adaptive_proposal_freeze_sha256: str | None = None
                adaptive_executable_freeze_sha256: str | None = None
                adaptive_executable_freeze_validated = False
                adaptive_diagnostic_audit: Mapping[str, Any] | None = None
                adaptive_applied: AppliedReviewOperations | None = None
                cached: tuple[Mapping[str, Any], str] | None = None
                response_failure: BaseException | None = None
                cached_failure = _load_request_bound_review_failure(
                    failure_cache_path,
                    request_sha256=request_sha256,
                    review_round=round_index,
                    review_attempt=attempt_index,
                    expected_current_names=[str(spec["name"]) for spec in workspace_specs],
                )
                if adaptive_attempt and cached_failure is not None and response_cache_path.exists():
                    raise RuntimeError(
                        "adaptive review attempt has both an execution and failure artifact"
                    )
                if adaptive_attempt and cached_failure is None:
                    if (
                        adaptive_catalog is None
                        or exact_spent_authentication is None
                        or workspace_adaptive_registry is None
                    ):
                        raise RuntimeError("later adaptive hierarchy lost authenticated inputs")
                    assert hierarchical_family_explanations is not None
                    assert hierarchical_approved_runner_identity is not None
                    assert hierarchical_approved_cache_identity is not None
                    assert self.hierarchical_review_evidence_policy is not None
                    assert self.hierarchical_discovery_job_cache_root is not None
                    adaptive_diagnostics, adaptive_diagnostic_audit = self._adaptive_diagnostics(
                        context["diagnostics"],
                        current_registry=workspace_adaptive_registry,
                    )
                    adaptive_builder = AdaptiveHierarchicalStage1Reconsideration(
                        catalog=adaptive_catalog,
                        exact_spent_authentication=exact_spent_authentication,
                        family_explanations=dict(hierarchical_family_explanations),
                        current_registry=workspace_adaptive_registry,
                        diagnostics=adaptive_diagnostics,
                        config=self.hierarchical_review_evidence_policy.adaptive_config(),
                    )
                    adaptive_cache = AuthenticatedHierarchicalDiscoveryJobCache(
                        root=(
                            self.hierarchical_discovery_job_cache_root
                            / f"outer_fold_{outer_fold:03d}"
                        ),
                        config=self.hierarchical_discovery_job_cache_config,
                    )
                    try:
                        adaptive_execution = adaptive_builder.execute_authenticated(
                            runner=self.hierarchical_discovery_runner,
                            job_cache=adaptive_cache,
                            approved_adaptive_identity=(
                                self.hierarchical_review_evidence_policy.as_dict()[
                                    "adaptive_reconsideration_identity"
                                ]
                            ),
                            approved_runner_identity=hierarchical_approved_runner_identity,
                            approved_cache_identity=hierarchical_approved_cache_identity,
                            current_specs=workspace_specs,
                            max_contracts=self.config.max_candidates,
                        )
                    except (TypeError, ValueError, RuntimeError) as exc:
                        response_failure = exc
                    else:
                        adaptive_applied = adaptive_execution.executable_revision.applied
                        adaptive_execution_record = adaptive_execution.as_dict()
                        adaptive_execution_sha256 = adaptive_execution.execution_sha256
                        adaptive_proposal_freeze_sha256 = (
                            adaptive_execution.frozen_round.freeze_sha256
                        )
                        adaptive_executable_freeze_sha256 = (
                            adaptive_execution.executable_revision.executable_freeze_sha256
                        )
                        adaptive_executable_freeze_validated = True
                        adaptive_response = adaptive_execution.frozen_round.proposal
                        cached = (
                            adaptive_response,
                            _content_sha256(list(adaptive_applied.specs)),
                        )
                        adaptive_artifact_body = {
                            "outer_fold": int(outer_fold),
                            "review_round": int(round_index),
                            "review_attempt": int(attempt_index),
                            "request_sha256": request_sha256,
                            "diagnostic_adapter_audit": adaptive_diagnostic_audit,
                            "authenticated_execution": adaptive_execution_record,
                            "proposal_frozen_before_executable_bridge": True,
                            "executable_revision_frozen_before_gate": True,
                            "complete_catalog_sent_to_legacy_review_agent": False,
                            "raw_response_persisted": False,
                            "raw_reasoning_persisted": False,
                            "gate_accessed": False,
                        }
                        prior_adaptive_artifact = _load_request_bound_adaptive_execution(
                            response_cache_path,
                            outer_fold=outer_fold,
                            request_sha256=request_sha256,
                            review_round=round_index,
                            review_attempt=attempt_index,
                            expected_diagnostic_adapter_audit=adaptive_diagnostic_audit,
                        )
                        if prior_adaptive_artifact is None:
                            _write_immutable_json(
                                response_cache_path,
                                adaptive_artifact_body,
                                schema=(ADAPTIVE_HIERARCHICAL_REVIEW_EXECUTION_SCHEMA_VERSION),
                            )
                        else:
                            prior_response, prior_applied, prior_execution = prior_adaptive_artifact
                            if (
                                _content_sha256(prior_response)
                                != _content_sha256(adaptive_response)
                                or _content_sha256(list(prior_applied.specs))
                                != _content_sha256(list(adaptive_applied.specs))
                                or prior_execution["frozen_round"]["freeze_sha256"]
                                != adaptive_proposal_freeze_sha256
                                or prior_execution["executable_revision"][
                                    "executable_freeze_sha256"
                                ]
                                != adaptive_executable_freeze_sha256
                            ):
                                raise RuntimeError(
                                    "current authenticated per-job replay differs from the "
                                    "immutable adaptive execution artifact"
                                )
                elif not adaptive_attempt:
                    cached = _load_request_bound_review_response(
                        response_cache_path,
                        request_sha256=request_sha256,
                        max_contracts=self.config.max_candidates,
                        review_round=round_index,
                        review_attempt=attempt_index,
                    )
                if cached is not None and cached_failure is not None:
                    raise RuntimeError(
                        "review attempt has both a valid response and a failure audit"
                    )
                if adaptive_attempt and response_failure is not None and cached_failure is None:
                    failure_identity = _sanitized_review_failure_identity(response_failure)
                    current_names = [str(spec["name"]) for spec in workspace_specs]
                    failure_body = {
                        "review_round": int(round_index),
                        "review_attempt": int(attempt_index),
                        "request_sha256": request_sha256,
                        "failure_type": "adaptive_hierarchy_or_executable_validation",
                        **failure_identity,
                        "failed_contract_names": current_names,
                        "completion_attempts": [],
                        "raw_response_persisted": False,
                        "raw_reasoning_persisted": False,
                        "row_level_values_persisted": False,
                        "gate_accessed": False,
                        "gate_consumed": False,
                        "outer_heldout_labels_used": False,
                    }
                    _write_immutable_json(
                        failure_cache_path,
                        failure_body,
                        schema=POST_EXTRACTION_REVIEW_FAILURE_SCHEMA_VERSION,
                    )
                    cached_failure = _load_request_bound_review_failure(
                        failure_cache_path,
                        request_sha256=request_sha256,
                        review_round=round_index,
                        review_attempt=attempt_index,
                        expected_current_names=current_names,
                    )
                    if cached_failure is None:  # pragma: no cover
                        raise RuntimeError("adaptive failure audit disappeared after creation")
                if not adaptive_attempt and cached is None and cached_failure is None:
                    returned_response_for_failure: Mapping[str, Any] | None = None
                    response_failure: BaseException | None = None
                    try:
                        raw_response, _raw_response_sha256 = self._invoke_review_agent(context)
                    except PostExtractionReviewResponseExhausted as exc:
                        response_failure = exc
                    else:
                        returned_response_for_failure = raw_response
                    if response_failure is None:
                        try:
                            validated = validate_post_extraction_review_response(
                                raw_response,
                                current_specs=workspace_specs,
                                available_diagnostic_ids=diagnostic_ids,
                                available_diagnostic_targets=diagnostic_targets,
                                available_evidence_ids=evidence_ids,
                                available_evidence_catalog=context["sanitized_evidence_catalog"],
                                max_operations=(self.config.post_extraction_review_max_operations),
                            )
                            response = validated.as_dict()
                            response_sha256 = _content_sha256(response)
                            if response_sha256 != validated.response_sha256:
                                raise RuntimeError(
                                    "review response canonical hash disagrees with validator"
                                )
                            applied = apply_post_extraction_review_operations(
                                workspace_specs,
                                validated,
                                max_contracts=self.config.max_candidates,
                            )
                            applied_specs_sha256 = _content_sha256(list(applied.specs))
                        except (TypeError, ValueError) as exc:
                            response_failure = exc
                    if response_failure is not None:
                        failure_identity = _sanitized_review_failure_identity(response_failure)
                        current_names = [str(spec["name"]) for spec in workspace_specs]
                        completion_attempts = _sanitized_review_failure_completion_attempts(
                            self.review_agent,
                            returned_response=returned_response_for_failure,
                        )
                        failure_type = (
                            "remote_reviewer_exhausted"
                            if isinstance(
                                response_failure,
                                PostExtractionReviewResponseExhausted,
                            )
                            else "runner_boundary_validation"
                        )
                        failure_body = {
                            "review_round": int(round_index),
                            "review_attempt": int(attempt_index),
                            "request_sha256": request_sha256,
                            "failure_type": failure_type,
                            **failure_identity,
                            "failed_contract_names": _sanitized_review_failure_targets(
                                self.review_agent,
                                returned_response=returned_response_for_failure,
                                current_names=current_names,
                            ),
                            "completion_attempts": completion_attempts,
                            "raw_response_persisted": False,
                            "raw_reasoning_persisted": False,
                            "row_level_values_persisted": False,
                            "gate_accessed": False,
                            "gate_consumed": False,
                            "outer_heldout_labels_used": False,
                        }
                        _write_immutable_json(
                            failure_cache_path,
                            failure_body,
                            schema=POST_EXTRACTION_REVIEW_FAILURE_SCHEMA_VERSION,
                        )
                        cached_failure = _load_request_bound_review_failure(
                            failure_cache_path,
                            request_sha256=request_sha256,
                            review_round=round_index,
                            review_attempt=attempt_index,
                            expected_current_names=current_names,
                        )
                        if cached_failure is None:  # pragma: no cover
                            raise RuntimeError("review failure audit disappeared after creation")
                    else:
                        _write_immutable_json(
                            response_cache_path,
                            {
                                "review_round": int(round_index),
                                "review_attempt": int(attempt_index),
                                "request_sha256": request_sha256,
                                "response": response,
                                "response_sha256": response_sha256,
                                "applied_specs_sha256": applied_specs_sha256,
                                "apply_policy_version": (
                                    POST_EXTRACTION_REVIEW_OPERATION_APPLY_POLICY_VERSION
                                ),
                                "max_contracts": int(self.config.max_candidates),
                                "raw_response_persisted": False,
                                "raw_reasoning_persisted": False,
                            },
                            schema=POST_EXTRACTION_REVIEW_RESPONSE_CACHE_SCHEMA_VERSION,
                        )
                        cached = _load_request_bound_review_response(
                            response_cache_path,
                            request_sha256=request_sha256,
                            max_contracts=self.config.max_candidates,
                            review_round=round_index,
                            review_attempt=attempt_index,
                        )
                        if cached is None:  # pragma: no cover
                            raise RuntimeError("review response cache disappeared after creation")

                if cached_failure is not None:
                    response_validation_rejections += 1
                    retry_available = attempt_index <= max_quality_retries
                    status = (
                        "review_response_validation_failed_pre_gate_retrying"
                        if retry_available
                        else "review_response_validation_retry_exhausted"
                    )
                    failure_sha256 = _content_sha256(cached_failure)
                    failure_feedback = {
                        "kind": "review_response_validation_retry_feedback",
                        "failed_review_round": int(round_index),
                        "failed_attempt": int(attempt_index),
                        "failure_type": str(cached_failure["failure_type"]),
                        "failure_code": str(cached_failure["failure_code"]),
                        "failure_message": str(cached_failure["failure_message"]),
                        "failure_issue_sha256": str(cached_failure["failure_issue_sha256"]),
                        "failed_contract_names": list(cached_failure["failed_contract_names"]),
                        "completion_attempt_count": len(cached_failure["completion_attempts"]),
                        "parsed_json_attempt_count": sum(
                            bool(row.get("parsed_json_object"))
                            for row in cached_failure["completion_attempts"]
                            if isinstance(row, Mapping)
                        ),
                        "failure_audit_sha256": failure_sha256,
                        "required_action": (
                            "return a new closed-schema response; do not repeat the "
                            "failed operation/evidence pairing"
                        ),
                        "same_gate_remains_sealed": True,
                        "gate_accessed": False,
                        "outer_heldout_labels_used": False,
                    }
                    terminal = persist_attempt(
                        attempt_dir=attempt_dir,
                        body={
                            "outer_fold": int(outer_fold),
                            "review_round": int(round_index),
                            "review_attempt": int(attempt_index),
                            "maximum_quality_retries_per_gate": max_quality_retries,
                            "status": status,
                            "spent_fold_ids_before_proposal": list(spent_fold_ids),
                            "spent_row_count": len(spent_ids),
                            "spent_evidence_context_audit": spent_evidence_audit,
                            "sanitized_context": context,
                            "request_sha256": request_sha256,
                            "request_cache_path": str(request_cache_path.resolve()),
                            "request_manifest_sha256": request_manifest_sha256,
                            "response_failure_path": str(failure_cache_path.resolve()),
                            "response_failure_sha256": failure_sha256,
                            "response_validation_failure": cached_failure,
                            "response_validation_retry_feedback_sha256": (
                                _content_sha256(failure_feedback)
                            ),
                            "accepted_round_baseline_specs_sha256": _content_sha256(
                                accepted_round_specs
                            ),
                            "workspace_specs_before_attempt_sha256": _content_sha256(
                                workspace_specs
                            ),
                            "workspace_extraction_before_attempt_sha256": (
                                workspace_extraction_sha256
                            ),
                            "workspace_stage_count_before_attempt": len(workspace_stage_history),
                            "workspace_stage_history": list(workspace_stage_history),
                            "candidate_workspace_policy_version": (
                                POST_EXTRACTION_REVIEW_CANDIDATE_WORKSPACE_POLICY_VERSION
                            ),
                            "workspace_advanced": False,
                            "workspace_specs_after_attempt_sha256": _content_sha256(
                                workspace_specs
                            ),
                            "workspace_extraction_after_attempt_sha256": (
                                workspace_extraction_sha256
                            ),
                            "workspace_accepted": False,
                            "gate_fold_id": int(gate_fold_id),
                            "gate_accessed": False,
                            "gate_consumed": False,
                            "same_gate_remains_sealed": True,
                            "retry_will_reuse_same_sealed_gate": retry_available,
                            "quality_retries_remaining": max(
                                0,
                                max_quality_retries - attempt_index + 1,
                            ),
                            "raw_response_persisted": False,
                            "raw_reasoning_persisted": False,
                            "row_level_numerical_vectors_persisted": False,
                            "outer_heldout_labels_used": False,
                        },
                        status=status,
                        attempt_index=attempt_index,
                        gate_accessed=False,
                        gate_consumed=False,
                    )
                    attempt_records.append(terminal)
                    if retry_available:
                        response_validation_retries += 1
                        retry_feedback.append(failure_feedback)
                        continue

                    response_validation_retry_exhausted = True
                    prior_gate_feedback.append(failure_feedback)
                    finish_round(
                        round_index=round_index,
                        terminal_attempt=terminal,
                        attempts=attempt_records,
                    )
                    round_finished = True
                    break

                response, cached_applied_specs_sha256 = cached
                if adaptive_attempt:
                    if adaptive_applied is None or adaptive_execution_record is None:
                        raise RuntimeError("adaptive executable freeze was not retained")
                    applied = adaptive_applied
                    proposal_stops = bool(response["converged"])
                    response_cache_authority = (
                        "authenticated_hierarchy_validated_and_executable_frozen"
                    )
                    adaptive_execution_records.append(
                        {
                            "review_round": int(round_index),
                            "review_attempt": int(attempt_index),
                            "exact_spent_authentication_sha256": (
                                exact_spent_authentication.authentication_sha256
                                if exact_spent_authentication is not None
                                else None
                            ),
                            "execution_sha256": adaptive_execution_sha256,
                            "proposal_freeze_sha256": adaptive_proposal_freeze_sha256,
                            "executable_freeze_sha256": adaptive_executable_freeze_sha256,
                            "outer_execution_artifact_used_as_authority": False,
                        }
                    )
                else:
                    validated = validate_post_extraction_review_response(
                        response,
                        current_specs=workspace_specs,
                        available_diagnostic_ids=diagnostic_ids,
                        available_diagnostic_targets=diagnostic_targets,
                        available_evidence_ids=evidence_ids,
                        available_evidence_catalog=context["sanitized_evidence_catalog"],
                        max_operations=self.config.post_extraction_review_max_operations,
                    )
                    applied = apply_post_extraction_review_operations(
                        workspace_specs,
                        validated,
                        max_contracts=self.config.max_candidates,
                    )
                    proposal_stops = validated.stops
                    response_cache_authority = (
                        "loaded_hash_verified_attempt_bound_response_"
                        "validated_operations_applied"
                    )
                applied_specs_sha256 = _content_sha256(list(applied.specs))
                if applied_specs_sha256 != cached_applied_specs_sha256:
                    raise RuntimeError("cached review application hash disagrees with replay")
                response_sha256 = _content_sha256(response)
                if not adaptive_attempt and response_sha256 != validated.response_sha256:
                    raise RuntimeError("cached review response hash disagrees with validator")
                candidate_adaptive_registry: tuple[AdaptiveCurrentFeature, ...] | None = None
                if hierarchical_review:
                    if workspace_adaptive_registry is None:
                        raise RuntimeError("hierarchical candidate workspace lost provenance")
                    candidate_adaptive_registry = self._transition_adaptive_registry(
                        before=workspace_adaptive_registry,
                        after_specs=applied.specs,
                        operation_audit=applied.operation_audit,
                        evidence_family_by_id=adaptive_evidence_family_by_id,
                    )
                accepted_registry_sha256 = (
                    None
                    if accepted_round_adaptive_registry is None
                    else _content_sha256(
                        [item.as_prompt_item() for item in accepted_round_adaptive_registry]
                    )
                )
                workspace_registry_sha256 = (
                    None
                    if workspace_adaptive_registry is None
                    else _content_sha256(
                        [item.as_prompt_item() for item in workspace_adaptive_registry]
                    )
                )
                candidate_registry_sha256 = (
                    None
                    if candidate_adaptive_registry is None
                    else _content_sha256(
                        [item.as_prompt_item() for item in candidate_adaptive_registry]
                    )
                )

                common_attempt_body = {
                    "outer_fold": int(outer_fold),
                    "review_round": int(round_index),
                    "review_attempt": int(attempt_index),
                    "maximum_quality_retries_per_gate": max_quality_retries,
                    "spent_fold_ids_before_proposal": list(spent_fold_ids),
                    "spent_row_count": len(spent_ids),
                    "spent_evidence_context_audit": spent_evidence_audit,
                    "sanitized_context": context,
                    "request_sha256": request_sha256,
                    "request_cache_path": str(request_cache_path.resolve()),
                    "request_manifest_sha256": request_manifest_sha256,
                    "response_sha256": response_sha256,
                    "response_cache_path": str(response_cache_path.resolve()),
                    "response_cache_authority": response_cache_authority,
                    "applied_specs_sha256": applied_specs_sha256,
                    "accepted_round_baseline_specs_sha256": _content_sha256(accepted_round_specs),
                    "workspace_specs_before_attempt_sha256": _content_sha256(workspace_specs),
                    "accepted_round_registry_provenance_sha256": accepted_registry_sha256,
                    "workspace_registry_provenance_before_attempt_sha256": (
                        workspace_registry_sha256
                    ),
                    "candidate_registry_provenance_sha256": candidate_registry_sha256,
                    "candidate_registry_private_items": (
                        None
                        if candidate_adaptive_registry is None
                        else [item.as_prompt_item() for item in candidate_adaptive_registry]
                    ),
                    "workspace_extraction_before_attempt_sha256": (workspace_extraction_sha256),
                    "workspace_stage_count_before_attempt": len(workspace_stage_history),
                    "workspace_stage_history": list(workspace_stage_history),
                    "candidate_workspace_policy_version": (
                        POST_EXTRACTION_REVIEW_CANDIDATE_WORKSPACE_POLICY_VERSION
                    ),
                    "validated_response": response,
                    "adaptive_hierarchy_execution_sha256": (adaptive_execution_sha256),
                    "adaptive_proposal_freeze_sha256": (adaptive_proposal_freeze_sha256),
                    "adaptive_executable_freeze_sha256": (adaptive_executable_freeze_sha256),
                    "gate_fold_id": int(gate_fold_id),
                    "raw_response_persisted": False,
                    "raw_reasoning_persisted": False,
                    "outer_heldout_labels_used": False,
                }

                # The request/response pair is frozen before this branch. The gate
                # remains wholly unavailable until the sealed spent-only candidate
                # workspace clears changed-contract quality and retained ontology checks.
                convergence_status = None
                if proposal_stops:
                    convergence_status = "agent_stop"
                elif list(applied.specs) == workspace_specs:
                    convergence_status = "no_semantic_change"
                if convergence_status is not None:
                    if not bool(workspace_ontology["passed"]):
                        unresolved_ontology_convergence_rejections += 1
                        retained_ontology_rejections += 1
                        retry_available = attempt_index <= max_quality_retries
                        status = (
                            "unresolved_ontology_convergence_rejected_retrying"
                            if retry_available
                            else "unresolved_ontology_convergence_retry_exhausted"
                        )
                        terminal = persist_attempt(
                            attempt_dir=attempt_dir,
                            body={
                                **common_attempt_body,
                                "status": status,
                                "proposed_convergence_status": convergence_status,
                                "operation_audit": list(applied.operation_audit),
                                "retained_registry_ontology_guard": workspace_ontology,
                                "workspace_advanced": False,
                                "workspace_specs_after_attempt_sha256": _content_sha256(
                                    workspace_specs
                                ),
                                "workspace_extraction_after_attempt_sha256": (
                                    workspace_extraction_sha256
                                ),
                                "gate_accessed": False,
                                "gate_consumed": False,
                                "same_gate_remains_sealed": True,
                                "retry_will_reuse_same_sealed_gate": retry_available,
                                "quality_retries_remaining": max(
                                    0,
                                    max_quality_retries - attempt_index + 1,
                                ),
                                "row_level_numerical_vectors_persisted": False,
                                "aggregate_numerical_diagnostics_persisted": True,
                            },
                            status=status,
                            attempt_index=attempt_index,
                            gate_accessed=False,
                            gate_consumed=False,
                        )
                        attempt_records.append(terminal)
                        if retry_available:
                            retained_ontology_retries += 1
                            retry_feedback.append(
                                self._ontology_retry_feedback_diagnostic(
                                    workspace_ontology,
                                    review_round=round_index,
                                    failed_attempt=attempt_index,
                                    response_sha256=response_sha256,
                                    operation_audit=applied.operation_audit,
                                    proposal_kind=convergence_status,
                                )
                            )
                            continue

                        unresolved_ontology_convergence_retry_exhausted = True
                        finish_round(
                            round_index=round_index,
                            terminal_attempt=terminal,
                            attempts=attempt_records,
                        )
                        round_finished = True
                        break

                    if workspace_specs == accepted_round_specs:
                        stopped_by_agent = True
                        status = convergence_status
                        terminal = persist_attempt(
                            attempt_dir=attempt_dir,
                            body={
                                **common_attempt_body,
                                "status": status,
                                "operation_audit": list(applied.operation_audit),
                                "retained_registry_ontology_guard": workspace_ontology,
                                "workspace_advanced": False,
                                "workspace_specs_after_attempt_sha256": _content_sha256(
                                    workspace_specs
                                ),
                                "workspace_extraction_after_attempt_sha256": (
                                    workspace_extraction_sha256
                                ),
                                "gate_accessed": False,
                                "gate_consumed": False,
                            },
                            status=status,
                            attempt_index=attempt_index,
                            gate_accessed=False,
                            gate_consumed=False,
                        )
                        attempt_records.append(terminal)
                        finish_round(
                            round_index=round_index,
                            terminal_attempt=terminal,
                            attempts=attempt_records,
                        )
                        round_finished = True
                        break

                    # A stop/no-change response on a safe, non-empty workspace means
                    # the cumulative draft is ready for one atomic untouched-gate
                    # comparison against the accepted round baseline.
                    candidate_specs = list(workspace_specs)
                    candidate_spent_extracted = workspace_spent_extracted
                    selective_audit = {
                        "candidate_contract_count": len(candidate_specs),
                        "selective_reextraction_spec_count": 0,
                        "selective_reextraction_names": [],
                        "reused_extraction_names": [str(spec["name"]) for spec in candidate_specs],
                        "role_only_changed_names": [],
                        "removed_names": [],
                        "added_names": [],
                        "provider_audit": None,
                        "role_only_columns_reused_without_remote_extraction": False,
                        "outer_heldout_labels_used": False,
                        "cache_overlay_enabled_for_this_scope": False,
                        "workspace_convergence_without_additional_extraction": True,
                    }
                else:
                    candidate_spent_extracted, selective_audit = (
                        self._candidate_extraction_projection(
                            label_free=spent_label_free,
                            current_extracted=workspace_spent_extracted,
                            current_specs=workspace_specs,
                            applied=applied,
                            use_cache_overlay=False,
                        )
                    )
                    candidate_specs = list(applied.specs)

                valid_operation_proposals += 1
                selective_audit = {
                    **dict(selective_audit),
                    "row_scope": "spent_rows_only",
                    "row_count": len(spent_ids),
                    "sealed_gate_texts_available_to_extractor": False,
                    "outer_heldout_texts_available_to_extractor": False,
                }
                candidate_spent = self._observable_review_rows(
                    row_ids=spent_ids,
                    extracted=candidate_spent_extracted,
                    data=data,
                    fold_by_row=fold_by_row,
                )
                candidate_quality = self._candidate_post_extraction_quality_guard(
                    candidate_spent,
                    candidate_specs,
                    spent_texts=spent_texts,
                    extraction_changed_names=self._cumulative_extraction_changed_names(
                        accepted_round_specs,
                        candidate_specs,
                    ),
                    scientific_policy=(
                        self.config.post_extraction_scientific_policy
                    ),
                )
                candidate_ontology = dict(candidate_quality["retained_registry_ontology_guard"])
                candidate_extraction_sha256 = self._extraction_projection_sha256(
                    candidate_spent_extracted,
                    candidate_specs,
                )
                candidate_quality_failed = not bool(candidate_quality["passed"])
                candidate_ontology_failed = not bool(candidate_ontology["passed"])
                if candidate_quality_failed or candidate_ontology_failed:
                    candidate_quality_rejections += int(candidate_quality_failed)
                    retained_ontology_rejections += int(candidate_ontology_failed)
                    retry_available = attempt_index <= max_quality_retries
                    workspace_failure_keys = self._ontology_failure_keys(workspace_ontology)
                    candidate_failure_keys = self._ontology_failure_keys(candidate_ontology)
                    workspace_advanced = bool(
                        retry_available
                        and not candidate_quality_failed
                        and candidate_ontology_failed
                        and candidate_failure_keys < workspace_failure_keys
                    )
                    stage_record = {
                        "attempt": int(attempt_index),
                        "response_sha256": response_sha256,
                        "workspace_specs_before_sha256": _content_sha256(workspace_specs),
                        "workspace_specs_after_sha256": _content_sha256(candidate_specs),
                        "workspace_extraction_before_sha256": workspace_extraction_sha256,
                        "workspace_extraction_after_sha256": candidate_extraction_sha256,
                        "hard_failure_count_before": len(workspace_failure_keys),
                        "hard_failure_count_after": len(candidate_failure_keys),
                        "hard_failure_set_strictly_reduced": bool(
                            candidate_failure_keys < workspace_failure_keys
                        ),
                        "changed_contract_quality_passed": not candidate_quality_failed,
                        "workspace_accepted": False,
                        "gate_accessed": False,
                        "gate_consumed": False,
                    }
                    if workspace_advanced:
                        status = "candidate_workspace_advanced_pre_gate_retrying"
                    elif candidate_quality_failed:
                        status = (
                            "candidate_quality_rejected_pre_gate_retrying"
                            if retry_available
                            else "quality_retry_exhausted"
                        )
                    else:
                        status = (
                            "retained_registry_ontology_rejected_pre_gate_retrying"
                            if retry_available
                            else "retained_registry_ontology_retry_exhausted"
                        )
                    terminal = persist_attempt(
                        attempt_dir=attempt_dir,
                        body={
                            **common_attempt_body,
                            "status": status,
                            "operation_audit": list(applied.operation_audit),
                            "selective_extraction": selective_audit,
                            "candidate_post_extraction_quality_guard": candidate_quality,
                            "retained_registry_ontology_guard": candidate_ontology,
                            "workspace_advanced": workspace_advanced,
                            "workspace_stage": stage_record,
                            "workspace_stage_history_after_attempt": [
                                *workspace_stage_history,
                                *([stage_record] if workspace_advanced else []),
                            ],
                            "workspace_specs_after_attempt_sha256": (
                                _content_sha256(candidate_specs)
                                if workspace_advanced
                                else _content_sha256(workspace_specs)
                            ),
                            "workspace_extraction_after_attempt_sha256": (
                                candidate_extraction_sha256
                                if workspace_advanced
                                else workspace_extraction_sha256
                            ),
                            "workspace_accepted": False,
                            "gate_accessed": False,
                            "gate_consumed": False,
                            "same_gate_remains_sealed": True,
                            "retry_will_reuse_same_sealed_gate": retry_available,
                            "quality_retries_remaining": max(
                                0,
                                max_quality_retries - attempt_index + 1,
                            ),
                            "row_level_numerical_vectors_persisted": False,
                            "aggregate_numerical_diagnostics_persisted": True,
                        },
                        status=status,
                        attempt_index=attempt_index,
                        gate_accessed=False,
                        gate_consumed=False,
                    )
                    attempt_records.append(terminal)
                    if retry_available:
                        candidate_quality_retries += 1
                        if candidate_quality_failed:
                            retry_feedback.append(
                                self._quality_retry_feedback_diagnostic(
                                    candidate_quality,
                                    review_round=round_index,
                                    failed_attempt=attempt_index,
                                    response_sha256=response_sha256,
                                    operation_audit=applied.operation_audit,
                                )
                            )
                        if candidate_ontology_failed:
                            retained_ontology_retries += 1
                            retry_feedback.append(
                                self._ontology_retry_feedback_diagnostic(
                                    candidate_ontology,
                                    review_round=round_index,
                                    failed_attempt=attempt_index,
                                    response_sha256=response_sha256,
                                    operation_audit=applied.operation_audit,
                                    proposal_kind="registry_revision",
                                    workspace_advanced=workspace_advanced,
                                )
                            )
                        if workspace_advanced:
                            if hierarchical_review and candidate_adaptive_registry is None:
                                raise RuntimeError(
                                    "staged hierarchical workspace lost candidate provenance"
                                )
                            candidate_workspace_stage_count += 1
                            workspace_specs = [
                                CandidateContract(spec).extraction_spec for spec in candidate_specs
                            ]
                            workspace_spent_extracted = candidate_spent_extracted
                            workspace_adaptive_registry = candidate_adaptive_registry
                            workspace_stage_history.append(stage_record)
                            workspace_operation_audit_history.extend(
                                json.loads(_canonical_json(row)) for row in applied.operation_audit
                            )
                        continue

                    quality_retry_exhausted = True
                    finish_round(
                        round_index=round_index,
                        terminal_attempt=terminal,
                        attempts=attempt_records,
                    )
                    round_finished = True
                    break

                # A quality-passing proposal is now immutable. Only at this point
                # may the current gate be materialized or an upstream provider run.
                if current_specs != accepted_round_specs:
                    raise RuntimeError(
                        "accepted review registry changed while a candidate workspace was sealed"
                    )
                final_delta_operation_audit = (
                    [] if convergence_status is not None else list(applied.operation_audit)
                )
                cumulative_operation_audit = [
                    *workspace_operation_audit_history,
                    *final_delta_operation_audit,
                ]
                cumulative_projection_plan = self._cumulative_review_projection_plan(
                    accepted_round_specs,
                    candidate_specs,
                    operation_audit=cumulative_operation_audit,
                )
                cumulative_proposal_sha256 = _content_sha256(
                    {
                        "accepted_round_baseline_specs_sha256": _content_sha256(
                            accepted_round_specs
                        ),
                        "candidate_specs_sha256": _content_sha256(candidate_specs),
                        "staged_response_sha256s": [
                            str(row["response_sha256"]) for row in workspace_stage_history
                        ],
                        "terminal_response_sha256": response_sha256,
                        "candidate_registry_provenance_sha256": candidate_registry_sha256,
                        "adaptive_proposal_freeze_sha256": (adaptive_proposal_freeze_sha256),
                        "adaptive_executable_freeze_sha256": (adaptive_executable_freeze_sha256),
                    }
                )
                pre_gate_candidate_record: Mapping[str, Any] | None = None
                if hierarchical_review:
                    if (
                        accepted_registry_sha256 is None
                        or workspace_registry_sha256 is None
                        or candidate_registry_sha256 is None
                        or candidate_adaptive_registry is None
                    ):
                        raise RuntimeError(
                            "hierarchical candidate provenance is incomplete before gate"
                        )
                    pre_gate_candidate_path = (
                        attempt_dir / "immutable_pre_gate_candidate_freeze.json"
                    )
                    pre_gate_candidate_body = {
                        "outer_fold": int(outer_fold),
                        "review_round": int(round_index),
                        "review_attempt": int(attempt_index),
                        "gate_fold_id": int(gate_fold_id),
                        "accepted_round_specs_sha256": _content_sha256(accepted_round_specs),
                        "candidate_specs_sha256": _content_sha256(candidate_specs),
                        "accepted_registry_provenance_sha256": (accepted_registry_sha256),
                        "workspace_registry_provenance_sha256": (workspace_registry_sha256),
                        "candidate_registry_provenance_sha256": (candidate_registry_sha256),
                        "candidate_registry_private_items": [
                            item.as_prompt_item() for item in candidate_adaptive_registry
                        ],
                        "cumulative_operation_audit_sha256": _content_sha256(
                            cumulative_operation_audit
                        ),
                        "cumulative_proposal_sha256": cumulative_proposal_sha256,
                        "adaptive_proposal_freeze_sha256": (adaptive_proposal_freeze_sha256),
                        "adaptive_executable_freeze_sha256": (adaptive_executable_freeze_sha256),
                        "proposal_and_executable_specs_frozen_before_gate": True,
                        "candidate_provenance_frozen_before_gate": True,
                        "gate_accessed": False,
                    }
                    pre_gate_candidate_content_sha256 = _write_immutable_json(
                        pre_gate_candidate_path,
                        pre_gate_candidate_body,
                        schema=ADAPTIVE_PRE_GATE_CANDIDATE_FREEZE_SCHEMA_VERSION,
                    )
                    pre_gate_candidate_record = {
                        "path": str(pre_gate_candidate_path.resolve()),
                        "content_sha256": pre_gate_candidate_content_sha256,
                        "candidate_registry_provenance_sha256": (candidate_registry_sha256),
                    }
                gate_ids = schedule.row_ids((gate_fold_id,))
                if adaptive_attempt:
                    if (
                        adaptive_execution_record is None
                        or candidate_adaptive_registry is None
                        or not adaptive_executable_freeze_validated
                    ):
                        raise RuntimeError(
                            "adaptive proposal/executable/provenance freeze missing before gate"
                        )
                    if exact_spent_authentication is None or (
                        exact_spent_authentication.still_sealed_gate_fingerprint
                        != row_set_fingerprint(gate_ids)
                    ):
                        raise RuntimeError(
                            "adaptive exact-spent authentication cites another sealed gate"
                        )
                gate_label_free = (
                    label_free.set_index("_oci_row_id", drop=False)
                    .loc[list(gate_ids)]
                    .reset_index(drop=True)
                )
                raw_current_gate_extracted, current_gate_provider_audit = self._extract(
                    gate_label_free,
                    current_specs,
                    use_cache_overlay=False,
                )
                current_gate_extracted = self._validated_extraction_projection(
                    raw_current_gate_extracted,
                    label_free=gate_label_free,
                    specs=current_specs,
                    source="immutable current-registry gate extraction",
                )
                candidate_gate_extracted, candidate_gate_extraction_audit = (
                    self._candidate_extraction_projection(
                        label_free=gate_label_free,
                        current_extracted=current_gate_extracted,
                        current_specs=accepted_round_specs,
                        applied=cumulative_projection_plan,
                        use_cache_overlay=False,
                    )
                )
                gate = self._observable_review_rows(
                    row_ids=gate_ids,
                    extracted=current_gate_extracted,
                    data=data,
                    fold_by_row=None,
                )
                candidate_gate = self._observable_review_rows(
                    row_ids=gate_ids,
                    extracted=candidate_gate_extracted,
                    data=data,
                    fold_by_row=None,
                )
                context_texts = list(spent_texts)
                gate_texts = gate_label_free[self.config.text_column].astype(str).tolist()
                first_gate_materialization_record: Mapping[str, Any] | None = None
                prebound_gate_provider: Any | None = None
                prebound_gate_provider_used = False
                if hierarchical_review:
                    if round_index == 1 and not self.gate_only_reference_review:
                        assert isinstance(
                            hierarchical_first_gate_materialization_intent,
                            FirstGateMaterializationIntent,
                        )
                        assert isinstance(
                            hierarchical_first_gate_catalog,
                            RoleNeutralEvidenceCatalog,
                        )
                        shared_provider = self.review_gate_source_provider
                        if shared_provider is None:
                            raise RuntimeError(
                                "hierarchical first-gate materialization lost its shared provider"
                            )
                        materialized = prepare_first_untouched_gate_direct_numerical(
                            outer_fold=outer_fold,
                            initial_spent_row_ids=accepted_round_spent.row_ids,
                            initial_spent_texts=context_texts,
                            initial_spent_treatment=accepted_round_spent.treatment,
                            initial_spent_outcome=accepted_round_spent.outcome,
                            initial_spent_inner_fold_ids=(
                                accepted_round_spent.inner_fold_ids or ()
                            ),
                            first_gate_row_ids=gate_ids,
                            first_gate_texts=gate_texts,
                            catalog=hierarchical_first_gate_catalog,
                            provider=shared_provider,
                            destination=(
                                attempt_dir / "first_gate_direct_upstream_numerical_manifest.json"
                            ),
                            bounds=(
                                self.first_untouched_gate_preparation_bounds
                            ),
                        )
                        materialized.verify()
                        realization_audit = {
                            **dict(materialized.audit),
                            "materialization_intent_sha256": (
                                hierarchical_first_gate_materialization_intent.content_sha256
                            ),
                            "exact_hierarchy_approval_completed_before_materialization": True,
                            "review_proposal_frozen_before_materialization": True,
                            "frozen_review_proposal_sha256": cumulative_proposal_sha256,
                        }
                        realization_attestation = (
                            hierarchical_first_gate_materialization_intent.verify_realization(
                                materialized.persisted_manifest.manifest,
                                preparation_audit=realization_audit,
                                bound_provider=materialized.bound_provider,
                            )
                        )
                        realization_path = (
                            attempt_dir / "first_gate_materialization_realization_attestation.json"
                        )
                        realization_file_sha256 = _write_immutable_plain_json(
                            realization_path,
                            realization_attestation.as_dict(),
                        )
                        prebound_gate_provider = materialized.bound_provider
                        first_gate_materialization_record = {
                            "intent_content_sha256": (
                                hierarchical_first_gate_materialization_intent.content_sha256
                            ),
                            "intent_source_cache_key": (
                                hierarchical_first_gate_materialization_intent.body[
                                    "source_cache_key"
                                ]
                            ),
                            "realization_attestation_path": str(realization_path),
                            "realization_attestation_file_sha256": realization_file_sha256,
                            "realization_attestation_content_sha256": (
                                realization_attestation.content_sha256
                            ),
                            "direct_manifest_path": str(materialized.persisted_manifest.path),
                            "direct_manifest_file_sha256": (
                                materialized.persisted_manifest.file_sha256
                            ),
                            "direct_manifest_content_sha256": (
                                materialized.persisted_manifest.manifest.content_sha256
                            ),
                            "preparation_audit": realization_audit,
                            "materialized_after_exact_approval": True,
                            "materialized_after_review_proposal_freeze": True,
                            "verified_before_first_gate_evaluation": True,
                            "gate_labels_supplied_to_provider": False,
                        }
                    if self.gate_only_reference_review:
                        (
                            source_view,
                            feature_bank_view,
                            gate_only_reference_audit,
                        ) = self._gate_only_reference_views(
                            outer_fold=outer_fold,
                            context_epoch=round_index - 1,
                            gate_row_ids=gate_ids,
                            context=accepted_round_spent,
                        )
                        if round_index == 1:
                            first_gate_materialization_record = {
                                **dict(gate_only_reference_audit),
                                "materialized_after_exact_approval": True,
                                "materialized_after_review_proposal_freeze": True,
                                "verified_before_first_gate_evaluation": True,
                                "gate_labels_supplied_to_provider": False,
                                "reference_only_existing_stage1_values": True,
                            }
                    else:
                        source_view, feature_bank_view, prebound_gate_provider_used = (
                            self._hierarchical_gate_views(
                                outer_fold=outer_fold,
                                gate_row_ids=gate_ids,
                                context=accepted_round_spent,
                                context_texts=context_texts,
                                gate_texts=gate_texts,
                                prebound_provider=prebound_gate_provider,
                            )
                        )
                else:
                    source_view = self._gate_source_view(
                        outer_fold=outer_fold,
                        gate_row_ids=gate_ids,
                        context=accepted_round_spent,
                        context_texts=context_texts,
                        gate_texts=gate_texts,
                    )
                    feature_bank_view = self._gate_feature_bank_view(
                        outer_fold=outer_fold,
                        gate_row_ids=gate_ids,
                        context=accepted_round_spent,
                        context_texts=context_texts,
                        gate_texts=gate_texts,
                    )
                source_lineage_audit = self._gate_view_lineage_audit(
                    source_view,
                    context=accepted_round_spent,
                )
                feature_bank_lineage_audit = self._gate_view_lineage_audit(
                    feature_bank_view,
                    context=accepted_round_spent,
                )
                decision = evaluate_untouched_gate_acceptance(
                    accepted_round_spent,
                    gate,
                    accepted_round_specs,
                    candidate_specs,
                    source_view=source_view,
                    feature_bank_view=feature_bank_view,
                    candidate_context=candidate_spent,
                    candidate_gate=candidate_gate,
                    config=self.config.post_extraction_review_config,
                    upstream_review_policy=self.config.upstream_review_policy,
                )
                if not isinstance(decision, GateAcceptanceDecision):
                    raise TypeError("review acceptance evaluator returned an invalid decision")
                gate_evaluated_proposals += 1
                accepted = bool(decision.accepted)
                accumulated_label_free = pd.concat(
                    [spent_label_free, gate_label_free],
                    ignore_index=True,
                )
                if accepted:
                    current_specs = [
                        CandidateContract(spec).extraction_spec for spec in candidate_specs
                    ]
                    current_extracted = self._combine_extraction_row_scopes(
                        [candidate_spent_extracted, candidate_gate_extracted],
                        label_free=accumulated_label_free,
                        specs=current_specs,
                        source="accepted accumulated review extraction",
                    )
                    if hierarchical_review:
                        if candidate_adaptive_registry is None:
                            raise RuntimeError("accepted hierarchy lost candidate provenance")
                        current_adaptive_registry = candidate_adaptive_registry
                else:
                    current_extracted = self._combine_extraction_row_scopes(
                        [accepted_round_spent_extracted, current_gate_extracted],
                        label_free=accumulated_label_free,
                        specs=current_specs,
                        source="rejected accumulated review extraction",
                    )
                    if hierarchical_review:
                        if accepted_round_adaptive_registry is None:
                            raise RuntimeError("rejected hierarchy lost baseline provenance")
                        current_adaptive_registry = accepted_round_adaptive_registry

                committed_registry_sha256 = (
                    None
                    if current_adaptive_registry is None
                    else _content_sha256(
                        [item.as_prompt_item() for item in current_adaptive_registry]
                    )
                )
                expected_committed_registry_sha256 = (
                    candidate_registry_sha256 if accepted else accepted_registry_sha256
                )
                if committed_registry_sha256 != expected_committed_registry_sha256:
                    raise RuntimeError("gate commit/revert changed adaptive registry provenance")

                # One gate is consumed after evaluation, including rejection.
                spent_fold_ids.append(int(gate_fold_id))
                gate_partition = next(
                    row
                    for row in schedule.audit["partitions"]
                    if int(row["fold_id"]) == int(gate_fold_id)
                )
                status = "accepted" if accepted else "rejected"
                sanitized_decision = self._sanitized_gate_decision(decision)
                next_feedback = self._prior_gate_feedback_diagnostic(
                    decision,
                    review_round=round_index,
                    response_sha256=cumulative_proposal_sha256,
                    operation_audit=cumulative_operation_audit,
                )
                terminal = persist_attempt(
                    attempt_dir=attempt_dir,
                    body={
                        **common_attempt_body,
                        "status": status,
                        "spent_fold_ids_before_proposal": list(spent_fold_ids[:-1]),
                        "spent_fold_ids_after_gate": list(spent_fold_ids),
                        "spent_row_count_before_proposal": len(spent_ids),
                        "operation_audit": list(applied.operation_audit),
                        "cumulative_operation_audit": cumulative_operation_audit,
                        "cumulative_proposal_sha256": cumulative_proposal_sha256,
                        "pre_gate_candidate_freeze": pre_gate_candidate_record,
                        "accepted_round_registry_provenance_sha256": (accepted_registry_sha256),
                        "candidate_registry_provenance_sha256": (candidate_registry_sha256),
                        "committed_registry_provenance_sha256": (committed_registry_sha256),
                        "gate_accept_commits_exact_candidate_registry": bool(accepted),
                        "gate_reject_restores_exact_baseline_registry": bool(not accepted),
                        "workspace_stage_history_after_attempt": list(workspace_stage_history),
                        "workspace_specs_after_attempt_sha256": _content_sha256(candidate_specs),
                        "workspace_extraction_after_attempt_sha256": (candidate_extraction_sha256),
                        "workspace_advanced": bool(workspace_stage_history),
                        "workspace_accepted": accepted,
                        "gate_accessed": True,
                        "gate_consumed": True,
                        "accepted_round_baseline_spent_extraction_sha256": (
                            accepted_round_spent_extraction_sha256
                        ),
                        "candidate_spent_extraction_sha256": candidate_extraction_sha256,
                        "selective_extraction": selective_audit,
                        "candidate_post_extraction_quality_guard": candidate_quality,
                        "retained_registry_ontology_guard": candidate_ontology,
                        "gate_extraction": {
                            "row_scope": "current_untouched_gate_only",
                            "row_count": len(gate_ids),
                            "performed_after_candidate_quality_passed": True,
                            "outer_heldout_texts_available_to_extractor": False,
                            "cache_overlay_enabled": False,
                            "current_registry_provider_audit": current_gate_provider_audit,
                            "candidate_registry_projection_audit": (
                                candidate_gate_extraction_audit
                            ),
                            "current_registry_extraction_sha256": (
                                self._extraction_projection_sha256(
                                    current_gate_extracted,
                                    accepted_round_specs,
                                )
                            ),
                            "candidate_registry_extraction_sha256": (
                                self._extraction_projection_sha256(
                                    candidate_gate_extracted,
                                    candidate_specs,
                                )
                            ),
                        },
                        "gate": {
                            "fold_id": int(gate_fold_id),
                            "row_count": int(gate_partition["row_count"]),
                            "row_fingerprint": gate_partition["row_fingerprint"],
                            "row_ids_exposed_to_review_agent": False,
                            "current_gate_data_exposed_before_proposal_freeze": False,
                            "providers_invoked_after_proposal_freeze": True,
                            "label_free_numerical_cache_materialized_before_discovery": False,
                            "prebound_acceptance_views_first_consumed_after_proposal_freeze": (
                                bool(prebound_gate_provider_used)
                            ),
                            "first_gate_materialization_intent_applicable": bool(
                                hierarchical_review
                                and round_index == 1
                                and not self.gate_only_reference_review
                            ),
                            "first_gate_materialization": first_gate_materialization_record,
                            "provider_bind_input_contract": (
                                "prefit_cumulative_exact_spent_and_gate_row_ids_only_v1"
                                if self.gate_only_reference_review
                                else "spent_observable_rows_plus_exact_gate_ids_and_text_only_v1"
                            ),
                            "gate_treatment_or_outcome_supplied_to_gate_providers": False,
                            "consumed_after_gate_evaluation": True,
                        },
                        "gate_source_lineage": source_lineage_audit,
                        "gate_feature_bank_lineage": feature_bank_lineage_audit,
                        "gate_source_catalog": self._opaque_gate_source_catalog(
                            source_view,
                            feature_bank_view,
                        ),
                        "acceptance": sanitized_decision,
                        "next_round_feedback_sha256": next_feedback["feedback_sha256"],
                        "selected_contracts_after_gate": current_specs,
                        "selected_contract_sha256_after_gate": [
                            extraction_contract_sha256(spec) for spec in current_specs
                        ],
                        "row_level_numerical_vectors_persisted": False,
                        "aggregate_numerical_diagnostics_persisted": True,
                    },
                    status=status,
                    attempt_index=attempt_index,
                    gate_accessed=True,
                    gate_consumed=True,
                )
                attempt_records.append(terminal)
                finish_round(
                    round_index=round_index,
                    terminal_attempt=terminal,
                    attempts=attempt_records,
                    extra={"gate_row_fingerprint": gate_partition["row_fingerprint"]},
                )
                prior_gate_feedback.append(next_feedback)
                round_finished = True
                break

            if not round_finished:  # pragma: no cover
                raise RuntimeError("post-extraction review round did not reach a terminal state")
            if (
                stopped_by_agent
                or quality_retry_exhausted
                or unresolved_ontology_convergence_retry_exhausted
                or response_validation_retry_exhausted
            ):
                break

        if response_validation_retry_exhausted:
            raise RuntimeError(
                "post-extraction review exhausted bounded response validation retries "
                "before gate access; inspect the immutable failure and round audits"
            )

        # Convergence and bounded round exhaustion are valid only if every
        # retained contract is free of locally grounded alternative-category
        # support on every row consumed so far.
        # Fail before touching any unconsumed gate or outer-heldout text.
        final_spent_ids = tuple(map(int, current_extracted["_oci_row_id"].tolist()))
        final_spent_label_free = (
            label_free.set_index("_oci_row_id", drop=False)
            .loc[list(final_spent_ids)]
            .reset_index(drop=True)
        )
        final_spent = self._observable_review_rows(
            row_ids=final_spent_ids,
            extracted=current_extracted,
            data=data,
            fold_by_row=fold_by_row,
        )
        final_ontology = self._retained_registry_ontology_guard(
            final_spent,
            current_specs,
            spent_texts=tuple(final_spent_label_free[self.config.text_column].astype(str).tolist()),
            scientific_policy=self.config.post_extraction_scientific_policy,
        )
        if not bool(final_ontology["passed"]):
            failure_path = (
                fold_dir / "post_extraction_review" / "unresolved_retained_registry_ontology.json"
            )
            _write_immutable_json(
                failure_path,
                {
                    "outer_fold": int(outer_fold),
                    "status": "unresolved_retained_registry_ontology",
                    "configured_rounds": rounds,
                    "rounds_completed": len(round_records),
                    "spent_fold_ids": list(spent_fold_ids),
                    "spent_row_count": len(final_spent_ids),
                    "retained_contract_sha256": [
                        extraction_contract_sha256(spec) for spec in current_specs
                    ],
                    "retained_registry_ontology_guard": final_ontology,
                    "unconsumed_gate_or_outer_heldout_text_extracted": False,
                    "outer_heldout_labels_used": False,
                    "row_level_numerical_vectors_persisted": False,
                    "raw_note_text_persisted": False,
                },
                schema=POST_EXTRACTION_REVIEW_UNRESOLVED_ONTOLOGY_SCHEMA_VERSION,
            )
            failed_names = ", ".join(map(str, final_ontology.get("failed_names") or ()))
            raise RuntimeError(
                "post-extraction review cannot freeze an unresolved retained registry: "
                "locally grounded category-ontology hazards for "
                f"{failed_names}; reasons="
                f"{final_ontology.get('failed_names_by_reason')}"
            )

        # The adaptive registry is now frozen. Only now may extraction touch
        # unconsumed review partitions or outer-heldout text. Preserve the
        # already evaluated row values and fill only the remaining rows. An
        # authenticated overlay is allowed to project the full dataset after
        # freeze, but it must reproduce every accumulated value exactly.
        accumulated_ids = tuple(map(int, current_extracted["_oci_row_id"].tolist()))
        remaining_label_free = label_free.loc[
            ~label_free["_oci_row_id"].isin(accumulated_ids)
        ].reset_index(drop=True)
        final_completion_audit: dict[str, Any]
        if self.cache_overlay is not None:
            raw_final, provider_audit = self._extract(
                label_free,
                current_specs,
                use_cache_overlay=True,
            )
            final_extracted = self._validated_extraction_projection(
                raw_final,
                label_free=label_free,
                specs=current_specs,
                source="post-freeze authenticated-overlay extraction",
            )
            accumulated_label_free = (
                label_free.set_index("_oci_row_id", drop=False)
                .loc[list(accumulated_ids)]
                .reset_index(drop=True)
            )
            replayed_accumulated = self._select_extraction_rows(
                final_extracted,
                label_free=accumulated_label_free,
                specs=current_specs,
                source="post-freeze overlay accumulated-row replay",
            )
            feature_columns = [
                column for spec in current_specs for column in expected_extraction_columns(spec)
            ]
            assert_frame_equal(
                replayed_accumulated[feature_columns].reset_index(drop=True),
                current_extracted[feature_columns].reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
                obj="post-freeze overlay replay of adaptively evaluated extraction",
            )
            final_completion_audit = {
                "mode": "full_authenticated_overlay_after_registry_freeze",
                "provider_audit": provider_audit,
                "remaining_row_count_before_full_projection": len(remaining_label_free),
                "accumulated_rows_replayed_exactly": True,
            }
        elif remaining_label_free.empty:
            final_extracted = self._select_extraction_rows(
                current_extracted,
                label_free=label_free,
                specs=current_specs,
                source="complete accumulated review extraction",
            )
            final_completion_audit = {
                "mode": "already_complete_after_registry_freeze",
                "provider_audit": None,
                "remaining_row_count_before_full_projection": 0,
                "accumulated_rows_replayed_exactly": True,
            }
        else:
            raw_remaining, provider_audit = self._extract(
                remaining_label_free,
                current_specs,
                use_cache_overlay=False,
            )
            remaining_extracted = self._validated_extraction_projection(
                raw_remaining,
                label_free=remaining_label_free,
                specs=current_specs,
                source="post-freeze remaining-row extraction",
            )
            final_extracted = self._combine_extraction_row_scopes(
                [current_extracted, remaining_extracted],
                label_free=label_free,
                specs=current_specs,
                source="post-freeze complete extraction",
            )
            final_completion_audit = {
                "mode": "remaining_rows_only_after_registry_freeze",
                "provider_audit": provider_audit,
                "remaining_row_count_before_full_projection": len(remaining_label_free),
                "accumulated_rows_replayed_exactly": True,
            }
        current_extracted = final_extracted

        audit = {
            "enabled": True,
            "configured_rounds": rounds,
            "rounds_completed": len(round_records),
            "stopped_by_agent_or_no_change": stopped_by_agent,
            "quality_retry_exhausted": quality_retry_exhausted,
            "configured_max_quality_retries_per_gate": max_quality_retries,
            "review_attempt_count": total_review_attempts,
            "partition_schedule": schedule.audit,
            "partition_provider": self.review_partition_provider_identity,
            "spent_evidence_provider": self.review_spent_evidence_provider_identity,
            "hierarchical_frozen_review_evidence": (
                None
                if frozen_hierarchical_review_evidence is None
                else frozen_hierarchical_review_evidence.as_binding_dict()
            ),
            "hierarchical_first_gate_materialization_intent": (
                None
                if hierarchical_first_gate_materialization_intent is None
                else {
                    "content_sha256": (
                        hierarchical_first_gate_materialization_intent.content_sha256
                    ),
                    "source_cache_key": (
                        hierarchical_first_gate_materialization_intent.body["source_cache_key"]
                    ),
                    "materialization_boundary": (
                        hierarchical_first_gate_materialization_intent.body[
                            "materialization_boundary"
                        ]
                    ),
                }
            ),
            "first_gate_numerical_materialization_before_discovery": False,
            "round_1_frozen_accepted_support_atoms_used": bool(hierarchical_review),
            "later_round_fresh_exact_spent_stage1_hierarchy_required": bool(
                hierarchical_review and rounds >= 2
            ),
            "same_frozen_accepted_support_atoms_authorized_for_later_rounds_by_policy": False,
            "same_frozen_accepted_support_atoms_used_for_every_executed_round": bool(
                hierarchical_review and not adaptive_execution_records
            ),
            "later_round_complete_catalog_sent_to_legacy_review_agent": False,
            "later_round_all_ten_architectures_incorporation_required": bool(
                hierarchical_review and rounds >= 2
            ),
            "later_round_all_ten_architectures_incorporated": bool(adaptive_execution_records),
            "adaptive_hierarchy_execution_count": len(adaptive_execution_records),
            "adaptive_hierarchy_executed_round_indices": sorted(
                {int(row["review_round"]) for row in adaptive_execution_records}
            ),
            "adaptive_hierarchy_execution_records": adaptive_execution_records,
            "original_content_addressed_review_evidence_ids_preserved": bool(hierarchical_review),
            "initial_adaptive_registry_provenance": initial_adaptive_registry_audit,
            "final_adaptive_registry_provenance_sha256": (
                None
                if current_adaptive_registry is None
                else _content_sha256([item.as_prompt_item() for item in current_adaptive_registry])
            ),
            "final_adaptive_registry_private_items": (
                None
                if current_adaptive_registry is None
                else [item.as_prompt_item() for item in current_adaptive_registry]
            ),
            "spent_evidence_context_epoch_policy": (_spent_evidence_context_epoch_policy_audit()),
            "gate_source_provider": self.review_gate_source_provider_identity,
            "gate_feature_bank_provider": self.review_gate_feature_bank_provider_identity,
            "required_source_signals": self.config.require_review_source_signals,
            "required_feature_banks": self.config.require_review_feature_banks,
            "degraded_review_without_all_upstream_explicitly_allowed": (
                self.config.allow_degraded_review_without_all_upstream
            ),
            "initial_contract_sha256": initial_hashes,
            "initial_selector_evidence_audit": initial_selector_evidence_audit,
            "initial_selector_future_gate_exclusion_verified": bool(
                initial_selector_evidence_audit
                and initial_selector_evidence_audit.get("full_outer_discovery_evidence_used")
                is False
                and initial_selector_evidence_audit.get(
                    "future_gate_text_or_labels_supplied_to_provider"
                )
                is False
                and initial_selector_evidence_audit.get("consumer_review_round") == 0
                and initial_selector_evidence_audit.get("spent_evidence_context_epoch") == 0
                and initial_selector_evidence_audit.get("context_epoch_policy_version")
                == SPENT_EVIDENCE_CONTEXT_EPOCH_POLICY_VERSION
            ),
            "final_contract_sha256": [extraction_contract_sha256(spec) for spec in current_specs],
            "round_audits": round_records,
            "initial_spent_partition_count": len(schedule.initial_spent_fold_ids),
            "valid_operation_proposal_count": valid_operation_proposals,
            "candidate_quality_rejection_count": candidate_quality_rejections,
            "candidate_quality_retry_count": candidate_quality_retries,
            "retained_ontology_rejection_count": retained_ontology_rejections,
            "retained_ontology_retry_count": retained_ontology_retries,
            "unresolved_ontology_convergence_rejection_count": (
                unresolved_ontology_convergence_rejections
            ),
            "unresolved_ontology_convergence_retry_exhausted": (
                unresolved_ontology_convergence_retry_exhausted
            ),
            "response_validation_rejection_count": response_validation_rejections,
            "response_validation_retry_count": response_validation_retries,
            "response_validation_retry_exhausted": response_validation_retry_exhausted,
            "candidate_workspace_policy_version": (
                POST_EXTRACTION_REVIEW_CANDIDATE_WORKSPACE_POLICY_VERSION
            ),
            "candidate_workspace_stage_count": candidate_workspace_stage_count,
            "candidate_workspace_stages_are_not_acceptances": True,
            "final_retained_registry_ontology_guard": final_ontology,
            "source_text_temporal_policy": source_text_temporal_policy_audit(),
            "gate_evaluated_proposal_count": gate_evaluated_proposals,
            "consumed_gate_count": len(spent_fold_ids) - len(schedule.initial_spent_fold_ids),
            "every_gate_evaluated_proposal_consumed_exactly_one_gate": bool(
                gate_evaluated_proposals
                == len(spent_fold_ids) - len(schedule.initial_spent_fold_ids)
            ),
            "prior_gate_feedback_diagnostic_count": len(prior_gate_feedback),
            "final_estimator_refit_scope": "complete_outer_train",
            "sealed_row_extraction_policy": (
                "initial_spent_only_then_gate_only_after_quality_then_remaining_after_freeze"
            ),
            "initial_extraction_saw_only_initial_spent_rows": True,
            "candidate_quality_extraction_saw_only_spent_rows": True,
            "retained_ontology_evaluated_all_contracts_on_spent_rows": True,
            "gate_extraction_started_only_after_candidate_quality_passed": True,
            "gate_provider_bind_input_contract": (
                "spent_observable_rows_plus_exact_gate_ids_and_text_only_v1"
            ),
            "gate_treatment_or_outcome_supplied_to_providers": False,
            "unconsumed_and_outer_heldout_text_extraction_started_after_registry_freeze": True,
            "post_freeze_extraction_completion": final_completion_audit,
            "outer_heldout_labels_used": False,
            "raw_reasoning_persisted": False,
            "row_level_numerical_vectors_persisted_in_round_audits": False,
            "aggregate_numerical_diagnostics_persisted_in_round_audits": bool(round_records),
        }
        return current_specs, current_extracted, audit

    def run(
        self,
        *,
        prepared_hierarchical_batch: PreparedHierarchicalDiscoveryBatch | None = None,
        hierarchy_execution_authorization: object | None = None,
    ) -> AllEvidenceFusionRunResult:
        if (prepared_hierarchical_batch is None) != (hierarchy_execution_authorization is None):
            raise ValueError(
                "prepared_hierarchical_batch and hierarchy_execution_authorization "
                "must be supplied together"
            )
        if prepared_hierarchical_batch is not None and not self.hierarchical_discovery_enabled:
            raise RuntimeError(
                "production hierarchy authorization requires hierarchical discovery mode"
            )
        hierarchical_preparation: PreparedHierarchicalDiscoveryBatch | None = None
        hierarchical_batch_result: ApprovedHierarchicalDiscoveryBatchResult | None = None
        production_runtime_binding: Mapping[str, Any] | None = None
        hierarchical_runtime_by_fold: dict[
            int,
            tuple[
                PreparedHierarchicalDiscoveryFold,
                Any,
                FrozenHierarchicalReviewEvidence,
            ],
        ] = {}
        if self.hierarchical_discovery_enabled:
            if prepared_hierarchical_batch is not None:
                if type(prepared_hierarchical_batch) is not PreparedHierarchicalDiscoveryBatch:
                    raise TypeError(
                        "production execution requires the concrete prepared hierarchy batch"
                    )
                if self.hierarchical_discovery_approved_batch_sha256 is not None:
                    raise ValueError(
                        "production internal authorization cannot be combined with a "
                        "caller-supplied hierarchy digest"
                    )
                if self.hierarchical_preparation_dir is None or (
                    prepared_hierarchical_batch.input_manifest_path.parent.resolve()
                    != self.hierarchical_preparation_dir.resolve()
                ):
                    raise ValueError("prepared hierarchy batch belongs to another runner root")
                hierarchical_preparation = prepared_hierarchical_batch
                hierarchical_batch_result = (
                    hierarchical_preparation.execute_with_internal_authorization(
                        authorization=hierarchy_execution_authorization,
                        runner=self,
                    )
                )
                production_runtime_binding = (
                    hierarchy_execution_authorization._consumed_runtime_binding_for_runner(
                        prepared_batch=hierarchical_preparation,
                        runner=self,
                    )
                )
            else:
                hierarchical_preparation = self.prepare_hierarchical_discovery_batch()
                approved_sha256 = self.hierarchical_discovery_approved_batch_sha256
                if approved_sha256 is None:
                    raise RuntimeError(
                        "hierarchical discovery is prepared but not approved; inspect "
                        f"{hierarchical_preparation.batch_packet_path} and rerun with "
                        "hierarchical_discovery_approved_batch_sha256="
                        f"{hierarchical_preparation.approval_sha256}"
                    )
                # The coordinator compares the approval before any fold preflight,
                # job-cache lookup, or remote JSON job.
                hierarchical_batch_result = hierarchical_preparation.execute(
                    approved_batch_sha256=approved_sha256
                )
            hierarchical_batch_result.validate_authentication()
            assert self.hierarchical_review_evidence_policy is not None
            if len(hierarchical_batch_result.ordered_fold_results) != len(
                hierarchical_preparation.folds
            ):
                raise RuntimeError("hierarchical batch result lost one or more outer folds")
            result_rows: list[dict[str, Any]] = []
            for prepared_fold, ordered_result in zip(
                hierarchical_preparation.folds,
                hierarchical_batch_result.ordered_fold_results,
            ):
                if ordered_result.outer_fold != prepared_fold.outer_fold:
                    raise RuntimeError("hierarchical result order differs from preparation")
                discovery_result = ordered_result.result
                discovery_result.validate_authentication()
                frozen_review = freeze_hierarchical_review_evidence(
                    catalog=prepared_fold.catalog,
                    completed=discovery_result.completed,
                    config=(self.hierarchical_review_evidence_policy.materializer_config()),
                )
                fold_result_path = (
                    self.hierarchical_preparation_dir
                    / f"outer_fold_{prepared_fold.outer_fold:03d}"
                    / "authenticated_hierarchical_discovery_result.json"
                )
                fold_result_sha256 = _write_immutable_json(
                    fold_result_path,
                    {
                        "outer_fold": prepared_fold.outer_fold,
                        "batch_approval_sha256": (hierarchical_batch_result.batch_approval_sha256),
                        "batch_result_sha256": hierarchical_batch_result.result_sha256,
                        "fold_result_binding": ordered_result.binding,
                        "compiled_registry_sha256": (
                            discovery_result.compiled_registry.registry_sha256
                        ),
                        "compiled_specs": discovery_result.compiled_registry.specs,
                        "frozen_review_evidence": frozen_review.as_binding_dict(),
                        "raw_reasoning_persisted": False,
                        "row_level_numerical_values_persisted": False,
                    },
                    schema=HIERARCHICAL_DISCOVERY_BATCH_RESULT_SCHEMA_VERSION,
                )
                result_rows.append(
                    {
                        "outer_fold": prepared_fold.outer_fold,
                        "fold_result_path": str(fold_result_path),
                        "fold_result_content_sha256": fold_result_sha256,
                        "fold_result_binding": ordered_result.binding,
                        "frozen_review_evidence_binding_sha256": (frozen_review.binding_sha256),
                    }
                )
                hierarchical_runtime_by_fold[prepared_fold.outer_fold] = (
                    prepared_fold,
                    discovery_result,
                    frozen_review,
                )
            assert self.hierarchical_preparation_dir is not None
            _write_immutable_json(
                self.hierarchical_preparation_dir / "authenticated_hierarchical_batch_result.json",
                {
                    "batch_approval_sha256": (hierarchical_batch_result.batch_approval_sha256),
                    "batch_result_sha256": hierarchical_batch_result.result_sha256,
                    "input_manifest_sha256": (hierarchical_batch_result.input_manifest_sha256),
                    "frozen_review_policy_sha256": (
                        hierarchical_batch_result.frozen_review_policy_sha256
                    ),
                    "ordered_fold_results": result_rows,
                    "all_fold_discovery_completed_before_per_fold_modeling": True,
                },
                schema=HIERARCHICAL_DISCOVERY_BATCH_RESULT_SCHEMA_VERSION,
            )

        current_cache_overlay_identity = _review_provider_identity(
            self.cache_overlay,
            label="cache_overlay",
        )
        if current_cache_overlay_identity != self.cache_overlay_identity:
            raise RuntimeError("frozen extraction-cache overlay identity changed before run")
        data, dataset_sha256 = _load_sanitized_dataset_snapshot(
            self.dataset_path,
            text_column=self.config.text_column,
            treatment_column=self.config.treatment_column,
            outcome_column=self.config.outcome_column,
        )
        if (
            hierarchical_preparation is not None
            and dataset_sha256 != hierarchical_preparation.dataset_sha256
        ):
            raise RuntimeError(
                "dataset bytes changed after hierarchical batch preparation/execution"
            )
        if (
            production_runtime_binding is not None
            and dataset_sha256 != production_runtime_binding["dataset_artifact"]["sha256"]
        ):
            raise RuntimeError("dataset differs from the production runtime authorization")
        external_validation: Mapping[str, Any] | None = None
        legacy = None
        tfidf = None
        reference_source: Mapping[str, Any] | None = None
        if self.reference_only_stage1_mode:
            direct_folds, split_rows = self._reference_only_outer_split_rows()
            folds = list(direct_folds)
            reference_source = self._reference_only_source_identity()
            if dataset_sha256 != reference_source[
                "prepared_cohort_artifact_sha256"
            ]:
                raise RuntimeError(
                    "dataset differs from the reference-only runtime binding"
                )
            if production_runtime_binding is not None and (
                production_runtime_binding.get(
                    "reference_only_runtime_binding_content_sha256"
                )
                != reference_source["runtime_binding_content_sha256"]
            ):
                raise RuntimeError(
                    "reference-only hierarchy authorization belongs to another "
                    "provider runtime"
                )
        else:
            assert self.tfidf_handoff_path is not None
            assert self.legacy_handoff_path is not None
            if self.tfidf_validator is not None:
                external_validation = self.tfidf_validator(
                    dataset=data.drop(columns=["_oci_row_id"]),
                    handoff_path=self.tfidf_handoff_path,
                )
                if (
                    not isinstance(external_validation, Mapping)
                    or external_validation.get("status") != "passed"
                ):
                    raise RuntimeError(
                        "external TF-IDF handoff validation did not pass"
                    )
            legacy = load_legacy_full_outer_evidence(self.legacy_handoff_path)
            tfidf = load_resealed_tfidf_handoff(
                self.tfidf_handoff_path,
                dataset_row_count=len(data),
                require_registry_seal=self.config.require_registry_seal,
            )
            if production_runtime_binding is not None and (
                legacy.artifact_sha256
                != production_runtime_binding[
                    "legacy_handoff_artifact"
                ]["sha256"]
                or tfidf.artifact_sha256
                != production_runtime_binding[
                    "tfidf_handoff_artifact"
                ]["sha256"]
            ):
                raise RuntimeError(
                    "legacy or TF-IDF handoff differs from the production "
                    "runtime authorization"
                )
            folds = sorted(tfidf.full_rows_by_outer_fold)
            if set(legacy.rows_by_outer_fold) != set(folds):
                raise ValueError(
                    "legacy and TF-IDF full-outer fold sets do not match exactly"
                )
            split_rows = {
                outer_fold: {
                    "fit_row_ids": tuple(
                        map(
                            int,
                            tfidf.full_rows_by_outer_fold[outer_fold][
                                "fit_row_ids"
                            ],
                        )
                    ),
                    "heldout_row_ids": tuple(
                        map(
                            int,
                            tfidf.full_rows_by_outer_fold[outer_fold][
                                "heldout_row_ids"
                            ],
                        )
                    ),
                }
                for outer_fold in folds
            }
        legacy_split_audit: Mapping[str, Any] | None = None
        if (
            not self.reference_only_stage1_mode
            and production_runtime_binding is not None
            and (
            (production_runtime_binding["legacy_primary_predictions_artifact"] is None)
            != (self.legacy_primary_predictions_path is None)
            )
        ):
            raise RuntimeError(
                "legacy primary split registration changed after production authorization"
            )
        if self.legacy_primary_predictions_path is not None:
            assert tfidf is not None
            legacy_heldout, legacy_primary_sha256 = (
                _load_outer_splits_from_primary_predictions_snapshot(
                    self.legacy_primary_predictions_path,
                    dataset_row_count=len(data),
                )
            )
            if set(legacy_heldout) != set(folds):
                raise ValueError("legacy primary-prediction fold set does not match TF-IDF")
            for outer_fold in folds:
                tfidf_heldout = set(
                    map(
                        int,
                        tfidf.full_rows_by_outer_fold[outer_fold]["heldout_row_ids"],
                    )
                )
                if set(legacy_heldout[outer_fold]) != tfidf_heldout:
                    raise ValueError(
                        f"legacy and TF-IDF heldout splits differ for fold {outer_fold}"
                    )
            legacy_split_audit = {
                "path": str(self.legacy_primary_predictions_path),
                "sha256": legacy_primary_sha256,
                "columns_read": ["_oci_row_id", "outer_fold_or_cv_fold"],
                "matches_tfidf_outer_splits": True,
            }
            if (
                production_runtime_binding is not None
                and legacy_primary_sha256
                != production_runtime_binding["legacy_primary_predictions_artifact"]["sha256"]
            ):
                raise RuntimeError(
                    "legacy primary splits differ from the production runtime authorization"
                )
        unexpected_pools = set(self.candidate_pool_paths) - set(folds)
        unexpected_queries = set(self.query_evidence_by_fold) - set(folds)
        unexpected_orphans = set(self.tfidf_orphan_artifacts_by_fold) - set(folds)
        if unexpected_pools or unexpected_queries or unexpected_orphans:
            raise ValueError(
                "candidate/query/orphan artifact registry contains an unknown outer fold"
            )
        missing_required_queries = set(folds) - set(self.query_evidence_by_fold)
        if (
            self.config.require_neural_query_moments
            and self.config.post_extraction_review_rounds == 0
            and missing_required_queries
        ):
            raise ValueError(
                "learned neural query-moment evidence is required but unregistered for "
                f"outer folds {sorted(missing_required_queries)}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        input_manifest = {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "initial_discovery": (
                {
                    "mode": "hierarchical_all_active_stage1_architectures",
                    "preparation_input_manifest_path": str(
                        hierarchical_preparation.input_manifest_path
                    ),
                    "preparation_input_manifest_sha256": (
                        hierarchical_preparation.input_manifest_sha256
                    ),
                    "batch_packet_path": str(hierarchical_preparation.batch_packet_path),
                    "batch_approval_sha256": hierarchical_preparation.approval_sha256,
                    "batch_result_sha256": hierarchical_batch_result.result_sha256,
                    "frozen_review_policy_sha256": (
                        hierarchical_batch_result.frozen_review_policy_sha256
                    ),
                    "all_fold_discovery_completed_before_per_fold_modeling": True,
                    "legacy_compact_fusion_agent_used": False,
                }
                if (hierarchical_preparation is not None and hierarchical_batch_result is not None)
                else {
                    "mode": "legacy_compact_fusion",
                    "legacy_compact_fusion_agent_used": True,
                }
            ),
            "source_text_temporal_policy": source_text_temporal_policy_audit(),
            "spent_evidence_context_epoch_policy": (_spent_evidence_context_epoch_policy_audit()),
            # Bind every prediction-affecting runner option at the first
            # immutable boundary.  Without this, reusing an output directory
            # with (for example) a different extraction model or head seed can
            # pass the input-manifest check, reuse an old fusion response, and
            # do substantial work before a later fold artifact finally
            # detects the mismatch.
            "effective_runner_config": asdict(self.config),
            "post_extraction_review_providers": {
                "spent_only_discovery": self.review_spent_evidence_provider_identity,
                "partition": self.review_partition_provider_identity,
                "calibrated_gate_sources": self.review_gate_source_provider_identity,
                "role_aware_gate_feature_banks": (self.review_gate_feature_bank_provider_identity),
                "numerical_cache_persistence": "provider_defined_and_identity_bound",
            },
            "post_extraction_review_extraction_semantics": {
                "adaptive_review_enabled": bool(self.config.post_extraction_review_rounds > 0),
                "provider_declares_request_group_dependency": bool(
                    getattr(
                        self.extraction_provider,
                        "extraction_request_group_dependent",
                        False,
                    )
                ),
                "contract_local_request_semantics_verified": bool(
                    callable(
                        getattr(
                            self.extraction_provider,
                            "adaptive_review_contract_local_extraction",
                            None,
                        )
                    )
                    and self.extraction_provider.adaptive_review_contract_local_extraction()
                ),
                "required_for_selective_review": True,
                "enforcement_version": ADAPTIVE_REVIEW_CONTRACT_LOCAL_EXTRACTION_VERSION,
            },
            "post_extraction_review_response_boundary": {
                "prompt_version": POST_EXTRACTION_REVIEW_PROMPT_VERSION,
                "response_schema_version": POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
                "request_schema_version": POST_EXTRACTION_REVIEW_REQUEST_SCHEMA_VERSION,
                "response_cache_schema_version": (
                    POST_EXTRACTION_REVIEW_RESPONSE_CACHE_SCHEMA_VERSION
                ),
                "failure_cache_schema_version": (POST_EXTRACTION_REVIEW_FAILURE_SCHEMA_VERSION),
                "round_audit_schema_version": POST_EXTRACTION_REVIEW_ROUND_SCHEMA_VERSION,
                "operation_apply_policy_version": (
                    POST_EXTRACTION_REVIEW_OPERATION_APPLY_POLICY_VERSION
                ),
                "ordered_extraction_projection_sha256_version": (
                    ORDERED_EXTRACTION_PROJECTION_SHA256_VERSION
                ),
                "fresh_response_normalization_version": (
                    POST_EXTRACTION_REVIEW_FRESH_NORMALIZATION_VERSION
                ),
                "grounding_repair_version": (POST_EXTRACTION_REVIEW_GROUNDING_REPAIR_VERSION),
                "response_validation_retry_policy_version": (
                    POST_EXTRACTION_REVIEW_RESPONSE_VALIDATION_RETRY_POLICY_VERSION
                ),
                "cached_failure_replay_enabled": True,
                "invalid_raw_response_persisted": False,
                "invalid_raw_reasoning_persisted": False,
                "cached_response_normalization_enabled": False,
                "candidate_workspace_policy_version": (
                    POST_EXTRACTION_REVIEW_CANDIDATE_WORKSPACE_POLICY_VERSION
                ),
            },
            "final_upstream_model_inputs": {
                "producer": self.final_upstream_producer_identity,
                "active": self.final_upstream_producer is not None,
                "required": self.config.require_final_upstream_inputs,
                "neural_query_inputs_required": (
                    self.config.require_final_upstream_neural_query_inputs
                ),
                "producer_api_accepts_outer_heldout_labels": False,
                "activation_boundary": "after_post_extraction_registry_freeze",
            },
            "final_ite_estimator": {
                "schema_version": FINAL_ITE_ESTIMATOR_AUDIT_SCHEMA_VERSION,
                "mode": (
                    FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID
                    if (
                        self.raw_final_upstream_producer is not None
                        or self.reference_only_stage1_mode
                    )
                    else "structured_interaction_head_degraded_fallback"
                ),
                "strict_causal_forest_active": (
                    self.raw_final_upstream_producer is not None
                    or self.reference_only_stage1_mode
                ),
                "strict_causal_forest_required": self.config.require_final_causal_forest,
                "exact_raw_runtime_producer": self.raw_final_upstream_producer_identity,
                "forest_backend": self.final_causal_forest_backend_identity,
                "fixed_prior_working_backend_active": bool(
                    (
                        self.raw_final_upstream_producer is not None
                        or self.reference_only_stage1_mode
                    )
                    and type(self.final_causal_forest_backend) is FixedCausalForestHeadBackend
                    and not self.final_causal_forest_backend_was_injected
                ),
                "test_backend_injected": self.final_causal_forest_backend_was_injected,
                "structured_interaction_fallback_allowed_only_without_exact_runtime": True,
                "outer_heldout_label_input_channel": False,
            },
            "conditional_extraction_cache_overlay": {
                "active": self.cache_overlay is not None,
                "overlay_identity": self.cache_overlay_identity,
                "declared_artifacts_authenticated_only_on_exact_hit": True,
                "authenticated_hit_provenance_persisted_in_extraction_report": True,
            },
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": dataset_sha256,
            "sanitized_dataset_row_count": len(data),
            "sanitized_dataset_text_fingerprint": ordered_dataset_text_fingerprint(
                data,
                text_column=self.config.text_column,
            ),
            "stage1_reference_source": reference_source,
            "legacy_handoff_path": (
                None if legacy is None else legacy.artifact_path
            ),
            "legacy_handoff_sha256": (
                None if legacy is None else legacy.artifact_sha256
            ),
            "legacy_non_full_contexts_reduced_not_exposed_raw": (
                None if legacy is None else legacy.ignored_non_full_context_count
            ),
            "legacy_exact_inner_contexts_validated_and_used_for_recurrence": (
                None if legacy is None else legacy.validated_inner_context_count
            ),
            "legacy_inner_contexts_per_outer": (
                None if legacy is None else legacy.inner_contexts_per_outer
            ),
            "legacy_exact_inner_recurrence_schema_version": (EXACT_INNER_RECURRENCE_VERSION),
            "legacy_exact_inner_recurrence_group_count": (
                None
                if legacy is None
                else legacy.exact_inner_recurrence_group_count
            ),
            "legacy_exact_inner_recurrent_term_count": (
                None
                if legacy is None
                else legacy.exact_inner_recurrent_term_count
            ),
            "legacy_diagnostic_fields_removed_before_fusion": (
                None if legacy is None else legacy.dropped_diagnostic_field_count
            ),
            "legacy_primary_prediction_split_audit": legacy_split_audit,
            "tfidf_handoff_path": None if tfidf is None else tfidf.artifact_path,
            "tfidf_handoff_sha256": (
                None if tfidf is None else tfidf.artifact_sha256
            ),
            "tfidf_split_registry_content_hash": (
                None if tfidf is None else tfidf.split_registry_content_hash
            ),
            "tfidf_structural_validation": (
                None if tfidf is None else tfidf.structural_validation
            ),
            "tfidf_external_validation": external_validation,
            "tfidf_orphan_ngram_evidence": {
                "enabled": self.config.include_tfidf_orphan_ngrams,
                "required": self.config.require_tfidf_orphan_ngrams,
                "adapter_config": (
                    None
                    if self.config.orphan_ngram_adapter is None
                    else asdict(self.config.orphan_ngram_adapter)
                ),
                "explicit_per_fold_registry": {
                    str(fold): {
                        "path": str(artifact.path),
                        "declared_sha256": artifact.artifact_sha256,
                    }
                    for fold, artifact in sorted(self.tfidf_orphan_artifacts_by_fold.items())
                },
            },
            "neural_query_moment_evidence": {
                "requirement_mode": (
                    "adaptive_context_fit"
                    if (
                        self.config.post_extraction_review_rounds > 0
                        and self.config.require_final_upstream_neural_query_inputs
                    )
                    else (
                        "authenticated_fold_artifact"
                        if self.config.require_neural_query_moments
                        else "optional_authenticated_artifact_or_sparse_fallback"
                    )
                ),
                "required_for_every_fold": (
                    self.config.require_neural_query_moments
                    and self.config.post_extraction_review_rounds == 0
                ),
                "adaptive_context_fit_required": (
                    self.config.post_extraction_review_rounds > 0
                    and self.config.require_final_upstream_neural_query_inputs
                ),
                "sparse_fallback_enabled_when_unregistered": (
                    self.config.post_extraction_review_rounds == 0
                    and self.config.derive_sparse_query_moments_when_missing
                ),
                "registered_outer_folds": sorted(self.query_evidence_by_fold),
                "registered_artifact_usage": (
                    "adaptive_audit_only_excluded_from_selector_and_model_inputs"
                    if self.config.post_extraction_review_rounds > 0
                    else "nonadaptive_selector_evidence"
                ),
                "sparse_fallback_outer_folds": (
                    sorted(set(folds) - set(self.query_evidence_by_fold))
                    if (
                        self.config.post_extraction_review_rounds == 0
                        and self.config.derive_sparse_query_moments_when_missing
                    )
                    else []
                ),
                "per_fold_registry": {
                    str(fold): {
                        "path": str(artifact.path),
                        "artifact_sha256": artifact.artifact_sha256,
                        "outer_fold": artifact.outer_fold,
                        "scope": artifact.scope,
                        "fit_row_fingerprint": artifact.fit_row_fingerprint,
                        "heldout_row_fingerprint": artifact.heldout_row_fingerprint,
                    }
                    for fold, artifact in sorted(self.query_evidence_by_fold.items())
                },
            },
            "configured_columns_are_allowlisted_only": True,
            "source_dataset_non_allowlisted_columns_dropped_immediately": True,
        }
        input_manifest_path = self.output_dir / "immutable_input_manifest.json"
        input_manifest_hash = _write_immutable_json(
            input_manifest_path,
            input_manifest,
            schema=RUNNER_SCHEMA_VERSION,
        )

        fold_predictions: list[pd.DataFrame] = []
        fold_manifests: list[Path] = []
        label_free = data[["_oci_row_id", self.config.text_column]].copy()
        for outer_fold in folds:
            fold_dir = self.output_dir / f"outer_fold_{outer_fold:03d}"
            split_row = split_rows[outer_fold]
            full = (
                split_row
                if self.reference_only_stage1_mode
                else tfidf.full_rows_by_outer_fold[outer_fold]
            )
            train_ids = tuple(map(int, split_row["fit_row_ids"]))
            heldout_ids = tuple(map(int, split_row["heldout_row_ids"]))
            provenance = FoldEvidenceProvenance(
                outer_fold=outer_fold,
                train_row_ids=train_ids,
                heldout_row_ids=heldout_ids,
                scope="outer_train",
                artifact_id=f"all-evidence-outer-{outer_fold}",
            )
            completed_fold_manifest = (
                fold_dir / "immutable_fold_manifest.json"
            )
            completed_fold_prediction = (
                fold_dir / "frozen_predictions.parquet"
            )
            if completed_fold_manifest.is_file():
                try:
                    wrapped = json.loads(
                        completed_fold_manifest.read_text(
                            encoding="utf-8"
                        )
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"completed fold {outer_fold} manifest is unreadable"
                    ) from exc
                body = wrapped.get("body") if isinstance(wrapped, Mapping) else None
                if (
                    not isinstance(body, Mapping)
                    or wrapped.get("schema_version")
                    != FOLD_MANIFEST_SCHEMA_VERSION
                    or wrapped.get("content_sha256")
                    != _content_sha256(body)
                    or body.get("input_manifest_content_sha256")
                    != input_manifest_hash
                    or body.get("outer_fold") != outer_fold
                    or body.get("split_fingerprint")
                    != provenance.split_fingerprint
                    or body.get("train_row_fingerprint")
                    != row_set_fingerprint(train_ids)
                    or body.get("heldout_row_fingerprint")
                    != row_set_fingerprint(heldout_ids)
                    or body.get("stage1_reference_source")
                    != reference_source
                    or body.get("outer_heldout_outcomes_used") is not False
                    or body.get("oracle_columns_written") is not False
                    or Path(str(body.get("prediction_path", ""))).resolve()
                    != completed_fold_prediction.resolve()
                    or not completed_fold_prediction.is_file()
                    or body.get("prediction_sha256")
                    != sha256_file(completed_fold_prediction)
                ):
                    raise RuntimeError(
                        f"completed fold {outer_fold} checkpoint is invalid"
                    )
                selected_contracts = body.get("selected_contracts")
                if (
                    not isinstance(selected_contracts, list)
                    or not all(
                        isinstance(contract, Mapping)
                        for contract in selected_contracts
                    )
                    or body.get("selected_contract_sha256")
                    != [
                        extraction_contract_sha256(contract)
                        for contract in selected_contracts
                    ]
                ):
                    raise RuntimeError(
                        f"completed fold {outer_fold} selected-variable "
                        "checkpoint is invalid"
                    )
                expected_strict_forest = bool(
                    self.raw_final_upstream_producer is not None
                    or self.reference_only_stage1_mode
                )
                completed_estimator = body.get("final_ite_estimator")
                if (
                    not isinstance(completed_estimator, Mapping)
                    or completed_estimator.get("mode")
                    != (
                        FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID
                        if expected_strict_forest
                        else "structured_interaction_head_degraded_fallback"
                    )
                    or completed_estimator.get("strict_causal_forest_active")
                    != expected_strict_forest
                    or (
                        expected_strict_forest
                        and completed_estimator.get("forest_backend_identity")
                        != self.final_causal_forest_backend_identity
                    )
                ):
                    raise RuntimeError(
                        f"completed fold {outer_fold} final-estimator "
                        "checkpoint is invalid"
                    )
                prediction = pd.read_parquet(
                    completed_fold_prediction
                )
                _reject_forbidden_columns(
                    prediction.columns,
                    source=f"resumed fold {outer_fold} prediction",
                )
                if (
                    not {
                        "_oci_row_id",
                        "outer_fold",
                        "pred_ite_prob",
                    }.issubset(prediction.columns)
                    or body.get("prediction_columns")
                    != list(prediction.columns)
                    or tuple(
                        map(int, prediction["_oci_row_id"].tolist())
                    )
                    != heldout_ids
                    or set(
                        map(int, prediction["outer_fold"].tolist())
                    )
                    != {outer_fold}
                    or not np.isfinite(
                        prediction["pred_ite_prob"].to_numpy(dtype=float)
                    ).all()
                ):
                    raise RuntimeError(
                        f"completed fold {outer_fold} prediction is invalid"
                    )
                fold_predictions.append(prediction)
                fold_manifests.append(completed_fold_manifest)
                continue
            legacy_row = (
                None
                if legacy is None
                else legacy.rows_by_outer_fold[outer_fold]
            )
            if legacy_row is not None and int(
                legacy_row.get("n_rows", 0)
            ) not in {0, len(train_ids)}:
                raise ValueError(
                    f"legacy full-outer row count mismatch for fold {outer_fold}"
                )
            review_schedule: ReviewPartitionSchedule | None = None
            initial_selector_evidence_audit: Mapping[str, Any] | None = None
            query_audit: Mapping[str, Any] | None = None
            candidate_audit: Mapping[str, Any] | None = None
            hierarchical_runtime = hierarchical_runtime_by_fold.get(outer_fold)
            hierarchical_discovery_result: Any | None = None
            frozen_hierarchical_review: FrozenHierarchicalReviewEvidence | None = None
            hierarchical_first_gate_materialization_intent: (
                FirstGateMaterializationIntent | None
            ) = None
            hierarchical_reference_numerical_contract: (
                AuthenticatedReferenceOnlyDirectNumericalContract | None
            ) = None
            hierarchical_first_gate_catalog: RoleNeutralEvidenceCatalog | None = None
            if hierarchical_runtime is not None:
                (
                    prepared_hierarchical_fold,
                    hierarchical_discovery_result,
                    frozen_hierarchical_review,
                ) = hierarchical_runtime
                review_schedule = prepared_hierarchical_fold.schedule
                hierarchical_first_gate_materialization_intent = (
                    prepared_hierarchical_fold.first_gate_materialization_intent
                )
                hierarchical_reference_numerical_contract = (
                    prepared_hierarchical_fold.reference_only_direct_numerical_contract
                )
                hierarchical_first_gate_catalog = prepared_hierarchical_fold.catalog
                if set(
                    row_id for rows in review_schedule.row_ids_by_fold.values() for row_id in rows
                ) != set(train_ids):
                    raise RuntimeError(
                        "hierarchical prepared review schedule differs from final outer train"
                    )
                initial_selector_evidence_audit = dict(
                    prepared_hierarchical_fold.initial_spent_evidence_audit
                )
                orphan_audit = {
                    "excluded_from_adaptive_initial_selector": True,
                    "reason": "full_outer_artifact_depends_on_sealed_review_rows",
                    "hierarchical_spent_only_catalog_used": True,
                }
                query_audit = {
                    "full_outer_query_evidence_excluded_from_adaptive_initial_selector": True,
                    "spent_only_provider_used": True,
                    "hierarchical_catalog_contains_neural_query_moments": True,
                }
                if outer_fold in self.candidate_pool_paths:
                    candidate_audit = {
                        "registered_but_excluded_from_adaptive_initial_selector": True,
                        "reason": "candidate_pool_not_authenticated_to_initial_spent_rows",
                    }
            elif self.config.post_extraction_review_rounds > 0:
                outer_train = (
                    data.set_index("_oci_row_id", drop=False)
                    .loc[list(train_ids)]
                    .reset_index(drop=True)
                )
                review_schedule = self._review_schedule(
                    outer_train=outer_train,
                    outer_fold=outer_fold,
                )
                # Full-outer artifacts and candidate pools already aggregate
                # future gates. They remain available for nonadaptive runs but
                # cannot seed an adaptive selector.
                request, initial_selector_evidence_audit = self._spent_fusion_request(
                    data=data,
                    schedule=review_schedule,
                    spent_fold_ids=review_schedule.initial_spent_fold_ids,
                    outer_fold=outer_fold,
                    review_round=0,
                )
                orphan_audit = {
                    "excluded_from_adaptive_initial_selector": True,
                    "reason": "full_outer_artifact_depends_on_sealed_review_rows",
                }
                query_audit = {
                    "full_outer_query_evidence_excluded_from_adaptive_initial_selector": True,
                    "spent_only_provider_used": True,
                }
                if outer_fold in self.candidate_pool_paths:
                    candidate_audit = {
                        "registered_but_excluded_from_adaptive_initial_selector": True,
                        "reason": "candidate_pool_not_authenticated_to_initial_spent_rows",
                    }
            else:
                full_discovery = full.get("discovery") or {}
                tfidf_payload = {
                    "outer_fold": outer_fold,
                    "scope": "full_outer_train",
                    "discovery": {
                        "topic_banks": full_discovery.get("topic_banks") or {},
                        "topic_score_tests": full_discovery.get("topic_score_tests") or {},
                        "exact_inner_recurrence": full_discovery.get("exact_inner_recurrence"),
                    },
                }
                orphan_branch, orphan_audit = self._adapt_orphan_ngram_evidence(
                    outer_fold=outer_fold,
                    full_row=full,
                )
                if orphan_branch is not None:
                    tfidf_payload["discovery"]["effect_orphan_ngram_branch"] = orphan_branch
                inputs = [
                    FoldEvidenceInput(LEGACY_ALL_SOURCE, legacy_row, provenance),
                    FoldEvidenceInput(TFIDF_TOPIC_SOURCE, tfidf_payload, provenance),
                ]
                if outer_fold in self.query_evidence_by_fold:
                    adapted_query, query_audit = _load_query_evidence(
                        self.query_evidence_by_fold[outer_fold],
                        provenance=provenance,
                        config=self.config.query_moment_adapter,
                    )
                    if (
                        self.config.require_neural_query_moments
                        and not query_audit["artifact_declared_full_partition"]
                    ):
                        raise ValueError(
                            "required neural query-moment evidence must declare exact fit and "
                            f"heldout row IDs inside the hashed artifact for fold {outer_fold}"
                        )
                    inputs.append(adapted_query.as_fold_evidence_input())
                elif self.config.derive_sparse_query_moments_when_missing:
                    indexed_data = data.set_index("_oci_row_id", drop=False)
                    outer_train = indexed_data.loc[list(train_ids)]
                    adapted_query = derive_sparse_query_moment_evidence(
                        provenance=provenance,
                        outer_train_row_ids=outer_train["_oci_row_id"].tolist(),
                        outer_train_texts=outer_train[self.config.text_column].tolist(),
                        treatment=outer_train[self.config.treatment_column].tolist(),
                        outcome=outer_train[self.config.outcome_column].tolist(),
                        tfidf_topic_evidence=tfidf_payload,
                        config=self.config.query_moment_adapter,
                    )
                    inputs.append(adapted_query.as_fold_evidence_input())
                    query_audit = adapted_query.audit

                candidates: list[CandidateContract] = []
                if outer_fold in self.candidate_pool_paths:
                    candidates, candidate_audit = load_candidate_pool(
                        self.candidate_pool_paths[outer_fold],
                        expected_outer_fold=outer_fold,
                    )
                request = prepare_all_evidence_fusion(
                    inputs,
                    candidates=candidates,
                    max_candidates=self.config.max_candidates,
                )
            if hierarchical_discovery_result is not None:
                assert hierarchical_batch_result is not None
                assert hierarchical_preparation is not None
                hierarchical_discovery_result.validate_authentication()
                specs = hierarchical_discovery_result.compiled_registry.specs
                request_hash = hierarchical_discovery_result.wrapper_approval_sha256
                response_hash = hierarchical_discovery_result.result_sha256
                response_cache_path = (
                    self.hierarchical_preparation_dir
                    / f"outer_fold_{outer_fold:03d}"
                    / "authenticated_hierarchical_discovery_result.json"
                )
                staged_fusion_audit = None
                staged_fusion_audit_status = (
                    "not_applicable_hierarchical_architecture_at_a_time_discovery"
                )
                fusion_prompt_version = "hierarchical_all_architecture_discovery_precommitted_jobs"
                fusion_source_family_coverage = {
                    family: len(hierarchical_runtime[0].catalog.family_atoms(family))
                    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
                }
                fusion_response_audit = {
                    "mode": "hierarchical_all_active_stage1_architectures",
                    "batch_approval_sha256": (hierarchical_batch_result.batch_approval_sha256),
                    "batch_result_sha256": hierarchical_batch_result.result_sha256,
                    "wrapper_approval_sha256": (
                        hierarchical_discovery_result.wrapper_approval_sha256
                    ),
                    "inner_precommit_sha256": (
                        hierarchical_discovery_result.inner_precommit_sha256
                    ),
                    "completion_sha256": (
                        hierarchical_discovery_result.completed.completion_sha256
                    ),
                    "compiled_registry_sha256": (
                        hierarchical_discovery_result.compiled_registry.registry_sha256
                    ),
                    "runner_trace_sha256": (
                        hierarchical_discovery_result.runner_trace.trace_sha256
                    ),
                    "all_active_architectures_incorporated": True,
                    "one_architecture_at_a_time_before_cross_architecture_integration": True,
                    "legacy_all_evidence_dump_prompt_used": False,
                }
                initial_discovery_mode = "hierarchical_all_active_stage1_architectures"
            else:
                request_hash = _content_sha256(request.context())
                response_cache_path = fold_dir / "immutable_fusion_response.json"
                cached_response_record = _load_request_bound_fusion_response(
                    response_cache_path,
                    request_sha256=request_hash,
                )
                if cached_response_record is None:
                    response, response_hash = self._invoke_agent(request)
                    # Never persist an invalid remote response.
                    fusion_result = validate_all_evidence_fusion_response(request, response)
                    staged_fusion_audit = _validate_request_bound_staged_fusion_audit(
                        getattr(self.fusion_agent, "last_stage_audit", None),
                        request_sha256=request_hash,
                        response_sha256=response_hash,
                        outer_fold=outer_fold,
                        split_fingerprint=request.split_fingerprint,
                    )
                    staged_fusion_audit_status = (
                        "captured_and_request_bound"
                        if staged_fusion_audit is not None
                        else "unavailable_not_exposed_by_agent"
                    )
                    _write_immutable_json(
                        response_cache_path,
                        {
                            "request_sha256": request_hash,
                            "response": response,
                            "staged_fusion_audit": staged_fusion_audit,
                            "staged_fusion_audit_status": staged_fusion_audit_status,
                        },
                        schema=FUSION_RESPONSE_CACHE_SCHEMA_VERSION,
                    )
                else:
                    (
                        response,
                        cached_staged_fusion_audit,
                        staged_fusion_audit_status,
                    ) = cached_response_record
                    response_hash = _content_sha256(response)
                    # Revalidate cached JSON against the reconstructed immutable
                    # request before trusting any candidate IDs or contracts.
                    fusion_result = validate_all_evidence_fusion_response(request, response)
                    staged_fusion_audit = _validate_request_bound_staged_fusion_audit(
                        cached_staged_fusion_audit,
                        request_sha256=request_hash,
                        response_sha256=response_hash,
                        outer_fold=outer_fold,
                        split_fingerprint=request.split_fingerprint,
                    )
                specs = (
                    fusion_result.selected_specs
                    if fusion_result.mode == "select"
                    else list(fusion_result.proposed_specs)
                )
                fusion_prompt_version = request.context()["prompt_version"]
                fusion_source_family_coverage = request.source_family_coverage
                fusion_response_audit = fusion_result.response_audit
                initial_discovery_mode = "legacy_compact_fusion"
            if not specs:
                raise ValueError(f"fusion selected no executable contracts for fold {outer_fold}")
            if len({str(spec["name"]) for spec in specs}) != len(specs):
                raise ValueError("fusion returned duplicate explicit feature names")

            initial_extraction_label_free = label_free
            initial_extraction_scope = "full_dataset_nonadaptive"
            if review_schedule is not None:
                initial_spent_ids = review_schedule.row_ids(review_schedule.initial_spent_fold_ids)
                initial_extraction_label_free = (
                    label_free.set_index("_oci_row_id", drop=False)
                    .loc[list(initial_spent_ids)]
                    .reset_index(drop=True)
                )
                initial_extraction_scope = "initial_spent_rows_only"
            raw_extracted, extraction_audit = self._extract(
                initial_extraction_label_free,
                specs,
                use_cache_overlay=review_schedule is None,
            )
            extracted = self._validated_extraction_projection(
                raw_extracted,
                label_free=initial_extraction_label_free,
                specs=specs,
                source="initial extracted frame",
            )
            extraction_audit = {
                **dict(extraction_audit),
                "initial_extraction_scope": initial_extraction_scope,
                "initial_extraction_row_count": len(initial_extraction_label_free),
                "sealed_review_gate_texts_available_to_initial_extractor": False,
                "outer_heldout_texts_available_to_initial_extractor": bool(review_schedule is None),
                "cache_overlay_enabled_for_initial_extraction": bool(
                    self.cache_overlay is not None and review_schedule is None
                ),
            }
            initial_specs = [CandidateContract(spec).extraction_spec for spec in specs]
            specs, extracted, post_extraction_review_audit = self._run_post_extraction_review(
                data=data,
                label_free=label_free,
                outer_fold=outer_fold,
                train_ids=train_ids,
                initial_specs=initial_specs,
                initial_extracted=extracted,
                fold_dir=fold_dir,
                review_schedule=review_schedule,
                initial_selector_evidence_audit=initial_selector_evidence_audit,
                frozen_hierarchical_review_evidence=frozen_hierarchical_review,
                hierarchical_first_gate_materialization_intent=(
                    hierarchical_first_gate_materialization_intent
                ),
                hierarchical_first_gate_catalog=hierarchical_first_gate_catalog,
                hierarchical_approved_runner_identity=(
                    None
                    if hierarchical_discovery_result is None
                    else hierarchical_discovery_result.runner_trace.runner_identity
                ),
                hierarchical_approved_cache_identity=(
                    None
                    if hierarchical_discovery_result is None
                    else hierarchical_discovery_result.runner_trace.cache_identity
                ),
                hierarchical_family_explanations=(
                    None
                    if hierarchical_discovery_result is None
                    else dict(prepared_hierarchical_fold.agent.family_explanations)
                ),
            )
            model_frame = data.merge(
                extracted.drop(columns=[self.config.text_column], errors="ignore"),
                on="_oci_row_id",
                how="left",
                validate="one_to_one",
            )
            train = model_frame.set_index("_oci_row_id", drop=False).loc[list(train_ids)]
            heldout = model_frame.set_index("_oci_row_id", drop=False).loc[list(heldout_ids)]
            encoder = FoldTrainExplicitEncoder().fit(train, specs)
            explicit_train_x = encoder.transform(train)
            explicit_heldout_x = encoder.transform(heldout)
            train_x = explicit_train_x
            heldout_x = explicit_heldout_x
            explicit_feature_count = int(explicit_train_x.shape[1])
            final_model_input_names = list(encoder.feature_names_)
            final_upstream_audit: dict[str, Any] = {
                "enabled": False,
                "required": self.config.require_final_upstream_inputs,
                "neural_query_inputs_required": (
                    self.config.require_final_upstream_neural_query_inputs
                ),
                "direct_upstream_numerical_signals_used_as_final_model_inputs": False,
                "outer_heldout_labels_passed_to_producer": False,
                "activation_boundary": "after_post_extraction_registry_freeze",
                "row_level_numerical_vectors_persisted_in_runner_audit": False,
            }
            upstream_modifier_indices: tuple[int, ...] = ()
            head_regularization_grid = self.config.regularization_grid
            package: AuthenticatedFinalContextFitUpstreamBank | None = None
            upstream: _FinalUpstreamHeadInputs | None = None
            direct_fold_view: Any | None = None
            causal_forest_active = (
                self.raw_final_upstream_producer is not None
                or self.reference_only_stage1_mode
            )
            if self.reference_only_stage1_mode:
                bank = self.reference_only_numerical_bank
                meta_inner_fold_ids = bank.get_meta_inner_fold_ids(
                    outer_fold=outer_fold,
                    exact_outer_train_row_ids=train_ids,
                )
                direct_fold_view = bank.produce(
                    outer_fold=outer_fold,
                    outer_train_row_ids=train_ids,
                    outer_train_texts=tuple(
                        train[self.config.text_column].tolist()
                    ),
                    outer_train_treatment=train[
                        self.config.treatment_column
                    ].to_numpy(dtype=float),
                    outer_train_outcome=train[
                        self.config.outcome_column
                    ].to_numpy(dtype=float),
                    outer_heldout_row_ids=heldout_ids,
                    outer_heldout_texts=tuple(
                        heldout[self.config.text_column].tolist()
                    ),
                    meta_inner_fold_ids=meta_inner_fold_ids,
                )
                direct_fold_view.verify_authenticated_content()
                final_upstream_audit = {
                    "enabled": True,
                    "required": self.config.require_final_upstream_inputs,
                    "neural_query_inputs_required": (
                        self.config.require_final_upstream_neural_query_inputs
                    ),
                    "mode": (
                        "authenticated_role_neutral_direct_numerical_references"
                    ),
                    "reference_manifest_content_sha256": (
                        bank.manifest["content_sha256"]
                    ),
                    "runtime_binding_content_sha256": (
                        self.reference_only_stage1_runtime_payload[
                            "content_sha256"
                        ]
                    ),
                    "meta_inner_fold_ids_source": (
                        "authenticated_exact_inner_scope_plan"
                    ),
                    "meta_inner_fold_count": len(
                        set(map(int, meta_inner_fold_ids))
                    ),
                    "post_extraction_registry_frozen_before_production": True,
                    "activation_boundary": (
                        "after_post_extraction_registry_freeze"
                    ),
                    "direct_upstream_numerical_signals_used_as_final_model_inputs": (
                        True
                    ),
                    "combined_numerical_payload_persisted": False,
                    "fit_or_refit_performed": False,
                    "outer_heldout_labels_passed_to_producer": False,
                    "row_level_numerical_vectors_persisted_in_runner_audit": False,
                }
            elif self.final_upstream_producer is not None:
                producer_identity_sha256 = self._assert_final_upstream_producer_identity()
                if causal_forest_active:
                    raw_runtime_identity_sha256 = (
                        self._assert_raw_final_upstream_producer_identity()
                    )
                    if raw_runtime_identity_sha256 != producer_identity_sha256:
                        raise ValueError("final package and exact raw runtime identities diverged")
                meta_inner_fold_ids, meta_inner_audit = _build_final_upstream_meta_inner_fold_ids(
                    train,
                    n_splits=self.config.final_upstream_meta_inner_folds,
                    random_state=self.config.random_state,
                    outer_fold=outer_fold,
                    treatment_column=self.config.treatment_column,
                    outcome_column=self.config.outcome_column,
                    outcome_type=self.config.outcome_type,
                )
                package = self.final_upstream_producer.produce(
                    outer_fold=outer_fold,
                    outer_train_row_ids=train_ids,
                    outer_train_texts=tuple(train[self.config.text_column].tolist()),
                    outer_train_treatment=train[self.config.treatment_column].to_numpy(dtype=float),
                    outer_train_outcome=train[self.config.outcome_column].to_numpy(dtype=float),
                    outer_heldout_row_ids=heldout_ids,
                    outer_heldout_texts=tuple(heldout[self.config.text_column].tolist()),
                    meta_inner_fold_ids=meta_inner_fold_ids,
                )
                if self._assert_final_upstream_producer_identity() != producer_identity_sha256:
                    raise ValueError("final upstream producer identity changed during production")
                if causal_forest_active and (
                    self._assert_raw_final_upstream_producer_identity() != producer_identity_sha256
                ):
                    raise ValueError("exact raw runtime identity changed during production")
                upstream = _prepare_final_upstream_head_inputs(
                    package,
                    outer_fold=outer_fold,
                    expected_train_row_ids=train_ids,
                    expected_heldout_row_ids=heldout_ids,
                    expected_meta_inner_fold_ids=meta_inner_fold_ids,
                    expected_producer_identity_sha256=producer_identity_sha256,
                    require_neural_query_inputs=(
                        self.config.require_final_upstream_neural_query_inputs
                    ),
                )
                final_upstream_audit = {
                    **dict(upstream.audit),
                    "required": self.config.require_final_upstream_inputs,
                    "activation_boundary": "after_post_extraction_registry_freeze",
                    "post_extraction_registry_frozen_before_production": True,
                    "meta_inner_partition": meta_inner_audit,
                }
                if not causal_forest_active:
                    train_x = np.column_stack((explicit_train_x, upstream.train_values))
                    heldout_x = np.column_stack((explicit_heldout_x, upstream.heldout_values))
                    upstream_modifier_indices = tuple(
                        explicit_feature_count + index for index in upstream.modifier_indices
                    )
                    final_model_input_names.extend(upstream.model_input_names)
                    if len(final_model_input_names) != len(set(final_model_input_names)):
                        raise ValueError("final model input namespace contains duplicate columns")
                    head_regularization_grid = (
                        float(self.config.final_upstream_head_regularization),
                    )
                    final_upstream_audit.update(
                        {
                            "model_input_schema": {
                                "column_count": len(final_model_input_names),
                                "explicit_extraction_column_count": explicit_feature_count,
                                "upstream_column_count": len(upstream.model_input_names),
                                "upstream_model_input_names": list(upstream.model_input_names),
                                "schema_sha256": _content_sha256(final_model_input_names),
                            },
                            "head_regularization_policy": {
                                "grid": list(head_regularization_grid),
                                "singleton_grid_precommitted_in_runner_config": True,
                                "adaptive_regularization_choice_performed": False,
                                "observed_inner_validation_loss_may_be_audited_but_cannot_select": (
                                    True
                                ),
                            },
                            "modifier_routing": {
                                "interact_all_features": self.config.interact_all_features,
                                "all_calibrated_tau_sources_are_treatment_modifiers": True,
                                "raw_features_interact_only_for_effect_modifier_role_when_modifier_only": (
                                    True
                                ),
                            },
                        }
                    )

            head_tuning_audit: Mapping[str, Any] | None = None
            if causal_forest_active and direct_fold_view is not None:
                if self.final_causal_forest_backend is None:
                    raise RuntimeError(
                        "reference-only Stage 2 has no strict causal-forest backend"
                    )
                current_backend_identity = _review_provider_identity(
                    self.final_causal_forest_backend,
                    label="final_causal_forest_backend",
                )
                if (
                    current_backend_identity
                    != self.final_causal_forest_backend_identity
                ):
                    raise ValueError(
                        "final causal-forest backend identity changed"
                    )
                direct_blocks = direct_fold_view.forest_blocks()
                if (
                    direct_blocks.train_row_ids != train_ids
                    or direct_blocks.heldout_row_ids != heldout_ids
                ):
                    raise ValueError(
                        "direct numerical forest blocks changed fold row order"
                    )
                spec_roles = {
                    str(spec["name"]): frozenset(
                        map(str, spec.get("roles") or ())
                    )
                    for spec in specs
                }
                encoded_spec_names = tuple(
                    map(str, encoder.feature_spec_names_)
                )
                if len(encoded_spec_names) != explicit_train_x.shape[1]:
                    raise RuntimeError(
                        "explicit encoder lost its per-column feature lineage"
                    )
                explicit_effect_indices = tuple(
                    index
                    for index, name in enumerate(encoded_spec_names)
                    if "effect_modifier" in spec_roles.get(name, frozenset())
                )
                explicit_control_indices = tuple(
                    index
                    for index, name in enumerate(encoded_spec_names)
                    if "confounder" in spec_roles.get(name, frozenset())
                )

                def append_columns(
                    base: np.ndarray,
                    explicit: np.ndarray,
                    indices: tuple[int, ...],
                ) -> np.ndarray:
                    if not indices:
                        return np.asarray(base, dtype=np.float64)
                    return np.column_stack(
                        (
                            np.asarray(base, dtype=np.float64),
                            np.asarray(
                                explicit[:, list(indices)],
                                dtype=np.float64,
                            ),
                        )
                    )

                effect_train = append_columns(
                    direct_blocks.effect_train_values,
                    explicit_train_x,
                    explicit_effect_indices,
                )
                effect_heldout = append_columns(
                    direct_blocks.effect_heldout_values,
                    explicit_heldout_x,
                    explicit_effect_indices,
                )
                control_train = append_columns(
                    direct_blocks.control_train_values,
                    explicit_train_x,
                    explicit_control_indices,
                )
                control_heldout = append_columns(
                    direct_blocks.control_heldout_values,
                    explicit_heldout_x,
                    explicit_control_indices,
                )
                if effect_train.shape[1] < 1 or control_train.shape[1] < 1:
                    raise ValueError(
                        "strict direct causal forest requires nonempty effect "
                        "and control feature blocks"
                    )
                raw_tau = self.final_causal_forest_backend.fit_predict(
                    effect_train=np.array(effect_train, copy=True),
                    control_train=np.array(control_train, copy=True),
                    treatment=train[
                        self.config.treatment_column
                    ].to_numpy(dtype=float),
                    outcome=train[
                        self.config.outcome_column
                    ].to_numpy(dtype=float),
                    effect_heldout=np.array(effect_heldout, copy=True),
                    control_heldout=np.array(control_heldout, copy=True),
                )
                if (
                    _review_provider_identity(
                        self.final_causal_forest_backend,
                        label="final_causal_forest_backend",
                    )
                    != current_backend_identity
                ):
                    raise ValueError(
                        "final causal-forest backend identity changed during fit"
                    )
                pred_ite = np.asarray(raw_tau, dtype=np.float64)
                if (
                    pred_ite.shape != (len(heldout_ids),)
                    or not np.isfinite(pred_ite).all()
                ):
                    raise ValueError(
                        "strict direct causal forest returned malformed treatment effects"
                    )
                probability_difference_tolerance = float(
                    64 * np.finfo(np.float64).eps
                )
                if (
                    np.any(
                        pred_ite
                        < (-1.0 - probability_difference_tolerance)
                    )
                    or np.any(
                        pred_ite
                        > (1.0 + probability_difference_tolerance)
                    )
                ):
                    raise ValueError(
                        "strict direct causal forest returned a treatment effect "
                        "outside the binary probability-difference estimand bounds"
                    )
                pred_y0 = None
                pred_y1 = None
                backend_fit_audit_method = getattr(
                    self.final_causal_forest_backend,
                    "fit_audit",
                    None,
                )
                backend_fit_audit = (
                    None
                    if not callable(backend_fit_audit_method)
                    else json.loads(
                        _canonical_json(backend_fit_audit_method())
                    )
                )
                final_model_input_names = [
                    *direct_blocks.effect_names,
                    *(
                        f"explicit_effect__{index:06d}"
                        for index in explicit_effect_indices
                    ),
                    *direct_blocks.control_names,
                    *(
                        f"explicit_control__{index:06d}"
                        for index in explicit_control_indices
                    ),
                ]
                forest_receipt = {
                    "schema_version": (
                        "role_neutral_direct_strict_causal_forest_receipt_v1"
                    ),
                    "outer_fold": outer_fold,
                    "backend_identity": current_backend_identity,
                    "backend_fit_audit": backend_fit_audit,
                    "reference_manifest_content_sha256": (
                        direct_blocks.reference_manifest_content_sha256
                    ),
                    "effect_train_sha256": _numerical_array_sha256(
                        effect_train
                    ),
                    "effect_heldout_sha256": _numerical_array_sha256(
                        effect_heldout
                    ),
                    "control_train_sha256": _numerical_array_sha256(
                        control_train
                    ),
                    "control_heldout_sha256": _numerical_array_sha256(
                        control_heldout
                    ),
                    "treatment_sha256": _numerical_array_sha256(
                        train[self.config.treatment_column].to_numpy(
                            dtype=float
                        )
                    ),
                    "outcome_sha256": _numerical_array_sha256(
                        train[self.config.outcome_column].to_numpy(dtype=float)
                    ),
                    "tau_sha256": _numerical_array_sha256(pred_ite),
                    "probability_difference_bounds": [-1.0, 1.0],
                    "probability_difference_validation_tolerance": (
                        probability_difference_tolerance
                    ),
                    "probability_difference_bounds_validated": True,
                    "probability_difference_values_clipped": False,
                    "effect_column_count": int(effect_train.shape[1]),
                    "control_column_count": int(control_train.shape[1]),
                    "explicit_effect_column_count": len(
                        explicit_effect_indices
                    ),
                    "explicit_control_column_count": len(
                        explicit_control_indices
                    ),
                    "fit_row_count": len(train_ids),
                    "prediction_row_count": len(heldout_ids),
                    "strict_causal_forest_only": True,
                    "structured_or_nonforest_fallback_used": False,
                    "outer_heldout_labels_used": False,
                    "potential_outcome_columns_emitted": False,
                }
                forest_receipt["content_sha256"] = _content_sha256(
                    forest_receipt
                )
                final_upstream_audit.update(
                    {
                        "model_input_schema": {
                            "estimator": (
                                FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID
                            ),
                            "effect_x_column_count": int(
                                effect_train.shape[1]
                            ),
                            "control_w_column_count": int(
                                control_train.shape[1]
                            ),
                            "column_count": len(final_model_input_names),
                            "schema_sha256": _content_sha256(
                                final_model_input_names
                            ),
                        },
                        "causal_forest_role_routing": {
                            "native_effect_columns": len(
                                direct_blocks.effect_names
                            ),
                            "native_control_columns": len(
                                direct_blocks.control_names
                            ),
                            "explicit_effect_columns": len(
                                explicit_effect_indices
                            ),
                            "explicit_control_columns": len(
                                explicit_control_indices
                            ),
                        },
                        "structured_interaction_head_used": False,
                    }
                )
                final_estimator_audit = {
                    "schema_version": FINAL_ITE_ESTIMATOR_AUDIT_SCHEMA_VERSION,
                    "mode": FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID,
                    "strict_causal_forest_active": True,
                    "strict_causal_forest_required": (
                        self.config.require_final_causal_forest
                    ),
                    "reference_only_role_neutral_runtime": True,
                    "forest_backend_identity": current_backend_identity,
                    "forest_receipt": forest_receipt,
                    "potential_outcome_reconstruction": (
                        "not_emitted_direct_cate_estimand_only"
                    ),
                    "structured_interaction_head_constructed": False,
                    "outer_heldout_labels_used": False,
                    "row_level_numerical_values_persisted": False,
                }
            elif causal_forest_active:
                if package is None or upstream is None:
                    raise RuntimeError(
                        "exact raw runtime was active without an authenticated final package"
                    )
                if self.final_causal_forest_backend is None:
                    raise RuntimeError("exact raw runtime has no final causal-forest backend")
                current_backend_identity = _review_provider_identity(
                    self.final_causal_forest_backend,
                    label="final_causal_forest_backend",
                )
                if current_backend_identity != self.final_causal_forest_backend_identity:
                    raise ValueError("final causal-forest backend identity changed")
                assert self.raw_final_upstream_producer is not None
                if self.coordinate_preserving_nuisance_view_names is not None:
                    if (
                        self.coordinate_preserving_producer_precommit_sha256 is None
                        or self.coordinate_preserving_nuisance_contract_sha256 is None
                    ):
                        raise RuntimeError(
                            "coordinate-preserving nuisance precommitments are missing"
                        )
                    nuisance_derivation = derive_exact_nuisance_from_coordinate_preserved_stage1(
                        package,
                        runtime_producer=self.raw_final_upstream_producer,
                        bow_view_names=self.coordinate_preserving_nuisance_view_names,
                        precommitted_producer_identity_sha256=(
                            self.coordinate_preserving_producer_precommit_sha256
                        ),
                        precommitted_coordinate_contract_sha256=(
                            self.coordinate_preserving_nuisance_contract_sha256
                        ),
                    )
                    nuisance_bridge_mode = "coordinate_preserving_v3"
                else:
                    nuisance_derivation = derive_exact_nuisance_from_runtime_stable_stage1(
                        package,
                        runtime_producer=self.raw_final_upstream_producer,
                    )
                    nuisance_bridge_mode = "stable_family_summary_v2"
                nuisance_derivation.verify_authenticated_content(
                    package,
                    runtime_producer=self.raw_final_upstream_producer,
                )
                explicit_block, explicit_forest_audit = _seal_final_forest_explicit_block(
                    package,
                    encoder=encoder,
                    specs=specs,
                    train_values=explicit_train_x,
                    heldout_values=explicit_heldout_x,
                )
                forest_adapter = StrictOuterHonestFinalCausalForestAdapter(
                    backend=self.final_causal_forest_backend
                )
                forest_tau = forest_adapter.fit_predict(
                    package,
                    outer_train_row_ids=train_ids,
                    treatment=train[self.config.treatment_column].to_numpy(dtype=float),
                    outcome=train[self.config.outcome_column].to_numpy(dtype=float),
                    exact_nuisance=nuisance_derivation.nuisance,
                    explicit_features=explicit_block,
                )
                nuisance_derivation.verify_authenticated_content(
                    package,
                    runtime_producer=self.raw_final_upstream_producer,
                )
                forest_tau.verify_authenticated_content()
                pred_y0, pred_y1, pred_ite, reconstruction_audit = (
                    _reconstruct_forest_potential_outcomes(
                        forest_tau.values,
                        exact_nuisance=nuisance_derivation.nuisance,
                        outcome_type=self.config.outcome_type,
                    )
                )
                forest_audit = dict(forest_adapter.audit_record())
                routing = forest_audit["routing"]
                x_count = int(sum(routing["effect_columns"].values()))
                w_count = int(sum(routing["control_columns"].values()))
                final_model_input_names = [
                    *(f"final_causal_forest__x__{index:03d}" for index in range(1, x_count + 1)),
                    *(f"final_causal_forest__w__{index:03d}" for index in range(1, w_count + 1)),
                ]
                final_upstream_audit.update(
                    {
                        "model_input_schema": {
                            "estimator": FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID,
                            "effect_x_column_count": x_count,
                            "control_w_column_count": w_count,
                            "column_count": len(final_model_input_names),
                            "explicit_extraction_encoded_column_count": (explicit_feature_count),
                            "schema_sha256": _content_sha256(final_model_input_names),
                        },
                        "causal_forest_role_routing": dict(routing),
                        "structured_interaction_head_used": False,
                    }
                )
                final_estimator_audit: Mapping[str, Any] = {
                    "schema_version": FINAL_ITE_ESTIMATOR_AUDIT_SCHEMA_VERSION,
                    "mode": FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID,
                    "strict_causal_forest_active": True,
                    "strict_causal_forest_required": (self.config.require_final_causal_forest),
                    "exact_raw_runtime_identity": self.raw_final_upstream_producer_identity,
                    "final_package_producer_identity": self.final_upstream_producer_identity,
                    "forest_backend_identity": current_backend_identity,
                    "fixed_prior_working_backend_active": bool(
                        type(self.final_causal_forest_backend) is FixedCausalForestHeadBackend
                        and not self.final_causal_forest_backend_was_injected
                    ),
                    "test_backend_injected": self.final_causal_forest_backend_was_injected,
                    "authenticated_nuisance_derivation_mode": nuisance_bridge_mode,
                    "authenticated_stable_nuisance_derivation": dict(
                        nuisance_derivation.audit_record()
                    ),
                    "explicit_feature_role_routing": dict(explicit_forest_audit),
                    "forest_adapter": forest_audit,
                    "potential_outcome_reconstruction": dict(reconstruction_audit),
                    "structured_interaction_head_constructed": False,
                    "outer_heldout_labels_used": False,
                    "row_level_numerical_values_persisted": False,
                }
                package.verify_authenticated_content()
            else:
                head = StructuredInteractionHead(
                    outcome_type=self.config.outcome_type,
                    regularization_grid=head_regularization_grid,
                    inner_folds=self.config.interaction_inner_folds,
                    interact_all_features=self.config.interact_all_features,
                    random_state=self.config.random_state + outer_fold,
                )
                modifier_indices: list[int] | None = None
                if not self.config.interact_all_features:
                    modifier_names = {
                        str(spec["name"])
                        for spec in specs
                        if "effect_modifier" in set(spec.get("roles") or [])
                    }
                    modifier_indices = [
                        index
                        for index, name in enumerate(encoder.feature_spec_names_)
                        if name in modifier_names
                    ]
                    modifier_indices.extend(upstream_modifier_indices)
                    if not modifier_indices:
                        raise ValueError("fusion selected no effect-modifier interaction columns")
                head.fit(
                    train_x,
                    train[self.config.treatment_column].to_numpy(dtype=float),
                    train[self.config.outcome_column].to_numpy(dtype=float),
                    modifier_indices=modifier_indices,
                )
                if self.final_upstream_producer is not None:
                    selected_regularization = float(head.tuning_result_.selected_regularization)
                    if selected_regularization != float(
                        self.config.final_upstream_head_regularization
                    ):
                        raise RuntimeError(
                            "final head changed the precommitted singleton regularization"
                        )
                    final_upstream_audit["head_regularization_policy"] = {
                        **final_upstream_audit["head_regularization_policy"],
                        "selected_regularization": selected_regularization,
                        "selection_equals_precommitted_singleton": True,
                    }
                    assert package is not None
                    package.verify_authenticated_content()
                pred_y0, pred_y1 = head.predict_potential_outcomes(heldout_x)
                pred_ite = pred_y1 - pred_y0
                head_tuning_audit = asdict(head.tuning_result_)
                final_estimator_audit = {
                    "schema_version": FINAL_ITE_ESTIMATOR_AUDIT_SCHEMA_VERSION,
                    "mode": "structured_interaction_head_degraded_fallback",
                    "strict_causal_forest_active": False,
                    "strict_causal_forest_required": False,
                    "reason": "no_exact_raw_final_upstream_runtime_was_supplied",
                    "structured_interaction_head_constructed": True,
                    "outer_heldout_labels_used": False,
                }
            prediction_columns: dict[str, Any] = {
                "_oci_row_id": np.asarray(heldout_ids, dtype=int),
                "outer_fold": int(outer_fold),
                "pred_ite_prob": pred_ite,
            }
            if pred_y0 is not None or pred_y1 is not None:
                if pred_y0 is None or pred_y1 is None:
                    raise RuntimeError(
                        "potential-outcome predictions must be emitted as a pair"
                    )
                prediction_columns.update(
                    {
                        "pred_y0_prob": pred_y0,
                        "pred_y1_prob": pred_y1,
                    }
                )
            prediction = pd.DataFrame(prediction_columns)
            _reject_forbidden_columns(prediction.columns, source="frozen prediction")
            fold_prediction_path = fold_dir / "frozen_predictions.parquet"
            fold_prediction_sha = _write_immutable_parquet(fold_prediction_path, prediction)
            fold_manifest_body = {
                "input_manifest_content_sha256": input_manifest_hash,
                "outer_fold": outer_fold,
                "source_text_temporal_policy": source_text_temporal_policy_audit(),
                "spent_evidence_context_epoch_policy": (
                    _spent_evidence_context_epoch_policy_audit()
                ),
                "split_fingerprint": provenance.split_fingerprint,
                "train_row_fingerprint": row_set_fingerprint(train_ids),
                "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
                "train_row_count": len(train_ids),
                "heldout_row_count": len(heldout_ids),
                "stage1_reference_source": reference_source,
                "legacy_handoff_sha256": (
                    None if legacy is None else legacy.artifact_sha256
                ),
                "tfidf_handoff_sha256": (
                    None if tfidf is None else tfidf.artifact_sha256
                ),
                "tfidf_orphan_ngram_evidence": orphan_audit,
                "query_evidence": query_audit,
                "candidate_pool": candidate_audit,
                "initial_selector_evidence_scope": (
                    "spent_only_with_all_future_review_partitions_sealed"
                    if self.config.post_extraction_review_rounds > 0
                    else "full_outer_train_nonadaptive"
                ),
                "initial_selector_evidence_audit": initial_selector_evidence_audit,
                "initial_discovery_mode": initial_discovery_mode,
                "fusion_prompt_version": fusion_prompt_version,
                "fusion_request_sha256": request_hash,
                "fusion_source_family_coverage": fusion_source_family_coverage,
                "fusion_response_sha256": response_hash,
                "fusion_response_cache_path": str(response_cache_path.resolve()),
                "fusion_response_validated_before_cache_write_or_after_cache_load": True,
                "staged_fusion_audit": {
                    "status": staged_fusion_audit_status,
                    "persisted_with_request_bound_response_cache": bool(
                        hierarchical_discovery_result is None
                    ),
                    "audit": staged_fusion_audit,
                },
                "fusion_response_audit": fusion_response_audit,
                "hierarchical_discovery": (
                    None
                    if hierarchical_discovery_result is None
                    else {
                        "batch_approval_sha256": (hierarchical_batch_result.batch_approval_sha256),
                        "batch_result_sha256": hierarchical_batch_result.result_sha256,
                        "wrapper_approval_sha256": (
                            hierarchical_discovery_result.wrapper_approval_sha256
                        ),
                        "completion_sha256": (
                            hierarchical_discovery_result.completed.completion_sha256
                        ),
                        "compiled_registry_sha256": (
                            hierarchical_discovery_result.compiled_registry.registry_sha256
                        ),
                        "frozen_review_evidence_binding_sha256": (
                            frozen_hierarchical_review.binding_sha256
                        ),
                        "all_fold_discovery_completed_before_this_fold_modeling": True,
                        "first_gate_direct_numerical_contract_kind": (
                            "authenticated_reference_only_direct_numerical_contract"
                            if hierarchical_reference_numerical_contract
                            is not None
                            else "first_gate_materialization_intent"
                        ),
                        "first_gate_direct_numerical_contract_sha256": (
                            hierarchical_reference_numerical_contract.content_sha256
                            if hierarchical_reference_numerical_contract
                            is not None
                            else (
                                hierarchical_first_gate_materialization_intent.content_sha256
                            )
                        ),
                        "first_gate_materialization_intent_sha256": (
                            None
                            if hierarchical_first_gate_materialization_intent
                            is None
                            else hierarchical_first_gate_materialization_intent.content_sha256
                        ),
                        "first_gate_label_free_cache_materialized_before_discovery": (
                            hierarchical_reference_numerical_contract is not None
                        ),
                        "first_gate_cache_materialization_deferred_until_proposal_freeze": (
                            hierarchical_reference_numerical_contract is None
                        ),
                        "first_gate_reference_projection_already_fit": (
                            hierarchical_reference_numerical_contract is not None
                        ),
                        "first_gate_values_or_coordinate_metadata_exposed_to_discovery": False,
                    }
                ),
                "initial_selected_contracts": initial_specs,
                "initial_selected_contract_sha256": [
                    extraction_contract_sha256(spec) for spec in initial_specs
                ],
                "post_extraction_review": post_extraction_review_audit,
                "selected_contracts": specs,
                "selected_contract_sha256": [extraction_contract_sha256(spec) for spec in specs],
                "extraction": extraction_audit,
                "encoder": encoder.state_dict(),
                "final_upstream_model_inputs": final_upstream_audit,
                "final_ite_estimator": dict(final_estimator_audit),
                "final_model_input_schema_sha256": _content_sha256(final_model_input_names),
                "final_model_input_column_count": len(final_model_input_names),
                "head_tuning": head_tuning_audit,
                "observed_label_use": {
                    "spent_only_initial_discovery": bool(
                        self.config.post_extraction_review_rounds > 0
                    ),
                    "spent_review_diagnostics": bool(self.config.post_extraction_review_rounds > 0),
                    "untouched_gate_acceptance": bool(
                        self.config.post_extraction_review_rounds > 0
                    ),
                    "complete_outer_train_final_head_fit_and_tuning": (
                        not causal_forest_active and self.final_upstream_producer is None
                    ),
                    "complete_outer_train_final_head_fit": not causal_forest_active,
                    "complete_outer_train_final_causal_forest_fit": (causal_forest_active),
                    "outer_train_only_causal_forest_tuning": bool(
                        causal_forest_active
                        and self.final_causal_forest_backend_identity is not None
                        and self.final_causal_forest_backend_identity["identity"].get(
                            "tune_model", False
                        )
                    ),
                    "adaptive_final_head_regularization_choice": (
                        not causal_forest_active
                        and self.final_upstream_producer is None
                        and len(self.config.regularization_grid) > 1
                    ),
                    "precommitted_singleton_final_head_regularization": (
                        not causal_forest_active and self.final_upstream_producer is not None
                    ),
                    "outer_heldout": False,
                },
                "outer_heldout_outcomes_used": False,
                "prediction_path": str(fold_prediction_path.resolve()),
                "prediction_sha256": fold_prediction_sha,
                "prediction_columns": list(prediction.columns),
                "oracle_columns_written": False,
            }
            fold_manifest_path = fold_dir / "immutable_fold_manifest.json"
            _write_immutable_json(
                fold_manifest_path,
                fold_manifest_body,
                schema=FOLD_MANIFEST_SCHEMA_VERSION,
            )
            fold_predictions.append(prediction)
            fold_manifests.append(fold_manifest_path)

        combined = (
            pd.concat(fold_predictions, ignore_index=True)
            .sort_values("_oci_row_id")
            .reset_index(drop=True)
        )
        if len(combined) != len(data) or combined["_oci_row_id"].duplicated().any():
            raise RuntimeError("outer-fold predictions do not cover each dataset row exactly once")
        if combined["_oci_row_id"].tolist() != data["_oci_row_id"].tolist():
            raise RuntimeError("outer-fold prediction row identities are incomplete")
        _reject_forbidden_columns(combined.columns, source="combined frozen predictions")
        prediction_path = self.output_dir / "frozen_predictions.parquet"
        prediction_sha = _write_immutable_parquet(prediction_path, combined)
        run_manifest_body = {
            "source_text_temporal_policy": source_text_temporal_policy_audit(),
            "spent_evidence_context_epoch_policy": (_spent_evidence_context_epoch_policy_audit()),
            "input_manifest_path": str(input_manifest_path.resolve()),
            "input_manifest_content_sha256": input_manifest_hash,
            "fold_manifest_paths": [str(path.resolve()) for path in fold_manifests],
            "fold_count": len(fold_manifests),
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": prediction_sha,
            "prediction_row_count": len(combined),
            "prediction_columns": list(combined.columns),
            "outer_test_rows_predicted_once": True,
            "final_ite_estimator": {
                "schema_version": FINAL_ITE_ESTIMATOR_AUDIT_SCHEMA_VERSION,
                "mode": (
                    FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID
                    if (
                        self.raw_final_upstream_producer is not None
                        or self.reference_only_stage1_mode
                    )
                    else "structured_interaction_head_degraded_fallback"
                ),
                "strict_causal_forest_active_for_every_fold": (
                    self.raw_final_upstream_producer is not None
                    or self.reference_only_stage1_mode
                ),
                "strict_causal_forest_required": self.config.require_final_causal_forest,
                "fixed_prior_working_backend_active": bool(
                    (
                        self.raw_final_upstream_producer is not None
                        or self.reference_only_stage1_mode
                    )
                    and type(self.final_causal_forest_backend) is FixedCausalForestHeadBackend
                    and not self.final_causal_forest_backend_was_injected
                ),
                "reference_only_role_neutral_runtime": (
                    self.reference_only_stage1_mode
                ),
            },
            "oracle_columns_written": False,
            "remote_dependencies_injected": True,
        }
        run_manifest_path = self.output_dir / "immutable_run_manifest.json"
        _write_immutable_json(
            run_manifest_path,
            run_manifest_body,
            schema=FROZEN_PREDICTION_SCHEMA_VERSION,
        )
        return AllEvidenceFusionRunResult(
            prediction_path=prediction_path,
            run_manifest_path=run_manifest_path,
            fold_manifest_paths=tuple(fold_manifests),
            prediction_sha256=prediction_sha,
        )


def evaluate_frozen_all_evidence_predictions(
    *,
    prediction_path: Path | str,
    expected_prediction_sha256: str,
    oracle_frame: pd.DataFrame,
    output_dir: Path | str,
    oracle_ite_column: str,
) -> Mapping[str, Any]:
    """Join oracle effects only after authenticating a frozen prediction file."""

    prediction_path = Path(prediction_path).resolve()
    expected_sha = str(expected_prediction_sha256).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise ValueError("expected_prediction_sha256 must be a lowercase SHA-256 digest")
    # Authenticate the exact bytes that will be parsed. Reading the path once
    # into an immutable buffer closes the hash/open race from a concurrent path
    # replacement during posthoc oracle evaluation.
    try:
        with prediction_path.open("rb") as handle:
            frozen_bytes = handle.read()
    except OSError as exc:
        raise ValueError("frozen prediction artifact could not be read") from exc
    frozen_sha = hashlib.sha256(frozen_bytes).hexdigest()
    if frozen_sha != expected_sha:
        raise ValueError("frozen prediction SHA-256 differs from the pre-oracle manifest")
    predictions = pd.read_parquet(io.BytesIO(frozen_bytes))
    _reject_forbidden_columns(predictions.columns, source="frozen prediction")
    required = {"_oci_row_id", "outer_fold", "pred_ite_prob"}
    if not required <= set(predictions.columns):
        raise ValueError(
            f"frozen predictions are missing columns: {sorted(required - set(predictions))}"
        )
    if predictions["_oci_row_id"].duplicated().any():
        raise ValueError("frozen predictions contain duplicate row IDs")
    if oracle_ite_column not in oracle_frame.columns or "_oci_row_id" not in oracle_frame.columns:
        raise ValueError("oracle frame lacks row ID or requested oracle ITE column")
    oracle = oracle_frame[["_oci_row_id", oracle_ite_column]].copy()
    if oracle["_oci_row_id"].duplicated().any():
        raise ValueError("oracle frame contains duplicate row IDs")
    evaluated = predictions.merge(oracle, on="_oci_row_id", how="left", validate="one_to_one")
    if evaluated[oracle_ite_column].isna().any():
        raise ValueError("oracle ITE is missing for one or more frozen predictions")

    def metrics(frame: pd.DataFrame) -> dict[str, Any]:
        truth = frame[oracle_ite_column].to_numpy(dtype=float)
        estimate = frame["pred_ite_prob"].to_numpy(dtype=float)
        error = estimate - truth
        correlation = None
        if len(frame) >= 2 and np.std(truth) > 0 and np.std(estimate) > 0:
            correlation = float(np.corrcoef(truth, estimate)[0, 1])
        return {
            "n": len(frame),
            "pearson_correlation": correlation,
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(math.sqrt(np.mean(np.square(error)))),
            "mean_error": float(np.mean(error)),
        }

    body = {
        "frozen_prediction_path": str(prediction_path),
        "frozen_prediction_sha256": frozen_sha,
        "oracle_ite_column": oracle_ite_column,
        "overall": metrics(evaluated),
        "per_fold": [
            {"outer_fold": int(fold), **metrics(frame)}
            for fold, frame in evaluated.groupby("outer_fold", sort=True)
        ],
        "oracle_join_performed_posthoc": True,
    }
    output = Path(output_dir)
    evaluated_path = output / "posthoc_predictions_with_oracle.parquet"
    _write_immutable_parquet(evaluated_path, evaluated)
    _write_immutable_json(
        output / "posthoc_oracle_metrics.json",
        body,
        schema=POSTHOC_EVALUATION_SCHEMA_VERSION,
    )
    return body
