"""Authenticated hierarchy dependency contract for arbitrary-cohort Stage 1.

The Stage 1 wrapper must not restate or loosely approximate the downstream
feature-discovery protocol.  This module imports the live hierarchy interfaces,
checks the exact production semantic versions, authenticates their implementation
bundles and standalone execution modules, and returns one content-addressed
identity for the immutable Stage 1 request.

This identity is intentionally stricter than an import-success check.  It rejects
the historical all-architecture raw-evidence dump, exact-coverage arrays, and any
hierarchy implementation predating architecture-local dossiers plus exhaustive,
ID-addressed raw-evidence pages and recursive folds.  It does not attest that Stage 1 has emitted
genuine native family proofs; the separate non-bypassable producer/e2e gates remain
closed until those artifacts exist.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

PRODUCTION_STAGE1_HIERARCHY_CONTRACT_SCHEMA_VERSION = (
    "production_stage1_hierarchical_discovery_contract_v5"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MODULE_ROOT = Path(__file__).resolve().parent

_REQUIRED_FAMILY_ORDER = (
    "bow_nuisance",
    "bow_r_loss",
    "htr_neural",
    "matched_pair_uplift",
    "embedding_whole_cohort",
    "embedding_clustered",
    "tfidf_semantic_retrieval_contrasts",
    "tfidf_topics",
    "tfidf_orphan_ngrams",
    "neural_query_moments",
)

_MODULE_NAMES = {
    "interfaces": "oci.inference.all_evidence_discovery_interfaces",
    "response_contract": "oci.inference.hierarchical_discovery_response_contract",
    "orchestrator": "oci.inference.hierarchical_all_architecture_discovery",
    "job_cache": "oci.inference.hierarchical_discovery_job_cache",
    "json_runner": "oci.inference.openai_compatible_json_discovery_job_runner",
    "approved_agent": "oci.inference.approved_hierarchical_discovery_agent",
    "approved_batch": "oci.inference.approved_hierarchical_discovery_batch",
    "adaptive_hierarchy": "oci.inference.adaptive_hierarchical_stage1_reconsideration",
    "frozen_review": "oci.inference.frozen_hierarchical_review_evidence",
    "review_provider": "oci.inference.review_spent_evidence_provider",
    "cumulative_spent": "oci.inference.stage1_cumulative_spent_evidence",
    "catalog": "oci.inference.lossless_stage1_evidence_catalog",
    "production_handoff": "oci.inference.production_stage1_hierarchy_handoff",
    "fusion_runner": "oci.inference.all_evidence_fusion_runner",
}

# These are compatibility pins, not duplicate interface definitions.  The actual
# schemas, normalizers, renderers, and validators are imported from the modules
# above and their bytes are authenticated below.
_REQUIRED_VERSION_ROWS = (
    ("interfaces", "DISCOVERY_INTERFACE_SCHEMA_VERSION", "all_evidence_discovery_interfaces_v10"),
    (
        "interfaces",
        "DISCOVERY_WIRE_NORMALIZATION_VERSION",
        "atomic_occurrence_compiler_normalization_v3",
    ),
    ("interfaces", "INTERPRET_JOB_VERSION", "interpret_complementary_evidence_chunk_v5"),
    ("interfaces", "CONSOLIDATE_JOB_VERSION", "lossless_candidate_consolidation_v4"),
    ("interfaces", "COVERAGE_CRITIC_JOB_VERSION", "complete_evidence_coverage_critic_v4"),
    ("interfaces", "REJECTION_CRITIC_JOB_VERSION", "complete_rejection_critic_v4"),
    (
        "interfaces",
        "CROSS_ARCHITECTURE_PLANNER_JOB_VERSION",
        "cross_architecture_lookback_planner_v4",
    ),
    (
        "interfaces",
        "CROSS_ARCHITECTURE_INTEGRATION_JOB_VERSION",
        "cross_architecture_integration_v4",
    ),
    ("interfaces", "ARCHITECTURE_DOSSIER_VERSION", "complete_architecture_dossier_v2"),
    (
        "interfaces",
        "EXTRACTION_DEFINITION_JOB_VERSION",
        "grounded_extraction_definition_v5",
    ),
    (
        "response_contract",
        "HIERARCHICAL_DISCOVERY_RESPONSE_CONTRACT_VERSION",
        "hierarchical_discovery_dynamic_response_contract_v9",
    ),
    (
        "response_contract",
        "HIERARCHICAL_DISCOVERY_EXACT_COVERAGE_REPRESENTATION",
        "closed_object_keyed_by_authenticated_identifier_v1",
    ),
    (
        "orchestrator",
        "HIERARCHICAL_DISCOVERY_ORCHESTRATOR_VERSION",
        "hierarchical_all_architecture_discovery_orchestrator_v12",
    ),
    (
        "orchestrator",
        "HIERARCHICAL_DISCOVERY_PRECOMMIT_VERSION",
        "hierarchical_discovery_precommit_v11",
    ),
    ("orchestrator", "DISCOVERY_JSON_JOB_VERSION", "hierarchical_discovery_json_job_v7"),
    (
        "orchestrator",
        "DISCOVERY_EXECUTION_LEDGER_VERSION",
        "hierarchical_discovery_execution_ledger_v6",
    ),
    (
        "orchestrator",
        "COMPLETED_HIERARCHICAL_DISCOVERY_VERSION",
        "completed_hierarchical_discovery_v7",
    ),
    (
        "orchestrator",
        "DISCOVERY_RESPONSE_REPAIR_POLICY_VERSION",
        "authenticated_bounded_hierarchy_response_repair_v7",
    ),
    (
        "orchestrator",
        "DISCOVERY_RESPONSE_ATTEMPT_TRACE_VERSION",
        "authenticated_hierarchy_response_attempt_trace_v5",
    ),
    (
        "orchestrator",
        "HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_VERSION",
        "hierarchical_discovery_implementation_bundle_v5",
    ),
    (
        "job_cache",
        "HIERARCHICAL_DISCOVERY_JOB_CACHE_VERSION",
        "authenticated_hierarchical_discovery_job_cache_v3",
    ),
    (
        "job_cache",
        "HIERARCHICAL_DISCOVERY_JOB_CACHE_IDENTITY_VERSION",
        "authenticated_hierarchical_discovery_job_cache_identity_v3",
    ),
    (
        "job_cache",
        "HIERARCHICAL_DISCOVERY_JOB_CACHE_LOOKUP_VERSION",
        "authenticated_hierarchical_discovery_job_cache_lookup_v1",
    ),
    (
        "job_cache",
        "HIERARCHICAL_DISCOVERY_JOB_CACHE_ENTRY_VERSION",
        "authenticated_hierarchical_discovery_job_cache_entry_v3",
    ),
    (
        "job_cache",
        "HIERARCHICAL_DISCOVERY_JOB_CACHE_HIT_VERSION",
        "authenticated_hierarchical_discovery_job_cache_hit_v3",
    ),
    (
        "job_cache",
        "HIERARCHICAL_DISCOVERY_CACHE_RESPONSE_TRACE_VERSION",
        "authenticated_cache_response_attempt_trace_v2",
    ),
    (
        "json_runner",
        "OPENAI_JSON_DISCOVERY_RUNNER_VERSION",
        "openai_json_discovery_job_runner_v13",
    ),
    (
        "approved_agent",
        "APPROVED_HIERARCHICAL_DISCOVERY_AGENT_VERSION",
        "approved_hierarchical_discovery_agent_v9",
    ),
    (
        "approved_agent",
        "APPROVED_HIERARCHICAL_DISCOVERY_PRECOMMIT_VERSION",
        "approved_hierarchical_discovery_precommit_v9",
    ),
    (
        "approved_agent",
        "AUTHENTICATED_RUNNER_EXECUTION_TRACE_VERSION",
        "authenticated_json_discovery_runner_and_cache_execution_trace_v8",
    ),
    (
        "approved_agent",
        "APPROVED_HIERARCHICAL_DISCOVERY_RESULT_VERSION",
        "approved_hierarchical_discovery_result_v7",
    ),
    (
        "approved_batch",
        "APPROVED_HIERARCHICAL_DISCOVERY_BATCH_COORDINATOR_VERSION",
        "approved_hierarchical_discovery_batch_coordinator_v4",
    ),
    (
        "approved_batch",
        "APPROVED_HIERARCHICAL_DISCOVERY_BATCH_PRECOMMIT_VERSION",
        "approved_hierarchical_discovery_batch_precommit_v4",
    ),
    (
        "approved_batch",
        "APPROVED_HIERARCHICAL_DISCOVERY_BATCH_RESULT_VERSION",
        "approved_hierarchical_discovery_batch_result_v4",
    ),
    (
        "adaptive_hierarchy",
        "ADAPTIVE_HIERARCHY_VERSION",
        "adaptive_hierarchical_stage1_reconsideration_v7",
    ),
    (
        "adaptive_hierarchy",
        "ADAPTIVE_PLANNER_INTERFACE_VERSION",
        "adaptive_stage1_lookback_planner_v4",
    ),
    (
        "adaptive_hierarchy",
        "ADAPTIVE_PROPOSER_INTERFACE_VERSION",
        "adaptive_registry_revision_proposer_v4",
    ),
    (
        "adaptive_hierarchy",
        "ADAPTIVE_AUTHENTICATED_EXECUTION_VERSION",
        "authenticated_adaptive_hierarchical_stage1_execution_v7",
    ),
    (
        "adaptive_hierarchy",
        "ADAPTIVE_PROMPT_CONTRACT_VERSION",
        "adaptive_hierarchical_stage1_prompt_contract_v7",
    ),
    (
        "adaptive_hierarchy",
        "ADAPTIVE_IMPLEMENTATION_BUNDLE_VERSION",
        "adaptive_hierarchical_implementation_bundle_v8",
    ),
    (
        "frozen_review",
        "FROZEN_HIERARCHICAL_REVIEW_EVIDENCE_SCHEMA_VERSION",
        "frozen_hierarchical_review_evidence_v2",
    ),
    (
        "frozen_review",
        "FROZEN_HIERARCHICAL_REVIEW_EVIDENCE_POLICY_VERSION",
        "round_1_accepted_routed_feature_support_only_v3",
    ),
    (
        "review_provider",
        "REVIEW_SPENT_EVIDENCE_CACHE_VERSION",
        "context_fit_review_spent_evidence_cache_v4",
    ),
    (
        "cumulative_spent",
        "CUMULATIVE_SPENT_REQUEST_SCHEMA",
        "production_stage1_hierarchy_spent_family_request_v1",
    ),
    (
        "cumulative_spent",
        "CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA",
        "cumulative_spent_stage1_family_fit_audit_v1",
    ),
    (
        "cumulative_spent",
        "CUMULATIVE_SPENT_FAMILY_ARTIFACT_SCHEMA",
        "cumulative_spent_stage1_family_artifact_v1",
    ),
    (
        "cumulative_spent",
        "CUMULATIVE_SPENT_EVIDENCE_BUNDLE_SCHEMA",
        "cumulative_spent_stage1_evidence_bundle_v1",
    ),
    ("catalog", "ARCHITECTURE_CHUNK_PLAN_SCHEMA_VERSION", "complete_architecture_chunk_plan_v5"),
    (
        "production_handoff",
        "STAGE1_HIERARCHY_HANDOFF_SCHEMA",
        "authenticated_production_stage1_hierarchy_handoff_v5",
    ),
    (
        "production_handoff",
        "INTERNAL_HIERARCHY_AUTHORIZATION_SCHEMA",
        "production_internal_hierarchy_execution_authorization_v5",
    ),
    (
        "production_handoff",
        "INTERNAL_HIERARCHY_PREPARATION_BINDING_SCHEMA",
        "production_internal_hierarchy_preparation_binding_v2",
    ),
    ("fusion_runner", "RUNNER_SCHEMA_VERSION", "all_evidence_fusion_outer_runner_v20"),
    (
        "fusion_runner",
        "PRODUCTION_HIERARCHY_RUNTIME_BINDING_SCHEMA",
        "production_hierarchy_same_process_runner_binding_v1",
    ),
    (
        "fusion_runner",
        "HIERARCHICAL_DISCOVERY_PREPARATION_INPUT_SCHEMA_VERSION",
        "hierarchical_all_evidence_runner_preparation_input_v2",
    ),
    (
        "fusion_runner",
        "HIERARCHICAL_DISCOVERY_BATCH_PACKET_SCHEMA_VERSION",
        "hierarchical_all_evidence_runner_batch_packet_v1",
    ),
)

_HIERARCHY_BUNDLE_FILES = frozenset(
    {
        "hierarchical_all_architecture_discovery.py",
        "all_evidence_discovery_interfaces.py",
        "hierarchical_discovery_response_contract.py",
        "lossless_stage1_evidence_catalog.py",
    }
)
_ADAPTIVE_BUNDLE_FILES = frozenset(
    {
        "adaptive_hierarchical_stage1_reconsideration.py",
        "all_evidence_discovery_interfaces.py",
        "hierarchical_all_architecture_discovery.py",
        "hierarchical_discovery_response_contract.py",
        "all_evidence_fusion.py",
        "all_evidence_post_extraction_review.py",
        "lossless_stage1_evidence_catalog.py",
        "stage1_architecture_explanations.py",
    }
)

def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_modules() -> dict[str, ModuleType]:
    modules: dict[str, ModuleType] = {}
    for role, module_name in _MODULE_NAMES.items():
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                f"required production hierarchy dependency is unavailable: {module_name}"
            ) from exc
        module_path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
        if module_path.parent != _MODULE_ROOT or module_path.suffix != ".py":
            raise RuntimeError(
                f"production hierarchy dependency resolved outside the local source tree: "
                f"{module_name}"
            )
        modules[role] = module
    return modules


def _observed_versions(modules: Mapping[str, ModuleType]) -> dict[str, dict[str, str]]:
    versions: dict[str, dict[str, str]] = {}
    for role, attribute, expected in _REQUIRED_VERSION_ROWS:
        value = getattr(modules[role], attribute, None)
        if value != expected:
            raise RuntimeError(
                f"production hierarchy dependency {role}.{attribute} is {value!r}; "
                f"required {expected!r}"
            )
        versions.setdefault(role, {})[attribute] = str(value)
    return {role: dict(sorted(values.items())) for role, values in sorted(versions.items())}


def _authenticate_bundle(
    value: Any,
    *,
    label: str,
    hash_key: str,
    expected_schema: str,
    expected_files: frozenset[str],
    expected_keys: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError(f"{label} is not a closed implementation bundle")
    result = copy.deepcopy(dict(value))
    declared = result.pop(hash_key, None)
    if _SHA256.fullmatch(str(declared or "")) is None or _sha(result) != declared:
        raise RuntimeError(f"{label} content hash is invalid")
    if result.get("schema_version") != expected_schema:
        raise RuntimeError(f"{label} schema version is not the pinned production version")
    files = result.get("files")
    if not isinstance(files, Mapping) or set(files) != expected_files:
        raise RuntimeError(f"{label} dependency file set is incomplete")
    for filename, digest in files.items():
        path = (_MODULE_ROOT / str(filename)).resolve(strict=True)
        if path.parent != _MODULE_ROOT or not path.is_file():
            raise RuntimeError(f"{label} dependency path escapes the local source tree")
        if _SHA256.fullmatch(str(digest or "")) is None or _sha_file(path) != digest:
            raise RuntimeError(f"{label} dependency bytes changed: {filename}")
    return copy.deepcopy(dict(value))


def current_production_stage1_hierarchy_contract_identity() -> dict[str, Any]:
    """Return the current exact imported hierarchy contract and byte identity."""

    modules = _load_modules()
    versions = _observed_versions(modules)
    interfaces = modules["interfaces"]
    if tuple(getattr(interfaces, "ACTIVE_STAGE1_CONCEPT_FAMILIES", ())) != (_REQUIRED_FAMILY_ORDER):
        raise RuntimeError("production hierarchy does not cover the exact ten-family order")

    orchestrator = modules["orchestrator"]
    hierarchy_bundle = _authenticate_bundle(
        orchestrator.hierarchical_discovery_implementation_bundle(),
        label="base hierarchy implementation bundle",
        hash_key="implementation_bundle_sha256",
        expected_schema="hierarchical_discovery_implementation_bundle_v5",
        expected_files=_HIERARCHY_BUNDLE_FILES,
        expected_keys=frozenset(
            {
                "schema_version",
                "files",
                "discovery_interface_schema_version",
                "wire_normalization_version",
                "response_contract_version",
                "local_json_schema_validator",
                "exact_coverage_representation",
                "job_interface_versions",
                "implementation_bundle_sha256",
            }
        ),
    )
    if (
        hierarchy_bundle.get("discovery_interface_schema_version")
        != versions["interfaces"]["DISCOVERY_INTERFACE_SCHEMA_VERSION"]
        or hierarchy_bundle.get("wire_normalization_version")
        != versions["interfaces"]["DISCOVERY_WIRE_NORMALIZATION_VERSION"]
        or hierarchy_bundle.get("response_contract_version")
        != versions["response_contract"]["HIERARCHICAL_DISCOVERY_RESPONSE_CONTRACT_VERSION"]
        or not isinstance(hierarchy_bundle.get("local_json_schema_validator"), Mapping)
        or hierarchy_bundle.get("exact_coverage_representation")
        != versions["response_contract"]["HIERARCHICAL_DISCOVERY_EXACT_COVERAGE_REPRESENTATION"]
    ):
        raise RuntimeError("base hierarchy implementation bundle changed its interface bindings")

    adaptive = modules["adaptive_hierarchy"]
    adaptive_bundle = _authenticate_bundle(
        adaptive.adaptive_hierarchical_implementation_bundle(),
        label="adaptive hierarchy implementation bundle",
        hash_key="implementation_bundle_sha256",
        expected_schema="adaptive_hierarchical_implementation_bundle_v8",
        expected_files=_ADAPTIVE_BUNDLE_FILES,
        expected_keys=frozenset(
            {
                "schema_version",
                "files",
                "local_json_schema_validator",
                "implementation_bundle_sha256",
            }
        ),
    )
    if not isinstance(adaptive_bundle.get("local_json_schema_validator"), Mapping):
        raise RuntimeError("adaptive hierarchy bundle lacks its local schema validator identity")

    hierarchy_config = {
        "schema_version": (
            "production_hierarchical_discovery_configuration_contract_v1"
        ),
        "required_constructor_fields": sorted(
            orchestrator.HierarchicalDiscoveryConfig.__dataclass_fields__
        ),
        "production_values_supplied_by_authenticated_scientific_protocol": True,
        "component_defaults_are_production_scientific_values": False,
        "wire_budget_is_required_scientific_configuration": True,
        "configured_capacities_may_page_or_fail_closed_but_never_truncate": True,
    }

    module_files = {
        role: {
            "module": module.__name__,
            "filename": Path(str(module.__file__)).name,
            "sha256": _sha_file(Path(str(module.__file__)).resolve(strict=True)),
        }
        for role, module in sorted(modules.items())
    }
    module_files["stage1_hierarchy_contract"] = {
        "module": __name__,
        "filename": Path(__file__).name,
        "sha256": _sha_file(Path(__file__).resolve(strict=True)),
    }
    body = {
        "schema_version": PRODUCTION_STAGE1_HIERARCHY_CONTRACT_SCHEMA_VERSION,
        "required_family_order": list(_REQUIRED_FAMILY_ORDER),
        "semantic_versions": versions,
        "base_hierarchy_implementation_bundle": hierarchy_bundle,
        "adaptive_hierarchy_implementation_bundle": adaptive_bundle,
        "standalone_module_files": module_files,
        "hierarchical_discovery_config": hierarchy_config,
        "feature_discovery_workflow": {
            "levels": [
                "lossless_all_ten_catalog_without_global_top_k",
                "complete_family_pure_chunks_interpreted_one_architecture_at_a_time",
                "within_architecture_consolidation_and_complete_coverage_critique",
                "exhaustive_candidate_pair_pages_then_complete_link_groups",
                "one_raw_support_item_per_page_then_recursive_folds",
            ],
            "architecture_local_before_cross_architecture_integration": True,
            "all_ten_architectures_must_be_interpreted": True,
            "global_top_k_allowed": False,
            "raw_all_architecture_evidence_dump_allowed": False,
            "family_mixed_interpretation_chunks_allowed": False,
            "dossiers_replace_raw_catalogs_at_cross_architecture_planning": True,
            "atomic_occurrence_assignment_required": True,
            "candidate_relationships_compiler_derived": True,
            "fixed_slot_consolidation_required": True,
        },
        "lossless_raw_evidence_hierarchy": {
            "resolver": "exact_authenticated_catalog_evidence_id_only",
            "model_written_or_unknown_ids_allowed": False,
            "one_raw_evidence_item_per_page": True,
            "maximum_fold_inputs_configured_by_hierarchy_wire_budget": True,
            "maximum_fresh_inputs_after_first_fold_derived_from_configured_fan_in": (
                True
            ),
            "integration_rejection_and_extraction_all_use_pages_and_folds": True,
            "every_input_receives_an_explicit_disposition": True,
            "semantic_sampling_or_truncation": False,
            "legacy_configured_lookback_and_feature_caps_are_not_semantic_limits": True,
        },
        "wire_contract": {
            "exact_coverage_representation": versions["response_contract"][
                "HIERARCHICAL_DISCOVERY_EXACT_COVERAGE_REPRESENTATION"
            ],
            "wire_normalization_version": versions["interfaces"][
                "DISCOVERY_WIRE_NORMALIZATION_VERSION"
            ],
            "legacy_exact_length_enum_array_allowed": False,
            "duplicate_identifier_can_satisfy_exact_coverage": False,
            "raw_wire_and_normalized_projection_hashes_required": True,
        },
        "adaptive_review_contract": {
            "all_ten_compact_dossiers_required": True,
            "accepted_support_only_catalog_allowed": False,
            "raw_all_architecture_evidence_dump_allowed": False,
            "exact_cumulative_spent_catalog_required": True,
            "lossless_id_addressed_pages_and_recursive_folds_required": True,
            "every_target_evidence_candidate_and_diagnostic_page_required": True,
            "every_proposal_judged_before_capacity_disposition": True,
            "extraction_reviews_every_support_item_before_terminal_fold": True,
            "semantic_sampling_or_truncation_allowed": False,
            "spent_evidence_projection_uses_exhaustive_vocabulary_and_no_term_cap": True,
        },
        "tfidf_nested_training_only_calibration": {
            "families": list(_REQUIRED_FAMILY_ORDER[6:9]),
            "all_three_paths_require_truthful_training_scope_policy_records": True,
            "label_based_nested_selection_families": list(_REQUIRED_FAMILY_ORDER[7:9]),
            "deterministic_exhaustive_no_selection_families": [_REQUIRED_FAMILY_ORDER[6]],
            "selection_inside_each_applicable_registered_training_scope": True,
            "semantic_retrieval_replay_partitions_are_nonselecting_canaries": True,
            "semantic_retrieval_projection_vocabulary_or_output_cap_allowed": False,
            "semantic_retrieval_nested_calibration_labels_accessed": False,
            "selection_frozen_before_registered_heldout_text_transform": True,
            "registered_heldout_treatment_or_outcome_available": False,
            "hierarchy_partitions_reused_as_calibration_folds": False,
            "interaction_crossfit_reused_as_calibration_folds": False,
        },
        "digest_execution_policy": {
            "low_level_exact_digest_binding_retained": True,
            "digest_carried_by_one_authorized_cohort_invocation": True,
            "end_user_digest_entry_required": False,
            "manual_digest_approval_required": False,
            "one_shot_provider_bound_orchestrator_required": True,
            "registered_json_parsed_from_authenticated_byte_snapshot": True,
            "strict_duplicate_json_keys_rejected": True,
            "bundle_root_and_all_registered_paths_descriptor_anchored": True,
            "intermediate_and_final_symlinks_followed": False,
            "loader_to_handoff_manifest_reopen_allowed": False,
            "preparation_input_wrapper_schema": (
                "hierarchical_all_evidence_runner_preparation_input_v2"
            ),
            "preparation_batch_wrapper_schema": (
                "hierarchical_all_evidence_runner_batch_packet_v1"
            ),
            "expected_digests_from_in_memory_prepared_batch_required": True,
            "caller_mapping_or_bare_expected_digests_accepted": False,
            "exact_concrete_prepared_batch_capability_required": True,
            "prepared_batch_capability_process_local_and_one_shot": True,
            "execution_authorization_exact_typed_and_one_shot": True,
            "exact_same_process_runner_and_runtime_objects_required": True,
            "caller_replay_registrations_accepted": False,
            "runtime_provider_identities_reauthenticated_before_authorization_and_execution": (
                True
            ),
            "runtime_scientific_input_file_hashes_reauthenticated": True,
            "exact_coordinator_and_precommit_objects_required": True,
            "canonical_unbound_coordinator_execute_required": True,
            "exact_authenticated_batch_result_type_required": True,
            "cross_process_path_replay_belongs_only_to_standalone_cli": True,
        },
        "remote_runtime_identity_policy": {
            "one_canonical_explicit_http_or_https_endpoint_required": True,
            "explicit_localhost_endpoint_allowed": True,
            "endpoint_credentials_query_or_fragment_allowed": False,
            "endpoint_pool_fallback_or_substitution_allowed": False,
            "one_exact_explicit_model_name_required": True,
            "model_autodiscovery_pool_fallback_or_substitution_allowed": False,
            "response_model_must_equal_requested_model": True,
            "response_finish_reason_must_equal_stop": True,
            "response_metadata_checked_before_content_semantics_and_cache": True,
            "response_policy_applies_to_initial_invalid_and_repair_responses": True,
            "guarded_client_paths": [
                "hierarchical_discovery",
                "proposal_and_post_extraction_review",
                "explicit_feature_extraction",
            ],
            "deployment_metadata_required_for_execution": False,
            "deployment_metadata_if_present_is_authoritative": False,
            "compiled_deployment_digest_pin_required": False,
            "caller_supplied_digest_or_approval_can_authorize": False,
            "immediate_canary_is_separate_operational_gate": True,
        },
        "native_proof_policy": {
            "component_emitted_exact_inner_and_cumulative_spent_proofs_required": True,
            "typed_cumulative_spent_all_ten_producer_boundary_required": True,
            "cumulative_spent_producer_receives_spent_text_treatment_outcome": True,
            "cumulative_spent_producer_receives_sealed_row_ids_only": True,
            "cumulative_spent_producer_receives_sealed_text_or_labels": False,
            "wrapper_assembled_or_schema_only_proof_sufficient": False,
            "production_readiness_without_genuine_binder_validation_and_e2e_allowed": False,
        },
    }
    return {**body, "content_sha256": _sha(body)}


def validate_production_stage1_hierarchy_contract_identity(
    value: Any,
) -> dict[str, Any]:
    """Recompute and require the exact current imported hierarchy identity."""

    if not isinstance(value, Mapping):
        raise ValueError("Stage 1 request has no hierarchical discovery contract identity")
    supplied = copy.deepcopy(dict(value))
    body = dict(supplied)
    declared = body.pop("content_sha256", None)
    if _SHA256.fullmatch(str(declared or "")) is None or _sha(body) != declared:
        raise ValueError("Stage 1 hierarchical discovery contract hash is invalid")
    current = current_production_stage1_hierarchy_contract_identity()
    if supplied != current:
        raise ValueError(
            "Stage 1 hierarchical discovery contract differs from the current imported "
            "hierarchy implementation"
        )
    return supplied


def production_stage1_hierarchy_architecture_bindings(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the closed Stage 1 architecture contract for this hierarchy."""

    validated = validate_production_stage1_hierarchy_contract_identity(identity)
    return {
        "required_families": list(_REQUIRED_FAMILY_ORDER),
        "all_families_required_nonzero_per_scope": True,
        "legacy_exact_inner_evidence_refit": True,
        "legacy_full_outer_evidence_reused_for_inner": False,
        "tfidf_uses_canonical_registry": True,
        "neural_query_uses_canonical_registry": True,
        "semantic_retrieval_is_separate_from_embedding_structural_views": True,
        "legacy_concept_projection": "lossless_no_prompt_compactor_v1",
        "raw_evidence_sidecars_authenticated_and_prompt_hidden": True,
        "matched_pair_subproducer_proofs_required": ["bow", "htr"],
        "tfidf_resume_policy": "sealed_complete_component_only_no_partial_checkpoint_reuse_v1",
        "hierarchy_accumulated_spent_scope_index_required": True,
        "hierarchy_partition_authority": (
            "canonical_stage1_inner_heldout_partitions_in_registry_order"
        ),
        "hierarchy_component_emitted_catalogs_and_proofs_required": True,
        "hierarchy_typed_cumulative_spent_all_ten_producer_boundary_required": True,
        "hierarchy_cumulative_spent_sealed_rows_are_id_only": True,
        "hierarchy_independent_runtime_stage1_refit_allowed": False,
        "hierarchical_discovery_contract_identity_sha256": validated["content_sha256"],
        "production_discovery_mode": "hierarchical",
        "all_ten_architectures_interpreted_separately_before_integration": True,
        "within_architecture_consolidation_and_coverage_required": True,
        "cross_architecture_integration_uses_compact_dossiers": True,
        "lossless_exact_id_raw_evidence_pages_and_recursive_folds_required": True,
        "raw_all_architecture_prompt_allowed": False,
        "global_top_k_before_discovery_allowed": False,
        "family_mixed_interpretation_chunks_allowed": False,
        "exact_coverage_keyed_object_required": True,
        "legacy_exact_coverage_array_allowed": False,
        "adaptive_review_retains_all_ten_architecture_dossiers": True,
        "tfidf_truthful_training_scope_policy_required_for_all_three_paths": True,
        "tfidf_nested_label_based_selection_required_for_topic_and_orphan": True,
        "semantic_retrieval_deterministic_exhaustive_no_selection_required": True,
        "end_user_digest_entry_required": False,
        "manual_digest_approval_required": False,
        "registered_json_parsed_from_authenticated_byte_snapshot": True,
        "strict_duplicate_json_keys_rejected": True,
        "bundle_paths_descriptor_anchored_without_symlink_following": True,
        "loader_to_handoff_manifest_reopen_allowed": False,
        "preparation_wrapper_schema_versions_pinned": True,
        "expected_digests_from_in_memory_prepared_batch_required": True,
        "exact_one_shot_prepared_batch_capability_required": True,
        "exact_one_shot_execution_authorization_required": True,
        "exact_one_shot_same_process_runner_binding_required": True,
        "production_caller_replay_registrations_allowed": False,
        "runtime_provider_identities_and_scientific_file_hashes_reauthenticated": True,
        "exact_coordinator_precommit_and_result_types_required": True,
    }


def validate_production_stage1_hierarchy_request_bindings(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one request's hierarchy identity, policy, and no-prompt security."""

    if not isinstance(request, Mapping):
        raise TypeError("Stage 1 build request must be a mapping")
    identity = validate_production_stage1_hierarchy_contract_identity(
        request.get("hierarchical_discovery_contract_identity")
    )
    architecture = request.get("architecture_contract")
    if not isinstance(architecture, Mapping):
        raise ValueError("Stage 1 request has no architecture contract")
    expected = production_stage1_hierarchy_architecture_bindings(identity)
    if dict(architecture) != expected:
        missing = sorted(set(expected) - set(architecture))
        extra = sorted(set(architecture) - set(expected))
        changed = sorted(
            key for key in set(expected) & set(architecture) if architecture[key] != expected[key]
        )
        raise ValueError(
            "Stage 1 request weakens or changes its hierarchical discovery bindings: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    security = request.get("security")
    if not isinstance(security, Mapping):
        raise ValueError("Stage 1 request has no security contract")
    if (
        security.get("remote_clients_constructed") is not False
        or security.get("remote_calls_allowed") is not False
        or security.get("manual_digest_approval_required") is not False
        or security.get("raw_evidence_sidecars_visible_to_prompts") is not False
    ):
        raise ValueError("Stage 1 request weakens its local-build or no-prompt security policy")
    return identity


__all__ = [
    "PRODUCTION_STAGE1_HIERARCHY_CONTRACT_SCHEMA_VERSION",
    "current_production_stage1_hierarchy_contract_identity",
    "production_stage1_hierarchy_architecture_bindings",
    "validate_production_stage1_hierarchy_contract_identity",
    "validate_production_stage1_hierarchy_request_bindings",
]
