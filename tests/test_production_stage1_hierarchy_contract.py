from __future__ import annotations

import copy
import hashlib
import json

import pytest

from oci.inference import all_evidence_discovery_interfaces as interfaces
from oci.inference import hierarchical_all_architecture_discovery as hierarchy
from oci.inference.production_stage1_hierarchy_contract import (
    current_production_stage1_hierarchy_contract_identity,
    production_stage1_hierarchy_architecture_bindings,
    validate_production_stage1_hierarchy_contract_identity,
    validate_production_stage1_hierarchy_request_bindings,
)


def _sha(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _minimal_request(identity: dict) -> dict:
    identity = copy.deepcopy(identity)
    return {
        "hierarchical_discovery_contract_identity": identity,
        "architecture_contract": {
            "required_families": list(interfaces.ACTIVE_STAGE1_CONCEPT_FAMILIES),
            **production_stage1_hierarchy_architecture_bindings(identity),
        },
        "security": {
            "remote_clients_constructed": False,
            "remote_calls_allowed": False,
            "manual_digest_approval_required": False,
            "raw_evidence_sidecars_visible_to_prompts": False,
        },
    }


def test_contract_imports_and_authenticates_complete_hierarchical_stack():
    identity = current_production_stage1_hierarchy_contract_identity()
    assert validate_production_stage1_hierarchy_contract_identity(identity) == identity
    assert tuple(identity["required_family_order"]) == interfaces.ACTIVE_STAGE1_CONCEPT_FAMILIES

    versions = identity["semantic_versions"]
    assert (
        versions["interfaces"]["DISCOVERY_INTERFACE_SCHEMA_VERSION"]
        == "all_evidence_discovery_interfaces_v10"
    )
    assert (
        versions["interfaces"]["DISCOVERY_WIRE_NORMALIZATION_VERSION"]
        == "atomic_occurrence_compiler_normalization_v3"
    )
    assert (
        versions["response_contract"]["HIERARCHICAL_DISCOVERY_EXACT_COVERAGE_REPRESENTATION"]
        == "closed_object_keyed_by_authenticated_identifier_v1"
    )
    assert (
        versions["orchestrator"]["HIERARCHICAL_DISCOVERY_ORCHESTRATOR_VERSION"]
        == "hierarchical_all_architecture_discovery_orchestrator_v12"
    )
    assert (
        versions["approved_agent"]["APPROVED_HIERARCHICAL_DISCOVERY_AGENT_VERSION"]
        == "approved_hierarchical_discovery_agent_v9"
    )
    assert (
        versions["approved_agent"]["AUTHENTICATED_RUNNER_EXECUTION_TRACE_VERSION"]
        == "authenticated_json_discovery_runner_and_cache_execution_trace_v8"
    )
    assert (
        versions["adaptive_hierarchy"]["ADAPTIVE_IMPLEMENTATION_BUNDLE_VERSION"]
        == "adaptive_hierarchical_implementation_bundle_v8"
    )
    assert (
        versions["frozen_review"]["FROZEN_HIERARCHICAL_REVIEW_EVIDENCE_POLICY_VERSION"]
        == "round_1_accepted_routed_feature_support_only_v3"
    )
    assert (
        versions["review_provider"]["REVIEW_SPENT_EVIDENCE_CACHE_VERSION"]
        == "context_fit_review_spent_evidence_cache_v4"
    )

    for name in (
        "base_hierarchy_implementation_bundle",
        "adaptive_hierarchy_implementation_bundle",
    ):
        bundle = copy.deepcopy(identity[name])
        declared = bundle.pop("implementation_bundle_sha256")
        assert _sha(bundle) == declared

    assert (
        identity["standalone_module_files"]["stage1_hierarchy_contract"]["filename"]
        == "production_stage1_hierarchy_contract.py"
    )
    assert identity["standalone_module_files"]["production_handoff"]["filename"] == (
        "production_stage1_hierarchy_handoff.py"
    )
    assert identity["standalone_module_files"]["fusion_runner"]["filename"] == (
        "all_evidence_fusion_runner.py"
    )

    body = dict(identity)
    declared = body.pop("content_sha256")
    assert _sha(body) == declared


def test_contract_encodes_architecture_local_then_lossless_cross_architecture_workflow():
    identity = current_production_stage1_hierarchy_contract_identity()
    workflow = identity["feature_discovery_workflow"]
    assert workflow["architecture_local_before_cross_architecture_integration"] is True
    assert workflow["all_ten_architectures_must_be_interpreted"] is True
    assert workflow["global_top_k_allowed"] is False
    assert workflow["raw_all_architecture_evidence_dump_allowed"] is False
    assert workflow["family_mixed_interpretation_chunks_allowed"] is False
    assert "within_architecture_consolidation" in workflow["levels"][2]
    assert workflow["levels"][4] == "one_raw_support_item_per_page_then_recursive_folds"

    hierarchy = identity["lossless_raw_evidence_hierarchy"]
    assert hierarchy["resolver"] == "exact_authenticated_catalog_evidence_id_only"
    assert hierarchy["model_written_or_unknown_ids_allowed"] is False
    assert hierarchy["one_raw_evidence_item_per_page"] is True
    assert hierarchy[
        "maximum_fold_inputs_configured_by_hierarchy_wire_budget"
    ] is True
    assert hierarchy[
        "maximum_fresh_inputs_after_first_fold_derived_from_configured_fan_in"
    ] is True
    assert hierarchy["integration_rejection_and_extraction_all_use_pages_and_folds"] is True
    assert hierarchy["every_input_receives_an_explicit_disposition"] is True
    assert hierarchy["semantic_sampling_or_truncation"] is False
    assert hierarchy["legacy_configured_lookback_and_feature_caps_are_not_semantic_limits"] is True

    wire = identity["wire_contract"]
    assert wire["legacy_exact_length_enum_array_allowed"] is False
    assert wire["duplicate_identifier_can_satisfy_exact_coverage"] is False
    assert wire["raw_wire_and_normalized_projection_hashes_required"] is True

    adaptive = identity["adaptive_review_contract"]
    assert adaptive["lossless_id_addressed_pages_and_recursive_folds_required"] is True
    assert adaptive["every_target_evidence_candidate_and_diagnostic_page_required"] is True
    assert adaptive["every_proposal_judged_before_capacity_disposition"] is True
    assert adaptive["extraction_reviews_every_support_item_before_terminal_fold"] is True
    assert adaptive["semantic_sampling_or_truncation_allowed"] is False
    assert adaptive["spent_evidence_projection_uses_exhaustive_vocabulary_and_no_term_cap"] is True


def test_contract_binds_truthful_tfidf_training_scope_policies_and_internal_digest_carry():
    identity = current_production_stage1_hierarchy_contract_identity()
    tfidf = identity["tfidf_nested_training_only_calibration"]
    assert tuple(tfidf["families"]) == interfaces.ACTIVE_STAGE1_CONCEPT_FAMILIES[6:9]
    assert tfidf["all_three_paths_require_truthful_training_scope_policy_records"] is True
    assert tuple(tfidf["label_based_nested_selection_families"]) == (
        interfaces.ACTIVE_STAGE1_CONCEPT_FAMILIES[7:9]
    )
    assert tuple(tfidf["deterministic_exhaustive_no_selection_families"]) == (
        interfaces.ACTIVE_STAGE1_CONCEPT_FAMILIES[6],
    )
    assert tfidf["selection_inside_each_applicable_registered_training_scope"] is True
    assert tfidf["semantic_retrieval_replay_partitions_are_nonselecting_canaries"] is True
    assert tfidf["semantic_retrieval_projection_vocabulary_or_output_cap_allowed"] is False
    assert tfidf["semantic_retrieval_nested_calibration_labels_accessed"] is False
    assert tfidf["selection_frozen_before_registered_heldout_text_transform"] is True
    assert tfidf["registered_heldout_treatment_or_outcome_available"] is False
    assert tfidf["hierarchy_partitions_reused_as_calibration_folds"] is False

    digest = identity["digest_execution_policy"]
    assert digest["low_level_exact_digest_binding_retained"] is True
    assert digest["digest_carried_by_one_authorized_cohort_invocation"] is True
    assert digest["end_user_digest_entry_required"] is False
    assert digest["manual_digest_approval_required"] is False
    assert digest["registered_json_parsed_from_authenticated_byte_snapshot"] is True
    assert digest["strict_duplicate_json_keys_rejected"] is True
    assert digest["bundle_root_and_all_registered_paths_descriptor_anchored"] is True
    assert digest["intermediate_and_final_symlinks_followed"] is False
    assert digest["loader_to_handoff_manifest_reopen_allowed"] is False
    assert digest["preparation_input_wrapper_schema"].endswith("_v2")
    assert digest["preparation_batch_wrapper_schema"].endswith("_v1")
    assert digest["expected_digests_from_in_memory_prepared_batch_required"] is True
    assert digest["caller_mapping_or_bare_expected_digests_accepted"] is False
    assert digest["exact_concrete_prepared_batch_capability_required"] is True
    assert digest["prepared_batch_capability_process_local_and_one_shot"] is True
    assert digest["execution_authorization_exact_typed_and_one_shot"] is True
    assert digest["exact_same_process_runner_and_runtime_objects_required"] is True
    assert digest["caller_replay_registrations_accepted"] is False
    assert (
        digest["runtime_provider_identities_reauthenticated_before_authorization_and_execution"]
        is True
    )
    assert digest["runtime_scientific_input_file_hashes_reauthenticated"] is True
    assert digest["exact_coordinator_and_precommit_objects_required"] is True
    assert digest["canonical_unbound_coordinator_execute_required"] is True
    assert digest["exact_authenticated_batch_result_type_required"] is True
    assert digest["cross_process_path_replay_belongs_only_to_standalone_cli"] is True


def test_contract_uses_runtime_response_identity_without_deployment_pin():
    identity = current_production_stage1_hierarchy_contract_identity()
    policy = identity["remote_runtime_identity_policy"]
    assert policy["one_canonical_explicit_http_or_https_endpoint_required"] is True
    assert policy["explicit_localhost_endpoint_allowed"] is True
    assert policy["endpoint_pool_fallback_or_substitution_allowed"] is False
    assert policy["one_exact_explicit_model_name_required"] is True
    assert policy["model_autodiscovery_pool_fallback_or_substitution_allowed"] is False
    assert policy["response_model_must_equal_requested_model"] is True
    assert policy["response_finish_reason_must_equal_stop"] is True
    assert policy["response_metadata_checked_before_content_semantics_and_cache"] is True
    assert policy["response_policy_applies_to_initial_invalid_and_repair_responses"] is True
    assert policy["guarded_client_paths"] == [
        "hierarchical_discovery",
        "proposal_and_post_extraction_review",
        "explicit_feature_extraction",
    ]
    assert policy["deployment_metadata_required_for_execution"] is False
    assert policy["deployment_metadata_if_present_is_authoritative"] is False
    assert policy["compiled_deployment_digest_pin_required"] is False
    assert policy["caller_supplied_digest_or_approval_can_authorize"] is False
    assert policy["immediate_canary_is_separate_operational_gate"] is True
    assert "served_deployment_attestation" not in identity["semantic_versions"]
    assert "served_deployment_attestation" not in identity["standalone_module_files"]


def test_contract_fails_closed_on_old_interface_version(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        interfaces,
        "DISCOVERY_INTERFACE_SCHEMA_VERSION",
        "all_evidence_discovery_interfaces_v4",
    )
    with pytest.raises(RuntimeError, match="required 'all_evidence_discovery_interfaces_v10'"):
        current_production_stage1_hierarchy_contract_identity()


def test_contract_fails_closed_on_rehashed_but_false_implementation_bundle(
    monkeypatch: pytest.MonkeyPatch,
):
    original = hierarchy.hierarchical_discovery_implementation_bundle

    def false_bundle():
        value = copy.deepcopy(original())
        value["files"]["all_evidence_discovery_interfaces.py"] = "f" * 64
        body = dict(value)
        body.pop("implementation_bundle_sha256")
        value["implementation_bundle_sha256"] = _sha(body)
        return value

    monkeypatch.setattr(hierarchy, "hierarchical_discovery_implementation_bundle", false_bundle)
    with pytest.raises(RuntimeError, match="dependency bytes changed"):
        current_production_stage1_hierarchy_contract_identity()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("raw_all_architecture_prompt_allowed", True),
        ("raw_all_architecture_evidence_dump_allowed", True),
        ("global_top_k_before_discovery_allowed", True),
        ("family_mixed_interpretation_chunks_allowed", True),
        ("legacy_exact_coverage_array_allowed", True),
        ("lossless_exact_id_raw_evidence_pages_and_recursive_folds_required", False),
        ("manual_digest_approval_required", True),
    ],
)
def test_request_binding_rejects_old_flat_or_array_contract(field: str, bad_value):
    identity = current_production_stage1_hierarchy_contract_identity()
    request = _minimal_request(identity)
    assert validate_production_stage1_hierarchy_request_bindings(request) == identity
    request["architecture_contract"][field] = bad_value
    with pytest.raises(ValueError, match="weakens or changes"):
        validate_production_stage1_hierarchy_request_bindings(request)


def test_request_binding_rejects_identity_tamper_and_manual_security_prompt():
    identity = current_production_stage1_hierarchy_contract_identity()
    request = _minimal_request(identity)
    request["hierarchical_discovery_contract_identity"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="contract hash is invalid"):
        validate_production_stage1_hierarchy_request_bindings(request)

    request = _minimal_request(identity)
    request["security"]["manual_digest_approval_required"] = True
    with pytest.raises(ValueError, match="no-prompt security"):
        validate_production_stage1_hierarchy_request_bindings(request)


def test_contract_does_not_over_attest_missing_native_proof_or_e2e_readiness():
    identity = current_production_stage1_hierarchy_contract_identity()
    policy = identity["native_proof_policy"]
    assert policy["component_emitted_exact_inner_and_cumulative_spent_proofs_required"] is True
    assert policy["typed_cumulative_spent_all_ten_producer_boundary_required"] is True
    assert policy["cumulative_spent_producer_receives_spent_text_treatment_outcome"] is True
    assert policy["cumulative_spent_producer_receives_sealed_row_ids_only"] is True
    assert policy["cumulative_spent_producer_receives_sealed_text_or_labels"] is False
    assert policy["wrapper_assembled_or_schema_only_proof_sufficient"] is False
    assert policy["production_readiness_without_genuine_binder_validation_and_e2e_allowed"] is False
