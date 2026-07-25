from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT,
    DIRECT_UPSTREAM_NUMERICAL_CHANNEL,
    OUTCOME_AXIS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TREATMENT_AXIS,
    DiscoveryEvidenceItem,
    ExtractionDefinitionRequest,
    canonical_json,
    content_sha256,
    extraction_vocabulary_grounding_policy,
    render_interpret_evidence_chunk_messages,
)
from oci.inference.approved_hierarchical_discovery_batch import (
    APPROVED_HIERARCHICAL_DISCOVERY_BATCH_PRECOMMIT_VERSION,
    FROZEN_REVIEW_EVIDENCE_POLICY_VERSION,
    ApprovedHierarchicalDiscoveryBatchPrecommit,
    FrozenReviewEvidencePolicyBinding,
)
from oci.inference.approved_hierarchical_discovery_agent import (
    APPROVED_HIERARCHICAL_DISCOVERY_AGENT_VERSION,
    APPROVED_HIERARCHICAL_DISCOVERY_PRECOMMIT_VERSION,
)
from oci.inference.adaptive_hierarchical_stage1_reconsideration import (
    AdaptiveReconsiderationConfig,
    adaptive_hierarchical_stage1_reconsideration_identity,
)
from oci.inference.frozen_hierarchical_review_evidence import (
    frozen_hierarchical_review_evidence_identity,
)
from oci.inference.hierarchical_all_architecture_discovery import (
    CONSOLIDATE_ARCHITECTURE_JOB,
    COVERAGE_CRITIC_JOB,
    CROSS_ARCHITECTURE_INTEGRATION_JOB,
    CROSS_ARCHITECTURE_PLANNER_JOB,
    DISCOVERY_JOB_LEDGER_VERSION,
    EXTRACTION_DEFINITION_JOB,
    HIERARCHICAL_DISCOVERY_ORCHESTRATOR_VERSION,
    HIERARCHICAL_DISCOVERY_PRECOMMIT_VERSION,
    INTERPRET_CHUNK_JOB,
    MAX_RENDERED_DISCOVERY_PROMPT_BYTES,
    DiscoveryJobLedger,
    DiscoveryJobSettings,
    DiscoveryJsonJob,
    _render_extraction_messages,
    discovery_response_repair_policy_identity,
)
from oci.inference.offline_hierarchical_discovery_review_packet import (
    AuthenticatedPromptFile,
    build_offline_extraction_definition_prompt_preview,
    compose_offline_hierarchical_discovery_review_packet,
)


def _digest(label: str) -> str:
    return content_sha256({"label": label})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _evidence(*, first_member_count: int = 1) -> tuple[DiscoveryEvidenceItem, ...]:
    result = []
    for index, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES):
        result.append(
            DiscoveryEvidenceItem(
                evidence_id=f"evidence.{index:02d}",
                source_family=family,
                observable_axes=(TREATMENT_AXIS, OUTCOME_AXIS),
                member_ids=tuple(
                    f"member.{index:02d}.{member_index:02d}"
                    for member_index in range(first_member_count if index == 0 else 1)
                ),
                content={
                    "phrase": f"marker level {index}",
                    "signed_clue": "positive",
                },
            )
        )
    return tuple(result)


def _initial_jobs(
    evidence: tuple[DiscoveryEvidenceItem, ...],
) -> tuple[DiscoveryJsonJob, ...]:
    return tuple(
        DiscoveryJsonJob.create(
            job_kind=INTERPRET_CHUNK_JOB,
            scope=f"{item.source_family}.chunk_001",
            dependencies=(),
            settings=DiscoveryJobSettings.selector(),
            messages=render_interpret_evidence_chunk_messages(
                family_explanation=(
                    f"The {item.source_family} architecture supplied one uncertain readable clue."
                ),
                evidence=(item,),
            ),
            input_bindings={
                "catalog_sha256": _digest("catalog"),
                "chunk_plan_sha256": _digest("chunks"),
                "chunk_id": f"chunk.{item.source_family}",
                "source_family": item.source_family,
            },
        )
        for item in evidence
    )


def _runner_identity() -> dict[str, Any]:
    body = {
        "schema_version": "test_runner_v1",
        "implementation": {"file_sha256": _digest("runner-file")},
        "endpoint_urls": ["http://offline.invalid/v1"],
        "model": {"name": "explicit-test-model", "resolution": "explicit_only_no_autodiscovery"},
        "authentication": {
            "api_key_mode": "empty_placeholder",
            "api_key_sha256": _digest("empty-key"),
        },
        "request_timeout_seconds": 10.0,
        "retry": {
            "max_retries": 2,
            "max_attempts": 3,
            "sdk_internal_max_retries": 0,
        },
        "max_tokens": 32_768,
        "response_semantics": {
            "messages": "exact_job_messages_without_augmentation",
            "selector_thinking": {
                "enabled": True,
                "thinking_token_budget": 5_000,
            },
            "extraction_thinking": {
                "enabled": False,
                "thinking_token_budget_field": "omitted",
            },
        },
        "client_factory": {"mode": "test_never_called"},
    }
    return {**body, "identity_sha256": content_sha256(body)}


def _cache_binding(tmp_path: Path) -> dict[str, Any]:
    body = {
        "schema_version": "authenticated_hierarchical_discovery_job_cache_identity_v1",
        "cache_version": "authenticated_hierarchical_discovery_job_cache_v1",
        "mode": "read_write_immutable",
        "root_envelope": {
            "kind": "machine_local_absolute_path",
            "absolute_path": str((tmp_path / "selector-cache").resolve()),
        },
        "config": {
            "max_entry_bytes": 32_000_000,
            "file_mode": 0o600,
            "directory_mode": 0o700,
            "write_policy": "exclusive_create_never_overwrite",
            "replay_policy": "strict_bytes_then_same_semantic_validator",
            "symlink_policy": "reject_cache_root_namespace_and_entry_symlinks",
        },
        "implementation_file_sha256": _digest("cache-file"),
        "entry_schema_version": "cache-entry-v1",
        "hit_metadata_schema_version": "cache-hit-v1",
    }
    identity = {**body, "identity_sha256": content_sha256(body)}
    return {
        "mode": "authenticated_immutable",
        "class": "test.AuthenticatedImmutableCache",
        "identity": identity,
        "identity_sha256": content_sha256(identity),
        "implementation_file_sha256": _digest("cache-file"),
    }


def _compiler_binding() -> dict[str, Any]:
    identity = {
        "schema_version": "test_compiler_v1",
        "max_candidates": 16,
        "identity_sha256": _digest("compiler-inner"),
    }
    return {
        "class": "test.Compiler",
        "identity": identity,
        "identity_sha256": content_sha256(identity),
        "implementation_file_sha256": _digest("compiler-file"),
    }


def _batch_precommit(
    tmp_path: Path,
    *,
    cache_enabled: bool = True,
    deferred_first_gate_intent: bool = False,
    first_member_count: int = 1,
    semantic_member_cap: int = 3,
    adaptive_semantic_member_cap: int | None = None,
    mutate: Any | None = None,
    mutate_batch: Any | None = None,
) -> tuple[
    ApprovedHierarchicalDiscoveryBatchPrecommit,
    tuple[DiscoveryEvidenceItem, ...],
    tuple[DiscoveryJsonJob, ...],
]:
    evidence = _evidence(first_member_count=first_member_count)
    jobs = _initial_jobs(evidence)
    ledger = DiscoveryJobLedger.build(jobs).as_dict()
    assert ledger["schema_version"] == DISCOVERY_JOB_LEDGER_VERSION
    runner = _runner_identity()
    config = {
        "max_rendered_prompt_bytes": MAX_RENDERED_DISCOVERY_PROMPT_BYTES,
        "max_semantic_member_ids_per_chunk": semantic_member_cap,
        "max_integrated_features": 16,
        "selector_thinking_enabled": True,
        "selector_thinking_token_budget": 5_000,
        "extraction_definition_thinking_enabled": False,
        "extraction_definition_thinking_token_budget": 0,
    }
    hierarchy_implementation_sha256 = _digest("hierarchy-file")
    hierarchy_bundle_body = {
        "schema_version": "hierarchical-discovery-test-bundle-v1",
        "files": {
            "hierarchical_all_architecture_discovery.py": hierarchy_implementation_sha256,
            "all_evidence_discovery_interfaces.py": _digest("interfaces-file"),
            "hierarchical_discovery_response_contract.py": _digest("response-contract-file"),
            "lossless_stage1_evidence_catalog.py": _digest("catalog-file"),
        },
    }
    hierarchy_bundle = {
        **hierarchy_bundle_body,
        "implementation_bundle_sha256": content_sha256(hierarchy_bundle_body),
    }
    hierarchy_packet = {
        "schema_version": HIERARCHICAL_DISCOVERY_PRECOMMIT_VERSION,
        "orchestrator_version": HIERARCHICAL_DISCOVERY_ORCHESTRATOR_VERSION,
        "orchestrator_implementation_file_sha256": hierarchy_implementation_sha256,
        "orchestrator_implementation_bundle": hierarchy_bundle,
        "orchestrator_implementation_bundle_sha256": hierarchy_bundle[
            "implementation_bundle_sha256"
        ],
        "catalog_binding": {
            "catalog_sha256": _digest("catalog"),
            "split_fingerprint": _digest("split"),
            "outer_fold": 1,
            "scope": "spent",
            "inner_fold": None,
            "atom_count": len(evidence),
        },
        "chunk_plan_binding": {
            "plan_sha256": _digest("chunks"),
            "chunk_count": len(jobs),
            "max_semantic_member_ids_per_chunk": semantic_member_cap,
            "delivery_audit": {
                "all_catalog_atoms_delivered_exactly_once": True,
                "all_catalog_semantic_member_ids_delivered_exactly_once": True,
                "catalog_semantic_member_id_count": sum(len(item.member_ids) for item in evidence),
                "observed_semantic_member_id_delivery_count": sum(
                    len(item.member_ids) for item in evidence
                ),
                "max_semantic_member_ids_per_chunk": semantic_member_cap,
                "observed_max_semantic_member_ids_per_chunk": first_member_count,
                "non_grounding_numerical_summaries_delivered": False,
            },
        },
        "runner_identity": runner,
        "config": config,
        "response_repair_policy": discovery_response_repair_policy_identity(),
        "dossier_direct_numerical_bindings": [],
        "initial_job_ledger": ledger,
        "downstream_contract": {
            "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "role_routing": "deterministic_observable_axis_rules_after_integration",
        },
        "assurances": {
            "raw_atoms_delivered_exactly_once": True,
            "mixed_architecture_interpretation_jobs": False,
            "direct_row_level_numerical_values_accepted": False,
            "bounded_response_repair_implemented": True,
            "unvalidated_response_cache_write_allowed": False,
        },
    }
    direct_contract_sha256 = _digest(
        "first-gate-intent" if deferred_first_gate_intent else "direct-manifest"
    )
    modern_family_bindings = []
    modern_dossier_bindings = []
    if deferred_first_gate_intent:
        evidence_by_family = {item.source_family: item for item in evidence}
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
            semantic_ids = [evidence_by_family[family].evidence_id]
            signal_count = 0 if family == TFIDF_SEMANTIC_RETRIEVAL else 1
            zero_reason = (
                "semantic_retrieval_has_no_independent_row_signal" if signal_count == 0 else ""
            )
            modern_family_bindings.append(
                {
                    "source_family": family,
                    "semantic_atom_ids": semantic_ids,
                    "semantic_atom_ids_sha256": content_sha256(semantic_ids),
                    "semantic_atom_count": len(semantic_ids),
                    "signal_count": signal_count,
                    "numerical_zero_reason": zero_reason,
                }
            )
            modern_dossier_bindings.append(
                {
                    "source_family": family,
                    "channel": DIRECT_UPSTREAM_NUMERICAL_CHANNEL,
                    "direct_numerical_contract_kind": (
                        DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
                    ),
                    "direct_numerical_contract_sha256": direct_contract_sha256,
                    "signal_count": signal_count,
                    "zero_reason": zero_reason,
                    "concept_grounding_allowed": False,
                }
            )
        hierarchy_packet["direct_numerical_contract_binding"] = {
            "direct_numerical_contract_kind": (DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT),
            "direct_numerical_contract_sha256": direct_contract_sha256,
            "model_facing": False,
        }
        hierarchy_packet["dossier_direct_numerical_bindings"] = modern_dossier_bindings
    hierarchy_sha = content_sha256(hierarchy_packet)
    family_counts = {family: 1 for family in ACTIVE_STAGE1_CONCEPT_FAMILIES}
    cache_binding = (
        {
            **_cache_binding(tmp_path),
            "validator_code_sha256": hierarchy_bundle["implementation_bundle_sha256"],
        }
        if cache_enabled
        else {
            "mode": "disabled",
            "cache_lookup_allowed": False,
            "cache_write_allowed": False,
            "validator_code_sha256": hierarchy_bundle["implementation_bundle_sha256"],
        }
    )
    wrapper = {
        "schema_version": APPROVED_HIERARCHICAL_DISCOVERY_PRECOMMIT_VERSION,
        "agent_version": APPROVED_HIERARCHICAL_DISCOVERY_AGENT_VERSION,
        "agent_implementation_file_sha256": _digest("agent-file"),
        "catalog_binding": {
            "catalog_sha256": _digest("catalog"),
            "outer_fold": 1,
            "scope": "spent",
            "inner_fold": None,
            "split_fingerprint": _digest("split"),
            "atom_count": len(evidence),
            "family_atom_counts": family_counts,
        },
        "chunk_plan_binding": {
            "plan_sha256": _digest("chunks"),
            "chunk_count": len(jobs),
            "max_atoms_per_chunk": 2,
            "max_bytes_per_chunk": 48_000,
            "max_semantic_member_ids_per_chunk": semantic_member_cap,
        },
        "direct_numerical_manifest_binding": {
            "manifest_sha256": _digest("direct-manifest"),
            "semantic_catalog_sha256": _digest("catalog"),
            "signal_count": 100,
            "families": [
                {"source_family": family, "semantic_atom_count": 1}
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            ],
            "row_values_included": False,
            "matrix_metadata_included": False,
            "coordinate_metadata_included": False,
            "concept_grounding_allowed": False,
        },
        "direct_numerical_dossier_bindings": [],
        "hierarchy_precommit": {
            "precommit_sha256": hierarchy_sha,
            "packet": hierarchy_packet,
        },
        "runner_identity": runner,
        "job_cache_binding": cache_binding,
        "compiler_binding": _compiler_binding(),
        "config_bounds": config,
        "assurances": {
            "all_active_architectures_bound": True,
            "all_catalog_atoms_delivered_exactly_once": True,
            "direct_manifest_authenticated_locally_in_full": True,
            "direct_row_level_numerical_values_in_packet": False,
            "direct_coordinate_metadata_in_packet": False,
            "unapproved_remote_execution_allowed": False,
            "final_dossiers_revalidated_against_full_manifest": True,
            "runner_retry_records_authenticated_in_final_result": True,
            "cache_hits_authenticated_in_final_result": True,
            "cache_lookup_before_wrapper_approval_allowed": False,
            "cache_write_before_semantic_validation_allowed": False,
        },
    }
    if deferred_first_gate_intent:
        wrapper.pop("direct_numerical_manifest_binding")
        wrapper["direct_numerical_contract_binding"] = {
            "direct_numerical_contract_kind": (DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT),
            "direct_numerical_contract_sha256": direct_contract_sha256,
            "source_cache_key": _digest("first-gate-source-cache"),
            "stable_output_schema_sha256": _digest("stable-output-schema"),
            "semantic_catalog_sha256": _digest("catalog"),
            "expected_shared_lineage_sha256": _digest("expected-lineage"),
            "lineage_scope": "exact_spent_oof_and_label_free_first_gate_rows",
            "signal_count": sum(row["signal_count"] for row in modern_family_bindings),
            "families": modern_family_bindings,
            "materialization_state": ("deferred_until_after_approval_and_proposal_freeze"),
            "row_values_included": False,
            "matrix_metadata_included": False,
            "coordinate_metadata_included": False,
            "coordinate_to_semantic_atom_linkage": False,
            "concept_grounding_allowed": False,
        }
        wrapper["direct_numerical_dossier_bindings"] = modern_dossier_bindings
        wrapper["assurances"].pop("direct_manifest_authenticated_locally_in_full")
        wrapper["assurances"].pop("final_dossiers_revalidated_against_full_manifest")
        wrapper["assurances"].update(
            {
                "direct_numerical_contract_authenticated_locally_in_full": True,
                "direct_numerical_contract_kind": (
                    DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
                ),
                "direct_numerical_contract_materialized": False,
                "final_dossiers_revalidated_against_approved_contract": True,
            }
        )
    if mutate is not None:
        mutate(wrapper)
    wrapper_sha = content_sha256(wrapper)
    fold_row = {
        "ordinal": 1,
        "outer_fold": 1,
        "split_fingerprint_sha256": _digest("split"),
        "catalog_sha256": _digest("catalog"),
        "chunk_plan_sha256": _digest("chunks"),
        "direct_numerical_manifest_sha256": _digest("direct-manifest"),
        "hierarchy_precommit_sha256": hierarchy_sha,
        "wrapper_approval_sha256": wrapper_sha,
        "wrapper_packet": wrapper,
    }
    if deferred_first_gate_intent:
        fold_row.pop("direct_numerical_manifest_sha256")
        fold_row.update(
            {
                "direct_numerical_contract_kind": (
                    DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
                ),
                "direct_numerical_contract_sha256": direct_contract_sha256,
            }
        )
    adaptive_identity = adaptive_hierarchical_stage1_reconsideration_identity(
        config=AdaptiveReconsiderationConfig(
            max_semantic_member_ids_per_chunk=(
                semantic_member_cap
                if adaptive_semantic_member_cap is None
                else adaptive_semantic_member_cap
            )
        )
    )
    review_policy = FrozenReviewEvidencePolicyBinding(
        max_evidence_ids=32,
        max_evidence_bytes=64_000,
        review_materializer_identity=frozen_hierarchical_review_evidence_identity(),
        adaptive_reconsideration_identity=adaptive_identity,
    )
    batch_packet = {
        "schema_version": APPROVED_HIERARCHICAL_DISCOVERY_BATCH_PRECOMMIT_VERSION,
        "coordinator_version": "approved_hierarchical_discovery_batch_coordinator_v1",
        "coordinator_code_identity": {
            "class": "test.Coordinator",
            "implementation_file_sha256": _digest("batch-file"),
        },
        "input_manifest_sha256": _digest("input-manifest"),
        "frozen_review_evidence_policy": review_policy.as_dict(),
        "ordered_outer_folds": [1],
        "ordered_folds": [fold_row],
        "common_bindings": {
            "runner_identity": runner,
            "compiler_binding": _compiler_binding(),
            "hierarchy_config_identity": config,
            "architecture_chunk_limits": {
                "max_atoms_per_chunk": 2,
                "max_bytes_per_chunk": 48_000,
                "max_semantic_member_ids_per_chunk": semantic_member_cap,
            },
        },
        "assurances": {
            "all_fold_wrapper_packets_included_in_full": True,
            "outer_folds_unique_complete_and_one_based": True,
            "all_fold_static_preflights_before_first_cache_lookup": True,
            "all_fold_static_preflights_before_first_remote_call": True,
            "wrong_or_missing_batch_approval_rejected_before_preflight": True,
            "per_fold_execution_uses_exact_wrapper_approval_sha256": True,
            "mixed_runner_compiler_or_hierarchy_config_allowed": False,
            "round_1_frozen_review_evidence_is_accepted_support_only": True,
            "architecture_wide_review_evidence_dump_allowed": False,
            "later_round_feature_rediscovery_uses_fresh_exact_spent_catalog": True,
            "later_round_all_ten_architectures_required": True,
            "later_round_architecture_at_a_time_interpretation_required": True,
            "later_round_compact_ten_dossier_planner_required": True,
            "later_round_bounded_requested_id_lookback_only": True,
            "later_round_executable_definition_uses_requested_atoms_only": True,
            "later_round_proposal_frozen_before_next_gate": True,
            "same_frozen_review_evidence_used_for_every_round": False,
            "adaptive_reconsideration_identity_authenticated": True,
            "ordered_batch_results_content_authenticated": True,
        },
    }
    if deferred_first_gate_intent:
        batch_packet["common_bindings"][
            "direct_numerical_contract_kind"
        ] = DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
    if mutate_batch is not None:
        mutate_batch(batch_packet)
    return ApprovedHierarchicalDiscoveryBatchPrecommit.create(batch_packet), evidence, jobs


def _extraction_preview(item: DiscoveryEvidenceItem, dependency_job_id: str) -> DiscoveryJsonJob:
    del dependency_job_id
    return build_offline_extraction_definition_prompt_preview(
        canonical_name="marker_level",
        evidence=(item,),
        supporting_evidence_ids=(item.evidence_id,),
        value_shape_hypothesis="continuous",
    )


def _prompt_file(tmp_path: Path, name: str, payload: bytes) -> AuthenticatedPromptFile:
    path = tmp_path / name
    path.write_bytes(payload)
    return AuthenticatedPromptFile(
        path=path,
        expected_sha256=_sha256_bytes(payload),
        display_name=name,
    )


def _rehash_first_fold(packet: dict[str, Any]) -> None:
    row = packet["ordered_folds"][0]
    wrapper = row["wrapper_packet"]
    hierarchy = wrapper["hierarchy_precommit"]
    hierarchy["precommit_sha256"] = content_sha256(hierarchy["packet"])
    row["hierarchy_precommit_sha256"] = hierarchy["precommit_sha256"]
    row["wrapper_approval_sha256"] = content_sha256(wrapper)


def _rehash_review_policy(packet: dict[str, Any]) -> None:
    policy = packet["frozen_review_evidence_policy"]
    body = {key: value for key, value in policy.items() if key != "policy_sha256"}
    policy["policy_sha256"] = content_sha256(body)


def _rehash_adaptive_prompt_contract(packet: dict[str, Any]) -> None:
    policy = packet["frozen_review_evidence_policy"]
    contract = policy["adaptive_reconsideration_identity"]["prompt_contract"]
    body = {key: value for key, value in contract.items() if key != "prompt_contract_sha256"}
    contract["prompt_contract_sha256"] = content_sha256(body)
    _rehash_review_policy(packet)


def _rehash_adaptive_implementation_bundle(packet: dict[str, Any]) -> None:
    policy = packet["frozen_review_evidence_policy"]
    bundle = policy["adaptive_reconsideration_identity"]["implementation_bundle"]
    body = {key: value for key, value in bundle.items() if key != "implementation_bundle_sha256"}
    bundle["implementation_bundle_sha256"] = content_sha256(body)
    _rehash_review_policy(packet)


def test_live_v2_policy_as_dict_matches_the_offline_closed_phased_schema(
    tmp_path: Path,
) -> None:
    batch, _, _ = _batch_precommit(tmp_path)
    policy = batch.packet["frozen_review_evidence_policy"]

    assert policy["schema_version"] == FROZEN_REVIEW_EVIDENCE_POLICY_VERSION
    assert set(policy) == {
        "schema_version",
        "max_evidence_ids",
        "max_evidence_bytes",
        "accepted_support_only",
        "review_materializer_identity",
        "adaptive_reconsideration_identity",
        "evidence_selection_rule",
        "architecture_wide_single_prompt_evidence_dump_allowed",
        "round_1_feature_rediscovery_allowed",
        "later_round_feature_rediscovery_allowed",
        "same_frozen_evidence_used_for_every_round",
        "policy_sha256",
    }
    body = {key: value for key, value in policy.items() if key != "policy_sha256"}
    assert policy["policy_sha256"] == content_sha256(body)
    adaptive_identity = policy["adaptive_reconsideration_identity"]
    assert set(adaptive_identity) == {
        "schema_version",
        "authenticated_execution_version",
        "executable_bridge_version",
        "implementation_file_sha256",
        "implementation_bundle",
        "config",
        "config_sha256",
        "prompt_contract",
        "phase_policy",
    }
    prompt_contract = adaptive_identity["prompt_contract"]
    prompt_body = {
        key: value for key, value in prompt_contract.items() if key != "prompt_contract_sha256"
    }
    assert prompt_contract["prompt_contract_sha256"] == content_sha256(prompt_body)
    implementation_bundle = adaptive_identity["implementation_bundle"]
    bundle_body = {
        key: value
        for key, value in implementation_bundle.items()
        if key != "implementation_bundle_sha256"
    }
    assert implementation_bundle["implementation_bundle_sha256"] == content_sha256(bundle_body)
    assert (
        implementation_bundle["files"]["adaptive_hierarchical_stage1_reconsideration.py"]
        == adaptive_identity["implementation_file_sha256"]
    )


def test_complete_packet_combines_exact_controls_real_prompts_and_all_audits(
    tmp_path: Path,
) -> None:
    batch, evidence, jobs = _batch_precommit(tmp_path)
    historical_bytes = b'{"historical": "byte-exact\\ncontrol"}\n'
    old_bytes = b"old hierarchy prompt ablation\n"
    packet = compose_offline_hierarchical_discovery_review_packet(
        batch_precommit=batch,
        representative_outer_fold=1,
        representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[3],
        extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
        extraction_preview_outer_fold=1,
        historical_prompt=_prompt_file(tmp_path, "historical.txt", historical_bytes),
        old_hierarchy_prompt=_prompt_file(tmp_path, "old.txt", old_bytes),
    )

    body = packet.packet
    assert packet.approval_ready is True
    assert body["run_level_batch_precommit"]["approval_sha256"] == batch.approval_sha256
    artifacts = body["comparison_artifacts"]
    assert base64.b64decode(artifacts["historical_prompt"]["bytes_base64"]) == historical_bytes
    assert base64.b64decode(artifacts["old_hierarchy_prompt"]["bytes_base64"]) == old_bytes
    real = body["representative_real_family_prompt"]
    assert real["source_family"] == ACTIVE_STAGE1_CONCEPT_FAMILIES[3]
    assert real["evidence_count"] == 1
    assert real["exact_messages"][0]["content"].startswith("You interpret")
    assert body["role_routing_policy_review"]["reviewed_conclusions"][
        "treatment_only_is_not_confounder_adjustment"
    ]
    assert body["extraction_definition_prompt"]["settings"] == {
        "thinking_enabled": False,
        "thinking_token_budget": 0,
        "response_format": "json",
    }
    assert body["model_hidden_field_audit"]["internal_machine_field_leaks_detected"] == []
    context = body["context_size_audit"]
    assert len(context["initial_job_audits"]) == len(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    assert [row["source_family"] for row in context["initial_job_audits"]] == list(
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    assert all(
        row["thinking_token_budget"] == 5000
        for row in (audit["settings"] for audit in context["initial_job_audits"])
    )
    assert context["every_reviewed_job_within_configured_guard"] is True
    assert context["every_initial_job_within_semantic_member_bound"] is True
    assert all(
        row["semantic_member_id_count"] == 1
        and row["max_semantic_member_ids_per_chunk"] == 3
        and row["semantic_member_id_headroom"] == 2
        for row in context["initial_job_audits"]
    )
    assert context["semantic_repair_prompt_implemented"] is True
    repair = context["response_repair_policy"]
    assert repair["maximum_repair_attempts"] == 1
    assert repair["message_sequence"] == ["system", "user", "assistant", "user"]
    assert repair["diagnostic_policy"] == (
        "fixed_category_only_no_exception_text_no_model_identifiers_v1"
    )
    assert repair["prior_response_content_model_visible"] is False
    assert repair["prior_response_content_persisted"] is False
    assert set(repair["repair_assistant_placeholders"]) == {
        "local_json_schema_validation_failure",
        "raw_transport_budget_failure",
        "strict_json_parse_failure",
    }
    assert context["repair_cache_policy"].startswith("only_validated_final_response")
    identities = body["immutable_precommit_and_cache_identities"]
    assert (
        identities["cache_namespace_identities"][0]["job_cache_binding"]["identity"]["config"][
            "write_policy"
        ]
        == "exclusive_create_never_overwrite"
    )
    assert body["review_scope"] == {
        "purpose": "human_review_before_any_new_remote_discovery_comparison",
        "remote_execution_authorized": False,
        "batch_execute_called": False,
        "job_cache_lookup_performed": False,
        "final_output_touched": False,
    }
    phased = body["phased_adaptive_review_policy"]
    assert phased["round_1"]["feature_rediscovery_allowed"] is False
    assert phased["rounds_2_and_later"] == {
        "evidence_scope": "fresh_exact_accumulated_spent_stage1_catalog",
        "feature_rediscovery_allowed": True,
        "all_ten_architectures_required": True,
        "architecture_at_a_time_interpretation": True,
        "consolidation_and_complete_coverage_per_architecture": True,
        "planner_input": (
            "exactly_ten_compact_dossiers_current_registry_and_sanitized_diagnostics"
        ),
        "raw_lookback": "bounded_planner_requested_evidence_ids_only",
        "complete_raw_catalog_dump_allowed": False,
        "future_gate_text_or_labels_allowed": False,
        "proposal_frozen_before_next_gate": True,
    }
    assurances = body["provenance_and_honesty_assurances"]
    assert assurances["round_1_post_extraction_review_is_accepted_support_only"] is True
    assert assurances["rounds_2_and_later_use_fresh_exact_accumulated_spent_stage1_catalog"] is True
    assert assurances["same_frozen_evidence_used_for_every_review_round"] is False
    assert assurances["later_round_planner_raw_catalog_atom_count"] == 0
    assert assurances["complete_raw_catalog_dump_allowed"] is False
    assert body["review_readiness"]["phased_adaptive_review_policy_authenticated"] is True
    prompt_contract = phased["later_round_static_prompt_contract"]
    assert len(prompt_contract["stages"]) == 6
    assert prompt_contract["stage_order"] == [
        INTERPRET_CHUNK_JOB,
        CONSOLIDATE_ARCHITECTURE_JOB,
        COVERAGE_CRITIC_JOB,
        CROSS_ARCHITECTURE_PLANNER_JOB,
        CROSS_ARCHITECTURE_INTEGRATION_JOB,
        EXTRACTION_DEFINITION_JOB,
    ]
    assert all(
        stage["settings"]
        == {
            "thinking_enabled": True,
            "thinking_token_budget": 5000,
            "response_format": "json",
        }
        for stage in prompt_contract["stages"][:-1]
    )
    assert prompt_contract["stages"][-1]["settings"] == {
        "thinking_enabled": False,
        "thinking_token_budget": 0,
        "response_format": "json",
    }
    assert all(
        set(stage)
        == {
            "stage",
            "template_version",
            "settings",
            "system_instruction",
            "dynamic_inputs",
            "user_payload_top_level_keys",
            "dynamic_payload_shapes",
            "static_user_payload_literals",
            "dynamic_user_payload_paths",
            "output_schema",
        }
        for stage in prompt_contract["stages"]
    )
    assert all(
        stage["user_payload_top_level_keys"][0] == "job"
        and stage["user_payload_top_level_keys"][-1] == "output_schema"
        and stage["dynamic_user_payload_paths"]
        and stage["dynamic_payload_shapes"]
        and stage["output_schema"]
        for stage in prompt_contract["stages"]
    )
    assert all(
        set(stage["static_user_payload_literals"]) == {"job"}
        for stage in prompt_contract["stages"][:-1]
    )
    expected_vocabulary_policy = {
        key: value
        for key, value in extraction_vocabulary_grounding_policy().items()
        if key != "schema_version"
    }
    definition_stage = prompt_contract["stages"][-1]
    assert definition_stage["static_user_payload_literals"] == {
        "job": "define_one_extraction_feature",
        "vocabulary_grounding_policy": expected_vocabulary_policy,
    }
    assert all(
        not path.startswith("vocabulary_grounding_policy")
        for path in definition_stage["dynamic_user_payload_paths"]
    )
    assert prompt_contract["dynamic_fold_content_in_static_contract"] is False
    phased_variants = prompt_contract["phased_stage_variants"]
    assert len(phased_variants) == 8
    assert len({(stage["stage"], stage["request_job"]) for stage in phased_variants}) == len(
        phased_variants
    )
    assert all(
        stage["request_job"] == stage["static_user_payload_literals"]["job"]
        and stage["user_payload_top_level_keys"][0] == "job"
        and stage["user_payload_top_level_keys"][-1] == "output_schema"
        for stage in phased_variants
    )
    prompt_body = {
        key: value for key, value in prompt_contract.items() if key != "prompt_contract_sha256"
    }
    assert prompt_contract["prompt_contract_sha256"] == content_sha256(prompt_body)
    implementation_bundle = phased["adaptive_implementation_bundle"]
    assert (
        implementation_bundle
        == phased["adaptive_reconsideration_identity"]["implementation_bundle"]
    )
    assert set(implementation_bundle["files"]) == {
        "adaptive_hierarchical_stage1_reconsideration.py",
        "all_evidence_discovery_interfaces.py",
        "hierarchical_all_architecture_discovery.py",
        "hierarchical_discovery_response_contract.py",
        "all_evidence_fusion.py",
        "all_evidence_post_extraction_review.py",
        "lossless_stage1_evidence_catalog.py",
        "stage1_architecture_explanations.py",
    }
    local_validator = implementation_bundle["local_json_schema_validator"]
    assert local_validator["implementation"] == ("jsonschema.validators.Draft202012Validator")
    assert local_validator["dependency_versions"]
    assert local_validator["resolved_module_file_sha256"]
    bundle_body = {
        key: value
        for key, value in implementation_bundle.items()
        if key != "implementation_bundle_sha256"
    }
    assert implementation_bundle["implementation_bundle_sha256"] == content_sha256(bundle_body)
    identities = body["immutable_precommit_and_cache_identities"]
    assert (
        identities["adaptive_reconsideration_config_sha256"]
        == phased["adaptive_reconsideration_identity"]["config_sha256"]
    )
    assert (
        identities["adaptive_static_prompt_contract_sha256"]
        == prompt_contract["prompt_contract_sha256"]
    )
    assert (
        identities["adaptive_implementation_bundle_sha256"]
        == implementation_bundle["implementation_bundle_sha256"]
    )
    assert packet.packet_sha256 == content_sha256(body)
    assert "Phased post-extraction review policy" in packet.render_markdown()
    assert "Exact static authorization envelope" in packet.render_markdown()
    assert "Authenticated adaptive implementation bundle" in packet.render_markdown()
    assert implementation_bundle["implementation_bundle_sha256"] in packet.render_markdown()
    assert prompt_contract["stages"][3]["system_instruction"] in packet.render_markdown()
    rendered = packet.render_markdown()
    assert "Exact user-payload top-level keys" in rendered
    assert "Exact static user-payload literals" in rendered
    assert "Dynamic user-payload paths" in rendered
    assert "Exact dynamic payload shapes" in rendered
    assert canonical_json(prompt_contract["stages"][4]["dynamic_payload_shapes"]) in rendered
    assert canonical_json(prompt_contract["stages"][5]["static_user_payload_literals"]) in rendered
    definition_section = rendered.split(f"#### `{EXTRACTION_DEFINITION_JOB}`", maxsplit=1)[1]
    static_section = definition_section.split("Exact static user-payload literals:", maxsplit=1)[
        1
    ].split("Dynamic user-payload paths", maxsplit=1)[0]
    dynamic_section = definition_section.split(
        "Dynamic user-payload paths (not populated in this offline packet):", maxsplit=1
    )[1].split("Exact dynamic payload shapes:", maxsplit=1)[0]
    assert "vocabulary_grounding_policy" in static_section
    assert "vocabulary_grounding_policy" not in dynamic_section
    assert "Real architecture-local family prompt" in packet.render_markdown()


def test_deferred_first_gate_intent_packet_is_normalized_without_manifest_mislabeling(
    tmp_path: Path,
) -> None:
    batch, evidence, jobs = _batch_precommit(
        tmp_path,
        deferred_first_gate_intent=True,
    )
    packet = compose_offline_hierarchical_discovery_review_packet(
        batch_precommit=batch,
        representative_outer_fold=1,
        representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
        extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
        extraction_preview_outer_fold=1,
    )

    identities = packet.packet["immutable_precommit_and_cache_identities"]
    fold = identities["fold_precommit_identities"][0]
    expected_sha256 = _digest("first-gate-intent")
    assert fold["direct_numerical_contract_kind"] == (
        DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
    )
    assert fold["direct_numerical_contract_sha256"] == expected_sha256
    assert "direct_numerical_manifest_sha256" not in fold
    serialized_messages = canonical_json(
        packet.packet["representative_real_family_prompt"]["exact_messages"]
    )
    assert expected_sha256 not in serialized_messages
    assert "direct_numerical_contract" not in serialized_messages


@pytest.mark.parametrize(
    ("mutate_batch", "message"),
    (
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"].update(
                    {
                        "evidence_selection_rule": (
                            "exact_supporting_evidence_ids_of_hierarchy_accepted_features_only"
                        )
                    }
                ),
                _rehash_review_policy(packet),
            ),
            "changed required phase behavior",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"].update(
                    {"later_round_feature_rediscovery_allowed": False}
                ),
                _rehash_review_policy(packet),
            ),
            "changed required phase behavior",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"].update(
                    {"same_frozen_evidence_used_for_every_round": True}
                ),
                _rehash_review_policy(packet),
            ),
            "changed required phase behavior",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"].update(
                    {"architecture_wide_single_prompt_evidence_dump_allowed": True}
                ),
                _rehash_review_policy(packet),
            ),
            "changed required phase behavior",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"].update(
                    {"implementation_file_sha256": _digest("unreviewed-adaptive-code")}
                ),
                _rehash_review_policy(packet),
            ),
            "primary implementation differs from its dependency bundle",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "config"
                ].update({"max_total_lookback_ids": 23}),
                _rehash_review_policy(packet),
            ),
            "config_sha256 does not authenticate",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "phase_policy"
                ].update({"bounded_requested_id_pages_only": False}),
                _rehash_review_policy(packet),
            ),
            "does not close every required boundary",
        ),
    ),
)
def test_phased_review_policy_rejects_rehashed_global_or_weakened_policy(
    tmp_path: Path,
    mutate_batch: Any,
    message: str,
) -> None:
    batch, evidence, jobs = _batch_precommit(tmp_path, mutate_batch=mutate_batch)

    with pytest.raises(ValueError, match=message):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
            extraction_preview_outer_fold=1,
        )


@pytest.mark.parametrize(
    ("mutate_batch", "message"),
    (
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "prompt_contract"
                ]["stages"][3].update(
                    {"system_instruction": "Use one unrestricted raw catalog dump."}
                ),
                _rehash_adaptive_prompt_contract(packet),
            ),
            "differs from current production templates",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "prompt_contract"
                ]["stages"][0]["settings"].update({"thinking_token_budget": 4999}),
                _rehash_adaptive_prompt_contract(packet),
            ),
            "differs from its authenticated config",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "prompt_contract"
                ]["stages"][-1]["settings"].update({"thinking_enabled": True}),
                _rehash_adaptive_prompt_contract(packet),
            ),
            "reasoning must be disabled",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "prompt_contract"
                ]["stages"][0]["user_payload_top_level_keys"].insert(1, "unreviewed_context"),
                _rehash_adaptive_prompt_contract(packet),
            ),
            "differs from current production templates",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "prompt_contract"
                ]["stages"][1]["static_user_payload_literals"].update(
                    {"job": "unreviewed_consolidation_job"}
                ),
                _rehash_adaptive_prompt_contract(packet),
            ),
            "differs from current production templates",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "prompt_contract"
                ]["stages"][2]["dynamic_user_payload_paths"].append("output_schema"),
                _rehash_adaptive_prompt_contract(packet),
            ),
            "dynamic user payload paths are invalid",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "prompt_contract"
                ]["stages"][4]["dynamic_payload_shapes"].update(
                    {"unreviewed_context_keys": ["oracle"]}
                ),
                _rehash_adaptive_prompt_contract(packet),
            ),
            "differs from current production templates",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "prompt_contract"
                ]["stages"][5]["output_schema"].update({"unreviewed_field": "value"}),
                _rehash_adaptive_prompt_contract(packet),
            ),
            "differs from current production templates",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "prompt_contract"
                ]["stages"][5]["static_user_payload_literals"].pop("vocabulary_grounding_policy"),
                _rehash_adaptive_prompt_contract(packet),
            ),
            "static user payload literals differ from their closed stage schema",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "prompt_contract"
                ]["stages"][5]["static_user_payload_literals"][
                    "vocabulary_grounding_policy"
                ].update(
                    {"clinical_vocabulary": "unreviewed_vocabulary_source"}
                ),
                _rehash_adaptive_prompt_contract(packet),
            ),
            "static vocabulary grounding policy differs from current production",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "prompt_contract"
                ]["phased_stage_variants"][0].update({"request_job": "unreviewed_relation_job"}),
                _rehash_adaptive_prompt_contract(packet),
            ),
            "request job differs from its static job literal",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "prompt_contract"
                ]["phased_stage_variants"][0]["settings"].update({"thinking_token_budget": 4999}),
                _rehash_adaptive_prompt_contract(packet),
            ),
            "differs from its authenticated config",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "prompt_contract"
                ]["stages"][5]["static_user_payload_literals"][
                    "vocabulary_grounding_policy"
                ].update(
                    {"unreviewed_policy_field": True}
                ),
                _rehash_adaptive_prompt_contract(packet),
            ),
            "static vocabulary grounding policy differs from current production",
        ),
    ),
)
def test_adaptive_static_prompt_contract_rejects_rehashed_template_or_reasoning_changes(
    tmp_path: Path,
    mutate_batch: Any,
    message: str,
) -> None:
    batch, evidence, jobs = _batch_precommit(tmp_path, mutate_batch=mutate_batch)

    with pytest.raises(ValueError, match=message):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
            extraction_preview_outer_fold=1,
        )


@pytest.mark.parametrize(
    ("mutate_batch", "message"),
    (
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "implementation_bundle"
                ]["files"].update(
                    {"all_evidence_fusion.py": _digest("changed-compiler-dependency")}
                ),
                _rehash_adaptive_implementation_bundle(packet),
            ),
            "differs from current dependencies",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "implementation_bundle"
                ].update({"implementation_bundle_sha256": _digest("unbound-bundle")}),
                _rehash_review_policy(packet),
            ),
            "does not authenticate its dependencies",
        ),
        (
            lambda packet: (
                packet["frozen_review_evidence_policy"]["adaptive_reconsideration_identity"][
                    "implementation_bundle"
                ]["local_json_schema_validator"]["resolved_module_file_sha256"].update(
                    {"jsonschema.validators": _digest("changed-local-validator")}
                ),
                _rehash_adaptive_implementation_bundle(packet),
            ),
            "differs from current dependencies",
        ),
    ),
)
def test_adaptive_implementation_bundle_rejects_rehashed_dependency_or_digest_changes(
    tmp_path: Path,
    mutate_batch: Any,
    message: str,
) -> None:
    batch, evidence, jobs = _batch_precommit(tmp_path, mutate_batch=mutate_batch)

    with pytest.raises(ValueError, match=message):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
            extraction_preview_outer_fold=1,
        )


def test_phased_review_policy_accepts_a_fully_authenticated_bounded_adaptive_config(
    tmp_path: Path,
) -> None:
    def mutate(packet: dict[str, Any]) -> None:
        policy = packet["frozen_review_evidence_policy"]
        config = AdaptiveReconsiderationConfig(max_total_lookback_ids=23)
        policy["adaptive_reconsideration_identity"] = (
            adaptive_hierarchical_stage1_reconsideration_identity(config=config)
        )
        _rehash_review_policy(packet)

    batch, evidence, jobs = _batch_precommit(tmp_path, mutate_batch=mutate)
    packet = compose_offline_hierarchical_discovery_review_packet(
        batch_precommit=batch,
        representative_outer_fold=1,
        representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
        extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
        extraction_preview_outer_fold=1,
    )

    identity = packet.packet["phased_adaptive_review_policy"]["adaptive_reconsideration_identity"]
    assert identity["config"]["max_total_lookback_ids"] == 23
    assert identity["config_sha256"] == content_sha256(identity["config"])


@pytest.mark.parametrize(
    ("mutate_batch", "message"),
    (
        (
            lambda packet: packet["ordered_folds"][0].update(
                {"direct_numerical_manifest_sha256": _digest("first-gate-intent")}
            ),
            "mislabels its contract as a manifest",
        ),
        (
            lambda packet: (
                packet["ordered_folds"][0]["wrapper_packet"][
                    "direct_numerical_contract_binding"
                ].update({"coordinates": []}),
                _rehash_first_fold(packet),
            ),
            "unexpected closed schema",
        ),
        (
            lambda packet: (
                packet["ordered_folds"][0]["wrapper_packet"]["hierarchy_precommit"]["packet"][
                    "direct_numerical_contract_binding"
                ].update({"direct_numerical_contract_sha256": _digest("different-intent")}),
                _rehash_first_fold(packet),
            ),
            "inner hierarchy binds a different",
        ),
        (
            lambda packet: (
                packet["ordered_folds"][0]["wrapper_packet"]["direct_numerical_contract_binding"][
                    "families"
                ][0].update({"semantic_atom_ids": ["evidence.alien"]}),
                _rehash_first_fold(packet),
            ),
            "evidence IDs differ",
        ),
    ),
)
def test_deferred_intent_normalization_rejects_rehashed_ambiguous_or_inexact_packets(
    tmp_path: Path,
    mutate_batch: Any,
    message: str,
) -> None:
    batch, evidence, jobs = _batch_precommit(
        tmp_path,
        deferred_first_gate_intent=True,
        mutate_batch=mutate_batch,
    )

    with pytest.raises(ValueError, match=message):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
            extraction_preview_outer_fold=1,
        )


def test_missing_optional_controls_are_explicit_and_not_invented(tmp_path: Path) -> None:
    batch, evidence, jobs = _batch_precommit(tmp_path)
    packet = compose_offline_hierarchical_discovery_review_packet(
        batch_precommit=batch,
        representative_outer_fold=1,
        representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
        extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
        extraction_preview_outer_fold=1,
    )

    assert packet.approval_ready is False
    readiness = packet.packet["review_readiness"]
    assert readiness["missing_optional_comparison_artifacts"] == [
        "historical_prompt",
        "old_hierarchy_prompt",
    ]
    assert packet.packet["comparison_artifacts"]["historical_prompt"]["content_invented"] is False


def test_supplied_comparison_prompt_requires_exact_known_bytes(tmp_path: Path) -> None:
    batch, evidence, jobs = _batch_precommit(tmp_path)
    path = tmp_path / "historical.txt"
    path.write_text("changed", encoding="utf-8")
    bad = AuthenticatedPromptFile(
        path=path,
        expected_sha256=_digest("different"),
        display_name="historical",
    )

    with pytest.raises(ValueError, match="differ from expected_sha256"):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
            extraction_preview_outer_fold=1,
            historical_prompt=bad,
        )


def test_extraction_preview_must_be_non_executable_and_use_exact_fold_evidence(
    tmp_path: Path,
) -> None:
    batch, evidence, jobs = _batch_precommit(tmp_path)
    request = ExtractionDefinitionRequest(
        canonical_name="marker_level",
        evidence=(evidence[0],),
        supporting_evidence_ids=(evidence[0].evidence_id,),
        value_shape_hypothesis="continuous",
    )
    executable = DiscoveryJsonJob.create(
        job_kind=EXTRACTION_DEFINITION_JOB,
        scope="marker_level",
        dependencies=(jobs[0].job_id,),
        settings=DiscoveryJobSettings.extraction(),
        messages=_render_extraction_messages(request=request),
        input_bindings={"supporting_evidence_ids": [evidence[0].evidence_id]},
    )
    with pytest.raises(ValueError, match="non-executable offline preview"):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=executable,
            extraction_preview_outer_fold=1,
        )

    alien = DiscoveryEvidenceItem(
        evidence_id="alien.evidence",
        source_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
        observable_axes=(OUTCOME_AXIS,),
        content={"phrase": "alien marker"},
    )
    with pytest.raises(ValueError, match="outside its prepared fold"):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=_extraction_preview(alien, jobs[0].job_id),
            extraction_preview_outer_fold=1,
        )


def test_disabled_cache_namespace_fails_closed(tmp_path: Path) -> None:
    batch, evidence, jobs = _batch_precommit(tmp_path, cache_enabled=False)
    with pytest.raises(ValueError, match="authenticated immutable job-cache namespace"):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
            extraction_preview_outer_fold=1,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_policy", "exact current response-repair policy"),
        ("mutated_policy", "exact current response-repair policy"),
        ("legacy_precommit", "exact current response-repair policy"),
        ("missing_repair_assurance", "bounded response-repair assurance"),
        ("unsafe_cache_assurance", "unvalidated response cache write"),
    ],
)
def test_each_fold_must_bind_current_repair_policy_and_assurances(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    def mutate(wrapper: dict[str, Any]) -> None:
        hierarchy = wrapper["hierarchy_precommit"]
        inner = hierarchy["packet"]
        if mutation == "missing_policy":
            inner.pop("response_repair_policy")
        elif mutation == "mutated_policy":
            policy = inner["response_repair_policy"]
            policy["maximum_repair_attempts"] = 2
            policy_body = {key: value for key, value in policy.items() if key != "policy_sha256"}
            policy["policy_sha256"] = content_sha256(policy_body)
        elif mutation == "legacy_precommit":
            inner["schema_version"] = "hierarchical_discovery_precommit_v1"
            inner["orchestrator_version"] = (
                "hierarchical_all_architecture_discovery_orchestrator_v2"
            )
            inner.pop("response_repair_policy")
            inner["assurances"].pop("bounded_response_repair_implemented")
            inner["assurances"].pop("unvalidated_response_cache_write_allowed")
        elif mutation == "missing_repair_assurance":
            inner["assurances"].pop("bounded_response_repair_implemented")
        else:
            inner["assurances"]["unvalidated_response_cache_write_allowed"] = True
        hierarchy["precommit_sha256"] = content_sha256(inner)

    batch, evidence, jobs = _batch_precommit(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match=match):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
            extraction_preview_outer_fold=1,
        )


def test_nested_hierarchy_precommit_tampering_fails_even_under_a_new_batch_hash(
    tmp_path: Path,
) -> None:
    def mutate(wrapper: dict[str, Any]) -> None:
        wrapper["hierarchy_precommit"]["packet"]["config"]["selector_thinking_token_budget"] = 4999

    batch, evidence, jobs = _batch_precommit(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="hierarchy precommit SHA-256"):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
            extraction_preview_outer_fold=1,
        )


def test_offline_review_rejects_cross_bound_semantic_member_mismatch(
    tmp_path: Path,
) -> None:
    def mutate(wrapper: dict[str, Any]) -> None:
        wrapper["chunk_plan_binding"]["max_semantic_member_ids_per_chunk"] = 2

    batch, evidence, jobs = _batch_precommit(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="semantic-member chunk bounds differ"):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
            extraction_preview_outer_fold=1,
        )


def test_offline_review_recomputes_and_rejects_fully_resealed_overbound_job(
    tmp_path: Path,
) -> None:
    batch, evidence, jobs = _batch_precommit(
        tmp_path,
        first_member_count=2,
        semantic_member_cap=1,
    )
    with pytest.raises(ValueError, match="exceeds max_semantic_member_ids_per_chunk"):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
            extraction_preview_outer_fold=1,
        )


def test_offline_review_rejects_initial_adaptive_semantic_member_cap_divergence(
    tmp_path: Path,
) -> None:
    batch, evidence, jobs = _batch_precommit(
        tmp_path,
        semantic_member_cap=3,
        adaptive_semantic_member_cap=2,
    )
    with pytest.raises(ValueError, match="initial and adaptive architecture chunk limits differ"):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
            extraction_preview_outer_fold=1,
        )


def test_persistence_is_content_addressed_preparation_only_and_never_overwrites(
    tmp_path: Path,
) -> None:
    batch, evidence, jobs = _batch_precommit(tmp_path)
    packet = compose_offline_hierarchical_discovery_review_packet(
        batch_precommit=batch,
        representative_outer_fold=1,
        representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
        extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
        extraction_preview_outer_fold=1,
    )
    final_output = tmp_path / "final-output"
    final_output.mkdir()
    sentinel = final_output / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    preparation = tmp_path / "offline-preparation"

    persisted = packet.persist(preparation_directory=preparation)
    persisted.validate_authentication()
    assert persisted.packet_sha256 in persisted.packet_json_path.name
    assert persisted.packet_sha256 in persisted.packet_markdown_path.name
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert set(preparation.iterdir()) == {
        persisted.packet_json_path,
        persisted.packet_markdown_path,
        persisted.manifest_path,
    }
    with pytest.raises(FileExistsError, match="fresh and absent"):
        packet.persist(preparation_directory=preparation)


def test_context_audit_fails_before_persistence_for_oversized_prompt(tmp_path: Path) -> None:
    def mutate(wrapper: dict[str, Any]) -> None:
        inner = wrapper["hierarchy_precommit"]["packet"]
        jobs = inner["initial_job_ledger"]["jobs"]
        job = jobs[0]
        payload = json.loads(job["messages"][1]["content"])
        payload["family_explanation"] = "x" * (MAX_RENDERED_DISCOVERY_PROMPT_BYTES + 100)
        job["messages"][1]["content"] = canonical_json(payload)
        messages = job["messages"]
        envelope = job["input_bindings"]["authenticated_model_message_envelope"]
        envelope["sha256"] = content_sha256(messages)
        envelope["byte_count"] = len(canonical_json(messages).encode("utf-8"))
        identity = {key: value for key, value in job.items() if key != "job_id"}
        job["job_id"] = f"job_{content_sha256(identity)}"
        ledger_identity = {
            "schema_version": inner["initial_job_ledger"]["schema_version"],
            "jobs": jobs,
        }
        inner["initial_job_ledger"]["ledger_sha256"] = content_sha256(ledger_identity)
        wrapper["hierarchy_precommit"]["precommit_sha256"] = content_sha256(inner)

    batch, evidence, jobs = _batch_precommit(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="exceeds its configured byte guard"):
        compose_offline_hierarchical_discovery_review_packet(
            batch_precommit=batch,
            representative_outer_fold=1,
            representative_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            extraction_definition_preview=_extraction_preview(evidence[0], jobs[0].job_id),
            extraction_preview_outer_fold=1,
        )
