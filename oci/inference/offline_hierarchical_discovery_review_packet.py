"""Compose and persist the offline hierarchical-discovery review packet.

This module is deliberately outside every execution path.  It accepts an
already-authenticated run-level batch precommit plus one explicitly preview-only
extraction-definition job, revalidates their exact model-facing messages, and
produces human- and machine-readable review artifacts.  It never invokes a fold
agent, a job cache, a model runner, or a final-output writer.

Historical comparison prompts are optional inputs.  When supplied, their known
SHA-256 digest is mandatory and their exact bytes are embedded as base64.  No
normalization, truncation, or reconstruction is permitted.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT,
    DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST,
    DIRECT_NUMERICAL_CONTRACT_KINDS,
    DIRECT_UPSTREAM_NUMERICAL_CHANNEL,
    EXTRACTION_SUPPORT_AXIS,
    HETEROGENEITY_AXIS,
    OBSERVABLE_AXES,
    OUTCOME_AXIS,
    PAIR_UPLIFT_AXIS,
    ROLE_ROUTING_VERSION,
    TFIDF_SEMANTIC_RETRIEVAL,
    TREATMENT_AXIS,
    DiscoveryEvidenceItem,
    ExtractionDefinitionRequest,
    canonical_json,
    content_sha256,
    extraction_vocabulary_grounding_policy,
    route_concept_roles,
)
from .approved_hierarchical_discovery_batch import (
    FROZEN_REVIEW_EVIDENCE_POLICY_VERSION,
    ApprovedHierarchicalDiscoveryBatchPrecommit,
)
from .adaptive_hierarchical_stage1_reconsideration import (
    AdaptiveReconsiderationConfig,
    adaptive_hierarchical_implementation_bundle,
    adaptive_hierarchical_stage1_prompt_contract,
    adaptive_hierarchical_stage1_reconsideration_identity,
)
from .frozen_hierarchical_review_evidence import (
    frozen_hierarchical_review_evidence_identity,
)
from .hierarchical_all_architecture_discovery import (
    AUTHENTICATED_MESSAGE_ENVELOPE_BINDING,
    CONSOLIDATE_ARCHITECTURE_JOB,
    COVERAGE_CRITIC_JOB,
    CROSS_ARCHITECTURE_INTEGRATION_JOB,
    CROSS_ARCHITECTURE_PLANNER_JOB,
    DISCOVERY_JOB_LEDGER_VERSION,
    EXTRACTION_DEFINITION_JOB,
    INTERPRET_CHUNK_JOB,
    DiscoveryJsonJob,
    DiscoveryJobSettings,
    _render_extraction_messages,
    discovery_response_repair_policy_identity,
)

OFFLINE_HIERARCHICAL_DISCOVERY_REVIEW_PACKET_VERSION = (
    "offline_hierarchical_discovery_review_packet_v5"
)
OFFLINE_HIERARCHICAL_DISCOVERY_REVIEW_MANIFEST_VERSION = (
    "offline_hierarchical_discovery_review_manifest_v1"
)
MAX_AUTHENTICATED_COMPARISON_PROMPT_BYTES = 16_000_000

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_JOB_ID = re.compile(r"job_[0-9a-f]{64}\Z")
_MACHINE_EXACT_KEYS = frozenset(
    {
        "schema_version",
        "catalog_id",
        "cache_id",
        "cache_key",
        "manifest_id",
        "producer_id",
        "producer_identity",
        "split_fingerprint",
        "deterministic_role_routing",
        "role_routing_sha256",
        "direct_numerical_contract_kind",
    }
)
_MACHINE_TEXT_TOKENS = (
    "schema_version",
    "catalog_sha256",
    "coverage_audit_sha256",
    "manifest_sha256",
    "split_fingerprint",
    "producer_identity",
    "producer_id",
    "cache_id",
    "cache_key",
    "deterministic_role_routing",
    "role_routing_sha256",
    "direct_numerical_contract_kind",
    "direct_numerical_contract_sha256",
)
_POLICY_TEXT_TOKENS = (
    "temporal_policy",
    "temporal policy",
    "current_date",
    "current date",
)
_FORBIDDEN_EVIDENCE_KEY = re.compile(
    r"(?:^|_)(?:"
    r"oracle|ground_truth|true_ite|true_cate|true_effect|"
    r"row_id|row_ids|patient_id|patient_ids|mrn|medical_record_number|"
    r"heldout_row_ids|validation_row_ids|test_row_ids|"
    r"raw_vector|raw_vectors|embedding_vector|embedding_vectors|activations|"
    r"backend_path|artifact_path|cache_path|full_note|full_notes|raw_note|raw_notes"
    r")(?:_|$)",
    flags=re.IGNORECASE,
)
_INTERNAL_FIELD_HINT = re.compile(
    r"(?:^|_)(?:"
    r"schema|sha256|fingerprint|precommit|approval|binding|identity|cache|"
    r"manifest|ledger|implementation|runner|compiler|dependency|settings|"
    r"job_id|producer|lineage"
    r")(?:_|$)",
    flags=re.IGNORECASE,
)
_MODERN_INTENT_CONTRACT_KEYS = frozenset(
    {
        "direct_numerical_contract_kind",
        "direct_numerical_contract_sha256",
        "source_cache_key",
        "stable_output_schema_sha256",
        "semantic_catalog_sha256",
        "expected_shared_lineage_sha256",
        "lineage_scope",
        "signal_count",
        "families",
        "materialization_state",
        "row_values_included",
        "matrix_metadata_included",
        "coordinate_metadata_included",
        "coordinate_to_semantic_atom_linkage",
        "concept_grounding_allowed",
    }
)
_MODERN_REALIZED_CONTRACT_KEYS = frozenset(
    {
        "direct_numerical_contract_kind",
        "direct_numerical_contract_sha256",
        "source_cache_schema",
        "source_cache_key",
        "source_manifest_sha256",
        "producer_identity_sha256",
        "stable_output_schema_sha256",
        "semantic_catalog_sha256",
        "shared_lineage_sha256",
        "lineage_scope",
        "signal_count",
        "families",
        "row_values_included",
        "matrix_metadata_included",
        "coordinate_metadata_included",
        "coordinate_to_semantic_atom_linkage",
        "concept_grounding_allowed",
    }
)
_MODERN_NUMERICAL_FAMILY_KEYS = frozenset(
    {
        "source_family",
        "semantic_atom_ids",
        "semantic_atom_ids_sha256",
        "semantic_atom_count",
        "signal_count",
        "numerical_zero_reason",
    }
)
_MODERN_DOSSIER_BINDING_KEYS = frozenset(
    {
        "source_family",
        "channel",
        "direct_numerical_contract_kind",
        "direct_numerical_contract_sha256",
        "signal_count",
        "zero_reason",
        "concept_grounding_allowed",
    }
)
_PHASED_REVIEW_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "max_evidence_ids",
        "max_evidence_bytes",
        "accepted_support_only",
        "review_materializer_identity",
        "evidence_selection_rule",
        "round_1_feature_rediscovery_allowed",
        "later_round_feature_rediscovery_allowed",
        "same_frozen_evidence_used_for_every_round",
        "architecture_wide_single_prompt_evidence_dump_allowed",
        "adaptive_reconsideration_identity",
        "policy_sha256",
    }
)
_ADAPTIVE_RECONSIDERATION_IDENTITY_KEYS = frozenset(
    {
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
)
_ADAPTIVE_IMPLEMENTATION_BUNDLE_KEYS = frozenset(
    {
        "schema_version",
        "files",
        "local_json_schema_validator",
        "implementation_bundle_sha256",
    }
)
_ADAPTIVE_PROMPT_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "stage_order",
        "stages",
        "phased_stage_variants",
        "dynamic_fold_content_in_static_contract",
        "complete_catalog_single_prompt_authorized",
        "direct_or_non_grounding_numerical_context_authorized",
        "row_note_oracle_or_temporal_policy_context_authorized",
        "prompt_contract_sha256",
    }
)
_ADAPTIVE_PROMPT_STAGE_KEYS = frozenset(
    {
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
)
_ADAPTIVE_PHASED_PROMPT_STAGE_KEYS = frozenset({*_ADAPTIVE_PROMPT_STAGE_KEYS, "request_job"})
_ADAPTIVE_PROMPT_STAGE_ORDER = (
    INTERPRET_CHUNK_JOB,
    CONSOLIDATE_ARCHITECTURE_JOB,
    COVERAGE_CRITIC_JOB,
    CROSS_ARCHITECTURE_PLANNER_JOB,
    CROSS_ARCHITECTURE_INTEGRATION_JOB,
    EXTRACTION_DEFINITION_JOB,
)
_ADAPTIVE_RECONSIDERATION_CONFIG_KEYS = frozenset(
    {
        "max_atoms_per_chunk",
        "max_bytes_per_chunk",
        "max_semantic_member_ids_per_chunk",
        "max_lookback_ids_per_target",
        "max_total_lookback_ids",
        "max_total_lookback_bytes",
        "max_operations",
        "max_rendered_prompt_bytes",
        "selector_thinking_token_budget",
        "hierarchy_wire_budget",
    }
)
_ADAPTIVE_PHASE_POLICY_KEYS = frozenset(
    {
        "round_1_initial_frozen_support_may_be_reused",
        "later_round_fresh_exact_spent_catalog_required",
        "later_round_all_ten_architectures_required",
        "architecture_at_a_time_interpretation_required",
        "lossless_exhaustive_family_relation_pages_required",
        "complete_link_family_compiler_required",
        "terminating_group_definition_folds_required",
        "lossless_atomic_coverage_pages_required",
        "compact_ten_dossier_planner_required",
        "exhaustive_target_evidence_planner_pages_required",
        "terminating_target_folds_required",
        "bounded_requested_id_pages_only",
        "every_revision_proposal_judged_and_ledgered",
        "final_operation_capacity_after_explicit_dispositions_only",
        "one_raw_support_item_per_extraction_page_required",
        "terminating_extraction_support_folds_required",
        "proposal_freeze_before_next_gate_required",
        "complete_catalog_dump_forbidden",
        "direct_numerical_model_context_forbidden",
        "non_grounding_numerical_model_context_forbidden",
        "row_or_note_model_context_forbidden",
        "oracle_or_temporal_policy_model_context_forbidden",
    }
)


def _clone(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one JSON object")
    return value


def _direct_numerical_contract_identity(
    value: Any,
) -> tuple[dict[str, Any], str, str, bool]:
    """Normalize modern deferred contracts and legacy realized-manifest bindings."""

    binding = dict(_clone(_mapping(value, label="direct numerical contract binding")))
    kind = binding.get("direct_numerical_contract_kind")
    contract_sha256 = binding.get("direct_numerical_contract_sha256")
    legacy = kind is None and contract_sha256 is None and "manifest_sha256" in binding
    if legacy:
        kind = DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
        contract_sha256 = binding.get("manifest_sha256")
    if kind not in DIRECT_NUMERICAL_CONTRACT_KINDS:
        raise ValueError("direct numerical contract kind is unsupported")
    contract_sha256 = _require_sha256(
        contract_sha256,
        label="direct numerical contract SHA-256",
    )
    if not legacy and "manifest_sha256" in binding:
        raise ValueError("modern direct numerical contract contains a legacy manifest label")
    if "manifest_sha256" in binding and (
        kind != DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
        or binding.get("manifest_sha256") != contract_sha256
    ):
        raise ValueError("direct numerical manifest identity is inconsistent")
    if kind == DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT:
        if binding.get("materialization_state") != (
            "deferred_until_after_approval_and_proposal_freeze"
        ):
            raise ValueError("first-gate numerical materialization is not safely deferred")
        if "source_manifest_sha256" in binding:
            raise ValueError("deferred first-gate contract unexpectedly binds realized bytes")
    return binding, kind, contract_sha256, legacy


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be one non-negative integer")
    return value


def _validate_modern_direct_numerical_contract(
    binding: Mapping[str, Any],
    *,
    kind: str,
    contract_sha256: str,
    catalog_sha256: str,
    expected_evidence_ids_by_family: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    expected_keys = (
        _MODERN_INTENT_CONTRACT_KEYS
        if kind == DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT
        else _MODERN_REALIZED_CONTRACT_KEYS
    )
    if set(binding) != expected_keys:
        raise ValueError("modern direct numerical contract has an unexpected closed schema")
    if binding["direct_numerical_contract_kind"] != kind or (
        binding["direct_numerical_contract_sha256"] != contract_sha256
    ):
        raise ValueError("modern direct numerical contract identity changed during validation")
    if binding["semantic_catalog_sha256"] != catalog_sha256:
        raise ValueError("direct numerical contract binds a different semantic catalog")
    for name in (
        "source_cache_key",
        "stable_output_schema_sha256",
    ):
        _require_sha256(binding[name], label=f"direct numerical contract {name}")
    if not isinstance(binding["lineage_scope"], str) or not binding["lineage_scope"]:
        raise ValueError("direct numerical contract lineage_scope must be non-empty")
    if kind == DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT:
        _require_sha256(
            binding["expected_shared_lineage_sha256"],
            label="expected first-gate shared lineage SHA-256",
        )
        if binding["materialization_state"] != (
            "deferred_until_after_approval_and_proposal_freeze"
        ):
            raise ValueError("first-gate numerical materialization is not safely deferred")
    else:
        if (
            not isinstance(binding["source_cache_schema"], str)
            or not binding["source_cache_schema"]
        ):
            raise ValueError("realized numerical contract source cache schema is empty")
        for name in (
            "source_manifest_sha256",
            "producer_identity_sha256",
            "shared_lineage_sha256",
        ):
            _require_sha256(binding[name], label=f"realized numerical contract {name}")
    for flag in (
        "row_values_included",
        "matrix_metadata_included",
        "coordinate_metadata_included",
        "coordinate_to_semantic_atom_linkage",
        "concept_grounding_allowed",
    ):
        if binding[flag] is not False:
            raise ValueError(f"unsafe direct numerical contract flag: {flag}")

    family_values = binding["families"]
    if not isinstance(family_values, list) or len(family_values) != len(
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    ):
        raise ValueError("modern numerical contract does not contain exactly ten families")
    by_family: dict[str, dict[str, Any]] = {}
    total_signal_count = 0
    for expected_family, value in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, family_values):
        row = dict(_clone(_mapping(value, label="direct numerical family")))
        if set(row) != _MODERN_NUMERICAL_FAMILY_KEYS:
            raise ValueError("modern numerical family binding has an unexpected closed schema")
        if row["source_family"] != expected_family:
            raise ValueError("modern numerical families changed canonical architecture order")
        exact_ids = list(expected_evidence_ids_by_family[expected_family])
        if row["semantic_atom_ids"] != exact_ids:
            raise ValueError(f"direct numerical contract evidence IDs differ for {expected_family}")
        if row["semantic_atom_ids_sha256"] != content_sha256(exact_ids):
            raise ValueError(
                f"direct numerical contract evidence hash differs for {expected_family}"
            )
        if row["semantic_atom_count"] != len(exact_ids):
            raise ValueError(
                f"direct numerical contract evidence count differs for {expected_family}"
            )
        signal_count = _nonnegative_int(
            row["signal_count"], label=f"{expected_family} signal_count"
        )
        zero_reason = row["numerical_zero_reason"]
        if not isinstance(zero_reason, str) or bool(signal_count) == bool(zero_reason):
            raise ValueError(
                f"direct numerical contract zero reason is inconsistent for {expected_family}"
            )
        if expected_family == TFIDF_SEMANTIC_RETRIEVAL and signal_count != 0:
            raise ValueError("semantic retrieval unexpectedly has an independent row signal")
        if expected_family != TFIDF_SEMANTIC_RETRIEVAL and signal_count == 0:
            raise ValueError(f"active numerical architecture has no signals: {expected_family}")
        total_signal_count += signal_count
        by_family[expected_family] = row
    if _nonnegative_int(binding["signal_count"], label="direct numerical signal_count") != (
        total_signal_count
    ):
        raise ValueError("direct numerical contract total signal count is inconsistent")
    return by_family


def _validate_modern_hierarchy_numerical_bindings(
    *,
    wrapper: Mapping[str, Any],
    inner: Mapping[str, Any],
    kind: str,
    contract_sha256: str,
    numerical_families: Mapping[str, Mapping[str, Any]],
) -> None:
    inner_contract = _mapping(
        inner.get("direct_numerical_contract_binding"),
        label="inner direct numerical contract binding",
    )
    if set(inner_contract) != {
        "direct_numerical_contract_kind",
        "direct_numerical_contract_sha256",
        "model_facing",
    }:
        raise ValueError("inner direct numerical contract has an unexpected closed schema")
    if (
        inner_contract["direct_numerical_contract_kind"] != kind
        or inner_contract["direct_numerical_contract_sha256"] != contract_sha256
        or inner_contract["model_facing"] is not False
    ):
        raise ValueError("inner hierarchy binds a different direct numerical contract")
    wrapper_values = wrapper.get("direct_numerical_dossier_bindings")
    inner_values = inner.get("dossier_direct_numerical_bindings")
    if not isinstance(wrapper_values, list) or not isinstance(inner_values, list):
        raise ValueError("modern hierarchy omits direct numerical dossier bindings")
    if canonical_json(wrapper_values) != canonical_json(inner_values):
        raise ValueError("wrapper and inner numerical dossier bindings differ")
    if len(wrapper_values) != len(ACTIVE_STAGE1_CONCEPT_FAMILIES):
        raise ValueError("modern hierarchy does not bind exactly ten numerical dossiers")
    for expected_family, value in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, wrapper_values):
        row = _mapping(value, label="direct numerical dossier binding")
        if set(row) != _MODERN_DOSSIER_BINDING_KEYS:
            raise ValueError("direct numerical dossier binding has an unexpected closed schema")
        expected = numerical_families[expected_family]
        if (
            row["source_family"] != expected_family
            or row["channel"] != DIRECT_UPSTREAM_NUMERICAL_CHANNEL
            or row["direct_numerical_contract_kind"] != kind
            or row["direct_numerical_contract_sha256"] != contract_sha256
            or row["signal_count"] != expected["signal_count"]
            or row["zero_reason"] != expected["numerical_zero_reason"]
            or row["concept_grounding_allowed"] is not False
        ):
            raise ValueError("direct numerical dossier binding differs from its contract family")


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be one non-empty string")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be one positive integer")
    return value


def _implementation_file_sha256() -> str:
    return _sha256_bytes(Path(__file__).read_bytes())


def _module_file_sha256(value: Any) -> str:
    source = inspect.getsourcefile(value)
    if source is None:
        raise ValueError("reviewed implementation source cannot be authenticated")
    return _sha256_bytes(Path(source).resolve(strict=True).read_bytes())


def _validate_content_addressed_identity(value: Any, *, label: str) -> dict[str, Any]:
    identity = dict(_clone(_mapping(value, label=label)))
    declared = _require_sha256(identity.get("identity_sha256"), label=f"{label} SHA-256")
    body = {key: row for key, row in identity.items() if key != "identity_sha256"}
    if declared != content_sha256(body):
        raise ValueError(f"{label} SHA-256 does not authenticate its complete identity")
    return identity


def _validate_component_binding(value: Any, *, label: str) -> dict[str, Any]:
    binding = dict(_clone(_mapping(value, label=label)))
    required = {"class", "identity", "identity_sha256", "implementation_file_sha256"}
    if not required <= set(binding):
        raise ValueError(f"{label} is missing a content-addressed component field")
    _nonempty_string(binding["class"], label=f"{label} class")
    _require_sha256(binding["implementation_file_sha256"], label=f"{label} implementation SHA-256")
    identity = _mapping(binding["identity"], label=f"{label} identity")
    declared = _require_sha256(binding["identity_sha256"], label=f"{label} identity_sha256")
    if declared != content_sha256(identity):
        raise ValueError(f"{label} identity_sha256 does not authenticate its identity")
    return binding


def _validate_adaptive_implementation_bundle(value: Any) -> dict[str, Any]:
    bundle = dict(_clone(_mapping(value, label="adaptive implementation bundle")))
    if set(bundle) != _ADAPTIVE_IMPLEMENTATION_BUNDLE_KEYS:
        raise ValueError("adaptive implementation bundle has an unexpected closed schema")
    _nonempty_string(
        bundle["schema_version"], label="adaptive implementation bundle schema_version"
    )
    files = _mapping(bundle["files"], label="adaptive implementation bundle files")
    if not files:
        raise ValueError("adaptive implementation bundle files cannot be empty")
    for filename, digest in files.items():
        _nonempty_string(filename, label="adaptive implementation dependency filename")
        _require_sha256(digest, label=f"adaptive implementation dependency {filename}")
    validator = _mapping(
        bundle["local_json_schema_validator"],
        label="adaptive local JSON Schema validator identity",
    )
    validator_keys = {
        "schema_version",
        "draft",
        "implementation",
        "distribution",
        "distribution_version",
        "dependency_versions",
        "resolved_module_file_sha256",
    }
    if set(validator) != validator_keys:
        raise ValueError("adaptive local JSON Schema validator has an unexpected closed schema")
    for key in (
        "schema_version",
        "draft",
        "implementation",
        "distribution",
        "distribution_version",
    ):
        _nonempty_string(
            validator[key],
            label=f"adaptive local JSON Schema validator {key}",
        )
    dependency_versions = _mapping(
        validator["dependency_versions"],
        label="adaptive local JSON Schema validator dependency versions",
    )
    if not dependency_versions:
        raise ValueError("adaptive local JSON Schema validator dependencies cannot be empty")
    for name, version in dependency_versions.items():
        _nonempty_string(name, label="adaptive local JSON Schema validator dependency name")
        _nonempty_string(
            version,
            label=f"adaptive local JSON Schema validator dependency {name} version",
        )
    module_files = _mapping(
        validator["resolved_module_file_sha256"],
        label="adaptive local JSON Schema validator module files",
    )
    if not module_files:
        raise ValueError("adaptive local JSON Schema validator module files cannot be empty")
    for module_name, digest in module_files.items():
        _nonempty_string(
            module_name,
            label="adaptive local JSON Schema validator module name",
        )
        _require_sha256(
            digest,
            label=f"adaptive local JSON Schema validator module {module_name}",
        )
    declared = _require_sha256(
        bundle["implementation_bundle_sha256"],
        label="adaptive implementation bundle SHA-256",
    )
    body = {key: row for key, row in bundle.items() if key != "implementation_bundle_sha256"}
    if declared != content_sha256(body):
        raise ValueError(
            "adaptive implementation bundle SHA-256 does not authenticate its dependencies"
        )
    if canonical_json(bundle) != canonical_json(adaptive_hierarchical_implementation_bundle()):
        raise ValueError("adaptive implementation bundle differs from current dependencies")
    return bundle


def _validate_adaptive_prompt_stage(
    value: Any,
    *,
    label: str,
    expected_stage: str | None = None,
    phased: bool = False,
    expected_selector_thinking_token_budget: int,
) -> dict[str, Any]:
    stage = dict(_clone(_mapping(value, label=label)))
    expected_keys = _ADAPTIVE_PHASED_PROMPT_STAGE_KEYS if phased else _ADAPTIVE_PROMPT_STAGE_KEYS
    if set(stage) != expected_keys:
        raise ValueError("adaptive prompt stage has an unexpected closed schema")
    stage_kind = _nonempty_string(stage["stage"], label=f"{label} stage")
    if stage_kind not in _ADAPTIVE_PROMPT_STAGE_ORDER:
        raise ValueError("adaptive prompt stage has an unknown stage kind")
    if expected_stage is not None and stage_kind != expected_stage:
        raise ValueError("adaptive prompt stages changed authenticated order")
    request_job: str | None = None
    if phased:
        request_job = _nonempty_string(stage["request_job"], label=f"{label} request job")
    _nonempty_string(stage["template_version"], label=f"{label} template version")
    instruction = _nonempty_string(
        stage["system_instruction"],
        label=f"{label} system instruction",
    )
    lowered = instruction.casefold()
    leaked_policy = [token for token in _POLICY_TEXT_TOKENS if token in lowered]
    if leaked_policy:
        raise ValueError(f"adaptive static system instruction exposes policy text: {leaked_policy}")
    dynamic_inputs = stage["dynamic_inputs"]
    if (
        not isinstance(dynamic_inputs, list)
        or not dynamic_inputs
        or any(not isinstance(item, str) or not item for item in dynamic_inputs)
        or len(dynamic_inputs) != len(set(dynamic_inputs))
    ):
        raise ValueError("adaptive prompt stage dynamic inputs are invalid")
    user_payload_keys = stage["user_payload_top_level_keys"]
    if (
        not isinstance(user_payload_keys, list)
        or not user_payload_keys
        or any(not isinstance(item, str) or not item for item in user_payload_keys)
        or len(user_payload_keys) != len(set(user_payload_keys))
    ):
        raise ValueError("adaptive prompt stage user payload keys are invalid")
    if user_payload_keys[0] != "job" or user_payload_keys[-1] != "output_schema":
        raise ValueError(
            "adaptive prompt stage user payload must begin with job and end with output_schema"
        )
    if "hierarchy_wire_budget" not in user_payload_keys:
        raise ValueError(
            "adaptive prompt stage omits its authenticated hierarchy wire budget"
        )
    dynamic_shapes = _mapping(
        stage["dynamic_payload_shapes"],
        label=f"{label} dynamic payload shapes",
    )
    if not dynamic_shapes:
        raise ValueError("adaptive prompt stage dynamic payload shapes cannot be empty")
    static_literals = _mapping(
        stage["static_user_payload_literals"],
        label=f"{label} static user payload literals",
    )
    expected_static_literal_keys = {"job"}
    if "vocabulary_grounding_policy" in user_payload_keys:
        expected_static_literal_keys.add("vocabulary_grounding_policy")
    if set(static_literals) != expected_static_literal_keys:
        raise ValueError(
            "adaptive prompt stage static user payload literals differ from their "
            "closed stage schema"
        )
    static_job = _nonempty_string(
        static_literals["job"],
        label=f"{label} static job literal",
    )
    if request_job is not None and static_job != request_job:
        raise ValueError("adaptive phased prompt request job differs from its static job literal")
    if "vocabulary_grounding_policy" in expected_static_literal_keys:
        vocabulary_policy = _mapping(
            static_literals["vocabulary_grounding_policy"],
            label="adaptive extraction-definition static vocabulary grounding policy",
        )
        expected_vocabulary_policy = {
            key: row
            for key, row in extraction_vocabulary_grounding_policy().items()
            if key != "schema_version"
        }
        if canonical_json(vocabulary_policy) != canonical_json(expected_vocabulary_policy):
            raise ValueError(
                "adaptive extraction-definition static vocabulary grounding policy "
                "differs from current production"
            )
    dynamic_paths = stage["dynamic_user_payload_paths"]
    if (
        not isinstance(dynamic_paths, list)
        or not dynamic_paths
        or any(not isinstance(item, str) or not item for item in dynamic_paths)
        or len(dynamic_paths) != len(set(dynamic_paths))
    ):
        raise ValueError("adaptive prompt stage dynamic user payload paths are invalid")
    top_level_dynamic_keys = {path.split(".", maxsplit=1)[0] for path in dynamic_paths}
    if not top_level_dynamic_keys <= set(user_payload_keys):
        raise ValueError(
            "adaptive prompt stage dynamic paths escape its authenticated user payload"
        )
    if "hierarchy_wire_budget" not in top_level_dynamic_keys:
        raise ValueError(
            "adaptive prompt stage does not authenticate its hierarchy wire budget path"
        )
    output_schema = _mapping(stage["output_schema"], label=f"{label} output schema")
    if not output_schema:
        raise ValueError("adaptive prompt stage output schema cannot be empty")
    settings = _mapping(stage["settings"], label=f"{label} settings")
    if (
        set(settings)
        != {
            "thinking_enabled",
            "thinking_token_budget",
            "response_format",
        }
        or settings["response_format"] != "json"
    ):
        raise ValueError("adaptive prompt stage settings have an unexpected closed schema")
    if stage_kind == EXTRACTION_DEFINITION_JOB:
        if settings["thinking_enabled"] is not False or settings["thinking_token_budget"] != 0:
            raise ValueError("adaptive extraction-definition reasoning must be disabled")
    elif settings["thinking_enabled"] is not True or (
        settings["thinking_token_budget"]
        != expected_selector_thinking_token_budget
    ):
        raise ValueError(
            "adaptive selector reasoning differs from its authenticated config"
        )
    return stage


def _validate_adaptive_prompt_contract(
    value: Any,
    *,
    expected_selector_thinking_token_budget: int,
) -> dict[str, Any]:
    """Authenticate the exact static authorization envelope for later calls."""

    contract = dict(_clone(_mapping(value, label="adaptive prompt contract")))
    if set(contract) != _ADAPTIVE_PROMPT_CONTRACT_KEYS:
        raise ValueError("adaptive prompt contract has an unexpected closed schema")
    contract_sha256 = _require_sha256(
        contract["prompt_contract_sha256"],
        label="adaptive prompt contract SHA-256",
    )
    body = {key: row for key, row in contract.items() if key != "prompt_contract_sha256"}
    if contract_sha256 != content_sha256(body):
        raise ValueError("adaptive prompt contract SHA-256 does not authenticate its content")
    if contract["stage_order"] != list(_ADAPTIVE_PROMPT_STAGE_ORDER):
        raise ValueError("adaptive prompt contract changed the required stage order")
    stages = contract["stages"]
    if not isinstance(stages, list) or len(stages) != len(_ADAPTIVE_PROMPT_STAGE_ORDER):
        raise ValueError("adaptive prompt contract must expose exactly six stages")
    for index, (expected_stage, stage_value) in enumerate(
        zip(_ADAPTIVE_PROMPT_STAGE_ORDER, stages)
    ):
        _validate_adaptive_prompt_stage(
            stage_value,
            label=f"adaptive prompt stage[{index}]",
            expected_stage=expected_stage,
            expected_selector_thinking_token_budget=(
                expected_selector_thinking_token_budget
            ),
        )
    phased_variants = contract["phased_stage_variants"]
    if not isinstance(phased_variants, list) or not phased_variants:
        raise ValueError("adaptive prompt contract must expose phased stage variants")
    seen_variants: set[tuple[str, str]] = set()
    for index, stage_value in enumerate(phased_variants):
        stage = _validate_adaptive_prompt_stage(
            stage_value,
            label=f"adaptive phased prompt stage[{index}]",
            phased=True,
            expected_selector_thinking_token_budget=(
                expected_selector_thinking_token_budget
            ),
        )
        identity = (stage["stage"], stage["request_job"])
        if identity in seen_variants:
            raise ValueError("adaptive phased prompt variants cannot contain duplicates")
        seen_variants.add(identity)
    for flag in (
        "dynamic_fold_content_in_static_contract",
        "complete_catalog_single_prompt_authorized",
        "direct_or_non_grounding_numerical_context_authorized",
        "row_note_oracle_or_temporal_policy_context_authorized",
    ):
        if contract[flag] is not False:
            raise ValueError(f"adaptive prompt contract does not close forbidden behavior: {flag}")
    if canonical_json(contract) != canonical_json(
        adaptive_hierarchical_stage1_prompt_contract(
            selector_thinking_token_budget=(
                expected_selector_thinking_token_budget
            )
        )
    ):
        raise ValueError("adaptive prompt contract differs from current production templates")
    return contract


def _validate_phased_review_policy(value: Any) -> dict[str, Any]:
    """Authenticate the round-scoped review policy and its exact implementation.

    ``accepted_support_only`` is intentionally retained as a compatibility
    field, but in v2 it describes round 1 only.  Later rounds must use the
    separately authenticated adaptive hierarchy; they cannot reuse the frozen
    round-1 support catalog or broaden one prompt into a raw catalog dump.
    """

    policy = dict(_clone(_mapping(value, label="phased review evidence policy")))
    if set(policy) != _PHASED_REVIEW_POLICY_KEYS:
        raise ValueError("phased review evidence policy has an unexpected closed schema")
    if policy["schema_version"] != FROZEN_REVIEW_EVIDENCE_POLICY_VERSION or (
        FROZEN_REVIEW_EVIDENCE_POLICY_VERSION != "frozen_review_evidence_policy_v2"
    ):
        raise ValueError("offline review requires the phased v2 review policy")
    policy_sha256 = _require_sha256(policy["policy_sha256"], label="phased review policy_sha256")
    policy_body = {key: row for key, row in policy.items() if key != "policy_sha256"}
    if policy_sha256 != content_sha256(policy_body):
        raise ValueError("phased review policy SHA-256 does not authenticate its policy")
    _positive_int(policy["max_evidence_ids"], label="round-1 max_evidence_ids")
    _positive_int(policy["max_evidence_bytes"], label="round-1 max_evidence_bytes")
    expected_phase_values = {
        "accepted_support_only": True,
        "evidence_selection_rule": (
            "round_1_exact_supporting_evidence_ids_of_hierarchy_accepted_features_only"
        ),
        "round_1_feature_rediscovery_allowed": False,
        "later_round_feature_rediscovery_allowed": True,
        "same_frozen_evidence_used_for_every_round": False,
        "architecture_wide_single_prompt_evidence_dump_allowed": False,
    }
    for key, expected in expected_phase_values.items():
        if policy[key] != expected:
            raise ValueError(f"phased review policy changed required phase behavior: {key}")

    materializer_identity = dict(
        _clone(
            _mapping(
                policy["review_materializer_identity"],
                label="round-1 review materializer identity",
            )
        )
    )
    if canonical_json(materializer_identity) != canonical_json(
        frozen_hierarchical_review_evidence_identity()
    ):
        raise ValueError(
            "round-1 review materializer identity differs from the current implementation"
        )

    adaptive_identity = dict(
        _clone(
            _mapping(
                policy["adaptive_reconsideration_identity"],
                label="adaptive reconsideration identity",
            )
        )
    )
    if set(adaptive_identity) != _ADAPTIVE_RECONSIDERATION_IDENTITY_KEYS:
        raise ValueError("adaptive reconsideration identity has an unexpected closed schema")
    _require_sha256(
        adaptive_identity["implementation_file_sha256"],
        label="adaptive reconsideration implementation_file_sha256",
    )
    config = dict(_clone(_mapping(adaptive_identity["config"], label="adaptive review config")))
    if set(config) != _ADAPTIVE_RECONSIDERATION_CONFIG_KEYS:
        raise ValueError("adaptive review config has an unexpected closed schema")
    if adaptive_identity["config_sha256"] != content_sha256(config):
        raise ValueError("adaptive review config_sha256 does not authenticate its config")
    implementation_bundle = _validate_adaptive_implementation_bundle(
        adaptive_identity["implementation_bundle"]
    )
    if (
        implementation_bundle["files"].get("adaptive_hierarchical_stage1_reconsideration.py")
        != adaptive_identity["implementation_file_sha256"]
    ):
        raise ValueError("adaptive primary implementation differs from its dependency bundle")
    selector_thinking_token_budget = _positive_int(
        config.get("selector_thinking_token_budget"),
        label="adaptive selector_thinking_token_budget",
    )
    _validate_adaptive_prompt_contract(
        adaptive_identity["prompt_contract"],
        expected_selector_thinking_token_budget=selector_thinking_token_budget,
    )
    phase_policy = _mapping(adaptive_identity["phase_policy"], label="adaptive review phase policy")
    if set(phase_policy) != _ADAPTIVE_PHASE_POLICY_KEYS or any(
        phase_policy[key] is not True for key in _ADAPTIVE_PHASE_POLICY_KEYS
    ):
        raise ValueError("adaptive review phase policy does not close every required boundary")
    try:
        adaptive_config = AdaptiveReconsiderationConfig.from_mapping(config)
    except (TypeError, ValueError) as exc:
        raise ValueError("adaptive review config is invalid") from exc
    current_identity = adaptive_hierarchical_stage1_reconsideration_identity(config=adaptive_config)
    if canonical_json(adaptive_identity) != canonical_json(current_identity):
        raise ValueError(
            "adaptive reconsideration identity differs from the current implementation/config"
        )
    return policy


@dataclass(frozen=True)
class AuthenticatedPromptFile:
    """A local prompt artifact whose exact bytes have a known digest."""

    path: str | os.PathLike[str]
    expected_sha256: str
    display_name: str
    max_bytes: int = MAX_AUTHENTICATED_COMPARISON_PROMPT_BYTES

    def __post_init__(self) -> None:
        raw_path = os.fspath(self.path)
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("prompt artifact path must be non-empty")
        _require_sha256(self.expected_sha256, label="prompt artifact expected_sha256")
        _nonempty_string(self.display_name, label="prompt artifact display_name")
        _positive_int(self.max_bytes, label="prompt artifact max_bytes")

    def snapshot(self, *, artifact_kind: str) -> dict[str, Any]:
        """Read one stable regular file without following a final symlink."""

        _nonempty_string(artifact_kind, label="artifact_kind")
        path = Path(os.path.abspath(os.fspath(self.path)))
        try:
            before = path.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"prompt artifact does not exist: {path}") from exc
        if stat.S_ISLNK(before.st_mode):
            raise ValueError("prompt artifact cannot be a symlink")
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("prompt artifact must be a regular file")
        if before.st_size > self.max_bytes:
            raise ValueError("prompt artifact exceeds its authenticated byte bound")
        payload = path.read_bytes()
        after = path.lstat()
        stable_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if not stable_identity or len(payload) != before.st_size:
            raise ValueError("prompt artifact changed while it was being read")
        observed_sha256 = _sha256_bytes(payload)
        if observed_sha256 != self.expected_sha256:
            raise ValueError("prompt artifact bytes differ from expected_sha256")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("model-facing prompt artifact must be valid UTF-8") from exc
        return {
            "status": "supplied_and_authenticated",
            "artifact_kind": artifact_kind,
            "display_name": self.display_name,
            "source_path": str(path),
            "sha256": observed_sha256,
            "byte_count": len(payload),
            "encoding": "base64_of_exact_source_bytes",
            "bytes_base64": base64.b64encode(payload).decode("ascii"),
            "utf8_text": text,
            "no_normalization_or_truncation": True,
        }


def build_offline_extraction_definition_prompt_preview(
    *,
    canonical_name: str,
    evidence: Sequence[DiscoveryEvidenceItem],
    supporting_evidence_ids: Sequence[str],
    value_shape_hypothesis: str = "ambiguous",
    allowed_aliases: Sequence[str] = (),
    allowed_units: Sequence[str] = (),
    allowed_categories: Sequence[str] = (),
    allowed_distinguish_from: Sequence[str] = (),
) -> DiscoveryJsonJob:
    """Build a non-executable preview with the exact production prompt renderer.

    The caller supplies a review-only candidate name and exact evidence objects
    from one prepared fold.  The composer later proves that those evidence
    bytes occur in the selected batch fold.  This helper does not claim that
    the preview candidate was discovered or accepted.
    """

    items = tuple(evidence)
    support = tuple(supporting_evidence_ids)
    if not items:
        raise ValueError("extraction preview evidence cannot be empty")
    if not all(isinstance(item, DiscoveryEvidenceItem) for item in items):
        raise TypeError("extraction preview evidence must contain DiscoveryEvidenceItem entries")
    for item in items:
        item.__post_init__()
    by_id = {item.evidence_id: item for item in items}
    if len(by_id) != len(items):
        raise ValueError("extraction preview evidence IDs cannot repeat")
    if not support or len(support) != len(set(support)) or set(support) != set(by_id):
        raise ValueError(
            "extraction preview support IDs must exactly equal the supplied evidence IDs"
        )
    request = ExtractionDefinitionRequest(
        canonical_name=canonical_name,
        evidence=items,
        supporting_evidence_ids=support,
        value_shape_hypothesis=value_shape_hypothesis,
        allowed_aliases=tuple(allowed_aliases),
        allowed_units=tuple(allowed_units),
        allowed_categories=tuple(allowed_categories),
        allowed_distinguish_from=tuple(allowed_distinguish_from),
    )
    request.__post_init__()
    routing = route_concept_roles(evidence=items, supporting_evidence_ids=support)
    vocabulary_policy = extraction_vocabulary_grounding_policy()
    return DiscoveryJsonJob.create(
        job_kind=EXTRACTION_DEFINITION_JOB,
        scope=f"offline.review.preview.{canonical_name}",
        dependencies=(),
        settings=DiscoveryJobSettings.extraction(),
        messages=_render_extraction_messages(request=request),
        input_bindings={
            "offline_review_preview": True,
            "preview_not_valid_for_execution": True,
            "preview_candidate_was_discovered_or_accepted": False,
            "supporting_evidence_ids": list(support),
            "supporting_evidence_sha256": content_sha256([item.as_prompt_item() for item in items]),
            "value_shape_hypothesis": value_shape_hypothesis,
            "vocabulary_grounding_policy": vocabulary_policy,
            "vocabulary_grounding_policy_sha256": content_sha256(vocabulary_policy),
            "deterministic_role_routing": routing.audit(),
            "role_routing_sha256": content_sha256(routing.audit()),
            "production_extraction_prompt_renderer_file_sha256": _module_file_sha256(
                _render_extraction_messages
            ),
        },
    )


def _missing_optional_prompt(*, artifact_kind: str) -> dict[str, Any]:
    return {
        "status": "not_supplied",
        "artifact_kind": artifact_kind,
        "required_before_remote_prompt_quality_comparison": True,
        "content_invented": False,
    }


def _message_payload(messages: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(messages) != 2:
        raise ValueError("reviewed discovery jobs require exactly two messages")
    if [row.get("role") for row in messages] != ["system", "user"]:
        raise ValueError("reviewed discovery jobs require system then user messages")
    for index, row in enumerate(messages):
        if set(row) != {"role", "content"}:
            raise ValueError(f"messages[{index}] has an unexpected schema")
        _nonempty_string(row["content"], label=f"messages[{index}].content")
    try:
        payload = json.loads(str(messages[1]["content"]))
    except json.JSONDecodeError as exc:
        raise ValueError("reviewed user message must be one JSON object") from exc
    return _mapping(payload, label="reviewed user-message payload")


def _scan_model_visible_value(value: Any, *, path: str, observed_keys: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string JSON key")
            normalized = key.casefold()
            observed_keys.add(normalized)
            if normalized in _MACHINE_EXACT_KEYS or normalized.endswith("_sha256"):
                raise ValueError(f"model-facing prompt exposes machine field {path}.{key}")
            if _FORBIDDEN_EVIDENCE_KEY.search(normalized):
                raise ValueError(
                    f"model-facing prompt exposes forbidden evidence field {path}.{key}"
                )
            _scan_model_visible_value(child, path=f"{path}.{key}", observed_keys=observed_keys)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_model_visible_value(
                child,
                path=f"{path}[{index}]",
                observed_keys=observed_keys,
            )


def _audit_model_visible_messages(messages: Sequence[Mapping[str, Any]]) -> set[str]:
    payload = _message_payload(messages)
    combined = "\n".join(str(row["content"]).casefold() for row in messages)
    matched_machine = [token for token in _MACHINE_TEXT_TOKENS if token in combined]
    if matched_machine:
        raise ValueError(f"model-facing prompt exposes machine text: {matched_machine}")
    matched_policy = [token for token in _POLICY_TEXT_TOKENS if token in combined]
    if matched_policy:
        raise ValueError(f"model-facing prompt exposes forbidden policy text: {matched_policy}")
    observed_keys: set[str] = set()
    _scan_model_visible_value(payload, path="user_message", observed_keys=observed_keys)
    return observed_keys


def _validate_message_envelope(
    job: Mapping[str, Any],
    *,
    max_rendered_prompt_bytes: int,
) -> dict[str, Any]:
    configured_limit = _positive_int(
        max_rendered_prompt_bytes,
        label="max_rendered_prompt_bytes",
    )
    messages = job.get("messages")
    if not isinstance(messages, list):
        raise TypeError("discovery job messages must be a JSON list")
    visible_keys = _audit_model_visible_messages(messages)
    bindings = _mapping(job.get("input_bindings"), label="discovery job input_bindings")
    envelope = _mapping(
        bindings.get(AUTHENTICATED_MESSAGE_ENVELOPE_BINDING),
        label="authenticated message envelope",
    )
    required = {
        "schema_version",
        "serialization",
        "sha256",
        "byte_count",
        "byte_limit_binding",
    }
    if set(envelope) != required:
        raise ValueError("authenticated message envelope has an unexpected closed schema")
    rendered = canonical_json(messages).encode("utf-8")
    if envelope["serialization"] != "canonical_json_utf8_message_array_v1":
        raise ValueError("authenticated message serialization is unsupported")
    if envelope["sha256"] != content_sha256(messages):
        raise ValueError("message envelope SHA-256 differs from exact messages")
    if envelope["byte_count"] != len(rendered):
        raise ValueError("message envelope byte count differs from exact messages")
    if envelope["byte_limit_binding"] != (
        "content_addressed_orchestrator_runtime_config_v1"
    ):
        raise ValueError("message envelope changed its runtime byte-limit binding")
    if len(rendered) > configured_limit:
        raise ValueError("reviewed model-facing prompt exceeds its configured byte guard")
    return {
        "messages_sha256": envelope["sha256"],
        "rendered_message_array_byte_count": len(rendered),
        "configured_max_rendered_prompt_bytes": configured_limit,
        "headroom_bytes": configured_limit - len(rendered),
        "system_content_utf8_bytes": len(messages[0]["content"].encode("utf-8")),
        "user_content_utf8_bytes": len(messages[1]["content"].encode("utf-8")),
        "within_configured_guard": True,
        "model_visible_json_keys": sorted(visible_keys),
    }


def _validate_job_dict(
    value: Any,
    *,
    expected_kind: str | None = None,
    expected_selector_thinking_token_budget: int,
    max_rendered_prompt_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    job = dict(_clone(_mapping(value, label="discovery job")))
    required = {
        "job_id",
        "schema_version",
        "job_kind",
        "scope",
        "dependencies",
        "settings",
        "messages",
        "input_bindings",
    }
    if set(job) != required:
        raise ValueError("discovery job has an unexpected closed schema")
    if not isinstance(job["job_id"], str) or _JOB_ID.fullmatch(job["job_id"]) is None:
        raise ValueError("discovery job_id is not content addressed")
    identity = {key: row for key, row in job.items() if key != "job_id"}
    if job["job_id"] != f"job_{content_sha256(identity)}":
        raise ValueError("discovery job_id does not authenticate the complete job")
    if expected_kind is not None and job["job_kind"] != expected_kind:
        raise ValueError(f"reviewed job must have kind {expected_kind}")
    dependencies = job["dependencies"]
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) or _JOB_ID.fullmatch(item) is None for item in dependencies
    ):
        raise ValueError("discovery job dependencies are invalid")
    if len(dependencies) != len(set(dependencies)):
        raise ValueError("discovery job dependencies contain duplicates")
    settings = _mapping(job["settings"], label="discovery job settings")
    if set(settings) != {"thinking_enabled", "thinking_token_budget", "response_format"}:
        raise ValueError("discovery job settings have an unexpected closed schema")
    if settings["response_format"] != "json":
        raise ValueError("discovery job response format must be JSON")
    if job["job_kind"] == EXTRACTION_DEFINITION_JOB:
        if settings["thinking_enabled"] is not False or settings["thinking_token_budget"] != 0:
            raise ValueError("extraction-definition reasoning must be disabled")
    else:
        if settings["thinking_enabled"] is not True or (
            settings["thinking_token_budget"]
            != expected_selector_thinking_token_budget
        ):
            raise ValueError(
                "selector reasoning differs from its authenticated hierarchy config"
            )
    audit = _validate_message_envelope(
        job,
        max_rendered_prompt_bytes=max_rendered_prompt_bytes,
    )
    return job, audit


def _validate_initial_ledger(value: Any) -> tuple[list[dict[str, Any]], str]:
    ledger = _mapping(value, label="initial discovery job ledger")
    if set(ledger) != {"schema_version", "jobs", "ledger_sha256"}:
        raise ValueError("initial discovery job ledger has an unexpected closed schema")
    if ledger["schema_version"] != DISCOVERY_JOB_LEDGER_VERSION:
        raise ValueError("initial discovery job ledger version is unsupported")
    jobs = ledger["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("initial discovery job ledger cannot be empty")
    identity = {"schema_version": ledger["schema_version"], "jobs": jobs}
    if ledger["ledger_sha256"] != content_sha256(identity):
        raise ValueError("initial discovery ledger SHA-256 does not authenticate its jobs")
    return jobs, _require_sha256(ledger["ledger_sha256"], label="initial ledger_sha256")


def _validated_prompt_member_ids(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be one non-empty list")
    member_ids = tuple(
        _nonempty_string(member_id, label=f"{label}[{index}]")
        for index, member_id in enumerate(value)
    )
    if len(member_ids) != len(set(member_ids)):
        raise ValueError(f"{label} cannot contain duplicates")
    return member_ids


def _fold_jobs_and_bindings(
    batch_precommit: ApprovedHierarchicalDiscoveryBatchPrecommit,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    if not isinstance(batch_precommit, ApprovedHierarchicalDiscoveryBatchPrecommit):
        raise TypeError("batch_precommit must be ApprovedHierarchicalDiscoveryBatchPrecommit")
    batch_precommit.__post_init__()
    packet = batch_precommit.packet
    if content_sha256(packet) != batch_precommit.approval_sha256:
        raise ValueError("batch approval SHA-256 differs from its complete packet")
    _require_sha256(packet.get("input_manifest_sha256"), label="input_manifest_sha256")
    ordered_folds = packet.get("ordered_folds")
    if not isinstance(ordered_folds, list) or not ordered_folds:
        raise ValueError("batch packet must contain ordered fold packets")
    expected_order = list(range(1, len(ordered_folds) + 1))
    if packet.get("ordered_outer_folds") != expected_order:
        raise ValueError("batch outer folds are not complete and one-based")

    by_fold: dict[int, list[dict[str, Any]]] = {}
    fold_identities: list[dict[str, Any]] = []
    internal_root = _clone(packet)
    for expected_fold, row_value in zip(expected_order, ordered_folds):
        row = _mapping(row_value, label=f"ordered_folds[{expected_fold - 1}]")
        if row.get("outer_fold") != expected_fold:
            raise ValueError("batch fold packets are missing, duplicated, or out of order")
        wrapper = _mapping(row.get("wrapper_packet"), label="fold wrapper packet")
        wrapper_sha256 = _require_sha256(
            row.get("wrapper_approval_sha256"), label="wrapper_approval_sha256"
        )
        if content_sha256(wrapper) != wrapper_sha256:
            raise ValueError("fold wrapper approval does not authenticate its complete packet")
        hierarchy = _mapping(wrapper.get("hierarchy_precommit"), label="hierarchy_precommit")
        inner = _mapping(hierarchy.get("packet"), label="hierarchy precommit packet")
        inner_sha256 = _require_sha256(
            hierarchy.get("precommit_sha256"), label="hierarchy precommit_sha256"
        )
        if content_sha256(inner) != inner_sha256:
            raise ValueError("hierarchy precommit SHA-256 differs from its packet")
        current_repair_policy = discovery_response_repair_policy_identity()
        if canonical_json(inner.get("response_repair_policy")) != canonical_json(
            current_repair_policy
        ):
            raise ValueError(
                "hierarchy precommit does not bind the exact current response-repair policy"
            )
        inner_assurances = _mapping(
            inner.get("assurances"), label="hierarchy response-repair assurances"
        )
        if inner_assurances.get("bounded_response_repair_implemented") is not True:
            raise ValueError("hierarchy precommit lacks bounded response-repair assurance")
        if inner_assurances.get("unvalidated_response_cache_write_allowed") is not False:
            raise ValueError("hierarchy precommit permits an unvalidated response cache write")
        jobs, ledger_sha256 = _validate_initial_ledger(inner.get("initial_job_ledger"))
        catalog = _mapping(wrapper.get("catalog_binding"), label="catalog_binding")
        if catalog.get("outer_fold") != expected_fold:
            raise ValueError("fold wrapper catalog cites a different outer fold")
        inner_catalog = _mapping(inner.get("catalog_binding"), label="inner catalog_binding")
        for key in (
            "catalog_sha256",
            "split_fingerprint",
            "outer_fold",
            "scope",
            "inner_fold",
            "atom_count",
        ):
            if inner_catalog.get(key) != catalog.get(key):
                raise ValueError(f"wrapper and hierarchy catalog bindings differ at {key}")
        family_counts = _mapping(
            catalog.get("family_atom_counts"), label="catalog family_atom_counts"
        )
        # Canonical JSON sorts object keys, so architecture order is carried by
        # the ordered job ledger/downstream contract rather than this count map.
        if set(family_counts) != set(ACTIVE_STAGE1_CONCEPT_FAMILIES):
            raise ValueError("catalog does not bind exactly all ten architectures")
        expected_atom_count = sum(
            _positive_int(family_counts[family], label=f"{family} atom count")
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        )
        if catalog.get("atom_count") != expected_atom_count:
            raise ValueError("catalog atom count differs from its architecture counts")

        chunk = _mapping(wrapper.get("chunk_plan_binding"), label="chunk_plan_binding")
        inner_chunk = _mapping(inner.get("chunk_plan_binding"), label="inner chunk-plan binding")
        if inner_chunk.get("plan_sha256") != chunk.get("plan_sha256") or inner_chunk.get(
            "chunk_count"
        ) != chunk.get("chunk_count"):
            raise ValueError("wrapper and hierarchy chunk-plan bindings differ")
        architecture_chunk_limits = {
            key: _positive_int(chunk.get(key), label=key)
            for key in (
                "max_atoms_per_chunk",
                "max_bytes_per_chunk",
                "max_semantic_member_ids_per_chunk",
            )
        }
        semantic_member_cap = architecture_chunk_limits["max_semantic_member_ids_per_chunk"]
        if inner_chunk.get("max_semantic_member_ids_per_chunk") != semantic_member_cap:
            raise ValueError("wrapper and hierarchy semantic-member chunk bounds differ")
        wrapper_config = _mapping(wrapper.get("config_bounds"), label="wrapper config bounds")
        if canonical_json(inner.get("config")) != canonical_json(wrapper_config):
            raise ValueError("wrapper and hierarchy configuration identities differ")
        if wrapper_config.get("max_semantic_member_ids_per_chunk") != semantic_member_cap:
            raise ValueError("chunk plan and hierarchy config semantic-member bounds differ")
        max_rendered_prompt_bytes = _positive_int(
            wrapper_config.get("max_rendered_prompt_bytes"),
            label="hierarchy max_rendered_prompt_bytes",
        )
        selector_thinking_token_budget = _positive_int(
            wrapper_config.get("selector_thinking_token_budget"),
            label="hierarchy selector_thinking_token_budget",
        )

        seen_evidence_ids: set[str] = set()
        seen_semantic_member_ids: set[str] = set()
        observed_evidence_ids_by_family: dict[str, list[str]] = {
            family: [] for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        }
        observed_counts = {family: 0 for family in ACTIVE_STAGE1_CONCEPT_FAMILIES}
        family_sequence: list[str] = []
        validated_jobs: list[dict[str, Any]] = []
        for raw_job in jobs:
            job, _ = _validate_job_dict(
                raw_job,
                expected_kind=INTERPRET_CHUNK_JOB,
                expected_selector_thinking_token_budget=(
                    selector_thinking_token_budget
                ),
                max_rendered_prompt_bytes=max_rendered_prompt_bytes,
            )
            if job["dependencies"]:
                raise ValueError(
                    "initial architecture interpretation jobs cannot have dependencies"
                )
            bindings = _mapping(job["input_bindings"], label="initial job input_bindings")
            family = bindings.get("source_family")
            if family not in observed_counts:
                raise ValueError("initial interpretation job cites an inactive architecture")
            payload = _message_payload(job["messages"])
            evidence = payload.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise ValueError("initial interpretation prompt contains no real evidence")
            payload_families = {
                _mapping(item, label="prompt evidence item").get("source_family")
                for item in evidence
            }
            if payload_families != {family}:
                raise ValueError("one initial interpretation prompt mixes architectures")
            if payload.get("family_explanation") is None:
                raise ValueError("initial prompt omits its architecture explanation")
            if not family_sequence or family_sequence[-1] != family:
                family_sequence.append(family)
            job_semantic_member_count = 0
            for item_value in evidence:
                item = _mapping(item_value, label="prompt evidence item")
                evidence_id = _nonempty_string(item.get("evidence_id"), label="evidence_id")
                if evidence_id in seen_evidence_ids:
                    raise ValueError("one fold delivers an evidence atom more than once")
                seen_evidence_ids.add(evidence_id)
                observed_counts[family] += 1
                observed_evidence_ids_by_family[family].append(evidence_id)
                member_ids = _validated_prompt_member_ids(
                    item.get("member_ids"),
                    label=f"initial job {job['job_id']} member_ids",
                )
                overlap = seen_semantic_member_ids.intersection(member_ids)
                if overlap:
                    raise ValueError("one fold delivers a semantic member ID more than once")
                seen_semantic_member_ids.update(member_ids)
                job_semantic_member_count += len(member_ids)
            if job_semantic_member_count > semantic_member_cap:
                raise ValueError(
                    "initial interpretation prompt exceeds " "max_semantic_member_ids_per_chunk"
                )
            validated_jobs.append(job)
        if tuple(family_sequence) != ACTIVE_STAGE1_CONCEPT_FAMILIES:
            raise ValueError("initial jobs are not architecture-at-a-time in canonical order")
        if observed_counts != dict(family_counts):
            raise ValueError("initial prompts do not losslessly match catalog architecture counts")
        if len(seen_evidence_ids) != expected_atom_count:
            raise ValueError("initial prompt delivery lost catalog evidence")
        if chunk.get("chunk_count") != len(validated_jobs):
            raise ValueError("chunk-plan count differs from initial interpretation jobs")
        delivery = _mapping(inner_chunk.get("delivery_audit"), label="delivery audit")
        if delivery.get("all_catalog_atoms_delivered_exactly_once") is not True:
            raise ValueError("batch does not assure lossless initial evidence delivery")
        if delivery.get("all_catalog_semantic_member_ids_delivered_exactly_once") is not True:
            raise ValueError("batch does not assure exact-once semantic-member delivery")
        if delivery.get("max_semantic_member_ids_per_chunk") != semantic_member_cap:
            raise ValueError("delivery audit cites a different semantic-member chunk bound")
        if delivery.get("observed_semantic_member_id_delivery_count") != len(
            seen_semantic_member_ids
        ):
            raise ValueError("delivery audit semantic-member count differs from initial prompts")
        if delivery.get("catalog_semantic_member_id_count") != len(seen_semantic_member_ids):
            raise ValueError("initial prompts do not preserve every catalog semantic member")
        observed_max_members = delivery.get("observed_max_semantic_member_ids_per_chunk")
        if (
            isinstance(observed_max_members, bool)
            or not isinstance(observed_max_members, int)
            or observed_max_members < 1
            or observed_max_members > semantic_member_cap
        ):
            raise ValueError("delivery audit has an invalid observed semantic-member maximum")
        if delivery.get("non_grounding_numerical_summaries_delivered") is not False:
            raise ValueError("direct numerical summaries entered concept-bearing prompts")

        downstream = _mapping(inner.get("downstream_contract"), label="downstream contract")
        if downstream.get("architecture_order") != list(ACTIVE_STAGE1_CONCEPT_FAMILIES):
            raise ValueError("hierarchy downstream contract changed canonical architecture order")
        if downstream.get("role_routing") != (
            "deterministic_observable_axis_rules_after_integration"
        ):
            raise ValueError("hierarchy downstream contract changed deterministic role routing")

        runner_identity = _validate_content_addressed_identity(
            wrapper.get("runner_identity"), label="runner identity"
        )
        if canonical_json(inner.get("runner_identity")) != canonical_json(runner_identity):
            raise ValueError("wrapper and hierarchy runner identities differ")
        if (
            wrapper_config.get("selector_thinking_enabled") is not True
            or wrapper_config.get("selector_thinking_token_budget")
            != selector_thinking_token_budget
        ):
            raise ValueError("hierarchy configuration changed selector reasoning")
        if wrapper_config.get("extraction_definition_thinking_enabled") is not False or (
            wrapper_config.get("extraction_definition_thinking_token_budget") != 0
        ):
            raise ValueError("hierarchy configuration changed extraction reasoning")
        compiler_binding = _validate_component_binding(
            wrapper.get("compiler_binding"), label="compiler binding"
        )

        has_modern_contract = "direct_numerical_contract_binding" in wrapper
        has_legacy_manifest = "direct_numerical_manifest_binding" in wrapper
        if has_modern_contract == has_legacy_manifest:
            raise ValueError(
                "fold wrapper must contain exactly one direct numerical contract binding"
            )
        (
            direct_contract,
            direct_contract_kind,
            direct_contract_sha256,
            legacy_contract_binding,
        ) = _direct_numerical_contract_identity(
            wrapper.get(
                "direct_numerical_contract_binding",
                wrapper.get("direct_numerical_manifest_binding"),
            )
        )
        if direct_contract.get("semantic_catalog_sha256") != catalog.get("catalog_sha256"):
            raise ValueError("direct numerical contract binds a different semantic catalog")
        if (
            direct_contract.get("row_values_included") is not False
            or direct_contract.get("matrix_metadata_included") is not False
            or direct_contract.get("coordinate_metadata_included") is not False
            or direct_contract.get("coordinate_to_semantic_atom_linkage", False) is not False
        ):
            raise ValueError("unsafe direct numerical content entered the offline wrapper")
        if direct_contract.get("concept_grounding_allowed") is not False:
            raise ValueError("direct numerical contract permits concept grounding")
        numerical_families = direct_contract.get("families")
        if not isinstance(numerical_families, list) or [
            _mapping(value, label="direct numerical family").get("source_family")
            for value in numerical_families
        ] != list(ACTIVE_STAGE1_CONCEPT_FAMILIES):
            raise ValueError("direct numerical contract does not bind all ten architectures")
        if not legacy_contract_binding:
            validated_numerical_families = _validate_modern_direct_numerical_contract(
                direct_contract,
                kind=direct_contract_kind,
                contract_sha256=direct_contract_sha256,
                catalog_sha256=catalog["catalog_sha256"],
                expected_evidence_ids_by_family=observed_evidence_ids_by_family,
            )
            _validate_modern_hierarchy_numerical_bindings(
                wrapper=wrapper,
                inner=inner,
                kind=direct_contract_kind,
                contract_sha256=direct_contract_sha256,
                numerical_families=validated_numerical_families,
            )

        cache_binding = _mapping(wrapper.get("job_cache_binding"), label="job_cache_binding")
        if cache_binding.get("mode") != "authenticated_immutable":
            raise ValueError(
                "offline review requires an authenticated immutable job-cache namespace per fold"
            )
        cache_component = _validate_component_binding(
            cache_binding, label="job cache component binding"
        )
        cache_identity = _mapping(cache_binding.get("identity"), label="cache identity")
        outer_cache_identity_sha256 = _require_sha256(
            cache_binding.get("identity_sha256"), label="cache component identity_sha256"
        )
        if outer_cache_identity_sha256 != content_sha256(cache_identity):
            raise ValueError("cache component binding does not authenticate its identity")
        declared_cache_sha256 = _require_sha256(
            cache_identity.get("identity_sha256"), label="cache identity_sha256"
        )
        cache_body = {
            key: value for key, value in cache_identity.items() if key != "identity_sha256"
        }
        if declared_cache_sha256 != content_sha256(cache_body):
            raise ValueError("cache identity_sha256 does not authenticate its closed identity")
        if cache_identity.get("mode") != "read_write_immutable":
            raise ValueError("job cache is not bound to immutable read/write mode")
        cache_config = _mapping(cache_identity.get("config"), label="cache config")
        if cache_config.get("write_policy") != "exclusive_create_never_overwrite":
            raise ValueError("job cache does not use exclusive immutable writes")
        root = _mapping(cache_identity.get("root_envelope"), label="cache root envelope")
        cache_path = root.get("absolute_path")
        if (
            root.get("kind") != "machine_local_absolute_path"
            or not isinstance(cache_path, str)
            or not Path(cache_path).is_absolute()
        ):
            raise ValueError("job cache does not bind one absolute machine-local namespace")

        wrapper_assurances = _mapping(wrapper.get("assurances"), label="wrapper assurances")
        for assurance in (
            "all_active_architectures_bound",
            "all_catalog_atoms_delivered_exactly_once",
            "cache_hits_authenticated_in_final_result",
        ):
            if wrapper_assurances.get(assurance) is not True:
                raise ValueError(f"fold wrapper lacks required assurance: {assurance}")
        contract_assurances = (
            (
                "direct_manifest_authenticated_locally_in_full",
                "final_dossiers_revalidated_against_full_manifest",
            )
            if legacy_contract_binding
            else (
                "direct_numerical_contract_authenticated_locally_in_full",
                "final_dossiers_revalidated_against_approved_contract",
            )
        )
        for assurance in contract_assurances:
            if wrapper_assurances.get(assurance) is not True:
                raise ValueError(f"fold wrapper lacks required assurance: {assurance}")
        if not legacy_contract_binding:
            if wrapper_assurances.get("direct_numerical_contract_kind") != (direct_contract_kind):
                raise ValueError("fold assurance changed the direct numerical contract kind")
            expected_materialized = (
                direct_contract_kind == DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
            )
            if wrapper_assurances.get("direct_numerical_contract_materialized") is not (
                expected_materialized
            ):
                raise ValueError("fold assurance changed numerical materialization state")
        for forbidden in (
            "direct_row_level_numerical_values_in_packet",
            "direct_coordinate_metadata_in_packet",
            "unapproved_remote_execution_allowed",
            "cache_lookup_before_wrapper_approval_allowed",
            "cache_write_before_semantic_validation_allowed",
        ):
            if wrapper_assurances.get(forbidden) is not False:
                raise ValueError(f"fold wrapper does not close forbidden behavior: {forbidden}")

        expected_fold_summary = {
            "split_fingerprint_sha256": catalog.get("split_fingerprint"),
            "catalog_sha256": catalog.get("catalog_sha256"),
            "chunk_plan_sha256": chunk.get("plan_sha256"),
            (
                "direct_numerical_manifest_sha256"
                if legacy_contract_binding
                else "direct_numerical_contract_sha256"
            ): direct_contract_sha256,
            "hierarchy_precommit_sha256": inner_sha256,
        }
        if legacy_contract_binding:
            if {
                "direct_numerical_contract_kind",
                "direct_numerical_contract_sha256",
            } & set(row):
                raise ValueError("legacy fold summary contains modern numerical contract fields")
        else:
            if "direct_numerical_manifest_sha256" in row:
                raise ValueError("modern fold summary mislabels its contract as a manifest")
            if row.get("direct_numerical_contract_kind") != direct_contract_kind:
                raise ValueError("batch fold summary changed direct numerical contract kind")
        for key, expected_value in expected_fold_summary.items():
            if row.get(key) != expected_value:
                raise ValueError(f"batch fold summary differs from wrapper at {key}")

        by_fold[expected_fold] = validated_jobs
        fold_identities.append(
            {
                "outer_fold": expected_fold,
                "split_fingerprint_sha256": _require_sha256(
                    row.get("split_fingerprint_sha256"), label="fold split fingerprint"
                ),
                "catalog_sha256": _require_sha256(
                    row.get("catalog_sha256"), label="fold catalog_sha256"
                ),
                "chunk_plan_sha256": _require_sha256(
                    row.get("chunk_plan_sha256"), label="fold chunk_plan_sha256"
                ),
                "direct_numerical_contract_kind": direct_contract_kind,
                "direct_numerical_contract_sha256": direct_contract_sha256,
                "legacy_direct_numerical_contract_binding": legacy_contract_binding,
                "hierarchy_precommit_sha256": inner_sha256,
                "initial_job_ledger_sha256": ledger_sha256,
                "wrapper_approval_sha256": wrapper_sha256,
                "runner_identity": runner_identity,
                "compiler_binding": compiler_binding,
                "job_cache_binding": cache_component,
                "hierarchy_config_identity": _clone(wrapper_config),
                "max_semantic_member_ids_per_chunk": semantic_member_cap,
                "observed_semantic_member_id_count": len(seen_semantic_member_ids),
                "architecture_chunk_limits": architecture_chunk_limits,
            }
        )

    policy = _validate_phased_review_policy(packet.get("frozen_review_evidence_policy"))
    adaptive_config = _mapping(
        _mapping(
            policy.get("adaptive_reconsideration_identity"),
            label="adaptive reconsideration identity",
        ).get("config"),
        label="adaptive reconsideration config",
    )
    adaptive_chunk_limits = {
        key: adaptive_config.get(key)
        for key in (
            "max_atoms_per_chunk",
            "max_bytes_per_chunk",
            "max_semantic_member_ids_per_chunk",
        )
    }
    if fold_identities[0]["architecture_chunk_limits"] != adaptive_chunk_limits:
        raise ValueError("initial and adaptive architecture chunk limits differ")

    common = _mapping(packet.get("common_bindings"), label="common batch bindings")
    first = fold_identities[0]
    legacy_modes = {row["legacy_direct_numerical_contract_binding"] for row in fold_identities}
    if len(legacy_modes) != 1:
        raise ValueError("batch mixes legacy and modern direct numerical contracts")
    expected_common = {
        "runner_identity": first["runner_identity"],
        "compiler_binding": first["compiler_binding"],
        "hierarchy_config_identity": first["hierarchy_config_identity"],
        "architecture_chunk_limits": first["architecture_chunk_limits"],
    }
    if not first["legacy_direct_numerical_contract_binding"]:
        expected_common["direct_numerical_contract_kind"] = first["direct_numerical_contract_kind"]
    for fold_row in fold_identities[1:]:
        for key, expected_value in expected_common.items():
            if canonical_json(fold_row[key]) != canonical_json(expected_value):
                raise ValueError(f"batch folds have mixed {key}")
    for key, expected_value in expected_common.items():
        if canonical_json(common.get(key)) != canonical_json(expected_value):
            raise ValueError(f"batch common binding differs from fold {key}")
        sha_key = f"{key}_sha256"
        if sha_key in common and common[sha_key] != content_sha256(expected_value):
            raise ValueError(f"batch common {sha_key} does not authenticate {key}")
    return by_fold, fold_identities, internal_root, policy


def _evidence_catalog_from_jobs(jobs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for job in jobs:
        payload = _message_payload(job["messages"])
        for item_value in payload["evidence"]:
            item = dict(_clone(_mapping(item_value, label="prompt evidence item")))
            evidence_id = item["evidence_id"]
            if evidence_id in result:
                raise ValueError("fold prompt evidence IDs are not unique")
            result[evidence_id] = item
    return result


def _validate_extraction_preview(
    job: DiscoveryJsonJob,
    *,
    evidence_catalog: Mapping[str, Mapping[str, Any]],
    expected_selector_thinking_token_budget: int,
    max_rendered_prompt_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(job, DiscoveryJsonJob):
        raise TypeError("extraction_definition_preview must be one DiscoveryJsonJob")
    job.__post_init__()
    value, audit = _validate_job_dict(
        job.as_dict(),
        expected_kind=EXTRACTION_DEFINITION_JOB,
        expected_selector_thinking_token_budget=(
            expected_selector_thinking_token_budget
        ),
        max_rendered_prompt_bytes=max_rendered_prompt_bytes,
    )
    bindings = _mapping(value["input_bindings"], label="extraction preview input_bindings")
    if bindings.get("offline_review_preview") is not True or (
        bindings.get("preview_not_valid_for_execution") is not True
    ):
        raise ValueError(
            "extraction prompt must be explicitly bound as a non-executable offline preview"
        )
    payload = _message_payload(value["messages"])
    evidence = payload.get("evidence")
    supporting = payload.get("supporting_evidence_ids")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("extraction preview must carry real supporting evidence")
    if (
        not isinstance(supporting, list)
        or not supporting
        or len(supporting) != len(set(supporting))
    ):
        raise ValueError("extraction preview supporting evidence IDs are invalid")
    if {item.get("evidence_id") for item in evidence if isinstance(item, Mapping)} != set(
        supporting
    ):
        raise ValueError("extraction preview evidence differs from its support IDs")
    for item_value in evidence:
        item = _mapping(item_value, label="extraction preview evidence item")
        evidence_id = item.get("evidence_id")
        original = evidence_catalog.get(evidence_id)
        if original is None:
            raise ValueError("extraction preview cites evidence outside its prepared fold")
        expected_projection = {
            "evidence_id": original["evidence_id"],
            "source_family": original["source_family"],
            "member_ids": original["member_ids"],
            "content": original["content"],
        }
        if canonical_json(item) != canonical_json(expected_projection):
            raise ValueError("extraction preview changed its exact supporting raw evidence")
    return value, audit


def _representative_job(jobs: Sequence[Mapping[str, Any]], *, family: str) -> dict[str, Any]:
    if family not in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        raise ValueError("representative_family must be one active Stage-1 architecture")
    matches = [
        job
        for job in jobs
        if _mapping(job["input_bindings"], label="job input_bindings").get("source_family")
        == family
    ]
    if not matches:
        raise ValueError("prepared fold has no prompt for representative_family")
    return dict(_clone(matches[0]))


def _role_routing_review() -> dict[str, Any]:
    cases = (
        ("treatment_only", (TREATMENT_AXIS,)),
        ("outcome_only", (OUTCOME_AXIS,)),
        ("treatment_plus_outcome", (TREATMENT_AXIS, OUTCOME_AXIS)),
        ("heterogeneity", (HETEROGENEITY_AXIS,)),
        ("pair_uplift", (PAIR_UPLIFT_AXIS,)),
        ("extraction_support", (EXTRACTION_SUPPORT_AXIS,)),
    )
    audits: dict[str, Any] = {}
    for index, (label, axes) in enumerate(cases):
        item = DiscoveryEvidenceItem(
            evidence_id=f"policy.probe.{index}",
            source_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            observable_axes=axes,
            content={"policy_probe": label},
        )
        routed = route_concept_roles(evidence=(item,), supporting_evidence_ids=(item.evidence_id,))
        audits[label] = routed.audit()
    conclusions = {
        "treatment_only_is_not_confounder_adjustment": (
            audits["treatment_only"]["adjustment_roles"] == []
        ),
        "outcome_only_is_prognostic_not_confounder_adjustment": (
            audits["outcome_only"]["adjustment_roles"] == ["prognostic_adjustment"]
        ),
        "treatment_plus_outcome_routes_to_confounder_adjustment": (
            audits["treatment_plus_outcome"]["adjustment_roles"] == ["confounder_adjustment"]
        ),
        "heterogeneity_routes_to_effect_modifier": audits["heterogeneity"]["effect_modifier"],
        "pair_uplift_routes_to_effect_modifier": audits["pair_uplift"]["effect_modifier"],
        "extraction_support_does_not_create_a_causal_role": (
            audits["extraction_support"]["adjustment_roles"] == []
            and audits["extraction_support"]["effect_modifier"] is False
            and audits["extraction_support"]["extraction_definition_support"] is True
        ),
    }
    if not all(conclusions.values()):
        raise ValueError("deterministic role-routing implementation violates reviewed policy")
    source = inspect.getsource(route_concept_roles).encode("utf-8")
    return {
        "policy_version": ROLE_ROUTING_VERSION,
        "mechanism": "deterministic_observable_axis_rules_no_model_call",
        "observable_axes": list(OBSERVABLE_AXES),
        "implementation_module_file_sha256": _module_file_sha256(route_concept_roles),
        "route_function_source_sha256": _sha256_bytes(source),
        "structural_probe_results": audits,
        "reviewed_conclusions": conclusions,
        "outside_knowledge_can_invent_roles_or_features": False,
    }


def _collect_internal_machine_fields(value: Any) -> tuple[list[str], dict[str, int]]:
    counts: dict[str, int] = {}

    def visit(child: Any, *, inside_messages: bool = False) -> None:
        if isinstance(child, Mapping):
            for key, row in child.items():
                if not isinstance(key, str):
                    continue
                next_inside = inside_messages or key == "messages"
                if not next_inside and (
                    key.casefold() in _MACHINE_EXACT_KEYS
                    or key.casefold().endswith("_sha256")
                    or _INTERNAL_FIELD_HINT.search(key) is not None
                ):
                    counts[key] = counts.get(key, 0) + 1
                visit(row, inside_messages=next_inside)
        elif isinstance(child, list):
            for row in child:
                visit(row, inside_messages=inside_messages)

    visit(value)
    return sorted(counts), {key: counts[key] for key in sorted(counts)}


def _identity_summary(
    *,
    batch_precommit: ApprovedHierarchicalDiscoveryBatchPrecommit,
    fold_rows: Sequence[Any],
    phased_review_policy: Mapping[str, Any],
) -> dict[str, Any]:
    packet = batch_precommit.packet
    adaptive_identity = _mapping(
        phased_review_policy["adaptive_reconsideration_identity"],
        label="adaptive reconsideration identity",
    )
    return {
        "batch_approval_sha256": batch_precommit.approval_sha256,
        "input_manifest_sha256": packet["input_manifest_sha256"],
        "phased_review_policy_sha256": phased_review_policy["policy_sha256"],
        "round_1_review_materializer_identity": _clone(
            phased_review_policy["review_materializer_identity"]
        ),
        "adaptive_reconsideration_identity": _clone(adaptive_identity),
        "adaptive_reconsideration_config_sha256": adaptive_identity["config_sha256"],
        "adaptive_implementation_bundle": _clone(adaptive_identity["implementation_bundle"]),
        "adaptive_implementation_bundle_sha256": adaptive_identity["implementation_bundle"][
            "implementation_bundle_sha256"
        ],
        "adaptive_static_prompt_contract_sha256": adaptive_identity["prompt_contract"][
            "prompt_contract_sha256"
        ],
        "fold_precommit_identities": [
            {
                key: _clone(row[key])
                for key in (
                    "outer_fold",
                    "split_fingerprint_sha256",
                    "catalog_sha256",
                    "chunk_plan_sha256",
                    "direct_numerical_contract_kind",
                    "direct_numerical_contract_sha256",
                    "hierarchy_precommit_sha256",
                    "initial_job_ledger_sha256",
                    "wrapper_approval_sha256",
                    "runner_identity",
                    "compiler_binding",
                )
            }
            for row in fold_rows
        ],
        "cache_namespace_identities": [
            {
                "outer_fold": row["outer_fold"],
                "job_cache_binding": _clone(row["job_cache_binding"]),
            }
            for row in fold_rows
        ],
        "all_identities_are_content_addressed": True,
        "cache_entries_are_exclusive_create_never_overwrite": True,
    }


def _prompt_projection(job: Mapping[str, Any]) -> dict[str, Any]:
    payload = _message_payload(job["messages"])
    return {
        "source_job_id": job["job_id"],
        "system_message": job["messages"][0],
        "user_job_name": payload.get("job"),
        "family_explanation": payload.get("family_explanation"),
        "output_schema": _clone(payload.get("output_schema")),
        "projection_note": (
            "This section projects exact framing fields from the real prompt below; "
            "it does not reconstruct or replace the authenticated job messages."
        ),
    }


@dataclass(frozen=True)
class OfflineHierarchicalDiscoveryReviewPacket:
    """One immutable in-memory packet, ready only for offline human review."""

    packet_sha256: str
    _packet_json: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_sha256(self.packet_sha256, label="review packet_sha256")
        try:
            packet = json.loads(self._packet_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("offline review packet is invalid JSON") from exc
        if not isinstance(packet, Mapping):
            raise TypeError("offline review packet must be one JSON object")
        if content_sha256(packet) != self.packet_sha256:
            raise ValueError("packet_sha256 does not authenticate the complete review packet")

    @classmethod
    def create(cls, packet: Mapping[str, Any]) -> "OfflineHierarchicalDiscoveryReviewPacket":
        detached = _clone(packet)
        return cls(
            packet_sha256=content_sha256(detached),
            _packet_json=canonical_json(detached),
        )

    @property
    def packet(self) -> dict[str, Any]:
        return json.loads(self._packet_json)

    @property
    def approval_ready(self) -> bool:
        return self.packet["review_readiness"]["comparison_packet_complete"] is True

    def render_json(self, *, indent: int = 2) -> str:
        if isinstance(indent, bool) or not isinstance(indent, int) or indent < 0:
            raise ValueError("indent must be a non-negative integer")
        return json.dumps(
            {"packet_sha256": self.packet_sha256, "packet": self.packet},
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    def render_markdown(self) -> str:
        return _render_markdown(self)

    def persist(
        self, *, preparation_directory: str | os.PathLike[str]
    ) -> "PersistedOfflineReviewPacket":
        """Exclusively create content-addressed JSON/Markdown in a fresh directory."""

        self.__post_init__()
        raw = os.fspath(preparation_directory)
        if not isinstance(raw, str) or not raw:
            raise ValueError("preparation_directory must be one non-empty path")
        target = Path(os.path.abspath(raw))
        if target.exists() or target.is_symlink():
            raise FileExistsError("preparation directory must be fresh and absent")
        if not target.parent.exists() or not target.parent.is_dir():
            raise ValueError("preparation directory parent must already exist")

        json_bytes = canonical_json(
            {"packet_sha256": self.packet_sha256, "packet": self.packet}
        ).encode("utf-8")
        markdown_bytes = self.render_markdown().encode("utf-8")
        json_sha256 = _sha256_bytes(json_bytes)
        markdown_sha256 = _sha256_bytes(markdown_bytes)
        json_name = f"offline_review_packet_{self.packet_sha256}.json"
        markdown_name = f"offline_review_packet_{self.packet_sha256}.md"
        manifest = {
            "schema_version": OFFLINE_HIERARCHICAL_DISCOVERY_REVIEW_MANIFEST_VERSION,
            "packet_sha256": self.packet_sha256,
            "preparation_directory": str(target),
            "artifacts": [
                {
                    "kind": "canonical_json",
                    "filename": json_name,
                    "sha256": json_sha256,
                    "byte_count": len(json_bytes),
                },
                {
                    "kind": "markdown_review",
                    "filename": markdown_name,
                    "sha256": markdown_sha256,
                    "byte_count": len(markdown_bytes),
                },
            ],
            "write_policy": "fresh_directory_exclusive_files_never_overwrite",
            "final_output_touched": False,
        }
        manifest_bytes = canonical_json(manifest).encode("utf-8")
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        manifest_name = f"offline_review_manifest_{manifest_sha256}.json"

        target.mkdir(mode=0o700)
        paths_and_bytes = (
            (target / json_name, json_bytes),
            (target / markdown_name, markdown_bytes),
            (target / manifest_name, manifest_bytes),
        )
        for path, payload in paths_and_bytes:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            path.chmod(0o400)
            if path.read_bytes() != payload:
                raise IOError("persisted offline review artifact failed byte verification")
        return PersistedOfflineReviewPacket(
            preparation_directory=target,
            packet_json_path=target / json_name,
            packet_markdown_path=target / markdown_name,
            manifest_path=target / manifest_name,
            packet_sha256=self.packet_sha256,
            packet_json_sha256=json_sha256,
            packet_markdown_sha256=markdown_sha256,
            manifest_sha256=manifest_sha256,
        )


@dataclass(frozen=True)
class PersistedOfflineReviewPacket:
    preparation_directory: Path
    packet_json_path: Path
    packet_markdown_path: Path
    manifest_path: Path
    packet_sha256: str
    packet_json_sha256: str
    packet_markdown_sha256: str
    manifest_sha256: str

    def validate_authentication(self) -> None:
        for label, value in (
            ("packet_sha256", self.packet_sha256),
            ("packet_json_sha256", self.packet_json_sha256),
            ("packet_markdown_sha256", self.packet_markdown_sha256),
            ("manifest_sha256", self.manifest_sha256),
        ):
            _require_sha256(value, label=label)
        if _sha256_bytes(self.packet_json_path.read_bytes()) != self.packet_json_sha256:
            raise ValueError("persisted review JSON failed authentication")
        if _sha256_bytes(self.packet_markdown_path.read_bytes()) != (self.packet_markdown_sha256):
            raise ValueError("persisted review Markdown failed authentication")
        if _sha256_bytes(self.manifest_path.read_bytes()) != self.manifest_sha256:
            raise ValueError("persisted review manifest failed authentication")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("packet_sha256") != self.packet_sha256:
            raise ValueError("persisted manifest cites a different review packet")


def compose_offline_hierarchical_discovery_review_packet(
    *,
    batch_precommit: ApprovedHierarchicalDiscoveryBatchPrecommit,
    representative_outer_fold: int,
    representative_family: str,
    extraction_definition_preview: DiscoveryJsonJob,
    extraction_preview_outer_fold: int,
    historical_prompt: AuthenticatedPromptFile | None = None,
    old_hierarchy_prompt: AuthenticatedPromptFile | None = None,
) -> OfflineHierarchicalDiscoveryReviewPacket:
    """Revalidate and compose the complete offline-only review packet.

    The two comparison controls are optional so an incomplete preparation can
    be inspected.  Their absence is explicit and makes ``approval_ready``
    false; no placeholder prompt bytes are ever manufactured.
    """

    _positive_int(representative_outer_fold, label="representative_outer_fold")
    _positive_int(extraction_preview_outer_fold, label="extraction_preview_outer_fold")
    by_fold, fold_identities, internal_batch, phased_review_policy = _fold_jobs_and_bindings(
        batch_precommit
    )
    if representative_outer_fold not in by_fold:
        raise ValueError("representative_outer_fold is absent from the prepared batch")
    if extraction_preview_outer_fold not in by_fold:
        raise ValueError("extraction_preview_outer_fold is absent from the prepared batch")

    representative = _representative_job(
        by_fold[representative_outer_fold], family=representative_family
    )
    hierarchy_config_by_fold = {
        row["outer_fold"]: _mapping(
            row["hierarchy_config_identity"],
            label=f"fold {row['outer_fold']} hierarchy config",
        )
        for row in fold_identities
    }
    representative_config = hierarchy_config_by_fold[representative_outer_fold]
    representative_payload = _message_payload(representative["messages"])
    representative_audit = _validate_message_envelope(
        representative,
        max_rendered_prompt_bytes=_positive_int(
            representative_config.get("max_rendered_prompt_bytes"),
            label="representative max_rendered_prompt_bytes",
        ),
    )
    extraction_config = hierarchy_config_by_fold[extraction_preview_outer_fold]
    extraction_job, extraction_audit = _validate_extraction_preview(
        extraction_definition_preview,
        evidence_catalog=_evidence_catalog_from_jobs(by_fold[extraction_preview_outer_fold]),
        expected_selector_thinking_token_budget=_positive_int(
            extraction_config.get("selector_thinking_token_budget"),
            label="extraction preview selector_thinking_token_budget",
        ),
        max_rendered_prompt_bytes=_positive_int(
            extraction_config.get("max_rendered_prompt_bytes"),
            label="extraction preview max_rendered_prompt_bytes",
        ),
    )

    all_initial_audits: list[dict[str, Any]] = []
    all_visible_keys: set[str] = set()
    semantic_member_caps_by_fold = {
        row["outer_fold"]: row["max_semantic_member_ids_per_chunk"] for row in fold_identities
    }
    for outer_fold, jobs in by_fold.items():
        for ordinal, job in enumerate(jobs, start=1):
            fold_config = hierarchy_config_by_fold[outer_fold]
            audit = _validate_message_envelope(
                job,
                max_rendered_prompt_bytes=_positive_int(
                    fold_config.get("max_rendered_prompt_bytes"),
                    label=f"fold {outer_fold} max_rendered_prompt_bytes",
                ),
            )
            payload = _message_payload(job["messages"])
            semantic_member_id_count = sum(
                len(
                    _validated_prompt_member_ids(
                        _mapping(item, label="prompt evidence item").get("member_ids"),
                        label=f"initial job {job['job_id']} member_ids",
                    )
                )
                for item in payload["evidence"]
            )
            semantic_member_cap = semantic_member_caps_by_fold[outer_fold]
            if semantic_member_id_count > semantic_member_cap:
                raise ValueError(
                    "initial interpretation prompt exceeds its authenticated "
                    "semantic-member bound"
                )
            all_visible_keys.update(audit.pop("model_visible_json_keys"))
            all_initial_audits.append(
                {
                    "outer_fold": outer_fold,
                    "job_ordinal": ordinal,
                    "job_id": job["job_id"],
                    "job_kind": job["job_kind"],
                    "scope": job["scope"],
                    "source_family": _mapping(job["input_bindings"], label="job input_bindings")[
                        "source_family"
                    ],
                    "evidence_count": len(payload["evidence"]),
                    "semantic_member_id_count": semantic_member_id_count,
                    "max_semantic_member_ids_per_chunk": semantic_member_cap,
                    "semantic_member_id_headroom": (semantic_member_cap - semantic_member_id_count),
                    "settings": _clone(job["settings"]),
                    **audit,
                }
            )
    all_visible_keys.update(extraction_audit.pop("model_visible_json_keys"))
    extraction_context_audit = {
        "outer_fold": extraction_preview_outer_fold,
        "job_id": extraction_job["job_id"],
        "job_kind": extraction_job["job_kind"],
        "scope": extraction_job["scope"],
        "settings": _clone(extraction_job["settings"]),
        **extraction_audit,
    }

    historical = (
        _missing_optional_prompt(artifact_kind="byte_exact_historical_model_facing_prompt")
        if historical_prompt is None
        else historical_prompt.snapshot(artifact_kind="byte_exact_historical_model_facing_prompt")
    )
    old_hierarchy = (
        _missing_optional_prompt(artifact_kind="old_hierarchy_prompt_quality_ablation")
        if old_hierarchy_prompt is None
        else old_hierarchy_prompt.snapshot(artifact_kind="old_hierarchy_prompt_quality_ablation")
    )
    missing_controls = [
        label
        for label, row in (
            ("historical_prompt", historical),
            ("old_hierarchy_prompt", old_hierarchy),
        )
        if row["status"] != "supplied_and_authenticated"
    ]

    internal_field_names, internal_field_counts = _collect_internal_machine_fields(
        {
            "batch": internal_batch,
            "extraction_preview_job": extraction_job,
        }
    )
    role_review = _role_routing_review()
    context_rows = [*all_initial_audits, extraction_context_audit]
    largest = max(context_rows, key=lambda row: row["rendered_message_array_byte_count"])
    batch_packet = batch_precommit.packet
    batch_assurances = _mapping(batch_packet.get("assurances"), label="batch assurances")
    required_batch_true = (
        "all_fold_wrapper_packets_included_in_full",
        "outer_folds_unique_complete_and_one_based",
        "all_fold_static_preflights_before_first_cache_lookup",
        "all_fold_static_preflights_before_first_remote_call",
        "wrong_or_missing_batch_approval_rejected_before_preflight",
        "round_1_frozen_review_evidence_is_accepted_support_only",
        "later_round_feature_rediscovery_uses_fresh_exact_spent_catalog",
        "later_round_all_ten_architectures_required",
        "later_round_architecture_at_a_time_interpretation_required",
        "later_round_compact_ten_dossier_planner_required",
        "later_round_bounded_requested_id_lookback_only",
        "later_round_executable_definition_uses_requested_atoms_only",
        "later_round_proposal_frozen_before_next_gate",
        "adaptive_reconsideration_identity_authenticated",
        "ordered_batch_results_content_authenticated",
    )
    if any(batch_assurances.get(key) is not True for key in required_batch_true):
        raise ValueError("batch packet lacks one or more required run-level assurances")
    if batch_assurances.get("architecture_wide_review_evidence_dump_allowed") is not False:
        raise ValueError("batch permits an architecture-wide post-review evidence dump")
    if batch_assurances.get("same_frozen_review_evidence_used_for_every_round") is not False:
        raise ValueError("batch reuses frozen round-1 evidence for later adaptive rounds")

    adaptive_identity = _mapping(
        phased_review_policy["adaptive_reconsideration_identity"],
        label="adaptive reconsideration identity",
    )

    packet = {
        "schema_version": OFFLINE_HIERARCHICAL_DISCOVERY_REVIEW_PACKET_VERSION,
        "composer_identity": {
            "module": "oci.inference.offline_hierarchical_discovery_review_packet",
            "implementation_file_sha256": _implementation_file_sha256(),
            "operation": "offline_read_validate_render_and_preparation_only_write",
            "network_or_model_runner_call_supported": False,
        },
        "review_scope": {
            "purpose": "human_review_before_any_new_remote_discovery_comparison",
            "remote_execution_authorized": False,
            "batch_execute_called": False,
            "job_cache_lookup_performed": False,
            "final_output_touched": False,
        },
        "run_level_batch_precommit": {
            "approval_sha256": batch_precommit.approval_sha256,
            "packet": batch_packet,
        },
        "phased_adaptive_review_policy": {
            "policy_sha256": phased_review_policy["policy_sha256"],
            "round_1": {
                "evidence_scope": (
                    "frozen_exact_supporting_evidence_of_initially_accepted_features"
                ),
                "feature_rediscovery_allowed": False,
                "raw_evidence_id_bound": phased_review_policy["max_evidence_ids"],
                "raw_evidence_byte_bound": phased_review_policy["max_evidence_bytes"],
                "review_materializer_identity": _clone(
                    phased_review_policy["review_materializer_identity"]
                ),
            },
            "rounds_2_and_later": {
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
            },
            "adaptive_reconsideration_identity": _clone(adaptive_identity),
            "adaptive_implementation_bundle": _clone(adaptive_identity["implementation_bundle"]),
            "later_round_static_prompt_contract": _clone(adaptive_identity["prompt_contract"]),
            "implementation_and_config_authenticated_against_current_source": True,
            "prompt_contract_authenticated_against_current_production_templates": True,
            "implementation_bundle_authenticated_against_current_dependencies": True,
        },
        "comparison_artifacts": {
            "historical_prompt": historical,
            "old_hierarchy_prompt": old_hierarchy,
        },
        "proposed_plain_language_discovery_prompt": _prompt_projection(representative),
        "representative_real_family_prompt": {
            "outer_fold": representative_outer_fold,
            "source_family": representative_family,
            "job_id": representative["job_id"],
            "scope": representative["scope"],
            "settings": _clone(representative["settings"]),
            "family_explanation": representative_payload["family_explanation"],
            "evidence_count": len(representative_payload["evidence"]),
            "semantic_member_id_count": sum(
                len(
                    _validated_prompt_member_ids(
                        _mapping(row, label="representative prompt evidence item").get(
                            "member_ids"
                        ),
                        label="representative prompt member_ids",
                    )
                )
                for row in representative_payload["evidence"]
            ),
            "max_semantic_member_ids_per_chunk": semantic_member_caps_by_fold[
                representative_outer_fold
            ],
            "evidence_ids": [row["evidence_id"] for row in representative_payload["evidence"]],
            "exact_messages": _clone(representative["messages"]),
            "messages_sha256": representative_audit["messages_sha256"],
            "rendered_message_array_byte_count": representative_audit[
                "rendered_message_array_byte_count"
            ],
        },
        "role_routing_policy_review": role_review,
        "extraction_definition_prompt": {
            "preview_only": True,
            "not_valid_for_execution_or_cache_replay": True,
            "outer_fold": extraction_preview_outer_fold,
            "job_id": extraction_job["job_id"],
            "scope": extraction_job["scope"],
            "settings": _clone(extraction_job["settings"]),
            "exact_messages": _clone(extraction_job["messages"]),
            "messages_sha256": extraction_context_audit["messages_sha256"],
            "rendered_message_array_byte_count": extraction_context_audit[
                "rendered_message_array_byte_count"
            ],
            "model_hidden_input_bindings": _clone(extraction_job["input_bindings"]),
        },
        "model_hidden_field_audit": {
            "machine_exact_keys_forbidden_in_model_messages": sorted(_MACHINE_EXACT_KEYS),
            "machine_sha256_suffix_forbidden_in_model_messages": True,
            "forbidden_evidence_key_pattern": _FORBIDDEN_EVIDENCE_KEY.pattern,
            "forbidden_temporal_policy_text": list(_POLICY_TEXT_TOKENS),
            "internal_machine_field_names": internal_field_names,
            "internal_machine_field_occurrence_counts": internal_field_counts,
            "model_visible_json_key_names": sorted(all_visible_keys),
            "internal_machine_field_leaks_detected": [],
            "raw_row_values_visible": False,
            "direct_coordinate_metadata_visible": False,
            "patient_ids_or_full_notes_visible": False,
            "oracle_or_fresh_validation_fields_visible": False,
            "future_gate_text_or_labels_visible": False,
            "role_routing_audit_visible_to_extraction_model": False,
            "all_reviewed_messages_passed": True,
        },
        "context_size_audit": {
            "guard_serialization": "canonical_json_utf8_message_array_v1",
            "configured_max_rendered_prompt_bytes": _positive_int(
                representative_config.get("max_rendered_prompt_bytes"),
                label="review packet max_rendered_prompt_bytes",
            ),
            "initial_job_audits": all_initial_audits,
            "extraction_preview_job_audit": extraction_context_audit,
            "audited_job_count": len(context_rows),
            "largest_reviewed_job": {
                "job_id": largest["job_id"],
                "rendered_message_array_byte_count": largest["rendered_message_array_byte_count"],
                "headroom_bytes": largest["headroom_bytes"],
            },
            "every_reviewed_job_within_configured_guard": True,
            "every_initial_job_within_semantic_member_bound": True,
            "semantic_repair_prompt_implemented": True,
            "response_repair_policy": discovery_response_repair_policy_identity(),
            "repair_context_policy": (
                "one_cumulative_system_user_assistant_user_sequence_rechecked_against_guard"
            ),
            "repair_diagnostic_policy": (
                "fixed_category_only_no_exception_text_no_model_identifiers"
            ),
            "repair_cache_policy": (
                "only_validated_final_response_with_exact_message_sequence_hash_trace"
            ),
            "transport_retry_context_policy": (
                "same_exact_authenticated_message_sequence_and_byte_count_on_every_attempt"
            ),
            "selector_reasoning_tokens": _positive_int(
                representative_config.get("selector_thinking_token_budget"),
                label="review packet selector_thinking_token_budget",
            ),
            "extraction_reasoning_enabled": False,
        },
        "adaptive_static_prompt_contract_audit": {
            "prompt_contract_sha256": adaptive_identity["prompt_contract"][
                "prompt_contract_sha256"
            ],
            "stage_order": list(_ADAPTIVE_PROMPT_STAGE_ORDER),
            "stage_count": len(_ADAPTIVE_PROMPT_STAGE_ORDER),
            "selector_stage_count": len(_ADAPTIVE_PROMPT_STAGE_ORDER) - 1,
            "selector_reasoning_tokens": _positive_int(
                _mapping(
                    adaptive_identity["config"],
                    label="adaptive prompt audit config",
                ).get("selector_thinking_token_budget"),
                label="adaptive prompt audit selector_thinking_token_budget",
            ),
            "extraction_definition_reasoning_enabled": False,
            "dynamic_fold_content_included": False,
            "future_gate_text_or_labels_included": False,
            "complete_raw_catalog_single_prompt_authorized": False,
            "exact_static_system_instructions_and_output_schemas_in_packet": True,
            "authenticated_against_current_production_templates": True,
        },
        "adaptive_implementation_bundle_audit": {
            "implementation_bundle_sha256": adaptive_identity["implementation_bundle"][
                "implementation_bundle_sha256"
            ],
            "dependency_files": _clone(adaptive_identity["implementation_bundle"]["files"]),
            "dependency_file_count": len(adaptive_identity["implementation_bundle"]["files"]),
            "primary_module_hash_matches_bundle": True,
            "authenticated_against_current_dependency_bytes": True,
        },
        "provenance_and_honesty_assurances": {
            "all_ten_stage1_architectures_present_per_fold": True,
            "architecture_at_a_time_initial_prompts": True,
            "every_catalog_atom_delivered_exactly_once_per_fold": True,
            "no_global_top_k_before_initial_discovery": True,
            "direct_numerical_values_do_not_ground_feature_names": True,
            "all_fold_precommits_exist_before_first_cache_or_remote_call": True,
            "wrong_or_missing_batch_approval_allows_no_execution": True,
            "selector_reasoning_is_exactly_5000": True,
            "extraction_reasoning_is_disabled": True,
            "round_1_post_extraction_review_is_accepted_support_only": True,
            "rounds_2_and_later_use_fresh_exact_accumulated_spent_stage1_catalog": True,
            "rounds_2_and_later_allow_bounded_feature_rediscovery": True,
            "same_frozen_evidence_used_for_every_review_round": False,
            "all_ten_architectures_reinterpreted_separately_in_later_rounds": True,
            "one_compact_dossier_per_architecture_before_later_round_planning": True,
            "later_round_planner_raw_catalog_atom_count": 0,
            "later_round_raw_lookback_is_bounded_requested_ids_only": True,
            "complete_raw_catalog_dump_allowed": False,
            "future_gate_text_or_labels_model_visible": False,
            "proposal_freeze_before_next_gate_required": True,
            "adaptive_reconsideration_implementation_and_config_authenticated": True,
            "adaptive_reconsideration_dependency_bundle_authenticated": True,
            "later_round_static_prompt_contract_authenticated": True,
            "later_round_selector_reasoning_is_exactly_5000": True,
            "later_round_extraction_definition_reasoning_is_disabled": True,
            "later_round_dynamic_fold_content_in_static_approval_envelope": False,
            "historical_control_bytes_modified": False,
            "oracle_information_used_for_packet_composition": False,
            "remote_calls_made_by_packet_composer": False,
            "final_output_writes_made_by_packet_composer": False,
            "assurance_basis": {
                "batch_assurances": _clone(batch_assurances),
                "fold_count": len(by_fold),
                "canonical_architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
                "role_routing_policy_version": ROLE_ROUTING_VERSION,
                "phased_review_policy_sha256": phased_review_policy["policy_sha256"],
                "adaptive_reconsideration_config_sha256": adaptive_identity["config_sha256"],
                "adaptive_implementation_bundle_sha256": adaptive_identity["implementation_bundle"][
                    "implementation_bundle_sha256"
                ],
                "adaptive_static_prompt_contract_sha256": adaptive_identity["prompt_contract"][
                    "prompt_contract_sha256"
                ],
            },
        },
        "immutable_precommit_and_cache_identities": _identity_summary(
            batch_precommit=batch_precommit,
            fold_rows=fold_identities,
            phased_review_policy=phased_review_policy,
        ),
        "review_readiness": {
            "mandatory_authenticated_sections_complete": True,
            "phased_adaptive_review_policy_authenticated": True,
            "round_1_frozen_then_later_exact_spent_hierarchy_ready": True,
            "proposal_freeze_precedes_next_gate": True,
            "adaptive_static_prompt_contract_authenticated_and_reviewable": True,
            "adaptive_implementation_bundle_authenticated_and_reviewable": True,
            "missing_optional_comparison_artifacts": missing_controls,
            "comparison_packet_complete": not missing_controls,
            "remote_execution_authorized": False,
            "required_next_action": (
                "supply_and_authenticate_missing_comparison_prompts"
                if missing_controls
                else "show_packet_to_user_and_wait_for_explicit_approval"
            ),
        },
    }
    return OfflineHierarchicalDiscoveryReviewPacket.create(packet)


def _markdown_fence(value: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{value}\n{fence}"


def _render_exact_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for index, message in enumerate(messages, start=1):
        rows.extend(
            [
                f"Message {index} (`{message['role']}`):",
                "",
                _markdown_fence(str(message["content"])),
                "",
            ]
        )
    return "\n".join(rows).rstrip()


def _render_comparison_artifact(title: str, artifact: Mapping[str, Any]) -> list[str]:
    if artifact["status"] != "supplied_and_authenticated":
        return [f"### {title}", "", "Not supplied; no placeholder content was invented."]
    return [
        f"### {title}",
        "",
        f"SHA-256: `{artifact['sha256']}`; bytes: `{artifact['byte_count']}`.",
        "",
        (
            "The JSON packet carries base64 of the exact source bytes. The block below is the "
            "strict UTF-8 review rendering of those same authenticated bytes."
        ),
        "",
        _markdown_fence(str(artifact["utf8_text"])),
    ]


def _render_markdown(packet: OfflineHierarchicalDiscoveryReviewPacket) -> str:
    body = packet.packet
    comparison = body["comparison_artifacts"]
    real = body["representative_real_family_prompt"]
    extraction = body["extraction_definition_prompt"]
    role = body["role_routing_policy_review"]
    hidden = body["model_hidden_field_audit"]
    contexts = body["context_size_audit"]
    identities = body["immutable_precommit_and_cache_identities"]
    phased = body["phased_adaptive_review_policy"]
    implementation_bundle = phased["adaptive_implementation_bundle"]
    prompt_contract = phased["later_round_static_prompt_contract"]
    readiness = body["review_readiness"]
    lines = [
        "# Offline hierarchical-discovery approval packet",
        "",
        f"Packet SHA-256: `{packet.packet_sha256}`",
        "",
        "This is an offline review artifact. It authorizes no remote execution.",
        "",
        "## Review readiness",
        "",
        f"Comparison packet complete: `{str(readiness['comparison_packet_complete']).lower()}`.",
        "",
    ]
    if readiness["missing_optional_comparison_artifacts"]:
        lines.extend(
            [
                "Missing comparison artifacts: "
                + ", ".join(readiness["missing_optional_comparison_artifacts"])
                + ".",
                "",
            ]
        )
    lines.extend(
        [
            "## Phased post-extraction review policy",
            "",
            (
                "Round 1 is limited to the frozen exact supporting evidence of initially "
                "accepted features. Rounds 2 and later rebuild a fresh authenticated catalog "
                "from exact accumulated-spent Stage-1 evidence."
            ),
            "",
            (
                "Each later round interprets all ten architectures separately, consolidates "
                "and audits coverage within each architecture, and gives the planner exactly "
                "ten compact dossiers plus the current registry and sanitized diagnostics."
            ),
            "",
            (
                "Raw evidence is available only through bounded planner-requested IDs. A "
                "complete raw catalog dump, future gate text or labels, and reuse of the "
                "round-1 frozen catalog for every round are forbidden. The proposal is frozen "
                "before the next gate."
            ),
            "",
            f"Policy SHA-256: `{phased['policy_sha256']}`.",
            "",
            (
                "Adaptive implementation SHA-256: "
                f"`{phased['adaptive_reconsideration_identity']['implementation_file_sha256']}`; "
                "config SHA-256: "
                f"`{phased['adaptive_reconsideration_identity']['config_sha256']}`."
            ),
            "",
        ]
    )
    lines.extend(
        [
            "### Authenticated adaptive implementation bundle",
            "",
            (
                "The adaptive cache validator and approval identity bind the primary module "
                "together with every local renderer, validator, compiler, and evidence-catalog "
                "dependency listed below."
            ),
            "",
            (
                "Implementation-bundle SHA-256: "
                f"`{implementation_bundle['implementation_bundle_sha256']}`."
            ),
            "",
        ]
    )
    for filename, digest in implementation_bundle["files"].items():
        lines.append(f"- `{filename}`: `{digest}`")
    lines.append("")
    lines.extend(
        [
            "### Exact static authorization envelope for later-round model calls",
            "",
            (
                "Dynamic fold evidence, registries, diagnostics, requested lookback atoms, "
                "and future-gate content are intentionally absent. The exact static system "
                "instructions, settings, dynamic-input slots, and output shapes follow."
            ),
            "",
            f"Prompt-contract SHA-256: `{prompt_contract['prompt_contract_sha256']}`.",
            "",
        ]
    )
    for stage in prompt_contract["stages"]:
        settings = stage["settings"]
        lines.extend(
            [
                f"#### `{stage['stage']}`",
                "",
                (
                    f"Template `{stage['template_version']}`; thinking enabled "
                    f"`{str(settings['thinking_enabled']).lower()}`; token budget "
                    f"`{settings['thinking_token_budget']}`; response format "
                    f"`{settings['response_format']}`."
                ),
                "",
                "Exact system instruction:",
                "",
                _markdown_fence(str(stage["system_instruction"])),
                "",
                "Dynamic input slots (not populated in this offline packet):",
                "",
                _markdown_fence(canonical_json(stage["dynamic_inputs"])),
                "",
                "Exact user-payload top-level keys:",
                "",
                _markdown_fence(canonical_json(stage["user_payload_top_level_keys"])),
                "",
                "Exact static user-payload literals:",
                "",
                _markdown_fence(canonical_json(stage["static_user_payload_literals"])),
                "",
                "Dynamic user-payload paths (not populated in this offline packet):",
                "",
                _markdown_fence(canonical_json(stage["dynamic_user_payload_paths"])),
                "",
                "Exact dynamic payload shapes:",
                "",
                _markdown_fence(canonical_json(stage["dynamic_payload_shapes"])),
                "",
                "Exact output shape:",
                "",
                _markdown_fence(canonical_json(stage["output_schema"])),
                "",
            ]
        )
    lines.extend(["## Prompt-quality comparison artifacts", ""])
    lines.extend(
        _render_comparison_artifact(
            "Byte-exact historical model-facing prompt",
            comparison["historical_prompt"],
        )
    )
    lines.extend([""])
    lines.extend(
        _render_comparison_artifact(
            "Old hierarchy prompt (ablation only)",
            comparison["old_hierarchy_prompt"],
        )
    )
    lines.extend(
        [
            "",
            "## Proposed plain-language discovery framing",
            "",
            _render_exact_messages(
                [body["proposed_plain_language_discovery_prompt"]["system_message"]]
            ),
            "",
            "The output schema and family explanation are exact projections from the real job below; the JSON packet retains both fields and the complete authenticated job.",
            "",
            "## Real architecture-local family prompt",
            "",
            f"Outer fold `{real['outer_fold']}`, family `{real['source_family']}`, job `{real['job_id']}`.",
            "",
            f"Rendered canonical message-array bytes: `{real['rendered_message_array_byte_count']}`; SHA-256: `{real['messages_sha256']}`.",
            "",
            _render_exact_messages(real["exact_messages"]),
            "",
            "## Deterministic role-routing review",
            "",
            f"Policy: `{role['policy_version']}`. No language-model call assigns roles.",
            "",
        ]
    )
    for key, value in role["reviewed_conclusions"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(
        [
            "",
            "## Extraction-definition prompt preview",
            "",
            "This evidence-backed prompt is explicitly preview-only and is not a selected feature or executable discovery result.",
            "",
            f"Reasoning enabled: `{str(extraction['settings']['thinking_enabled']).lower()}`; budget: `{extraction['settings']['thinking_token_budget']}`.",
            "",
            _render_exact_messages(extraction["exact_messages"]),
            "",
            "## Fields hidden from model visibility",
            "",
            "Every reviewed proposed message passed the closed machine-field, sensitive-evidence, and temporal-policy scans.",
            "",
            "Internal machine field names present only in authenticated envelopes:",
            "",
        ]
    )
    for name in hidden["internal_machine_field_names"]:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "Prohibited value classes include row-level numerical values, direct-coordinate metadata, patient identifiers, full notes, oracle fields, and fresh validation fields.",
            "",
            "## Exact context-size audit",
            "",
            "Configured guard: "
            f"`{contexts['configured_max_rendered_prompt_bytes']}` canonical UTF-8 "
            "message-array bytes.",
            "",
            "| Fold | Job | Family | Members | Member cap | Bytes | Byte headroom | Thinking |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in contexts["initial_job_audits"]:
        lines.append(
            "| "
            f"{row['outer_fold']} | `{row['job_id']}` | `{row['source_family']}` | "
            f"{row['semantic_member_id_count']} | "
            f"{row['max_semantic_member_ids_per_chunk']} | "
            f"{row['rendered_message_array_byte_count']} | {row['headroom_bytes']} | "
            f"{row['settings']['thinking_token_budget']} |"
        )
    preview = contexts["extraction_preview_job_audit"]
    lines.append(
        "| "
        f"{preview['outer_fold']} | `{preview['job_id']}` | extraction preview | "
        "n/a | n/a | "
        f"{preview['rendered_message_array_byte_count']} | {preview['headroom_bytes']} | 0 |"
    )
    lines.extend(
        [
            "",
            "Transport retries reuse the same exact authenticated message sequence. After a strict parse or semantic-validation failure, exactly one separately authenticated cumulative repair sequence may run; its fixed sanitized instruction and complete message-sequence hash are bound to the result and immutable cache entry.",
            "",
            "## Provenance, honesty, and immutable identities",
            "",
            f"Batch approval SHA-256: `{identities['batch_approval_sha256']}`.",
            "",
            f"Input manifest SHA-256: `{identities['input_manifest_sha256']}`.",
            "",
            (
                "Phased review policy SHA-256: "
                f"`{identities['phased_review_policy_sha256']}`; adaptive config SHA-256: "
                f"`{identities['adaptive_reconsideration_config_sha256']}`; implementation "
                "bundle SHA-256: "
                f"`{identities['adaptive_implementation_bundle_sha256']}`; static prompt "
                "contract SHA-256: "
                f"`{identities['adaptive_static_prompt_contract_sha256']}`."
            ),
            "",
        ]
    )
    for row in identities["fold_precommit_identities"]:
        lines.append(
            f"- Fold {row['outer_fold']}: wrapper `{row['wrapper_approval_sha256']}`, "
            f"hierarchy `{row['hierarchy_precommit_sha256']}`, catalog `{row['catalog_sha256']}`."
        )
    lines.extend(["", "Authenticated immutable cache namespaces:", ""])
    for row in identities["cache_namespace_identities"]:
        root = row["job_cache_binding"]["identity"]["root_envelope"]["absolute_path"]
        digest = row["job_cache_binding"]["identity"]["identity_sha256"]
        lines.append(f"- Fold {row['outer_fold']}: `{root}` (`{digest}`).")
    lines.extend(
        [
            "",
            "All ten active architectures are delivered one at a time in every fold, with every catalog atom delivered exactly once before cross-architecture integration. The same architecture-local hierarchy is required again over each later round's fresh exact-spent catalog; no prompt receives a complete raw multi-architecture dump. Direct numerical values remain non-grounding. The composer made no cache lookup, model call, oracle read, or final-output write.",
            "",
            "## Approval boundary",
            "",
            "Reviewing this packet does not execute it. Any later remote comparison still requires the exact run-level batch approval SHA-256 through the separate execution boundary.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "MAX_AUTHENTICATED_COMPARISON_PROMPT_BYTES",
    "OFFLINE_HIERARCHICAL_DISCOVERY_REVIEW_MANIFEST_VERSION",
    "OFFLINE_HIERARCHICAL_DISCOVERY_REVIEW_PACKET_VERSION",
    "AuthenticatedPromptFile",
    "OfflineHierarchicalDiscoveryReviewPacket",
    "PersistedOfflineReviewPacket",
    "build_offline_extraction_definition_prompt_preview",
    "compose_offline_hierarchical_discovery_review_packet",
]
