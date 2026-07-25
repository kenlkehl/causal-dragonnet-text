"""Offline-first orchestration for hierarchical all-architecture discovery.

This module composes the strict interfaces in
``all_evidence_discovery_interfaces`` with the lossless architecture chunks in
``lossless_stage1_evidence_catalog``.  It deliberately has no model client or
network implementation.  A caller may inject a JSON job runner only after
reviewing and approving the content-addressed offline precommit packet.

The workflow keeps raw concept-bearing evidence architecture-local until a
cross-architecture planner explicitly requests a bounded set of exact evidence
IDs.  Row-level numerical values are never accepted by this API.  Only a
manifest digest, signal count, and explicit zero reason are bound into each
architecture dossier.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
    DISCOVERY_INTERFACE_SCHEMA_VERSION,
    DISCOVERY_WIRE_NORMALIZATION_VERSION,
    CONSOLIDATE_JOB_VERSION,
    COVERAGE_CRITIC_JOB_VERSION,
    CROSS_ARCHITECTURE_INTEGRATION_JOB_VERSION,
    CROSS_ARCHITECTURE_PLANNER_JOB_VERSION,
    DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST,
    DIRECT_NUMERICAL_CONTRACT_KINDS,
    DIRECT_UPSTREAM_NUMERICAL_CHANNEL,
    EXTRACTION_DEFINITION_JOB_VERSION,
    INTERPRET_JOB_VERSION,
    REJECTION_CRITIC_JOB_VERSION,
    ArchitectureDossier,
    DiscoveryCandidate,
    DiscoveryEvidenceItem,
    ExtractionDefinitionRequest,
    RoleRoutingResult,
    bounded_candidate_relation_pages,
    candidate_definition_fold_batches,
    canonical_json,
    compile_complete_link_candidate_groups,
    consolidate_candidate_context,
    content_sha256,
    cross_architecture_planner_context,
    extraction_vocabulary_grounding_policy,
    interpretation_model_view,
    render_interpret_evidence_chunk_messages,
    route_concept_roles,
    validate_consolidation_response,
    validate_candidate_relation_page_response,
    validate_coverage_critic_response,
    validate_extraction_definition_response,
    validate_interpret_evidence_chunk_response,
    validate_rejection_critic_response,
)
from .lossless_stage1_evidence_catalog import (
    DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK,
    ArchitectureChunkPlan,
    ArchitectureEvidenceChunk,
    RoleNeutralEvidenceCatalog,
    audit_complete_architecture_delivery,
    validate_role_neutral_catalog,
)
from .hierarchical_discovery_job_cache import (
    AuthenticatedHierarchicalDiscoveryJobCache,
)
from .hierarchical_discovery_response_contract import (
    HIERARCHICAL_DISCOVERY_EXACT_COVERAGE_REPRESENTATION,
    HIERARCHICAL_DISCOVERY_RESPONSE_CONTRACT_VERSION,
    HierarchyWireBudget,
    LEGACY_HIERARCHY_WIRE_BUDGET,
    attach_hierarchical_discovery_response_contract,
    build_hierarchical_discovery_response_contract,
)

HIERARCHICAL_DISCOVERY_ORCHESTRATOR_VERSION = (
    "hierarchical_all_architecture_discovery_orchestrator_v12"
)
HIERARCHICAL_DISCOVERY_PRECOMMIT_VERSION = "hierarchical_discovery_precommit_v11"
DISCOVERY_JSON_JOB_VERSION = "hierarchical_discovery_json_job_v7"
DISCOVERY_JOB_LEDGER_VERSION = "hierarchical_discovery_job_ledger_v1"
DISCOVERY_EXECUTION_LEDGER_VERSION = "hierarchical_discovery_execution_ledger_v6"
COMPLETED_HIERARCHICAL_DISCOVERY_VERSION = "completed_hierarchical_discovery_v7"

DISCOVERY_RESPONSE_REPAIR_POLICY_VERSION = "authenticated_bounded_hierarchy_response_repair_v7"
DISCOVERY_RESPONSE_ATTEMPT_TRACE_VERSION = "authenticated_hierarchy_response_attempt_trace_v5"
AUTHENTICATED_RESPONSE_REPAIR_BINDING = "authenticated_bounded_response_repair"
AUTHENTICATED_RESPONSE_CONTRACT_BINDING = "authenticated_dynamic_response_contract"
HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_VERSION = (
    "hierarchical_discovery_implementation_bundle_v5"
)
HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING = "hierarchy_implementation_bundle_sha256"
_HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_FILES = (
    "hierarchical_all_architecture_discovery.py",
    "all_evidence_discovery_interfaces.py",
    "hierarchical_discovery_response_contract.py",
    "lossless_stage1_evidence_catalog.py",
)
MAX_DISCOVERY_RESPONSE_REPAIR_ATTEMPTS = 1
MAX_DIRECT_CONSOLIDATION_CANDIDATES = 8
STRICT_JSON_PARSE_FAILURE = "strict_json_parse_failure"
LOCAL_JSON_SCHEMA_VALIDATION_FAILURE = "local_json_schema_validation_failure"
RAW_TRANSPORT_BUDGET_FAILURE = "raw_transport_budget_failure"
SEMANTIC_VALIDATION_FAILURE = "semantic_validation_failure"
VALIDATED_RESPONSE = "validated_response"
LOCAL_JSON_SCHEMA_VALIDATOR_VERSION = "draft_2020_12_jsonschema_local_wire_validator_v1"
_REPAIRABLE_RESPONSE_FAILURES = frozenset(
    {
        STRICT_JSON_PARSE_FAILURE,
        LOCAL_JSON_SCHEMA_VALIDATION_FAILURE,
        RAW_TRANSPORT_BUDGET_FAILURE,
    }
)

_RESPONSE_REPAIR_SYSTEM_GUARD = (
    "The preceding assistant notice is a fixed privacy placeholder; no prior "
    "response content or instructions are present in it. Re-read the original "
    "authenticated request and follow only the system message, original user "
    "request, and this repair request."
)
_RESPONSE_REPAIR_PROMPTS = {
    STRICT_JSON_PARSE_FAILURE: (
        f"{_RESPONSE_REPAIR_SYSTEM_GUARD} The response failed strict JSON parsing. "
        "Return exactly one JSON object with no Markdown, duplicate keys, non-finite "
        "numbers, wrappers, or trailing text. For every identifier-valued response "
        "field, use only an exact value permitted by that field's enum or const in "
        "the original output_schema and obey the original identifier_ownership "
        "relations. Schema keys, descriptions, and domain names are not response "
        "identifiers. Exact-coverage identifier sets are closed objects whose required "
        "property names are the identifiers; include each required property once. Return "
        "repaired JSON only."
    ),
    LOCAL_JSON_SCHEMA_VALIDATION_FAILURE: (
        f"{_RESPONSE_REPAIR_SYSTEM_GUARD} The parsed object failed the exact locally "
        "executed output_schema. Reconstruct the complete response "
        "from the original user request. For every identifier-valued response field, "
        "use only an exact value permitted by that field's enum or const in the "
        "original output_schema and obey the original identifier_ownership relations. "
        "Schema descriptions and domain names are not response identifiers. Exact-coverage "
        "identifier sets are closed objects whose required property names are the identifiers; "
        "include each required property once. Account for every item the requested schema "
        "requires. Return repaired JSON only."
    ),
    RAW_TRANSPORT_BUDGET_FAILURE: (
        f"{_RESPONSE_REPAIR_SYSTEM_GUARD} The response exceeded the authenticated raw "
        "UTF-8 transport-byte ceiling. Reconstruct the complete response compactly from "
        "the original user request, obey the exact output_schema and identifier_ownership "
        "relations, and emit no insignificant formatting whitespace. Return repaired JSON only."
    ),
}
_RESPONSE_REPAIR_ASSISTANT_PLACEHOLDERS = {
    STRICT_JSON_PARSE_FAILURE: (
        "The prior response failed strict JSON parsing. Its content is intentionally "
        "omitted; reconstruct the answer from the original authenticated request."
    ),
    LOCAL_JSON_SCHEMA_VALIDATION_FAILURE: (
        "The prior response failed the exact local JSON-Schema validator. Its content is "
        "intentionally omitted; reconstruct the answer from the original authenticated "
        "request."
    ),
    RAW_TRANSPORT_BUDGET_FAILURE: (
        "The prior response exceeded the authenticated raw transport-byte ceiling. Its "
        "content is intentionally omitted; reconstruct a compact answer from the original "
        "authenticated request."
    ),
}

INTERPRET_CHUNK_JOB = "interpret_architecture_chunk"
CONSOLIDATE_ARCHITECTURE_JOB = "consolidate_architecture_candidates"
COVERAGE_CRITIC_JOB = "audit_architecture_coverage"
CROSS_ARCHITECTURE_PLANNER_JOB = "plan_cross_architecture_integration"
CROSS_ARCHITECTURE_INTEGRATION_JOB = "integrate_cross_architecture_candidates"
REJECTION_CRITIC_JOB = "audit_rejected_candidates"
EXTRACTION_DEFINITION_JOB = "define_one_extraction_feature"

EXTRACTION_DEFINITION_SYSTEM_PROMPT = (
    "Define extraction for exactly one accepted patient feature from the supplied raw-evidence "
    "page or exhaustive evidence-review accumulators. Address every authenticated fold input "
    "exactly once when input_dispositions is requested; selection is allowed only through an "
    "explicit integrated, not_selected, or conflict_preserved disposition. Clinical aliases, "
    "units, categories, and distinctions must occur literally in the authenticated raw evidence "
    "represented by those inputs. The reserved as_documented scale and "
    "not_mentioned/mentioned pair are extraction mechanics, not clinical ontology; do not mix "
    "the mechanical pair with clinical categories. Preserve the supplied value-shape hypothesis "
    "or return unresolved. Do not add causal claims or a second feature. Set "
    "supporting_evidence_reviewed true; the complete feature support relation is compiler-owned "
    "and must not be repeated. Return JSON only."
)

# Backward-compatible generic component defaults.  Production entry points
# bind both values from the required scientific Stage-2 protocol and do not
# treat either value as an engine ceiling.
SELECTOR_THINKING_TOKEN_BUDGET = 5_000
MAX_RENDERED_DISCOVERY_PROMPT_BYTES = 220_000
AUTHENTICATED_MESSAGE_ENVELOPE_BINDING = "authenticated_model_message_envelope"
_JOB_KINDS = frozenset(
    {
        INTERPRET_CHUNK_JOB,
        CONSOLIDATE_ARCHITECTURE_JOB,
        COVERAGE_CRITIC_JOB,
        CROSS_ARCHITECTURE_PLANNER_JOB,
        CROSS_ARCHITECTURE_INTEGRATION_JOB,
        REJECTION_CRITIC_JOB,
        EXTRACTION_DEFINITION_JOB,
    }
)
_SELECTOR_JOB_KINDS = _JOB_KINDS - {EXTRACTION_DEFINITION_JOB}
_JOB_INTERFACE_VERSIONS = {
    INTERPRET_CHUNK_JOB: INTERPRET_JOB_VERSION,
    CONSOLIDATE_ARCHITECTURE_JOB: CONSOLIDATE_JOB_VERSION,
    COVERAGE_CRITIC_JOB: COVERAGE_CRITIC_JOB_VERSION,
    CROSS_ARCHITECTURE_PLANNER_JOB: CROSS_ARCHITECTURE_PLANNER_JOB_VERSION,
    CROSS_ARCHITECTURE_INTEGRATION_JOB: CROSS_ARCHITECTURE_INTEGRATION_JOB_VERSION,
    REJECTION_CRITIC_JOB: REJECTION_CRITIC_JOB_VERSION,
    EXTRACTION_DEFINITION_JOB: EXTRACTION_DEFINITION_JOB_VERSION,
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_JOB_ID = re.compile(r"job_[0-9a-f]{64}\Z")
_OPAQUE = re.compile(r"[a-z][a-z0-9_.:-]*\Z")
_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
_UNSAFE_MODEL_TEXT = re.compile(r"[\x00-\x1f\x7f-\x9f\ud800-\udfff]")
_DISALLOWED_PROMPT_POLICY_TEXT = (
    "temporal_policy",
    "temporal policy",
    "current_date",
    "current date",
)
_DISALLOWED_PROMPT_MACHINE_TEXT = (
    "schema_version",
    "catalog_sha256",
    "coverage_audit_sha256",
    "manifest_sha256",
    "direct_numerical_contract_kind",
    "direct_numerical_contract_sha256",
    "split_fingerprint",
    "producer_identity",
    "producer_id",
    "cache_id",
    "cache_key",
    "deterministic_role_routing",
    "role_routing_sha256",
)


def _clone(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _sha(value: Any) -> str:
    return content_sha256(value)


def hierarchical_discovery_implementation_bundle(
    *, refresh_local_validator: bool = False
) -> dict[str, Any]:
    """Authenticate every local base-hierarchy renderer/validator dependency."""

    base = Path(__file__).resolve().parent
    files: dict[str, str] = {}
    for filename in _HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_FILES:
        path = base / filename
        if not path.is_file():
            raise ValueError(f"hierarchy implementation dependency is missing: {filename}")
        files[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    body = {
        "schema_version": HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_VERSION,
        "files": files,
        "discovery_interface_schema_version": DISCOVERY_INTERFACE_SCHEMA_VERSION,
        "wire_normalization_version": DISCOVERY_WIRE_NORMALIZATION_VERSION,
        "response_contract_version": HIERARCHICAL_DISCOVERY_RESPONSE_CONTRACT_VERSION,
        "local_json_schema_validator": local_json_schema_validator_identity(
            refresh=refresh_local_validator
        ),
        "exact_coverage_representation": (HIERARCHICAL_DISCOVERY_EXACT_COVERAGE_REPRESENTATION),
        "job_interface_versions": dict(sorted(_JOB_INTERFACE_VERSIONS.items())),
    }
    return {**body, "implementation_bundle_sha256": _sha(body)}


_LOCAL_JSON_SCHEMA_VALIDATOR_IDENTITY_SNAPSHOT: dict[str, Any] | None = None


def _compute_local_json_schema_validator_identity() -> dict[str, Any]:

    distribution_names = (
        "jsonschema",
        "jsonschema-specifications",
        "referencing",
        "rpds-py",
        "attrs",
    )
    module_names = (
        "jsonschema.validators",
        "jsonschema._keywords",
        "jsonschema._types",
        "jsonschema._utils",
        "jsonschema.exceptions",
        "referencing._core",
        "referencing.jsonschema",
        "rpds",
        "rpds.rpds",
        "attr",
        "attr._make",
    )
    module_files: dict[str, str] = {}
    for module_name in module_names:
        module_path = getattr(import_module(module_name), "__file__", None)
        if not isinstance(module_path, str) or not Path(module_path).is_file():
            raise ValueError(f"local schema validator module has no file: {module_name}")
        module_files[module_name] = hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
    return {
        "schema_version": LOCAL_JSON_SCHEMA_VALIDATOR_VERSION,
        "draft": "https://json-schema.org/draft/2020-12/schema",
        "implementation": "jsonschema.validators.Draft202012Validator",
        "distribution": "jsonschema",
        "distribution_version": distribution_version("jsonschema"),
        "dependency_versions": {name: distribution_version(name) for name in distribution_names},
        "resolved_module_file_sha256": module_files,
    }


def local_json_schema_validator_identity(*, refresh: bool = False) -> dict[str, Any]:
    """Return the installed validator snapshot, optionally rereading code bytes.

    Ordinary job compilation reuses the immutable process snapshot. Approved
    execution boundaries pass ``refresh=True`` and compare the fresh identity
    with the precommitted snapshot before any cache or transport action.
    """

    global _LOCAL_JSON_SCHEMA_VALIDATOR_IDENTITY_SNAPSHOT
    if _LOCAL_JSON_SCHEMA_VALIDATOR_IDENTITY_SNAPSHOT is None:
        _LOCAL_JSON_SCHEMA_VALIDATOR_IDENTITY_SNAPSHOT = (
            _compute_local_json_schema_validator_identity()
        )
    if refresh:
        return _clone(_compute_local_json_schema_validator_identity())
    return _clone(_LOCAL_JSON_SCHEMA_VALIDATOR_IDENTITY_SNAPSHOT)


def _exact_mapping(value: Any, *, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one JSON object")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{label} keys differ; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _exact_keyed_rows(
    value: Any,
    *,
    identifiers: Sequence[str],
    label: str,
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    ordered = tuple(identifiers)
    if len(ordered) != len(set(ordered)):
        raise ValueError(f"{label} expected identifiers cannot contain duplicates")
    mapping = _exact_mapping(value, keys=set(ordered), label=label)
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for identifier in ordered:
        row = mapping[identifier]
        if not isinstance(row, Mapping):
            raise TypeError(f"{label}.{identifier} must be one JSON object")
        rows.append((identifier, row))
    return tuple(rows)


def _string(value: Any, *, label: str, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if _UNSAFE_MODEL_TEXT.search(value) is not None:
        raise ValueError(f"{label} contains a forbidden control or surrogate code point")
    if not empty and not value.strip():
        raise ValueError(f"{label} cannot be empty")
    return value


def _identifier(value: Any, *, label: str) -> str:
    result = _string(value, label=label)
    if _OPAQUE.fullmatch(result) is None:
        raise ValueError(f"{label} must be an opaque lowercase identifier")
    return result


def _feature_name(value: Any, *, label: str) -> str:
    result = _string(value, label=label)
    if _NAME.fullmatch(result) is None:
        raise ValueError(f"{label} must be lower snake_case")
    return result


def _string_list(
    value: Any,
    *,
    label: str,
    empty: bool = False,
    identifiers: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    if not value and not empty:
        raise ValueError(f"{label} cannot be empty")
    result = tuple(
        (
            _identifier(item, label=f"{label}[{index}]")
            if identifiers
            else _string(item, label=f"{label}[{index}]")
        )
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{label} cannot contain duplicates")
    return result


def _assert_no_policy_prompt_text(messages: Sequence[Mapping[str, str]]) -> None:
    text = "\n".join(str(message.get("content", "")).casefold() for message in messages)
    matched = [token for token in _DISALLOWED_PROMPT_POLICY_TEXT if token in text]
    if matched:
        raise ValueError(f"prompt contains disallowed policy text: {matched}")


def _assert_no_machine_prompt_text(messages: Sequence[Mapping[str, str]]) -> None:
    text = "\n".join(str(message.get("content", "")).casefold() for message in messages)
    matched = [token for token in _DISALLOWED_PROMPT_MACHINE_TEXT if token in text]
    if matched:
        raise ValueError(f"prompt contains internal machine metadata: {matched}")
    forbidden_exact_keys = {
        "schema_version",
        "catalog_id",
        "cache_id",
        "cache_key",
        "manifest_id",
        "producer_id",
        "producer_identity",
        "split_fingerprint",
        "deterministic_role_routing",
    }

    def visit(value: Any, *, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} contains a non-string JSON key")
                normalized = key.casefold()
                if normalized in forbidden_exact_keys or normalized.endswith("_sha256"):
                    raise ValueError(f"prompt contains internal machine metadata key: {path}.{key}")
                visit(child, path=f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path=f"{path}[{index}]")

    for index, message in enumerate(messages):
        try:
            parsed = json.loads(str(message.get("content", "")))
        except json.JSONDecodeError:
            continue
        visit(parsed, path=f"messages[{index}].content")


def _validated_messages(values: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], ...]:
    messages = tuple(values)
    if len(messages) not in {2, 4}:
        raise ValueError(
            "discovery jobs require either the initial system/user sequence or "
            "one cumulative system/user/assistant/user repair sequence"
        )
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        row = _exact_mapping(
            message,
            keys={"role", "content"},
            label=f"messages[{index}]",
        )
        role = _string(row["role"], label=f"messages[{index}].role")
        content = _string(row["content"], label=f"messages[{index}].content")
        normalized.append({"role": role, "content": content})
    roles = tuple(row["role"] for row in normalized)
    if roles not in {("system", "user"), ("system", "user", "assistant", "user")}:
        raise ValueError(
            "discovery message order must be system/user or cumulative "
            "system/user/assistant/user repair"
        )
    # The assistant repair notice is a fixed producer-controlled privacy
    # placeholder.  All four messages can therefore retain the ordinary scans.
    _assert_no_policy_prompt_text(normalized)
    _assert_no_machine_prompt_text(normalized)
    return tuple(normalized)


def _authenticated_message_envelope(
    *, job_kind: str, messages: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    rendered = canonical_json(list(messages)).encode("utf-8")
    return {
        "schema_version": _JOB_INTERFACE_VERSIONS[job_kind],
        "serialization": "canonical_json_utf8_message_array_v1",
        "sha256": content_sha256(list(messages)),
        "byte_count": len(rendered),
        "byte_limit_binding": "content_addressed_orchestrator_runtime_config_v1",
    }


def _validated_model_response_contract(
    *,
    job_kind: str,
    messages: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute the exact dynamic contract from designated request fields."""

    try:
        request = json.loads(str(messages[1]["content"]))
    except (IndexError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("discovery user request must be one strict JSON object") from exc
    if not isinstance(request, Mapping):
        raise TypeError("discovery user request must be one JSON object")
    schema, ownership = build_hierarchical_discovery_response_contract(
        job_kind=job_kind,
        request=request,
    )
    if request.get("output_schema") != schema:
        raise ValueError("model-facing output_schema differs from its designated-field derivation")
    if request.get("identifier_ownership") != ownership:
        raise ValueError(
            "model-facing identifier_ownership differs from its designated-field derivation"
        )
    return schema, ownership


def _authenticated_response_contract(
    *,
    job_kind: str,
    messages: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    schema, ownership = _validated_model_response_contract(
        job_kind=job_kind,
        messages=messages,
    )
    schema_json = canonical_json(schema)
    ownership_json = canonical_json(ownership)
    body = {
        "schema_version": HIERARCHICAL_DISCOVERY_RESPONSE_CONTRACT_VERSION,
        "serialization": "canonical_json_utf8_v1",
        "response_schema": schema,
        "response_schema_canonical_json_utf8": schema_json,
        "response_schema_sha256": _sha(schema),
        "response_schema_byte_count": len(schema_json.encode("utf-8")),
        "identifier_ownership": ownership,
        "identifier_ownership_canonical_json_utf8": ownership_json,
        "identifier_ownership_sha256": _sha(ownership),
        "identifier_ownership_byte_count": len(ownership_json.encode("utf-8")),
        "local_json_schema_validator": local_json_schema_validator_identity(),
    }
    return {**body, "binding_sha256": _sha(body)}


def _validate_authenticated_response_contract_binding(
    *,
    job_kind: str,
    messages: Sequence[Mapping[str, str]],
    bindings: Mapping[str, Any],
) -> None:
    expected = _authenticated_response_contract(job_kind=job_kind, messages=messages)
    if bindings.get(AUTHENTICATED_RESPONSE_CONTRACT_BINDING) != expected:
        raise ValueError("input_bindings do not authenticate the exact dynamic response contract")


def discovery_response_repair_policy_identity() -> dict[str, Any]:
    """Return the closed, content-addressed one-repair policy."""

    body = {
        "schema_version": DISCOVERY_RESPONSE_REPAIR_POLICY_VERSION,
        "maximum_repair_attempts": MAX_DISCOVERY_RESPONSE_REPAIR_ATTEMPTS,
        "repairable_failure_categories": sorted(_REPAIRABLE_RESPONSE_FAILURES),
        "message_sequence": ["system", "user", "assistant", "user"],
        "prior_response_projection": {
            STRICT_JSON_PARSE_FAILURE: "sha256_of_exact_failed_transport_content_v1",
            LOCAL_JSON_SCHEMA_VALIDATION_FAILURE: (
                "sha256_of_canonical_json_schema_invalid_parsed_object_v1"
            ),
            RAW_TRANSPORT_BUDGET_FAILURE: ("sha256_of_exact_failed_transport_content_v1"),
        },
        "prior_response_content_model_visible": False,
        "prior_response_content_persisted": False,
        "implementation_bundle_binding": (
            "exact_current_hierarchy_implementation_bundle_sha256_copied_from_original_job_v1"
        ),
        "repair_assistant_placeholders": dict(_RESPONSE_REPAIR_ASSISTANT_PLACEHOLDERS),
        "diagnostic_policy": ("fixed_category_only_no_exception_text_no_model_identifiers_v1"),
        "repair_prompts": dict(_RESPONSE_REPAIR_PROMPTS),
        "context_guard_source": "content_addressed_orchestrator_runtime_config_v1",
        "selector_thinking_token_budget_source": (
            "authenticated_discovery_job_settings_v1"
        ),
        "extraction_thinking_enabled": False,
        "cache_policy": "validated_final_response_only_with_attempt_trace_v1",
    }
    return {**body, "policy_sha256": _sha(body)}


class DiscoveryResponseRepairExhausted(ValueError):
    """The single authenticated response repair did not validate."""


class DiscoveryWireSchemaValidationFailure(ValueError):
    """A parsed wire response failed its exact authenticated JSON Schema."""


class DiscoverySemanticNormalizationFailure(ValueError):
    """A schema-valid response could not be safely normalized locally."""


def _validate_local_discovery_wire_schema(*, job: "DiscoveryJsonJob", response: Any) -> None:
    """Execute the exact schema and authenticated interpret wire bound locally."""

    try:
        Draft202012Validator(job.response_schema).validate(response)
    except JsonSchemaValidationError as exc:
        raise DiscoveryWireSchemaValidationFailure(
            "discovery wire response failed its exact local JSON Schema"
        ) from exc
    if job.job_kind != INTERPRET_CHUNK_JOB:
        return
    budget = job.identifier_ownership.get("ownership", {}).get("wire_response_budget")
    if not isinstance(budget, Mapping):
        raise DiscoveryWireSchemaValidationFailure(
            "interpret response contract lacks its authenticated wire budget"
        )
    maximum_canonical_json_bytes = budget.get("maximum_canonical_json_bytes")
    if (
        isinstance(maximum_canonical_json_bytes, bool)
        or not isinstance(maximum_canonical_json_bytes, int)
        or maximum_canonical_json_bytes < 1
    ):
        raise DiscoveryWireSchemaValidationFailure(
            "interpret response contract has an invalid authenticated canonical JSON budget"
        )
    try:
        actual_wire_bytes = len(canonical_json(response).encode("utf-8"))
    except (UnicodeEncodeError, ValueError) as exc:
        raise DiscoveryWireSchemaValidationFailure(
            "discovery wire response is not valid UTF-8 model text"
        ) from exc
    if actual_wire_bytes > maximum_canonical_json_bytes:
        raise DiscoveryWireSchemaValidationFailure(
            "interpret response exceeds its authenticated canonical JSON budget"
        )


@dataclass(frozen=True)
class DiscoveryJobSettings:
    """Closed inference settings enforced by job type."""

    thinking_enabled: bool
    thinking_token_budget: int
    response_format: str = "json"

    def __post_init__(self) -> None:
        if not isinstance(self.thinking_enabled, bool):
            raise TypeError("thinking_enabled must be boolean")
        if isinstance(self.thinking_token_budget, bool) or not isinstance(
            self.thinking_token_budget, int
        ):
            raise TypeError("thinking_token_budget must be an integer")
        if self.thinking_token_budget < 0:
            raise ValueError("thinking_token_budget cannot be negative")
        if self.response_format != "json":
            raise ValueError("response_format must be json")

    @classmethod
    def selector(
        cls,
        thinking_token_budget: int = SELECTOR_THINKING_TOKEN_BUDGET,
    ) -> "DiscoveryJobSettings":
        return cls(
            thinking_enabled=True,
            thinking_token_budget=thinking_token_budget,
        )

    @classmethod
    def extraction(cls) -> "DiscoveryJobSettings":
        return cls(thinking_enabled=False, thinking_token_budget=0)

    def validate_for(self, job_kind: str) -> None:
        if job_kind in _SELECTOR_JOB_KINDS:
            if not self.thinking_enabled or self.thinking_token_budget < 1:
                raise ValueError(
                    "selector jobs require thinking enabled with a positive "
                    "authenticated token budget"
                )
        elif job_kind == EXTRACTION_DEFINITION_JOB:
            if self.thinking_enabled or self.thinking_token_budget != 0:
                raise ValueError("extraction-definition jobs require thinking disabled")
        else:
            raise ValueError(f"unknown discovery job kind: {job_kind!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "thinking_enabled": self.thinking_enabled,
            "thinking_token_budget": self.thinking_token_budget,
            "response_format": self.response_format,
        }


@dataclass(frozen=True)
class DiscoveryJsonJob:
    """One immutable, content-addressed JSON job."""

    job_id: str
    job_kind: str
    scope: str
    dependencies: tuple[str, ...]
    settings: DiscoveryJobSettings
    _messages_json: str = field(repr=False)
    _input_bindings_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if _JOB_ID.fullmatch(self.job_id) is None:
            raise ValueError("job_id must be content addressed")
        if self.job_kind not in _JOB_KINDS:
            raise ValueError("job_kind is unsupported")
        _identifier(self.scope, label="job scope")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("job dependencies cannot contain duplicates")
        if any(_JOB_ID.fullmatch(value) is None for value in self.dependencies):
            raise ValueError("job dependencies must be content-addressed job IDs")
        if not isinstance(self.settings, DiscoveryJobSettings):
            raise TypeError("settings must be DiscoveryJobSettings")
        self.settings.validate_for(self.job_kind)
        messages = _validated_messages(json.loads(self._messages_json))
        bindings = json.loads(self._input_bindings_json)
        if not isinstance(bindings, Mapping):
            raise TypeError("input_bindings must be one JSON object")
        expected_envelope = _authenticated_message_envelope(
            job_kind=self.job_kind,
            messages=messages,
        )
        if bindings.get(AUTHENTICATED_MESSAGE_ENVELOPE_BINDING) != expected_envelope:
            raise ValueError("input_bindings do not authenticate exact rendered messages")
        _validate_authenticated_response_contract_binding(
            job_kind=self.job_kind,
            messages=messages,
            bindings=bindings,
        )
        _validate_response_repair_binding(
            job_kind=self.job_kind,
            messages=messages,
            bindings=bindings,
        )
        if self.job_id != f"job_{_sha(self._identity_without_id())}":
            raise ValueError("job_id does not authenticate job content")

    @classmethod
    def create(
        cls,
        *,
        job_kind: str,
        scope: str,
        dependencies: Sequence[str],
        settings: DiscoveryJobSettings,
        messages: Sequence[Mapping[str, Any]],
        input_bindings: Mapping[str, Any],
    ) -> "DiscoveryJsonJob":
        normalized_messages = _validated_messages(messages)
        bindings = _clone(input_bindings)
        if not isinstance(bindings, Mapping):
            raise TypeError("input_bindings must be one JSON object")
        if AUTHENTICATED_MESSAGE_ENVELOPE_BINDING in bindings:
            raise ValueError(f"input_bindings reserve {AUTHENTICATED_MESSAGE_ENVELOPE_BINDING!r}")
        if AUTHENTICATED_RESPONSE_CONTRACT_BINDING in bindings:
            raise ValueError(f"input_bindings reserve {AUTHENTICATED_RESPONSE_CONTRACT_BINDING!r}")
        bindings[AUTHENTICATED_MESSAGE_ENVELOPE_BINDING] = _authenticated_message_envelope(
            job_kind=job_kind,
            messages=normalized_messages,
        )
        bindings[AUTHENTICATED_RESPONSE_CONTRACT_BINDING] = _authenticated_response_contract(
            job_kind=job_kind,
            messages=normalized_messages,
        )
        _validate_authenticated_response_contract_binding(
            job_kind=job_kind,
            messages=normalized_messages,
            bindings=bindings,
        )
        _validate_response_repair_binding(
            job_kind=job_kind,
            messages=normalized_messages,
            bindings=bindings,
        )
        dependency_tuple = tuple(dependencies)
        identity = {
            "schema_version": DISCOVERY_JSON_JOB_VERSION,
            "job_kind": job_kind,
            "scope": scope,
            "dependencies": list(dependency_tuple),
            "settings": settings.as_dict(),
            "messages": list(normalized_messages),
            "input_bindings": bindings,
        }
        return cls(
            job_id=f"job_{_sha(identity)}",
            job_kind=job_kind,
            scope=scope,
            dependencies=dependency_tuple,
            settings=settings,
            _messages_json=canonical_json(normalized_messages),
            _input_bindings_json=canonical_json(bindings),
        )

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        return tuple(json.loads(self._messages_json))

    @property
    def input_bindings(self) -> dict[str, Any]:
        return json.loads(self._input_bindings_json)

    @property
    def response_schema(self) -> dict[str, Any]:
        schema, _ownership = _validated_model_response_contract(
            job_kind=self.job_kind,
            messages=self.messages,
        )
        return _clone(schema)

    @property
    def identifier_ownership(self) -> dict[str, Any]:
        _schema, ownership = _validated_model_response_contract(
            job_kind=self.job_kind,
            messages=self.messages,
        )
        return _clone(ownership)

    @property
    def rendered_messages_bytes(self) -> bytes:
        """Return the exact canonical UTF-8 message array authenticated by the job."""

        return canonical_json(list(self.messages)).encode("utf-8")

    def _identity_without_id(self) -> dict[str, Any]:
        return {
            "schema_version": DISCOVERY_JSON_JOB_VERSION,
            "job_kind": self.job_kind,
            "scope": self.scope,
            "dependencies": list(self.dependencies),
            "settings": self.settings.as_dict(),
            "messages": list(self.messages),
            "input_bindings": self.input_bindings,
        }

    def as_dict(self) -> dict[str, Any]:
        return {"job_id": self.job_id, **self._identity_without_id()}


def _validate_response_repair_binding(
    *,
    job_kind: str,
    messages: Sequence[Mapping[str, str]],
    bindings: Mapping[str, Any],
) -> None:
    repair = bindings.get(AUTHENTICATED_RESPONSE_REPAIR_BINDING)
    if len(messages) == 2:
        if repair is not None:
            raise ValueError("an initial discovery job cannot carry a response-repair binding")
        return
    if len(messages) != 4:
        raise AssertionError("validated message sequence has an unsupported length")
    expected_bundle_sha256 = hierarchical_discovery_implementation_bundle()[
        "implementation_bundle_sha256"
    ]
    if bindings.get(HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING) != expected_bundle_sha256:
        raise ValueError(
            "response-repair job does not bind the current hierarchy implementation bundle"
        )
    row = _exact_mapping(
        repair,
        keys={
            "policy_sha256",
            "original_job_id",
            "repair_attempt_number",
            "failure_category",
            "original_messages_sha256",
            "prior_response_content_sha256",
            "assistant_placeholder_sha256",
            "repair_prompt_sha256",
        },
        label="authenticated response-repair binding",
    )
    policy = discovery_response_repair_policy_identity()
    if row["policy_sha256"] != policy["policy_sha256"]:
        raise ValueError("response-repair binding cites a different policy")
    if (
        not isinstance(row["original_job_id"], str)
        or _JOB_ID.fullmatch(row["original_job_id"]) is None
    ):
        raise ValueError("response-repair original_job_id is invalid")
    if row["repair_attempt_number"] != 1:
        raise ValueError("response repair is bounded to exactly one attempt")
    category = row["failure_category"]
    if category not in _REPAIRABLE_RESPONSE_FAILURES:
        raise ValueError("response-repair failure category is not admitted")
    for label in (
        "original_messages_sha256",
        "prior_response_content_sha256",
        "assistant_placeholder_sha256",
        "repair_prompt_sha256",
    ):
        value = row[label]
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"response-repair {label} is invalid")
    if row["original_messages_sha256"] != _sha(list(messages[:2])):
        raise ValueError("response repair changed the original authenticated messages")
    expected_placeholder = _RESPONSE_REPAIR_ASSISTANT_PLACEHOLDERS[category]
    if messages[2] != {"role": "assistant", "content": expected_placeholder}:
        raise ValueError("response repair changed the fixed privacy-preserving placeholder")
    if (
        row["assistant_placeholder_sha256"]
        != hashlib.sha256(expected_placeholder.encode("utf-8")).hexdigest()
    ):
        raise ValueError("response-repair assistant placeholder SHA-256 is invalid")
    expected_prompt = _RESPONSE_REPAIR_PROMPTS[category]
    if messages[3] != {"role": "user", "content": expected_prompt}:
        raise ValueError("response repair changed the fixed sanitized repair prompt")
    if row["repair_prompt_sha256"] != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest():
        raise ValueError("response-repair prompt SHA-256 is invalid")
    if job_kind not in _JOB_KINDS:
        raise ValueError("response-repair job kind is unsupported")


def _build_response_repair_job(
    *,
    original_job: DiscoveryJsonJob,
    prior_response_content: str,
    failure_category: str,
) -> DiscoveryJsonJob:
    if failure_category not in _REPAIRABLE_RESPONSE_FAILURES:
        raise ValueError("response failure category is not repairable")
    if not isinstance(prior_response_content, str):
        raise TypeError("prior failed response content must be a string")
    return _build_response_repair_job_from_projection_sha256(
        original_job=original_job,
        prior_response_content_sha256=hashlib.sha256(
            prior_response_content.encode("utf-8")
        ).hexdigest(),
        failure_category=failure_category,
    )


def _build_response_repair_job_from_projection_sha256(
    *,
    original_job: DiscoveryJsonJob,
    prior_response_content_sha256: str,
    failure_category: str,
) -> DiscoveryJsonJob:
    """Reconstruct the exact privacy-preserving repair job from authenticated inputs."""

    if not isinstance(original_job, DiscoveryJsonJob):
        raise TypeError("original_job must be a DiscoveryJsonJob")
    if failure_category not in _REPAIRABLE_RESPONSE_FAILURES:
        raise ValueError("response failure category is not repairable")
    if (
        not isinstance(prior_response_content_sha256, str)
        or _SHA256.fullmatch(prior_response_content_sha256) is None
    ):
        raise ValueError("prior response content SHA-256 is invalid")
    original_bundle_sha256 = original_job.input_bindings.get(
        HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING
    )
    current_bundle_sha256 = hierarchical_discovery_implementation_bundle()[
        "implementation_bundle_sha256"
    ]
    if original_bundle_sha256 != current_bundle_sha256:
        raise ValueError(
            "original discovery job does not bind the current hierarchy " "implementation bundle"
        )
    prompt = _RESPONSE_REPAIR_PROMPTS[failure_category]
    assistant_placeholder = _RESPONSE_REPAIR_ASSISTANT_PLACEHOLDERS[failure_category]
    policy = discovery_response_repair_policy_identity()
    binding = {
        "policy_sha256": policy["policy_sha256"],
        "original_job_id": original_job.job_id,
        "repair_attempt_number": 1,
        "failure_category": failure_category,
        "original_messages_sha256": _sha(list(original_job.messages)),
        "prior_response_content_sha256": prior_response_content_sha256,
        "assistant_placeholder_sha256": hashlib.sha256(
            assistant_placeholder.encode("utf-8")
        ).hexdigest(),
        "repair_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    return DiscoveryJsonJob.create(
        job_kind=original_job.job_kind,
        scope=f"{original_job.scope}.response_repair_001",
        dependencies=(),
        settings=original_job.settings,
        messages=(
            *original_job.messages,
            {"role": "assistant", "content": assistant_placeholder},
            {"role": "user", "content": prompt},
        ),
        input_bindings={
            AUTHENTICATED_RESPONSE_REPAIR_BINDING: binding,
            HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING: (original_bundle_sha256),
        },
    )


@dataclass(frozen=True)
class DiscoveryJobLedger:
    jobs: tuple[DiscoveryJsonJob, ...]
    ledger_sha256: str

    def __post_init__(self) -> None:
        if not self.jobs:
            raise ValueError("job ledger cannot be empty")
        seen: set[str] = set()
        for job in self.jobs:
            if not isinstance(job, DiscoveryJsonJob):
                raise TypeError("job ledger contains a non-job entry")
            missing = set(job.dependencies) - seen
            if missing:
                raise ValueError(f"job dependencies are absent or forward references: {missing}")
            if job.job_id in seen:
                raise ValueError("job ledger contains duplicate job IDs")
            seen.add(job.job_id)
        if _SHA256.fullmatch(self.ledger_sha256) is None:
            raise ValueError("ledger_sha256 must be a lowercase SHA-256")
        if self.ledger_sha256 != _sha(self._identity_without_sha()):
            raise ValueError("job ledger SHA-256 does not authenticate")

    @classmethod
    def build(cls, jobs: Sequence[DiscoveryJsonJob]) -> "DiscoveryJobLedger":
        values = tuple(jobs)
        identity = {
            "schema_version": DISCOVERY_JOB_LEDGER_VERSION,
            "jobs": [job.as_dict() for job in values],
        }
        return cls(jobs=values, ledger_sha256=_sha(identity))

    def _identity_without_sha(self) -> dict[str, Any]:
        return {
            "schema_version": DISCOVERY_JOB_LEDGER_VERSION,
            "jobs": [job.as_dict() for job in self.jobs],
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._identity_without_sha(), "ledger_sha256": self.ledger_sha256}


def _response_attempt_entry(
    *,
    job: DiscoveryJsonJob,
    validation_outcome: str,
    raw_response_projection_sha256: str,
    normalized_validated_response_sha256: str | None = None,
) -> dict[str, Any]:
    if validation_outcome not in {*_REPAIRABLE_RESPONSE_FAILURES, VALIDATED_RESPONSE}:
        raise ValueError("response attempt has an unsupported validation outcome")
    if (
        not isinstance(raw_response_projection_sha256, str)
        or _SHA256.fullmatch(raw_response_projection_sha256) is None
    ):
        raise ValueError("raw response projection SHA-256 is invalid")
    if validation_outcome == VALIDATED_RESPONSE:
        if (
            not isinstance(normalized_validated_response_sha256, str)
            or _SHA256.fullmatch(normalized_validated_response_sha256) is None
        ):
            raise ValueError("validated response attempt requires a normalized SHA-256")
    elif normalized_validated_response_sha256 is not None:
        raise ValueError("failed response attempt cannot claim a normalized validated SHA-256")
    envelope = job.input_bindings[AUTHENTICATED_MESSAGE_ENVELOPE_BINDING]
    repair_binding = job.input_bindings.get(AUTHENTICATED_RESPONSE_REPAIR_BINDING)
    return {
        "attempt_number": 0,  # assigned by _response_attempt_trace
        "job_id": job.job_id,
        "job_kind": job.job_kind,
        "job_sha256": _sha(job.as_dict()),
        "input_bindings_sha256": _sha(job.input_bindings),
        "messages_sha256": envelope["sha256"],
        "rendered_message_array_byte_count": envelope["byte_count"],
        "response_repair_binding": _clone(repair_binding),
        "validation_outcome": validation_outcome,
        "raw_response_projection_sha256": raw_response_projection_sha256,
        "normalized_validated_response_sha256": normalized_validated_response_sha256,
    }


def _response_attempt_trace(
    *,
    logical_job: DiscoveryJsonJob,
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for index, attempt in enumerate(attempts, start=1):
        row = _clone(attempt)
        row["attempt_number"] = index
        rows.append(row)
    body = {
        "schema_version": DISCOVERY_RESPONSE_ATTEMPT_TRACE_VERSION,
        "policy_sha256": discovery_response_repair_policy_identity()["policy_sha256"],
        "logical_job_id": logical_job.job_id,
        "attempts": rows,
    }
    return {**body, "trace_sha256": _sha(body)}


def _validated_response_attempt_trace(
    *,
    logical_job: DiscoveryJsonJob,
    validated_response_sha256: str,
    trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row = _exact_mapping(
        trace,
        keys={
            "schema_version",
            "policy_sha256",
            "logical_job_id",
            "attempts",
            "trace_sha256",
        },
        label="response attempt trace",
    )
    if row["schema_version"] != DISCOVERY_RESPONSE_ATTEMPT_TRACE_VERSION:
        raise ValueError("response attempt trace has the wrong schema version")
    if row["policy_sha256"] != discovery_response_repair_policy_identity()["policy_sha256"]:
        raise ValueError("response attempt trace cites a different repair policy")
    if row["logical_job_id"] != logical_job.job_id:
        raise ValueError("response attempt trace cites a different logical job")
    body = {key: value for key, value in row.items() if key != "trace_sha256"}
    if row["trace_sha256"] != _sha(body):
        raise ValueError("response attempt trace SHA-256 does not authenticate")
    attempts = row["attempts"]
    if not isinstance(attempts, list) or len(attempts) not in {1, 2}:
        raise ValueError("response attempt trace must contain one initial and at most one repair")
    expected_keys = {
        "attempt_number",
        "job_id",
        "job_kind",
        "job_sha256",
        "input_bindings_sha256",
        "messages_sha256",
        "rendered_message_array_byte_count",
        "response_repair_binding",
        "validation_outcome",
        "raw_response_projection_sha256",
        "normalized_validated_response_sha256",
    }
    normalized: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        item = _exact_mapping(
            attempt,
            keys=expected_keys,
            label=f"response attempt trace attempt[{index - 1}]",
        )
        if item["attempt_number"] != index:
            raise ValueError("response attempt numbers must be contiguous")
        if item["job_kind"] != logical_job.job_kind:
            raise ValueError("response attempt changed the logical job kind")
        for label in (
            "job_sha256",
            "input_bindings_sha256",
            "messages_sha256",
            "raw_response_projection_sha256",
        ):
            if not isinstance(item[label], str) or _SHA256.fullmatch(item[label]) is None:
                raise ValueError(f"response attempt {label} is invalid")
        repair_binding = item["response_repair_binding"]
        if repair_binding is not None and not isinstance(repair_binding, Mapping):
            raise TypeError("response attempt repair binding must be one JSON object or null")
        byte_count = item["rendered_message_array_byte_count"]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 1
        ):
            raise ValueError("response attempt carries an invalid rendered byte count")
        if item["validation_outcome"] not in {
            *_REPAIRABLE_RESPONSE_FAILURES,
            VALIDATED_RESPONSE,
        }:
            raise ValueError("response attempt validation outcome is unsupported")
        normalized_sha256 = item["normalized_validated_response_sha256"]
        if item["validation_outcome"] == VALIDATED_RESPONSE:
            if (
                not isinstance(normalized_sha256, str)
                or _SHA256.fullmatch(normalized_sha256) is None
            ):
                raise ValueError("validated attempt normalized response SHA-256 is invalid")
        elif normalized_sha256 is not None:
            raise ValueError("failed attempt cannot carry a normalized response SHA-256")
        normalized.append(dict(item))
    first = normalized[0]
    if len(normalized) == 1:
        if first["validation_outcome"] != VALIDATED_RESPONSE:
            raise ValueError("an unrepaired response attempt must be validated")
    else:
        second = normalized[1]
        if first["validation_outcome"] not in _REPAIRABLE_RESPONSE_FAILURES:
            raise ValueError("a repair must follow one admitted initial failure")
        if second["validation_outcome"] != VALIDATED_RESPONSE:
            raise ValueError("the single response repair must be validated")
    expected_first = _response_attempt_entry(
        job=logical_job,
        validation_outcome=first["validation_outcome"],
        raw_response_projection_sha256=first["raw_response_projection_sha256"],
        normalized_validated_response_sha256=(first["normalized_validated_response_sha256"]),
    )
    expected_first["attempt_number"] = 1
    if canonical_json(first) != canonical_json(expected_first):
        raise ValueError("response attempt trace changed the exact initial authenticated job")
    if len(normalized) == 2:
        expected_repair_job = _build_response_repair_job_from_projection_sha256(
            original_job=logical_job,
            prior_response_content_sha256=first["raw_response_projection_sha256"],
            failure_category=first["validation_outcome"],
        )
        expected_second = _response_attempt_entry(
            job=expected_repair_job,
            validation_outcome=VALIDATED_RESPONSE,
            raw_response_projection_sha256=normalized[1]["raw_response_projection_sha256"],
            normalized_validated_response_sha256=validated_response_sha256,
        )
        expected_second["attempt_number"] = 2
        if canonical_json(normalized[1]) != canonical_json(expected_second):
            raise ValueError("response attempt trace changed the exact deterministic repair job")
    if normalized[-1]["normalized_validated_response_sha256"] != validated_response_sha256:
        raise ValueError("final response attempt differs from the validated response")
    return _clone(row)


@dataclass(frozen=True)
class ValidatedDiscoveryJobResult:
    job_id: str
    response_sha256: str
    response_attempt_trace_sha256: str
    _response_json: str = field(repr=False)
    _response_attempt_trace_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if _JOB_ID.fullmatch(self.job_id) is None:
            raise ValueError("result job_id is invalid")
        if _SHA256.fullmatch(self.response_sha256) is None:
            raise ValueError("response_sha256 must be a lowercase SHA-256")
        response = json.loads(self._response_json)
        if not isinstance(response, Mapping):
            raise TypeError("validated response must be one JSON object")
        if self.response_sha256 != _sha(response):
            raise ValueError("response_sha256 does not authenticate the validated response")
        if _SHA256.fullmatch(self.response_attempt_trace_sha256) is None:
            raise ValueError("response_attempt_trace_sha256 must be a lowercase SHA-256")
        trace = json.loads(self._response_attempt_trace_json)
        if not isinstance(trace, Mapping):
            raise TypeError("response attempt trace must be one JSON object")
        if self.response_attempt_trace_sha256 != _sha(trace):
            raise ValueError("response_attempt_trace_sha256 does not authenticate its trace")

    @classmethod
    def create(
        cls,
        *,
        job: DiscoveryJsonJob,
        validated_response: Mapping[str, Any],
        response_attempt_trace: Mapping[str, Any] | None = None,
    ) -> "ValidatedDiscoveryJobResult":
        response = _clone(validated_response)
        trace = (
            _validated_response_attempt_trace(
                logical_job=job,
                validated_response_sha256=_sha(response),
                trace=response_attempt_trace,
            )
            if response_attempt_trace is not None
            else _response_attempt_trace(
                logical_job=job,
                attempts=(
                    _response_attempt_entry(
                        job=job,
                        validation_outcome=VALIDATED_RESPONSE,
                        raw_response_projection_sha256=_sha(response),
                        normalized_validated_response_sha256=_sha(response),
                    ),
                ),
            )
        )
        return cls(
            job_id=job.job_id,
            response_sha256=_sha(response),
            response_attempt_trace_sha256=_sha(trace),
            _response_json=canonical_json(response),
            _response_attempt_trace_json=canonical_json(trace),
        )

    @property
    def response(self) -> dict[str, Any]:
        return json.loads(self._response_json)

    @property
    def response_attempt_trace(self) -> dict[str, Any]:
        return json.loads(self._response_attempt_trace_json)

    @property
    def raw_wire_response_sha256(self) -> str:
        return str(self.response_attempt_trace["attempts"][-1]["raw_response_projection_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "raw_wire_response_sha256": self.raw_wire_response_sha256,
            "response_sha256": self.response_sha256,
            "response": self.response,
            "response_attempt_trace_sha256": self.response_attempt_trace_sha256,
            "response_attempt_trace": self.response_attempt_trace,
        }


@dataclass(frozen=True)
class DiscoveryExecutionLedger:
    job_ledger: DiscoveryJobLedger
    results: tuple[ValidatedDiscoveryJobResult, ...]
    execution_sha256: str

    def __post_init__(self) -> None:
        if len(self.results) != len(self.job_ledger.jobs):
            raise ValueError("execution ledger requires one result per job")
        expected = tuple(job.job_id for job in self.job_ledger.jobs)
        observed = tuple(result.job_id for result in self.results)
        if observed != expected:
            raise ValueError("execution results must follow exact job-ledger order")
        if _SHA256.fullmatch(self.execution_sha256) is None:
            raise ValueError("execution_sha256 must be a lowercase SHA-256")
        if self.execution_sha256 != _sha(self._identity_without_sha()):
            raise ValueError("execution ledger SHA-256 does not authenticate")

    @classmethod
    def build(
        cls,
        *,
        jobs: Sequence[DiscoveryJsonJob],
        results: Sequence[ValidatedDiscoveryJobResult],
    ) -> "DiscoveryExecutionLedger":
        ledger = DiscoveryJobLedger.build(jobs)
        values = tuple(results)
        identity = {
            "schema_version": DISCOVERY_EXECUTION_LEDGER_VERSION,
            "job_ledger_sha256": ledger.ledger_sha256,
            "results": [result.as_dict() for result in values],
        }
        return cls(
            job_ledger=ledger,
            results=values,
            execution_sha256=_sha(identity),
        )

    def _identity_without_sha(self) -> dict[str, Any]:
        return {
            "schema_version": DISCOVERY_EXECUTION_LEDGER_VERSION,
            "job_ledger_sha256": self.job_ledger.ledger_sha256,
            "results": [result.as_dict() for result in self.results],
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._identity_without_sha(),
            "execution_sha256": self.execution_sha256,
            "job_ledger": self.job_ledger.as_dict(),
        }


@runtime_checkable
class JsonDiscoveryJobRunner(Protocol):
    """Injected JSON runner; this module supplies no transport implementation."""

    def identity(self) -> Mapping[str, Any]:
        pass

    def run_json(self, *, job: DiscoveryJsonJob) -> Mapping[str, Any]:
        pass


@dataclass(frozen=True)
class DirectNumericalDossierBinding:
    """The only numerical-channel shape admitted by discovery orchestration."""

    source_family: str
    # Compatibility input for the realized-manifest path.  This is never
    # populated for a pre-fit intent and is omitted from authenticated output.
    manifest_sha256: str = ""
    signal_count: int = 0
    zero_reason: str = ""
    direct_numerical_contract_kind: str = ""
    direct_numerical_contract_sha256: str = ""

    def __post_init__(self) -> None:
        if self.source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("direct numerical binding has an inactive source family")
        contract_kind = self.direct_numerical_contract_kind
        contract_sha256 = self.direct_numerical_contract_sha256
        if not contract_kind and not contract_sha256 and self.manifest_sha256:
            contract_kind = DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
            contract_sha256 = self.manifest_sha256
            object.__setattr__(self, "direct_numerical_contract_kind", contract_kind)
            object.__setattr__(self, "direct_numerical_contract_sha256", contract_sha256)
        if contract_kind not in DIRECT_NUMERICAL_CONTRACT_KINDS:
            raise ValueError("direct_numerical_contract_kind is unsupported")
        if _SHA256.fullmatch(contract_sha256) is None:
            raise ValueError("direct_numerical_contract_sha256 must be a lowercase SHA-256")
        if contract_kind == DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST:
            if self.manifest_sha256 and self.manifest_sha256 != contract_sha256:
                raise ValueError("manifest and generic numerical contract digests differ")
            if not self.manifest_sha256:
                object.__setattr__(self, "manifest_sha256", contract_sha256)
        elif self.manifest_sha256:
            raise ValueError("a pre-fit intent cannot expose a manifest SHA-256")
        if isinstance(self.signal_count, bool) or not isinstance(self.signal_count, int):
            raise TypeError("signal_count must be a non-negative integer")
        if self.signal_count < 0:
            raise ValueError("signal_count cannot be negative")
        _string(self.zero_reason, label="zero_reason", empty=True)
        if self.signal_count == 0 and not self.zero_reason:
            raise ValueError("zero signal_count requires an explicit zero_reason")
        if self.signal_count > 0 and self.zero_reason:
            raise ValueError("nonzero signal_count cannot have a zero_reason")

    def as_dossier_dict(self) -> dict[str, Any]:
        return {
            "source_family": self.source_family,
            "channel": DIRECT_UPSTREAM_NUMERICAL_CHANNEL,
            "direct_numerical_contract_kind": self.direct_numerical_contract_kind,
            "direct_numerical_contract_sha256": self.direct_numerical_contract_sha256,
            "signal_count": self.signal_count,
            "zero_reason": self.zero_reason,
            "concept_grounding_allowed": False,
        }


@dataclass(frozen=True)
class HierarchicalDiscoveryConfig:
    max_rendered_prompt_bytes: int = MAX_RENDERED_DISCOVERY_PROMPT_BYTES
    selector_thinking_token_budget: int = SELECTOR_THINKING_TOKEN_BUDGET
    max_semantic_member_ids_per_chunk: int = DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK
    max_cross_architecture_lookback_ids_per_group: int = 8
    max_cross_architecture_lookback_bytes_per_group: int = 96_000
    max_extraction_lookback_ids_per_feature: int = 8
    max_extraction_lookback_bytes_per_feature: int = 96_000
    max_rejection_lookback_ids_per_candidate: int = 24
    max_rejection_lookback_bytes_per_candidate: int = 48_000
    max_integrated_features: int = 16
    wire_budget: HierarchyWireBudget = field(
        default_factory=lambda: LEGACY_HIERARCHY_WIRE_BUDGET
    )

    def __post_init__(self) -> None:
        if not isinstance(self.wire_budget, HierarchyWireBudget):
            raise TypeError("wire_budget must be a HierarchyWireBudget")
        nonnegative = (
            "max_cross_architecture_lookback_ids_per_group",
            "max_rejection_lookback_ids_per_candidate",
        )
        positive = (
            "max_rendered_prompt_bytes",
            "selector_thinking_token_budget",
            "max_semantic_member_ids_per_chunk",
            "max_cross_architecture_lookback_bytes_per_group",
            "max_extraction_lookback_ids_per_feature",
            "max_extraction_lookback_bytes_per_feature",
            "max_rejection_lookback_bytes_per_candidate",
            "max_integrated_features",
        )
        for name in (*nonnegative, *positive):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if name in nonnegative and value < 0:
                raise ValueError(f"{name} cannot be negative")
            if name in positive and value < 1:
                raise ValueError(f"{name} must be positive")
        if (
            self.max_semantic_member_ids_per_chunk
            > self.wire_budget.max_interpret_members_per_job
        ):
            raise ValueError(
                "max_semantic_member_ids_per_chunk exceeds the authenticated "
                "interpret response budget"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_rendered_prompt_bytes": self.max_rendered_prompt_bytes,
            "max_semantic_member_ids_per_chunk": (self.max_semantic_member_ids_per_chunk),
            "max_cross_architecture_lookback_ids_per_group": (
                self.max_cross_architecture_lookback_ids_per_group
            ),
            "max_cross_architecture_lookback_bytes_per_group": (
                self.max_cross_architecture_lookback_bytes_per_group
            ),
            "max_extraction_lookback_ids_per_feature": (
                self.max_extraction_lookback_ids_per_feature
            ),
            "max_extraction_lookback_bytes_per_feature": (
                self.max_extraction_lookback_bytes_per_feature
            ),
            "max_rejection_lookback_ids_per_candidate": (
                self.max_rejection_lookback_ids_per_candidate
            ),
            "max_rejection_lookback_bytes_per_candidate": (
                self.max_rejection_lookback_bytes_per_candidate
            ),
            "max_integrated_features": self.max_integrated_features,
            "hierarchy_wire_budget": self.wire_budget.as_dict(),
            "legacy_lookback_and_feature_cap_fields_apply_semantic_truncation": False,
            "selector_thinking_enabled": True,
            "selector_thinking_token_budget": self.selector_thinking_token_budget,
            "extraction_definition_thinking_enabled": False,
            "extraction_definition_thinking_token_budget": 0,
        }


@dataclass(frozen=True)
class HierarchicalDiscoveryPrecommit:
    precommit_sha256: str
    _packet_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.precommit_sha256) is None:
            raise ValueError("precommit_sha256 must be a lowercase SHA-256")
        packet = json.loads(self._packet_json)
        if not isinstance(packet, Mapping):
            raise TypeError("precommit packet must be one JSON object")
        if self.precommit_sha256 != _sha(packet):
            raise ValueError("precommit SHA-256 does not authenticate the offline packet")

    @classmethod
    def create(cls, packet: Mapping[str, Any]) -> "HierarchicalDiscoveryPrecommit":
        detached = _clone(packet)
        return cls(precommit_sha256=_sha(detached), _packet_json=canonical_json(detached))

    @property
    def packet(self) -> dict[str, Any]:
        return json.loads(self._packet_json)

    def render_json(self, *, indent: int = 2) -> str:
        if isinstance(indent, bool) or not isinstance(indent, int) or indent < 0:
            raise ValueError("indent must be a non-negative integer")
        return json.dumps(
            {
                "precommit_sha256": self.precommit_sha256,
                "packet": self.packet,
            },
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )


@dataclass(frozen=True)
class IntegratedCanonicalFeature:
    canonical_name: str
    description: str
    member_candidate_ids: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    source_families: tuple[str, ...]
    value_shape_hypothesis: str
    unresolved_ambiguity: str
    allowed_aliases: tuple[str, ...] = ()
    allowed_units: tuple[str, ...] = ()
    allowed_categories: tuple[str, ...] = ()
    allowed_distinguish_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _feature_name(self.canonical_name, label="canonical_name")
        _string(self.description, label="description")
        if not self.member_candidate_ids or not self.supporting_evidence_ids:
            raise ValueError("integrated features require candidate and evidence support")
        for label, values, identifiers in (
            ("member_candidate_ids", self.member_candidate_ids, True),
            ("supporting_evidence_ids", self.supporting_evidence_ids, True),
            ("source_families", self.source_families, False),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} cannot contain duplicates")
            for index, value in enumerate(values):
                if identifiers:
                    _identifier(value, label=f"{label}[{index}]")
                else:
                    _string(value, label=f"{label}[{index}]")
        if not self.source_families or not set(self.source_families) <= (
            ACTIVE_STAGE1_CONCEPT_FAMILY_SET
        ):
            raise ValueError("integrated feature source families are empty or inactive")
        if self.value_shape_hypothesis not in {"continuous", "categorical", "ambiguous"}:
            raise ValueError("integrated feature value shape is invalid")
        _string(self.unresolved_ambiguity, label="unresolved_ambiguity", empty=True)
        for label, values in (
            ("allowed_aliases", self.allowed_aliases),
            ("allowed_units", self.allowed_units),
            ("allowed_categories", self.allowed_categories),
            ("allowed_distinguish_from", self.allowed_distinguish_from),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} cannot contain duplicates")
            for index, value in enumerate(values):
                _string(value, label=f"{label}[{index}]")


@dataclass(frozen=True)
class RoutedIntegratedFeature:
    feature: IntegratedCanonicalFeature
    role_routing: RoleRoutingResult

    def __post_init__(self) -> None:
        if not isinstance(self.feature, IntegratedCanonicalFeature):
            raise TypeError("feature must be IntegratedCanonicalFeature")
        if not isinstance(self.role_routing, RoleRoutingResult):
            raise TypeError("role_routing must be RoleRoutingResult")

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.feature.canonical_name,
            "description": self.feature.description,
            "member_candidate_ids": list(self.feature.member_candidate_ids),
            "supporting_evidence_ids": list(self.feature.supporting_evidence_ids),
            "source_families": list(self.feature.source_families),
            "value_shape_hypothesis": self.feature.value_shape_hypothesis,
            "unresolved_ambiguity": self.feature.unresolved_ambiguity,
            "role_routing": self.role_routing.audit(),
        }


@dataclass(frozen=True)
class CompletedHierarchicalDiscovery:
    precommit_sha256: str
    dossiers: tuple[ArchitectureDossier, ...]
    routed_features: tuple[RoutedIntegratedFeature, ...]
    rejected_candidate_ids: tuple[str, ...]
    requested_lookback_evidence_ids: tuple[str, ...]
    extraction_job_ids: tuple[str, ...]
    execution_ledger: DiscoveryExecutionLedger
    completion_sha256: str
    _planner_response_json: str = field(repr=False)
    _integration_response_json: str = field(repr=False)
    _rejection_critic_response_json: str = field(repr=False)
    _extraction_definitions_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.precommit_sha256) is None:
            raise ValueError("precommit_sha256 must be a lowercase SHA-256")
        if _SHA256.fullmatch(self.completion_sha256) is None:
            raise ValueError("completion_sha256 must be a lowercase SHA-256")
        if tuple(dossier.source_family for dossier in self.dossiers) != (
            ACTIVE_STAGE1_CONCEPT_FAMILIES
        ):
            raise ValueError("completed discovery requires all ten dossiers in canonical order")
        for label, values in (
            ("rejected_candidate_ids", self.rejected_candidate_ids),
            ("requested_lookback_evidence_ids", self.requested_lookback_evidence_ids),
            ("extraction_job_ids", self.extraction_job_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} cannot contain duplicates")
        if not isinstance(self.execution_ledger, DiscoveryExecutionLedger):
            raise TypeError("execution_ledger must be DiscoveryExecutionLedger")
        extraction_jobs = tuple(
            job.job_id
            for job in self.execution_ledger.job_ledger.jobs
            if job.job_kind == EXTRACTION_DEFINITION_JOB
        )
        if extraction_jobs != self.extraction_job_ids:
            raise ValueError("extraction_job_ids differ from the authenticated job ledger")
        definitions = self.extraction_definitions
        if set(definitions) != {routed.feature.canonical_name for routed in self.routed_features}:
            raise ValueError("extraction definitions must cover every routed feature exactly once")
        integration_dispositions = self.integration_response.get("candidate_dispositions")
        if not isinstance(integration_dispositions, list):
            raise TypeError("integration candidate_dispositions must be a JSON list")
        integration_rejected_ids: list[str] = []
        for index, raw in enumerate(integration_dispositions):
            if not isinstance(raw, Mapping):
                raise TypeError(f"integration candidate_dispositions[{index}] must be an object")
            candidate_id = _identifier(
                raw.get("candidate_id"),
                label=f"integration candidate_dispositions[{index}].candidate_id",
            )
            decision = raw.get("decision")
            if decision not in {"accept", "reject"}:
                raise ValueError(f"integration candidate_dispositions[{index}].decision is invalid")
            if decision == "reject":
                integration_rejected_ids.append(candidate_id)
        if self.rejected_candidate_ids != tuple(integration_rejected_ids):
            raise ValueError(
                "rejected_candidate_ids differ from the authenticated integration dispositions"
            )
        identity = {
            "schema_version": COMPLETED_HIERARCHICAL_DISCOVERY_VERSION,
            "precommit_sha256": self.precommit_sha256,
            "dossiers": [row.as_authenticated_dict() for row in self.dossiers],
            "planner_response": self.planner_response,
            "requested_lookback_evidence_ids": list(self.requested_lookback_evidence_ids),
            "integration_response": self.integration_response,
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "rejection_critic_response": self.rejection_critic_response,
            "routed_features": [row.as_dict() for row in self.routed_features],
            "extraction_definitions": definitions,
            "execution_sha256": self.execution_ledger.execution_sha256,
        }
        if self.completion_sha256 != _sha(identity):
            raise ValueError("completion_sha256 does not authenticate completed discovery")

    @property
    def planner_response(self) -> dict[str, Any]:
        return json.loads(self._planner_response_json)

    @property
    def integration_response(self) -> dict[str, Any]:
        return json.loads(self._integration_response_json)

    @property
    def rejection_critic_response(self) -> dict[str, Any]:
        return json.loads(self._rejection_critic_response_json)

    @property
    def extraction_definitions(self) -> dict[str, Any]:
        return json.loads(self._extraction_definitions_json)


class CoverageCriticRequiresRevision(RuntimeError):
    """Raised when an architecture-local critic finds unresolved loss."""


class RejectionCriticRequiresRevision(RuntimeError):
    """Raised when a rejection critic does not uphold every rejection."""


def _family_catalog_sha256(
    catalog: RoleNeutralEvidenceCatalog, family: str, evidence: Sequence[DiscoveryEvidenceItem]
) -> str:
    return _sha(
        {
            "catalog_sha256": catalog.catalog_sha256,
            "source_family": family,
            "evidence": [item.as_prompt_item() for item in evidence],
        }
    )


def _candidate_from_interpretation(
    *,
    job: DiscoveryJsonJob,
    family: str,
    concept: Mapping[str, Any],
) -> DiscoveryCandidate:
    identity = {
        "job_id": job.job_id,
        "source_family": family,
        "concept": concept,
    }
    return DiscoveryCandidate(
        candidate_id=f"candidate_{_sha(identity)}",
        feature_name=str(concept["feature_name"]),
        description=str(concept["description"]),
        supporting_evidence_ids=tuple(concept["supporting_evidence_ids"]),
        source_families=(family,),
        value_shape_hypothesis=str(concept["value_shape_hypothesis"]),
        unresolved_ambiguity=str(concept["unresolved_ambiguity"]),
    )


def _candidate_from_consolidation(*, family: str, concept: Mapping[str, Any]) -> DiscoveryCandidate:
    identity = {
        "source_family": family,
        "canonical_concept": concept,
    }
    return DiscoveryCandidate(
        candidate_id=f"candidate_{_sha(identity)}",
        feature_name=str(concept["canonical_name"]),
        description=str(concept["description"]),
        supporting_evidence_ids=tuple(concept["supporting_evidence_ids"]),
        source_families=(family,),
        value_shape_hypothesis=str(concept["value_shape_hypothesis"]),
        unresolved_ambiguity=str(concept["unresolved_ambiguity"]),
    )


def _render_consolidation_messages(
    *,
    source_family: str,
    candidates: Sequence[DiscoveryCandidate],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    if candidates:
        context = consolidate_candidate_context(
            source_family=source_family,
            candidates=candidates,
        )
    else:
        context = {
            "job": "consolidate_candidate_ledger",
            "source_family": source_family,
            "candidates": [],
        }
    context.pop("schema_version", None)
    request = attach_hierarchical_discovery_response_contract(
        job_kind=CONSOLIDATE_ARCHITECTURE_JOB,
        request=context,
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": (
                "Consolidate candidates from exactly one Stage 1 architecture. Merge only "
                "clear aliases and preserve every candidate, supporting evidence ID, source "
                "family, ambiguity, and measurement distinction. candidate_assignments is "
                "keyed by every exact candidate_id; assign each candidate to one fixed "
                "cluster_slot and define every fixed slot in slot_definitions. Do not assign "
                "causal roles, define extraction, estimate effects, or reject a candidate. "
                "Return JSON only."
            ),
        },
        {"role": "user", "content": canonical_json(request)},
    )


def _bounded_candidate_projection(candidate: DiscoveryCandidate) -> dict[str, Any]:
    """Keep relation/definition prompts bounded while authenticating full support."""

    return {
        "candidate_id": candidate.candidate_id,
        "feature_name": candidate.feature_name,
        "description": candidate.description,
        "source_families": list(candidate.source_families),
        "value_shape_hypothesis": candidate.value_shape_hypothesis,
        "unresolved_ambiguity": candidate.unresolved_ambiguity,
        "supporting_evidence_count": len(candidate.supporting_evidence_ids),
    }


def _render_candidate_relation_page_messages(
    *,
    job: str,
    job_kind: str,
    source_family: str | None,
    anchor: DiscoveryCandidate,
    peers: Sequence[DiscoveryCandidate],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    peer_tuple = tuple(peers)
    request: dict[str, Any] = {
        "job": job,
        "anchor_candidate_id": anchor.candidate_id,
        "peer_candidate_ids": [peer.candidate_id for peer in peer_tuple],
        "anchor_candidate": _bounded_candidate_projection(anchor),
        "peer_candidates": [_bounded_candidate_projection(peer) for peer in peer_tuple],
    }
    if source_family is not None:
        request["source_family"] = source_family
    payload = attach_hierarchical_discovery_response_contract(
        job_kind=job_kind,
        request=request,
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": (
                "Judge every exact anchor-to-peer candidate pair independently. Use relation "
                "same_construct only for the same patient-level construct; use distinct for "
                "different measurements and uncertain when the compact evidence cannot decide. "
                "Do not infer transitive merges, repeat support IDs, assign causal roles, define "
                "extraction, or estimate effects. comparisons is keyed by every exact later peer "
                "candidate ID. Return JSON only."
            ),
        },
        {"role": "user", "content": canonical_json(payload)},
    )


def _render_candidate_definition_fold_messages(
    *,
    job: str,
    job_kind: str,
    group_id: str,
    fold_index: int,
    candidates: Sequence[DiscoveryCandidate],
    prior_accumulator: Mapping[str, Any] | None,
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    candidate_tuple = tuple(candidates)
    request = attach_hierarchical_discovery_response_contract(
        job_kind=job_kind,
        request={
            "job": job,
            "group_id": group_id,
            "fold_index": fold_index,
            "member_candidate_ids": [item.candidate_id for item in candidate_tuple],
            "prior_accumulator": (
                _clone(prior_accumulator) if prior_accumulator is not None else None
            ),
            "fresh_candidates": [_bounded_candidate_projection(item) for item in candidate_tuple],
        },
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": (
                "Define one canonical patient-level construct for this compiler-proven "
                "complete-link group. Fold the prior accumulator, when supplied, together with "
                "every fresh member without dropping measurement distinctions or unresolved "
                "ambiguity. Membership, provenance, and evidence support are compiler-owned; do "
                "not repeat them, assign causal roles, define extraction, or estimate effects. "
                "Return JSON only."
            ),
        },
        {"role": "user", "content": canonical_json(request)},
    )


def _validate_candidate_definition_fold_response(response: Any) -> dict[str, str]:
    if not isinstance(response, Mapping):
        raise TypeError("candidate definition fold response must be one JSON object")
    if set(response) != {
        "canonical_name",
        "description",
        "unresolved_ambiguity",
        "reason",
    }:
        raise ValueError("candidate definition fold response keys differ from its contract")
    canonical_name = _string(response["canonical_name"], label="canonical_name")
    if _NAME.fullmatch(canonical_name) is None:
        raise ValueError("candidate definition canonical_name must be lower_snake_case")
    return {
        "canonical_name": canonical_name,
        "description": _string(response["description"], label="description"),
        "unresolved_ambiguity": _string(
            response["unresolved_ambiguity"],
            label="unresolved_ambiguity",
            empty=True,
        ),
        "reason": _string(response["reason"], label="reason"),
    }


def _compile_bounded_consolidation(
    *,
    source_family: str,
    candidates: Sequence[DiscoveryCandidate],
    grouped: Mapping[str, Any],
    definitions_by_group_id: Mapping[str, Mapping[str, Any]],
    wire_budget: HierarchyWireBudget,
) -> dict[str, Any]:
    items = tuple(candidates)
    by_id = {item.candidate_id: item for item in items}
    groups = grouped.get("groups")
    if not isinstance(groups, list):
        raise TypeError("complete-link compiler did not return group rows")
    used_names: set[str] = set()
    name_by_group: dict[str, str] = {}
    disambiguations: list[dict[str, str]] = []
    concepts: list[dict[str, Any]] = []
    group_by_candidate: dict[str, str] = {}
    for raw_group in groups:
        if not isinstance(raw_group, Mapping):
            raise TypeError("complete-link group row must be one object")
        group_id = _string(raw_group.get("group_id"), label="group_id")
        members_raw = raw_group.get("member_candidate_ids")
        if not isinstance(members_raw, list) or not members_raw:
            raise ValueError("complete-link group must have non-empty membership")
        members = tuple(_string(value, label="member_candidate_id") for value in members_raw)
        if not set(members) <= set(by_id):
            raise ValueError("complete-link group cites an unknown candidate")
        definition = definitions_by_group_id.get(group_id)
        if not isinstance(definition, Mapping):
            raise ValueError("complete-link group lacks its final definition fold")
        proposed = _string(definition.get("canonical_name"), label="canonical_name")
        derived, event = _derive_unique_integration_name(
            proposed=proposed,
            slot=f"group_{_sha(group_id)[:8]}",
            used=used_names,
            wire_budget=wire_budget,
        )
        if event is not None:
            disambiguations.append(
                {
                    "group_id": group_id,
                    "proposed_canonical_name": proposed,
                    "derived_canonical_name": derived,
                    "reason": event["reason"],
                }
            )
        used_names.add(derived)
        name_by_group[group_id] = derived
        group_by_candidate.update({member: group_id for member in members})
        supporting_evidence = tuple(
            dict.fromkeys(
                evidence_id
                for member in members
                for evidence_id in by_id[member].supporting_evidence_ids
            )
        )
        shapes = {by_id[member].value_shape_hypothesis for member in members}
        concepts.append(
            {
                "canonical_name": derived,
                "description": _string(definition.get("description"), label="description"),
                "member_candidate_ids": list(members),
                "supporting_evidence_ids": list(supporting_evidence),
                "source_families": [source_family],
                "value_shape_hypothesis": (next(iter(shapes)) if len(shapes) == 1 else "ambiguous"),
                "unresolved_ambiguity": _string(
                    definition.get("unresolved_ambiguity"),
                    label="unresolved_ambiguity",
                    empty=True,
                ),
            }
        )
    if set(group_by_candidate) != set(by_id):
        raise ValueError("complete-link groups do not preserve every candidate")
    return _clone(
        {
            "canonical_concepts": concepts,
            "candidate_dispositions": [
                {
                    "candidate_id": item.candidate_id,
                    "canonical_name": name_by_group[group_by_candidate[item.candidate_id]],
                    "reason": "exhaustive pair judgments plus complete-link compilation",
                }
                for item in items
            ],
            "wire_normalization_audit": {
                "audit_version": "bounded_complete_link_consolidation_compiler_v1",
                "relation_compiler": _clone(grouped["pair_relation_audit"]),
                "groups": _clone(groups),
                "final_group_definitions": {
                    key: _clone(value) for key, value in definitions_by_group_id.items()
                },
                "canonical_name_disambiguations": disambiguations,
                "membership_support_and_provenance_compiler_owned": True,
            },
        }
    )


def _compile_bounded_cross_architecture_plan(
    *,
    candidates: Sequence[DiscoveryCandidate],
    grouped: Mapping[str, Any],
    definitions_by_group_id: Mapping[str, Mapping[str, Any]],
    evidence_by_id: Mapping[str, DiscoveryEvidenceItem],
    wire_budget: HierarchyWireBudget,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    items = tuple(candidates)
    by_id = {item.candidate_id: item for item in items}
    groups = grouped.get("groups")
    if not isinstance(groups, list):
        raise TypeError("complete-link planner did not return group rows")
    used_names: set[str] = set()
    provisional_groups: list[dict[str, Any]] = []
    raw_requests: list[dict[str, Any]] = []
    lookback_by_group: dict[str, tuple[str, ...]] = {}
    support_review_audits: list[dict[str, Any]] = []
    disambiguations: list[dict[str, str]] = []
    globally_requested: set[str] = set()
    for raw_group in groups:
        if not isinstance(raw_group, Mapping):
            raise TypeError("complete-link planner group must be one object")
        group_id = _string(raw_group.get("group_id"), label="group_id")
        member_ids_raw = raw_group.get("member_candidate_ids")
        if not isinstance(member_ids_raw, list) or not member_ids_raw:
            raise ValueError("complete-link planner group must have members")
        member_ids = tuple(_string(value, label="member_candidate_id") for value in member_ids_raw)
        members = tuple(by_id[value] for value in member_ids)
        definition = definitions_by_group_id.get(group_id)
        if not isinstance(definition, Mapping):
            raise ValueError("complete-link planner group lacks its final definition")
        proposed = _string(definition.get("canonical_name"), label="canonical_name")
        name, event = _derive_unique_integration_name(
            proposed=proposed,
            slot=f"group_{_sha(group_id)[:8]}",
            used=used_names,
            wire_budget=wire_budget,
        )
        used_names.add(name)
        if event is not None:
            disambiguations.append(
                {
                    "group_id": group_id,
                    "proposed_provisional_name": proposed,
                    "derived_provisional_name": name,
                    "reason": event["reason"],
                }
            )
        provisional_groups.append(
            {
                "member_candidate_ids": list(member_ids),
                "provisional_name": name,
                "reason": _string(definition.get("reason"), label="reason"),
            }
        )
        complete_support = tuple(
            dict.fromkeys(
                evidence_id
                for candidate in members
                for evidence_id in candidate.supporting_evidence_ids
            )
        )
        unknown = set(complete_support) - set(evidence_by_id)
        if unknown:
            raise ValueError("complete provisional-group support is outside the catalog")
        lookback_by_group[group_id] = complete_support
        new_global_ids = tuple(
            evidence_id for evidence_id in complete_support if evidence_id not in globally_requested
        )
        globally_requested.update(new_global_ids)
        raw_requests.append(
            {
                "group_id": group_id,
                "evidence_ids": list(complete_support),
                "question": (
                    "Review every exact support item in independent raw-evidence pages, then "
                    "recursively fold every page without sampling."
                ),
                "reason": "Exhaustive per-group raw-evidence reconsideration.",
            }
        )
        support_review_audits.append(
            {
                "group_id": group_id,
                "complete_supporting_evidence_ids": list(complete_support),
                "complete_support_count": len(complete_support),
                "one_raw_evidence_item_per_page": True,
                "all_support_scheduled_exactly_once_for_this_group": True,
            }
        )
    return (
        _clone(
            {
                "provisional_groups": provisional_groups,
                "raw_evidence_requests": raw_requests,
                "wire_normalization_audit": {
                    "audit_version": "lossless_complete_link_planner_compiler_v2",
                    "relation_compiler": _clone(grouped["pair_relation_audit"]),
                    "groups": _clone(groups),
                    "final_group_definitions": {
                        key: _clone(value) for key, value in definitions_by_group_id.items()
                    },
                    "provisional_name_disambiguations": disambiguations,
                    "per_group_support_review_audits": support_review_audits,
                    "global_lookback_truncation": False,
                    "raw_support_sampling": False,
                    "every_group_support_item_is_page_scheduled": True,
                },
            }
        ),
        lookback_by_group,
    )


def _render_bounded_group_integration_messages(
    *,
    group_id: str,
    definition: Mapping[str, Any],
    members: Sequence[DiscoveryCandidate],
    lookback: Sequence[Mapping[str, Any]],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    member_tuple = tuple(members)
    source_families = tuple(
        dict.fromkeys(family for candidate in member_tuple for family in candidate.source_families)
    )
    shapes = tuple(dict.fromkeys(candidate.value_shape_hypothesis for candidate in member_tuple))
    support = tuple(
        dict.fromkeys(
            evidence_id
            for candidate in member_tuple
            for evidence_id in candidate.supporting_evidence_ids
        )
    )
    request = attach_hierarchical_discovery_response_contract(
        job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
        request={
            "job": "integrate_cross_architecture_group",
            "group_id": group_id,
            "provisional_definition": _clone(definition),
            "compiler_owned_relations": {
                "member_candidate_count": len(member_tuple),
                "supporting_evidence_count": len(support),
                "source_families": list(source_families),
                "value_shape_hypotheses": list(shapes),
            },
            "requested_raw_evidence_lookback": list(lookback),
        },
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": (
                "Make one final accept-or-reject decision for this exact provisional group. "
                "Use its bounded accumulator and only the supplied deterministic raw-evidence "
                "lookback. Preserve distinct measurements. Membership, complete support, source "
                "families, and value shape are compiler-owned and must not be repeated. Accept "
                "only a concrete patient-level feature. Do not assign causal roles, define "
                "extraction, or estimate effects. Return JSON only."
            ),
        },
        {"role": "user", "content": canonical_json(request)},
    )


def _validate_bounded_group_integration_wire(response: Any) -> dict[str, str]:
    row = _exact_mapping(
        response,
        keys={
            "decision",
            "canonical_name",
            "description",
            "unresolved_ambiguity",
            "reason",
        },
        label="bounded group integration response",
    )
    decision = _string(row["decision"], label="decision")
    if decision not in {"accept", "reject"}:
        raise ValueError("bounded group integration decision is invalid")
    name = _string(row["canonical_name"], label="canonical_name", empty=decision == "reject")
    description = _string(row["description"], label="description", empty=decision == "reject")
    ambiguity = _string(
        row["unresolved_ambiguity"],
        label="unresolved_ambiguity",
        empty=True,
    )
    if decision == "accept":
        _feature_name(name, label="canonical_name")
    elif name or description or ambiguity:
        raise ValueError("rejected bounded group must leave definition fields empty")
    return {
        "decision": decision,
        "canonical_name": name,
        "description": description,
        "unresolved_ambiguity": ambiguity,
        "reason": _string(row["reason"], label="reason"),
    }


def _review_accumulator_id(
    *, kind: str, scope: str, fold_index: int, response: Mapping[str, Any]
) -> str:
    return f"{kind}_accumulator_{_sha({'scope': scope, 'fold_index': fold_index, 'response': response})}"


def _render_integration_evidence_page_messages(
    *,
    group_id: str,
    definition: Mapping[str, Any],
    members: Sequence[DiscoveryCandidate],
    evidence: DiscoveryEvidenceItem,
    review_id: str,
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    request = attach_hierarchical_discovery_response_contract(
        job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
        request={
            "job": "review_integration_group_evidence",
            "group_id": group_id,
            "review_id": review_id,
            "evidence_id": evidence.evidence_id,
            "provisional_definition": _clone(definition),
            "group_members": [_bounded_candidate_projection(member) for member in members],
            "raw_evidence": {
                "evidence_id": evidence.evidence_id,
                "source_family": evidence.source_family,
                "member_ids": list(evidence.member_ids),
                "content": _clone(evidence.content),
            },
        },
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": (
                "Review exactly one authenticated raw evidence item against one provisional "
                "cross-architecture group. Decide whether it supports the same patient-level "
                "measurement, contradicts the group, reveals a distinct measurement, or remains "
                "ambiguous. Preserve a concrete distinct name only when the relationship is "
                "distinct_measurement. This page is one member of an exhaustive compiler-owned "
                "support schedule; do not infer that unseen support is absent. Do not assign "
                "causal roles, define extraction, or estimate effects. Set reviewed_evidence "
                "true and return JSON only."
            ),
        },
        {"role": "user", "content": canonical_json(request)},
    )


def _validate_integration_evidence_page_response(response: Any) -> dict[str, Any]:
    row = _exact_mapping(
        response,
        keys={
            "relationship",
            "proposed_distinct_name",
            "measurement_summary",
            "unresolved_ambiguity",
            "reason",
            "reviewed_evidence",
        },
        label="integration evidence page response",
    )
    relationship = _string(row["relationship"], label="relationship")
    if relationship not in {
        "supports_group",
        "contradicts_group",
        "distinct_measurement",
        "ambiguous",
    }:
        raise ValueError("integration evidence relationship is invalid")
    proposed = _string(
        row["proposed_distinct_name"],
        label="proposed_distinct_name",
        empty=relationship != "distinct_measurement",
    )
    if relationship == "distinct_measurement":
        _feature_name(proposed, label="proposed_distinct_name")
    elif proposed:
        raise ValueError("non-distinct integration evidence cannot propose a distinct name")
    if row["reviewed_evidence"] is not True:
        raise ValueError("integration evidence page did not acknowledge its raw evidence")
    return _clone(
        {
            "relationship": relationship,
            "proposed_distinct_name": proposed,
            "measurement_summary": _string(row["measurement_summary"], label="measurement_summary"),
            "unresolved_ambiguity": _string(
                row["unresolved_ambiguity"],
                label="unresolved_ambiguity",
                empty=True,
            ),
            "reason": _string(row["reason"], label="reason"),
            "reviewed_evidence": True,
        }
    )


def _render_integration_evidence_fold_messages(
    *,
    group_id: str,
    fold_index: int,
    definition: Mapping[str, Any],
    review_inputs: Sequence[tuple[str, Mapping[str, Any]]],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    inputs = tuple(review_inputs)
    request = attach_hierarchical_discovery_response_contract(
        job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
        request={
            "job": "fold_integration_group_evidence_reviews",
            "group_id": group_id,
            "fold_index": fold_index,
            "review_input_ids": [review_id for review_id, _ in inputs],
            "provisional_definition": _clone(definition),
            "review_inputs": [
                {"review_input_id": review_id, "review": _clone(review)}
                for review_id, review in inputs
            ],
        },
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": (
                "Fold every exact evidence review input into one lossless group accumulator. "
                "input_dispositions must address every authenticated input exactly once and "
                "preserve contradictions, distinct measurements, and ambiguity explicitly. "
                "Accept only when the complete transitive support establishes one concrete "
                "patient-level feature without silently collapsing a distinct measurement; "
                "otherwise reject for upstream reconsideration. The compiler owns membership, "
                "provenance, and complete support. Set complete_support_reviewed true. Do not "
                "assign causal roles, define extraction, or estimate effects. Return JSON only."
            ),
        },
        {"role": "user", "content": canonical_json(request)},
    )


def _validate_integration_evidence_fold_response(
    response: Any,
    *,
    review_input_ids: Sequence[str],
) -> dict[str, Any]:
    row = _exact_mapping(
        response,
        keys={
            "decision",
            "canonical_name",
            "description",
            "unresolved_ambiguity",
            "input_dispositions",
            "complete_support_reviewed",
            "reason",
        },
        label="integration evidence fold response",
    )
    base = _validate_bounded_group_integration_wire(
        {
            key: row[key]
            for key in (
                "decision",
                "canonical_name",
                "description",
                "unresolved_ambiguity",
                "reason",
            )
        }
    )
    dispositions: dict[str, dict[str, str]] = {}
    allowed = {
        "integrated",
        "contradiction_preserved",
        "distinct_measurement_preserved",
        "ambiguity_preserved",
    }
    for review_id, raw in _exact_keyed_rows(
        row["input_dispositions"],
        identifiers=tuple(review_input_ids),
        label="input_dispositions",
    ):
        disposition = _exact_mapping(
            raw,
            keys={"action", "reason"},
            label=f"input_dispositions.{review_id}",
        )
        action = _string(disposition["action"], label=f"{review_id}.action")
        if action not in allowed:
            raise ValueError("integration fold input disposition action is invalid")
        dispositions[review_id] = {
            "action": action,
            "reason": _string(disposition["reason"], label=f"{review_id}.reason"),
        }
    if row["complete_support_reviewed"] is not True:
        raise ValueError("integration fold did not acknowledge complete transitive support")
    return _clone(
        {
            **base,
            "input_dispositions": dispositions,
            "complete_support_reviewed": True,
        }
    )


def _render_rejection_evidence_page_messages(
    *,
    candidate: DiscoveryCandidate,
    integration_disposition: Mapping[str, Any],
    evidence: DiscoveryEvidenceItem,
    review_id: str,
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    request = attach_hierarchical_discovery_response_contract(
        job_kind=REJECTION_CRITIC_JOB,
        request={
            "job": "review_rejection_candidate_evidence",
            "candidate_id": candidate.candidate_id,
            "review_id": review_id,
            "evidence_id": evidence.evidence_id,
            "rejected_candidate": _bounded_candidate_projection(candidate),
            "integration_rejection": _clone(integration_disposition),
            "raw_evidence": {
                "evidence_id": evidence.evidence_id,
                "source_family": evidence.source_family,
                "member_ids": list(evidence.member_ids),
                "content": _clone(evidence.content),
            },
        },
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": (
                "Independently review exactly one authenticated raw evidence item for one "
                "rejected candidate. State whether it supports upholding, restoring, or splitting "
                "the candidate, or remains ambiguous. Propose a name only for restore/split. This "
                "is one page of an exhaustive compiler-owned candidate-support schedule; do not "
                "infer that unseen support is absent. Set reviewed_evidence true. Do not assign "
                "causal roles, define extraction, estimate effects, or invent evidence. Return "
                "JSON only."
            ),
        },
        {"role": "user", "content": canonical_json(request)},
    )


def _validate_rejection_evidence_page_response(response: Any) -> dict[str, Any]:
    row = _exact_mapping(
        response,
        keys={
            "signal",
            "proposed_name",
            "measurement_summary",
            "reason",
            "reviewed_evidence",
        },
        label="rejection evidence page response",
    )
    signal = _string(row["signal"], label="signal")
    if signal not in {
        "supports_uphold",
        "supports_restore",
        "supports_split",
        "ambiguous",
    }:
        raise ValueError("rejection evidence signal is invalid")
    proposed = _string(
        row["proposed_name"],
        label="proposed_name",
        empty=signal in {"supports_uphold", "ambiguous"},
    )
    if signal in {"supports_restore", "supports_split"}:
        _feature_name(proposed, label="proposed_name")
    elif proposed:
        raise ValueError("uphold/ambiguous rejection evidence cannot propose a name")
    if row["reviewed_evidence"] is not True:
        raise ValueError("rejection evidence page did not acknowledge its raw evidence")
    return _clone(
        {
            "signal": signal,
            "proposed_name": proposed,
            "measurement_summary": _string(row["measurement_summary"], label="measurement_summary"),
            "reason": _string(row["reason"], label="reason"),
            "reviewed_evidence": True,
        }
    )


def _render_rejection_evidence_fold_messages(
    *,
    candidate: DiscoveryCandidate,
    fold_index: int,
    review_inputs: Sequence[tuple[str, Mapping[str, Any]]],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    inputs = tuple(review_inputs)
    request = attach_hierarchical_discovery_response_contract(
        job_kind=REJECTION_CRITIC_JOB,
        request={
            "job": "fold_rejection_candidate_evidence_reviews",
            "candidate_id": candidate.candidate_id,
            "fold_index": fold_index,
            "review_input_ids": [review_id for review_id, _ in inputs],
            "rejected_candidate": _bounded_candidate_projection(candidate),
            "review_inputs": [
                {"review_input_id": review_id, "review": _clone(review)}
                for review_id, review in inputs
            ],
        },
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": (
                "Fold every exact evidence-review input for this rejected candidate. Address "
                "every input exactly once in input_dispositions, explicitly preserving ambiguity "
                "and explaining overruled signals. Decide uphold, restore, or split only after the "
                "complete transitive support has been processed. Candidate membership and full "
                "support are compiler-owned and will be restored intact; do not repeat support "
                "IDs. Set complete_support_reviewed true. Do not assign causal roles, define "
                "extraction, or estimate effects. Return JSON only."
            ),
        },
        {"role": "user", "content": canonical_json(request)},
    )


def _validate_rejection_evidence_fold_response(
    response: Any,
    *,
    review_input_ids: Sequence[str],
) -> dict[str, Any]:
    row = _exact_mapping(
        response,
        keys={
            "decision",
            "proposed_name",
            "measurement_summary",
            "input_dispositions",
            "complete_support_reviewed",
            "reason",
        },
        label="rejection evidence fold response",
    )
    decision = _string(row["decision"], label="decision")
    if decision not in {"uphold", "restore", "split"}:
        raise ValueError("rejection evidence fold decision is invalid")
    proposed = _string(
        row["proposed_name"],
        label="proposed_name",
        empty=decision == "uphold",
    )
    if decision == "uphold":
        if proposed:
            raise ValueError("upheld rejection cannot propose a name")
    else:
        _feature_name(proposed, label="proposed_name")
    dispositions: dict[str, dict[str, str]] = {}
    for review_id, raw in _exact_keyed_rows(
        row["input_dispositions"],
        identifiers=tuple(review_input_ids),
        label="input_dispositions",
    ):
        disposition = _exact_mapping(
            raw,
            keys={"action", "reason"},
            label=f"input_dispositions.{review_id}",
        )
        action = _string(disposition["action"], label=f"{review_id}.action")
        if action not in {"integrated", "overruled", "ambiguity_preserved"}:
            raise ValueError("rejection fold input disposition action is invalid")
        dispositions[review_id] = {
            "action": action,
            "reason": _string(disposition["reason"], label=f"{review_id}.reason"),
        }
    if row["complete_support_reviewed"] is not True:
        raise ValueError("rejection fold did not acknowledge complete transitive support")
    return _clone(
        {
            "decision": decision,
            "proposed_name": proposed,
            "measurement_summary": _string(row["measurement_summary"], label="measurement_summary"),
            "input_dispositions": dispositions,
            "complete_support_reviewed": True,
            "reason": _string(row["reason"], label="reason"),
        }
    )


def _render_extraction_evidence_page_messages(
    *,
    request: ExtractionDefinitionRequest,
    evidence: DiscoveryEvidenceItem,
    review_id: str,
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    payload = attach_hierarchical_discovery_response_contract(
        job_kind=EXTRACTION_DEFINITION_JOB,
        request={
            "job": "review_extraction_feature_evidence",
            "canonical_name": request.canonical_name,
            "review_id": review_id,
            "evidence_id": evidence.evidence_id,
            "value_shape_hypothesis": request.value_shape_hypothesis,
            "raw_evidence": {
                "evidence_id": evidence.evidence_id,
                "source_family": evidence.source_family,
                "member_ids": list(evidence.member_ids),
                "content": _clone(evidence.content),
            },
        },
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": (
                "Review exactly one authenticated support item for extraction of the named "
                "patient feature. Record the measurement and shape observation plus only aliases, "
                "units, categories, and distinctions that occur literally in this raw evidence "
                "item. This is one page of an exhaustive compiler-owned support schedule; do not "
                "infer that unseen support is absent or finalize the extraction alone. Set "
                "reviewed_evidence true. Do not add causal claims or a second feature. Return JSON "
                "only."
            ),
        },
        {"role": "user", "content": canonical_json(payload)},
    )


def _validate_extraction_evidence_page_response(
    response: Any,
    *,
    request: ExtractionDefinitionRequest,
    evidence: DiscoveryEvidenceItem,
) -> dict[str, Any]:
    row = _exact_mapping(
        response,
        keys={
            "measurement_observation",
            "shape_observation",
            "literal_aliases",
            "literal_units",
            "literal_categories",
            "literal_distinctions",
            "missing_or_ambiguous",
            "reviewed_evidence",
        },
        label="extraction evidence page response",
    )
    shape = _string(row["shape_observation"], label="shape_observation")
    if shape not in {"continuous", "categorical", "ambiguous", "unresolved"}:
        raise ValueError("extraction evidence shape observation is invalid")

    def literal_values(field: str) -> tuple[str, ...]:
        value = row[field]
        if not isinstance(value, list):
            raise TypeError(f"{field} must be one JSON list")
        values = tuple(_string(item, label=f"{field}[{index}]") for index, item in enumerate(value))
        if len(values) != len(set(values)):
            raise ValueError(f"{field} cannot contain duplicates")
        return values

    aliases = literal_values("literal_aliases")
    units = literal_values("literal_units")
    categories = literal_values("literal_categories")
    distinctions = literal_values("literal_distinctions")
    # Reuse the production literal-grounding validator against exactly this
    # page's raw evidence.  Page summaries therefore cannot smuggle invented
    # vocabulary into later folds.
    ExtractionDefinitionRequest(
        canonical_name=request.canonical_name,
        evidence=(evidence,),
        supporting_evidence_ids=(evidence.evidence_id,),
        value_shape_hypothesis=request.value_shape_hypothesis,
        allowed_aliases=aliases,
        allowed_units=units,
        allowed_categories=categories,
        allowed_distinguish_from=distinctions,
    )
    if row["reviewed_evidence"] is not True:
        raise ValueError("extraction evidence page did not acknowledge its raw evidence")
    return _clone(
        {
            "measurement_observation": _string(
                row["measurement_observation"], label="measurement_observation"
            ),
            "shape_observation": shape,
            "literal_aliases": list(aliases),
            "literal_units": list(units),
            "literal_categories": list(categories),
            "literal_distinctions": list(distinctions),
            "missing_or_ambiguous": _string(
                row["missing_or_ambiguous"], label="missing_or_ambiguous"
            ),
            "reviewed_evidence": True,
        }
    )


def _render_extraction_evidence_fold_messages(
    *,
    request: ExtractionDefinitionRequest,
    fold_index: int,
    review_inputs: Sequence[tuple[str, Mapping[str, Any]]],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    inputs = tuple(review_inputs)
    vocabulary_grounding_policy = extraction_vocabulary_grounding_policy()
    vocabulary_grounding_policy.pop("schema_version")
    payload = attach_hierarchical_discovery_response_contract(
        job_kind=EXTRACTION_DEFINITION_JOB,
        request={
            "job": "fold_extraction_evidence_definitions",
            "canonical_name": request.canonical_name,
            "value_shape_hypothesis": request.value_shape_hypothesis,
            "fold_index": fold_index,
            "review_input_ids": [review_id for review_id, _ in inputs],
            "review_inputs": [
                {"review_input_id": review_id, "review": _clone(review)}
                for review_id, review in inputs
            ],
            "planner_lookback_constraints": {
                "aliases": list(request.allowed_aliases),
                "units": list(request.allowed_units),
                "categories": list(request.allowed_categories),
                "distinguish_from": list(request.allowed_distinguish_from),
            },
            "vocabulary_grounding_policy": vocabulary_grounding_policy,
        },
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": EXTRACTION_DEFINITION_SYSTEM_PROMPT,
        },
        {"role": "user", "content": canonical_json(payload)},
    )


def _validate_extraction_evidence_fold_response(
    response: Any,
    *,
    request: ExtractionDefinitionRequest,
    review_input_ids: Sequence[str],
) -> dict[str, Any]:
    row = _exact_mapping(
        response,
        keys={
            "feature_name",
            "measurement",
            "representation",
            "aliases",
            "distinguish_from",
            "missing_or_ambiguous",
            "input_dispositions",
            "supporting_evidence_reviewed",
        },
        label="extraction evidence fold response",
    )
    dispositions: dict[str, dict[str, str]] = {}
    for review_id, raw in _exact_keyed_rows(
        row["input_dispositions"],
        identifiers=tuple(review_input_ids),
        label="input_dispositions",
    ):
        disposition = _exact_mapping(
            raw,
            keys={"action", "reason"},
            label=f"input_dispositions.{review_id}",
        )
        action = _string(disposition["action"], label=f"{review_id}.action")
        if action not in {"integrated", "not_selected", "conflict_preserved"}:
            raise ValueError("extraction fold input disposition action is invalid")
        dispositions[review_id] = {
            "action": action,
            "reason": _string(disposition["reason"], label=f"{review_id}.reason"),
        }
    definition_wire = {
        key: _clone(row[key])
        for key in (
            "feature_name",
            "measurement",
            "representation",
            "aliases",
            "distinguish_from",
            "missing_or_ambiguous",
            "supporting_evidence_reviewed",
        )
    }
    definition = validate_extraction_definition_response(definition_wire, request=request)
    return _clone(
        {
            "definition": definition,
            "fold_wire": {**definition_wire, "input_dispositions": dispositions},
        }
    )


def _validate_consolidation_allowing_empty(
    response: Any,
    *,
    source_family: str,
    candidates: Sequence[DiscoveryCandidate],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> dict[str, Any]:
    return validate_consolidation_response(
        response,
        source_family=source_family,
        candidates=candidates,
        wire_budget=wire_budget,
    )


def _render_coverage_messages(
    *,
    family: str,
    evidence: Sequence[DiscoveryEvidenceItem],
    interpretation_responses: Sequence[Mapping[str, Any]],
    consolidation_response: Mapping[str, Any],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    request = attach_hierarchical_discovery_response_contract(
        job_kind=COVERAGE_CRITIC_JOB,
        request={
            "job": "audit_architecture_chunk_coverage",
            "source_family": family,
            "evidence": [item.as_prompt_item() for item in evidence],
            "chunk_interpretations": [
                interpretation_model_view(row) for row in interpretation_responses
            ],
            "consolidation": consolidation_response,
        },
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": (
                "Audit one original evidence chunk from one completed Stage 1 architecture "
                "for semantic loss. Review every supplied evidence atom and member disposition "
                "against every consolidated concept in that architecture. Report any omitted "
                "concept, improper merge, or lost support. The supplied interpretation remains "
                "chunk-local, but the concept catalog is the complete family catalog so that a "
                "wrong cross-chunk merge or missing support relation remains discoverable. Do not "
                "compare other architectures, assign causal roles, define extraction, or use "
                "numerical signals to name a feature. reviewed_evidence_ids is keyed by every "
                "exact evidence_id with value true. Return JSON only."
            ),
        },
        {"role": "user", "content": canonical_json(request)},
    )


def _render_atomic_coverage_messages(
    *,
    family: str,
    evidence: DiscoveryEvidenceItem,
    interpretation_response: Mapping[str, Any],
    consolidation_response: Mapping[str, Any],
    canonical_names: Sequence[str],
    page_index: int,
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    atomic_review_id = f"coverage_review_{_sha({'evidence_id': evidence.evidence_id, 'page_index': page_index, 'canonical_names': list(canonical_names)})}"
    request = attach_hierarchical_discovery_response_contract(
        job_kind=COVERAGE_CRITIC_JOB,
        request={
            "job": "audit_architecture_atomic_coverage",
            "atomic_review_id": atomic_review_id,
            "evidence_id": evidence.evidence_id,
            "canonical_names": list(canonical_names),
            "source_family": family,
            "evidence": evidence.as_prompt_item(),
            "chunk_interpretation": interpretation_model_view(interpretation_response),
            "consolidation_page": _clone(consolidation_response),
        },
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": (
                "Audit this exact original evidence atom and its member dispositions against one "
                "exhaustive bounded page of the architecture's consolidated concepts. Report "
                "omitted concepts, improper merges, or lost support. Every architecture concept "
                "is scheduled on exactly one page for this evidence atom. Evidence support is "
                "fixed and compiler-derived; do not repeat support IDs. "
                "affected_canonical_names may use only the exact page names. Set "
                "reviewed_atomic_review true. Do not compare other evidence or architectures, "
                "assign causal roles, define extraction, or estimate effects. Return JSON only."
            ),
        },
        {"role": "user", "content": canonical_json(request)},
    )


def _validate_atomic_coverage_response(
    response: Any,
    *,
    evidence_id: str,
    canonical_names: Sequence[str],
) -> dict[str, Any]:
    root = _exact_mapping(
        response,
        keys={"findings", "reviewed_atomic_review"},
        label="atomic coverage response",
    )
    if root["reviewed_atomic_review"] is not True:
        raise ValueError("atomic coverage response did not acknowledge its exact review")
    findings = root["findings"]
    if not isinstance(findings, list):
        raise TypeError("atomic coverage findings must be one JSON list")
    compiler_completed_findings: list[dict[str, Any]] = []
    for raw in findings:
        if not isinstance(raw, Mapping):
            raise TypeError("atomic coverage finding must be one JSON object")
        action = raw.get("action")
        compiler_completed_findings.append(
            {
                **_clone(raw),
                "supporting_evidence_ids": ([] if action == "no_change" else [evidence_id]),
            }
        )
    normalized = validate_coverage_critic_response(
        {
            "findings": compiler_completed_findings,
            "reviewed_evidence_ids": {evidence_id: True},
        },
        evidence_ids=(evidence_id,),
        canonical_names=tuple(canonical_names),
    )
    return _clone(
        {
            **normalized,
            "atomic_wire_response": _clone(root),
        }
    )


def _chunk_scoped_consolidation_view(
    *,
    consolidation_response: Mapping[str, Any],
    chunk_candidates: Sequence[DiscoveryCandidate],
    chunk_evidence_ids: Sequence[str],
) -> dict[str, Any]:
    """Project a validated family consolidation onto one original raw-evidence chunk."""

    candidate_ids = {candidate.candidate_id for candidate in chunk_candidates}
    evidence_ids = set(chunk_evidence_ids)
    concepts: list[dict[str, Any]] = []
    for raw in consolidation_response["canonical_concepts"]:
        members = [
            candidate_id
            for candidate_id in raw["member_candidate_ids"]
            if candidate_id in candidate_ids
        ]
        if not members:
            continue
        support = [
            evidence_id
            for evidence_id in raw["supporting_evidence_ids"]
            if evidence_id in evidence_ids
        ]
        concepts.append(
            {
                **raw,
                "member_candidate_ids": members,
                "supporting_evidence_ids": support,
            }
        )
    dispositions = [
        row
        for row in consolidation_response["candidate_dispositions"]
        if row["candidate_id"] in candidate_ids
    ]
    if {row["candidate_id"] for row in dispositions} != candidate_ids:
        raise RuntimeError("chunk-scoped consolidation lost a chunk candidate disposition")
    projected_support = {
        evidence_id for concept in concepts for evidence_id in concept["supporting_evidence_ids"]
    }
    expected_support = {
        evidence_id
        for candidate in chunk_candidates
        for evidence_id in candidate.supporting_evidence_ids
    }
    if projected_support != expected_support:
        raise RuntimeError("chunk-scoped consolidation lost chunk evidence support")
    return _clone(
        {
            "canonical_concepts": concepts,
            "candidate_dispositions": dispositions,
        }
    )


def _compile_coverage_revision_findings(
    *,
    consolidation_response: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
    source_family: str,
    candidates: Sequence[DiscoveryCandidate],
    wire_budget: HierarchyWireBudget,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one bounded critic revision deterministically without another selector."""

    corrected = _clone(consolidation_response)
    concepts = corrected["canonical_concepts"]
    if not isinstance(concepts, list):
        raise TypeError("coverage revision compiler requires canonical concept rows")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    used_names = {str(concept["canonical_name"]) for concept in concepts}
    events: list[dict[str, Any]] = []

    for finding_index, raw in enumerate(findings):
        finding = _clone(raw)
        action = str(finding["action"])
        if action == "no_change":
            continue
        support = tuple(str(value) for value in finding["supporting_evidence_ids"])
        affected_names = tuple(str(value) for value in finding["affected_canonical_names"])
        affected = [concept for concept in concepts if concept["canonical_name"] in affected_names]
        event: dict[str, Any] = {
            "finding_index": finding_index,
            "action": action,
            "affected_canonical_names": list(affected_names),
            "supporting_evidence_ids": list(support),
        }
        if action == "restore_support":
            for concept in affected:
                concept["supporting_evidence_ids"] = list(
                    dict.fromkeys((*concept["supporting_evidence_ids"], *support))
                )
            event["compiled_canonical_name"] = ""
            events.append(event)
            continue

        proposed = str(finding["proposed_name"])
        derived, disambiguation = _derive_unique_integration_name(
            proposed=proposed,
            slot=f"coverage_{finding_index + 1:03d}",
            used=used_names,
            wire_budget=wire_budget,
        )
        used_names.add(derived)
        if action == "split_concept":
            source_members = tuple(
                dict.fromkeys(
                    member_id
                    for concept in affected
                    for member_id in concept["member_candidate_ids"]
                )
            )
            member_ids = (
                tuple(
                    member_id
                    for member_id in source_members
                    if member_id in candidate_by_id
                    and set(candidate_by_id[member_id].supporting_evidence_ids) & set(support)
                )
                or source_members
            )
            shapes = {str(concept["value_shape_hypothesis"]) for concept in affected}
        else:
            member_ids = tuple(
                candidate.candidate_id
                for candidate in candidates
                if set(candidate.supporting_evidence_ids) & set(support)
            )
            shapes = {candidate_by_id[member_id].value_shape_hypothesis for member_id in member_ids}
        shape = next(iter(shapes)) if len(shapes) == 1 else "ambiguous"
        concepts.append(
            {
                "canonical_name": derived,
                "description": str(finding["description"]),
                "member_candidate_ids": list(member_ids),
                "supporting_evidence_ids": list(support),
                "source_families": [source_family],
                "value_shape_hypothesis": shape,
                "unresolved_ambiguity": "",
            }
        )
        event["compiled_canonical_name"] = derived
        if disambiguation is not None:
            event["canonical_name_disambiguation"] = disambiguation
        events.append(event)

    prior_audit = corrected.get("wire_normalization_audit")
    corrected["wire_normalization_audit"] = {
        "prior_consolidation_audit": _clone(prior_audit),
        "coverage_revision_compiler_version": "bounded_coverage_revision_compiler_v1",
        "events": events,
    }
    return corrected, {
        "compiler_version": "bounded_coverage_revision_compiler_v1",
        "finding_count": len(findings),
        "events": events,
        "corrected_consolidation_sha256": _sha(corrected),
    }


def _render_integration_messages(
    *,
    dossiers: Sequence[ArchitectureDossier],
    planner_response: Mapping[str, Any],
    lookback: Sequence[Mapping[str, Any]],
    maximum_integrated_features: int,
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    architecture_context = cross_architecture_planner_context(dossiers)
    architecture_context.pop("schema_version")
    request = attach_hierarchical_discovery_response_contract(
        job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
        request={
            "job": "integrate_cross_architecture_candidates",
            "architecture_context": architecture_context,
            "validated_planner_response": {
                "provisional_groups": planner_response["provisional_groups"],
                "raw_evidence_requests": planner_response["raw_evidence_requests"],
            },
            "requested_raw_evidence_lookback": list(lookback),
            "maximum_integrated_features": maximum_integrated_features,
        },
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": (
                "Integrate the ten compact architecture dossiers using the validated provisional "
                "groups and only the exact requested raw evidence supplied here. Preserve distinct "
                "measurements. candidate_routes is keyed by every exact candidate_id; route each "
                "candidate to reject or to one compiler-owned integration slot, then define every "
                "fixed slot. Unused slot definitions are ignored. Reject only when evidence cannot "
                "support a concrete patient-level feature. Candidate membership, evidence support, "
                "source families, value shape, and extraction constraints are compiler-derived. "
                "Do not assign causal roles, define extraction, estimate effects, or use numerical "
                "manifests to name features. Return JSON only."
            ),
        },
        {"role": "user", "content": canonical_json(request)},
    )


def _derive_unique_integration_name(
    *,
    proposed: str,
    slot: str,
    used: set[str],
    wire_budget: HierarchyWireBudget,
) -> tuple[str, dict[str, str] | None]:
    """Preserve distinct active slots under a bounded deterministic name."""

    if not isinstance(wire_budget, HierarchyWireBudget):
        raise TypeError(
            "duplicate-name compilation requires an explicit HierarchyWireBudget"
        )
    if proposed not in used:
        return proposed, None
    ordinal = 1
    while True:
        suffix = f"_{slot}" if ordinal == 1 else f"_{slot}_{ordinal}"
        available = wire_budget.max_generated_name_chars - len(suffix)
        if available < 1:
            raise ValueError(
                "generated-name wire budget cannot encode the required "
                "compiler-owned suffix"
            )
        prefix = proposed[:available].rstrip("_") or "feature"
        derived = f"{prefix}{suffix}"
        if derived not in used:
            return derived, {
                "integration_slot": slot,
                "proposed_canonical_name": proposed,
                "derived_canonical_name": derived,
                "reason": "distinct active slots cannot share canonical identity",
            }
        ordinal += 1


def validate_cross_architecture_integration_response(
    response: Any,
    *,
    dossiers: Sequence[ArchitectureDossier],
    lookback: Sequence[Mapping[str, Any]],
    maximum_integrated_features: int,
    wire_budget: HierarchyWireBudget,
) -> tuple[dict[str, Any], tuple[IntegratedCanonicalFeature, ...]]:
    """Compile exact candidate routes into lossless integrated features."""

    cross_architecture_planner_context(dossiers)
    if (
        isinstance(maximum_integrated_features, bool)
        or not isinstance(maximum_integrated_features, int)
        or maximum_integrated_features < 1
    ):
        raise ValueError("maximum_integrated_features must be a positive integer")
    candidates = {
        candidate.candidate_id: candidate
        for dossier in dossiers
        for candidate in dossier.architecture_candidates
    }
    candidate_order = tuple(candidates)
    seen_lookback_ids: set[str] = set()
    for index, raw in enumerate(lookback):
        row = _exact_mapping(
            raw,
            keys={"evidence_id", "source_family", "observable_axes", "member_ids", "content"},
            label=f"lookback[{index}]",
        )
        evidence_id = _identifier(row["evidence_id"], label=f"lookback[{index}].evidence_id")
        if evidence_id in seen_lookback_ids:
            raise ValueError("lookback evidence IDs cannot repeat")
        seen_lookback_ids.add(evidence_id)

    slots = tuple(
        f"integration_slot_{index:03d}"
        for index in range(
            1,
            min(maximum_integrated_features, len(candidate_order)) + 1,
        )
    )
    root = _exact_mapping(
        response,
        keys={"candidate_routes", "slot_definitions"},
        label="cross-architecture integration response",
    )
    route_by_candidate: dict[str, str] = {}
    reason_by_candidate: dict[str, str] = {}
    route_audit: list[dict[str, str]] = []
    for candidate_id, raw_route in _exact_keyed_rows(
        root["candidate_routes"],
        identifiers=candidate_order,
        label="candidate_routes",
    ):
        label = f"candidate_routes.{candidate_id}"
        route_row = _exact_mapping(raw_route, keys={"route", "reason"}, label=label)
        route = _string(route_row["route"], label=f"{label}.route")
        if route != "reject" and route not in slots:
            raise ValueError(f"{label}.route is not compiler-owned")
        reason = _string(route_row["reason"], label=f"{label}.reason")
        route_by_candidate[candidate_id] = route
        reason_by_candidate[candidate_id] = reason
        route_audit.append({"candidate_id": candidate_id, "route": route})

    definitions: dict[str, dict[str, str]] = {}
    definition_audit: list[dict[str, str]] = []
    for slot, raw_definition in _exact_keyed_rows(
        root["slot_definitions"],
        identifiers=slots,
        label="slot_definitions",
    ):
        label = f"slot_definitions.{slot}"
        definition = _exact_mapping(
            raw_definition,
            keys={"canonical_name", "description", "unresolved_ambiguity"},
            label=label,
        )
        parsed = {
            "canonical_name": _feature_name(
                definition["canonical_name"], label=f"{label}.canonical_name"
            ),
            "description": _string(definition["description"], label=f"{label}.description"),
            "unresolved_ambiguity": _string(
                definition["unresolved_ambiguity"],
                label=f"{label}.unresolved_ambiguity",
                empty=True,
            ),
        }
        definitions[slot] = parsed
        definition_audit.append({"integration_slot": slot, **parsed})

    assigned_slots = set(route_by_candidate.values()) - {"reject"}
    active_slots = tuple(slot for slot in slots if slot in assigned_slots)
    used_names: set[str] = set()
    name_by_slot: dict[str, str] = {}
    disambiguations: list[dict[str, str]] = []
    for slot in active_slots:
        name, disambiguation = _derive_unique_integration_name(
            proposed=definitions[slot]["canonical_name"],
            slot=slot,
            used=used_names,
            wire_budget=wire_budget,
        )
        used_names.add(name)
        name_by_slot[slot] = name
        if disambiguation is not None:
            disambiguations.append(disambiguation)

    features: list[IntegratedCanonicalFeature] = []
    normalized_features: list[dict[str, Any]] = []
    for slot in active_slots:
        name = name_by_slot[slot]
        members = tuple(
            candidate_id
            for candidate_id in candidate_order
            if route_by_candidate[candidate_id] == slot
        )
        support = tuple(
            dict.fromkeys(
                evidence_id
                for member in members
                for evidence_id in candidates[member].supporting_evidence_ids
            )
        )
        families = tuple(
            dict.fromkeys(
                family for member in members for family in candidates[member].source_families
            )
        )
        member_shapes = {candidates[member].value_shape_hypothesis for member in members}
        shape = next(iter(member_shapes)) if len(member_shapes) == 1 else "ambiguous"
        description = definitions[slot]["description"]
        ambiguity = definitions[slot]["unresolved_ambiguity"]
        features.append(
            IntegratedCanonicalFeature(
                canonical_name=name,
                description=description,
                member_candidate_ids=members,
                supporting_evidence_ids=support,
                source_families=families,
                value_shape_hypothesis=shape,
                unresolved_ambiguity=ambiguity,
            )
        )
        normalized_features.append(
            {
                "canonical_name": name,
                "description": description,
                "member_candidate_ids": list(members),
                "supporting_evidence_ids": list(support),
                "source_families": list(families),
                "value_shape_hypothesis": shape,
                "unresolved_ambiguity": ambiguity,
                "extraction_constraints": {
                    "aliases": [],
                    "units": [],
                    "categories": [],
                    "distinguish_from": [],
                },
            }
        )

    normalized_dispositions = [
        {
            "candidate_id": candidate_id,
            "decision": ("reject" if route_by_candidate[candidate_id] == "reject" else "accept"),
            "canonical_name": (
                ""
                if route_by_candidate[candidate_id] == "reject"
                else name_by_slot[route_by_candidate[candidate_id]]
            ),
            "reason": reason_by_candidate[candidate_id],
        }
        for candidate_id in candidate_order
    ]
    normalized_root = {
        "canonical_features": normalized_features,
        "candidate_dispositions": normalized_dispositions,
        "wire_normalization_audit": {
            "audit_version": "fixed_slot_integration_normalization_audit_v1",
            "slot_policy": "candidate_route_then_compiler_derived_feature_relations_v1",
            "derived_relation_fields": [
                "canonical_features.member_candidate_ids",
                "canonical_features.supporting_evidence_ids",
                "canonical_features.source_families",
                "canonical_features.value_shape_hypothesis",
                "canonical_features.extraction_constraints",
                "candidate_dispositions.decision",
                "candidate_dispositions.canonical_name",
            ],
            "candidate_routes": route_audit,
            "slot_definitions": definition_audit,
            "active_slots": list(active_slots),
            "unused_slots": [slot for slot in slots if slot not in assigned_slots],
            "canonical_name_disambiguations": disambiguations,
            "maximum_integrated_features": maximum_integrated_features,
        },
    }
    return _clone(normalized_root), tuple(features)


def _render_rejection_critic_messages(
    *,
    candidates: Mapping[str, DiscoveryCandidate],
    integration_response: Mapping[str, Any],
    lookback: Sequence[Mapping[str, Any]],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    disposition_by_id = {
        row["candidate_id"]: row for row in integration_response["candidate_dispositions"]
    }
    visible_evidence_ids = {str(row["evidence_id"]) for row in lookback}
    rejected_candidates = []
    for candidate_id, candidate in candidates.items():
        candidate_view = candidate.as_prompt_item()
        candidate_view["supporting_evidence_ids"] = [
            evidence_id
            for evidence_id in candidate.supporting_evidence_ids
            if evidence_id in visible_evidence_ids
        ]
        rejected_candidates.append(
            {
                **candidate_view,
                "integration_rejection": disposition_by_id[candidate_id],
            }
        )
    request = attach_hierarchical_discovery_response_contract(
        job_kind=REJECTION_CRITIC_JOB,
        request={
            "job": "audit_every_rejected_candidate",
            "rejected_candidates": rejected_candidates,
            "requested_raw_evidence_lookback": [
                row
                for row in lookback
                if row["evidence_id"]
                in {
                    evidence_id
                    for candidate in candidates.values()
                    for evidence_id in candidate.supporting_evidence_ids
                }
            ],
        },
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": (
                "Independently reconsider every rejected candidate. Preserve candidate boundaries "
                "and use only its bounded deterministic support references plus the exact raw "
                "lookback. The complete support relation remains compiler-owned. Do not assign "
                "causal roles, define extraction, estimate effects, or invent evidence. "
                "reconsiderations is keyed by every exact rejected candidate_id."
            ),
        },
        {"role": "user", "content": canonical_json(request)},
    )


def _compile_rejection_reconsideration(
    *,
    integration_response: Mapping[str, Any],
    integrated_features: Sequence[IntegratedCanonicalFeature],
    candidate: DiscoveryCandidate,
    reconsideration: Mapping[str, Any],
    evidence_by_id: Mapping[str, DiscoveryEvidenceItem],
    maximum_integrated_features: int,
    wire_budget: HierarchyWireBudget,
) -> tuple[dict[str, Any], tuple[IntegratedCanonicalFeature, ...], dict[str, Any]]:
    """Compile one restore/split critic decision into the frozen integration state."""

    decision = str(reconsideration["decision"])
    if decision not in {"restore", "split"}:
        raise ValueError("rejection continuation requires restore or split")
    current = tuple(integrated_features)
    support = tuple(candidate.supporting_evidence_ids)
    if not support:
        raise ValueError("rejection continuation candidate lacks compiler-owned support")
    unknown = set(support) - set(evidence_by_id)
    if unknown:
        raise ValueError("rejection continuation cites evidence outside the catalog")
    used_names = {feature.canonical_name for feature in current}
    name, disambiguation = _derive_unique_integration_name(
        proposed=str(reconsideration["proposed_name"]),
        slot=f"rejection_{_sha(candidate.candidate_id)[:8]}",
        used=used_names,
        wire_budget=wire_budget,
    )
    families = tuple(candidate.source_families)
    feature = IntegratedCanonicalFeature(
        canonical_name=name,
        description=candidate.description,
        member_candidate_ids=(candidate.candidate_id,),
        supporting_evidence_ids=support,
        source_families=families,
        value_shape_hypothesis=candidate.value_shape_hypothesis,
        unresolved_ambiguity=candidate.unresolved_ambiguity,
    )
    revised = _clone(integration_response)
    dispositions = revised.get("candidate_dispositions")
    canonical_features = revised.get("canonical_features")
    if not isinstance(dispositions, list) or not isinstance(canonical_features, list):
        raise TypeError("rejection continuation received an invalid integration projection")
    matching = [row for row in dispositions if row["candidate_id"] == candidate.candidate_id]
    if len(matching) != 1 or matching[0]["decision"] != "reject":
        raise ValueError("rejection continuation candidate is not exactly one rejection")
    matching[0].update(
        {
            "decision": "accept",
            "canonical_name": name,
            "reason": str(reconsideration["reason"]),
        }
    )
    canonical_features.append(
        {
            "canonical_name": name,
            "description": candidate.description,
            "member_candidate_ids": [candidate.candidate_id],
            "supporting_evidence_ids": list(support),
            "source_families": list(families),
            "value_shape_hypothesis": candidate.value_shape_hypothesis,
            "unresolved_ambiguity": candidate.unresolved_ambiguity,
            "extraction_constraints": {
                "aliases": [],
                "units": [],
                "categories": [],
                "distinguish_from": [],
            },
        }
    )
    event = {
        "compiler_version": "lossless_rejection_reconsideration_compiler_v2",
        "candidate_id": candidate.candidate_id,
        "decision": decision,
        "compiled_canonical_name": name,
        "supporting_evidence_ids": list(support),
        "complete_candidate_support_sha256": _sha(list(candidate.supporting_evidence_ids)),
        "complete_candidate_support_restored_without_sampling": True,
        "legacy_direct_slot_bound": maximum_integrated_features,
        "legacy_direct_slot_bound_did_not_truncate_reconsideration": True,
        "canonical_name_disambiguation": disambiguation,
    }
    prior_audit = revised.get("wire_normalization_audit")
    if not isinstance(prior_audit, Mapping):
        raise TypeError("integration wire normalization audit is missing")
    compiler_events = list(prior_audit.get("rejection_reconsideration_events", []))
    compiler_events.append(event)
    revised["wire_normalization_audit"] = {
        **_clone(prior_audit),
        "rejection_reconsideration_compiler_version": (
            "lossless_rejection_reconsideration_compiler_v2"
        ),
        "rejection_reconsideration_events": compiler_events,
    }
    return revised, (*current, feature), event


def _render_extraction_messages(
    *,
    request: ExtractionDefinitionRequest,
    model_evidence: Sequence[DiscoveryEvidenceItem] | None = None,
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    visible_evidence = tuple(request.evidence if model_evidence is None else model_evidence)
    visible_ids = tuple(item.evidence_id for item in visible_evidence)
    if not set(visible_ids) <= set(request.supporting_evidence_ids):
        raise ValueError("extraction model evidence is outside compiler-owned feature support")
    vocabulary_grounding_policy = extraction_vocabulary_grounding_policy()
    vocabulary_grounding_policy.pop("schema_version")
    payload = attach_hierarchical_discovery_response_contract(
        job_kind=EXTRACTION_DEFINITION_JOB,
        request={
            "job": "define_one_extraction_feature",
            "canonical_name": request.canonical_name,
            "value_shape_hypothesis": request.value_shape_hypothesis,
            "supporting_evidence_ids": list(visible_ids),
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "source_family": item.source_family,
                    "member_ids": list(item.member_ids),
                    "content": _clone(item.content),
                }
                for item in visible_evidence
            ],
            "planner_lookback_constraints": {
                "aliases": list(request.allowed_aliases),
                "units": list(request.allowed_units),
                "categories": list(request.allowed_categories),
                "distinguish_from": list(request.allowed_distinguish_from),
            },
            "vocabulary_grounding_policy": vocabulary_grounding_policy,
        },
        wire_budget=wire_budget,
    )
    return (
        {
            "role": "system",
            "content": EXTRACTION_DEFINITION_SYSTEM_PROMPT,
        },
        {"role": "user", "content": canonical_json(payload)},
    )


class HierarchicalAllArchitectureDiscoveryOrchestrator:
    """Compile, precommit, and execute the strict hierarchical workflow."""

    def __init__(
        self,
        *,
        catalog: RoleNeutralEvidenceCatalog,
        chunk_plan: ArchitectureChunkPlan,
        family_explanations: Mapping[str, str],
        direct_numerical_bindings: Sequence[DirectNumericalDossierBinding],
        runner_identity: Mapping[str, Any],
        config: HierarchicalDiscoveryConfig | None = None,
        job_cache: AuthenticatedHierarchicalDiscoveryJobCache | None = None,
    ) -> None:
        validate_role_neutral_catalog(catalog)
        delivery_audit = audit_complete_architecture_delivery(catalog, chunk_plan)
        if delivery_audit["all_catalog_atoms_delivered_exactly_once"] is not True:
            raise ValueError("architecture chunk delivery is incomplete")
        if delivery_audit["non_grounding_numerical_summaries_delivered"] is not False:
            raise ValueError("non-grounding numerical summaries entered discovery chunks")
        explanations = dict(family_explanations)
        if set(explanations) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("family_explanations must cover exactly all active architectures")
        for family, explanation in explanations.items():
            _string(explanation, label=f"family_explanations[{family}]")
            _assert_no_policy_prompt_text(({"role": "user", "content": explanation},))
        bindings = tuple(direct_numerical_bindings)
        if len(bindings) != len(ACTIVE_STAGE1_CONCEPT_FAMILIES):
            raise ValueError("direct numerical bindings must cover exactly ten architectures")
        binding_by_family = {binding.source_family: binding for binding in bindings}
        if len(binding_by_family) != len(bindings) or (
            set(binding_by_family) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET
        ):
            raise ValueError("direct numerical bindings must be unique and complete")
        contract_identities = {
            (
                binding.direct_numerical_contract_kind,
                binding.direct_numerical_contract_sha256,
            )
            for binding in bindings
        }
        if len(contract_identities) != 1:
            raise ValueError("all dossier bindings must cite one direct numerical contract")
        self.direct_numerical_contract_kind, self.direct_numerical_contract_sha256 = next(
            iter(contract_identities)
        )
        normalized_runner_identity = _clone(runner_identity)
        if not isinstance(normalized_runner_identity, Mapping) or not normalized_runner_identity:
            raise ValueError("runner_identity must be one non-empty JSON object")

        self.catalog = catalog
        self.chunk_plan = chunk_plan
        self.family_explanations = {
            family: explanations[family] for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        }
        self.direct_numerical_bindings = tuple(
            binding_by_family[family] for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        )
        self._binding_by_family = binding_by_family
        self._runner_identity_json = canonical_json(normalized_runner_identity)
        self.config = config or HierarchicalDiscoveryConfig()
        if not isinstance(self.config, HierarchicalDiscoveryConfig):
            raise TypeError("config must be HierarchicalDiscoveryConfig")
        if (
            self.chunk_plan.max_semantic_member_ids_per_chunk
            != self.config.max_semantic_member_ids_per_chunk
        ):
            raise ValueError(
                "chunk plan max_semantic_member_ids_per_chunk differs from the " "hierarchy config"
            )
        if (
            self.chunk_plan.max_atoms_per_chunk
            > self.config.wire_budget.max_interpret_atoms_per_job
        ):
            raise ValueError(
                "chunk plan max_atoms_per_chunk exceeds the authenticated interpret "
                "response budget"
            )
        if job_cache is not None and not isinstance(
            job_cache, AuthenticatedHierarchicalDiscoveryJobCache
        ):
            raise TypeError("job_cache must be AuthenticatedHierarchicalDiscoveryJobCache")
        self.job_cache = job_cache
        self._implementation_file_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._implementation_bundle = hierarchical_discovery_implementation_bundle()
        self._delivery_audit = _clone(delivery_audit)
        self._evidence_by_id = {
            atom.evidence_id: atom.as_discovery_item() for atom in catalog.atoms
        }
        self._family_evidence = {
            family: tuple(
                atom.as_discovery_item() for atom in catalog.atoms if atom.source_family == family
            )
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        }
        if any(not rows for rows in self._family_evidence.values()):
            raise ValueError("every active architecture must contain concept-bearing evidence")
        self._initial_job_by_chunk_id: dict[str, DiscoveryJsonJob] = {}
        initial_jobs: list[DiscoveryJsonJob] = []
        for chunk in chunk_plan.chunks:
            job = self._build_interpret_job(chunk)
            self._initial_job_by_chunk_id[chunk.chunk_id] = job
            initial_jobs.append(job)
        self.initial_job_ledger = DiscoveryJobLedger.build(initial_jobs)
        self._audit_initial_prompt_delivery()
        self.precommit = HierarchicalDiscoveryPrecommit.create(self._offline_packet())

    @property
    def runner_identity(self) -> dict[str, Any]:
        return json.loads(self._runner_identity_json)

    @property
    def implementation_file_sha256(self) -> str:
        return self._implementation_file_sha256

    @property
    def implementation_bundle(self) -> dict[str, Any]:
        return _clone(self._implementation_bundle)

    @property
    def implementation_bundle_sha256(self) -> str:
        return str(self._implementation_bundle["implementation_bundle_sha256"])

    @property
    def cache_execution_metadata(self) -> tuple[dict[str, Any], ...]:
        if self.job_cache is None:
            return ()
        return self.job_cache.execution_metadata

    def _assert_implementation_bundle_unchanged(
        self, *, context: str, refresh_local_validator: bool = False
    ) -> None:
        current_file_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if current_file_sha256 != self._implementation_file_sha256:
            raise ValueError(f"hierarchy implementation changed {context}")
        if refresh_local_validator and (
            local_json_schema_validator_identity(refresh=True)
            != self._implementation_bundle["local_json_schema_validator"]
        ):
            raise ValueError(f"local JSON-Schema validator changed {context}")
        current_bundle = hierarchical_discovery_implementation_bundle()
        if current_bundle != self._implementation_bundle:
            raise ValueError(f"hierarchy implementation dependency bundle changed {context}")

    def _create_job(
        self,
        *,
        job_kind: str,
        scope: str,
        dependencies: Sequence[str],
        settings: DiscoveryJobSettings,
        messages: Sequence[Mapping[str, Any]],
        input_bindings: Mapping[str, Any],
    ) -> DiscoveryJsonJob:
        self._assert_implementation_bundle_unchanged(context="during job compilation")
        bindings = _clone(input_bindings)
        if HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING in bindings:
            raise ValueError(
                f"input_bindings reserve "
                f"{HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING!r}"
            )
        bindings[HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING] = (
            self.implementation_bundle_sha256
        )
        return DiscoveryJsonJob.create(
            job_kind=job_kind,
            scope=scope,
            dependencies=dependencies,
            settings=settings,
            messages=messages,
            input_bindings=bindings,
        )

    def _selector_settings(self) -> DiscoveryJobSettings:
        return DiscoveryJobSettings.selector(
            self.config.selector_thinking_token_budget
        )

    def _build_interpret_job(self, chunk: ArchitectureEvidenceChunk) -> DiscoveryJsonJob:
        evidence = tuple(self._evidence_by_id[str(row["evidence_id"])] for row in chunk.evidence)
        messages = render_interpret_evidence_chunk_messages(
            family_explanation=self.family_explanations[chunk.source_family],
            evidence=evidence,
            wire_budget=self.config.wire_budget,
        )
        return self._create_job(
            job_kind=INTERPRET_CHUNK_JOB,
            scope=f"{chunk.source_family}.chunk_{chunk.chunk_index:03d}",
            dependencies=(),
            settings=self._selector_settings(),
            messages=messages,
            input_bindings={
                "catalog_sha256": self.catalog.catalog_sha256,
                "chunk_plan_sha256": self.chunk_plan.plan_sha256,
                "chunk_id": chunk.chunk_id,
                "source_family": chunk.source_family,
            },
        )

    def _audit_initial_prompt_delivery(self) -> None:
        expected = {
            item.evidence_id: item.as_prompt_item() for item in self._evidence_by_id.values()
        }
        observed: dict[str, Any] = {}
        for job in self.initial_job_ledger.jobs:
            payload = json.loads(job.messages[1]["content"])
            families = {row["source_family"] for row in payload["evidence"]}
            if families != {job.input_bindings["source_family"]}:
                raise ValueError("one interpretation job contains mixed architectures")
            for row in payload["evidence"]:
                evidence_id = row["evidence_id"]
                if evidence_id in observed:
                    raise ValueError("one raw evidence atom entered multiple interpretation jobs")
                observed[evidence_id] = row
        if observed != expected:
            raise ValueError("interpretation jobs do not preserve every raw evidence atom")

    def _offline_packet(self) -> dict[str, Any]:
        return {
            "schema_version": HIERARCHICAL_DISCOVERY_PRECOMMIT_VERSION,
            "orchestrator_version": HIERARCHICAL_DISCOVERY_ORCHESTRATOR_VERSION,
            "orchestrator_implementation_file_sha256": (self._implementation_file_sha256),
            "orchestrator_implementation_bundle": self.implementation_bundle,
            "orchestrator_implementation_bundle_sha256": self.implementation_bundle_sha256,
            "catalog_binding": {
                "catalog_sha256": self.catalog.catalog_sha256,
                "split_fingerprint": self.catalog.split_fingerprint,
                "outer_fold": self.catalog.outer_fold,
                "scope": self.catalog.scope,
                "inner_fold": self.catalog.inner_fold,
                "atom_count": len(self.catalog.atoms),
            },
            "chunk_plan_binding": {
                "plan_sha256": self.chunk_plan.plan_sha256,
                "chunk_count": len(self.chunk_plan.chunks),
                "max_semantic_member_ids_per_chunk": (
                    self.chunk_plan.max_semantic_member_ids_per_chunk
                ),
                "delivery_audit": self._delivery_audit,
            },
            "runner_identity": self.runner_identity,
            "config": self.config.as_dict(),
            "response_repair_policy": discovery_response_repair_policy_identity(),
            "direct_numerical_contract_binding": {
                "direct_numerical_contract_kind": self.direct_numerical_contract_kind,
                "direct_numerical_contract_sha256": self.direct_numerical_contract_sha256,
                "model_facing": False,
            },
            "dossier_direct_numerical_bindings": [
                binding.as_dossier_dict() for binding in self.direct_numerical_bindings
            ],
            "initial_job_ledger": self.initial_job_ledger.as_dict(),
            "downstream_contract": {
                "discovery_interface_schema_version": DISCOVERY_INTERFACE_SCHEMA_VERSION,
                "wire_normalization_version": DISCOVERY_WIRE_NORMALIZATION_VERSION,
                "response_contract_version": HIERARCHICAL_DISCOVERY_RESPONSE_CONTRACT_VERSION,
                "exact_coverage_representation": (
                    HIERARCHICAL_DISCOVERY_EXACT_COVERAGE_REPRESENTATION
                ),
                "stage_order": [
                    INTERPRET_CHUNK_JOB,
                    CONSOLIDATE_ARCHITECTURE_JOB,
                    COVERAGE_CRITIC_JOB,
                    CROSS_ARCHITECTURE_PLANNER_JOB,
                    CROSS_ARCHITECTURE_INTEGRATION_JOB,
                    REJECTION_CRITIC_JOB,
                    EXTRACTION_DEFINITION_JOB,
                ],
                "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
                "cross_architecture_dossier_count": len(ACTIVE_STAGE1_CONCEPT_FAMILIES),
                "raw_lookback": {
                    "cross_architecture_integration": (
                        "every_group_support_item_once_then_recursive_8_input_folds"
                    ),
                    "rejection_critic": (
                        "every_rejected_candidate_support_item_once_then_recursive_8_input_folds"
                    ),
                    "extraction_definition": (
                        "every_accepted_feature_support_item_once_then_recursive_8_input_folds"
                    ),
                },
                "lossless_raw_evidence_hierarchy": {
                    "one_raw_evidence_item_per_page": True,
                    "maximum_fold_inputs": (
                        self.config.wire_budget.max_definition_fold_inputs
                    ),
                    "maximum_fresh_inputs_after_first_fold": (
                        self.config.wire_budget.max_definition_fold_inputs - 1
                    ),
                    "every_input_receives_an_explicit_disposition": True,
                    "semantic_sampling_or_truncation": False,
                    "legacy_configured_lookback_and_feature_caps_are_not_semantic_limits": True,
                },
                "rejection_review": (
                    "one_page_per_support_item_and_complete_fold_disposition_per_rejected_candidate"
                ),
                "role_routing": "deterministic_observable_axis_rules_after_integration",
                "extraction_definition": (
                    "one_accepted_feature_per_exhaustive_page_and_fold_hierarchy"
                ),
                "extraction_vocabulary_grounding": (extraction_vocabulary_grounding_policy()),
                "selector_settings": self._selector_settings().as_dict(),
                "extraction_settings": DiscoveryJobSettings.extraction().as_dict(),
            },
            "assurances": {
                "raw_atoms_delivered_exactly_once": True,
                "mixed_architecture_interpretation_jobs": False,
                "direct_row_level_numerical_values_accepted": False,
                "direct_numerical_concept_grounding": False,
                "inactive_sparse_query_family_present": False,
                "network_transport_implemented": False,
                "bounded_response_repair_implemented": True,
                "unvalidated_response_cache_write_allowed": False,
                "implementation_bundle_authenticated": True,
                "cache_validator_identity_is_implementation_bundle": True,
                "raw_wire_and_normalized_responses_authenticated_separately": True,
                "retired_global_evidence_dump_path_admitted": False,
                "retired_exact_coverage_identifier_arrays_admitted": False,
            },
        }

    def render_offline_precommit(self, *, indent: int = 2) -> str:
        return self.precommit.render_json(indent=indent)

    def _assert_runner_identity(self, runner: JsonDiscoveryJobRunner) -> None:
        if not callable(getattr(runner, "identity", None)) or not callable(
            getattr(runner, "run_json", None)
        ):
            raise TypeError("runner must implement identity() and run_json()")
        current = _clone(runner.identity())
        if canonical_json(current) != self._runner_identity_json:
            raise ValueError("runner identity differs from the reviewed precommit")

    def _run(
        self,
        *,
        runner: JsonDiscoveryJobRunner,
        job: DiscoveryJsonJob,
        validator: Any,
    ) -> tuple[dict[str, Any], ValidatedDiscoveryJobResult]:
        rendered_byte_count = len(job.rendered_messages_bytes)
        if rendered_byte_count > self.config.max_rendered_prompt_bytes:
            raise ValueError(
                "rendered discovery prompt exceeds the configured "
                f"{self.config.max_rendered_prompt_bytes}-byte guard"
            )
        self._assert_runner_identity(runner)
        self._assert_implementation_bundle_unchanged(context="after precommit")

        def validate_wire(raw_wire: Any) -> Mapping[str, Any]:
            _validate_local_discovery_wire_schema(job=job, response=raw_wire)
            try:
                normalized = validator(_clone(raw_wire))
            except (TypeError, ValueError) as exc:
                raise DiscoverySemanticNormalizationFailure(
                    "schema-valid discovery response could not be safely normalized"
                ) from exc
            if not isinstance(normalized, Mapping):
                raise DiscoverySemanticNormalizationFailure(
                    "discovery semantic normalizer did not return one JSON object"
                )
            return normalized

        if self.job_cache is not None:
            replay = self.job_cache.replay_validated(
                job=job,
                hierarchy_inner_precommit_sha256=self.precommit.precommit_sha256,
                runner_identity=self.runner_identity,
                validator_code_sha256=self.implementation_bundle_sha256,
                validator=validate_wire,
            )
            if replay is not None:
                self._assert_runner_identity(runner)
                self._assert_implementation_bundle_unchanged(context="during cache replay")
                detached = _clone(replay.validated_response)
                trace = _validated_response_attempt_trace(
                    logical_job=job,
                    validated_response_sha256=_sha(detached),
                    trace=replay.response_attempt_trace,
                )
                return detached, ValidatedDiscoveryJobResult.create(
                    job=job,
                    validated_response=detached,
                    response_attempt_trace=trace,
                )
        repair_job: DiscoveryJsonJob | None = None
        attempts: list[dict[str, Any]] = []
        try:
            raw = runner.run_json(job=job)
        except Exception as exc:
            category = getattr(exc, "discovery_response_failure_category", None)
            prior_content = getattr(exc, "failed_response_content", None)
            if category not in {
                STRICT_JSON_PARSE_FAILURE,
                RAW_TRANSPORT_BUDGET_FAILURE,
            } or not isinstance(prior_content, str):
                raise
            attempts.append(
                _response_attempt_entry(
                    job=job,
                    validation_outcome=category,
                    raw_response_projection_sha256=hashlib.sha256(
                        prior_content.encode("utf-8")
                    ).hexdigest(),
                )
            )
            failure_category = category
        else:
            self._assert_runner_identity(runner)
            raw_wire = _clone(raw)
            try:
                validated = validate_wire(raw_wire)
            except DiscoveryWireSchemaValidationFailure:
                prior_content = canonical_json(raw_wire)
                attempts.append(
                    _response_attempt_entry(
                        job=job,
                        validation_outcome=LOCAL_JSON_SCHEMA_VALIDATION_FAILURE,
                        raw_response_projection_sha256=_sha(raw_wire),
                    )
                )
                failure_category = LOCAL_JSON_SCHEMA_VALIDATION_FAILURE
            else:
                detached = _clone(validated)
                final_wire = raw_wire
                attempts.append(
                    _response_attempt_entry(
                        job=job,
                        validation_outcome=VALIDATED_RESPONSE,
                        raw_response_projection_sha256=_sha(final_wire),
                        normalized_validated_response_sha256=_sha(detached),
                    )
                )

        if attempts[-1]["validation_outcome"] != VALIDATED_RESPONSE:
            repair_job = _build_response_repair_job(
                original_job=job,
                prior_response_content=prior_content,
                failure_category=failure_category,
            )
            if len(repair_job.rendered_messages_bytes) > self.config.max_rendered_prompt_bytes:
                raise ValueError(
                    "cumulative response-repair prompt exceeds the configured "
                    f"{self.config.max_rendered_prompt_bytes}-byte guard"
                )
            self._assert_runner_identity(runner)
            try:
                repaired_raw = runner.run_json(job=repair_job)
            except Exception as exc:
                category = getattr(exc, "discovery_response_failure_category", None)
                if category in {
                    STRICT_JSON_PARSE_FAILURE,
                    RAW_TRANSPORT_BUDGET_FAILURE,
                }:
                    raise DiscoveryResponseRepairExhausted(
                        "hierarchical discovery exhausted its single authenticated "
                        "response repair after transport validation failed"
                    ) from exc
                raise
            self._assert_runner_identity(runner)
            repaired_wire = _clone(repaired_raw)
            try:
                repaired_validated = validate_wire(repaired_wire)
            except DiscoveryWireSchemaValidationFailure as exc:
                raise DiscoveryResponseRepairExhausted(
                    "hierarchical discovery exhausted its single authenticated "
                    "response repair after local JSON-Schema validation failed"
                ) from exc
            detached = _clone(repaired_validated)
            final_wire = repaired_wire
            attempts.append(
                _response_attempt_entry(
                    job=repair_job,
                    validation_outcome=VALIDATED_RESPONSE,
                    raw_response_projection_sha256=_sha(final_wire),
                    normalized_validated_response_sha256=_sha(detached),
                )
            )

        trace = _response_attempt_trace(logical_job=job, attempts=attempts)
        trace = _validated_response_attempt_trace(
            logical_job=job,
            validated_response_sha256=_sha(detached),
            trace=trace,
        )
        self._assert_runner_identity(runner)
        self._assert_implementation_bundle_unchanged(context="during execution")
        result = ValidatedDiscoveryJobResult.create(
            job=job,
            validated_response=detached,
            response_attempt_trace=trace,
        )
        if self.job_cache is not None:
            self.job_cache.store_validated(
                job=job,
                hierarchy_inner_precommit_sha256=self.precommit.precommit_sha256,
                runner_identity=self.runner_identity,
                validator_code_sha256=self.implementation_bundle_sha256,
                wire_response=final_wire,
                validated_response=detached,
                response_attempt_trace=trace,
            )
        return detached, result

    def _execute_coverage_audit(
        self,
        *,
        runner: JsonDiscoveryJobRunner,
        family: str,
        chunk: ArchitectureEvidenceChunk,
        chunk_evidence: Sequence[DiscoveryEvidenceItem],
        interpretation: Mapping[str, Any],
        interpret_job_id: str,
        consolidation_dependency_ids: Sequence[str],
        consolidation: Mapping[str, Any],
        chunk_consolidation: Mapping[str, Any],
        family_catalog_sha256: str,
        chunk_evidence_sha256: str,
        jobs: list[DiscoveryJsonJob],
        results: list[ValidatedDiscoveryJobResult],
    ) -> tuple[list[dict[str, Any]], tuple[str, ...], dict[str, Any]]:
        evidence_tuple = tuple(chunk_evidence)
        evidence_ids = tuple(item.evidence_id for item in evidence_tuple)
        # Coverage must be capable of finding a relation that the current
        # consolidation omitted.  Scheduling only concepts already linked to
        # this chunk/evidence makes that impossible by construction.  Review
        # every chunk evidence atom against the complete family concept set;
        # large sets are losslessly partitioned below.
        canonical_names = tuple(
            str(row["canonical_name"]) for row in consolidation["canonical_concepts"]
        )
        chunk_evidence_id_set = set(evidence_ids)
        # Keep every family concept visible while exposing only the current
        # chunk's support relation.  Other chunks' raw evidence remains in
        # their own authenticated review jobs; the critic needs the complete
        # concept vocabulary here, not unrelated raw identifiers/content.
        coverage_consolidation = _clone(
            {
                "canonical_concepts": [
                    {
                        **row,
                        "supporting_evidence_ids": [
                            evidence_id
                            for evidence_id in row["supporting_evidence_ids"]
                            if evidence_id in chunk_evidence_id_set
                        ],
                    }
                    for row in consolidation["canonical_concepts"]
                ],
                "candidate_dispositions": consolidation["candidate_dispositions"],
            }
        )
        common_bindings = {
            "catalog_sha256": self.catalog.catalog_sha256,
            "family_catalog_sha256": family_catalog_sha256,
            "chunk_id": chunk.chunk_id,
            "chunk_evidence_sha256": chunk_evidence_sha256,
            "interpretation_response_sha256": _sha(interpretation),
            "chunk_consolidation_sha256": _sha(chunk_consolidation),
            "consolidation_response_sha256": _sha(consolidation),
        }
        if len(canonical_names) <= self.config.wire_budget.max_findings_per_atomic_review:
            coverage_job = self._create_job(
                job_kind=COVERAGE_CRITIC_JOB,
                scope=f"{family}.chunk_{chunk.chunk_index:03d}",
                dependencies=(interpret_job_id, *consolidation_dependency_ids),
                settings=self._selector_settings(),
                messages=_render_coverage_messages(
                    family=family,
                    evidence=evidence_tuple,
                    interpretation_responses=(interpretation,),
                    consolidation_response=coverage_consolidation,
                    wire_budget=self.config.wire_budget,
                ),
                input_bindings={
                    **common_bindings,
                    "expected_reviewed_evidence_ids": list(evidence_ids),
                    "complete_family_canonical_names": list(canonical_names),
                    "complete_family_canonical_names_sha256": _sha(list(canonical_names)),
                },
            )
            jobs.append(coverage_job)
            coverage, coverage_result = self._run(
                runner=runner,
                job=coverage_job,
                validator=lambda raw: validate_coverage_critic_response(
                    raw,
                    evidence_ids=evidence_ids,
                    canonical_names=canonical_names,
                ),
            )
            results.append(coverage_result)
            return (
                list(coverage["findings"]),
                (coverage_job.job_id,),
                {
                    "coverage_mode": "bounded_direct_complete_family_v2",
                    "coverage_job_ids": [coverage_job.job_id],
                    "coverage_response_sha256": _sha(coverage),
                    "reviewed_evidence_ids": list(coverage["reviewed_evidence_ids"]),
                },
            )

        findings: list[dict[str, Any]] = []
        atomic_job_ids: list[str] = []
        atomic_audits: list[dict[str, Any]] = []
        for evidence in evidence_tuple:
            name_pages = tuple(
                canonical_names[
                    offset
                    : offset + self.config.wire_budget.max_findings_per_atomic_review
                ]
                for offset in range(
                    0,
                    len(canonical_names),
                    self.config.wire_budget.max_findings_per_atomic_review,
                )
            ) or ((),)
            for page_index, names in enumerate(name_pages):
                name_set = set(names)
                page_consolidation = {
                    "canonical_concepts": [
                        row
                        for row in coverage_consolidation["canonical_concepts"]
                        if row["canonical_name"] in name_set
                    ],
                    "candidate_dispositions": [
                        row
                        for row in coverage_consolidation["candidate_dispositions"]
                        if row["canonical_name"] in name_set
                    ],
                }
                atomic_job = self._create_job(
                    job_kind=COVERAGE_CRITIC_JOB,
                    scope=(
                        f"{family}.chunk_{chunk.chunk_index:03d}."
                        f"{evidence.evidence_id}.page_{page_index:06d}"
                    ),
                    dependencies=(interpret_job_id, *consolidation_dependency_ids),
                    settings=self._selector_settings(),
                    messages=_render_atomic_coverage_messages(
                        family=family,
                        evidence=evidence,
                        interpretation_response=interpretation,
                        consolidation_response=page_consolidation,
                        canonical_names=names,
                        page_index=page_index,
                        wire_budget=self.config.wire_budget,
                    ),
                    input_bindings={
                        **common_bindings,
                        "evidence_id": evidence.evidence_id,
                        "canonical_name_page": list(names),
                        "canonical_name_page_index": page_index,
                        "canonical_name_page_count": len(name_pages),
                        "complete_family_canonical_names_sha256": _sha(list(canonical_names)),
                    },
                )
                jobs.append(atomic_job)
                coverage, coverage_result = self._run(
                    runner=runner,
                    job=atomic_job,
                    validator=lambda raw, evidence_id=evidence.evidence_id, names=names: (
                        _validate_atomic_coverage_response(
                            raw,
                            evidence_id=evidence_id,
                            canonical_names=names,
                        )
                    ),
                )
                results.append(coverage_result)
                findings.extend(coverage["findings"])
                atomic_job_ids.append(atomic_job.job_id)
                atomic_audits.append(
                    {
                        "coverage_job_id": atomic_job.job_id,
                        "evidence_id": evidence.evidence_id,
                        "canonical_names": list(names),
                        "coverage_response_sha256": _sha(coverage),
                    }
                )
        return (
            findings,
            tuple(atomic_job_ids),
            {
                "coverage_mode": "atomic_evidence_complete_family_name_pages_v2",
                "coverage_job_ids": atomic_job_ids,
                "atomic_page_audits": atomic_audits,
                "reviewed_evidence_ids": list(evidence_ids),
                "complete_family_canonical_names": list(canonical_names),
                "every_evidence_name_pair_scheduled_exactly_once": True,
            },
        )

    def _execute_bounded_consolidation(
        self,
        *,
        runner: JsonDiscoveryJobRunner,
        source_family: str,
        candidates: Sequence[DiscoveryCandidate],
        interpretation_job_ids: Sequence[str],
        interpretation_response_sha256: Sequence[str],
        jobs: list[DiscoveryJsonJob],
        results: list[ValidatedDiscoveryJobResult],
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Execute exhaustive bounded pair pages and terminating definition folds."""

        items = tuple(candidates)
        by_id = {item.candidate_id: item for item in items}
        schedule = bounded_candidate_relation_pages(
            items,
            wire_budget=self.config.wire_budget,
        )
        normalized_pages: list[dict[str, Any]] = []
        relation_job_ids: list[str] = []
        for page_index, page in enumerate(schedule):
            anchor_id = str(page["anchor_candidate_id"])
            peer_ids = tuple(str(value) for value in page["peer_candidate_ids"])
            relation_job = self._create_job(
                job_kind=CONSOLIDATE_ARCHITECTURE_JOB,
                scope=f"{source_family}.relation_page_{page_index:06d}",
                dependencies=tuple(interpretation_job_ids),
                settings=self._selector_settings(),
                messages=_render_candidate_relation_page_messages(
                    job="compare_consolidation_candidate_relations",
                    job_kind=CONSOLIDATE_ARCHITECTURE_JOB,
                    source_family=source_family,
                    anchor=by_id[anchor_id],
                    peers=tuple(by_id[peer_id] for peer_id in peer_ids),
                    wire_budget=self.config.wire_budget,
                ),
                input_bindings={
                    "catalog_sha256": self.catalog.catalog_sha256,
                    "source_family": source_family,
                    "relation_page": page,
                    "candidate_projection_sha256": _sha(
                        [
                            _bounded_candidate_projection(by_id[candidate_id])
                            for candidate_id in (anchor_id, *peer_ids)
                        ]
                    ),
                    "interpretation_response_sha256": list(interpretation_response_sha256),
                },
            )
            jobs.append(relation_job)
            normalized, result = self._run(
                runner=runner,
                job=relation_job,
                validator=lambda raw, anchor_id=anchor_id, peer_ids=peer_ids: (
                    validate_candidate_relation_page_response(
                        raw,
                        anchor_candidate_id=anchor_id,
                        peer_candidate_ids=peer_ids,
                        wire_budget=self.config.wire_budget,
                    )
                ),
            )
            results.append(result)
            normalized_pages.append(normalized)
            relation_job_ids.append(relation_job.job_id)

        grouped = compile_complete_link_candidate_groups(
            candidate_ids=tuple(item.candidate_id for item in items),
            relation_pages=normalized_pages,
        )
        grouped_sha256 = _sha(grouped)
        definitions: dict[str, dict[str, Any]] = {}
        terminal_job_ids: list[str] = []
        for raw_group in grouped["groups"]:
            group_id = str(raw_group["group_id"])
            member_ids = tuple(str(value) for value in raw_group["member_candidate_ids"])
            if len(member_ids) == 1:
                singleton = by_id[member_ids[0]]
                definitions[group_id] = {
                    "canonical_name": singleton.feature_name,
                    "description": singleton.description,
                    "unresolved_ambiguity": singleton.unresolved_ambiguity,
                    "reason": "singleton definition preserved byte-exactly",
                }
                continue
            prior: dict[str, Any] | None = None
            prior_job_id: str | None = None
            for fold in candidate_definition_fold_batches(
                group_id=group_id,
                member_candidate_ids=member_ids,
                wire_budget=self.config.wire_budget,
            ):
                fresh_ids = tuple(str(value) for value in fold["member_candidate_ids"])
                dependencies = (
                    (prior_job_id,) if prior_job_id is not None else tuple(relation_job_ids)
                )
                fold_index = int(fold["fold_index"])
                fold_job = self._create_job(
                    job_kind=CONSOLIDATE_ARCHITECTURE_JOB,
                    scope=(f"{source_family}.{group_id}.definition_fold_{fold_index:06d}"),
                    dependencies=dependencies,
                    settings=self._selector_settings(),
                    messages=_render_candidate_definition_fold_messages(
                        job="fold_consolidation_group_definition",
                        job_kind=CONSOLIDATE_ARCHITECTURE_JOB,
                        group_id=group_id,
                        fold_index=fold_index,
                        candidates=tuple(by_id[value] for value in fresh_ids),
                        prior_accumulator=prior,
                        wire_budget=self.config.wire_budget,
                    ),
                    input_bindings={
                        "catalog_sha256": self.catalog.catalog_sha256,
                        "source_family": source_family,
                        "group_compiler_sha256": grouped_sha256,
                        "group_id": group_id,
                        "group_member_candidate_ids_sha256": _sha(list(member_ids)),
                        "fold": fold,
                        "prior_accumulator_sha256": (_sha(prior) if prior is not None else None),
                    },
                )
                jobs.append(fold_job)
                prior, fold_result = self._run(
                    runner=runner,
                    job=fold_job,
                    validator=_validate_candidate_definition_fold_response,
                )
                results.append(fold_result)
                prior_job_id = fold_job.job_id
            if prior is None or prior_job_id is None:
                raise AssertionError("multi-member group did not execute a definition fold")
            definitions[group_id] = prior
            terminal_job_ids.append(prior_job_id)

        consolidation = _compile_bounded_consolidation(
            source_family=source_family,
            candidates=items,
            grouped=grouped,
            definitions_by_group_id=definitions,
            wire_budget=self.config.wire_budget,
        )
        if not terminal_job_ids:
            terminal_job_ids.extend(relation_job_ids or interpretation_job_ids)
        return consolidation, tuple(terminal_job_ids)

    def _execute_bounded_cross_architecture(
        self,
        *,
        runner: JsonDiscoveryJobRunner,
        dossiers: Sequence[ArchitectureDossier],
        coverage_job_ids: Sequence[str],
        jobs: list[DiscoveryJsonJob],
        results: list[ValidatedDiscoveryJobResult],
    ) -> tuple[
        dict[str, Any],
        tuple[str, ...],
        dict[str, Any],
        tuple[IntegratedCanonicalFeature, ...],
        str,
    ]:
        """Plan and integrate arbitrary candidate counts with finite response pages."""

        dossier_tuple = tuple(dossiers)
        candidates = tuple(
            candidate for dossier in dossier_tuple for candidate in dossier.architecture_candidates
        )
        if not candidates:
            if not coverage_job_ids:
                raise ValueError("empty cross-architecture hierarchy lacks coverage dependencies")
            planner = {
                "provisional_groups": [],
                "raw_evidence_requests": [],
                "wire_normalization_audit": {
                    "audit_version": "lossless_complete_link_planner_compiler_v2",
                    "groups": [],
                    "raw_support_sampling": False,
                    "every_group_support_item_is_page_scheduled": True,
                    "empty_candidate_ledger": True,
                },
            }
            integration = {
                "canonical_features": [],
                "candidate_dispositions": [],
                "wire_normalization_audit": {
                    "audit_version": "lossless_paged_per_group_integration_compiler_v2",
                    "group_integration_wires": {},
                    "group_support_review_audits": {},
                    "all_provisional_groups_integrated_exactly_once": True,
                    "global_integrated_feature_truncation": False,
                    "raw_support_sampling": False,
                    "empty_candidate_ledger": True,
                },
            }
            return planner, (), integration, (), tuple(coverage_job_ids)[-1]
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        dossier_sha256 = [_sha(dossier.as_authenticated_dict()) for dossier in dossier_tuple]
        normalized_pages: list[dict[str, Any]] = []
        relation_job_ids: list[str] = []
        for page_index, page in enumerate(
            bounded_candidate_relation_pages(
                candidates,
                wire_budget=self.config.wire_budget,
            )
        ):
            anchor_id = str(page["anchor_candidate_id"])
            peer_ids = tuple(str(value) for value in page["peer_candidate_ids"])
            relation_job = self._create_job(
                job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
                scope=f"all_architectures.relation_page_{page_index:06d}",
                dependencies=tuple(coverage_job_ids),
                settings=self._selector_settings(),
                messages=_render_candidate_relation_page_messages(
                    job="compare_cross_architecture_candidate_relations",
                    job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
                    source_family=None,
                    anchor=by_id[anchor_id],
                    peers=tuple(by_id[value] for value in peer_ids),
                    wire_budget=self.config.wire_budget,
                ),
                input_bindings={
                    "dossier_sha256": dossier_sha256,
                    "relation_page": page,
                    "candidate_projection_sha256": _sha(
                        [
                            _bounded_candidate_projection(by_id[candidate_id])
                            for candidate_id in (anchor_id, *peer_ids)
                        ]
                    ),
                },
            )
            jobs.append(relation_job)
            normalized, result = self._run(
                runner=runner,
                job=relation_job,
                validator=lambda raw, anchor_id=anchor_id, peer_ids=peer_ids: (
                    validate_candidate_relation_page_response(
                        raw,
                        anchor_candidate_id=anchor_id,
                        peer_candidate_ids=peer_ids,
                        wire_budget=self.config.wire_budget,
                    )
                ),
            )
            results.append(result)
            normalized_pages.append(normalized)
            relation_job_ids.append(relation_job.job_id)

        grouped = compile_complete_link_candidate_groups(
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            relation_pages=normalized_pages,
        )
        grouped_sha256 = _sha(grouped)
        definitions: dict[str, dict[str, Any]] = {}
        terminal_by_group: dict[str, str] = {}
        for raw_group in grouped["groups"]:
            group_id = str(raw_group["group_id"])
            member_ids = tuple(str(value) for value in raw_group["member_candidate_ids"])
            if len(member_ids) == 1:
                singleton = by_id[member_ids[0]]
                definitions[group_id] = {
                    "canonical_name": singleton.feature_name,
                    "description": singleton.description,
                    "unresolved_ambiguity": singleton.unresolved_ambiguity,
                    "reason": "singleton definition preserved byte-exactly",
                }
                terminal_by_group[group_id] = (
                    relation_job_ids[-1] if relation_job_ids else tuple(coverage_job_ids)[-1]
                )
                continue
            prior: dict[str, Any] | None = None
            prior_job_id: str | None = None
            for fold in candidate_definition_fold_batches(
                group_id=group_id,
                member_candidate_ids=member_ids,
                wire_budget=self.config.wire_budget,
            ):
                fold_index = int(fold["fold_index"])
                fresh_ids = tuple(str(value) for value in fold["member_candidate_ids"])
                fold_job = self._create_job(
                    job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
                    scope=f"{group_id}.planner_definition_fold_{fold_index:06d}",
                    dependencies=(
                        (prior_job_id,) if prior_job_id is not None else tuple(relation_job_ids)
                    ),
                    settings=self._selector_settings(),
                    messages=_render_candidate_definition_fold_messages(
                        job="fold_cross_architecture_group_definition",
                        job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
                        group_id=group_id,
                        fold_index=fold_index,
                        candidates=tuple(by_id[value] for value in fresh_ids),
                        prior_accumulator=prior,
                        wire_budget=self.config.wire_budget,
                    ),
                    input_bindings={
                        "dossier_sha256": dossier_sha256,
                        "group_compiler_sha256": grouped_sha256,
                        "group_id": group_id,
                        "group_member_candidate_ids_sha256": _sha(list(member_ids)),
                        "fold": fold,
                        "prior_accumulator_sha256": (_sha(prior) if prior is not None else None),
                    },
                )
                jobs.append(fold_job)
                prior, fold_result = self._run(
                    runner=runner,
                    job=fold_job,
                    validator=_validate_candidate_definition_fold_response,
                )
                results.append(fold_result)
                prior_job_id = fold_job.job_id
            if prior is None or prior_job_id is None:
                raise AssertionError("planner group did not complete its definition folds")
            definitions[group_id] = prior
            terminal_by_group[group_id] = prior_job_id

        planner, lookback_ids_by_group = _compile_bounded_cross_architecture_plan(
            candidates=candidates,
            grouped=grouped,
            definitions_by_group_id=definitions,
            evidence_by_id=self._evidence_by_id,
            wire_budget=self.config.wire_budget,
        )
        requested_ids = tuple(
            dict.fromkeys(
                evidence_id
                for request in planner["raw_evidence_requests"]
                for evidence_id in request["evidence_ids"]
            )
        )
        integration_wires: dict[str, dict[str, Any]] = {}
        integration_review_audits: dict[str, dict[str, Any]] = {}
        prior_integration_job_id: str | None = None
        for raw_group in grouped["groups"]:
            group_id = str(raw_group["group_id"])
            member_ids = tuple(str(value) for value in raw_group["member_candidate_ids"])
            members = tuple(by_id[value] for value in member_ids)
            complete_support_ids = lookback_ids_by_group[group_id]
            if not complete_support_ids:
                raise ValueError("provisional integration group has no raw evidence support")
            support_sha256 = _sha(list(complete_support_ids))
            page_rows: list[tuple[str, dict[str, Any], str, str]] = []
            for evidence_index, evidence_id in enumerate(complete_support_ids):
                evidence = self._evidence_by_id[evidence_id]
                review_id = (
                    "integration_evidence_review_"
                    f"{_sha({'group_id': group_id, 'evidence_id': evidence_id})}"
                )
                dependencies = [terminal_by_group[group_id]]
                if prior_integration_job_id is not None:
                    dependencies.append(prior_integration_job_id)
                page_job = self._create_job(
                    job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
                    scope=f"{group_id}.evidence_page_{evidence_index:06d}",
                    dependencies=tuple(dict.fromkeys(dependencies)),
                    settings=self._selector_settings(),
                    messages=_render_integration_evidence_page_messages(
                        group_id=group_id,
                        definition=definitions[group_id],
                        members=members,
                        evidence=evidence,
                        review_id=review_id,
                        wire_budget=self.config.wire_budget,
                    ),
                    input_bindings={
                        "dossier_sha256": dossier_sha256,
                        "planner_response_sha256": _sha(planner),
                        "group_compiler_sha256": grouped_sha256,
                        "group_id": group_id,
                        "member_candidate_ids_sha256": _sha(list(member_ids)),
                        "complete_supporting_evidence_ids": list(complete_support_ids),
                        "complete_supporting_evidence_ids_sha256": support_sha256,
                        "evidence_review_id": review_id,
                        "evidence_id": evidence_id,
                        "evidence_index": evidence_index,
                        "evidence_count": len(complete_support_ids),
                        "raw_evidence_sha256": _sha(evidence.as_prompt_item()),
                    },
                )
                jobs.append(page_job)
                page_response, page_result = self._run(
                    runner=runner,
                    job=page_job,
                    validator=_validate_integration_evidence_page_response,
                )
                results.append(page_result)
                page_rows.append((review_id, page_response, page_job.job_id, evidence_id))

            consumed = 0
            fold_index = 0
            accumulator_id: str | None = None
            accumulator_response: dict[str, Any] | None = None
            accumulator_job_id: str | None = None
            fold_job_ids: list[str] = []
            fold_audits: list[dict[str, Any]] = []
            while consumed < len(page_rows):
                fresh_capacity = (
                    self.config.wire_budget.max_definition_fold_inputs
                    if accumulator_response is None
                    else self.config.wire_budget.max_definition_fold_inputs - 1
                )
                fresh = page_rows[consumed : consumed + fresh_capacity]
                review_inputs: list[tuple[str, Mapping[str, Any]]] = []
                dependencies: list[str] = []
                if accumulator_response is not None:
                    assert accumulator_id is not None and accumulator_job_id is not None
                    review_inputs.append((accumulator_id, accumulator_response))
                    dependencies.append(accumulator_job_id)
                review_inputs.extend((review_id, response) for review_id, response, _, _ in fresh)
                dependencies.extend(job_id for _, _, job_id, _ in fresh)
                review_input_ids = tuple(review_id for review_id, _ in review_inputs)
                fold_job = self._create_job(
                    job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
                    scope=f"{group_id}.evidence_fold_{fold_index:06d}",
                    dependencies=tuple(dict.fromkeys(dependencies)),
                    settings=self._selector_settings(),
                    messages=_render_integration_evidence_fold_messages(
                        group_id=group_id,
                        fold_index=fold_index,
                        definition=definitions[group_id],
                        review_inputs=review_inputs,
                        wire_budget=self.config.wire_budget,
                    ),
                    input_bindings={
                        "dossier_sha256": dossier_sha256,
                        "planner_response_sha256": _sha(planner),
                        "group_compiler_sha256": grouped_sha256,
                        "group_id": group_id,
                        "fold_index": fold_index,
                        "review_input_ids": list(review_input_ids),
                        "fresh_review_ids": [row[0] for row in fresh],
                        "fresh_evidence_ids": [row[3] for row in fresh],
                        "prior_accumulator_id": accumulator_id,
                        "prior_accumulator_response_sha256": (
                            _sha(accumulator_response) if accumulator_response is not None else None
                        ),
                        "complete_supporting_evidence_ids": list(complete_support_ids),
                        "complete_supporting_evidence_ids_sha256": support_sha256,
                    },
                )
                jobs.append(fold_job)
                fold_response, fold_result = self._run(
                    runner=runner,
                    job=fold_job,
                    validator=lambda raw, review_input_ids=review_input_ids: (
                        _validate_integration_evidence_fold_response(
                            raw,
                            review_input_ids=review_input_ids,
                        )
                    ),
                )
                results.append(fold_result)
                accumulator_response = fold_response
                accumulator_job_id = fold_job.job_id
                accumulator_id = _review_accumulator_id(
                    kind="integration_review",
                    scope=group_id,
                    fold_index=fold_index,
                    response=fold_response,
                )
                fold_job_ids.append(fold_job.job_id)
                fold_audits.append(
                    {
                        "fold_job_id": fold_job.job_id,
                        "fold_index": fold_index,
                        "review_input_ids": list(review_input_ids),
                        "fresh_review_ids": [row[0] for row in fresh],
                        "accumulator_id": accumulator_id,
                        "fold_response_sha256": _sha(fold_response),
                    }
                )
                consumed += len(fresh)
                fold_index += 1
            if accumulator_response is None or accumulator_job_id is None:
                raise AssertionError("integration support schedule produced no terminal fold")
            integration_wires[group_id] = accumulator_response
            integration_review_audits[group_id] = {
                "complete_supporting_evidence_ids": list(complete_support_ids),
                "complete_supporting_evidence_ids_sha256": support_sha256,
                "evidence_review_ids": [row[0] for row in page_rows],
                "evidence_page_job_ids": [row[2] for row in page_rows],
                "fold_job_ids": fold_job_ids,
                "fold_audits": fold_audits,
                "terminal_fold_job_id": accumulator_job_id,
                "every_support_item_reviewed_exactly_once": True,
                "all_page_reviews_transitively_folded": True,
                "raw_support_sampling": False,
            }
            prior_integration_job_id = accumulator_job_id

        if prior_integration_job_id is None:
            raise ValueError("bounded cross-architecture integration produced no groups")
        used_names: set[str] = set()
        features: list[IntegratedCanonicalFeature] = []
        normalized_features: list[dict[str, Any]] = []
        dispositions_by_candidate: dict[str, dict[str, str]] = {}
        disambiguations: list[dict[str, str]] = []
        for raw_group in grouped["groups"]:
            group_id = str(raw_group["group_id"])
            member_ids = tuple(str(value) for value in raw_group["member_candidate_ids"])
            members = tuple(by_id[value] for value in member_ids)
            wire = integration_wires[group_id]
            if wire["decision"] == "reject":
                dispositions_by_candidate.update(
                    {
                        candidate_id: {
                            "candidate_id": candidate_id,
                            "decision": "reject",
                            "canonical_name": "",
                            "reason": wire["reason"],
                        }
                        for candidate_id in member_ids
                    }
                )
                continue
            name, event = _derive_unique_integration_name(
                proposed=wire["canonical_name"],
                slot=f"group_{_sha(group_id)[:8]}",
                used=used_names,
                wire_budget=self.config.wire_budget,
            )
            used_names.add(name)
            if event is not None:
                disambiguations.append({"group_id": group_id, **event})
            support = tuple(
                dict.fromkeys(
                    evidence_id
                    for candidate in members
                    for evidence_id in candidate.supporting_evidence_ids
                )
            )
            families = tuple(
                dict.fromkeys(
                    family for candidate in members for family in candidate.source_families
                )
            )
            shapes = {candidate.value_shape_hypothesis for candidate in members}
            shape = next(iter(shapes)) if len(shapes) == 1 else "ambiguous"
            feature = IntegratedCanonicalFeature(
                canonical_name=name,
                description=wire["description"],
                member_candidate_ids=member_ids,
                supporting_evidence_ids=support,
                source_families=families,
                value_shape_hypothesis=shape,
                unresolved_ambiguity=wire["unresolved_ambiguity"],
            )
            features.append(feature)
            normalized_features.append(
                {
                    "canonical_name": name,
                    "description": wire["description"],
                    "member_candidate_ids": list(member_ids),
                    "supporting_evidence_ids": list(support),
                    "source_families": list(families),
                    "value_shape_hypothesis": shape,
                    "unresolved_ambiguity": wire["unresolved_ambiguity"],
                    "extraction_constraints": {
                        "aliases": [],
                        "units": [],
                        "categories": [],
                        "distinguish_from": [],
                    },
                }
            )
            dispositions_by_candidate.update(
                {
                    candidate_id: {
                        "candidate_id": candidate_id,
                        "decision": "accept",
                        "canonical_name": name,
                        "reason": wire["reason"],
                    }
                    for candidate_id in member_ids
                }
            )
        candidate_order = tuple(candidate.candidate_id for candidate in candidates)
        if set(dispositions_by_candidate) != set(candidate_order):
            raise ValueError("bounded integration lost a candidate disposition")
        integration = _clone(
            {
                "canonical_features": normalized_features,
                "candidate_dispositions": [
                    dispositions_by_candidate[candidate_id] for candidate_id in candidate_order
                ],
                "wire_normalization_audit": {
                    "audit_version": "lossless_paged_per_group_integration_compiler_v2",
                    "group_compiler_sha256": grouped_sha256,
                    "group_integration_wires": integration_wires,
                    "group_support_review_audits": integration_review_audits,
                    "canonical_name_disambiguations": disambiguations,
                    "all_provisional_groups_integrated_exactly_once": True,
                    "global_integrated_feature_truncation": False,
                    "raw_support_sampling": False,
                },
            }
        )
        return (
            planner,
            requested_ids,
            integration,
            tuple(features),
            prior_integration_job_id,
        )

    def execute(
        self,
        *,
        runner: JsonDiscoveryJobRunner,
        approved_precommit_sha256: str,
    ) -> CompletedHierarchicalDiscovery:
        """Execute only after exact offline-packet approval; no transport is built in."""

        if approved_precommit_sha256 != self.precommit.precommit_sha256:
            raise ValueError("approved precommit SHA-256 does not match the offline packet")
        self._assert_runner_identity(runner)
        self._assert_implementation_bundle_unchanged(
            context="before authenticated execution",
            refresh_local_validator=True,
        )
        if self.job_cache is not None:
            self.job_cache.begin_execution(
                hierarchy_inner_precommit_sha256=self.precommit.precommit_sha256,
                runner_identity=self.runner_identity,
            )
        jobs: list[DiscoveryJsonJob] = []
        results: list[ValidatedDiscoveryJobResult] = []
        dossiers: list[ArchitectureDossier] = []
        coverage_job_ids: list[str] = []
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
            family_chunks = tuple(
                chunk for chunk in self.chunk_plan.chunks if chunk.source_family == family
            )
            interpretations: list[dict[str, Any]] = []
            interpret_jobs: list[DiscoveryJsonJob] = []
            chunk_candidates_by_id: dict[str, tuple[DiscoveryCandidate, ...]] = {}
            chunk_candidates: list[DiscoveryCandidate] = []
            for chunk in family_chunks:
                job = self._initial_job_by_chunk_id[chunk.chunk_id]
                jobs.append(job)
                evidence = tuple(
                    self._evidence_by_id[str(row["evidence_id"])] for row in chunk.evidence
                )
                validated, result = self._run(
                    runner=runner,
                    job=job,
                    validator=lambda raw, evidence=evidence: (
                        validate_interpret_evidence_chunk_response(
                            raw,
                            evidence=evidence,
                            wire_budget=self.config.wire_budget,
                        )
                    ),
                )
                results.append(result)
                interpretations.append(validated)
                interpret_jobs.append(job)
                candidates = tuple(
                    _candidate_from_interpretation(
                        job=job,
                        family=family,
                        concept=concept,
                    )
                    for concept in validated["concepts"]
                )
                chunk_candidates_by_id[chunk.chunk_id] = candidates
                chunk_candidates.extend(candidates)

            chunk_candidate_tuple = tuple(chunk_candidates)
            interpretation_job_ids = tuple(job.job_id for job in interpret_jobs)
            interpretation_response_sha256 = tuple(_sha(row) for row in interpretations)
            if len(chunk_candidate_tuple) <= MAX_DIRECT_CONSOLIDATION_CANDIDATES:
                consolidation_job = self._create_job(
                    job_kind=CONSOLIDATE_ARCHITECTURE_JOB,
                    scope=family,
                    dependencies=interpretation_job_ids,
                    settings=self._selector_settings(),
                    messages=_render_consolidation_messages(
                        source_family=family,
                        candidates=chunk_candidate_tuple,
                        wire_budget=self.config.wire_budget,
                    ),
                    input_bindings={
                        "catalog_sha256": self.catalog.catalog_sha256,
                        "source_family": family,
                        "interpretation_response_sha256": list(interpretation_response_sha256),
                    },
                )
                jobs.append(consolidation_job)
                consolidation, consolidation_result = self._run(
                    runner=runner,
                    job=consolidation_job,
                    validator=lambda raw, family=family, candidates=(
                        chunk_candidate_tuple
                    ): _validate_consolidation_allowing_empty(
                        raw,
                        source_family=family,
                        candidates=candidates,
                        wire_budget=self.config.wire_budget,
                    ),
                )
                results.append(consolidation_result)
                consolidation_dependency_ids = (consolidation_job.job_id,)
            else:
                consolidation, consolidation_dependency_ids = self._execute_bounded_consolidation(
                    runner=runner,
                    source_family=family,
                    candidates=chunk_candidate_tuple,
                    interpretation_job_ids=interpretation_job_ids,
                    interpretation_response_sha256=(interpretation_response_sha256),
                    jobs=jobs,
                    results=results,
                )
            family_evidence = self._family_evidence[family]
            evidence_ids = tuple(item.evidence_id for item in family_evidence)
            family_catalog_sha256 = _family_catalog_sha256(self.catalog, family, family_evidence)
            coverage_disposition_ids: list[str] = []
            chunk_coverage_audits: list[dict[str, Any]] = []
            for chunk, interpretation, interpret_job in zip(
                family_chunks, interpretations, interpret_jobs
            ):
                chunk_evidence = tuple(
                    self._evidence_by_id[str(row["evidence_id"])] for row in chunk.evidence
                )
                chunk_evidence_ids = tuple(item.evidence_id for item in chunk_evidence)
                chunk_candidates_for_audit = chunk_candidates_by_id[chunk.chunk_id]
                chunk_consolidation = _chunk_scoped_consolidation_view(
                    consolidation_response=consolidation,
                    chunk_candidates=chunk_candidates_for_audit,
                    chunk_evidence_ids=chunk_evidence_ids,
                )
                chunk_evidence_sha256 = _sha(
                    {
                        "catalog_sha256": self.catalog.catalog_sha256,
                        "family_catalog_sha256": family_catalog_sha256,
                        "chunk_id": chunk.chunk_id,
                        "evidence": [item.as_prompt_item() for item in chunk_evidence],
                    }
                )
                coverage_findings, chunk_coverage_job_ids, coverage_audit = (
                    self._execute_coverage_audit(
                        runner=runner,
                        family=family,
                        chunk=chunk,
                        chunk_evidence=chunk_evidence,
                        interpretation=interpretation,
                        interpret_job_id=interpret_job.job_id,
                        consolidation_dependency_ids=consolidation_dependency_ids,
                        consolidation=consolidation,
                        chunk_consolidation=chunk_consolidation,
                        family_catalog_sha256=family_catalog_sha256,
                        chunk_evidence_sha256=chunk_evidence_sha256,
                        jobs=jobs,
                        results=results,
                    )
                )
                unresolved = [
                    finding for finding in coverage_findings if finding["action"] != "no_change"
                ]
                if unresolved:
                    consolidation, revision_audit = _compile_coverage_revision_findings(
                        consolidation_response=consolidation,
                        findings=unresolved,
                        source_family=family,
                        candidates=chunk_candidate_tuple,
                        wire_budget=self.config.wire_budget,
                    )
                    corrected_chunk_consolidation = _chunk_scoped_consolidation_view(
                        consolidation_response=consolidation,
                        chunk_candidates=chunk_candidates_for_audit,
                        chunk_evidence_ids=chunk_evidence_ids,
                    )
                    corrected_names = {
                        row["canonical_name"]
                        for row in corrected_chunk_consolidation["canonical_concepts"]
                    }
                    revised_names = {
                        event["compiled_canonical_name"]
                        for event in revision_audit["events"]
                        if event["compiled_canonical_name"]
                    }
                    for concept in consolidation["canonical_concepts"]:
                        if concept["canonical_name"] not in revised_names - corrected_names:
                            continue
                        support = [
                            evidence_id
                            for evidence_id in concept["supporting_evidence_ids"]
                            if evidence_id in set(chunk_evidence_ids)
                        ]
                        if not support:
                            continue
                        corrected_chunk_consolidation["canonical_concepts"].append(
                            {
                                **_clone(concept),
                                "member_candidate_ids": [
                                    candidate_id
                                    for candidate_id in concept["member_candidate_ids"]
                                    if candidate_id
                                    in {
                                        candidate.candidate_id
                                        for candidate in chunk_candidates_for_audit
                                    }
                                ],
                                "supporting_evidence_ids": support,
                            }
                        )
                    (
                        followup_findings,
                        followup_job_ids,
                        followup_audit,
                    ) = self._execute_coverage_audit(
                        runner=runner,
                        family=family,
                        chunk=chunk,
                        chunk_evidence=chunk_evidence,
                        interpretation=interpretation,
                        interpret_job_id=interpret_job.job_id,
                        consolidation_dependency_ids=chunk_coverage_job_ids,
                        consolidation=consolidation,
                        chunk_consolidation=corrected_chunk_consolidation,
                        family_catalog_sha256=family_catalog_sha256,
                        chunk_evidence_sha256=chunk_evidence_sha256,
                        jobs=jobs,
                        results=results,
                    )
                    still_unresolved = [
                        finding for finding in followup_findings if finding["action"] != "no_change"
                    ]
                    if still_unresolved:
                        raise CoverageCriticRequiresRevision(
                            f"architecture {family!r} chunk {chunk.chunk_index} has unresolved "
                            "coverage findings after one deterministic continuation"
                        )
                    chunk_coverage_job_ids = (
                        *chunk_coverage_job_ids,
                        *followup_job_ids,
                    )
                    coverage_audit = {
                        "coverage_mode": "bounded_revision_then_followup_v1",
                        "initial_audit": coverage_audit,
                        "revision_compiler_audit": revision_audit,
                        "followup_audit": followup_audit,
                        "coverage_job_ids": list(chunk_coverage_job_ids),
                        "reviewed_evidence_ids": list(chunk_evidence_ids),
                    }
                coverage_job_ids.extend(chunk_coverage_job_ids)
                coverage_disposition_ids.extend(chunk_evidence_ids)
                chunk_coverage_audits.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_evidence_sha256": chunk_evidence_sha256,
                        **coverage_audit,
                    }
                )
            architecture_candidates = tuple(
                _candidate_from_consolidation(family=family, concept=concept)
                for concept in consolidation["canonical_concepts"]
            )
            interpreted_disposition_ids = tuple(
                row["evidence_id"]
                for response in interpretations
                for row in response["evidence_dispositions"]
            )
            if set(interpreted_disposition_ids) != set(evidence_ids):
                raise RuntimeError("architecture interpretations lost catalog evidence")
            if set(coverage_disposition_ids) != set(evidence_ids):
                raise RuntimeError("chunk coverage audits lost catalog evidence")
            binding = self._binding_by_family[family]
            dossiers.append(
                ArchitectureDossier(
                    source_family=family,
                    catalog_sha256=self.catalog.catalog_sha256,
                    catalog_evidence_ids=evidence_ids,
                    coverage_disposition_ids=tuple(coverage_disposition_ids),
                    coverage_audit_sha256=_sha(
                        {
                            "catalog_sha256": self.catalog.catalog_sha256,
                            "family_catalog_sha256": family_catalog_sha256,
                            "chunk_coverage_audits": chunk_coverage_audits,
                        }
                    ),
                    architecture_candidates=architecture_candidates,
                    direct_numerical_signal_count=binding.signal_count,
                    direct_numerical_zero_reason=binding.zero_reason,
                    direct_numerical_contract_kind=(binding.direct_numerical_contract_kind),
                    direct_numerical_contract_sha256=(binding.direct_numerical_contract_sha256),
                    direct_numerical_manifest_sha256=binding.manifest_sha256,
                )
            )

        dossier_tuple = tuple(dossiers)
        # This call is also the exact ten-dossier and global candidate-ID guard.
        cross_architecture_planner_context(dossier_tuple)
        cross_executor = self._execute_bounded_cross_architecture
        (
            planner,
            requested_ids,
            integration,
            integrated_features,
            last_selector_job_id,
        ) = cross_executor(
            runner=runner,
            dossiers=dossier_tuple,
            coverage_job_ids=coverage_job_ids,
            jobs=jobs,
            results=results,
        )
        candidate_by_id = {
            candidate.candidate_id: candidate
            for dossier in dossier_tuple
            for candidate in dossier.architecture_candidates
        }
        rejected_candidate_ids = tuple(
            row["candidate_id"]
            for row in integration["candidate_dispositions"]
            if row["decision"] == "reject"
        )
        rejected = {
            candidate_id: candidate_by_id[candidate_id] for candidate_id in rejected_candidate_ids
        }

        rejection: dict[str, Any] = {
            "reconsiderations": [],
            "lossless_review_audits": [],
        }
        disposition_by_candidate_id = {
            str(row["candidate_id"]): row for row in integration["candidate_dispositions"]
        }
        for candidate_id, candidate in rejected.items():
            complete_support_ids = tuple(candidate.supporting_evidence_ids)
            support_sha256 = _sha(list(complete_support_ids))
            page_rows: list[tuple[str, dict[str, Any], str, str]] = []
            for evidence_index, evidence_id in enumerate(complete_support_ids):
                evidence = self._evidence_by_id[evidence_id]
                review_id = (
                    "rejection_evidence_review_"
                    f"{_sha({'candidate_id': candidate_id, 'evidence_id': evidence_id})}"
                )
                page_job = self._create_job(
                    job_kind=REJECTION_CRITIC_JOB,
                    scope=f"{candidate_id}.evidence_page_{evidence_index:06d}",
                    dependencies=(last_selector_job_id,),
                    settings=self._selector_settings(),
                    messages=_render_rejection_evidence_page_messages(
                        candidate=candidate,
                        integration_disposition=disposition_by_candidate_id[candidate_id],
                        evidence=evidence,
                        review_id=review_id,
                        wire_budget=self.config.wire_budget,
                    ),
                    input_bindings={
                        "integration_response_sha256": _sha(integration),
                        "rejected_candidate_id": candidate_id,
                        "complete_rejected_candidate_supporting_evidence_ids": list(
                            complete_support_ids
                        ),
                        "complete_rejected_candidate_support_sha256": support_sha256,
                        "evidence_review_id": review_id,
                        "evidence_id": evidence_id,
                        "evidence_index": evidence_index,
                        "evidence_count": len(complete_support_ids),
                        "raw_evidence_sha256": _sha(evidence.as_prompt_item()),
                    },
                )
                jobs.append(page_job)
                page_response, page_result = self._run(
                    runner=runner,
                    job=page_job,
                    validator=_validate_rejection_evidence_page_response,
                )
                results.append(page_result)
                page_rows.append((review_id, page_response, page_job.job_id, evidence_id))

            consumed = 0
            fold_index = 0
            accumulator_id: str | None = None
            accumulator_response: dict[str, Any] | None = None
            accumulator_job_id: str | None = None
            fold_job_ids: list[str] = []
            fold_audits: list[dict[str, Any]] = []
            while consumed < len(page_rows):
                fresh_capacity = (
                    self.config.wire_budget.max_definition_fold_inputs
                    if accumulator_response is None
                    else self.config.wire_budget.max_definition_fold_inputs - 1
                )
                fresh = page_rows[consumed : consumed + fresh_capacity]
                review_inputs: list[tuple[str, Mapping[str, Any]]] = []
                dependencies: list[str] = []
                if accumulator_response is not None:
                    assert accumulator_id is not None and accumulator_job_id is not None
                    review_inputs.append((accumulator_id, accumulator_response))
                    dependencies.append(accumulator_job_id)
                review_inputs.extend((review_id, response) for review_id, response, _, _ in fresh)
                dependencies.extend(job_id for _, _, job_id, _ in fresh)
                review_input_ids = tuple(review_id for review_id, _ in review_inputs)
                fold_job = self._create_job(
                    job_kind=REJECTION_CRITIC_JOB,
                    scope=f"{candidate_id}.evidence_fold_{fold_index:06d}",
                    dependencies=tuple(dict.fromkeys(dependencies)),
                    settings=self._selector_settings(),
                    messages=_render_rejection_evidence_fold_messages(
                        candidate=candidate,
                        fold_index=fold_index,
                        review_inputs=review_inputs,
                        wire_budget=self.config.wire_budget,
                    ),
                    input_bindings={
                        "integration_response_sha256": _sha(integration),
                        "rejected_candidate_id": candidate_id,
                        "fold_index": fold_index,
                        "review_input_ids": list(review_input_ids),
                        "fresh_review_ids": [row[0] for row in fresh],
                        "fresh_evidence_ids": [row[3] for row in fresh],
                        "prior_accumulator_id": accumulator_id,
                        "prior_accumulator_response_sha256": (
                            _sha(accumulator_response) if accumulator_response is not None else None
                        ),
                        "complete_rejected_candidate_supporting_evidence_ids": list(
                            complete_support_ids
                        ),
                        "complete_rejected_candidate_support_sha256": support_sha256,
                    },
                )
                jobs.append(fold_job)
                fold_response, fold_result = self._run(
                    runner=runner,
                    job=fold_job,
                    validator=lambda raw, review_input_ids=review_input_ids: (
                        _validate_rejection_evidence_fold_response(
                            raw,
                            review_input_ids=review_input_ids,
                        )
                    ),
                )
                results.append(fold_result)
                accumulator_response = fold_response
                accumulator_job_id = fold_job.job_id
                accumulator_id = _review_accumulator_id(
                    kind="rejection_review",
                    scope=candidate_id,
                    fold_index=fold_index,
                    response=fold_response,
                )
                fold_job_ids.append(fold_job.job_id)
                fold_audits.append(
                    {
                        "fold_job_id": fold_job.job_id,
                        "fold_index": fold_index,
                        "review_input_ids": list(review_input_ids),
                        "fresh_review_ids": [row[0] for row in fresh],
                        "accumulator_id": accumulator_id,
                        "fold_response_sha256": _sha(fold_response),
                    }
                )
                consumed += len(fresh)
                fold_index += 1
            if accumulator_response is None or accumulator_job_id is None:
                raise AssertionError("rejection support schedule produced no terminal fold")
            reconsideration = {
                "candidate_id": candidate_id,
                "decision": accumulator_response["decision"],
                "proposed_name": accumulator_response["proposed_name"],
                "supporting_evidence_ids": (
                    []
                    if accumulator_response["decision"] == "uphold"
                    else list(complete_support_ids)
                ),
                "reason": accumulator_response["reason"],
            }
            if reconsideration["decision"] != "uphold":
                integration, integrated_features, _ = _compile_rejection_reconsideration(
                    integration_response=integration,
                    integrated_features=integrated_features,
                    candidate=candidate,
                    reconsideration=reconsideration,
                    evidence_by_id=self._evidence_by_id,
                    maximum_integrated_features=self.config.max_integrated_features,
                    wire_budget=self.config.wire_budget,
                )
            rejection["reconsiderations"].append(reconsideration)
            rejection["lossless_review_audits"].append(
                {
                    "candidate_id": candidate_id,
                    "complete_supporting_evidence_ids": list(complete_support_ids),
                    "complete_supporting_evidence_ids_sha256": support_sha256,
                    "evidence_review_ids": [row[0] for row in page_rows],
                    "evidence_page_job_ids": [row[2] for row in page_rows],
                    "fold_job_ids": fold_job_ids,
                    "fold_audits": fold_audits,
                    "terminal_fold_job_id": accumulator_job_id,
                    "every_support_item_reviewed_exactly_once": True,
                    "all_page_reviews_transitively_folded": True,
                    "raw_support_sampling": False,
                }
            )
            last_selector_job_id = accumulator_job_id

        rejected_candidate_ids = tuple(
            row["candidate_id"]
            for row in integration["candidate_dispositions"]
            if row["decision"] == "reject"
        )

        routed: list[RoutedIntegratedFeature] = []
        extraction_job_ids: list[str] = []
        extraction_definitions: dict[str, Any] = {}
        all_evidence = tuple(self._evidence_by_id.values())
        for feature in integrated_features:
            routing = route_concept_roles(
                evidence=all_evidence,
                supporting_evidence_ids=feature.supporting_evidence_ids,
            )
            routed_feature = RoutedIntegratedFeature(feature=feature, role_routing=routing)
            routed.append(routed_feature)
            evidence = tuple(
                self._evidence_by_id[evidence_id] for evidence_id in feature.supporting_evidence_ids
            )
            request = ExtractionDefinitionRequest(
                canonical_name=feature.canonical_name,
                evidence=evidence,
                supporting_evidence_ids=feature.supporting_evidence_ids,
                value_shape_hypothesis=feature.value_shape_hypothesis,
                allowed_aliases=feature.allowed_aliases,
                allowed_units=feature.allowed_units,
                allowed_categories=feature.allowed_categories,
                allowed_distinguish_from=feature.allowed_distinguish_from,
            )
            complete_support_ids = tuple(feature.supporting_evidence_ids)
            support_sha256 = _sha(list(complete_support_ids))
            page_rows: list[tuple[str, dict[str, Any], str, str]] = []
            for evidence_index, evidence_item in enumerate(evidence):
                review_id = (
                    "extraction_evidence_review_"
                    f"{_sha({'canonical_name': feature.canonical_name, 'evidence_id': evidence_item.evidence_id})}"
                )
                page_job = self._create_job(
                    job_kind=EXTRACTION_DEFINITION_JOB,
                    scope=(f"{feature.canonical_name}.evidence_page_{evidence_index:06d}"),
                    dependencies=(last_selector_job_id,),
                    settings=DiscoveryJobSettings.extraction(),
                    messages=_render_extraction_evidence_page_messages(
                        request=request,
                        evidence=evidence_item,
                        review_id=review_id,
                        wire_budget=self.config.wire_budget,
                    ),
                    input_bindings={
                        "integration_response_sha256": _sha(integration),
                        "canonical_name": feature.canonical_name,
                        "complete_supporting_evidence_ids": list(complete_support_ids),
                        "complete_supporting_evidence_ids_sha256": support_sha256,
                        "evidence_review_id": review_id,
                        "evidence_id": evidence_item.evidence_id,
                        "evidence_index": evidence_index,
                        "evidence_count": len(complete_support_ids),
                        "raw_evidence_sha256": _sha(evidence_item.as_prompt_item()),
                        "value_shape_hypothesis": feature.value_shape_hypothesis,
                        "vocabulary_grounding_policy_sha256": _sha(
                            extraction_vocabulary_grounding_policy()
                        ),
                        "deterministic_role_routing": routing.audit(),
                        "role_routing_sha256": _sha(routing.audit()),
                    },
                )
                jobs.append(page_job)
                page_response, page_result = self._run(
                    runner=runner,
                    job=page_job,
                    validator=lambda raw, request=request, evidence_item=evidence_item: (
                        _validate_extraction_evidence_page_response(
                            raw,
                            request=request,
                            evidence=evidence_item,
                        )
                    ),
                )
                results.append(page_result)
                extraction_job_ids.append(page_job.job_id)
                page_rows.append(
                    (
                        review_id,
                        page_response,
                        page_job.job_id,
                        evidence_item.evidence_id,
                    )
                )

            consumed = 0
            fold_index = 0
            accumulator_id: str | None = None
            accumulator_wire: dict[str, Any] | None = None
            accumulator_job_id: str | None = None
            terminal_definition: dict[str, Any] | None = None
            while consumed < len(page_rows):
                fresh_capacity = (
                    self.config.wire_budget.max_definition_fold_inputs
                    if accumulator_wire is None
                    else self.config.wire_budget.max_definition_fold_inputs - 1
                )
                fresh = page_rows[consumed : consumed + fresh_capacity]
                review_inputs: list[tuple[str, Mapping[str, Any]]] = []
                dependencies: list[str] = []
                if accumulator_wire is not None:
                    assert accumulator_id is not None and accumulator_job_id is not None
                    review_inputs.append((accumulator_id, accumulator_wire))
                    dependencies.append(accumulator_job_id)
                review_inputs.extend((review_id, response) for review_id, response, _, _ in fresh)
                dependencies.extend(job_id for _, _, job_id, _ in fresh)
                review_input_ids = tuple(review_id for review_id, _ in review_inputs)
                fold_job = self._create_job(
                    job_kind=EXTRACTION_DEFINITION_JOB,
                    scope=f"{feature.canonical_name}.evidence_fold_{fold_index:06d}",
                    dependencies=tuple(dict.fromkeys(dependencies)),
                    settings=DiscoveryJobSettings.extraction(),
                    messages=_render_extraction_evidence_fold_messages(
                        request=request,
                        fold_index=fold_index,
                        review_inputs=review_inputs,
                        wire_budget=self.config.wire_budget,
                    ),
                    input_bindings={
                        "integration_response_sha256": _sha(integration),
                        "canonical_name": feature.canonical_name,
                        "fold_index": fold_index,
                        "review_input_ids": list(review_input_ids),
                        "fresh_review_ids": [row[0] for row in fresh],
                        "fresh_evidence_ids": [row[3] for row in fresh],
                        "prior_accumulator_id": accumulator_id,
                        "prior_accumulator_response_sha256": (
                            _sha(accumulator_wire) if accumulator_wire is not None else None
                        ),
                        "complete_supporting_evidence_ids": list(complete_support_ids),
                        "complete_supporting_evidence_ids_sha256": support_sha256,
                        "supporting_evidence_sha256": _sha(
                            [item.as_prompt_item() for item in evidence]
                        ),
                        "value_shape_hypothesis": feature.value_shape_hypothesis,
                        "vocabulary_grounding_policy_sha256": _sha(
                            extraction_vocabulary_grounding_policy()
                        ),
                        "vocabulary_grounding_policy": (extraction_vocabulary_grounding_policy()),
                        "deterministic_role_routing": routing.audit(),
                        "role_routing_sha256": _sha(routing.audit()),
                        "raw_support_sampling": False,
                    },
                )
                jobs.append(fold_job)
                fold_response, fold_result = self._run(
                    runner=runner,
                    job=fold_job,
                    validator=lambda raw, request=request, review_input_ids=review_input_ids: (
                        _validate_extraction_evidence_fold_response(
                            raw,
                            request=request,
                            review_input_ids=review_input_ids,
                        )
                    ),
                )
                results.append(fold_result)
                extraction_job_ids.append(fold_job.job_id)
                accumulator_wire = fold_response["fold_wire"]
                terminal_definition = fold_response["definition"]
                accumulator_job_id = fold_job.job_id
                accumulator_id = _review_accumulator_id(
                    kind="extraction_review",
                    scope=feature.canonical_name,
                    fold_index=fold_index,
                    response=accumulator_wire,
                )
                consumed += len(fresh)
                fold_index += 1
            if terminal_definition is None or accumulator_job_id is None:
                raise AssertionError("extraction support schedule produced no terminal fold")
            extraction_definitions[feature.canonical_name] = terminal_definition
            last_selector_job_id = accumulator_job_id

        self._assert_implementation_bundle_unchanged(
            context="at execution completion",
            refresh_local_validator=True,
        )
        execution = DiscoveryExecutionLedger.build(jobs=jobs, results=results)
        completion_identity = {
            "schema_version": COMPLETED_HIERARCHICAL_DISCOVERY_VERSION,
            "precommit_sha256": self.precommit.precommit_sha256,
            "dossiers": [row.as_authenticated_dict() for row in dossier_tuple],
            "planner_response": planner,
            "requested_lookback_evidence_ids": list(requested_ids),
            "integration_response": integration,
            "rejected_candidate_ids": list(rejected_candidate_ids),
            "rejection_critic_response": rejection,
            "routed_features": [row.as_dict() for row in routed],
            "extraction_definitions": extraction_definitions,
            "execution_sha256": execution.execution_sha256,
        }
        return CompletedHierarchicalDiscovery(
            precommit_sha256=self.precommit.precommit_sha256,
            dossiers=dossier_tuple,
            routed_features=tuple(routed),
            rejected_candidate_ids=rejected_candidate_ids,
            requested_lookback_evidence_ids=requested_ids,
            extraction_job_ids=tuple(extraction_job_ids),
            execution_ledger=execution,
            completion_sha256=_sha(completion_identity),
            _planner_response_json=canonical_json(planner),
            _integration_response_json=canonical_json(integration),
            _rejection_critic_response_json=canonical_json(rejection),
            _extraction_definitions_json=canonical_json(extraction_definitions),
        )


__all__ = [
    "AUTHENTICATED_MESSAGE_ENVELOPE_BINDING",
    "AUTHENTICATED_RESPONSE_CONTRACT_BINDING",
    "AUTHENTICATED_RESPONSE_REPAIR_BINDING",
    "COMPLETED_HIERARCHICAL_DISCOVERY_VERSION",
    "CONSOLIDATE_ARCHITECTURE_JOB",
    "COVERAGE_CRITIC_JOB",
    "CROSS_ARCHITECTURE_INTEGRATION_JOB",
    "CROSS_ARCHITECTURE_PLANNER_JOB",
    "CompletedHierarchicalDiscovery",
    "CoverageCriticRequiresRevision",
    "DirectNumericalDossierBinding",
    "DiscoveryExecutionLedger",
    "DiscoveryJobLedger",
    "DiscoveryJobSettings",
    "DiscoveryJsonJob",
    "DiscoveryResponseRepairExhausted",
    "DiscoverySemanticNormalizationFailure",
    "DiscoveryWireSchemaValidationFailure",
    "DISCOVERY_RESPONSE_ATTEMPT_TRACE_VERSION",
    "DISCOVERY_RESPONSE_REPAIR_POLICY_VERSION",
    "EXTRACTION_DEFINITION_JOB",
    "EXTRACTION_DEFINITION_SYSTEM_PROMPT",
    "HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING",
    "HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_VERSION",
    "HIERARCHICAL_DISCOVERY_ORCHESTRATOR_VERSION",
    "HierarchicalAllArchitectureDiscoveryOrchestrator",
    "HierarchicalDiscoveryConfig",
    "HierarchicalDiscoveryPrecommit",
    "INTERPRET_CHUNK_JOB",
    "IntegratedCanonicalFeature",
    "JsonDiscoveryJobRunner",
    "MAX_RENDERED_DISCOVERY_PROMPT_BYTES",
    "MAX_DISCOVERY_RESPONSE_REPAIR_ATTEMPTS",
    "LOCAL_JSON_SCHEMA_VALIDATION_FAILURE",
    "LOCAL_JSON_SCHEMA_VALIDATOR_VERSION",
    "REJECTION_CRITIC_JOB",
    "RAW_TRANSPORT_BUDGET_FAILURE",
    "RejectionCriticRequiresRevision",
    "RoutedIntegratedFeature",
    "SELECTOR_THINKING_TOKEN_BUDGET",
    "SEMANTIC_VALIDATION_FAILURE",
    "STRICT_JSON_PARSE_FAILURE",
    "VALIDATED_RESPONSE",
    "ValidatedDiscoveryJobResult",
    "discovery_response_repair_policy_identity",
    "hierarchical_discovery_implementation_bundle",
    "local_json_schema_validator_identity",
    "validate_cross_architecture_integration_response",
]
