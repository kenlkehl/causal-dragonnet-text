"""Architecture-at-a-time adaptive Stage-1 reconsideration.

This module is deliberately independent of the existing post-extraction review
runner.  It compiles the model jobs needed after a validation gate has been
consumed and an exact accumulated-spent Stage-1 catalog has been authenticated:

1. interpret complete raw evidence chunks from one architecture at a time;
2. consolidate candidates within each architecture;
3. audit complete within-architecture coverage;
4. show a cross-architecture planner only ten compact dossiers, the current
   feature registry, and sanitized aggregate diagnostics;
5. resolve every planner-requested evidence ID through bounded lossless pages
   and recursive folds; and
6. compile and freeze a bounded registry-revision proposal before another gate
   can be consumed.

Direct upstream numerical channels, non-grounding numerical summaries, row
data, notes, oracle values, and temporal-policy context have no model-facing
path through this interface.  The role-neutral catalog may contain authenticated
non-grounding summaries, but architecture chunking omits them and this module
never serializes them into a prompt or dossier.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .all_evidence_fusion import (
    CandidateContract,
    ground_evidence_to_extraction_contract,
)

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
    AS_DOCUMENTED_UNIT,
    CONSOLIDATE_JOB_VERSION,
    COVERAGE_CRITIC_JOB_VERSION,
    DiscoveryCandidate,
    DiscoveryEvidenceItem,
    EXTRACTION_DEFINITION_JOB_VERSION,
    ExtractionDefinitionRequest,
    INTERPRET_JOB_VERSION,
    INTERPRET_SYSTEM_PROMPT,
    MECHANICAL_MENTION_CATEGORIES,
    bounded_candidate_relation_pages,
    candidate_definition_fold_batches,
    canonical_json,
    compile_complete_link_candidate_groups,
    content_sha256,
    extraction_vocabulary_grounding_policy,
    interpretation_model_view,
    render_interpret_evidence_chunk_messages,
    revalidate_normalized_consolidation_response,
    revalidate_normalized_coverage_critic_response,
    revalidate_normalized_extraction_definition_response,
    revalidate_normalized_interpret_evidence_chunk_response,
    route_concept_roles,
    validate_candidate_relation_page_response,
    validate_consolidation_response,
    validate_coverage_critic_response,
    validate_extraction_definition_response,
    validate_interpret_evidence_chunk_response,
)
from .hierarchical_all_architecture_discovery import (
    CONSOLIDATE_ARCHITECTURE_JOB,
    COVERAGE_CRITIC_JOB,
    CROSS_ARCHITECTURE_INTEGRATION_JOB,
    CROSS_ARCHITECTURE_PLANNER_JOB,
    EXTRACTION_DEFINITION_JOB,
    EXTRACTION_DEFINITION_SYSTEM_PROMPT,
    HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING,
    INTERPRET_CHUNK_JOB,
    LOCAL_JSON_SCHEMA_VALIDATION_FAILURE,
    MAX_RENDERED_DISCOVERY_PROMPT_BYTES,
    RAW_TRANSPORT_BUDGET_FAILURE,
    STRICT_JSON_PARSE_FAILURE,
    VALIDATED_RESPONSE,
    DiscoveryResponseRepairExhausted,
    DiscoverySemanticNormalizationFailure,
    DiscoveryWireSchemaValidationFailure,
    DiscoveryJobSettings,
    DiscoveryJsonJob,
    JsonDiscoveryJobRunner,
    _bounded_candidate_projection,
    _compile_bounded_consolidation,
    _build_response_repair_job,
    _render_candidate_definition_fold_messages,
    _render_candidate_relation_page_messages,
    _render_extraction_evidence_fold_messages,
    _render_extraction_evidence_page_messages,
    _response_attempt_entry,
    _response_attempt_trace,
    _validate_atomic_coverage_response,
    _validate_candidate_definition_fold_response,
    _validate_extraction_evidence_fold_response,
    _validate_extraction_evidence_page_response,
    _validated_response_attempt_trace,
    hierarchical_discovery_implementation_bundle,
    _render_extraction_messages,
    _validate_local_discovery_wire_schema,
    local_json_schema_validator_identity,
)
from .hierarchical_discovery_job_cache import (
    AuthenticatedHierarchicalDiscoveryJobCache,
)
from .hierarchical_discovery_response_contract import (
    HIERARCHICAL_DISCOVERY_MAX_ADAPTIVE_REVIEW_TARGETS,
    HIERARCHICAL_DISCOVERY_MAX_ATOMS_PER_INTERPRET_JOB,
    HIERARCHICAL_DISCOVERY_MAX_FINDINGS_PER_ATOMIC_REVIEW,
    HIERARCHICAL_DISCOVERY_MAX_DEFINITION_FOLD_MEMBERS,
    HIERARCHICAL_DISCOVERY_MAX_GENERATED_NAME_LENGTH,
    HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB,
    attach_hierarchical_discovery_response_contract,
    build_hierarchical_discovery_response_contract,
)
from .all_evidence_post_extraction_review import (
    AppliedReviewOperations,
    extraction_semantics_sha256,
)
from .lossless_stage1_evidence_catalog import (
    DEFAULT_MAX_ATOMS_PER_ARCHITECTURE_CHUNK,
    DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK,
    DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK,
    ArchitectureEvidenceChunk,
    RoleNeutralEvidenceCatalog,
    audit_complete_architecture_delivery,
    build_complete_architecture_chunks,
    validate_role_neutral_catalog,
)

ADAPTIVE_HIERARCHY_VERSION = "adaptive_hierarchical_stage1_reconsideration_v7"
EXACT_SPENT_CATALOG_AUTHENTICATION_VERSION = "exact_spent_catalog_authentication_v1"
ADAPTIVE_DOSSIER_VERSION = "adaptive_architecture_dossier_v2"
ADAPTIVE_FAMILY_CONSOLIDATION_VERSION = "adaptive_family_consolidation_v1"
ADAPTIVE_CHUNK_COVERAGE_VERSION = "adaptive_chunk_coverage_v1"
ADAPTIVE_PLANNER_INTERFACE_VERSION = "adaptive_stage1_lookback_planner_v4"
ADAPTIVE_PROPOSER_INTERFACE_VERSION = "adaptive_registry_revision_proposer_v4"
ADAPTIVE_ROUND_FREEZE_VERSION = "frozen_adaptive_stage1_reconsideration_round_v3"
ADAPTIVE_AUTHENTICATED_EXECUTION_VERSION = "authenticated_adaptive_hierarchical_stage1_execution_v7"
ADAPTIVE_EXECUTABLE_BRIDGE_VERSION = "adaptive_executable_extraction_contract_bridge_v3"
ADAPTIVE_PROMPT_CONTRACT_VERSION = "adaptive_hierarchical_stage1_prompt_contract_v7"
ADAPTIVE_IMPLEMENTATION_BUNDLE_VERSION = "adaptive_hierarchical_implementation_bundle_v8"

_ADAPTIVE_PLANNER_NORMALIZATION_AUDIT_VERSION = "adaptive_planner_wire_normalization_audit_v1"
_ADAPTIVE_PROPOSER_NORMALIZATION_AUDIT_VERSION = "adaptive_proposer_wire_normalization_audit_v1"
_ADAPTIVE_PHASED_PLANNER_COMPILER_VERSION = "adaptive_phased_planner_compiler_v1"
_ADAPTIVE_PHASED_PROPOSER_COMPILER_VERSION = "adaptive_phased_proposer_compiler_v1"
_ADAPTIVE_PHASED_EXTRACTION_COMPILER_VERSION = "adaptive_phased_extraction_compiler_v1"

_ADAPTIVE_IMPLEMENTATION_BUNDLE_FILES = (
    "adaptive_hierarchical_stage1_reconsideration.py",
    "all_evidence_discovery_interfaces.py",
    "hierarchical_all_architecture_discovery.py",
    "hierarchical_discovery_response_contract.py",
    "all_evidence_fusion.py",
    "all_evidence_post_extraction_review.py",
    "lossless_stage1_evidence_catalog.py",
    "stage1_architecture_explanations.py",
)

NEW_MISSING_CONSTRUCT = "new_missing_construct"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.:-]*\Z")
_FEATURE_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
_FORBIDDEN_MODEL_KEY = re.compile(
    r"(?:^|_)(?:"
    r"oracle|ground_truth|true_ite|true_cate|true_effect|counterfactual_truth|"
    r"row_id|row_ids|raw_rows|row_values|row_records|"
    r"patient_id|patient_ids|mrn|record_id|record_ids|"
    r"raw_note|raw_notes|full_note|full_notes|note_text|notes|"
    r"raw_vector|raw_vectors|embedding_vector|embedding_vectors|activations|"
    r"backend_path|artifact_path|cache_path|"
    r"direct_upstream_numerical|direct_numerical|non_grounding_numerical|"
    r"temporal_policy|current_date|validation_labels|gate_labels"
    r")(?:_|$)",
    flags=re.IGNORECASE,
)
_FORBIDDEN_MODEL_TEXT = (
    "direct_upstream_numerical",
    "direct upstream numerical",
    "non_grounding_numerical",
    "non-grounding numerical",
    "temporal_policy",
    "temporal policy",
    "current_date",
    "current date",
    "ground_truth",
    "ground truth ite",
    "true effect",
    "true_ite",
    "true cate",
    "oracle",
    "oracle outcome",
    "raw note",
    "raw notes",
    "full note",
    "full notes",
    "file://",
    "http://",
    "https://",
)
_DIAGNOSTIC_KINDS = frozenset(
    {
        "extraction_missingness",
        "extraction_validity",
        "nuisance_residual",
        "heterogeneity",
        "redundancy",
        "source_preservation",
    }
)
_VALUE_SHAPES = frozenset({"continuous", "categorical", "ambiguous", "unresolved"})
_OPERATIONS = frozenset({"add", "drop", "merge", "split", "rename", "revise_definition"})

_ADAPTIVE_INTERPRET_SYSTEM = INTERPRET_SYSTEM_PROMPT
_ADAPTIVE_CONSOLIDATE_SYSTEM = (
    "Consolidate candidates from exactly one Stage 1 architecture. Merge only clear aliases and "
    "preserve every candidate, evidence reference, ambiguity, and measurement distinction. Do "
    "not compare architectures, assign roles, estimate effects, or reject a supported "
    "characteristic. candidate_assignments is keyed by every exact candidate_id; assign each "
    "candidate to one fixed cluster_slot and define every fixed slot. Return JSON only."
)
_ADAPTIVE_COVERAGE_SYSTEM = (
    "Audit one evidence chunk from one Stage 1 architecture for semantic loss. Review every "
    "supplied atom and member disposition against that family's consolidation. Report omitted "
    "characteristics, improper merges, or lost support. Do not compare architectures, assign "
    "roles, or estimate effects. reviewed_evidence_ids is keyed by every exact evidence_id with "
    "value true. Return JSON only."
)
_ADAPTIVE_RELATION_SYSTEM = (
    "Judge every exact anchor-to-peer candidate pair independently. Use relation "
    "same_construct only for the same patient-level construct; use distinct for "
    "different measurements and uncertain when the compact evidence cannot decide. "
    "Do not infer transitive merges, repeat support IDs, assign causal roles, define "
    "extraction, or estimate effects. comparisons is keyed by every exact later peer "
    "candidate ID. Return JSON only."
)
_ADAPTIVE_DEFINITION_FOLD_SYSTEM = (
    "Define one canonical patient-level construct for this compiler-proven "
    "complete-link group. Fold the prior accumulator, when supplied, together with "
    "every fresh member without dropping measurement distinctions or unresolved "
    "ambiguity. Membership, provenance, and evidence support are compiler-owned; do "
    "not repeat them, assign causal roles, define extraction, or estimate effects. "
    "Return JSON only."
)
_ADAPTIVE_ATOMIC_COVERAGE_SYSTEM = (
    "Audit this exact original evidence atom and its member dispositions against one "
    "bounded page of consolidated concepts. Report omitted concepts, improper merges, "
    "or lost support. Evidence support is fixed and compiler-derived; do not repeat "
    "support IDs. affected_canonical_names may use only the exact page names. Set "
    "reviewed_atomic_review true. Do not compare other evidence, assign causal roles, "
    "define extraction, or estimate effects. Return JSON only."
)
_ADAPTIVE_PROPOSAL_JUDGE_SYSTEM = (
    "Judge one exact adaptive registry-revision proposal after its bounded evidence page has "
    "been reviewed. Accept only a concrete, evidence-grounded patient-feature revision; reject "
    "unsupported, redundant, or measurement-collapsing revisions. The proposal identity, complete "
    "support restoration, cross-page grouping, conflicts, and final round capacity are compiler-owned. "
    "Do not assign causal roles, define extraction, or estimate effects. Return JSON only."
)
_ADAPTIVE_EXTRACTION_PAGE_SYSTEM = (
    "Review exactly one authenticated support item for extraction of the named patient feature. "
    "Record the measurement and shape observation plus only aliases, units, categories, and "
    "distinctions that occur literally in this raw evidence item. This is one page of an exhaustive "
    "compiler-owned support schedule; do not infer that unseen support is absent or finalize the "
    "extraction alone. Set reviewed_evidence true. Do not add causal claims or a second feature. "
    "Return JSON only."
)
_ADAPTIVE_PLANNER_SYSTEM = (
    "Plan a diagnostic-driven reconsideration using ten independently completed Stage 1 "
    "architecture dossiers. Use the current registry and observable aggregate diagnostics to "
    "identify existing features needing attention or a missing patient characteristic. Request "
    "the dossier-listed evidence IDs needed to resolve each ambiguity on this exact bounded page. "
    "Every target, evidence incidence, and diagnostic page is scheduled and folded by the compiler. Do not define "
    "operations, assign roles, estimate effects, or invent evidence. Return JSON only."
)
_ADAPTIVE_PROPOSER_SYSTEM = (
    "Propose at most one revision for this exact bounded page of the current patient-feature "
    "registry using the diagnostic plan, ten compact architecture dossiers, and only the exact "
    "requested evidence supplied here. You may add an evidence-grounded missing characteristic or target an existing feature "
    "for drop, merge, split, rename, or definition revision. Preserve distinct measurements. "
    "Cite supplied diagnostics and evidence. Every emitted proposal is separately judged, grouped, "
    "and explicitly dispositioned before the compiler applies conflict or round capacity; do not assign causal roles, estimate effects, or "
    "invent evidence. Return JSON only."
)
_ADAPTIVE_DEFINITION_SYSTEM = EXTRACTION_DEFINITION_SYSTEM_PROMPT


def _dynamic_output_schema_template(*response_keys: str) -> dict[str, Any]:
    """Static review template; runtime enums/consts come from designated fields."""

    return {
        "contract_kind": "strict_json_schema",
        "response_object_keys": list(response_keys),
        "additional_properties_allowed": False,
        "identifier_values": "dynamic_enums_and_consts_from_designated_request_fields",
        "ownership_relations": "dynamic_exact_request_scoped_contract",
        "example_identifier_values_present": False,
    }


def _interpret_output_schema() -> dict[str, Any]:
    return _dynamic_output_schema_template("concepts", "evidence_dispositions")


def _consolidation_output_schema() -> dict[str, Any]:
    return _dynamic_output_schema_template("candidate_assignments", "slot_definitions")


def _relation_output_schema() -> dict[str, Any]:
    return _dynamic_output_schema_template("comparisons")


def _definition_fold_output_schema() -> dict[str, Any]:
    return _dynamic_output_schema_template(
        "canonical_name",
        "description",
        "unresolved_ambiguity",
        "reason",
    )


def _coverage_output_schema() -> dict[str, Any]:
    return _dynamic_output_schema_template("findings", "reviewed_evidence_ids")


def _atomic_coverage_output_schema() -> dict[str, Any]:
    return _dynamic_output_schema_template("findings", "reviewed_atomic_review")


def _planner_output_schema() -> dict[str, Any]:
    return _dynamic_output_schema_template("review_targets", "no_lookback_needed")


def _proposer_output_schema() -> dict[str, Any]:
    return _dynamic_output_schema_template("operations", "converged")


def _definition_output_schema() -> dict[str, Any]:
    return _dynamic_output_schema_template(
        "feature_name",
        "measurement",
        "representation",
        "aliases",
        "distinguish_from",
        "missing_or_ambiguous",
        "supporting_evidence_ids",
    )


_USER_PAYLOAD_TOP_LEVEL_KEYS = {
    INTERPRET_CHUNK_JOB: (
        "job",
        "family_explanation",
        "evidence",
        "identifier_ownership",
        "output_schema",
    ),
    CONSOLIDATE_ARCHITECTURE_JOB: (
        "job",
        "source_family",
        "candidates",
        "identifier_ownership",
        "output_schema",
    ),
    COVERAGE_CRITIC_JOB: (
        "job",
        "source_family",
        "evidence",
        "chunk_interpretation",
        "family_consolidation",
        "identifier_ownership",
        "output_schema",
    ),
    CROSS_ARCHITECTURE_PLANNER_JOB: (
        "job",
        "architecture_dossiers",
        "current_registry",
        "diagnostics",
        "lookback_bounds",
        "identifier_ownership",
        "output_schema",
    ),
    CROSS_ARCHITECTURE_INTEGRATION_JOB: (
        "job",
        "architecture_dossiers",
        "current_registry",
        "diagnostics",
        "review_plan",
        "requested_evidence",
        "maximum_operations",
        "identifier_ownership",
        "output_schema",
    ),
    EXTRACTION_DEFINITION_JOB: (
        "job",
        "canonical_name",
        "value_shape_hypothesis",
        "supporting_evidence_ids",
        "evidence",
        "planner_lookback_constraints",
        "vocabulary_grounding_policy",
        "identifier_ownership",
        "output_schema",
    ),
}

_STATIC_JOB_LITERAL = {
    INTERPRET_CHUNK_JOB: "interpret_evidence_chunk",
    CONSOLIDATE_ARCHITECTURE_JOB: "consolidate_adaptive_architecture_candidates",
    COVERAGE_CRITIC_JOB: "audit_adaptive_architecture_coverage",
    CROSS_ARCHITECTURE_PLANNER_JOB: "plan_adaptive_stage1_reconsideration",
    CROSS_ARCHITECTURE_INTEGRATION_JOB: "propose_adaptive_registry_revision",
    EXTRACTION_DEFINITION_JOB: "define_one_extraction_feature",
}

_DYNAMIC_USER_PAYLOAD_PATHS = {
    INTERPRET_CHUNK_JOB: ("family_explanation", "evidence"),
    CONSOLIDATE_ARCHITECTURE_JOB: (
        "source_family",
        "candidates",
    ),
    COVERAGE_CRITIC_JOB: (
        "source_family",
        "evidence",
        "chunk_interpretation",
        "family_consolidation",
    ),
    CROSS_ARCHITECTURE_PLANNER_JOB: (
        "architecture_dossiers",
        "current_registry",
        "diagnostics",
        "lookback_bounds",
    ),
    CROSS_ARCHITECTURE_INTEGRATION_JOB: (
        "architecture_dossiers",
        "current_registry",
        "diagnostics",
        "review_plan",
        "requested_evidence",
        "maximum_operations",
    ),
    EXTRACTION_DEFINITION_JOB: (
        "canonical_name",
        "value_shape_hypothesis",
        "supporting_evidence_ids",
        "evidence",
        "planner_lookback_constraints",
    ),
}

_CROSS_ARCHITECTURE_DYNAMIC_SHAPES = {
    "architecture_dossier_keys": [
        "source_family",
        "coverage",
        "architecture_candidates",
    ],
    "dossier_coverage_keys": [
        "catalog_evidence_count",
        "coverage_disposition_count",
        "complete",
        "lookback_evidence_ids",
    ],
    "dossier_candidate_keys": [
        "candidate_id",
        "feature_name",
        "description",
        "supporting_evidence_ids",
        "source_families",
        "value_shape_hypothesis",
        "unresolved_ambiguity",
    ],
    "current_registry_item_keys": [
        "feature_name",
        "description",
        "value_shape_hypothesis",
        "source_families",
        "definition_summary",
        "support_provenance",
    ],
    "diagnostic_item_keys": [
        "diagnostic_id",
        "diagnostic_kind",
        "affected_features",
        "summary",
        "aggregate_metrics",
    ],
}


def _clone(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _sha(value: Any) -> str:
    return content_sha256(value)


def adaptive_hierarchical_implementation_bundle(
    *, refresh_local_validator: bool = False
) -> dict[str, Any]:
    """Authenticate every local renderer, validator, and compiler dependency.

    The composite SHA is also the immutable-cache validator identity.  This
    prevents a cache hit compiled under one prompt/validation implementation
    from being replayed after any dependency changes while the adaptive module
    itself remains byte-identical.
    """

    base = Path(__file__).resolve().parent
    files: dict[str, str] = {}
    for filename in _ADAPTIVE_IMPLEMENTATION_BUNDLE_FILES:
        path = base / filename
        if not path.is_file():
            raise ValueError(f"adaptive implementation dependency is missing: {filename}")
        files[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    body = {
        "schema_version": ADAPTIVE_IMPLEMENTATION_BUNDLE_VERSION,
        "files": files,
        "local_json_schema_validator": local_json_schema_validator_identity(
            refresh=refresh_local_validator
        ),
    }
    return {**body, "implementation_bundle_sha256": _sha(body)}


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_string(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{label} cannot be empty")
    return value


def _require_identifier(value: Any, *, label: str) -> str:
    result = _require_string(value, label=label)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ValueError(f"{label} must be an opaque lowercase identifier")
    return result


def _require_feature_name(value: Any, *, label: str) -> str:
    result = _require_string(value, label=label)
    if _FEATURE_NAME.fullmatch(result) is None:
        raise ValueError(f"{label} must be lower snake_case")
    return result


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


def _string_list(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
    feature_names: bool = False,
    identifiers: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    if not allow_empty and not value:
        raise ValueError(f"{label} cannot be empty")
    parsed: list[str] = []
    for index, item in enumerate(value):
        if feature_names:
            parsed.append(_require_feature_name(item, label=f"{label}[{index}]"))
        elif identifiers:
            parsed.append(_require_identifier(item, label=f"{label}[{index}]"))
        else:
            parsed.append(_require_string(item, label=f"{label}[{index}]"))
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{label} cannot contain duplicates")
    return tuple(parsed)


def _deduplicated_string_list(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
    feature_names: bool = False,
    identifiers: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate one wire list and retain the first occurrence deterministically."""

    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    if not allow_empty and not value:
        raise ValueError(f"{label} cannot be empty")
    retained: list[str] = []
    duplicates: list[str] = []
    for index, item in enumerate(value):
        if feature_names:
            parsed = _require_feature_name(item, label=f"{label}[{index}]")
        elif identifiers:
            parsed = _require_identifier(item, label=f"{label}[{index}]")
        else:
            parsed = _require_string(item, label=f"{label}[{index}]")
        if parsed in retained:
            duplicates.append(parsed)
        else:
            retained.append(parsed)
    return tuple(retained), tuple(duplicates)


def _compiler_unique_feature_name(
    proposed: str,
    *,
    unavailable: set[str],
    suffix: str,
) -> str:
    """Return a bounded lower-snake-case name outside ``unavailable``."""

    _require_feature_name(proposed, label="proposed compiler feature name")
    _require_feature_name(suffix, label="compiler feature-name suffix")
    if proposed not in unavailable:
        return proposed
    ordinal = 1
    while True:
        ending = f"_{suffix}" if ordinal == 1 else f"_{suffix}_{ordinal:03d}"
        prefix_limit = HIERARCHICAL_DISCOVERY_MAX_GENERATED_NAME_LENGTH - len(ending)
        prefix = proposed[:prefix_limit].rstrip("_")
        if not prefix:
            prefix = "feature"
        candidate = f"{prefix}{ending}"
        _require_feature_name(candidate, label="compiler-derived feature name")
        if candidate not in unavailable:
            return candidate
        ordinal += 1


def _scan_model_safe(value: Any, *, path: str) -> None:
    """Reject forbidden model-facing fields and policy text recursively."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            if key.casefold() in {"row", "rows", "patient_records"} or (
                _FORBIDDEN_MODEL_KEY.search(key)
            ):
                raise ValueError(f"{path}.{key} is forbidden in adaptive model context")
            _scan_model_safe(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_model_safe(child, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        lowered = value.casefold()
        matched = [term for term in _FORBIDDEN_MODEL_TEXT if term in lowered]
        if matched:
            raise ValueError(f"{path} contains forbidden adaptive model text: {matched}")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if value is not None and not isinstance(value, (bool, int, float)):
        raise TypeError(f"{path} is not JSON-safe")


def _sanitize_model_authored_text(value: str) -> tuple[str, tuple[str, ...]]:
    """Mask policy-channel terms before model-authored text enters another prompt."""

    result = value
    matched: list[str] = []
    for term in sorted(_FORBIDDEN_MODEL_TEXT, key=len, reverse=True):
        if term in result.casefold():
            result = re.sub(re.escape(term), "masked", result, flags=re.IGNORECASE)
            matched.append(term)
    return result, tuple(matched)


def _scan_messages(messages: Sequence[Mapping[str, str]]) -> None:
    for index, message in enumerate(messages):
        content = _require_string(message.get("content"), label=f"messages[{index}].content")
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            value = content
        _scan_model_safe(value, path=f"messages[{index}].content")


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validated_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be one non-empty JSON object")
    detached = _clone(value)
    declared = _require_sha(detached.get("identity_sha256"), label=f"{label}.identity_sha256")
    body = {key: row for key, row in detached.items() if key != "identity_sha256"}
    if declared != _sha(body):
        raise ValueError(f"{label}.identity_sha256 does not authenticate its identity")
    return detached


def _run_adaptive_remote_call_with_projection_authentication(
    *,
    runner: JsonDiscoveryJobRunner,
    job: DiscoveryJsonJob,
    runner_identity_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one wire call and authenticate its exact response projection.

    Repair stays outside this helper so one logical caller can bind an initial
    physical call and its optional repair into the shared base-policy trace.
    The boundary is intentionally extractable into the common executor without
    changing any job, cache, repair, or trace identity.
    """

    before = tuple(_clone(row) for row in runner.execution_metadata)
    try:
        raw = runner.run_json(job=job)
    except Exception as exc:
        after = tuple(_clone(row) for row in runner.execution_metadata)
        if after[: len(before)] != before or len(after) != len(before) + 1:
            raise ValueError(
                "runner must append exactly one immutable metadata record per adaptive call"
            ) from exc
        metadata = after[-1]
        if not isinstance(metadata, Mapping):
            raise TypeError("adaptive runner failure metadata must be one object") from exc
        if (
            metadata.get("job_id") != job.job_id
            or metadata.get("runner_identity_sha256") != runner_identity_sha256
        ):
            raise ValueError("adaptive runner failure metadata changed its job binding") from exc
        category = getattr(exc, "discovery_response_failure_category", None)
        failed_content = getattr(exc, "failed_response_content", None)
        if category in {STRICT_JSON_PARSE_FAILURE, RAW_TRANSPORT_BUDGET_FAILURE} and isinstance(
            failed_content, str
        ):
            if metadata.get("outcome") != "invalid_response":
                raise ValueError(
                    "adaptive transport-validation failure lacks invalid-response metadata"
                ) from exc
            attempts = metadata.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                raise ValueError(
                    "adaptive transport-validation failure lacks attempt metadata"
                ) from exc
            expected = hashlib.sha256(failed_content.encode("utf-8")).hexdigest()
            if attempts[-1].get("content_sha256") != expected:
                raise ValueError(
                    "adaptive invalid-response metadata does not authenticate raw content"
                ) from exc
        raise
    after = tuple(_clone(row) for row in runner.execution_metadata)
    if after[: len(before)] != before or len(after) != len(before) + 1:
        raise ValueError(
            "runner must append exactly one immutable metadata record per adaptive call"
        )
    metadata = after[-1]
    if not isinstance(metadata, Mapping):
        raise TypeError("adaptive runner success metadata must be one object")
    wire = _clone(raw)
    if not isinstance(wire, Mapping):
        raise TypeError("adaptive runner must return one JSON object")
    if (
        metadata.get("job_id") != job.job_id
        or metadata.get("runner_identity_sha256") != runner_identity_sha256
        or metadata.get("outcome") != "success"
    ):
        raise ValueError("adaptive runner metadata does not authenticate a successful job")
    if _require_sha(
        metadata.get("parsed_response_sha256"),
        label="runner metadata parsed_response_sha256",
    ) != _sha(wire):
        raise ValueError("runner metadata parsed-response SHA differs from JSON")
    return wire, _clone(metadata)


def _validate_atomic_adaptive_planner_page_response(
    response: Any,
    *,
    target: str,
    evidence_id: str,
    owning_family: str,
    registry_names: Sequence[str],
) -> dict[str, Any]:
    """Normalize one target/evidence planner page without any semantic slicing."""

    root = _exact_mapping(
        response,
        keys={"review_targets", "no_lookback_needed"},
        label="atomic adaptive planner page response",
    )
    rows = root["review_targets"]
    if not isinstance(rows, list):
        raise TypeError("atomic planner review_targets must be a JSON list")
    if len(rows) > 1:
        raise ValueError("atomic planner page may return at most its one exact target")
    if not isinstance(root["no_lookback_needed"], bool):
        raise TypeError("atomic planner no_lookback_needed must be boolean")
    if target != NEW_MISSING_CONSTRUCT and target not in set(registry_names):
        raise ValueError("atomic planner page target is absent from the current registry")
    normalized_rows: list[dict[str, Any]] = []
    normalization_events: list[dict[str, Any]] = []
    if rows:
        row = _exact_mapping(
            rows[0],
            keys={
                "target",
                "problem",
                "relevant_architectures",
                "requested_evidence_ids",
                "reason",
            },
            label="atomic planner review target",
        )
        observed_target = _require_identifier(row["target"], label="review target")
        if observed_target != target:
            raise ValueError("atomic planner returned a target outside its exact page")
        families, duplicate_families = _deduplicated_string_list(
            row["relevant_architectures"],
            label="atomic planner relevant_architectures",
            identifiers=True,
        )
        if not set(families) <= ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("atomic planner cites an inactive architecture")
        requested, duplicate_requested = _deduplicated_string_list(
            row["requested_evidence_ids"],
            label="atomic planner requested_evidence_ids",
            allow_empty=True,
            identifiers=True,
        )
        if set(requested) - {evidence_id}:
            raise ValueError("atomic planner requested evidence outside its exact page")
        retained_families = list(families)
        if requested and owning_family not in retained_families:
            retained_families.append(owning_family)
            normalization_events.append(
                {
                    "field": "relevant_architectures",
                    "action": "owning_architecture_added",
                    "values": [owning_family],
                }
            )
        for field_name, duplicates in (
            ("relevant_architectures", duplicate_families),
            ("requested_evidence_ids", duplicate_requested),
        ):
            if duplicates:
                normalization_events.append(
                    {
                        "field": field_name,
                        "action": "duplicate_occurrences_removed",
                        "values": list(duplicates),
                    }
                )
        problem, masked_problem = _sanitize_model_authored_text(
            _require_string(row["problem"], label="atomic planner problem")
        )
        reason, masked_reason = _sanitize_model_authored_text(
            _require_string(row["reason"], label="atomic planner reason")
        )
        for field_name, masked in (("problem", masked_problem), ("reason", masked_reason)):
            if masked:
                normalization_events.append(
                    {
                        "field": field_name,
                        "action": "policy_channel_terms_masked",
                        "values": list(masked),
                    }
                )
        normalized_rows.append(
            {
                "target": target,
                "problem": problem,
                "relevant_architectures": retained_families,
                "requested_evidence_ids": list(requested),
                "reason": reason,
            }
        )
    normalized_no_lookback = not any(row["requested_evidence_ids"] for row in normalized_rows)
    if root["no_lookback_needed"] is not normalized_no_lookback:
        normalization_events.append(
            {
                "field": "no_lookback_needed",
                "action": "compiler_derived_from_exact_page_request",
                "values": [str(normalized_no_lookback).lower()],
            }
        )
    return {
        "review_targets": normalized_rows,
        "no_lookback_needed": normalized_no_lookback,
        "page_normalization_audit": {
            "target": target,
            "evidence_id": evidence_id,
            "owning_family": owning_family,
            "normalization_events": normalization_events,
        },
    }


def _validate_atomic_adaptive_proposer_page_response(
    response: Any,
    *,
    planned_targets: Sequence[str],
    requested_evidence_ids: Sequence[str],
    diagnostic_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate one bounded proposal page while deferring cross-page support checks."""

    root = _exact_mapping(
        response,
        keys={"operations", "converged"},
        label="atomic adaptive proposer page response",
    )
    operations = root["operations"]
    if not isinstance(operations, list):
        raise TypeError("atomic proposer operations must be a JSON list")
    if len(operations) > 1:
        raise ValueError("atomic proposer page may emit at most one revision proposal")
    if not isinstance(root["converged"], bool):
        raise TypeError("atomic proposer converged must be boolean")
    if root["converged"] is not (not operations):
        raise ValueError("atomic proposer converged must be derived from its page operations")
    planned = set(planned_targets)
    available_evidence = set(requested_evidence_ids)
    available_diagnostics = set(diagnostic_ids)
    normalized: list[dict[str, Any]] = []
    normalization_events: list[dict[str, Any]] = []
    for index, raw in enumerate(operations):
        operation = _exact_mapping(
            raw,
            keys={
                "operation",
                "targets",
                "proposed_feature",
                "supporting_evidence_ids",
                "diagnostic_ids",
                "reason",
            },
            label=f"atomic proposer operations[{index}]",
        )
        kind = operation["operation"]
        if kind not in _OPERATIONS:
            raise ValueError("atomic proposer operation is invalid")
        targets, duplicate_targets = _deduplicated_string_list(
            operation["targets"],
            label="atomic proposer targets",
            feature_names=True,
        )
        support, duplicate_support = _deduplicated_string_list(
            operation["supporting_evidence_ids"],
            label="atomic proposer supporting_evidence_ids",
            allow_empty=True,
            identifiers=True,
        )
        cited_diagnostics, duplicate_diagnostics = _deduplicated_string_list(
            operation["diagnostic_ids"],
            label="atomic proposer diagnostic_ids",
            identifiers=True,
        )
        if not set(support) <= available_evidence:
            raise ValueError("atomic proposer cites evidence outside its exact page")
        if not set(cited_diagnostics) <= available_diagnostics:
            raise ValueError("atomic proposer cites a diagnostic outside its exact page")
        for field_name, duplicates in (
            ("targets", duplicate_targets),
            ("supporting_evidence_ids", duplicate_support),
            ("diagnostic_ids", duplicate_diagnostics),
        ):
            if duplicates:
                normalization_events.append(
                    {
                        "operation_index": index,
                        "field": field_name,
                        "action": "duplicate_occurrences_removed",
                        "values": list(duplicates),
                    }
                )
        proposed_raw = operation["proposed_feature"]
        if kind == "drop":
            if (
                len(targets) != 1
                or targets[0] not in planned
                or targets[0] == NEW_MISSING_CONSTRUCT
            ):
                raise ValueError("atomic drop must target its one planned existing feature")
            if support or not isinstance(proposed_raw, Mapping) or proposed_raw:
                raise ValueError("atomic drop requires no support and an empty proposed_feature")
            proposed: dict[str, Any] = {}
        else:
            if not support:
                raise ValueError("non-drop atomic proposal requires exact page evidence")
            proposed = AdaptiveHierarchicalStage1Reconsideration._validate_proposed_feature(
                proposed_raw,
                label="atomic proposer proposed_feature",
            )
            if kind == "add":
                if NEW_MISSING_CONSTRUCT not in planned or len(targets) != 1:
                    raise ValueError("atomic add requires the planned missing-construct target")
            elif kind == "merge":
                if len(targets) < 2 or not set(targets) <= planned:
                    raise ValueError("atomic merge requires planned existing targets")
            elif len(targets) != 1 or targets[0] not in planned:
                raise ValueError(f"atomic {kind} requires one planned existing target")
            for field_name in ("description", "definition_summary"):
                sanitized, masked = _sanitize_model_authored_text(str(proposed[field_name]))
                proposed[field_name] = sanitized
                if masked:
                    normalization_events.append(
                        {
                            "operation_index": index,
                            "field": f"proposed_feature.{field_name}",
                            "action": "policy_channel_terms_masked",
                            "values": list(masked),
                        }
                    )
        reason, masked_reason = _sanitize_model_authored_text(
            _require_string(operation["reason"], label="atomic proposer reason")
        )
        if masked_reason:
            normalization_events.append(
                {
                    "operation_index": index,
                    "field": "reason",
                    "action": "policy_channel_terms_masked",
                    "values": list(masked_reason),
                }
            )
        normalized.append(
            {
                "operation": kind,
                "targets": list(targets),
                "proposed_feature": proposed,
                "supporting_evidence_ids": list(support),
                "diagnostic_ids": list(cited_diagnostics),
                "reason": reason,
            }
        )
    return {
        "operations": normalized,
        "converged": not normalized,
        "page_normalization_audit": {
            "planned_targets": list(planned_targets),
            "requested_evidence_ids": list(requested_evidence_ids),
            "diagnostic_ids": list(diagnostic_ids),
            "normalization_events": normalization_events,
        },
    }


def _validate_adaptive_proposal_judgment(response: Any) -> dict[str, str]:
    row = _exact_mapping(
        response,
        keys={
            "decision",
            "canonical_name",
            "description",
            "unresolved_ambiguity",
            "reason",
        },
        label="adaptive proposal judgment",
    )
    decision = _require_string(row["decision"], label="proposal judgment decision")
    if decision not in {"accept", "reject"}:
        raise ValueError("adaptive proposal judgment decision is invalid")
    canonical_name = _require_string(
        row["canonical_name"],
        label="proposal judgment canonical_name",
        allow_empty=decision == "reject",
    )
    description = _require_string(
        row["description"],
        label="proposal judgment description",
        allow_empty=decision == "reject",
    )
    ambiguity = _require_string(
        row["unresolved_ambiguity"],
        label="proposal judgment unresolved_ambiguity",
        allow_empty=True,
    )
    if decision == "accept":
        _require_feature_name(canonical_name, label="proposal judgment canonical_name")
    elif canonical_name or description or ambiguity:
        raise ValueError("rejected adaptive proposal must leave definition fields empty")
    description, _ = _sanitize_model_authored_text(description)
    ambiguity, _ = _sanitize_model_authored_text(ambiguity)
    reason, _ = _sanitize_model_authored_text(
        _require_string(row["reason"], label="proposal judgment reason")
    )
    return {
        "decision": decision,
        "canonical_name": canonical_name,
        "description": description,
        "unresolved_ambiguity": ambiguity,
        "reason": reason,
    }


def _render_adaptive_proposal_judgment_messages(
    *,
    proposal_id: str,
    proposal: Mapping[str, Any],
    requested_raw_evidence_lookback: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    request = attach_hierarchical_discovery_response_contract(
        job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
        request={
            "job": "integrate_cross_architecture_group",
            "group_id": proposal_id,
            "proposal_id": proposal_id,
            "proposal": _clone(proposal),
            "compiler_owned_relations": {
                "one_proposal_per_judgment": True,
                "complete_support_restored_after_all_pages": True,
                "conflict_and_capacity_dispositions_deferred": True,
            },
            "requested_raw_evidence_lookback": [
                _clone(item) for item in requested_raw_evidence_lookback
            ],
        },
    )
    return (
        {"role": "system", "content": _ADAPTIVE_PROPOSAL_JUDGE_SYSTEM},
        {"role": "user", "content": canonical_json(request)},
    )


@dataclass(frozen=True)
class ExactSpentCatalogAuthentication:
    """Upstream-authenticated binding for one exact accumulated-spent catalog.

    This object does not authenticate Stage 1 itself.  It preserves the digest
    from the upstream authenticator and binds it to the exact catalog, accumulated
    spent scope, already-consumed gates, and the next still-sealed gate.
    """

    catalog_sha256: str
    split_fingerprint: str
    outer_fold: int
    catalog_scope: str
    inner_fold: int | None
    accumulated_spent_scope_sha256: str
    accumulated_spent_row_count: int
    consumed_gate_fingerprints: tuple[str, ...]
    still_sealed_gate_fingerprint: str
    upstream_authentication_sha256: str
    authentication_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("catalog_sha256", self.catalog_sha256),
            ("split_fingerprint", self.split_fingerprint),
            ("accumulated_spent_scope_sha256", self.accumulated_spent_scope_sha256),
            ("still_sealed_gate_fingerprint", self.still_sealed_gate_fingerprint),
            ("upstream_authentication_sha256", self.upstream_authentication_sha256),
            ("authentication_sha256", self.authentication_sha256),
        ):
            _require_sha(value, label=label)
        if isinstance(self.outer_fold, bool) or not isinstance(self.outer_fold, int):
            raise TypeError("outer_fold must be an integer")
        if self.outer_fold < 1:
            raise ValueError("outer_fold must be positive")
        _require_string(self.catalog_scope, label="catalog_scope")
        if self.inner_fold is not None and (
            isinstance(self.inner_fold, bool)
            or not isinstance(self.inner_fold, int)
            or self.inner_fold < 1
        ):
            raise ValueError("inner_fold must be a positive integer or None")
        _positive_integer(
            self.accumulated_spent_row_count,
            label="accumulated_spent_row_count",
        )
        if not self.consumed_gate_fingerprints:
            raise ValueError("adaptive reconsideration requires at least one consumed gate")
        if len(set(self.consumed_gate_fingerprints)) != len(self.consumed_gate_fingerprints):
            raise ValueError("consumed_gate_fingerprints cannot contain duplicates")
        for index, value in enumerate(self.consumed_gate_fingerprints):
            _require_sha(value, label=f"consumed_gate_fingerprints[{index}]")
        if self.still_sealed_gate_fingerprint in self.consumed_gate_fingerprints:
            raise ValueError("the still-sealed gate cannot already be consumed")
        if self.authentication_sha256 != _sha(self._identity_without_authentication_sha()):
            raise ValueError("authentication_sha256 does not bind exact-spent catalog state")

    @classmethod
    def create(
        cls,
        *,
        catalog: RoleNeutralEvidenceCatalog,
        accumulated_spent_scope_sha256: str,
        accumulated_spent_row_count: int,
        consumed_gate_fingerprints: Sequence[str],
        still_sealed_gate_fingerprint: str,
        upstream_authentication_sha256: str,
    ) -> "ExactSpentCatalogAuthentication":
        validate_role_neutral_catalog(catalog)
        values = tuple(consumed_gate_fingerprints)
        identity = {
            "schema_version": EXACT_SPENT_CATALOG_AUTHENTICATION_VERSION,
            "catalog_sha256": catalog.catalog_sha256,
            "split_fingerprint": catalog.split_fingerprint,
            "outer_fold": catalog.outer_fold,
            "catalog_scope": catalog.scope,
            "inner_fold": catalog.inner_fold,
            "accumulated_spent_scope_sha256": accumulated_spent_scope_sha256,
            "accumulated_spent_row_count": accumulated_spent_row_count,
            "consumed_gate_fingerprints": list(values),
            "still_sealed_gate_fingerprint": still_sealed_gate_fingerprint,
            "upstream_authentication_sha256": upstream_authentication_sha256,
        }
        return cls(
            catalog_sha256=catalog.catalog_sha256,
            split_fingerprint=catalog.split_fingerprint,
            outer_fold=catalog.outer_fold,
            catalog_scope=catalog.scope,
            inner_fold=catalog.inner_fold,
            accumulated_spent_scope_sha256=accumulated_spent_scope_sha256,
            accumulated_spent_row_count=accumulated_spent_row_count,
            consumed_gate_fingerprints=values,
            still_sealed_gate_fingerprint=still_sealed_gate_fingerprint,
            upstream_authentication_sha256=upstream_authentication_sha256,
            authentication_sha256=_sha(identity),
        )

    def _identity_without_authentication_sha(self) -> dict[str, Any]:
        return {
            "schema_version": EXACT_SPENT_CATALOG_AUTHENTICATION_VERSION,
            "catalog_sha256": self.catalog_sha256,
            "split_fingerprint": self.split_fingerprint,
            "outer_fold": self.outer_fold,
            "catalog_scope": self.catalog_scope,
            "inner_fold": self.inner_fold,
            "accumulated_spent_scope_sha256": self.accumulated_spent_scope_sha256,
            "accumulated_spent_row_count": self.accumulated_spent_row_count,
            "consumed_gate_fingerprints": list(self.consumed_gate_fingerprints),
            "still_sealed_gate_fingerprint": self.still_sealed_gate_fingerprint,
            "upstream_authentication_sha256": self.upstream_authentication_sha256,
        }

    def assert_matches(self, catalog: RoleNeutralEvidenceCatalog) -> None:
        validate_role_neutral_catalog(catalog)
        observed = (
            catalog.catalog_sha256,
            catalog.split_fingerprint,
            catalog.outer_fold,
            catalog.scope,
            catalog.inner_fold,
        )
        expected = (
            self.catalog_sha256,
            self.split_fingerprint,
            self.outer_fold,
            self.catalog_scope,
            self.inner_fold,
        )
        if observed != expected:
            raise ValueError("exact-spent authentication is bound to another catalog")

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._identity_without_authentication_sha(),
            "authentication_sha256": self.authentication_sha256,
        }


@dataclass(frozen=True)
class AdaptiveCurrentFeature:
    feature_name: str
    description: str
    value_shape_hypothesis: str
    source_families: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    definition_summary: str

    def __post_init__(self) -> None:
        _require_feature_name(self.feature_name, label="feature_name")
        _require_string(self.description, label="description")
        if self.value_shape_hypothesis not in _VALUE_SHAPES:
            raise ValueError("value_shape_hypothesis is invalid")
        if not self.source_families or len(set(self.source_families)) != len(self.source_families):
            raise ValueError("source_families must be non-empty and unique")
        unknown = set(self.source_families) - ACTIVE_STAGE1_CONCEPT_FAMILY_SET
        if unknown:
            raise ValueError(f"current feature cites unknown architectures: {sorted(unknown)}")
        if not self.supporting_evidence_ids or len(set(self.supporting_evidence_ids)) != len(
            self.supporting_evidence_ids
        ):
            raise ValueError("supporting_evidence_ids must be non-empty and unique")
        for index, evidence_id in enumerate(self.supporting_evidence_ids):
            _require_identifier(evidence_id, label=f"supporting_evidence_ids[{index}]")
        _require_string(self.definition_summary, label="definition_summary")
        _scan_model_safe(self.as_prompt_item(), path=f"registry.{self.feature_name}")

    def as_prompt_item(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "description": self.description,
            "value_shape_hypothesis": self.value_shape_hypothesis,
            "source_families": list(self.source_families),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "definition_summary": self.definition_summary,
        }


@dataclass(frozen=True)
class AdaptiveDiagnostic:
    diagnostic_id: str
    diagnostic_kind: str
    affected_features: tuple[str, ...]
    summary: str
    aggregate_metrics: Mapping[str, int | float | str | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.diagnostic_id, label="diagnostic_id")
        if self.diagnostic_kind not in _DIAGNOSTIC_KINDS:
            raise ValueError("diagnostic_kind is invalid")
        if len(set(self.affected_features)) != len(self.affected_features):
            raise ValueError("affected_features cannot contain duplicates")
        for index, feature_name in enumerate(self.affected_features):
            _require_feature_name(feature_name, label=f"affected_features[{index}]")
        _require_string(self.summary, label="summary")
        if not isinstance(self.aggregate_metrics, Mapping):
            raise TypeError("aggregate_metrics must be one flat JSON object")
        metrics: dict[str, int | float | str | bool | None] = {}
        for key, value in self.aggregate_metrics.items():
            _require_identifier(key, label="aggregate metric name")
            if _FORBIDDEN_MODEL_KEY.search(key):
                raise ValueError(f"forbidden aggregate diagnostic metric: {key}")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"aggregate diagnostic metric {key!r} must be finite")
            if value is not None and not isinstance(value, (bool, int, float, str)):
                raise TypeError("aggregate diagnostic metrics must be scalar")
            metrics[key] = value
        object.__setattr__(self, "aggregate_metrics", _clone(metrics))
        _scan_model_safe(self.as_prompt_item(), path=f"diagnostics.{self.diagnostic_id}")

    def as_prompt_item(self) -> dict[str, Any]:
        return {
            "diagnostic_id": self.diagnostic_id,
            "diagnostic_kind": self.diagnostic_kind,
            "affected_features": list(self.affected_features),
            "summary": self.summary,
            "aggregate_metrics": _clone(self.aggregate_metrics),
        }


@dataclass(frozen=True)
class AdaptiveReconsiderationConfig:
    max_atoms_per_chunk: int = DEFAULT_MAX_ATOMS_PER_ARCHITECTURE_CHUNK
    max_bytes_per_chunk: int = DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK
    max_semantic_member_ids_per_chunk: int = DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK
    max_lookback_ids_per_target: int = 8
    max_total_lookback_ids: int = 24
    max_total_lookback_bytes: int = 96_000
    max_operations: int = 4
    max_rendered_prompt_bytes: int = MAX_RENDERED_DISCOVERY_PROMPT_BYTES

    def __post_init__(self) -> None:
        for label in (
            "max_atoms_per_chunk",
            "max_bytes_per_chunk",
            "max_semantic_member_ids_per_chunk",
            "max_lookback_ids_per_target",
            "max_total_lookback_ids",
            "max_total_lookback_bytes",
            "max_operations",
            "max_rendered_prompt_bytes",
        ):
            _positive_integer(getattr(self, label), label=label)
        if self.max_lookback_ids_per_target > self.max_total_lookback_ids:
            raise ValueError("per-target lookback limit cannot exceed the total limit")
        if self.max_operations > HIERARCHICAL_DISCOVERY_MAX_ADAPTIVE_REVIEW_TARGETS:
            raise ValueError(
                "adaptive operation limit exceeds the fixed review-target response budget"
            )
        if self.max_rendered_prompt_bytes > MAX_RENDERED_DISCOVERY_PROMPT_BYTES:
            raise ValueError("adaptive prompt guard cannot exceed the immutable global guard")
        if self.max_atoms_per_chunk > HIERARCHICAL_DISCOVERY_MAX_ATOMS_PER_INTERPRET_JOB:
            raise ValueError("adaptive atom chunk bound exceeds the interpret response budget")
        if (
            self.max_semantic_member_ids_per_chunk
            > HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB
        ):
            raise ValueError("adaptive member chunk bound exceeds the interpret response budget")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_atoms_per_chunk": self.max_atoms_per_chunk,
            "max_bytes_per_chunk": self.max_bytes_per_chunk,
            "max_semantic_member_ids_per_chunk": self.max_semantic_member_ids_per_chunk,
            "max_lookback_ids_per_target": self.max_lookback_ids_per_target,
            "max_total_lookback_ids": self.max_total_lookback_ids,
            "max_total_lookback_bytes": self.max_total_lookback_bytes,
            "max_operations": self.max_operations,
            "max_rendered_prompt_bytes": self.max_rendered_prompt_bytes,
        }


def adaptive_hierarchical_stage1_prompt_contract() -> dict[str, Any]:
    """Expose the exact static model-call contract; dynamic fold content is absent."""

    selector = DiscoveryJobSettings.selector().as_dict()
    extraction = DiscoveryJobSettings.extraction().as_dict()
    stages = [
        {
            "stage": INTERPRET_CHUNK_JOB,
            "template_version": INTERPRET_JOB_VERSION,
            "settings": selector,
            "system_instruction": _ADAPTIVE_INTERPRET_SYSTEM,
            "dynamic_inputs": ["one_family_explanation", "one_family_evidence_chunk"],
            "user_payload_top_level_keys": list(_USER_PAYLOAD_TOP_LEVEL_KEYS[INTERPRET_CHUNK_JOB]),
            "dynamic_payload_shapes": {
                "evidence_item_keys": [
                    "evidence_id",
                    "source_family",
                    "observable_axes",
                    "member_ids",
                    "content",
                ]
            },
            "output_schema": _interpret_output_schema(),
        },
        {
            "stage": CONSOLIDATE_ARCHITECTURE_JOB,
            "template_version": CONSOLIDATE_JOB_VERSION,
            "settings": selector,
            "system_instruction": _ADAPTIVE_CONSOLIDATE_SYSTEM,
            "dynamic_inputs": ["one_family_interpreted_candidates"],
            "user_payload_top_level_keys": list(
                _USER_PAYLOAD_TOP_LEVEL_KEYS[CONSOLIDATE_ARCHITECTURE_JOB]
            ),
            "dynamic_payload_shapes": {
                "candidate_item_keys": [
                    "candidate_id",
                    "feature_name",
                    "description",
                    "supporting_evidence_ids",
                    "source_families",
                    "value_shape_hypothesis",
                    "unresolved_ambiguity",
                ],
            },
            "output_schema": _consolidation_output_schema(),
        },
        {
            "stage": COVERAGE_CRITIC_JOB,
            "template_version": COVERAGE_CRITIC_JOB_VERSION,
            "settings": selector,
            "system_instruction": _ADAPTIVE_COVERAGE_SYSTEM,
            "dynamic_inputs": [
                "one_family_evidence_chunk",
                "one_chunk_interpretation",
                "one_family_consolidation_projection",
            ],
            "user_payload_top_level_keys": list(_USER_PAYLOAD_TOP_LEVEL_KEYS[COVERAGE_CRITIC_JOB]),
            "dynamic_payload_shapes": {
                "evidence_item_keys": [
                    "evidence_id",
                    "source_family",
                    "observable_axes",
                    "member_ids",
                    "content",
                ],
                "chunk_interpretation_keys": [
                    "concepts",
                    "evidence_dispositions",
                ],
                "family_consolidation_keys": [
                    "canonical_concepts",
                    "candidate_dispositions",
                ],
            },
            "output_schema": _coverage_output_schema(),
        },
        {
            "stage": CROSS_ARCHITECTURE_PLANNER_JOB,
            "template_version": ADAPTIVE_PLANNER_INTERFACE_VERSION,
            "settings": selector,
            "system_instruction": _ADAPTIVE_PLANNER_SYSTEM,
            "dynamic_inputs": [
                "exactly_ten_compact_dossiers",
                "current_registry_without_raw_support_ids",
                "sanitized_observable_diagnostics",
            ],
            "user_payload_top_level_keys": list(
                _USER_PAYLOAD_TOP_LEVEL_KEYS[CROSS_ARCHITECTURE_PLANNER_JOB]
            ),
            "dynamic_payload_shapes": _clone(_CROSS_ARCHITECTURE_DYNAMIC_SHAPES),
            "output_schema": _planner_output_schema(),
        },
        {
            "stage": CROSS_ARCHITECTURE_INTEGRATION_JOB,
            "template_version": ADAPTIVE_PROPOSER_INTERFACE_VERSION,
            "settings": selector,
            "system_instruction": _ADAPTIVE_PROPOSER_SYSTEM,
            "dynamic_inputs": [
                "exactly_ten_compact_dossiers",
                "current_registry_without_raw_support_ids",
                "sanitized_observable_diagnostics",
                "validated_review_plan",
                "only_requested_current_catalog_atoms",
            ],
            "user_payload_top_level_keys": list(
                _USER_PAYLOAD_TOP_LEVEL_KEYS[CROSS_ARCHITECTURE_INTEGRATION_JOB]
            ),
            "dynamic_payload_shapes": {
                **_clone(_CROSS_ARCHITECTURE_DYNAMIC_SHAPES),
                "review_plan_keys": ["review_targets", "no_lookback_needed"],
                "requested_evidence_item_keys": [
                    "evidence_id",
                    "source_family",
                    "observable_axes",
                    "member_ids",
                    "content",
                ],
            },
            "output_schema": _proposer_output_schema(),
        },
        {
            "stage": EXTRACTION_DEFINITION_JOB,
            "template_version": EXTRACTION_DEFINITION_JOB_VERSION,
            "settings": extraction,
            "system_instruction": _ADAPTIVE_DEFINITION_SYSTEM,
            "dynamic_inputs": [
                "one_frozen_proposed_feature",
                "exactly_its_cited_requested_current_catalog_atoms",
            ],
            "user_payload_top_level_keys": list(
                _USER_PAYLOAD_TOP_LEVEL_KEYS[EXTRACTION_DEFINITION_JOB]
            ),
            "dynamic_payload_shapes": {
                "evidence_item_keys": [
                    "evidence_id",
                    "source_family",
                    "member_ids",
                    "content",
                ],
                "planner_lookback_constraints_keys": [
                    "aliases",
                    "units",
                    "categories",
                    "distinguish_from",
                ],
            },
            "output_schema": _definition_output_schema(),
        },
    ]
    for stage in stages:
        job_kind = stage["stage"]
        static_literals = {"job": _STATIC_JOB_LITERAL[job_kind]}
        if job_kind == EXTRACTION_DEFINITION_JOB:
            static_literals["vocabulary_grounding_policy"] = {
                key: value
                for key, value in extraction_vocabulary_grounding_policy().items()
                if key != "schema_version"
            }
        stage["static_user_payload_literals"] = static_literals
        stage["dynamic_user_payload_paths"] = [
            *_DYNAMIC_USER_PAYLOAD_PATHS[job_kind],
            "identifier_ownership",
            "output_schema",
        ]
    bounded_candidate_keys = [
        "candidate_id",
        "feature_name",
        "description",
        "source_families",
        "value_shape_hypothesis",
        "unresolved_ambiguity",
        "supporting_evidence_count",
    ]
    phased_stage_variants = [
        {
            "stage": CONSOLIDATE_ARCHITECTURE_JOB,
            "request_job": "compare_adaptive_candidate_relations",
            "template_version": CONSOLIDATE_JOB_VERSION,
            "settings": selector,
            "system_instruction": _ADAPTIVE_RELATION_SYSTEM,
            "dynamic_inputs": ["one_anchor_candidate", "at_most_seven_later_peers"],
            "user_payload_top_level_keys": [
                "job",
                "anchor_candidate_id",
                "peer_candidate_ids",
                "anchor_candidate",
                "peer_candidates",
                "source_family",
                "identifier_ownership",
                "output_schema",
            ],
            "dynamic_payload_shapes": {
                "candidate_item_keys": bounded_candidate_keys,
            },
            "output_schema": _relation_output_schema(),
            "static_user_payload_literals": {"job": "compare_adaptive_candidate_relations"},
            "dynamic_user_payload_paths": [
                "anchor_candidate_id",
                "peer_candidate_ids",
                "anchor_candidate",
                "peer_candidates",
                "source_family",
                "identifier_ownership",
                "output_schema",
            ],
        },
        {
            "stage": CONSOLIDATE_ARCHITECTURE_JOB,
            "request_job": "fold_adaptive_group_definition",
            "template_version": CONSOLIDATE_JOB_VERSION,
            "settings": selector,
            "system_instruction": _ADAPTIVE_DEFINITION_FOLD_SYSTEM,
            "dynamic_inputs": [
                "compiler_owned_group",
                "bounded_fresh_members",
                "prior_accumulator",
            ],
            "user_payload_top_level_keys": [
                "job",
                "group_id",
                "fold_index",
                "member_candidate_ids",
                "prior_accumulator",
                "fresh_candidates",
                "identifier_ownership",
                "output_schema",
            ],
            "dynamic_payload_shapes": {
                "candidate_item_keys": bounded_candidate_keys,
                "prior_accumulator_keys": [
                    "canonical_name",
                    "description",
                    "unresolved_ambiguity",
                    "reason",
                ],
            },
            "output_schema": _definition_fold_output_schema(),
            "static_user_payload_literals": {"job": "fold_adaptive_group_definition"},
            "dynamic_user_payload_paths": [
                "group_id",
                "fold_index",
                "member_candidate_ids",
                "prior_accumulator",
                "fresh_candidates",
                "identifier_ownership",
                "output_schema",
            ],
        },
        {
            "stage": COVERAGE_CRITIC_JOB,
            "request_job": "audit_adaptive_atomic_coverage",
            "template_version": COVERAGE_CRITIC_JOB_VERSION,
            "settings": selector,
            "system_instruction": _ADAPTIVE_ATOMIC_COVERAGE_SYSTEM,
            "dynamic_inputs": ["one_evidence_atom", "at_most_four_canonical_names"],
            "user_payload_top_level_keys": [
                "job",
                "atomic_review_id",
                "evidence_id",
                "canonical_names",
                "source_family",
                "evidence",
                "chunk_interpretation",
                "consolidation_page",
                "identifier_ownership",
                "output_schema",
            ],
            "dynamic_payload_shapes": {
                "evidence_item_keys": [
                    "evidence_id",
                    "source_family",
                    "observable_axes",
                    "member_ids",
                    "content",
                ],
                "chunk_interpretation_keys": ["concepts", "evidence_dispositions"],
                "consolidation_page_keys": [
                    "canonical_concepts",
                    "candidate_dispositions",
                ],
            },
            "output_schema": _atomic_coverage_output_schema(),
            "static_user_payload_literals": {"job": "audit_adaptive_atomic_coverage"},
            "dynamic_user_payload_paths": [
                "atomic_review_id",
                "evidence_id",
                "canonical_names",
                "source_family",
                "evidence",
                "chunk_interpretation",
                "consolidation_page",
                "identifier_ownership",
                "output_schema",
            ],
        },
        {
            "stage": CROSS_ARCHITECTURE_PLANNER_JOB,
            "request_job": "compare_cross_architecture_candidate_relations",
            "template_version": ADAPTIVE_PLANNER_INTERFACE_VERSION,
            "settings": selector,
            "system_instruction": _ADAPTIVE_RELATION_SYSTEM,
            "dynamic_inputs": ["one_revision_anchor", "at_most_seven_later_peers"],
            "user_payload_top_level_keys": [
                "job",
                "anchor_candidate_id",
                "peer_candidate_ids",
                "anchor_candidate",
                "peer_candidates",
                "identifier_ownership",
                "output_schema",
            ],
            "dynamic_payload_shapes": {
                "candidate_item_keys": bounded_candidate_keys,
            },
            "output_schema": _relation_output_schema(),
            "static_user_payload_literals": {
                "job": "compare_cross_architecture_candidate_relations"
            },
            "dynamic_user_payload_paths": [
                "anchor_candidate_id",
                "peer_candidate_ids",
                "anchor_candidate",
                "peer_candidates",
                "identifier_ownership",
                "output_schema",
            ],
        },
        {
            "stage": CROSS_ARCHITECTURE_PLANNER_JOB,
            "request_job": "fold_cross_architecture_group_definition",
            "template_version": ADAPTIVE_PLANNER_INTERFACE_VERSION,
            "settings": selector,
            "system_instruction": _ADAPTIVE_DEFINITION_FOLD_SYSTEM,
            "dynamic_inputs": [
                "compiler_owned_revision_or_target_group",
                "bounded_fresh_members",
                "prior_accumulator",
            ],
            "user_payload_top_level_keys": [
                "job",
                "group_id",
                "fold_index",
                "member_candidate_ids",
                "prior_accumulator",
                "fresh_candidates",
                "identifier_ownership",
                "output_schema",
            ],
            "dynamic_payload_shapes": {
                "candidate_item_keys": bounded_candidate_keys,
                "prior_accumulator_keys": [
                    "canonical_name",
                    "description",
                    "unresolved_ambiguity",
                    "reason",
                ],
            },
            "output_schema": _definition_fold_output_schema(),
            "static_user_payload_literals": {"job": "fold_cross_architecture_group_definition"},
            "dynamic_user_payload_paths": [
                "group_id",
                "fold_index",
                "member_candidate_ids",
                "prior_accumulator",
                "fresh_candidates",
                "identifier_ownership",
                "output_schema",
            ],
        },
        {
            "stage": CROSS_ARCHITECTURE_INTEGRATION_JOB,
            "request_job": "integrate_cross_architecture_group",
            "template_version": ADAPTIVE_PROPOSER_INTERFACE_VERSION,
            "settings": selector,
            "system_instruction": _ADAPTIVE_PROPOSAL_JUDGE_SYSTEM,
            "dynamic_inputs": ["one_revision_proposal", "its_exact_bounded_raw_evidence"],
            "user_payload_top_level_keys": [
                "job",
                "group_id",
                "proposal_id",
                "proposal",
                "compiler_owned_relations",
                "requested_raw_evidence_lookback",
                "identifier_ownership",
                "output_schema",
            ],
            "dynamic_payload_shapes": {
                "proposal_keys": [
                    "operation",
                    "targets",
                    "proposed_feature",
                    "supporting_evidence_ids",
                    "diagnostic_ids",
                    "reason",
                ],
                "evidence_item_keys": [
                    "evidence_id",
                    "source_family",
                    "observable_axes",
                    "member_ids",
                    "content",
                ],
            },
            "output_schema": _dynamic_output_schema_template(
                "decision",
                "canonical_name",
                "description",
                "unresolved_ambiguity",
                "reason",
            ),
            "static_user_payload_literals": {"job": "integrate_cross_architecture_group"},
            "dynamic_user_payload_paths": [
                "group_id",
                "proposal_id",
                "proposal",
                "compiler_owned_relations",
                "requested_raw_evidence_lookback",
                "identifier_ownership",
                "output_schema",
            ],
        },
        {
            "stage": EXTRACTION_DEFINITION_JOB,
            "request_job": "review_extraction_feature_evidence",
            "template_version": EXTRACTION_DEFINITION_JOB_VERSION,
            "settings": extraction,
            "system_instruction": _ADAPTIVE_EXTRACTION_PAGE_SYSTEM,
            "dynamic_inputs": ["one_feature", "one_exact_raw_support_item"],
            "user_payload_top_level_keys": [
                "job",
                "canonical_name",
                "review_id",
                "evidence_id",
                "value_shape_hypothesis",
                "raw_evidence",
                "identifier_ownership",
                "output_schema",
            ],
            "dynamic_payload_shapes": {
                "evidence_item_keys": [
                    "evidence_id",
                    "source_family",
                    "member_ids",
                    "content",
                ],
            },
            "output_schema": _dynamic_output_schema_template(
                "measurement_observation",
                "shape_observation",
                "literal_aliases",
                "literal_units",
                "literal_categories",
                "literal_distinctions",
                "missing_or_ambiguous",
                "reviewed_evidence",
            ),
            "static_user_payload_literals": {"job": "review_extraction_feature_evidence"},
            "dynamic_user_payload_paths": [
                "canonical_name",
                "review_id",
                "evidence_id",
                "value_shape_hypothesis",
                "raw_evidence",
                "identifier_ownership",
                "output_schema",
            ],
        },
        {
            "stage": EXTRACTION_DEFINITION_JOB,
            "request_job": "fold_extraction_evidence_definitions",
            "template_version": EXTRACTION_DEFINITION_JOB_VERSION,
            "settings": extraction,
            "system_instruction": _ADAPTIVE_DEFINITION_SYSTEM,
            "dynamic_inputs": [
                "one_feature",
                "at_most_eight_fresh_or_prior_review_inputs",
            ],
            "user_payload_top_level_keys": [
                "job",
                "canonical_name",
                "value_shape_hypothesis",
                "fold_index",
                "review_input_ids",
                "review_inputs",
                "planner_lookback_constraints",
                "vocabulary_grounding_policy",
                "identifier_ownership",
                "output_schema",
            ],
            "dynamic_payload_shapes": {
                "review_input_item_keys": ["review_input_id", "review"],
                "planner_lookback_constraints_keys": [
                    "aliases",
                    "units",
                    "categories",
                    "distinguish_from",
                ],
            },
            "output_schema": _dynamic_output_schema_template(
                "feature_name",
                "measurement",
                "representation",
                "aliases",
                "distinguish_from",
                "missing_or_ambiguous",
                "input_dispositions",
                "supporting_evidence_reviewed",
            ),
            "static_user_payload_literals": {
                "job": "fold_extraction_evidence_definitions",
                "vocabulary_grounding_policy": {
                    key: value
                    for key, value in extraction_vocabulary_grounding_policy().items()
                    if key != "schema_version"
                },
            },
            "dynamic_user_payload_paths": [
                "canonical_name",
                "value_shape_hypothesis",
                "fold_index",
                "review_input_ids",
                "review_inputs",
                "planner_lookback_constraints",
                "identifier_ownership",
                "output_schema",
            ],
        },
    ]
    body = {
        "schema_version": ADAPTIVE_PROMPT_CONTRACT_VERSION,
        "stage_order": [row["stage"] for row in stages],
        "stages": stages,
        "phased_stage_variants": phased_stage_variants,
        "dynamic_fold_content_in_static_contract": False,
        "complete_catalog_single_prompt_authorized": False,
        "direct_or_non_grounding_numerical_context_authorized": False,
        "row_note_oracle_or_temporal_policy_context_authorized": False,
    }
    return {**body, "prompt_contract_sha256": _sha(body)}


def adaptive_hierarchical_stage1_reconsideration_identity(
    config: AdaptiveReconsiderationConfig | None = None,
) -> dict[str, Any]:
    """Return the closed phased-policy identity authorized before execution."""

    chosen = config or AdaptiveReconsiderationConfig()
    if not isinstance(chosen, AdaptiveReconsiderationConfig):
        raise TypeError("config has the wrong type")
    config_dict = chosen.as_dict()
    prompt_contract = adaptive_hierarchical_stage1_prompt_contract()
    return {
        "schema_version": ADAPTIVE_HIERARCHY_VERSION,
        "authenticated_execution_version": ADAPTIVE_AUTHENTICATED_EXECUTION_VERSION,
        "executable_bridge_version": ADAPTIVE_EXECUTABLE_BRIDGE_VERSION,
        "implementation_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_bundle": adaptive_hierarchical_implementation_bundle(),
        "config": config_dict,
        "config_sha256": _sha(config_dict),
        "prompt_contract": prompt_contract,
        "phase_policy": {
            "round_1_initial_frozen_support_may_be_reused": True,
            "later_round_fresh_exact_spent_catalog_required": True,
            "later_round_all_ten_architectures_required": True,
            "architecture_at_a_time_interpretation_required": True,
            "lossless_exhaustive_family_relation_pages_required": True,
            "complete_link_family_compiler_required": True,
            "terminating_group_definition_folds_required": True,
            "lossless_atomic_coverage_pages_required": True,
            "compact_ten_dossier_planner_required": True,
            "exhaustive_target_evidence_planner_pages_required": True,
            "terminating_target_folds_required": True,
            "bounded_requested_id_pages_only": True,
            "every_revision_proposal_judged_and_ledgered": True,
            "final_operation_capacity_after_explicit_dispositions_only": True,
            "one_raw_support_item_per_extraction_page_required": True,
            "terminating_extraction_support_folds_required": True,
            "proposal_freeze_before_next_gate_required": True,
            "complete_catalog_dump_forbidden": True,
            "direct_numerical_model_context_forbidden": True,
            "non_grounding_numerical_model_context_forbidden": True,
            "row_or_note_model_context_forbidden": True,
            "oracle_or_temporal_policy_model_context_forbidden": True,
        },
    }


def _assert_adaptive_job_prompt_contract(
    *,
    job_kind: str,
    messages: Sequence[Mapping[str, str]],
    settings: DiscoveryJobSettings,
) -> None:
    """Authenticate one rendered job against the exact static payload contract."""

    contract = adaptive_hierarchical_stage1_prompt_contract()
    if len(messages) != 2 or messages[1].get("role") != "user":
        raise ValueError(f"{job_kind} must contain exactly one JSON user payload")
    try:
        request = json.loads(messages[1]["content"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"{job_kind} user payload must be strict JSON") from exc
    if not isinstance(request, Mapping):
        raise ValueError(f"{job_kind} user payload must be one JSON object")
    variants = {
        (row["stage"], row["request_job"]): row for row in contract["phased_stage_variants"]
    }
    stage = variants.get((job_kind, request.get("job")))
    if stage is None:
        stage = {row["stage"]: row for row in contract["stages"]}.get(job_kind)
    if stage is None:
        raise ValueError(f"adaptive prompt contract has no stage {job_kind!r}")
    if messages[0] != {
        "role": "system",
        "content": stage["system_instruction"],
    }:
        raise ValueError(f"{job_kind} system prompt differs from the approved contract")
    if settings.as_dict() != stage["settings"]:
        raise ValueError(f"{job_kind} settings differ from the approved contract")
    if not isinstance(request, Mapping) or set(request) != set(
        stage["user_payload_top_level_keys"]
    ):
        raise ValueError(f"{job_kind} user payload differs from its closed top-level schema")
    static_literals = stage["static_user_payload_literals"]
    for field_name, expected_literal in static_literals.items():
        if request.get(field_name) != expected_literal:
            raise ValueError(
                f"{job_kind} static {field_name} literal differs from the approved contract"
            )
    dynamic_roots = {str(path).split(".", 1)[0] for path in stage["dynamic_user_payload_paths"]}
    static_roots = set(static_literals)
    if set(request) != {"output_schema", *static_roots, *dynamic_roots}:
        raise ValueError(f"{job_kind} has an undeclared dynamic user-payload slot")

    def assert_cross_architecture_shapes() -> None:
        shapes = stage["dynamic_payload_shapes"]
        for dossier in request["architecture_dossiers"]:
            if set(dossier) != set(shapes["architecture_dossier_keys"]):
                raise ValueError(f"{job_kind} architecture dossier shape changed")
            if set(dossier["coverage"]) != set(shapes["dossier_coverage_keys"]):
                raise ValueError(f"{job_kind} dossier coverage shape changed")
            for candidate in dossier["architecture_candidates"]:
                if set(candidate) != set(shapes["dossier_candidate_keys"]):
                    raise ValueError(f"{job_kind} dossier candidate shape changed")
        for field_name, schema_name in (
            ("current_registry", "current_registry_item_keys"),
            ("diagnostics", "diagnostic_item_keys"),
        ):
            for row in request[field_name]:
                if set(row) != set(shapes[schema_name]):
                    raise ValueError(f"{job_kind} {field_name} item shape changed")

    request_job = request["job"]
    if request_job in {
        "compare_adaptive_candidate_relations",
        "compare_cross_architecture_candidate_relations",
    }:
        item_fields = stage["dynamic_payload_shapes"]["candidate_item_keys"]
        dynamic_rows = [request["anchor_candidate"], *request["peer_candidates"]]
    elif request_job in {
        "fold_adaptive_group_definition",
        "fold_cross_architecture_group_definition",
    }:
        item_fields = stage["dynamic_payload_shapes"]["candidate_item_keys"]
        dynamic_rows = request["fresh_candidates"]
        accumulator = request["prior_accumulator"]
        if accumulator is not None and (
            not isinstance(accumulator, Mapping)
            or set(accumulator) != set(stage["dynamic_payload_shapes"]["prior_accumulator_keys"])
        ):
            raise ValueError("adaptive definition-fold accumulator shape changed")
    elif request_job == "audit_adaptive_atomic_coverage":
        item_fields = stage["dynamic_payload_shapes"]["evidence_item_keys"]
        dynamic_rows = [request["evidence"]]
        for field_name, schema_name in (
            ("chunk_interpretation", "chunk_interpretation_keys"),
            ("consolidation_page", "consolidation_page_keys"),
        ):
            if set(request[field_name]) != set(stage["dynamic_payload_shapes"][schema_name]):
                raise ValueError(f"{job_kind} {field_name} payload shape changed")
    elif request_job == "integrate_cross_architecture_group":
        shapes = stage["dynamic_payload_shapes"]
        if not isinstance(request["proposal"], Mapping) or set(request["proposal"]) != set(
            shapes["proposal_keys"]
        ):
            raise ValueError("adaptive proposal-judgment payload shape changed")
        item_fields = shapes["evidence_item_keys"]
        dynamic_rows = request["requested_raw_evidence_lookback"]
    elif request_job == "review_extraction_feature_evidence":
        item_fields = stage["dynamic_payload_shapes"]["evidence_item_keys"]
        dynamic_rows = [request["raw_evidence"]]
    elif request_job == "fold_extraction_evidence_definitions":
        shapes = stage["dynamic_payload_shapes"]
        constraints = request["planner_lookback_constraints"]
        if not isinstance(constraints, Mapping) or set(constraints) != set(
            shapes["planner_lookback_constraints_keys"]
        ):
            raise ValueError("adaptive extraction-fold planner constraints changed")
        item_fields = shapes["review_input_item_keys"]
        dynamic_rows = request["review_inputs"]
    elif job_kind == INTERPRET_CHUNK_JOB:
        item_fields = stage["dynamic_payload_shapes"]["evidence_item_keys"]
        dynamic_rows = request["evidence"]
    elif job_kind == CONSOLIDATE_ARCHITECTURE_JOB:
        item_fields = stage["dynamic_payload_shapes"]["candidate_item_keys"]
        dynamic_rows = request["candidates"]
    elif job_kind == COVERAGE_CRITIC_JOB:
        item_fields = stage["dynamic_payload_shapes"]["evidence_item_keys"]
        dynamic_rows = request["evidence"]
        for field_name, schema_name in (
            ("chunk_interpretation", "chunk_interpretation_keys"),
            ("family_consolidation", "family_consolidation_keys"),
        ):
            if set(request[field_name]) != set(stage["dynamic_payload_shapes"][schema_name]):
                raise ValueError(f"{job_kind} {field_name} payload shape changed")
    elif job_kind == CROSS_ARCHITECTURE_PLANNER_JOB:
        dynamic_rows = request["architecture_dossiers"]
        item_fields = stage["dynamic_payload_shapes"]["architecture_dossier_keys"]
        assert_cross_architecture_shapes()
    elif job_kind == CROSS_ARCHITECTURE_INTEGRATION_JOB:
        dynamic_rows = request["requested_evidence"]
        item_fields = stage["dynamic_payload_shapes"]["requested_evidence_item_keys"]
        assert_cross_architecture_shapes()
        if set(request["review_plan"]) != set(stage["dynamic_payload_shapes"]["review_plan_keys"]):
            raise ValueError("adaptive proposer review-plan payload shape changed")
    else:
        dynamic_rows = request["evidence"]
        item_fields = stage["dynamic_payload_shapes"]["evidence_item_keys"]
        constraints = request["planner_lookback_constraints"]
        if set(constraints) != set(
            stage["dynamic_payload_shapes"]["planner_lookback_constraints_keys"]
        ):
            raise ValueError("adaptive definition planner constraints changed")
    if not isinstance(dynamic_rows, list) or any(
        not isinstance(row, Mapping) or set(row) != set(item_fields) for row in dynamic_rows
    ):
        raise ValueError(f"{job_kind} dynamic item payload shape changed")
    expected_output, expected_ownership = build_hierarchical_discovery_response_contract(
        job_kind=job_kind,
        request=request,
    )
    if request["output_schema"] != expected_output:
        raise ValueError(f"{job_kind} nested output schema differs from the approved contract")
    if request["identifier_ownership"] != expected_ownership:
        raise ValueError(
            f"{job_kind} identifier ownership differs from the approved dynamic contract"
        )


@dataclass(frozen=True)
class AdaptiveFamilyConsolidation:
    """Content-addressed lossless result of one family's phased consolidation."""

    source_family: str
    candidate_ids: tuple[str, ...]
    relation_job_ids: tuple[str, ...]
    definition_job_ids: tuple[str, ...]
    terminal_dependency_ids: tuple[str, ...]
    normalized_response_sha256: str
    compiler_audit_sha256: str
    consolidation_id: str
    _normalized_response_json: str = field(repr=False)
    _compiler_audit_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("adaptive family consolidation has an inactive architecture")
        for label, values, allow_empty in (
            ("candidate_ids", self.candidate_ids, True),
            ("relation_job_ids", self.relation_job_ids, True),
            ("definition_job_ids", self.definition_job_ids, True),
            ("terminal_dependency_ids", self.terminal_dependency_ids, False),
        ):
            if (not allow_empty and not values) or len(values) != len(set(values)):
                raise ValueError(f"{label} has invalid or duplicate identifiers")
            for index, value in enumerate(values):
                _require_identifier(value, label=f"{label}[{index}]")
        for label, value in (
            ("normalized_response_sha256", self.normalized_response_sha256),
            ("compiler_audit_sha256", self.compiler_audit_sha256),
        ):
            _require_sha(value, label=label)
        normalized = json.loads(self._normalized_response_json)
        audit = json.loads(self._compiler_audit_json)
        if not isinstance(normalized, Mapping) or not isinstance(audit, Mapping):
            raise TypeError("adaptive family consolidation payloads must be JSON objects")
        if _sha(normalized) != self.normalized_response_sha256:
            raise ValueError("normalized_response_sha256 does not authenticate consolidation")
        if _sha(audit) != self.compiler_audit_sha256:
            raise ValueError("compiler_audit_sha256 does not authenticate consolidation audit")
        expected_id = f"adaptive_consolidation_{_sha(self._identity_without_id())}"
        if self.consolidation_id != expected_id:
            raise ValueError("consolidation_id does not authenticate the phased compilation")

    @classmethod
    def create(
        cls,
        *,
        source_family: str,
        candidate_ids: Sequence[str],
        relation_job_ids: Sequence[str],
        definition_job_ids: Sequence[str],
        terminal_dependency_ids: Sequence[str],
        normalized_response: Mapping[str, Any],
        compiler_audit: Mapping[str, Any],
    ) -> "AdaptiveFamilyConsolidation":
        response = _clone(normalized_response)
        audit = _clone(compiler_audit)
        values = {
            "schema_version": ADAPTIVE_FAMILY_CONSOLIDATION_VERSION,
            "source_family": source_family,
            "candidate_ids": list(candidate_ids),
            "relation_job_ids": list(relation_job_ids),
            "definition_job_ids": list(definition_job_ids),
            "terminal_dependency_ids": list(terminal_dependency_ids),
            "normalized_response_sha256": _sha(response),
            "compiler_audit_sha256": _sha(audit),
        }
        return cls(
            source_family=source_family,
            candidate_ids=tuple(candidate_ids),
            relation_job_ids=tuple(relation_job_ids),
            definition_job_ids=tuple(definition_job_ids),
            terminal_dependency_ids=tuple(terminal_dependency_ids),
            normalized_response_sha256=values["normalized_response_sha256"],
            compiler_audit_sha256=values["compiler_audit_sha256"],
            consolidation_id=f"adaptive_consolidation_{_sha(values)}",
            _normalized_response_json=canonical_json(response),
            _compiler_audit_json=canonical_json(audit),
        )

    def _identity_without_id(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_FAMILY_CONSOLIDATION_VERSION,
            "source_family": self.source_family,
            "candidate_ids": list(self.candidate_ids),
            "relation_job_ids": list(self.relation_job_ids),
            "definition_job_ids": list(self.definition_job_ids),
            "terminal_dependency_ids": list(self.terminal_dependency_ids),
            "normalized_response_sha256": self.normalized_response_sha256,
            "compiler_audit_sha256": self.compiler_audit_sha256,
        }

    @property
    def normalized_response(self) -> dict[str, Any]:
        return json.loads(self._normalized_response_json)

    @property
    def compiler_audit(self) -> dict[str, Any]:
        return json.loads(self._compiler_audit_json)

    def compilation_record(self) -> dict[str, Any]:
        return {**self._identity_without_id(), "consolidation_id": self.consolidation_id}


@dataclass(frozen=True)
class AdaptiveChunkCoverage:
    """Content-addressed aggregation of one losslessly paged chunk review."""

    source_family: str
    chunk_id: str
    evidence_ids: tuple[str, ...]
    coverage_job_ids: tuple[str, ...]
    normalized_response_sha256: str
    compiler_audit_sha256: str
    coverage_id: str
    _normalized_response_json: str = field(repr=False)
    _compiler_audit_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("adaptive chunk coverage has an inactive architecture")
        _require_identifier(self.chunk_id, label="chunk_id")
        for label, values in (
            ("evidence_ids", self.evidence_ids),
            ("coverage_job_ids", self.coverage_job_ids),
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{label} must be non-empty and unique")
            for index, value in enumerate(values):
                _require_identifier(value, label=f"{label}[{index}]")
        _require_sha(self.normalized_response_sha256, label="normalized_response_sha256")
        _require_sha(self.compiler_audit_sha256, label="compiler_audit_sha256")
        normalized = json.loads(self._normalized_response_json)
        audit = json.loads(self._compiler_audit_json)
        if (
            _sha(normalized) != self.normalized_response_sha256
            or _sha(audit) != self.compiler_audit_sha256
        ):
            raise ValueError("adaptive chunk coverage authentication changed")
        if tuple(normalized.get("reviewed_evidence_ids", ())) != self.evidence_ids:
            raise ValueError("adaptive chunk coverage lost an evidence disposition")
        expected_id = f"adaptive_coverage_{_sha(self._identity_without_id())}"
        if self.coverage_id != expected_id:
            raise ValueError("coverage_id does not authenticate the paged compilation")

    @classmethod
    def create(
        cls,
        *,
        source_family: str,
        chunk_id: str,
        evidence_ids: Sequence[str],
        coverage_job_ids: Sequence[str],
        normalized_response: Mapping[str, Any],
        compiler_audit: Mapping[str, Any],
    ) -> "AdaptiveChunkCoverage":
        response = _clone(normalized_response)
        audit = _clone(compiler_audit)
        values = {
            "schema_version": ADAPTIVE_CHUNK_COVERAGE_VERSION,
            "source_family": source_family,
            "chunk_id": chunk_id,
            "evidence_ids": list(evidence_ids),
            "coverage_job_ids": list(coverage_job_ids),
            "normalized_response_sha256": _sha(response),
            "compiler_audit_sha256": _sha(audit),
        }
        return cls(
            source_family=source_family,
            chunk_id=chunk_id,
            evidence_ids=tuple(evidence_ids),
            coverage_job_ids=tuple(coverage_job_ids),
            normalized_response_sha256=values["normalized_response_sha256"],
            compiler_audit_sha256=values["compiler_audit_sha256"],
            coverage_id=f"adaptive_coverage_{_sha(values)}",
            _normalized_response_json=canonical_json(response),
            _compiler_audit_json=canonical_json(audit),
        )

    def _identity_without_id(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_CHUNK_COVERAGE_VERSION,
            "source_family": self.source_family,
            "chunk_id": self.chunk_id,
            "evidence_ids": list(self.evidence_ids),
            "coverage_job_ids": list(self.coverage_job_ids),
            "normalized_response_sha256": self.normalized_response_sha256,
            "compiler_audit_sha256": self.compiler_audit_sha256,
        }

    @property
    def normalized_response(self) -> dict[str, Any]:
        return json.loads(self._normalized_response_json)

    @property
    def compiler_audit(self) -> dict[str, Any]:
        return json.loads(self._compiler_audit_json)

    def compilation_record(self) -> dict[str, Any]:
        return {**self._identity_without_id(), "coverage_id": self.coverage_id}


@dataclass(frozen=True)
class AdaptiveArchitectureDossier:
    """Numerical-free compact architecture summary with a complete private audit."""

    source_family: str
    catalog_sha256: str
    catalog_evidence_ids: tuple[str, ...]
    coverage_disposition_ids: tuple[str, ...]
    architecture_candidates: tuple[DiscoveryCandidate, ...]
    interpretation_job_ids: tuple[str, ...]
    consolidation_job_id: str
    coverage_job_ids: tuple[str, ...]
    dossier_sha256: str

    def __post_init__(self) -> None:
        if self.source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("adaptive dossier has an inactive architecture")
        _require_sha(self.catalog_sha256, label="catalog_sha256")
        _require_sha(self.dossier_sha256, label="dossier_sha256")
        for label, values in (
            ("catalog_evidence_ids", self.catalog_evidence_ids),
            ("coverage_disposition_ids", self.coverage_disposition_ids),
            ("interpretation_job_ids", self.interpretation_job_ids),
            ("coverage_job_ids", self.coverage_job_ids),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{label} must be non-empty and unique")
            for index, value in enumerate(values):
                _require_identifier(value, label=f"{label}[{index}]")
        _require_identifier(self.consolidation_job_id, label="consolidation_job_id")
        if set(self.catalog_evidence_ids) != set(self.coverage_disposition_ids):
            raise ValueError("adaptive dossier coverage must disposition every catalog atom")
        candidate_ids = [candidate.candidate_id for candidate in self.architecture_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("adaptive dossier candidate IDs cannot repeat")
        available = set(self.catalog_evidence_ids)
        for candidate in self.architecture_candidates:
            if candidate.source_families != (self.source_family,):
                raise ValueError("adaptive dossier candidates must remain architecture-local")
            if not set(candidate.supporting_evidence_ids) <= available:
                raise ValueError("adaptive dossier candidate cites another catalog")
        if self.dossier_sha256 != _sha(self._identity_without_sha()):
            raise ValueError("dossier_sha256 does not authenticate the adaptive dossier")

    @classmethod
    def create(
        cls,
        *,
        source_family: str,
        catalog_sha256: str,
        catalog_evidence_ids: Sequence[str],
        coverage_disposition_ids: Sequence[str],
        architecture_candidates: Sequence[DiscoveryCandidate],
        interpretation_job_ids: Sequence[str],
        consolidation_job_id: str,
        coverage_job_ids: Sequence[str],
    ) -> "AdaptiveArchitectureDossier":
        values = {
            "schema_version": ADAPTIVE_DOSSIER_VERSION,
            "source_family": source_family,
            "catalog_sha256": catalog_sha256,
            "catalog_evidence_ids": list(catalog_evidence_ids),
            "coverage_disposition_ids": list(coverage_disposition_ids),
            "architecture_candidates": [
                candidate.as_prompt_item() for candidate in architecture_candidates
            ],
            "interpretation_job_ids": list(interpretation_job_ids),
            "consolidation_job_id": consolidation_job_id,
            "coverage_job_ids": list(coverage_job_ids),
        }
        return cls(
            source_family=source_family,
            catalog_sha256=catalog_sha256,
            catalog_evidence_ids=tuple(catalog_evidence_ids),
            coverage_disposition_ids=tuple(coverage_disposition_ids),
            architecture_candidates=tuple(architecture_candidates),
            interpretation_job_ids=tuple(interpretation_job_ids),
            consolidation_job_id=consolidation_job_id,
            coverage_job_ids=tuple(coverage_job_ids),
            dossier_sha256=_sha(values),
        )

    def _identity_without_sha(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_DOSSIER_VERSION,
            "source_family": self.source_family,
            "catalog_sha256": self.catalog_sha256,
            "catalog_evidence_ids": list(self.catalog_evidence_ids),
            "coverage_disposition_ids": list(self.coverage_disposition_ids),
            "architecture_candidates": [
                candidate.as_prompt_item() for candidate in self.architecture_candidates
            ],
            "interpretation_job_ids": list(self.interpretation_job_ids),
            "consolidation_job_id": self.consolidation_job_id,
            "coverage_job_ids": list(self.coverage_job_ids),
        }

    def as_prompt_item(self) -> dict[str, Any]:
        """Return the compact model view; no raw atom content or numerical channel."""

        return {
            "source_family": self.source_family,
            "coverage": {
                "catalog_evidence_count": len(self.catalog_evidence_ids),
                "coverage_disposition_count": len(self.coverage_disposition_ids),
                "complete": True,
                "lookback_evidence_ids": list(self.catalog_evidence_ids),
            },
            "architecture_candidates": [
                candidate.as_prompt_item() for candidate in self.architecture_candidates
            ],
        }

    def as_authenticated_dict(self) -> dict[str, Any]:
        return {**self._identity_without_sha(), "dossier_sha256": self.dossier_sha256}


@dataclass(frozen=True)
class ResolvedAdaptiveLookback:
    requested_evidence_ids: tuple[str, ...]
    canonical_size_bytes: int
    lookback_sha256: str
    total_catalog_atom_count: int
    _items_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if len(set(self.requested_evidence_ids)) != len(self.requested_evidence_ids):
            raise ValueError("requested_evidence_ids cannot contain duplicates")
        for index, evidence_id in enumerate(self.requested_evidence_ids):
            _require_identifier(evidence_id, label=f"requested_evidence_ids[{index}]")
        if isinstance(self.canonical_size_bytes, bool) or not isinstance(
            self.canonical_size_bytes, int
        ):
            raise TypeError("canonical_size_bytes must be an integer")
        if self.canonical_size_bytes < 2:
            raise ValueError("canonical_size_bytes is invalid")
        _positive_integer(self.total_catalog_atom_count, label="total_catalog_atom_count")
        _require_sha(self.lookback_sha256, label="lookback_sha256")
        items = json.loads(self._items_json)
        if not isinstance(items, list):
            raise TypeError("lookback items must be one JSON list")
        observed_ids = tuple(str(item.get("evidence_id") or "") for item in items)
        if observed_ids != self.requested_evidence_ids:
            raise ValueError("resolved lookback item order differs from the request")
        if len(canonical_json(items).encode("utf-8")) != self.canonical_size_bytes:
            raise ValueError("lookback byte accounting does not authenticate")
        if _sha(items) != self.lookback_sha256:
            raise ValueError("lookback_sha256 does not authenticate resolved atoms")

    @property
    def items(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(self._items_json))

    def audit(self) -> dict[str, Any]:
        return {
            "requested_evidence_ids": list(self.requested_evidence_ids),
            "resolved_atom_count": len(self.requested_evidence_ids),
            "total_catalog_atom_count": self.total_catalog_atom_count,
            "canonical_size_bytes": self.canonical_size_bytes,
            "lookback_sha256": self.lookback_sha256,
            "all_catalog_atoms_returned": (
                len(self.requested_evidence_ids) == self.total_catalog_atom_count
            ),
            "all_catalog_atoms_rendered_in_one_model_prompt": False,
            "deterministic_id_order_preserved": True,
        }


class AdaptiveCoverageRequiresRevision(RuntimeError):
    """Raised when architecture-local coverage finds unresolved semantic loss."""

    def __init__(self, findings: Sequence[Mapping[str, Any]]) -> None:
        self.findings = tuple(_clone(findings))
        super().__init__(
            "architecture-local coverage found unresolved additions, splits, or support loss"
        )


@dataclass(frozen=True)
class FrozenAdaptiveReconsiderationRound:
    exact_spent_authentication_sha256: str
    catalog_sha256: str
    chunk_plan_sha256: str
    dossier_sha256s: tuple[str, ...]
    current_registry_sha256: str
    diagnostics_sha256: str
    planner_job_id: str
    planner_response_sha256: str
    lookback_sha256: str
    proposer_job_id: str
    proposal_sha256: str
    still_sealed_gate_fingerprint: str
    freeze_sha256: str
    _proposal_json: str = field(repr=False)
    _audit_json: str = field(repr=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("exact_spent_authentication_sha256", self.exact_spent_authentication_sha256),
            ("catalog_sha256", self.catalog_sha256),
            ("chunk_plan_sha256", self.chunk_plan_sha256),
            ("current_registry_sha256", self.current_registry_sha256),
            ("diagnostics_sha256", self.diagnostics_sha256),
            ("planner_response_sha256", self.planner_response_sha256),
            ("lookback_sha256", self.lookback_sha256),
            ("proposal_sha256", self.proposal_sha256),
            ("still_sealed_gate_fingerprint", self.still_sealed_gate_fingerprint),
            ("freeze_sha256", self.freeze_sha256),
        ):
            _require_sha(value, label=label)
        if len(self.dossier_sha256s) != len(ACTIVE_STAGE1_CONCEPT_FAMILIES):
            raise ValueError("frozen adaptive round must bind exactly ten dossiers")
        for index, value in enumerate(self.dossier_sha256s):
            _require_sha(value, label=f"dossier_sha256s[{index}]")
        _require_identifier(self.planner_job_id, label="planner_job_id")
        _require_identifier(self.proposer_job_id, label="proposer_job_id")
        proposal = json.loads(self._proposal_json)
        audit = json.loads(self._audit_json)
        if _sha(proposal) != self.proposal_sha256:
            raise ValueError("proposal_sha256 does not authenticate the frozen proposal")
        if self.freeze_sha256 != _sha(self._identity_without_freeze_sha()):
            raise ValueError("freeze_sha256 does not authenticate the adaptive round")
        if audit.get("proposal_frozen_before_next_gate") is not True:
            raise ValueError("adaptive round lacks its pre-gate proposal freeze assertion")

    @property
    def proposal(self) -> dict[str, Any]:
        return json.loads(self._proposal_json)

    @property
    def audit(self) -> dict[str, Any]:
        return json.loads(self._audit_json)

    def _identity_without_freeze_sha(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_ROUND_FREEZE_VERSION,
            "exact_spent_authentication_sha256": self.exact_spent_authentication_sha256,
            "catalog_sha256": self.catalog_sha256,
            "chunk_plan_sha256": self.chunk_plan_sha256,
            "dossier_sha256s": list(self.dossier_sha256s),
            "current_registry_sha256": self.current_registry_sha256,
            "diagnostics_sha256": self.diagnostics_sha256,
            "planner_job_id": self.planner_job_id,
            "planner_response_sha256": self.planner_response_sha256,
            "lookback_sha256": self.lookback_sha256,
            "proposer_job_id": self.proposer_job_id,
            "proposal_sha256": self.proposal_sha256,
            "still_sealed_gate_fingerprint": self.still_sealed_gate_fingerprint,
            "proposal": self.proposal,
            "audit": self.audit,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._identity_without_freeze_sha(), "freeze_sha256": self.freeze_sha256}


@dataclass(frozen=True)
class FrozenAdaptiveExecutableRevision:
    """Final executable-contract freeze derived from one frozen proposal."""

    proposal_freeze_sha256: str
    definition_job_ids: tuple[str, ...]
    definition_response_sha256s: tuple[str, ...]
    applied_specs_sha256: str
    executable_freeze_sha256: str
    _applied_json: str = field(repr=False)
    _audit_json: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_sha(self.proposal_freeze_sha256, label="proposal_freeze_sha256")
        _require_sha(self.applied_specs_sha256, label="applied_specs_sha256")
        _require_sha(self.executable_freeze_sha256, label="executable_freeze_sha256")
        if len(self.definition_job_ids) != len(self.definition_response_sha256s):
            raise ValueError("definition job and response SHA counts differ")
        if len(set(self.definition_job_ids)) != len(self.definition_job_ids):
            raise ValueError("definition_job_ids cannot repeat")
        for index, job_id in enumerate(self.definition_job_ids):
            _require_identifier(job_id, label=f"definition_job_ids[{index}]")
        for index, digest in enumerate(self.definition_response_sha256s):
            _require_sha(digest, label=f"definition_response_sha256s[{index}]")
        applied = json.loads(self._applied_json)
        audit = json.loads(self._audit_json)
        if not isinstance(applied, Mapping) or not isinstance(audit, Mapping):
            raise TypeError("executable revision JSON payloads are malformed")
        required = {
            "specs",
            "reextract_specs",
            "removed_names",
            "added_names",
            "extraction_changed_names",
            "role_only_changed_names",
            "operation_audit",
        }
        if set(applied) != required:
            raise ValueError("executable revision applied payload has a wrong closed schema")
        if self.applied_specs_sha256 != _sha(applied["specs"]):
            raise ValueError("applied_specs_sha256 does not authenticate executable specs")
        if audit.get("executable_revision_frozen_before_next_gate") is not True:
            raise ValueError("executable revision lacks its pre-gate freeze assertion")
        identity = {
            "schema_version": ADAPTIVE_EXECUTABLE_BRIDGE_VERSION,
            "proposal_freeze_sha256": self.proposal_freeze_sha256,
            "definition_job_ids": list(self.definition_job_ids),
            "definition_response_sha256s": list(self.definition_response_sha256s),
            "applied_specs_sha256": self.applied_specs_sha256,
            "applied": applied,
            "audit": audit,
        }
        if self.executable_freeze_sha256 != _sha(identity):
            raise ValueError("executable_freeze_sha256 does not authenticate the bridge")

    @property
    def applied(self) -> AppliedReviewOperations:
        row = json.loads(self._applied_json)
        return AppliedReviewOperations(
            specs=tuple(row["specs"]),
            reextract_specs=tuple(row["reextract_specs"]),
            removed_names=tuple(row["removed_names"]),
            added_names=tuple(row["added_names"]),
            extraction_changed_names=tuple(row["extraction_changed_names"]),
            role_only_changed_names=tuple(row["role_only_changed_names"]),
            operation_audit=tuple(row["operation_audit"]),
        )

    @property
    def audit(self) -> dict[str, Any]:
        return json.loads(self._audit_json)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_EXECUTABLE_BRIDGE_VERSION,
            "proposal_freeze_sha256": self.proposal_freeze_sha256,
            "definition_job_ids": list(self.definition_job_ids),
            "definition_response_sha256s": list(self.definition_response_sha256s),
            "applied_specs_sha256": self.applied_specs_sha256,
            "applied": json.loads(self._applied_json),
            "audit": self.audit,
            "executable_freeze_sha256": self.executable_freeze_sha256,
        }


@dataclass(frozen=True)
class ExecutedAdaptiveReconsiderationRound:
    """Authenticated transport/cache result for one frozen adaptive round."""

    frozen_round: FrozenAdaptiveReconsiderationRound
    executable_revision: FrozenAdaptiveExecutableRevision
    dossiers: tuple[AdaptiveArchitectureDossier, ...]
    lookback: ResolvedAdaptiveLookback
    runner_identity_sha256: str
    cache_identity_sha256: str
    execution_sha256: str
    _audit_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_round, FrozenAdaptiveReconsiderationRound):
            raise TypeError("frozen_round has the wrong type")
        self.frozen_round.__post_init__()
        if not isinstance(self.executable_revision, FrozenAdaptiveExecutableRevision):
            raise TypeError("executable_revision has the wrong type")
        self.executable_revision.__post_init__()
        if self.executable_revision.proposal_freeze_sha256 != self.frozen_round.freeze_sha256:
            raise ValueError("executable revision derives from another proposal freeze")
        if tuple(item.source_family for item in self.dossiers) != (ACTIVE_STAGE1_CONCEPT_FAMILIES):
            raise ValueError("executed adaptive round requires ten ordered dossiers")
        if self.frozen_round.dossier_sha256s != tuple(
            item.dossier_sha256 for item in self.dossiers
        ):
            raise ValueError("executed dossiers differ from the frozen round")
        if not isinstance(self.lookback, ResolvedAdaptiveLookback):
            raise TypeError("lookback has the wrong type")
        self.lookback.__post_init__()
        if self.lookback.lookback_sha256 != self.frozen_round.lookback_sha256:
            raise ValueError("executed lookback differs from the frozen round")
        _require_sha(self.runner_identity_sha256, label="runner_identity_sha256")
        _require_sha(self.cache_identity_sha256, label="cache_identity_sha256")
        _require_sha(self.execution_sha256, label="execution_sha256")
        audit = json.loads(self._audit_json)
        if not isinstance(audit, Mapping):
            raise TypeError("adaptive execution audit must be one JSON object")
        identity = {
            "schema_version": ADAPTIVE_AUTHENTICATED_EXECUTION_VERSION,
            "freeze_sha256": self.frozen_round.freeze_sha256,
            "executable_freeze_sha256": self.executable_revision.executable_freeze_sha256,
            "dossier_sha256s": [item.dossier_sha256 for item in self.dossiers],
            "lookback_sha256": self.lookback.lookback_sha256,
            "runner_identity_sha256": self.runner_identity_sha256,
            "cache_identity_sha256": self.cache_identity_sha256,
            "audit": audit,
        }
        if self.execution_sha256 != _sha(identity):
            raise ValueError("execution_sha256 does not authenticate the adaptive execution")

    @property
    def audit(self) -> dict[str, Any]:
        return json.loads(self._audit_json)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_AUTHENTICATED_EXECUTION_VERSION,
            "frozen_round": self.frozen_round.as_dict(),
            "executable_revision": self.executable_revision.as_dict(),
            "dossier_sha256s": [item.dossier_sha256 for item in self.dossiers],
            "lookback": self.lookback.audit(),
            "runner_identity_sha256": self.runner_identity_sha256,
            "cache_identity_sha256": self.cache_identity_sha256,
            "audit": self.audit,
            "execution_sha256": self.execution_sha256,
        }


class AdaptiveHierarchicalStage1Reconsideration:
    """Pure compiler/validator for one exact-spent adaptive reconsideration round."""

    def __init__(
        self,
        *,
        catalog: RoleNeutralEvidenceCatalog,
        exact_spent_authentication: ExactSpentCatalogAuthentication,
        family_explanations: Mapping[str, str],
        current_registry: Sequence[AdaptiveCurrentFeature],
        diagnostics: Sequence[AdaptiveDiagnostic],
        config: AdaptiveReconsiderationConfig | None = None,
    ) -> None:
        validate_role_neutral_catalog(catalog)
        if not isinstance(exact_spent_authentication, ExactSpentCatalogAuthentication):
            raise TypeError("exact_spent_authentication has the wrong type")
        exact_spent_authentication.assert_matches(catalog)
        counts = {
            family: len(catalog.family_atoms(family)) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        }
        missing = [family for family, count in counts.items() if count == 0]
        if missing:
            raise ValueError(f"adaptive exact-spent catalog misses architectures: {missing}")
        explanations = dict(family_explanations)
        if set(explanations) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("family_explanations must cover exactly all ten architectures")
        for family, explanation in explanations.items():
            _require_string(explanation, label=f"family_explanations[{family}]")
            _scan_model_safe(explanation, path=f"family_explanations.{family}")
        registry = tuple(current_registry)
        if any(not isinstance(item, AdaptiveCurrentFeature) for item in registry):
            raise TypeError("current_registry contains a non-feature entry")
        names = [item.feature_name for item in registry]
        if len(names) != len(set(names)):
            raise ValueError("current_registry feature names cannot repeat")
        if NEW_MISSING_CONSTRUCT in names:
            raise ValueError(
                "current_registry cannot use the reserved adaptive missing-construct target"
            )
        diagnostic_values = tuple(diagnostics)
        if not diagnostic_values:
            raise ValueError("adaptive reconsideration requires observable diagnostics")
        if any(not isinstance(item, AdaptiveDiagnostic) for item in diagnostic_values):
            raise TypeError("diagnostics contains a non-diagnostic entry")
        diagnostic_ids = [item.diagnostic_id for item in diagnostic_values]
        if len(diagnostic_ids) != len(set(diagnostic_ids)):
            raise ValueError("diagnostic IDs cannot repeat")
        unknown_affected = {
            feature
            for item in diagnostic_values
            for feature in item.affected_features
            if feature not in set(names)
        }
        if unknown_affected:
            raise ValueError(
                "diagnostics cite features absent from the current registry: "
                f"{sorted(unknown_affected)}"
            )
        chosen_config = config or AdaptiveReconsiderationConfig()
        if not isinstance(chosen_config, AdaptiveReconsiderationConfig):
            raise TypeError("config has the wrong type")
        chunk_plan = build_complete_architecture_chunks(
            catalog,
            max_atoms_per_chunk=chosen_config.max_atoms_per_chunk,
            max_bytes_per_chunk=chosen_config.max_bytes_per_chunk,
            max_semantic_member_ids_per_chunk=(chosen_config.max_semantic_member_ids_per_chunk),
        )
        delivery_audit = audit_complete_architecture_delivery(catalog, chunk_plan)
        if delivery_audit["all_catalog_atoms_delivered_exactly_once"] is not True:
            raise ValueError("adaptive chunks do not deliver every catalog atom exactly once")
        if delivery_audit["non_grounding_numerical_summaries_delivered"] is not False:
            raise ValueError("non-grounding numerical summaries entered adaptive chunks")

        self.catalog = catalog
        self.exact_spent_authentication = exact_spent_authentication
        self.family_explanations = {
            family: explanations[family] for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        }
        self.current_registry = registry
        self.diagnostics = diagnostic_values
        self.config = chosen_config
        self.chunk_plan = chunk_plan
        self.delivery_audit = _clone(delivery_audit)
        self._implementation_file_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._implementation_bundle = adaptive_hierarchical_implementation_bundle()
        self._atom_by_id = {atom.evidence_id: atom for atom in catalog.atoms}
        self._interpret_job_by_chunk_id: dict[str, DiscoveryJsonJob] = {}
        jobs: list[DiscoveryJsonJob] = []
        for chunk in chunk_plan.chunks:
            job = self._build_interpret_job(chunk)
            self._interpret_job_by_chunk_id[chunk.chunk_id] = job
            jobs.append(job)
        self.interpret_jobs = tuple(jobs)
        self._audit_interpret_delivery()

    @property
    def offline_contract(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_HIERARCHY_VERSION,
            "implementation_file_sha256": self._implementation_file_sha256,
            "implementation_bundle": _clone(self._implementation_bundle),
            "phased_policy_identity": (
                adaptive_hierarchical_stage1_reconsideration_identity(self.config)
            ),
            "exact_spent_authentication": self.exact_spent_authentication.as_dict(),
            "catalog_sha256": self.catalog.catalog_sha256,
            "chunk_plan_sha256": self.chunk_plan.plan_sha256,
            "delivery_audit": _clone(self.delivery_audit),
            "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "family_explanations_sha256": _sha(self.family_explanations),
            "current_registry_sha256": _sha(self._registry_private_items()),
            "diagnostics_sha256": _sha(self._diagnostic_prompt_items()),
            "config": self.config.as_dict(),
            "stage_order": [
                INTERPRET_CHUNK_JOB,
                "lossless_bounded_adaptive_candidate_relation_pages",
                "deterministic_complete_link_family_compilation",
                "terminating_adaptive_group_definition_folds",
                "lossless_bounded_adaptive_coverage_pages",
                "exhaustive_target_evidence_candidate_diagnostic_planner_pages",
                "terminating_adaptive_target_folds",
                "lossless_requested_id_resolution",
                "exhaustive_singleton_and_merge_pair_proposer_pages",
                "one_judgment_per_revision_proposal",
                "complete_link_revision_proposal_compilation",
                "explicit_conflict_and_capacity_dispositions",
                "freeze_validated_reconsideration_proposal",
                "one_raw_support_item_per_extraction_definition_page",
                "terminating_extraction_definition_folds",
                "deterministic_role_routing_and_candidate_contract_compilation",
                "freeze_executable_registry_revision",
            ],
            "assurances": {
                "all_ten_architectures_required": True,
                "one_architecture_per_interpretation": True,
                "every_unordered_family_candidate_pair_reviewed_exactly_once": True,
                "complete_link_prevents_transitive_false_positive_merges": True,
                "every_group_definition_folded_to_termination": True,
                "candidate_or_coverage_decision_truncation": False,
                "planner_and_proposer_lossless_paging_complete": True,
                "arbitrary_count_production_gate_open": True,
                "complete_family_coverage_required": True,
                "planner_compact_dossier_count": 10,
                "planner_raw_atom_count": 0,
                "lookback_requested_ids_only": True,
                "complete_catalog_single_prompt_forbidden": True,
                "complete_catalog_across_lossless_pages_supported": True,
                "extraction_support_lossless_paging_complete": True,
                "complete_operation_support_single_prompt_forbidden": True,
                "direct_numerical_model_context": False,
                "non_grounding_summary_model_context": False,
                "row_data_model_context": False,
                "note_text_model_context": False,
                "oracle_model_context": False,
                "temporal_policy_model_context": False,
                "extraction_definition_uses_requested_atoms_only": True,
                "extraction_definition_thinking_enabled": False,
                "proposer_summary_defines_categories_or_roles": False,
            },
        }

    @property
    def implementation_file_sha256(self) -> str:
        return self._implementation_file_sha256

    @property
    def implementation_bundle_sha256(self) -> str:
        return self._implementation_bundle["implementation_bundle_sha256"]

    @property
    def offline_contract_sha256(self) -> str:
        return _sha(self.offline_contract)

    @property
    def authenticated_cache_namespace_contract(self) -> dict[str, Any]:
        """Stable exact-catalog namespace shared by retries of one sealed round.

        Registry and diagnostic changes remain in each job ID/input binding, so
        planner/proposer/definition entries cannot cross-replay.  Keeping those
        retry-varying values out of the outer namespace permits the identical
        architecture interpretation/consolidation/coverage jobs to revalidate
        from cache instead of issuing duplicate remote calls.
        """

        return {
            "schema_version": ADAPTIVE_AUTHENTICATED_EXECUTION_VERSION,
            "phased_policy_identity": adaptive_hierarchical_stage1_reconsideration_identity(
                self.config
            ),
            "implementation_bundle": _clone(self._implementation_bundle),
            "exact_spent_authentication": self.exact_spent_authentication.as_dict(),
            "catalog_sha256": self.catalog.catalog_sha256,
            "chunk_plan_sha256": self.chunk_plan.plan_sha256,
            "family_explanations_sha256": _sha(self.family_explanations),
            "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "config": self.config.as_dict(),
        }

    @property
    def authenticated_cache_namespace_sha256(self) -> str:
        return _sha(self.authenticated_cache_namespace_contract)

    def _registry_prompt_items(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.current_registry:
            row = item.as_prompt_item()
            row.pop("supporting_evidence_ids")
            row["support_provenance"] = "private_authenticated_current_requested_id_lookback_only"
            rows.append(row)
        return rows

    def _registry_private_items(self) -> list[dict[str, Any]]:
        return [item.as_prompt_item() for item in self.current_registry]

    def _diagnostic_prompt_items(self) -> list[dict[str, Any]]:
        return [item.as_prompt_item() for item in self.diagnostics]

    def _create_job(
        self,
        *,
        job_kind: str,
        scope: str,
        dependencies: Sequence[str],
        messages: Sequence[Mapping[str, str]],
        input_bindings: Mapping[str, Any],
        settings: DiscoveryJobSettings | None = None,
    ) -> DiscoveryJsonJob:
        _scan_messages(messages)
        current_implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if current_implementation_sha256 != self._implementation_file_sha256:
            raise ValueError("adaptive hierarchy implementation changed during compilation")
        if adaptive_hierarchical_implementation_bundle() != self._implementation_bundle:
            raise ValueError("adaptive hierarchy dependency bundle changed during compilation")
        bindings = _clone(input_bindings)
        if "adaptive_implementation_file_sha256" in bindings:
            raise ValueError("adaptive_implementation_file_sha256 is reserved")
        if "adaptive_implementation_bundle_sha256" in bindings:
            raise ValueError("adaptive_implementation_bundle_sha256 is reserved")
        if HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING in bindings:
            raise ValueError(f"{HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING} is reserved")
        bindings["adaptive_implementation_file_sha256"] = self._implementation_file_sha256
        bindings["adaptive_implementation_bundle_sha256"] = self.implementation_bundle_sha256
        bindings[HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING] = (
            hierarchical_discovery_implementation_bundle()["implementation_bundle_sha256"]
        )
        chosen_settings = settings or DiscoveryJobSettings.selector()
        _assert_adaptive_job_prompt_contract(
            job_kind=job_kind,
            messages=messages,
            settings=chosen_settings,
        )
        job = DiscoveryJsonJob.create(
            job_kind=job_kind,
            scope=scope,
            dependencies=dependencies,
            settings=chosen_settings,
            messages=messages,
            input_bindings=bindings,
        )
        if len(job.rendered_messages_bytes) > self.config.max_rendered_prompt_bytes:
            raise ValueError("adaptive model prompt exceeds its fixed byte guard")
        return job

    def _build_interpret_job(self, chunk: ArchitectureEvidenceChunk) -> DiscoveryJsonJob:
        evidence = tuple(
            self._atom_by_id[str(row["evidence_id"])].as_discovery_item() for row in chunk.evidence
        )
        messages = render_interpret_evidence_chunk_messages(
            family_explanation=self.family_explanations[chunk.source_family],
            evidence=evidence,
        )
        if messages[0] != {"role": "system", "content": _ADAPTIVE_INTERPRET_SYSTEM}:
            raise ValueError("interpretation system prompt differs from the approved contract")
        return self._create_job(
            job_kind=INTERPRET_CHUNK_JOB,
            scope=f"adaptive.{chunk.source_family}.chunk_{chunk.chunk_index:03d}",
            dependencies=(),
            messages=messages,
            input_bindings={
                "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                "exact_spent_authentication_sha256": (
                    self.exact_spent_authentication.authentication_sha256
                ),
                "catalog_sha256": self.catalog.catalog_sha256,
                "chunk_plan_sha256": self.chunk_plan.plan_sha256,
                "chunk_id": chunk.chunk_id,
                "source_family": chunk.source_family,
            },
        )

    def _audit_interpret_delivery(self) -> None:
        expected = {
            atom.evidence_id: atom.as_discovery_item().as_prompt_item()
            for atom in self.catalog.atoms
        }
        observed: dict[str, dict[str, Any]] = {}
        for job in self.interpret_jobs:
            request = json.loads(job.messages[1]["content"])
            evidence = request["evidence"]
            families = {item["source_family"] for item in evidence}
            if len(families) != 1:
                raise ValueError("adaptive interpretation mixed Stage-1 architectures")
            for item in evidence:
                evidence_id = item["evidence_id"]
                if evidence_id in observed:
                    raise ValueError("adaptive interpretation duplicated a raw evidence atom")
                observed[evidence_id] = item
        if observed != expected:
            raise ValueError("adaptive interpretation did not deliver the exact complete catalog")

    @staticmethod
    def _responses_for_jobs(
        *,
        jobs: Sequence[DiscoveryJsonJob],
        responses: Mapping[str, Mapping[str, Any]],
        label: str,
    ) -> dict[str, Mapping[str, Any]]:
        if not isinstance(responses, Mapping):
            raise TypeError(f"{label} responses must be keyed by job ID")
        expected = {job.job_id for job in jobs}
        actual = set(responses)
        if actual != expected:
            raise ValueError(
                f"{label} response job IDs differ; missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        normalized: dict[str, Mapping[str, Any]] = {}
        for job_id, response in responses.items():
            if not isinstance(response, Mapping):
                raise TypeError(f"{label} response {job_id} must be one JSON object")
            normalized[job_id] = response
        return normalized

    def validate_interpretation_responses(
        self, responses: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        supplied = self._responses_for_jobs(
            jobs=self.interpret_jobs,
            responses=responses,
            label="interpretation",
        )
        validated: dict[str, dict[str, Any]] = {}
        chunk_by_job = {
            self._interpret_job_by_chunk_id[chunk.chunk_id].job_id: chunk
            for chunk in self.chunk_plan.chunks
        }
        for job in self.interpret_jobs:
            chunk = chunk_by_job[job.job_id]
            evidence = tuple(
                self._atom_by_id[str(row["evidence_id"])].as_discovery_item()
                for row in chunk.evidence
            )
            response = supplied[job.job_id]
            if isinstance(response.get("evidence_dispositions"), list):
                validated[job.job_id] = revalidate_normalized_interpret_evidence_chunk_response(
                    response,
                    evidence=evidence,
                )
            else:
                validated[job.job_id] = validate_interpret_evidence_chunk_response(
                    response,
                    evidence=evidence,
                )
        return validated

    def validate_interpretation_job_response(
        self,
        *,
        job: DiscoveryJsonJob,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate one interpretation for authenticated cache replay/store."""

        chunk_by_job = {
            self._interpret_job_by_chunk_id[chunk.chunk_id].job_id: chunk
            for chunk in self.chunk_plan.chunks
        }
        chunk = chunk_by_job.get(job.job_id)
        if chunk is None or job.job_kind != INTERPRET_CHUNK_JOB:
            raise ValueError("interpretation validator received an unknown adaptive job")
        evidence = tuple(
            self._atom_by_id[str(row["evidence_id"])].as_discovery_item() for row in chunk.evidence
        )
        return validate_interpret_evidence_chunk_response(response, evidence=evidence)

    def _interpret_candidates(
        self, responses: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, tuple[DiscoveryCandidate, ...]]:
        validated = self.validate_interpretation_responses(responses)
        candidates: dict[str, list[DiscoveryCandidate]] = {
            family: [] for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        }
        chunk_by_job = {
            self._interpret_job_by_chunk_id[chunk.chunk_id].job_id: chunk
            for chunk in self.chunk_plan.chunks
        }
        for job in self.interpret_jobs:
            family = chunk_by_job[job.job_id].source_family
            for concept in validated[job.job_id]["concepts"]:
                candidates[family].append(
                    self._candidate_from_interpretation(
                        job=job,
                        family=family,
                        concept=concept,
                    )
                )
        return {family: tuple(rows) for family, rows in candidates.items()}

    @staticmethod
    def _candidate_from_interpretation(
        *,
        job: DiscoveryJsonJob,
        family: str,
        concept: Mapping[str, Any],
    ) -> DiscoveryCandidate:
        identity = {
            "schema_version": ADAPTIVE_HIERARCHY_VERSION,
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

    @staticmethod
    def _render_consolidation_messages(
        *, source_family: str, candidates: Sequence[DiscoveryCandidate]
    ) -> tuple[dict[str, str], ...]:
        request = attach_hierarchical_discovery_response_contract(
            job_kind=CONSOLIDATE_ARCHITECTURE_JOB,
            request={
                "job": "consolidate_adaptive_architecture_candidates",
                "source_family": source_family,
                "candidates": [candidate.as_prompt_item() for candidate in candidates],
            },
        )
        return (
            {"role": "system", "content": _ADAPTIVE_CONSOLIDATE_SYSTEM},
            {"role": "user", "content": canonical_json(request)},
        )

    def build_consolidation_jobs(
        self, interpretation_responses: Mapping[str, Mapping[str, Any]]
    ) -> tuple[DiscoveryJsonJob, ...]:
        candidates = self._interpret_candidates(interpretation_responses)
        jobs: list[DiscoveryJsonJob] = []
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
            dependencies = tuple(
                self._interpret_job_by_chunk_id[chunk.chunk_id].job_id
                for chunk in self.chunk_plan.chunks
                if chunk.source_family == family
            )
            jobs.append(
                self._create_job(
                    job_kind=CONSOLIDATE_ARCHITECTURE_JOB,
                    scope=f"adaptive.{family}.consolidate",
                    dependencies=dependencies,
                    messages=self._render_consolidation_messages(
                        source_family=family,
                        candidates=candidates[family],
                    ),
                    input_bindings={
                        "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                        "exact_spent_authentication_sha256": (
                            self.exact_spent_authentication.authentication_sha256
                        ),
                        "catalog_sha256": self.catalog.catalog_sha256,
                        "source_family": family,
                        "interpretation_response_sha256s": [
                            _sha(interpretation_responses[job_id]) for job_id in dependencies
                        ],
                    },
                )
            )
        return tuple(jobs)

    @staticmethod
    def _validate_empty_consolidation_wire(response: Any) -> dict[str, Any]:
        return validate_consolidation_response(
            response,
            source_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            candidates=(),
        )

    @staticmethod
    def _revalidate_empty_consolidation_projection(response: Any) -> dict[str, Any]:
        return revalidate_normalized_consolidation_response(
            response,
            source_family=ACTIVE_STAGE1_CONCEPT_FAMILIES[0],
            candidates=(),
        )

    def validate_consolidation_responses(
        self,
        *,
        interpretation_responses: Mapping[str, Mapping[str, Any]],
        consolidation_responses: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        candidates = self._interpret_candidates(interpretation_responses)
        jobs = self.build_consolidation_jobs(interpretation_responses)
        supplied = self._responses_for_jobs(
            jobs=jobs,
            responses=consolidation_responses,
            label="consolidation",
        )
        validated: dict[str, dict[str, Any]] = {}
        for family, job in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, jobs):
            if candidates[family]:
                response = supplied[job.job_id]
                if isinstance(response.get("candidate_dispositions"), list):
                    validated[job.job_id] = revalidate_normalized_consolidation_response(
                        response,
                        source_family=family,
                        candidates=candidates[family],
                    )
                else:
                    validated[job.job_id] = validate_consolidation_response(
                        response,
                        source_family=family,
                        candidates=candidates[family],
                    )
            else:
                response = supplied[job.job_id]
                if isinstance(response.get("candidate_dispositions"), list):
                    validated[job.job_id] = self._revalidate_empty_consolidation_projection(
                        response
                    )
                else:
                    validated[job.job_id] = self._validate_empty_consolidation_wire(response)
        return validated

    def validate_consolidation_job_response(
        self,
        *,
        interpretation_responses: Mapping[str, Mapping[str, Any]],
        job: DiscoveryJsonJob,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate one architecture consolidation for authenticated caching."""

        jobs = self.build_consolidation_jobs(interpretation_responses)
        family_by_job = dict(zip((row.job_id for row in jobs), ACTIVE_STAGE1_CONCEPT_FAMILIES))
        family = family_by_job.get(job.job_id)
        if family is None or job.job_kind != CONSOLIDATE_ARCHITECTURE_JOB:
            raise ValueError("consolidation validator received an unknown adaptive job")
        candidates = self._interpret_candidates(interpretation_responses)[family]
        if candidates:
            return validate_consolidation_response(
                response,
                source_family=family,
                candidates=candidates,
            )
        return self._validate_empty_consolidation_wire(response)

    def _consolidated_candidates(
        self,
        *,
        interpretation_responses: Mapping[str, Mapping[str, Any]],
        consolidation_responses: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, tuple[DiscoveryCandidate, ...]]:
        jobs = self.build_consolidation_jobs(interpretation_responses)
        validated = self.validate_consolidation_responses(
            interpretation_responses=interpretation_responses,
            consolidation_responses=consolidation_responses,
        )
        result: dict[str, tuple[DiscoveryCandidate, ...]] = {}
        for family, job in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, jobs):
            rows: list[DiscoveryCandidate] = []
            for concept in validated[job.job_id]["canonical_concepts"]:
                identity = {
                    "schema_version": ADAPTIVE_HIERARCHY_VERSION,
                    "catalog_sha256": self.catalog.catalog_sha256,
                    "source_family": family,
                    "canonical_concept": concept,
                }
                rows.append(
                    DiscoveryCandidate(
                        candidate_id=f"candidate_{_sha(identity)}",
                        feature_name=str(concept["canonical_name"]),
                        description=str(concept["description"]),
                        supporting_evidence_ids=tuple(concept["supporting_evidence_ids"]),
                        source_families=(family,),
                        value_shape_hypothesis=str(concept["value_shape_hypothesis"]),
                        unresolved_ambiguity=str(concept["unresolved_ambiguity"]),
                    )
                )
            result[family] = tuple(rows)
        return result

    def _execute_phased_family_consolidations(
        self,
        *,
        interpretation_responses: Mapping[str, Mapping[str, Any]],
        run_job: Callable[
            [DiscoveryJsonJob, Callable[[Any], Mapping[str, Any]]],
            dict[str, Any],
        ],
    ) -> tuple[dict[str, AdaptiveFamilyConsolidation], tuple[dict[str, Any], ...]]:
        """Losslessly consolidate every family through bounded pair pages and folds."""

        interpretations = self.validate_interpretation_responses(interpretation_responses)
        candidates_by_family = self._interpret_candidates(interpretations)
        artifacts: dict[str, AdaptiveFamilyConsolidation] = {}
        compiler_records: list[dict[str, Any]] = []
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
            candidates = candidates_by_family[family]
            by_id = {candidate.candidate_id: candidate for candidate in candidates}
            family_chunks = tuple(
                chunk for chunk in self.chunk_plan.chunks if chunk.source_family == family
            )
            interpretation_jobs = tuple(
                self._interpret_job_by_chunk_id[chunk.chunk_id] for chunk in family_chunks
            )
            interpretation_job_ids = tuple(job.job_id for job in interpretation_jobs)
            interpretation_sha256s = tuple(
                _sha(interpretations[job.job_id]) for job in interpretation_jobs
            )
            schedule = bounded_candidate_relation_pages(candidates)
            normalized_pages: list[dict[str, Any]] = []
            relation_job_ids: list[str] = []
            relation_audits: list[dict[str, Any]] = []
            for page_index, page in enumerate(schedule):
                anchor_id = str(page["anchor_candidate_id"])
                peer_ids = tuple(str(value) for value in page["peer_candidate_ids"])
                messages = _render_candidate_relation_page_messages(
                    job="compare_adaptive_candidate_relations",
                    job_kind=CONSOLIDATE_ARCHITECTURE_JOB,
                    source_family=family,
                    anchor=by_id[anchor_id],
                    peers=tuple(by_id[peer_id] for peer_id in peer_ids),
                )
                relation_job = self._create_job(
                    job_kind=CONSOLIDATE_ARCHITECTURE_JOB,
                    scope=f"adaptive.{family}.relation_page_{page_index:06d}",
                    dependencies=interpretation_job_ids,
                    messages=messages,
                    input_bindings={
                        "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                        "exact_spent_authentication_sha256": (
                            self.exact_spent_authentication.authentication_sha256
                        ),
                        "catalog_sha256": self.catalog.catalog_sha256,
                        "source_family": family,
                        "relation_page": page,
                        "candidate_projection_sha256": _sha(
                            [
                                _bounded_candidate_projection(by_id[candidate_id])
                                for candidate_id in (anchor_id, *peer_ids)
                            ]
                        ),
                        "interpretation_response_sha256s": list(interpretation_sha256s),
                    },
                )
                normalized = run_job(
                    relation_job,
                    lambda raw, anchor_id=anchor_id, peer_ids=peer_ids: (
                        validate_candidate_relation_page_response(
                            raw,
                            anchor_candidate_id=anchor_id,
                            peer_candidate_ids=peer_ids,
                        )
                    ),
                )
                normalized_pages.append(normalized)
                relation_job_ids.append(relation_job.job_id)
                relation_audits.append(
                    {
                        "relation_page": page,
                        "job_id": relation_job.job_id,
                        "normalized_response_sha256": _sha(normalized),
                    }
                )

            grouped = compile_complete_link_candidate_groups(
                candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
                relation_pages=normalized_pages,
            )
            grouped_sha256 = _sha(grouped)
            definitions: dict[str, dict[str, Any]] = {}
            definition_job_ids: list[str] = []
            terminal_fold_job_ids: list[str] = []
            fold_audits: list[dict[str, Any]] = []
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
                ):
                    fresh_ids = tuple(str(value) for value in fold["member_candidate_ids"])
                    dependencies = (
                        (prior_job_id,) if prior_job_id is not None else tuple(relation_job_ids)
                    )
                    fold_index = int(fold["fold_index"])
                    messages = _render_candidate_definition_fold_messages(
                        job="fold_adaptive_group_definition",
                        job_kind=CONSOLIDATE_ARCHITECTURE_JOB,
                        group_id=group_id,
                        fold_index=fold_index,
                        candidates=tuple(by_id[value] for value in fresh_ids),
                        prior_accumulator=prior,
                    )
                    fold_job = self._create_job(
                        job_kind=CONSOLIDATE_ARCHITECTURE_JOB,
                        scope=(
                            f"adaptive.{family}.{group_id}.definition_fold_" f"{fold_index:06d}"
                        ),
                        dependencies=dependencies,
                        messages=messages,
                        input_bindings={
                            "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                            "exact_spent_authentication_sha256": (
                                self.exact_spent_authentication.authentication_sha256
                            ),
                            "catalog_sha256": self.catalog.catalog_sha256,
                            "source_family": family,
                            "group_compiler_sha256": grouped_sha256,
                            "group_id": group_id,
                            "group_member_candidate_ids_sha256": _sha(list(member_ids)),
                            "fold": fold,
                            "prior_accumulator_sha256": (
                                _sha(prior) if prior is not None else None
                            ),
                        },
                    )
                    prior = run_job(
                        fold_job,
                        _validate_candidate_definition_fold_response,
                    )
                    prior_job_id = fold_job.job_id
                    definition_job_ids.append(fold_job.job_id)
                    fold_audits.append(
                        {
                            "group_id": group_id,
                            "fold": fold,
                            "job_id": fold_job.job_id,
                            "normalized_response_sha256": _sha(prior),
                        }
                    )
                if prior is None or prior_job_id is None:
                    raise AssertionError("adaptive multi-member group did not terminate")
                definitions[group_id] = prior
                terminal_fold_job_ids.append(prior_job_id)

            normalized_consolidation = _compile_bounded_consolidation(
                source_family=family,
                candidates=candidates,
                grouped=grouped,
                definitions_by_group_id=definitions,
            )
            terminal_dependency_ids = (
                tuple(terminal_fold_job_ids) or tuple(relation_job_ids) or interpretation_job_ids
            )
            compiler_audit = {
                "schema_version": ADAPTIVE_FAMILY_CONSOLIDATION_VERSION,
                "source_family": family,
                "candidate_order": [candidate.candidate_id for candidate in candidates],
                "expected_unordered_pair_count": len(candidates) * (len(candidates) - 1) // 2,
                "relation_pages": relation_audits,
                "complete_link_compiler": grouped,
                "definition_folds": fold_audits,
                "final_definitions_sha256": _sha(definitions),
                "all_candidates_preserved": True,
                "candidate_or_decision_truncation_applied": False,
            }
            artifact = AdaptiveFamilyConsolidation.create(
                source_family=family,
                candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
                relation_job_ids=relation_job_ids,
                definition_job_ids=definition_job_ids,
                terminal_dependency_ids=terminal_dependency_ids,
                normalized_response=normalized_consolidation,
                compiler_audit=compiler_audit,
            )
            artifacts[family] = artifact
            compiler_records.append(
                {
                    "record_type": "adaptive_family_consolidation_compilation",
                    **artifact.compilation_record(),
                }
            )
        return artifacts, tuple(compiler_records)

    @staticmethod
    def _render_coverage_messages(
        *,
        family: str,
        evidence: Sequence[DiscoveryEvidenceItem],
        interpretation_response: Mapping[str, Any],
        consolidation_response: Mapping[str, Any],
    ) -> tuple[dict[str, str], ...]:
        request = attach_hierarchical_discovery_response_contract(
            job_kind=COVERAGE_CRITIC_JOB,
            request={
                "job": "audit_adaptive_architecture_coverage",
                "source_family": family,
                "evidence": [item.as_prompt_item() for item in evidence],
                "chunk_interpretation": interpretation_model_view(interpretation_response),
                "family_consolidation": consolidation_response,
            },
        )
        return (
            {"role": "system", "content": _ADAPTIVE_COVERAGE_SYSTEM},
            {"role": "user", "content": canonical_json(request)},
        )

    @staticmethod
    def _render_atomic_coverage_messages(
        *,
        family: str,
        evidence: DiscoveryEvidenceItem,
        interpretation_response: Mapping[str, Any],
        consolidation_response: Mapping[str, Any],
        canonical_names: Sequence[str],
        page_index: int,
    ) -> tuple[dict[str, str], ...]:
        atomic_review_id = f"coverage_review_{_sha({'evidence_id': evidence.evidence_id, 'page_index': page_index, 'canonical_names': list(canonical_names)})}"
        request = attach_hierarchical_discovery_response_contract(
            job_kind=COVERAGE_CRITIC_JOB,
            request={
                "job": "audit_adaptive_atomic_coverage",
                "atomic_review_id": atomic_review_id,
                "evidence_id": evidence.evidence_id,
                "canonical_names": list(canonical_names),
                "source_family": family,
                "evidence": evidence.as_prompt_item(),
                "chunk_interpretation": interpretation_model_view(interpretation_response),
                "consolidation_page": _clone(consolidation_response),
            },
        )
        return (
            {"role": "system", "content": _ADAPTIVE_ATOMIC_COVERAGE_SYSTEM},
            {"role": "user", "content": canonical_json(request)},
        )

    def _chunk_scoped_consolidation(
        self,
        *,
        chunk: ArchitectureEvidenceChunk,
        interpret_job: DiscoveryJsonJob,
        interpretation_response: Mapping[str, Any],
        consolidation_response: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Project a complete family consolidation onto one original evidence chunk."""

        chunk_candidates = tuple(
            self._candidate_from_interpretation(
                job=interpret_job,
                family=chunk.source_family,
                concept=concept,
            )
            for concept in interpretation_response["concepts"]
        )
        candidate_ids = {candidate.candidate_id for candidate in chunk_candidates}
        evidence_ids = {str(row["evidence_id"]) for row in chunk.evidence}
        concepts: list[dict[str, Any]] = []
        for concept in consolidation_response["canonical_concepts"]:
            members = [
                candidate_id
                for candidate_id in concept["member_candidate_ids"]
                if candidate_id in candidate_ids
            ]
            if not members:
                continue
            support = [
                evidence_id
                for evidence_id in concept["supporting_evidence_ids"]
                if evidence_id in evidence_ids
            ]
            concepts.append(
                {
                    **concept,
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
            raise RuntimeError("chunk-scoped adaptive coverage lost candidate dispositions")
        projected_support = {
            evidence_id
            for concept in concepts
            for evidence_id in concept["supporting_evidence_ids"]
        }
        expected_support = {
            evidence_id
            for candidate in chunk_candidates
            for evidence_id in candidate.supporting_evidence_ids
        }
        if projected_support != expected_support:
            raise RuntimeError("chunk-scoped adaptive coverage lost evidence support")
        return _clone(
            {
                "canonical_concepts": concepts,
                "candidate_dispositions": dispositions,
            }
        )

    def build_coverage_jobs(
        self,
        *,
        interpretation_responses: Mapping[str, Mapping[str, Any]],
        consolidation_responses: Mapping[str, Mapping[str, Any]],
    ) -> tuple[DiscoveryJsonJob, ...]:
        interpretations = self.validate_interpretation_responses(interpretation_responses)
        consolidation_jobs = self.build_consolidation_jobs(interpretation_responses)
        consolidations = self.validate_consolidation_responses(
            interpretation_responses=interpretation_responses,
            consolidation_responses=consolidation_responses,
        )
        consolidation_by_family = dict(zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, consolidation_jobs))
        jobs: list[DiscoveryJsonJob] = []
        for chunk in self.chunk_plan.chunks:
            interpret_job = self._interpret_job_by_chunk_id[chunk.chunk_id]
            consolidation_job = consolidation_by_family[chunk.source_family]
            evidence = tuple(
                self._atom_by_id[str(row["evidence_id"])].as_discovery_item()
                for row in chunk.evidence
            )
            scoped_consolidation = self._chunk_scoped_consolidation(
                chunk=chunk,
                interpret_job=interpret_job,
                interpretation_response=interpretations[interpret_job.job_id],
                consolidation_response=consolidations[consolidation_job.job_id],
            )
            jobs.append(
                self._create_job(
                    job_kind=COVERAGE_CRITIC_JOB,
                    scope=f"adaptive.{chunk.source_family}.coverage_{chunk.chunk_index:03d}",
                    dependencies=(interpret_job.job_id, consolidation_job.job_id),
                    messages=self._render_coverage_messages(
                        family=chunk.source_family,
                        evidence=evidence,
                        interpretation_response=interpretations[interpret_job.job_id],
                        consolidation_response=scoped_consolidation,
                    ),
                    input_bindings={
                        "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                        "exact_spent_authentication_sha256": (
                            self.exact_spent_authentication.authentication_sha256
                        ),
                        "catalog_sha256": self.catalog.catalog_sha256,
                        "chunk_id": chunk.chunk_id,
                        "source_family": chunk.source_family,
                        "interpretation_response_sha256": _sha(
                            interpretations[interpret_job.job_id]
                        ),
                        "consolidation_response_sha256": _sha(
                            consolidations[consolidation_job.job_id]
                        ),
                        "chunk_scoped_consolidation_sha256": _sha(scoped_consolidation),
                    },
                )
            )
        return tuple(jobs)

    def validate_coverage_responses(
        self,
        *,
        interpretation_responses: Mapping[str, Mapping[str, Any]],
        consolidation_responses: Mapping[str, Mapping[str, Any]],
        coverage_responses: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        coverage_jobs = self.build_coverage_jobs(
            interpretation_responses=interpretation_responses,
            consolidation_responses=consolidation_responses,
        )
        supplied = self._responses_for_jobs(
            jobs=coverage_jobs,
            responses=coverage_responses,
            label="coverage",
        )
        validated: dict[str, dict[str, Any]] = {}
        for chunk, job in zip(self.chunk_plan.chunks, coverage_jobs):
            evidence_ids = tuple(str(row["evidence_id"]) for row in chunk.evidence)
            request = json.loads(job.messages[1]["content"])
            canonical_names = tuple(
                str(concept["canonical_name"])
                for concept in request["family_consolidation"]["canonical_concepts"]
            )
            response = supplied[job.job_id]
            if isinstance(response.get("reviewed_evidence_ids"), list):
                validated[job.job_id] = revalidate_normalized_coverage_critic_response(
                    response,
                    evidence_ids=evidence_ids,
                    canonical_names=canonical_names,
                )
            else:
                validated[job.job_id] = validate_coverage_critic_response(
                    response,
                    evidence_ids=evidence_ids,
                    canonical_names=canonical_names,
                )
        return validated

    def validate_coverage_job_response(
        self,
        *,
        interpretation_responses: Mapping[str, Mapping[str, Any]],
        consolidation_responses: Mapping[str, Mapping[str, Any]],
        job: DiscoveryJsonJob,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate one chunk coverage result for authenticated caching."""

        jobs = self.build_coverage_jobs(
            interpretation_responses=interpretation_responses,
            consolidation_responses=consolidation_responses,
        )
        chunk_by_job = dict(zip((row.job_id for row in jobs), self.chunk_plan.chunks))
        chunk = chunk_by_job.get(job.job_id)
        if chunk is None or job.job_kind != COVERAGE_CRITIC_JOB:
            raise ValueError("coverage validator received an unknown adaptive job")
        request = json.loads(job.messages[1]["content"])
        canonical_names = tuple(
            str(concept["canonical_name"])
            for concept in request["family_consolidation"]["canonical_concepts"]
        )
        return validate_coverage_critic_response(
            response,
            evidence_ids=tuple(str(row["evidence_id"]) for row in chunk.evidence),
            canonical_names=canonical_names,
        )

    def compile_dossiers(
        self,
        *,
        interpretation_responses: Mapping[str, Mapping[str, Any]],
        consolidation_responses: Mapping[str, Mapping[str, Any]],
        coverage_responses: Mapping[str, Mapping[str, Any]],
    ) -> tuple[AdaptiveArchitectureDossier, ...]:
        consolidation_jobs = self.build_consolidation_jobs(interpretation_responses)
        coverage_jobs = self.build_coverage_jobs(
            interpretation_responses=interpretation_responses,
            consolidation_responses=consolidation_responses,
        )
        coverage = self.validate_coverage_responses(
            interpretation_responses=interpretation_responses,
            consolidation_responses=consolidation_responses,
            coverage_responses=coverage_responses,
        )
        actionable = [
            finding
            for job in coverage_jobs
            for finding in coverage[job.job_id]["findings"]
            if finding["action"] != "no_change"
        ]
        if actionable:
            raise AdaptiveCoverageRequiresRevision(actionable)
        candidates = self._consolidated_candidates(
            interpretation_responses=interpretation_responses,
            consolidation_responses=consolidation_responses,
        )
        consolidation_by_family = dict(zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, consolidation_jobs))
        dossiers: list[AdaptiveArchitectureDossier] = []
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
            family_chunks = tuple(
                chunk for chunk in self.chunk_plan.chunks if chunk.source_family == family
            )
            family_interpret_jobs = tuple(
                self._interpret_job_by_chunk_id[chunk.chunk_id] for chunk in family_chunks
            )
            family_coverage_jobs = tuple(
                job
                for chunk, job in zip(self.chunk_plan.chunks, coverage_jobs)
                if chunk.source_family == family
            )
            catalog_ids = tuple(
                atom.evidence_id for atom in self.catalog.atoms if atom.source_family == family
            )
            disposition_ids = tuple(
                evidence_id
                for job in family_coverage_jobs
                for evidence_id in coverage[job.job_id]["reviewed_evidence_ids"]
            )
            dossiers.append(
                AdaptiveArchitectureDossier.create(
                    source_family=family,
                    catalog_sha256=self.catalog.catalog_sha256,
                    catalog_evidence_ids=catalog_ids,
                    coverage_disposition_ids=disposition_ids,
                    architecture_candidates=candidates[family],
                    interpretation_job_ids=tuple(job.job_id for job in family_interpret_jobs),
                    consolidation_job_id=consolidation_by_family[family].job_id,
                    coverage_job_ids=tuple(job.job_id for job in family_coverage_jobs),
                )
            )
        return tuple(dossiers)

    def _execute_phased_chunk_coverage(
        self,
        *,
        interpretation_responses: Mapping[str, Mapping[str, Any]],
        family_consolidations: Mapping[str, AdaptiveFamilyConsolidation],
        run_job: Callable[
            [DiscoveryJsonJob, Callable[[Any], Mapping[str, Any]]],
            dict[str, Any],
        ],
    ) -> tuple[tuple[AdaptiveChunkCoverage, ...], tuple[dict[str, Any], ...]]:
        """Review every chunk losslessly, paging large name domains per evidence."""

        interpretations = self.validate_interpretation_responses(interpretation_responses)
        if set(family_consolidations) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("phased consolidations must cover all active architectures")
        coverages: list[AdaptiveChunkCoverage] = []
        compiler_records: list[dict[str, Any]] = []
        for chunk in self.chunk_plan.chunks:
            family = chunk.source_family
            consolidation = family_consolidations[family]
            if (
                not isinstance(consolidation, AdaptiveFamilyConsolidation)
                or consolidation.source_family != family
            ):
                raise ValueError("phased consolidation family binding changed")
            interpret_job = self._interpret_job_by_chunk_id[chunk.chunk_id]
            interpretation = interpretations[interpret_job.job_id]
            evidence = tuple(
                self._atom_by_id[str(row["evidence_id"])].as_discovery_item()
                for row in chunk.evidence
            )
            evidence_ids = tuple(item.evidence_id for item in evidence)
            scoped = self._chunk_scoped_consolidation(
                chunk=chunk,
                interpret_job=interpret_job,
                interpretation_response=interpretation,
                consolidation_response=consolidation.normalized_response,
            )
            canonical_names = tuple(
                str(row["canonical_name"]) for row in scoped["canonical_concepts"]
            )
            dependencies = tuple(
                dict.fromkeys((interpret_job.job_id, *consolidation.terminal_dependency_ids))
            )
            common_bindings = {
                "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                "exact_spent_authentication_sha256": (
                    self.exact_spent_authentication.authentication_sha256
                ),
                "catalog_sha256": self.catalog.catalog_sha256,
                "chunk_id": chunk.chunk_id,
                "source_family": family,
                "interpretation_response_sha256": _sha(interpretation),
                "family_consolidation_id": consolidation.consolidation_id,
                "consolidation_response_sha256": (consolidation.normalized_response_sha256),
                "chunk_scoped_consolidation_sha256": _sha(scoped),
            }
            findings: list[dict[str, Any]] = []
            coverage_job_ids: list[str] = []
            page_audits: list[dict[str, Any]] = []
            if len(canonical_names) <= HIERARCHICAL_DISCOVERY_MAX_FINDINGS_PER_ATOMIC_REVIEW:
                coverage_job = self._create_job(
                    job_kind=COVERAGE_CRITIC_JOB,
                    scope=f"adaptive.{family}.coverage_{chunk.chunk_index:03d}",
                    dependencies=dependencies,
                    messages=self._render_coverage_messages(
                        family=family,
                        evidence=evidence,
                        interpretation_response=interpretation,
                        consolidation_response=scoped,
                    ),
                    input_bindings={
                        **common_bindings,
                        "coverage_mode": "bounded_direct_chunk_v1",
                        "expected_reviewed_evidence_ids": list(evidence_ids),
                    },
                )
                normalized = run_job(
                    coverage_job,
                    lambda raw, evidence_ids=evidence_ids, canonical_names=canonical_names: (
                        validate_coverage_critic_response(
                            raw,
                            evidence_ids=evidence_ids,
                            canonical_names=canonical_names,
                        )
                    ),
                )
                findings.extend(normalized["findings"])
                coverage_job_ids.append(coverage_job.job_id)
                page_audits.append(
                    {
                        "coverage_job_id": coverage_job.job_id,
                        "evidence_ids": list(evidence_ids),
                        "canonical_names": list(canonical_names),
                        "normalized_response_sha256": _sha(normalized),
                    }
                )
                coverage_mode = "bounded_direct_chunk_v1"
            else:
                coverage_mode = "atomic_evidence_name_pages_v1"
                for evidence_item in evidence:
                    relevant_names = tuple(
                        str(concept["canonical_name"])
                        for concept in scoped["canonical_concepts"]
                        if evidence_item.evidence_id in concept["supporting_evidence_ids"]
                    )
                    name_pages = tuple(
                        relevant_names[
                            offset : offset + HIERARCHICAL_DISCOVERY_MAX_FINDINGS_PER_ATOMIC_REVIEW
                        ]
                        for offset in range(
                            0,
                            len(relevant_names),
                            HIERARCHICAL_DISCOVERY_MAX_FINDINGS_PER_ATOMIC_REVIEW,
                        )
                    ) or ((),)
                    for page_index, names in enumerate(name_pages):
                        name_set = set(names)
                        page_consolidation = {
                            "canonical_concepts": [
                                row
                                for row in scoped["canonical_concepts"]
                                if row["canonical_name"] in name_set
                            ],
                            "candidate_dispositions": [
                                row
                                for row in scoped["candidate_dispositions"]
                                if row["canonical_name"] in name_set
                            ],
                        }
                        atomic_job = self._create_job(
                            job_kind=COVERAGE_CRITIC_JOB,
                            scope=(
                                f"adaptive.{family}.coverage_{chunk.chunk_index:03d}."
                                f"{evidence_item.evidence_id}.page_{page_index:06d}"
                            ),
                            dependencies=dependencies,
                            messages=self._render_atomic_coverage_messages(
                                family=family,
                                evidence=evidence_item,
                                interpretation_response=interpretation,
                                consolidation_response=page_consolidation,
                                canonical_names=names,
                                page_index=page_index,
                            ),
                            input_bindings={
                                **common_bindings,
                                "coverage_mode": coverage_mode,
                                "evidence_id": evidence_item.evidence_id,
                                "canonical_name_page": list(names),
                                "canonical_name_page_index": page_index,
                                "canonical_name_page_count": len(name_pages),
                            },
                        )
                        normalized = run_job(
                            atomic_job,
                            lambda raw, evidence_id=evidence_item.evidence_id, names=names: (
                                _validate_atomic_coverage_response(
                                    raw,
                                    evidence_id=evidence_id,
                                    canonical_names=names,
                                )
                            ),
                        )
                        findings.extend(normalized["findings"])
                        coverage_job_ids.append(atomic_job.job_id)
                        page_audits.append(
                            {
                                "coverage_job_id": atomic_job.job_id,
                                "evidence_id": evidence_item.evidence_id,
                                "canonical_names": list(names),
                                "normalized_response_sha256": _sha(normalized),
                            }
                        )
            normalized_coverage = {
                "findings": findings,
                "reviewed_evidence_ids": list(evidence_ids),
            }
            compiler_audit = {
                "schema_version": ADAPTIVE_CHUNK_COVERAGE_VERSION,
                "coverage_mode": coverage_mode,
                "chunk_id": chunk.chunk_id,
                "source_family": family,
                "family_consolidation_id": consolidation.consolidation_id,
                "page_audits": page_audits,
                "reviewed_evidence_ids": list(evidence_ids),
                "all_pages_retained": True,
                "evidence_or_model_decision_truncation_applied": False,
            }
            artifact = AdaptiveChunkCoverage.create(
                source_family=family,
                chunk_id=chunk.chunk_id,
                evidence_ids=evidence_ids,
                coverage_job_ids=coverage_job_ids,
                normalized_response=normalized_coverage,
                compiler_audit=compiler_audit,
            )
            coverages.append(artifact)
            compiler_records.append(
                {
                    "record_type": "adaptive_chunk_coverage_compilation",
                    **artifact.compilation_record(),
                }
            )
        return tuple(coverages), tuple(compiler_records)

    def _compile_phased_dossiers(
        self,
        *,
        interpretation_responses: Mapping[str, Mapping[str, Any]],
        family_consolidations: Mapping[str, AdaptiveFamilyConsolidation],
        chunk_coverages: Sequence[AdaptiveChunkCoverage],
    ) -> tuple[AdaptiveArchitectureDossier, ...]:
        """Compile the existing public dossier type from authenticated phased artifacts."""

        interpretations = self.validate_interpretation_responses(interpretation_responses)
        coverage_values = tuple(chunk_coverages)
        if len(coverage_values) != len(self.chunk_plan.chunks):
            raise ValueError("phased coverage artifacts differ from the exact chunk plan")
        coverage_by_chunk = {item.chunk_id: item for item in coverage_values}
        if set(coverage_by_chunk) != {chunk.chunk_id for chunk in self.chunk_plan.chunks}:
            raise ValueError("phased coverage artifacts do not uniquely cover every chunk")
        actionable = [
            finding
            for coverage in coverage_values
            for finding in coverage.normalized_response["findings"]
            if finding["action"] != "no_change"
        ]
        if actionable:
            raise AdaptiveCoverageRequiresRevision(actionable)
        dossiers: list[AdaptiveArchitectureDossier] = []
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
            consolidation = family_consolidations[family]
            normalized = consolidation.normalized_response
            candidates: list[DiscoveryCandidate] = []
            for concept in normalized["canonical_concepts"]:
                identity = {
                    "schema_version": ADAPTIVE_HIERARCHY_VERSION,
                    "catalog_sha256": self.catalog.catalog_sha256,
                    "source_family": family,
                    "canonical_concept": concept,
                }
                candidates.append(
                    DiscoveryCandidate(
                        candidate_id=f"candidate_{_sha(identity)}",
                        feature_name=str(concept["canonical_name"]),
                        description=str(concept["description"]),
                        supporting_evidence_ids=tuple(concept["supporting_evidence_ids"]),
                        source_families=(family,),
                        value_shape_hypothesis=str(concept["value_shape_hypothesis"]),
                        unresolved_ambiguity=str(concept["unresolved_ambiguity"]),
                    )
                )
            family_chunks = tuple(
                chunk for chunk in self.chunk_plan.chunks if chunk.source_family == family
            )
            family_coverages = tuple(coverage_by_chunk[chunk.chunk_id] for chunk in family_chunks)
            interpretation_job_ids = tuple(
                self._interpret_job_by_chunk_id[chunk.chunk_id].job_id for chunk in family_chunks
            )
            if any(job_id not in interpretations for job_id in interpretation_job_ids):
                raise ValueError("phased dossier lost an interpretation dependency")
            catalog_ids = tuple(
                atom.evidence_id for atom in self.catalog.atoms if atom.source_family == family
            )
            disposition_ids = tuple(
                evidence_id
                for coverage in family_coverages
                for evidence_id in coverage.evidence_ids
            )
            coverage_job_ids = tuple(
                job_id for coverage in family_coverages for job_id in coverage.coverage_job_ids
            )
            dossiers.append(
                AdaptiveArchitectureDossier.create(
                    source_family=family,
                    catalog_sha256=self.catalog.catalog_sha256,
                    catalog_evidence_ids=catalog_ids,
                    coverage_disposition_ids=disposition_ids,
                    architecture_candidates=candidates,
                    interpretation_job_ids=interpretation_job_ids,
                    consolidation_job_id=consolidation.consolidation_id,
                    coverage_job_ids=coverage_job_ids,
                )
            )
        return tuple(dossiers)

    def _project_dossiers_for_atomic_evidence(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        evidence_id: str,
        candidate_id: str | None,
    ) -> list[dict[str, Any]]:
        """Return ten compact dossier shells with one exact support incidence."""

        ordered = self._validate_dossiers(dossiers)
        if evidence_id not in self._atom_by_id:
            raise ValueError("atomic dossier projection received unknown evidence")
        owning_family = self._atom_by_id[evidence_id].source_family
        projected: list[dict[str, Any]] = []
        observed_candidate = False
        for dossier in ordered:
            candidates: list[dict[str, Any]] = []
            if dossier.source_family == owning_family and candidate_id is not None:
                for candidate in dossier.architecture_candidates:
                    if candidate.candidate_id != candidate_id:
                        continue
                    if evidence_id not in candidate.supporting_evidence_ids:
                        raise ValueError(
                            "atomic dossier candidate does not own the scheduled evidence"
                        )
                    row = candidate.as_prompt_item()
                    row["supporting_evidence_ids"] = [evidence_id]
                    candidates.append(row)
                    observed_candidate = True
            projected.append(
                {
                    "source_family": dossier.source_family,
                    "coverage": {
                        "catalog_evidence_count": len(dossier.catalog_evidence_ids),
                        "coverage_disposition_count": len(dossier.coverage_disposition_ids),
                        "complete": True,
                        "lookback_evidence_ids": (
                            [evidence_id] if dossier.source_family == owning_family else []
                        ),
                    },
                    "architecture_candidates": candidates,
                }
            )
        if candidate_id is not None and not observed_candidate:
            raise ValueError("atomic dossier projection lost its scheduled candidate")
        return projected

    def _planner_page_schedule(
        self,
        dossiers: Sequence[AdaptiveArchitectureDossier],
    ) -> tuple[dict[str, Any], ...]:
        """Schedule every target/evidence/candidate/diagnostic page without sampling."""

        ordered = self._validate_dossiers(dossiers)
        targets = (*[item.feature_name for item in self.current_registry], NEW_MISSING_CONSTRUCT)
        diagnostics = tuple(item.diagnostic_id for item in self.diagnostics)
        rows: list[dict[str, Any]] = []
        ordinal = 0
        for target in targets:
            for atom in self.catalog.atoms:
                candidate_ids = tuple(
                    candidate.candidate_id
                    for dossier in ordered
                    if dossier.source_family == atom.source_family
                    for candidate in dossier.architecture_candidates
                    if atom.evidence_id in candidate.supporting_evidence_ids
                )
                candidate_pages: tuple[str | None, ...] = candidate_ids or (None,)
                for candidate_id in candidate_pages:
                    for diagnostic_id in diagnostics:
                        ordinal += 1
                        page_body = {
                            "target": target,
                            "evidence_id": atom.evidence_id,
                            "candidate_id": candidate_id,
                            "diagnostic_id": diagnostic_id,
                            "page_ordinal": ordinal,
                        }
                        rows.append(
                            {
                                **page_body,
                                "page_id": f"adaptive_planner_page_{_sha(page_body)}",
                            }
                        )
        return tuple(_clone(row) for row in rows)

    def _render_atomic_planner_messages(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        page: Mapping[str, Any],
    ) -> tuple[dict[str, str], ...]:
        target = str(page["target"])
        evidence_id = str(page["evidence_id"])
        candidate_id = page["candidate_id"]
        diagnostic_id = str(page["diagnostic_id"])
        registry = (
            []
            if target == NEW_MISSING_CONSTRUCT
            else [row for row in self._registry_prompt_items() if row["feature_name"] == target]
        )
        diagnostics = [
            row for row in self._diagnostic_prompt_items() if row["diagnostic_id"] == diagnostic_id
        ]
        if target != NEW_MISSING_CONSTRUCT and len(registry) != 1:
            raise ValueError("atomic planner target projection is not unique")
        if len(diagnostics) != 1:
            raise ValueError("atomic planner diagnostic projection is not unique")
        request = attach_hierarchical_discovery_response_contract(
            job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
            request={
                "job": "plan_adaptive_stage1_reconsideration",
                "architecture_dossiers": self._project_dossiers_for_atomic_evidence(
                    dossiers=dossiers,
                    evidence_id=evidence_id,
                    candidate_id=(str(candidate_id) if candidate_id is not None else None),
                ),
                "current_registry": registry,
                "diagnostics": diagnostics,
                "lookback_bounds": {
                    "max_ids_per_target": 1,
                    "max_total_ids": 1,
                    "max_total_bytes": self.config.max_total_lookback_bytes,
                },
            },
        )
        return (
            {"role": "system", "content": _ADAPTIVE_PLANNER_SYSTEM},
            {"role": "user", "content": canonical_json(request)},
        )

    def _execute_phased_adaptive_planner(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        run_job: Callable[[DiscoveryJsonJob, Callable[[Any], Mapping[str, Any]]], dict[str, Any]],
    ) -> tuple[dict[str, Any], ResolvedAdaptiveLookback, str, dict[str, Any]]:
        """Exhaustively page target/evidence decisions and recursively fold positives."""

        ordered = self._validate_dossiers(dossiers)
        schedule = self._planner_page_schedule(ordered)
        if not schedule:
            raise AssertionError("adaptive planner schedule cannot be empty")
        registry_names = tuple(item.feature_name for item in self.current_registry)
        dossier_dependencies = tuple(
            dict.fromkeys(job_id for dossier in ordered for job_id in dossier.coverage_job_ids)
        )
        page_records: list[dict[str, Any]] = []
        positive_by_target: dict[str, list[tuple[DiscoveryCandidate, str, dict[str, Any]]]] = {
            target: [] for target in (*registry_names, NEW_MISSING_CONSTRUCT)
        }
        for page in schedule:
            target = str(page["target"])
            evidence_id = str(page["evidence_id"])
            owning_family = self._atom_by_id[evidence_id].source_family
            job = self._create_job(
                job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
                scope=f"adaptive.cross_family.planner_page_{int(page['page_ordinal']):06d}",
                dependencies=dossier_dependencies,
                messages=self._render_atomic_planner_messages(
                    dossiers=ordered,
                    page=page,
                ),
                input_bindings={
                    "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                    "planner_interface_version": ADAPTIVE_PLANNER_INTERFACE_VERSION,
                    "exact_spent_authentication_sha256": (
                        self.exact_spent_authentication.authentication_sha256
                    ),
                    "catalog_sha256": self.catalog.catalog_sha256,
                    "chunk_plan_sha256": self.chunk_plan.plan_sha256,
                    "dossier_sha256s": [item.dossier_sha256 for item in ordered],
                    "planner_page": _clone(page),
                    "current_registry_sha256": _sha(self._registry_private_items()),
                    "diagnostics_sha256": _sha(self._diagnostic_prompt_items()),
                    "semantic_truncation_applied": False,
                },
            )
            response = run_job(
                job,
                lambda raw, target=target, evidence_id=evidence_id, owning_family=owning_family: (
                    _validate_atomic_adaptive_planner_page_response(
                        raw,
                        target=target,
                        evidence_id=evidence_id,
                        owning_family=owning_family,
                        registry_names=registry_names,
                    )
                ),
            )
            page_record = {
                **_clone(page),
                "job_id": job.job_id,
                "normalized_response": _clone(response),
                "normalized_response_sha256": _sha(response),
                "disposition": (
                    "review_requested" if response["review_targets"] else "no_review_requested"
                ),
            }
            page_records.append(page_record)
            if not response["review_targets"]:
                continue
            row = response["review_targets"][0]
            candidate_id = f"adaptive_planner_signal_{_sha({'job_id': job.job_id, 'row': row})}"
            registry_shape = next(
                (
                    item.value_shape_hypothesis
                    for item in self.current_registry
                    if item.feature_name == target
                ),
                "ambiguous",
            )
            positive_by_target[target].append(
                (
                    DiscoveryCandidate(
                        candidate_id=candidate_id,
                        feature_name=target,
                        description=str(row["problem"]),
                        supporting_evidence_ids=(evidence_id,),
                        source_families=tuple(row["relevant_architectures"]),
                        value_shape_hypothesis=registry_shape,
                        unresolved_ambiguity=str(row["reason"]),
                    ),
                    job.job_id,
                    _clone(row),
                )
            )

        compiled_targets: list[dict[str, Any]] = []
        target_records: list[dict[str, Any]] = []
        fold_job_ids: list[str] = []
        terminal_ids: list[str] = []
        catalog_order = {atom.evidence_id: index for index, atom in enumerate(self.catalog.atoms)}
        for target in (*registry_names, NEW_MISSING_CONSTRUCT):
            signals = positive_by_target[target]
            if not signals:
                target_records.append(
                    {
                        "target": target,
                        "positive_page_count": 0,
                        "signal_candidate_ids": [],
                        "folds": [],
                        "disposition": "no_review_requested_on_any_page",
                    }
                )
                continue
            candidates = tuple(item[0] for item in signals)
            if len(candidates) == 1:
                definition = {
                    "canonical_name": target,
                    "description": candidates[0].description,
                    "unresolved_ambiguity": candidates[0].unresolved_ambiguity,
                    "reason": signals[0][2]["reason"],
                }
                folds: list[dict[str, Any]] = []
                terminal_id = signals[0][1]
            else:
                group_id = f"adaptive_planner_target_{_sha({'target': target, 'signals': [item.candidate_id for item in candidates]})}"
                prior: dict[str, Any] | None = None
                prior_job_id: str | None = None
                folds = []
                by_id = {candidate.candidate_id: candidate for candidate in candidates}
                for fold in candidate_definition_fold_batches(
                    group_id=group_id,
                    member_candidate_ids=tuple(by_id),
                ):
                    fold_index = int(fold["fold_index"])
                    fresh = tuple(by_id[str(value)] for value in fold["member_candidate_ids"])
                    dependencies = (
                        (prior_job_id,)
                        if prior_job_id is not None
                        else tuple(item[1] for item in signals)
                    )
                    fold_job = self._create_job(
                        job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
                        scope=(
                            f"adaptive.cross_family.planner_target_{_sha(target)[:12]}"
                            f"_fold_{fold_index:06d}"
                        ),
                        dependencies=dependencies,
                        messages=_render_candidate_definition_fold_messages(
                            job="fold_cross_architecture_group_definition",
                            job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
                            group_id=group_id,
                            fold_index=fold_index,
                            candidates=fresh,
                            prior_accumulator=prior,
                        ),
                        input_bindings={
                            "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                            "planner_interface_version": ADAPTIVE_PLANNER_INTERFACE_VERSION,
                            "target": target,
                            "target_signal_candidate_ids_sha256": _sha(list(by_id)),
                            "fold": fold,
                            "prior_accumulator_sha256": (
                                _sha(prior) if prior is not None else None
                            ),
                            "semantic_truncation_applied": False,
                        },
                    )
                    prior = run_job(fold_job, _validate_candidate_definition_fold_response)
                    prior_job_id = fold_job.job_id
                    fold_job_ids.append(fold_job.job_id)
                    folds.append(
                        {
                            "fold": _clone(fold),
                            "job_id": fold_job.job_id,
                            "normalized_response": _clone(prior),
                            "normalized_response_sha256": _sha(prior),
                        }
                    )
                if prior is None or prior_job_id is None:
                    raise AssertionError("adaptive planner target fold did not terminate")
                definition = prior
                terminal_id = prior_job_id
            terminal_ids.append(terminal_id)
            requested = tuple(
                sorted(
                    {
                        evidence_id
                        for _candidate, _job_id, row in signals
                        for evidence_id in row["requested_evidence_ids"]
                    },
                    key=catalog_order.__getitem__,
                )
            )
            families = tuple(
                family
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
                if any(family in row["relevant_architectures"] for _, _, row in signals)
            )
            compiled_targets.append(
                {
                    "target": target,
                    "problem": str(definition["description"]),
                    "relevant_architectures": list(families),
                    "requested_evidence_ids": list(requested),
                    "reason": str(definition["reason"]),
                }
            )
            target_records.append(
                {
                    "target": target,
                    "positive_page_count": len(signals),
                    "signal_candidate_ids": [item.candidate_id for item in candidates],
                    "folds": folds,
                    "terminal_dependency_id": terminal_id,
                    "requested_evidence_ids": list(requested),
                    "disposition": "compiled_review_target",
                }
            )
        retained_requested = tuple(
            dict.fromkeys(
                evidence_id
                for target in compiled_targets
                for evidence_id in target["requested_evidence_ids"]
            )
        )
        planner_audit_body = {
            "audit_version": _ADAPTIVE_PHASED_PLANNER_COMPILER_VERSION,
            "schedule_sha256": _sha(list(schedule)),
            "expected_page_count": len(schedule),
            "page_records": page_records,
            "target_records": target_records,
            "retained_requested_evidence_ids": list(retained_requested),
            "all_targets_evidence_candidates_and_diagnostics_paged": True,
            "every_page_has_explicit_disposition": True,
            "target_or_evidence_truncation_applied": False,
            "recursive_target_folds_terminated": True,
        }
        planner_id = f"adaptive_planner_compilation_{_sha(planner_audit_body)}"
        planner = {
            "review_targets": compiled_targets,
            "no_lookback_needed": not retained_requested,
            "wire_normalization_audit": {
                "audit_version": _ADAPTIVE_PHASED_PLANNER_COMPILER_VERSION,
                "planner_compilation_id": planner_id,
                "review_targets_sha256": _sha(compiled_targets),
                "retained_requested_evidence_ids": list(retained_requested),
                "page_records_sha256": _sha(page_records),
                "target_records_sha256": _sha(target_records),
                "expected_page_count": len(schedule),
                "target_or_evidence_truncation_applied": False,
            },
        }
        lookback_items = [
            self._atom_by_id[evidence_id].as_discovery_item().as_prompt_item()
            for evidence_id in retained_requested
        ]
        lookback = ResolvedAdaptiveLookback(
            requested_evidence_ids=retained_requested,
            canonical_size_bytes=len(canonical_json(lookback_items).encode("utf-8")),
            lookback_sha256=_sha(lookback_items),
            total_catalog_atom_count=len(self.catalog.atoms),
            _items_json=canonical_json(lookback_items),
        )
        compiler_record = {
            "record_type": "adaptive_phased_planner_compilation",
            "schema_version": _ADAPTIVE_PHASED_PLANNER_COMPILER_VERSION,
            "planner_compilation_id": planner_id,
            "planner_response_sha256": _sha(planner),
            "lookback_sha256": lookback.lookback_sha256,
            "page_job_ids": [row["job_id"] for row in page_records],
            "fold_job_ids": fold_job_ids,
            "terminal_dependency_ids": terminal_ids,
            **planner_audit_body,
        }
        return _clone(planner), lookback, planner_id, _clone(compiler_record)

    def _render_atomic_proposer_messages(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        planner_response: Mapping[str, Any],
        target_scope: Sequence[str],
        evidence_id: str | None,
        candidate_id: str | None,
        diagnostic_id: str,
    ) -> tuple[dict[str, str], ...]:
        ordered = self._validate_dossiers(dossiers)
        target_set = set(target_scope)
        planner_rows = [
            row for row in planner_response["review_targets"] if row["target"] in target_set
        ]
        if {row["target"] for row in planner_rows} != target_set:
            raise ValueError("atomic proposer target scope differs from the phased planner")
        review_rows: list[dict[str, Any]] = []
        for row in planner_rows:
            review_rows.append(
                {
                    **_clone(row),
                    "requested_evidence_ids": (
                        [evidence_id]
                        if evidence_id is not None and evidence_id in row["requested_evidence_ids"]
                        else []
                    ),
                }
            )
        if evidence_id is None:
            dossier_projection = []
            for dossier in ordered:
                dossier_projection.append(
                    {
                        "source_family": dossier.source_family,
                        "coverage": {
                            "catalog_evidence_count": len(dossier.catalog_evidence_ids),
                            "coverage_disposition_count": len(dossier.coverage_disposition_ids),
                            "complete": True,
                            "lookback_evidence_ids": [],
                        },
                        "architecture_candidates": [],
                    }
                )
            requested_evidence: list[dict[str, Any]] = []
        else:
            dossier_projection = self._project_dossiers_for_atomic_evidence(
                dossiers=ordered,
                evidence_id=evidence_id,
                candidate_id=candidate_id,
            )
            requested_evidence = [
                self._atom_by_id[evidence_id].as_discovery_item().as_prompt_item()
            ]
        registry = [
            row for row in self._registry_prompt_items() if row["feature_name"] in target_set
        ]
        diagnostics = [
            row for row in self._diagnostic_prompt_items() if row["diagnostic_id"] == diagnostic_id
        ]
        if len(diagnostics) != 1:
            raise ValueError("atomic proposer diagnostic projection is not unique")
        request = attach_hierarchical_discovery_response_contract(
            job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
            request={
                "job": "propose_adaptive_registry_revision",
                "architecture_dossiers": dossier_projection,
                "current_registry": registry,
                "diagnostics": diagnostics,
                "review_plan": {
                    "review_targets": review_rows,
                    "no_lookback_needed": evidence_id is None,
                },
                "requested_evidence": requested_evidence,
                "maximum_operations": 1,
            },
        )
        return (
            {"role": "system", "content": _ADAPTIVE_PROPOSER_SYSTEM},
            {"role": "user", "content": canonical_json(request)},
        )

    def _proposer_page_schedule(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        planner_response: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        """Schedule singleton and merge-pair proposal pages over every requested atom."""

        ordered = self._validate_dossiers(dossiers)
        planner = self._validate_phased_planner_response(
            dossiers=ordered,
            response=planner_response,
        )
        rows_by_target = {row["target"]: row for row in planner["review_targets"]}
        target_scopes: list[tuple[str, ...]] = [(target,) for target in rows_by_target]
        existing = tuple(
            item.feature_name
            for item in self.current_registry
            if item.feature_name in rows_by_target
        )
        for left_index, left in enumerate(existing):
            for right in existing[left_index + 1 :]:
                target_scopes.append((left, right))
        diagnostics = tuple(item.diagnostic_id for item in self.diagnostics)
        rows: list[dict[str, Any]] = []
        ordinal = 0
        for target_scope in target_scopes:
            evidence_ids = tuple(
                dict.fromkeys(
                    evidence_id
                    for target in target_scope
                    for evidence_id in rows_by_target[target]["requested_evidence_ids"]
                )
            )
            evidence_pages: tuple[str | None, ...] = evidence_ids or (None,)
            for evidence_id in evidence_pages:
                if evidence_id is None:
                    candidate_pages: tuple[str | None, ...] = (None,)
                else:
                    candidate_ids = tuple(
                        candidate.candidate_id
                        for dossier in ordered
                        for candidate in dossier.architecture_candidates
                        if evidence_id in candidate.supporting_evidence_ids
                    )
                    candidate_pages = candidate_ids or (None,)
                for candidate_id in candidate_pages:
                    for diagnostic_id in diagnostics:
                        ordinal += 1
                        page_body = {
                            "target_scope": list(target_scope),
                            "evidence_id": evidence_id,
                            "candidate_id": candidate_id,
                            "diagnostic_id": diagnostic_id,
                            "page_ordinal": ordinal,
                        }
                        rows.append(
                            {
                                **page_body,
                                "page_id": f"adaptive_proposer_page_{_sha(page_body)}",
                            }
                        )
        return tuple(_clone(row) for row in rows)

    def _execute_phased_adaptive_proposer(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        planner_response: Mapping[str, Any],
        lookback: ResolvedAdaptiveLookback,
        planner_compilation_id: str,
        planner_dependency_ids: Sequence[str],
        run_job: Callable[[DiscoveryJsonJob, Callable[[Any], Mapping[str, Any]]], dict[str, Any]],
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Judge every bounded proposal, group it losslessly, then apply explicit capacity."""

        ordered = self._validate_dossiers(dossiers)
        planner = self._validate_phased_planner_response(
            dossiers=ordered,
            response=planner_response,
        )
        expected_lookback = self.resolve_requested_evidence(
            dossiers=ordered,
            planner_response=planner,
        )
        if expected_lookback.lookback_sha256 != lookback.lookback_sha256:
            raise ValueError("phased proposer received another exact planner lookback")
        planner_dependencies = tuple(dict.fromkeys(planner_dependency_ids))
        schedule = self._proposer_page_schedule(
            dossiers=ordered,
            planner_response=planner,
        )
        page_records: list[dict[str, Any]] = []
        raw_proposals: list[dict[str, Any]] = []
        diagnostic_ids = tuple(item.diagnostic_id for item in self.diagnostics)
        for page in schedule:
            target_scope = tuple(str(value) for value in page["target_scope"])
            evidence_id = page["evidence_id"]
            requested_ids = () if evidence_id is None else (str(evidence_id),)
            diagnostic_id = str(page["diagnostic_id"])
            job = self._create_job(
                job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
                scope=f"adaptive.cross_family.proposer_page_{int(page['page_ordinal']):06d}",
                dependencies=planner_dependencies,
                messages=self._render_atomic_proposer_messages(
                    dossiers=ordered,
                    planner_response=planner,
                    target_scope=target_scope,
                    evidence_id=(str(evidence_id) if evidence_id is not None else None),
                    candidate_id=(
                        str(page["candidate_id"]) if page["candidate_id"] is not None else None
                    ),
                    diagnostic_id=diagnostic_id,
                ),
                input_bindings={
                    "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                    "proposer_interface_version": ADAPTIVE_PROPOSER_INTERFACE_VERSION,
                    "planner_compilation_id": planner_compilation_id,
                    "planner_response_sha256": _sha(planner),
                    "lookback_sha256": lookback.lookback_sha256,
                    "proposer_page": _clone(page),
                    "maximum_operations_on_page": 1,
                    "semantic_truncation_applied": False,
                },
            )
            response = run_job(
                job,
                lambda raw, target_scope=target_scope, requested_ids=requested_ids, diagnostic_id=diagnostic_id: (
                    _validate_atomic_adaptive_proposer_page_response(
                        raw,
                        planned_targets=target_scope,
                        requested_evidence_ids=requested_ids,
                        diagnostic_ids=(diagnostic_id,),
                    )
                ),
            )
            page_record = {
                **_clone(page),
                "job_id": job.job_id,
                "normalized_response": _clone(response),
                "normalized_response_sha256": _sha(response),
                "disposition": (
                    "proposal_emitted" if response["operations"] else "no_revision_proposed"
                ),
            }
            page_records.append(page_record)
            for operation_index, operation in enumerate(response["operations"]):
                proposal_body = {
                    "page_job_id": job.job_id,
                    "operation_index": operation_index,
                    "operation": operation,
                }
                raw_proposals.append(
                    {
                        "proposal_id": f"adaptive_revision_proposal_{_sha(proposal_body)}",
                        "page_ordinal": int(page["page_ordinal"]),
                        "page_job_id": job.job_id,
                        "operation": _clone(operation),
                    }
                )

        proposal_judgments: list[dict[str, Any]] = []
        accepted: list[dict[str, Any]] = []
        for proposal in raw_proposals:
            operation = proposal["operation"]
            support = tuple(operation["supporting_evidence_ids"])
            evidence = [
                self._atom_by_id[evidence_id].as_discovery_item().as_prompt_item()
                for evidence_id in support
            ]
            judgment_job = self._create_job(
                job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
                scope=f"adaptive.cross_family.judge_{_sha(proposal['proposal_id'])[:16]}",
                dependencies=(proposal["page_job_id"],),
                messages=_render_adaptive_proposal_judgment_messages(
                    proposal_id=proposal["proposal_id"],
                    proposal=operation,
                    requested_raw_evidence_lookback=evidence,
                ),
                input_bindings={
                    "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                    "proposer_interface_version": ADAPTIVE_PROPOSER_INTERFACE_VERSION,
                    "planner_compilation_id": planner_compilation_id,
                    "proposal_id": proposal["proposal_id"],
                    "proposal_sha256": _sha(operation),
                    "complete_supporting_evidence_ids_sha256": _sha(list(support)),
                    "semantic_truncation_applied": False,
                },
            )
            judgment = run_job(judgment_job, _validate_adaptive_proposal_judgment)
            record = {
                "proposal_id": proposal["proposal_id"],
                "page_ordinal": proposal["page_ordinal"],
                "page_job_id": proposal["page_job_id"],
                "judgment_job_id": judgment_job.job_id,
                "operation": _clone(operation),
                "judgment": _clone(judgment),
                "disposition": (
                    "accepted_for_relation_compilation"
                    if judgment["decision"] == "accept"
                    else "model_rejected"
                ),
            }
            proposal_judgments.append(record)
            if judgment["decision"] == "accept":
                accepted.append(record)

        catalog_order = {atom.evidence_id: index for index, atom in enumerate(self.catalog.atoms)}
        diagnostic_order = {
            diagnostic.diagnostic_id: index for index, diagnostic in enumerate(self.diagnostics)
        }
        compiled_rows: list[tuple[int, dict[str, Any], str, tuple[str, ...]]] = []
        proposal_dispositions: dict[str, dict[str, Any]] = {
            record["proposal_id"]: {
                "proposal_id": record["proposal_id"],
                "disposition": record["disposition"],
                "reason": record["judgment"]["reason"],
            }
            for record in proposal_judgments
        }

        drop_groups: dict[str, list[dict[str, Any]]] = {}
        non_drop_partitions: dict[str, list[dict[str, Any]]] = {}
        for record in accepted:
            operation = record["operation"]
            if operation["operation"] == "drop":
                drop_groups.setdefault(str(operation["targets"][0]), []).append(record)
                continue
            partition_key = canonical_json(
                {
                    "operation": operation["operation"],
                    "targets": operation["targets"],
                }
            )
            non_drop_partitions.setdefault(partition_key, []).append(record)

        for target, records in drop_groups.items():
            compiled_group_id = f"adaptive_revision_group_{_sha({'operation': 'drop', 'target': target, 'proposals': [row['proposal_id'] for row in records]})}"
            diagnostic_values = tuple(
                sorted(
                    {
                        diagnostic_id
                        for row in records
                        for diagnostic_id in row["operation"]["diagnostic_ids"]
                    },
                    key=diagnostic_order.__getitem__,
                )
            )
            compiled_rows.append(
                (
                    min(int(row["page_ordinal"]) for row in records),
                    {
                        "operation": "drop",
                        "targets": [target],
                        "proposed_feature": {},
                        "supporting_evidence_ids": [],
                        "diagnostic_ids": list(diagnostic_values),
                        "reason": str(records[0]["judgment"]["reason"]),
                    },
                    compiled_group_id,
                    tuple(row["proposal_id"] for row in records),
                )
            )
            for record in records:
                proposal_dispositions[record["proposal_id"]] = {
                    "proposal_id": record["proposal_id"],
                    "disposition": "accepted_and_compiled",
                    "compiled_group_id": compiled_group_id,
                    "reason": "equivalent drop proposals coalesced after every judgment",
                }

        relation_records: list[dict[str, Any]] = []
        definition_records: list[dict[str, Any]] = []
        for partition_index, records in enumerate(non_drop_partitions.values(), start=1):
            candidates: list[DiscoveryCandidate] = []
            record_by_candidate: dict[str, dict[str, Any]] = {}
            for record in records:
                operation = record["operation"]
                judgment = record["judgment"]
                support = tuple(operation["supporting_evidence_ids"])
                families = tuple(
                    family
                    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
                    if any(
                        self._atom_by_id[evidence_id].source_family == family
                        for evidence_id in support
                    )
                )
                candidate = DiscoveryCandidate(
                    candidate_id=record["proposal_id"],
                    feature_name=str(judgment["canonical_name"]),
                    description=str(judgment["description"]),
                    supporting_evidence_ids=support,
                    source_families=families,
                    value_shape_hypothesis=str(
                        operation["proposed_feature"]["value_shape_hypothesis"]
                    ),
                    unresolved_ambiguity=str(judgment["unresolved_ambiguity"]),
                )
                candidates.append(candidate)
                record_by_candidate[candidate.candidate_id] = record
            relation_pages = bounded_candidate_relation_pages(candidates)
            relation_responses: dict[str, dict[str, Any]] = {}
            candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
            for page_index, page in enumerate(relation_pages, start=1):
                anchor = candidate_by_id[str(page["anchor_candidate_id"])]
                peers = tuple(
                    candidate_by_id[str(peer_id)] for peer_id in page["peer_candidate_ids"]
                )
                relation_job = self._create_job(
                    job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
                    scope=(
                        f"adaptive.cross_family.revision_partition_{partition_index:06d}"
                        f"_relation_{page_index:06d}"
                    ),
                    dependencies=tuple(
                        record_by_candidate[candidate.candidate_id]["judgment_job_id"]
                        for candidate in (anchor, *peers)
                    ),
                    messages=_render_candidate_relation_page_messages(
                        job="compare_cross_architecture_candidate_relations",
                        job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
                        source_family=None,
                        anchor=anchor,
                        peers=peers,
                    ),
                    input_bindings={
                        "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                        "planner_compilation_id": planner_compilation_id,
                        "revision_partition_index": partition_index,
                        "relation_page": page,
                        "semantic_truncation_applied": False,
                    },
                )
                normalized_relation = run_job(
                    relation_job,
                    lambda raw, anchor=anchor, peers=peers: (
                        validate_candidate_relation_page_response(
                            raw,
                            anchor=anchor,
                            peers=peers,
                        )
                    ),
                )
                relation_responses[relation_job.job_id] = normalized_relation
                relation_records.append(
                    {
                        "partition_index": partition_index,
                        "page": _clone(page),
                        "job_id": relation_job.job_id,
                        "normalized_response": _clone(normalized_relation),
                    }
                )
            grouped = compile_complete_link_candidate_groups(
                candidates=candidates,
                page_responses=relation_responses,
            )
            for group_index, group in enumerate(grouped["groups"], start=1):
                member_ids = tuple(str(value) for value in group["member_candidate_ids"])
                members = tuple(candidate_by_id[value] for value in member_ids)
                group_id = f"adaptive_revision_group_{_sha({'partition': partition_index, 'members': list(member_ids)})}"
                if len(members) == 1:
                    definition = {
                        "canonical_name": members[0].feature_name,
                        "description": members[0].description,
                        "unresolved_ambiguity": members[0].unresolved_ambiguity,
                        "reason": record_by_candidate[members[0].candidate_id]["judgment"][
                            "reason"
                        ],
                    }
                    folds: list[dict[str, Any]] = []
                else:
                    prior: dict[str, Any] | None = None
                    prior_job_id: str | None = None
                    folds = []
                    for fold in candidate_definition_fold_batches(
                        group_id=group_id,
                        member_candidate_ids=member_ids,
                    ):
                        fold_index = int(fold["fold_index"])
                        fresh = tuple(
                            candidate_by_id[str(value)] for value in fold["member_candidate_ids"]
                        )
                        dependencies = (
                            (prior_job_id,)
                            if prior_job_id is not None
                            else tuple(
                                record_by_candidate[value]["judgment_job_id"]
                                for value in member_ids
                            )
                        )
                        fold_job = self._create_job(
                            job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
                            scope=(
                                f"adaptive.cross_family.revision_partition_{partition_index:06d}"
                                f"_group_{group_index:06d}_fold_{fold_index:06d}"
                            ),
                            dependencies=dependencies,
                            messages=_render_candidate_definition_fold_messages(
                                job="fold_cross_architecture_group_definition",
                                job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
                                group_id=group_id,
                                fold_index=fold_index,
                                candidates=fresh,
                                prior_accumulator=prior,
                            ),
                            input_bindings={
                                "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                                "planner_compilation_id": planner_compilation_id,
                                "revision_group_id": group_id,
                                "fold": fold,
                                "prior_accumulator_sha256": (
                                    _sha(prior) if prior is not None else None
                                ),
                                "semantic_truncation_applied": False,
                            },
                        )
                        prior = run_job(
                            fold_job,
                            _validate_candidate_definition_fold_response,
                        )
                        prior_job_id = fold_job.job_id
                        folds.append(
                            {
                                "fold": _clone(fold),
                                "job_id": fold_job.job_id,
                                "normalized_response": _clone(prior),
                            }
                        )
                    if prior is None:
                        raise AssertionError("adaptive revision definition fold did not terminate")
                    definition = prior
                support = tuple(
                    sorted(
                        {
                            evidence_id
                            for member in members
                            for evidence_id in member.supporting_evidence_ids
                        },
                        key=catalog_order.__getitem__,
                    )
                )
                diagnostics = tuple(
                    sorted(
                        {
                            diagnostic_id
                            for member_id in member_ids
                            for diagnostic_id in record_by_candidate[member_id]["operation"][
                                "diagnostic_ids"
                            ]
                        },
                        key=diagnostic_order.__getitem__,
                    )
                )
                families = tuple(
                    family
                    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
                    if any(
                        self._atom_by_id[evidence_id].source_family == family
                        for evidence_id in support
                    )
                )
                shapes = {member.value_shape_hypothesis for member in members}
                operation_kind = str(record_by_candidate[member_ids[0]]["operation"]["operation"])
                operation_targets = list(record_by_candidate[member_ids[0]]["operation"]["targets"])
                if operation_kind == "add":
                    operation_targets = [str(definition["canonical_name"])]
                compiled_operation = {
                    "operation": operation_kind,
                    "targets": operation_targets,
                    "proposed_feature": {
                        "feature_name": str(definition["canonical_name"]),
                        "description": str(definition["description"]),
                        "value_shape_hypothesis": (
                            next(iter(shapes)) if len(shapes) == 1 else "ambiguous"
                        ),
                        "definition_summary": str(definition["reason"]),
                        "source_families": list(families),
                    },
                    "supporting_evidence_ids": list(support),
                    "diagnostic_ids": list(diagnostics),
                    "reason": str(definition["reason"]),
                }
                compiled_rows.append(
                    (
                        min(
                            int(record_by_candidate[value]["page_ordinal"]) for value in member_ids
                        ),
                        compiled_operation,
                        group_id,
                        member_ids,
                    )
                )
                definition_records.append(
                    {
                        "partition_index": partition_index,
                        "group_id": group_id,
                        "member_proposal_ids": list(member_ids),
                        "definition": _clone(definition),
                        "folds": folds,
                    }
                )
                for proposal_id in member_ids:
                    proposal_dispositions[proposal_id] = {
                        "proposal_id": proposal_id,
                        "disposition": "accepted_and_compiled",
                        "compiled_group_id": group_id,
                        "reason": "complete-link revision group compiled after explicit judgment",
                    }

        compiled_rows.sort(key=lambda item: (item[0], item[2]))
        compiled_operations = [row[1] for row in compiled_rows]
        normalized_proposal = self._normalize_proposer_wire_response(
            dossiers=ordered,
            planner_response=planner,
            lookback=lookback,
            response={
                "operations": compiled_operations,
                "converged": not compiled_operations,
            },
        )
        dropped_compiled = {
            int(row["operation_index"]): str(row["reason"])
            for row in normalized_proposal["wire_normalization_audit"]["dropped_operation_slots"]
        }
        for operation_index, (_ordinal, _operation, group_id, member_ids) in enumerate(
            compiled_rows
        ):
            if operation_index in dropped_compiled:
                disposition = "rejected_after_exhaustive_compilation"
                reason = dropped_compiled[operation_index]
            else:
                disposition = "accepted_into_bounded_final_operation"
                reason = "survived complete validation, conflict resolution, and round capacity"
            for proposal_id in member_ids:
                proposal_dispositions[proposal_id] = {
                    "proposal_id": proposal_id,
                    "disposition": disposition,
                    "compiled_group_id": group_id,
                    "reason": reason,
                }
        ordered_dispositions = [
            proposal_dispositions[proposal["proposal_id"]] for proposal in raw_proposals
        ]
        compiler_body = {
            "schema_version": _ADAPTIVE_PHASED_PROPOSER_COMPILER_VERSION,
            "planner_compilation_id": planner_compilation_id,
            "page_schedule_sha256": _sha(list(schedule)),
            "expected_page_count": len(schedule),
            "page_records": page_records,
            "raw_proposals": raw_proposals,
            "proposal_judgments": proposal_judgments,
            "proposal_dispositions": ordered_dispositions,
            "relation_records": relation_records,
            "definition_records": definition_records,
            "all_job_ids": [
                *[row["job_id"] for row in page_records],
                *[row["judgment_job_id"] for row in proposal_judgments],
                *[row["job_id"] for row in relation_records],
                *[fold["job_id"] for record in definition_records for fold in record["folds"]],
            ],
            "compiled_operations_before_global_dispositions": compiled_operations,
            "normalized_proposal_sha256": _sha(normalized_proposal),
            "every_page_and_proposal_has_an_explicit_disposition": True,
            "operation_slice_or_semantic_truncation_applied": False,
            "final_capacity_applied_only_after_exhaustive_validation": True,
        }
        proposer_id = f"adaptive_proposer_compilation_{_sha(compiler_body)}"
        compiler_record = {
            "record_type": "adaptive_phased_proposer_compilation",
            "proposer_compilation_id": proposer_id,
            **compiler_body,
        }
        return _clone(normalized_proposal), proposer_id, _clone(compiler_record)

    def _freeze_phased_round(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        planner_response: Mapping[str, Any],
        planner_compilation_id: str,
        planner_compiler_record: Mapping[str, Any],
        lookback: ResolvedAdaptiveLookback,
        proposer_response: Mapping[str, Any],
        proposer_compilation_id: str,
        proposer_compiler_record: Mapping[str, Any],
    ) -> FrozenAdaptiveReconsiderationRound:
        ordered = self._validate_dossiers(dossiers)
        planner = self._validate_phased_planner_response(
            dossiers=ordered,
            response=planner_response,
        )
        expected_lookback = self.resolve_requested_evidence(
            dossiers=ordered,
            planner_response=planner,
        )
        if expected_lookback.lookback_sha256 != lookback.lookback_sha256:
            raise ValueError("phased freeze received another exact lookback")
        proposal = self.validate_proposer_response(
            dossiers=ordered,
            planner_response=planner,
            lookback=lookback,
            response=proposer_response,
        )
        _require_identifier(planner_compilation_id, label="planner_compilation_id")
        _require_identifier(proposer_compilation_id, label="proposer_compilation_id")
        if planner_compiler_record.get("planner_compilation_id") != planner_compilation_id:
            raise ValueError("phased planner compiler record changed its identity")
        if proposer_compiler_record.get("proposer_compilation_id") != proposer_compilation_id:
            raise ValueError("phased proposer compiler record changed its identity")
        audit = {
            "schema_version": ADAPTIVE_ROUND_FREEZE_VERSION,
            "fresh_exact_spent_catalog_bound": True,
            "all_ten_architectures_completed_separately": True,
            "planner_compilation_id": planner_compilation_id,
            "planner_compiler_record_sha256": _sha(planner_compiler_record),
            "planner_page_count": planner_compiler_record["expected_page_count"],
            "planner_target_or_evidence_truncation_applied": False,
            "lookback": lookback.audit(),
            "proposer_compilation_id": proposer_compilation_id,
            "proposer_compiler_record_sha256": _sha(proposer_compiler_record),
            "proposer_page_count": proposer_compiler_record["expected_page_count"],
            "proposal_or_operation_truncation_applied": False,
            "every_revision_proposal_has_explicit_disposition": True,
            "complete_catalog_single_prompt_present": False,
            "direct_numerical_channel_present": False,
            "non_grounding_numerical_summary_present": False,
            "row_data_present": False,
            "note_text_present": False,
            "oracle_field_present": False,
            "temporal_policy_text_present": False,
            "proposal_frozen_before_next_gate": True,
        }
        identity = {
            "schema_version": ADAPTIVE_ROUND_FREEZE_VERSION,
            "exact_spent_authentication_sha256": (
                self.exact_spent_authentication.authentication_sha256
            ),
            "catalog_sha256": self.catalog.catalog_sha256,
            "chunk_plan_sha256": self.chunk_plan.plan_sha256,
            "dossier_sha256s": [item.dossier_sha256 for item in ordered],
            "current_registry_sha256": _sha(self._registry_private_items()),
            "diagnostics_sha256": _sha(self._diagnostic_prompt_items()),
            "planner_job_id": planner_compilation_id,
            "planner_response_sha256": _sha(planner),
            "lookback_sha256": lookback.lookback_sha256,
            "proposer_job_id": proposer_compilation_id,
            "proposal_sha256": _sha(proposal),
            "still_sealed_gate_fingerprint": (
                self.exact_spent_authentication.still_sealed_gate_fingerprint
            ),
            "proposal": proposal,
            "audit": audit,
        }
        return FrozenAdaptiveReconsiderationRound(
            exact_spent_authentication_sha256=(
                self.exact_spent_authentication.authentication_sha256
            ),
            catalog_sha256=self.catalog.catalog_sha256,
            chunk_plan_sha256=self.chunk_plan.plan_sha256,
            dossier_sha256s=tuple(item.dossier_sha256 for item in ordered),
            current_registry_sha256=_sha(self._registry_private_items()),
            diagnostics_sha256=_sha(self._diagnostic_prompt_items()),
            planner_job_id=planner_compilation_id,
            planner_response_sha256=_sha(planner),
            lookback_sha256=lookback.lookback_sha256,
            proposer_job_id=proposer_compilation_id,
            proposal_sha256=_sha(proposal),
            still_sealed_gate_fingerprint=(
                self.exact_spent_authentication.still_sealed_gate_fingerprint
            ),
            freeze_sha256=_sha(identity),
            _proposal_json=canonical_json(proposal),
            _audit_json=canonical_json(audit),
        )

    def _validate_dossiers(
        self, dossiers: Sequence[AdaptiveArchitectureDossier]
    ) -> tuple[AdaptiveArchitectureDossier, ...]:
        values = tuple(dossiers)
        if len(values) != len(ACTIVE_STAGE1_CONCEPT_FAMILIES):
            raise ValueError("cross-family adaptive work requires exactly ten dossiers")
        if any(not isinstance(item, AdaptiveArchitectureDossier) for item in values):
            raise TypeError("dossiers contains an invalid entry")
        by_family = {item.source_family: item for item in values}
        if len(by_family) != len(values) or set(by_family) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("adaptive dossiers must uniquely cover all active architectures")
        ordered = tuple(by_family[family] for family in ACTIVE_STAGE1_CONCEPT_FAMILIES)
        expected_ids = {
            family: {
                atom.evidence_id for atom in self.catalog.atoms if atom.source_family == family
            }
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        }
        for dossier in ordered:
            if dossier.catalog_sha256 != self.catalog.catalog_sha256:
                raise ValueError("adaptive dossier is bound to another exact-spent catalog")
            if set(dossier.catalog_evidence_ids) != expected_ids[dossier.source_family]:
                raise ValueError("adaptive dossier does not cover its complete architecture")
        candidate_ids = [
            candidate.candidate_id
            for dossier in ordered
            for candidate in dossier.architecture_candidates
        ]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("adaptive candidate IDs must be globally unique")
        return ordered

    def _cross_family_context(
        self, dossiers: Sequence[AdaptiveArchitectureDossier]
    ) -> dict[str, Any]:
        ordered = self._validate_dossiers(dossiers)
        return {
            "architecture_dossiers": [item.as_prompt_item() for item in ordered],
            "current_registry": self._registry_prompt_items(),
            "diagnostics": self._diagnostic_prompt_items(),
        }

    def build_planner_job(
        self, dossiers: Sequence[AdaptiveArchitectureDossier]
    ) -> DiscoveryJsonJob:
        ordered = self._validate_dossiers(dossiers)
        context = self._cross_family_context(ordered)
        request = attach_hierarchical_discovery_response_contract(
            job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
            request={
                "job": "plan_adaptive_stage1_reconsideration",
                **context,
                "lookback_bounds": {
                    "max_ids_per_target": self.config.max_lookback_ids_per_target,
                    "max_total_ids": self.config.max_total_lookback_ids,
                    "max_total_bytes": self.config.max_total_lookback_bytes,
                },
            },
        )
        messages = (
            {"role": "system", "content": _ADAPTIVE_PLANNER_SYSTEM},
            {"role": "user", "content": canonical_json(request)},
        )
        job = self._create_job(
            job_kind=CROSS_ARCHITECTURE_PLANNER_JOB,
            scope="adaptive.cross_family.planner",
            dependencies=tuple(
                job_id for dossier in ordered for job_id in dossier.coverage_job_ids
            ),
            messages=messages,
            input_bindings={
                "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                "planner_interface_version": ADAPTIVE_PLANNER_INTERFACE_VERSION,
                "exact_spent_authentication_sha256": (
                    self.exact_spent_authentication.authentication_sha256
                ),
                "catalog_sha256": self.catalog.catalog_sha256,
                "chunk_plan_sha256": self.chunk_plan.plan_sha256,
                "dossier_sha256s": [item.dossier_sha256 for item in ordered],
                "current_registry_sha256": _sha(self._registry_private_items()),
                "diagnostics_sha256": _sha(self._diagnostic_prompt_items()),
                "lookback_bounds": {
                    "max_ids_per_target": self.config.max_lookback_ids_per_target,
                    "max_total_ids": self.config.max_total_lookback_ids,
                    "max_total_bytes": self.config.max_total_lookback_bytes,
                },
            },
        )
        minimum_proposer_bytes = self._proposer_rendered_byte_count(
            dossiers=ordered,
            planner_response={"review_targets": [], "no_lookback_needed": True},
            requested_evidence_ids=(),
        )
        if minimum_proposer_bytes > self.config.max_rendered_prompt_bytes:
            raise ValueError(
                "adaptive planner context leaves no room for even the minimum proposer prompt"
            )
        self.audit_planner_prompt(job=job, dossiers=ordered)
        return job

    def audit_planner_prompt(
        self,
        *,
        job: DiscoveryJsonJob,
        dossiers: Sequence[AdaptiveArchitectureDossier],
    ) -> dict[str, Any]:
        self._validate_dossiers(dossiers)
        if job.job_kind != CROSS_ARCHITECTURE_PLANNER_JOB:
            raise ValueError("planner prompt audit received another job kind")
        bindings = job.input_bindings
        if bindings.get("catalog_sha256") != self.catalog.catalog_sha256:
            raise ValueError("planner prompt audit received another catalog binding")
        request = json.loads(job.messages[1]["content"])
        expected_keys = {
            "job",
            "architecture_dossiers",
            "current_registry",
            "diagnostics",
            "lookback_bounds",
            "identifier_ownership",
            "output_schema",
        }
        if set(request) != expected_keys:
            raise ValueError("adaptive planner contains unreviewed model context")
        prompt_dossiers = request["architecture_dossiers"]
        if len(prompt_dossiers) != len(ACTIVE_STAGE1_CONCEPT_FAMILIES):
            raise ValueError("adaptive planner did not receive exactly ten compact dossiers")
        for index, dossier in enumerate(prompt_dossiers):
            forbidden = set(dossier) - {"source_family", "coverage", "architecture_candidates"}
            if forbidden:
                raise ValueError(f"planner dossier {index} contains noncompact fields: {forbidden}")
            serialized = canonical_json(dossier)
            if '"content":' in serialized or '"evidence":' in serialized:
                raise ValueError("planner dossier contains raw evidence")
        serialized_prompt = canonical_json(request)
        forbidden_tokens = (
            "direct_upstream_numerical",
            "non_grounding_numerical_summaries",
            "raw_note",
            "full_note",
            "oracle",
            "temporal_policy",
        )
        if any(token in serialized_prompt.casefold() for token in forbidden_tokens):
            raise ValueError("adaptive planner prompt contains a forbidden evidence channel")
        return {
            "compact_dossier_count": len(prompt_dossiers),
            "architecture_order": [row["source_family"] for row in prompt_dossiers],
            "all_ten_architectures_present": True,
            "raw_atom_count": 0,
            "complete_catalog_dump_present": False,
            "direct_numerical_channel_present": False,
            "non_grounding_numerical_summary_present": False,
            "row_data_present": False,
            "note_text_present": False,
            "oracle_field_present": False,
            "temporal_policy_text_present": False,
        }

    def _normalize_planner_wire_response(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        ordered = self._validate_dossiers(dossiers)
        root = _exact_mapping(
            response,
            keys={"review_targets", "no_lookback_needed"},
            label="adaptive planner response",
        )
        targets = root["review_targets"]
        if not isinstance(targets, list):
            raise TypeError("review_targets must be a JSON list")
        if not isinstance(root["no_lookback_needed"], bool):
            raise TypeError("no_lookback_needed must be boolean")
        registry_names = {item.feature_name for item in self.current_registry}
        family_by_evidence = {
            evidence_id: dossier.source_family
            for dossier in ordered
            for evidence_id in dossier.catalog_evidence_ids
        }
        globally_requested: list[str] = []
        globally_requested_set: set[str] = set()
        normalized_targets: list[dict[str, Any]] = []
        normalization_events: list[dict[str, Any]] = []
        for index, raw in enumerate(targets):
            label = f"review_targets[{index}]"
            target = _exact_mapping(
                raw,
                keys={
                    "target",
                    "problem",
                    "relevant_architectures",
                    "requested_evidence_ids",
                    "reason",
                },
                label=label,
            )
            target_name = _require_identifier(target["target"], label=f"{label}.target")
            if target_name != NEW_MISSING_CONSTRUCT and target_name not in registry_names:
                raise ValueError(f"{label} cites an unknown registry feature")
            raw_problem = _require_string(target["problem"], label=f"{label}.problem")
            problem, masked_problem_terms = _sanitize_model_authored_text(raw_problem)
            if masked_problem_terms:
                normalization_events.append(
                    {
                        "target_index": index,
                        "field": "problem",
                        "action": "policy_channel_terms_masked",
                        "values": list(masked_problem_terms),
                    }
                )
            families, duplicate_families = _deduplicated_string_list(
                target["relevant_architectures"],
                label=f"{label}.relevant_architectures",
                identifiers=True,
            )
            if not set(families) <= ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
                raise ValueError(f"{label} cites an inactive architecture")
            requested, duplicate_requested = _deduplicated_string_list(
                target["requested_evidence_ids"],
                label=f"{label}.requested_evidence_ids",
                allow_empty=True,
                identifiers=True,
            )
            unknown = set(requested) - set(family_by_evidence)
            if unknown:
                raise ValueError(
                    f"{label} requests IDs absent from the authenticated dossiers: "
                    f"{sorted(unknown)}"
                )
            if duplicate_families:
                normalization_events.append(
                    {
                        "target_index": index,
                        "field": "relevant_architectures",
                        "action": "duplicate_occurrences_removed",
                        "values": list(duplicate_families),
                    }
                )
            if duplicate_requested:
                normalization_events.append(
                    {
                        "target_index": index,
                        "field": "requested_evidence_ids",
                        "action": "duplicate_occurrences_removed",
                        "values": list(duplicate_requested),
                    }
                )
            retained_families = list(families)
            retained_requested: list[str] = []
            for evidence_id in requested:
                if evidence_id in globally_requested_set:
                    retained_requested.append(evidence_id)
                    normalization_events.append(
                        {
                            "target_index": index,
                            "field": "requested_evidence_ids",
                            "action": "cross_target_evidence_reference_preserved",
                            "values": [evidence_id],
                        }
                    )
                else:
                    retained_requested.append(evidence_id)
                    globally_requested.append(evidence_id)
                    globally_requested_set.add(evidence_id)
                owning_family = family_by_evidence[evidence_id]
                if owning_family not in retained_families:
                    retained_families.append(owning_family)
                    normalization_events.append(
                        {
                            "target_index": index,
                            "field": "relevant_architectures",
                            "action": "owning_architecture_added",
                            "values": [owning_family],
                        }
                    )
            raw_reason = _require_string(target["reason"], label=f"{label}.reason")
            reason, masked_reason_terms = _sanitize_model_authored_text(raw_reason)
            if masked_reason_terms:
                normalization_events.append(
                    {
                        "target_index": index,
                        "field": "reason",
                        "action": "policy_channel_terms_masked",
                        "values": list(masked_reason_terms),
                    }
                )
            normalized_targets.append(
                {
                    "target": target_name,
                    "problem": problem,
                    "relevant_architectures": retained_families,
                    "requested_evidence_ids": retained_requested,
                    "reason": reason,
                }
            )
        normalized_no_lookback = not globally_requested
        rendered_proposer_bytes = self._proposer_rendered_byte_count(
            dossiers=ordered,
            planner_response={
                "review_targets": normalized_targets,
                "no_lookback_needed": normalized_no_lookback,
            },
            requested_evidence_ids=globally_requested,
        )
        if root["no_lookback_needed"] is not normalized_no_lookback:
            normalization_events.append(
                {
                    "target_index": -1,
                    "field": "no_lookback_needed",
                    "action": "compiler_derived_from_retained_requests",
                    "values": [str(normalized_no_lookback).lower()],
                }
            )
        return _clone(
            {
                "review_targets": normalized_targets,
                "no_lookback_needed": normalized_no_lookback,
                "wire_normalization_audit": {
                    "audit_version": _ADAPTIVE_PLANNER_NORMALIZATION_AUDIT_VERSION,
                    "wire_review_targets": targets,
                    "wire_no_lookback_needed": root["no_lookback_needed"],
                    "normalization_events": normalization_events,
                    "retained_requested_evidence_ids": globally_requested,
                    "rendered_proposer_bytes": rendered_proposer_bytes,
                    "lookback_bounds": {
                        "max_ids_per_target": self.config.max_lookback_ids_per_target,
                        "max_total_ids": self.config.max_total_lookback_ids,
                        "max_total_bytes": self.config.max_total_lookback_bytes,
                        "max_rendered_prompt_bytes": self.config.max_rendered_prompt_bytes,
                    },
                },
            }
        )

    def _revalidate_normalized_planner_response(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        root = _exact_mapping(
            response,
            keys={"review_targets", "no_lookback_needed", "wire_normalization_audit"},
            label="normalized adaptive planner response",
        )
        audit = _exact_mapping(
            root["wire_normalization_audit"],
            keys={
                "audit_version",
                "wire_review_targets",
                "wire_no_lookback_needed",
                "normalization_events",
                "retained_requested_evidence_ids",
                "rendered_proposer_bytes",
                "lookback_bounds",
            },
            label="adaptive planner wire_normalization_audit",
        )
        if audit["audit_version"] != _ADAPTIVE_PLANNER_NORMALIZATION_AUDIT_VERSION:
            raise ValueError("adaptive planner normalization audit version is invalid")
        reconstructed = self._normalize_planner_wire_response(
            dossiers=dossiers,
            response={
                "review_targets": audit["wire_review_targets"],
                "no_lookback_needed": audit["wire_no_lookback_needed"],
            },
        )
        if canonical_json(reconstructed) != canonical_json(root):
            raise ValueError("normalized adaptive planner response is not compiler-derived")
        return reconstructed

    def _validate_phased_planner_response(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Revalidate the compact output of the authenticated exhaustive page compiler."""

        self._validate_dossiers(dossiers)
        root = _exact_mapping(
            response,
            keys={"review_targets", "no_lookback_needed", "wire_normalization_audit"},
            label="phased adaptive planner response",
        )
        audit = _exact_mapping(
            root["wire_normalization_audit"],
            keys={
                "audit_version",
                "planner_compilation_id",
                "review_targets_sha256",
                "retained_requested_evidence_ids",
                "page_records_sha256",
                "target_records_sha256",
                "expected_page_count",
                "target_or_evidence_truncation_applied",
            },
            label="phased adaptive planner audit",
        )
        if audit["audit_version"] != _ADAPTIVE_PHASED_PLANNER_COMPILER_VERSION:
            raise ValueError("phased adaptive planner audit version is invalid")
        _require_identifier(audit["planner_compilation_id"], label="planner_compilation_id")
        for field_name in (
            "review_targets_sha256",
            "page_records_sha256",
            "target_records_sha256",
        ):
            _require_sha(audit[field_name], label=field_name)
        if audit["review_targets_sha256"] != _sha(root["review_targets"]):
            raise ValueError("phased adaptive planner target digest changed")
        if audit["target_or_evidence_truncation_applied"] is not False:
            raise ValueError("phased adaptive planner reports semantic truncation")
        expected_page_count = len(self._planner_page_schedule(dossiers))
        if audit["expected_page_count"] != expected_page_count:
            raise ValueError("phased adaptive planner page count differs from its exact schedule")
        targets = root["review_targets"]
        if not isinstance(targets, list):
            raise TypeError("phased adaptive planner review_targets must be a JSON list")
        registry_names = {item.feature_name for item in self.current_registry}
        allowed_targets = {*registry_names, NEW_MISSING_CONSTRUCT}
        seen_targets: set[str] = set()
        normalized_targets: list[dict[str, Any]] = []
        for index, raw in enumerate(targets):
            row = _exact_mapping(
                raw,
                keys={
                    "target",
                    "problem",
                    "relevant_architectures",
                    "requested_evidence_ids",
                    "reason",
                },
                label=f"phased review_targets[{index}]",
            )
            target = _require_identifier(row["target"], label=f"review_targets[{index}].target")
            if target not in allowed_targets or target in seen_targets:
                raise ValueError("phased adaptive planner has an unknown or duplicate target")
            seen_targets.add(target)
            families = _string_list(
                row["relevant_architectures"],
                label=f"review_targets[{index}].relevant_architectures",
                identifiers=True,
            )
            if not set(families) <= ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
                raise ValueError("phased adaptive planner cites an inactive architecture")
            requested = _string_list(
                row["requested_evidence_ids"],
                label=f"review_targets[{index}].requested_evidence_ids",
                allow_empty=True,
                identifiers=True,
            )
            if not set(requested) <= set(self._atom_by_id):
                raise ValueError("phased adaptive planner cites unknown evidence")
            if any(
                self._atom_by_id[evidence_id].source_family not in set(families)
                for evidence_id in requested
            ):
                raise ValueError("phased adaptive planner lost evidence-family ownership")
            normalized_targets.append(
                {
                    "target": target,
                    "problem": _require_string(
                        row["problem"], label=f"review_targets[{index}].problem"
                    ),
                    "relevant_architectures": list(families),
                    "requested_evidence_ids": list(requested),
                    "reason": _require_string(
                        row["reason"], label=f"review_targets[{index}].reason"
                    ),
                }
            )
        retained = tuple(
            dict.fromkeys(
                evidence_id
                for target in normalized_targets
                for evidence_id in target["requested_evidence_ids"]
            )
        )
        if audit["retained_requested_evidence_ids"] != list(retained):
            raise ValueError("phased adaptive planner retained-evidence ledger changed")
        if root["no_lookback_needed"] is not (not retained):
            raise ValueError("phased adaptive planner no_lookback_needed is not compiler-derived")
        return _clone(root)

    def validate_planner_response(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        if isinstance(response, Mapping) and "wire_normalization_audit" in response:
            audit = response.get("wire_normalization_audit")
            if (
                isinstance(audit, Mapping)
                and audit.get("audit_version") == _ADAPTIVE_PHASED_PLANNER_COMPILER_VERSION
            ):
                return self._validate_phased_planner_response(
                    dossiers=dossiers,
                    response=response,
                )
            return self._revalidate_normalized_planner_response(
                dossiers=dossiers,
                response=response,
            )
        return self._normalize_planner_wire_response(dossiers=dossiers, response=response)

    @staticmethod
    def _planner_model_view(response: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "review_targets": _clone(response["review_targets"]),
            "no_lookback_needed": response["no_lookback_needed"],
        }

    def _render_proposer_messages(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        planner_response: Mapping[str, Any],
        requested_evidence_ids: Sequence[str],
    ) -> tuple[dict[str, str], ...]:
        context = self._cross_family_context(dossiers)
        request = attach_hierarchical_discovery_response_contract(
            job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
            request={
                "job": "propose_adaptive_registry_revision",
                **context,
                "review_plan": self._planner_model_view(planner_response),
                "requested_evidence": [
                    self._atom_by_id[evidence_id].as_discovery_item().as_prompt_item()
                    for evidence_id in requested_evidence_ids
                ],
                "maximum_operations": self.config.max_operations,
            },
        )
        return (
            {"role": "system", "content": _ADAPTIVE_PROPOSER_SYSTEM},
            {"role": "user", "content": canonical_json(request)},
        )

    def _proposer_rendered_byte_count(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        planner_response: Mapping[str, Any],
        requested_evidence_ids: Sequence[str],
    ) -> int:
        messages = self._render_proposer_messages(
            dossiers=dossiers,
            planner_response=planner_response,
            requested_evidence_ids=requested_evidence_ids,
        )
        return len(canonical_json(list(messages)).encode("utf-8"))

    def resolve_requested_evidence(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        planner_response: Mapping[str, Any],
    ) -> ResolvedAdaptiveLookback:
        validated = self.validate_planner_response(dossiers=dossiers, response=planner_response)
        requested = tuple(
            dict.fromkeys(
                evidence_id
                for target in validated["review_targets"]
                for evidence_id in target["requested_evidence_ids"]
            )
        )
        items = [
            self._atom_by_id[evidence_id].as_discovery_item().as_prompt_item()
            for evidence_id in requested
        ]
        size = len(canonical_json(items).encode("utf-8"))
        return ResolvedAdaptiveLookback(
            requested_evidence_ids=requested,
            canonical_size_bytes=size,
            lookback_sha256=_sha(items),
            total_catalog_atom_count=len(self.catalog.atoms),
            _items_json=canonical_json(items),
        )

    def build_proposer_job(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        planner_job: DiscoveryJsonJob,
        planner_response: Mapping[str, Any],
        lookback: ResolvedAdaptiveLookback,
    ) -> DiscoveryJsonJob:
        ordered = self._validate_dossiers(dossiers)
        expected_planner_job = self.build_planner_job(ordered)
        if planner_job.job_id != expected_planner_job.job_id:
            raise ValueError("planner_job differs from the authenticated adaptive planner")
        validated_planner = self.validate_planner_response(
            dossiers=ordered,
            response=planner_response,
        )
        expected_lookback = self.resolve_requested_evidence(
            dossiers=ordered,
            planner_response=validated_planner,
        )
        if lookback.lookback_sha256 != expected_lookback.lookback_sha256:
            raise ValueError("lookback differs from deterministic planner-ID resolution")
        messages = self._render_proposer_messages(
            dossiers=ordered,
            planner_response=validated_planner,
            requested_evidence_ids=lookback.requested_evidence_ids,
        )
        job = self._create_job(
            job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
            scope="adaptive.cross_family.proposer",
            dependencies=(planner_job.job_id,),
            messages=messages,
            input_bindings={
                "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                "proposer_interface_version": ADAPTIVE_PROPOSER_INTERFACE_VERSION,
                "exact_spent_authentication_sha256": (
                    self.exact_spent_authentication.authentication_sha256
                ),
                "catalog_sha256": self.catalog.catalog_sha256,
                "chunk_plan_sha256": self.chunk_plan.plan_sha256,
                "dossier_sha256s": [item.dossier_sha256 for item in ordered],
                "current_registry_sha256": _sha(self._registry_private_items()),
                "diagnostics_sha256": _sha(self._diagnostic_prompt_items()),
                "planner_response_sha256": _sha(validated_planner),
                "lookback_sha256": lookback.lookback_sha256,
                "lookback_count": len(lookback.requested_evidence_ids),
                "lookback_size_bytes": lookback.canonical_size_bytes,
                "max_operations": self.config.max_operations,
            },
        )
        self.audit_proposer_prompt(job=job, dossiers=ordered, lookback=lookback)
        return job

    def audit_proposer_prompt(
        self,
        *,
        job: DiscoveryJsonJob,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        lookback: ResolvedAdaptiveLookback,
    ) -> dict[str, Any]:
        self._validate_dossiers(dossiers)
        if job.job_kind != CROSS_ARCHITECTURE_INTEGRATION_JOB:
            raise ValueError("proposer prompt audit received another job kind")
        bindings = job.input_bindings
        if bindings.get("catalog_sha256") != self.catalog.catalog_sha256:
            raise ValueError("proposer prompt audit received another catalog binding")
        if bindings.get("lookback_sha256") != lookback.lookback_sha256:
            raise ValueError("proposer prompt audit received another lookback binding")
        request = json.loads(job.messages[1]["content"])
        expected_keys = {
            "job",
            "architecture_dossiers",
            "current_registry",
            "diagnostics",
            "review_plan",
            "requested_evidence",
            "maximum_operations",
            "identifier_ownership",
            "output_schema",
        }
        if set(request) != expected_keys:
            raise ValueError("adaptive proposer contains unreviewed model context")
        if len(request["architecture_dossiers"]) != len(ACTIVE_STAGE1_CONCEPT_FAMILIES):
            raise ValueError("adaptive proposer did not receive exactly ten compact dossiers")
        observed = tuple(row["evidence_id"] for row in request["requested_evidence"])
        if observed != lookback.requested_evidence_ids:
            raise ValueError("adaptive proposer evidence differs from deterministic lookback")
        if len(observed) >= len(self.catalog.atoms):
            raise ValueError("adaptive proposer contains a complete raw evidence dump")
        dossier_json = canonical_json(request["architecture_dossiers"])
        if '"content":' in dossier_json or '"evidence":' in dossier_json:
            raise ValueError("adaptive proposer compact dossiers contain raw evidence")
        serialized_prompt = canonical_json(request).casefold()
        forbidden_tokens = (
            "direct_upstream_numerical",
            "non_grounding_numerical_summaries",
            "raw_note",
            "full_note",
            "oracle",
            "temporal_policy",
        )
        if any(token in serialized_prompt for token in forbidden_tokens):
            raise ValueError("adaptive proposer prompt contains a forbidden evidence channel")
        return {
            "compact_dossier_count": len(request["architecture_dossiers"]),
            "all_ten_architectures_present": True,
            "raw_atom_count": len(observed),
            "requested_evidence_ids": list(observed),
            "lookback_sha256": lookback.lookback_sha256,
            "complete_catalog_dump_present": False,
            "only_requested_raw_atoms_present": True,
            "direct_numerical_channel_present": False,
            "non_grounding_numerical_summary_present": False,
            "row_data_present": False,
            "note_text_present": False,
            "oracle_field_present": False,
            "temporal_policy_text_present": False,
        }

    @staticmethod
    def _validate_proposed_feature(value: Any, *, label: str) -> dict[str, Any]:
        row = _exact_mapping(
            value,
            keys={
                "feature_name",
                "description",
                "value_shape_hypothesis",
                "definition_summary",
                "source_families",
            },
            label=label,
        )
        feature_name = _require_feature_name(row["feature_name"], label=f"{label}.feature_name")
        description = _require_string(row["description"], label=f"{label}.description")
        if row["value_shape_hypothesis"] not in _VALUE_SHAPES:
            raise ValueError(f"{label}.value_shape_hypothesis is invalid")
        definition_summary = _require_string(
            row["definition_summary"], label=f"{label}.definition_summary"
        )
        families, _ = _deduplicated_string_list(
            row["source_families"],
            label=f"{label}.source_families",
            identifiers=True,
        )
        if not set(families) <= ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError(f"{label} cites an inactive source family")
        return {
            "feature_name": feature_name,
            "description": description,
            "value_shape_hypothesis": row["value_shape_hypothesis"],
            "definition_summary": definition_summary,
            "source_families": list(families),
        }

    def _normalize_proposer_wire_response(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        planner_response: Mapping[str, Any],
        lookback: ResolvedAdaptiveLookback,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._validate_dossiers(dossiers)
        planner = self.validate_planner_response(dossiers=dossiers, response=planner_response)
        expected_lookback = self.resolve_requested_evidence(
            dossiers=dossiers,
            planner_response=planner,
        )
        if lookback.lookback_sha256 != expected_lookback.lookback_sha256:
            raise ValueError("proposer validation received a non-deterministic lookback")
        root = _exact_mapping(
            response,
            keys={"operations", "converged"},
            label="adaptive proposer response",
        )
        operations = root["operations"]
        if not isinstance(operations, list):
            raise TypeError("operations must be a JSON list")
        if not isinstance(root["converged"], bool):
            raise TypeError("converged must be boolean")
        registry_by_name = {item.feature_name: item for item in self.current_registry}
        resolved_ids = set(lookback.requested_evidence_ids)
        diagnostic_ids = {item.diagnostic_id for item in self.diagnostics}
        planned_targets = {target["target"] for target in planner["review_targets"]}
        used_existing_targets: set[str] = set()
        proposed_result_names: set[str] = set()
        normalized_operations: list[dict[str, Any]] = []
        normalization_events: list[dict[str, Any]] = []
        dropped_operation_slots: list[dict[str, Any]] = []

        def record_event(
            index: int,
            *,
            field: str,
            action: str,
            values: Sequence[str],
        ) -> None:
            normalization_events.append(
                {
                    "operation_index": index,
                    "field": field,
                    "action": action,
                    "values": list(values),
                }
            )

        def drop_operation(index: int, *, reason: str) -> None:
            dropped_operation_slots.append({"operation_index": index, "reason": reason})

        for index, raw in enumerate(operations):
            label = f"operations[{index}]"
            operation = _exact_mapping(
                raw,
                keys={
                    "operation",
                    "targets",
                    "proposed_feature",
                    "supporting_evidence_ids",
                    "diagnostic_ids",
                    "reason",
                },
                label=label,
            )
            kind = operation["operation"]
            if kind not in _OPERATIONS:
                raise ValueError(f"{label}.operation is invalid")
            targets, duplicate_targets = _deduplicated_string_list(
                operation["targets"],
                label=f"{label}.targets",
                feature_names=True,
            )
            support, duplicate_support = _deduplicated_string_list(
                operation["supporting_evidence_ids"],
                label=f"{label}.supporting_evidence_ids",
                allow_empty=True,
                identifiers=True,
            )
            if not set(support) <= resolved_ids:
                raise ValueError(
                    f"{label} cites evidence outside the current exact-scope "
                    "requested evidence materialized by the authenticated paged lookback"
                )
            cited_diagnostics, duplicate_diagnostics = _deduplicated_string_list(
                operation["diagnostic_ids"],
                label=f"{label}.diagnostic_ids",
                identifiers=True,
            )
            if not set(cited_diagnostics) <= diagnostic_ids:
                raise ValueError(f"{label} cites an unknown diagnostic")
            reason = _require_string(operation["reason"], label=f"{label}.reason")
            for field_name, duplicates in (
                ("targets", duplicate_targets),
                ("supporting_evidence_ids", duplicate_support),
                ("diagnostic_ids", duplicate_diagnostics),
            ):
                if duplicates:
                    record_event(
                        index,
                        field=field_name,
                        action="duplicate_occurrences_removed",
                        values=duplicates,
                    )

            proposed_raw = operation["proposed_feature"]
            if kind == "drop":
                if support:
                    record_event(
                        index,
                        field="supporting_evidence_ids",
                        action="drop_support_removed",
                        values=support,
                    )
                    support = ()
                if not isinstance(proposed_raw, Mapping) or proposed_raw:
                    raise ValueError("drop operations require an empty proposed_feature")
                proposed: dict[str, Any] = {}
            else:
                proposed = self._validate_proposed_feature(
                    proposed_raw,
                    label=f"{label}.proposed_feature",
                )
                raw_families = tuple(proposed_raw["source_families"])
                if tuple(proposed["source_families"]) != raw_families:
                    record_event(
                        index,
                        field="proposed_feature.source_families",
                        action="duplicate_occurrences_removed",
                        values=raw_families,
                    )

            if kind == "add":
                if NEW_MISSING_CONSTRUCT not in planned_targets:
                    drop_operation(index, reason="missing_construct_was_not_planned")
                    continue
                if not support:
                    drop_operation(index, reason="add_lacks_requested_evidence")
                    continue
                unavailable = (
                    set(registry_by_name) | proposed_result_names | {NEW_MISSING_CONSTRUCT}
                )
                result_name = _compiler_unique_feature_name(
                    str(proposed["feature_name"]),
                    unavailable=unavailable,
                    suffix=f"adaptive_add_{index + 1:03d}",
                )
                if result_name != proposed["feature_name"]:
                    record_event(
                        index,
                        field="proposed_feature.feature_name",
                        action="compiler_disambiguated",
                        values=(str(proposed["feature_name"]), result_name),
                    )
                if targets != (result_name,):
                    record_event(
                        index,
                        field="targets",
                        action="derived_from_proposed_feature_name",
                        values=(*targets, result_name),
                    )
                targets = (result_name,)
                proposed["feature_name"] = result_name
                cited_families = {
                    self._atom_by_id[evidence_id].source_family for evidence_id in support
                }
                derived_families = tuple(
                    family for family in ACTIVE_STAGE1_CONCEPT_FAMILIES if family in cited_families
                )
                if tuple(proposed["source_families"]) != derived_families:
                    record_event(
                        index,
                        field="proposed_feature.source_families",
                        action="derived_from_cited_evidence",
                        values=derived_families,
                    )
                proposed["source_families"] = list(derived_families)
            else:
                if kind == "merge":
                    if len(targets) < 2:
                        drop_operation(index, reason="merge_has_fewer_than_two_distinct_targets")
                        continue
                elif len(targets) != 1:
                    drop_operation(index, reason=f"{kind}_does_not_have_one_distinct_target")
                    continue
                if not set(targets) <= set(registry_by_name):
                    raise ValueError(f"{label} targets a missing registry feature")
                if not set(targets) <= planned_targets:
                    raise ValueError(f"{label} was not selected by the planner")
                repeated = used_existing_targets.intersection(targets)
                if repeated:
                    drop_operation(index, reason="existing_target_already_used_by_earlier_slot")
                    continue
                if kind != "drop":
                    if not support:
                        drop_operation(index, reason=f"{kind}_lacks_requested_evidence")
                        continue
                    target_families = {
                        family
                        for target_name in targets
                        for family in registry_by_name[target_name].source_families
                    }
                    resolved_support_families = {
                        self._atom_by_id[evidence_id].source_family for evidence_id in support
                    }
                    if kind in {"rename", "revise_definition", "merge"}:
                        if not target_families <= resolved_support_families:
                            drop_operation(
                                index,
                                reason=f"{kind}_lacks_evidence_for_retained_architectures",
                            )
                            continue
                        derived_families = tuple(
                            family
                            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
                            if family in target_families or family in resolved_support_families
                        )
                    else:
                        derived_families = tuple(
                            family
                            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
                            if family in resolved_support_families
                        )
                    if tuple(proposed["source_families"]) != derived_families:
                        record_event(
                            index,
                            field="proposed_feature.source_families",
                            action="compiler_derived_provenance",
                            values=derived_families,
                        )
                    proposed["source_families"] = list(derived_families)

                    unavailable = (
                        (set(registry_by_name) - set(targets))
                        | proposed_result_names
                        | {NEW_MISSING_CONSTRUCT}
                    )
                    proposed_name = str(proposed["feature_name"])
                    if kind == "revise_definition":
                        result_name = targets[0]
                    else:
                        if kind in {"rename", "split"}:
                            unavailable.update(targets)
                        result_name = _compiler_unique_feature_name(
                            proposed_name,
                            unavailable=unavailable,
                            suffix=f"adaptive_{kind}_{index + 1:03d}",
                        )
                    if result_name != proposed_name:
                        record_event(
                            index,
                            field="proposed_feature.feature_name",
                            action=(
                                "definition_name_preserved"
                                if kind == "revise_definition"
                                else "compiler_disambiguated"
                            ),
                            values=(proposed_name, result_name),
                        )
                    proposed["feature_name"] = result_name
                used_existing_targets.update(targets)

            if kind != "drop":
                result_name = str(proposed["feature_name"])
                if result_name in proposed_result_names:
                    drop_operation(index, reason="result_name_used_by_earlier_operation")
                    continue
                proposed_result_names.add(result_name)
            normalized = {
                "operation": kind,
                "targets": list(targets),
                "proposed_feature": proposed,
                "supporting_evidence_ids": list(support),
                "diagnostic_ids": list(cited_diagnostics),
                "reason": reason,
            }
            if len(normalized_operations) >= self.config.max_operations:
                drop_operation(
                    index,
                    reason="round_capacity_after_exhaustive_operation_validation",
                )
                continue
            normalized_operations.append(normalized)
        normalized_converged = not normalized_operations
        if root["converged"] is not normalized_converged:
            record_event(
                -1,
                field="converged",
                action="compiler_derived_from_retained_operations",
                values=(str(normalized_converged).lower(),),
            )
        return _clone(
            {
                "operations": normalized_operations,
                "converged": normalized_converged,
                "wire_normalization_audit": {
                    "audit_version": _ADAPTIVE_PROPOSER_NORMALIZATION_AUDIT_VERSION,
                    "wire_operations": operations,
                    "wire_converged": root["converged"],
                    "normalization_events": normalization_events,
                    "dropped_operation_slots": dropped_operation_slots,
                    "maximum_operations": self.config.max_operations,
                },
            }
        )

    def _revalidate_normalized_proposer_response(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        planner_response: Mapping[str, Any],
        lookback: ResolvedAdaptiveLookback,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        root = _exact_mapping(
            response,
            keys={"operations", "converged", "wire_normalization_audit"},
            label="normalized adaptive proposer response",
        )
        audit = _exact_mapping(
            root["wire_normalization_audit"],
            keys={
                "audit_version",
                "wire_operations",
                "wire_converged",
                "normalization_events",
                "dropped_operation_slots",
                "maximum_operations",
            },
            label="adaptive proposer wire_normalization_audit",
        )
        if audit["audit_version"] != _ADAPTIVE_PROPOSER_NORMALIZATION_AUDIT_VERSION:
            raise ValueError("adaptive proposer normalization audit version is invalid")
        reconstructed = self._normalize_proposer_wire_response(
            dossiers=dossiers,
            planner_response=planner_response,
            lookback=lookback,
            response={
                "operations": audit["wire_operations"],
                "converged": audit["wire_converged"],
            },
        )
        if canonical_json(reconstructed) != canonical_json(root):
            raise ValueError("normalized adaptive proposer response is not compiler-derived")
        return reconstructed

    def validate_proposer_response(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        planner_response: Mapping[str, Any],
        lookback: ResolvedAdaptiveLookback,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        if isinstance(response, Mapping) and "wire_normalization_audit" in response:
            return self._revalidate_normalized_proposer_response(
                dossiers=dossiers,
                planner_response=planner_response,
                lookback=lookback,
                response=response,
            )
        return self._normalize_proposer_wire_response(
            dossiers=dossiers,
            planner_response=planner_response,
            lookback=lookback,
            response=response,
        )

    def freeze_round(
        self,
        *,
        dossiers: Sequence[AdaptiveArchitectureDossier],
        planner_job: DiscoveryJsonJob,
        planner_response: Mapping[str, Any],
        lookback: ResolvedAdaptiveLookback,
        proposer_job: DiscoveryJsonJob,
        proposer_response: Mapping[str, Any],
    ) -> FrozenAdaptiveReconsiderationRound:
        ordered = self._validate_dossiers(dossiers)
        expected_planner = self.build_planner_job(ordered)
        if planner_job.job_id != expected_planner.job_id:
            raise ValueError("cannot freeze a round with a different planner job")
        planner = self.validate_planner_response(dossiers=ordered, response=planner_response)
        expected_proposer = self.build_proposer_job(
            dossiers=ordered,
            planner_job=planner_job,
            planner_response=planner,
            lookback=lookback,
        )
        if proposer_job.job_id != expected_proposer.job_id:
            raise ValueError("cannot freeze a round with a different proposer job")
        proposal = self.validate_proposer_response(
            dossiers=ordered,
            planner_response=planner,
            lookback=lookback,
            response=proposer_response,
        )
        planner_audit = self.audit_planner_prompt(job=planner_job, dossiers=ordered)
        proposer_audit = self.audit_proposer_prompt(
            job=proposer_job,
            dossiers=ordered,
            lookback=lookback,
        )
        audit = {
            "schema_version": ADAPTIVE_ROUND_FREEZE_VERSION,
            "fresh_exact_spent_catalog_bound": True,
            "all_ten_architectures_completed_separately": True,
            "planner_prompt": planner_audit,
            "planner_wire_normalization": _clone(planner["wire_normalization_audit"]),
            "lookback": lookback.audit(),
            "proposer_prompt": proposer_audit,
            "proposer_wire_normalization": _clone(proposal["wire_normalization_audit"]),
            "complete_catalog_dump_present": False,
            "direct_numerical_channel_present": False,
            "non_grounding_numerical_summary_present": False,
            "row_data_present": False,
            "note_text_present": False,
            "oracle_field_present": False,
            "temporal_policy_text_present": False,
            "proposal_frozen_before_next_gate": True,
        }
        identity = {
            "schema_version": ADAPTIVE_ROUND_FREEZE_VERSION,
            "exact_spent_authentication_sha256": (
                self.exact_spent_authentication.authentication_sha256
            ),
            "catalog_sha256": self.catalog.catalog_sha256,
            "chunk_plan_sha256": self.chunk_plan.plan_sha256,
            "dossier_sha256s": [item.dossier_sha256 for item in ordered],
            "current_registry_sha256": _sha(self._registry_private_items()),
            "diagnostics_sha256": _sha(self._diagnostic_prompt_items()),
            "planner_job_id": planner_job.job_id,
            "planner_response_sha256": _sha(planner),
            "lookback_sha256": lookback.lookback_sha256,
            "proposer_job_id": proposer_job.job_id,
            "proposal_sha256": _sha(proposal),
            "still_sealed_gate_fingerprint": (
                self.exact_spent_authentication.still_sealed_gate_fingerprint
            ),
            "proposal": proposal,
            "audit": audit,
        }
        return FrozenAdaptiveReconsiderationRound(
            exact_spent_authentication_sha256=(
                self.exact_spent_authentication.authentication_sha256
            ),
            catalog_sha256=self.catalog.catalog_sha256,
            chunk_plan_sha256=self.chunk_plan.plan_sha256,
            dossier_sha256s=tuple(item.dossier_sha256 for item in ordered),
            current_registry_sha256=_sha(self._registry_private_items()),
            diagnostics_sha256=_sha(self._diagnostic_prompt_items()),
            planner_job_id=planner_job.job_id,
            planner_response_sha256=_sha(planner),
            lookback_sha256=lookback.lookback_sha256,
            proposer_job_id=proposer_job.job_id,
            proposal_sha256=_sha(proposal),
            still_sealed_gate_fingerprint=(
                self.exact_spent_authentication.still_sealed_gate_fingerprint
            ),
            freeze_sha256=_sha(identity),
            _proposal_json=canonical_json(proposal),
            _audit_json=canonical_json(audit),
        )

    def build_extraction_definition_jobs(
        self,
        *,
        frozen_round: FrozenAdaptiveReconsiderationRound,
        lookback: ResolvedAdaptiveLookback,
    ) -> tuple[tuple[DiscoveryJsonJob, ExtractionDefinitionRequest], ...]:
        """Define every proposed feature from only its cited requested atoms."""

        frozen_round.__post_init__()
        lookback.__post_init__()
        if frozen_round.catalog_sha256 != self.catalog.catalog_sha256:
            raise ValueError("frozen proposal is bound to another exact-spent catalog")
        if frozen_round.lookback_sha256 != lookback.lookback_sha256:
            raise ValueError("definition jobs received another authenticated paged lookback")
        available = set(lookback.requested_evidence_ids)
        jobs: list[tuple[DiscoveryJsonJob, ExtractionDefinitionRequest]] = []
        for operation_index, operation in enumerate(frozen_round.proposal["operations"]):
            if operation["operation"] == "drop":
                continue
            support = tuple(operation["supporting_evidence_ids"])
            if not support or not set(support) <= available:
                raise ValueError(
                    "extraction definition requires every cited operation atom to be "
                    "materialized by the bounded current lookback"
                )
            evidence = tuple(
                self._atom_by_id[evidence_id].as_discovery_item() for evidence_id in support
            )
            proposed = operation["proposed_feature"]
            request = ExtractionDefinitionRequest(
                canonical_name=str(proposed["feature_name"]),
                evidence=evidence,
                supporting_evidence_ids=support,
                value_shape_hypothesis=str(proposed["value_shape_hypothesis"]),
            )
            routing = route_concept_roles(
                evidence=evidence,
                supporting_evidence_ids=support,
            )
            definition_messages = _render_extraction_messages(request=request)
            if definition_messages[0] != {
                "role": "system",
                "content": _ADAPTIVE_DEFINITION_SYSTEM,
            }:
                raise ValueError("definition system prompt differs from the approved contract")
            job = self._create_job(
                job_kind=EXTRACTION_DEFINITION_JOB,
                scope=(
                    f"adaptive.executable.operation_{operation_index:03d}."
                    f"{request.canonical_name}"
                ),
                dependencies=(frozen_round.proposer_job_id,),
                settings=DiscoveryJobSettings.extraction(),
                messages=definition_messages,
                input_bindings={
                    "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                    "executable_bridge_version": ADAPTIVE_EXECUTABLE_BRIDGE_VERSION,
                    "proposal_freeze_sha256": frozen_round.freeze_sha256,
                    "operation_index": operation_index,
                    "operation": operation["operation"],
                    "canonical_name": request.canonical_name,
                    "supporting_evidence_ids": list(support),
                    "supporting_evidence_sha256": _sha(
                        [item.as_prompt_item() for item in evidence]
                    ),
                    "value_shape_hypothesis": request.value_shape_hypothesis,
                    "vocabulary_grounding_policy_sha256": _sha(
                        extraction_vocabulary_grounding_policy()
                    ),
                    "deterministic_role_routing": routing.audit(),
                    "role_routing_sha256": _sha(routing.audit()),
                    "lookback_sha256": lookback.lookback_sha256,
                    "definition_thinking_enabled": False,
                },
            )
            if job.settings != DiscoveryJobSettings.extraction():
                raise RuntimeError("adaptive extraction definition did not disable thinking")
            jobs.append((job, request))
        return tuple(jobs)

    def _execute_phased_extraction_definitions(
        self,
        *,
        frozen_round: FrozenAdaptiveReconsiderationRound,
        lookback: ResolvedAdaptiveLookback,
        proposer_compilation_id: str,
        proposer_dependency_ids: Sequence[str],
        run_job: Callable[[DiscoveryJsonJob, Callable[[Any], Mapping[str, Any]]], dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Review every cited support item once and recursively fold each definition."""

        frozen_round.__post_init__()
        lookback.__post_init__()
        _require_identifier(proposer_compilation_id, label="proposer_compilation_id")
        proposer_dependencies = tuple(dict.fromkeys(proposer_dependency_ids))
        if frozen_round.catalog_sha256 != self.catalog.catalog_sha256:
            raise ValueError("phased definitions received another exact-spent catalog")
        if frozen_round.lookback_sha256 != lookback.lookback_sha256:
            raise ValueError("phased definitions received another authenticated paged lookback")
        available = set(lookback.requested_evidence_ids)
        artifacts: dict[str, dict[str, Any]] = {}
        feature_records: list[dict[str, Any]] = []
        all_job_ids: list[str] = []
        all_response_sha256s: list[str] = []
        for operation_index, operation in enumerate(frozen_round.proposal["operations"]):
            if operation["operation"] == "drop":
                continue
            support = tuple(operation["supporting_evidence_ids"])
            if not support or not set(support) <= available:
                raise ValueError(
                    "phased extraction requires every cited operation atom in the exact lookback"
                )
            evidence = tuple(
                self._atom_by_id[evidence_id].as_discovery_item() for evidence_id in support
            )
            proposed = operation["proposed_feature"]
            request = ExtractionDefinitionRequest(
                canonical_name=str(proposed["feature_name"]),
                evidence=evidence,
                supporting_evidence_ids=support,
                value_shape_hypothesis=str(proposed["value_shape_hypothesis"]),
            )
            if request.canonical_name in artifacts:
                raise ValueError("phased extraction feature names cannot repeat")
            routing = route_concept_roles(
                evidence=evidence,
                supporting_evidence_ids=support,
            )
            support_sha256 = _sha(list(support))
            page_rows: list[tuple[str, dict[str, Any], str, str]] = []
            page_records: list[dict[str, Any]] = []
            for evidence_index, evidence_item in enumerate(evidence):
                review_id = (
                    "adaptive_extraction_review_"
                    f"{_sha({'canonical_name': request.canonical_name, 'evidence_id': evidence_item.evidence_id})}"
                )
                page_job = self._create_job(
                    job_kind=EXTRACTION_DEFINITION_JOB,
                    scope=(
                        f"adaptive.executable.operation_{operation_index:03d}."
                        f"evidence_page_{evidence_index:06d}"
                    ),
                    dependencies=proposer_dependencies,
                    settings=DiscoveryJobSettings.extraction(),
                    messages=_render_extraction_evidence_page_messages(
                        request=request,
                        evidence=evidence_item,
                        review_id=review_id,
                    ),
                    input_bindings={
                        "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                        "executable_bridge_version": ADAPTIVE_EXECUTABLE_BRIDGE_VERSION,
                        "proposal_freeze_sha256": frozen_round.freeze_sha256,
                        "proposer_compilation_id": proposer_compilation_id,
                        "operation_index": operation_index,
                        "canonical_name": request.canonical_name,
                        "complete_supporting_evidence_ids_sha256": support_sha256,
                        "evidence_review_id": review_id,
                        "evidence_id": evidence_item.evidence_id,
                        "evidence_index": evidence_index,
                        "evidence_count": len(evidence),
                        "raw_evidence_sha256": _sha(evidence_item.as_prompt_item()),
                        "value_shape_hypothesis": request.value_shape_hypothesis,
                        "vocabulary_grounding_policy_sha256": _sha(
                            extraction_vocabulary_grounding_policy()
                        ),
                        "deterministic_role_routing": routing.audit(),
                        "role_routing_sha256": _sha(routing.audit()),
                        "raw_support_sampling": False,
                        "semantic_truncation_applied": False,
                    },
                )
                page_response = run_job(
                    page_job,
                    lambda raw, request=request, evidence_item=evidence_item: (
                        _validate_extraction_evidence_page_response(
                            raw,
                            request=request,
                            evidence=evidence_item,
                        )
                    ),
                )
                page_rows.append(
                    (
                        review_id,
                        page_response,
                        page_job.job_id,
                        evidence_item.evidence_id,
                    )
                )
                page_records.append(
                    {
                        "review_id": review_id,
                        "evidence_id": evidence_item.evidence_id,
                        "job_id": page_job.job_id,
                        "normalized_response": _clone(page_response),
                        "normalized_response_sha256": _sha(page_response),
                        "disposition": "reviewed_exactly_once",
                    }
                )
                all_job_ids.append(page_job.job_id)
                all_response_sha256s.append(_sha(page_response))

            consumed = 0
            fold_index = 0
            accumulator_id: str | None = None
            accumulator_wire: dict[str, Any] | None = None
            accumulator_job_id: str | None = None
            terminal_definition: dict[str, Any] | None = None
            fold_records: list[dict[str, Any]] = []
            while consumed < len(page_rows):
                fresh_capacity = (
                    HIERARCHICAL_DISCOVERY_MAX_DEFINITION_FOLD_MEMBERS
                    if accumulator_wire is None
                    else HIERARCHICAL_DISCOVERY_MAX_DEFINITION_FOLD_MEMBERS - 1
                )
                fresh = page_rows[consumed : consumed + fresh_capacity]
                review_inputs: list[tuple[str, Mapping[str, Any]]] = []
                dependencies: list[str] = []
                if accumulator_wire is not None:
                    if accumulator_id is None or accumulator_job_id is None:
                        raise AssertionError("adaptive extraction accumulator lost its identity")
                    review_inputs.append((accumulator_id, accumulator_wire))
                    dependencies.append(accumulator_job_id)
                review_inputs.extend((review_id, response) for review_id, response, _, _ in fresh)
                dependencies.extend(job_id for _, _, job_id, _ in fresh)
                review_input_ids = tuple(review_id for review_id, _ in review_inputs)
                fold_job = self._create_job(
                    job_kind=EXTRACTION_DEFINITION_JOB,
                    scope=(
                        f"adaptive.executable.operation_{operation_index:03d}."
                        f"evidence_fold_{fold_index:06d}"
                    ),
                    dependencies=tuple(dict.fromkeys(dependencies)),
                    settings=DiscoveryJobSettings.extraction(),
                    messages=_render_extraction_evidence_fold_messages(
                        request=request,
                        fold_index=fold_index,
                        review_inputs=review_inputs,
                    ),
                    input_bindings={
                        "adaptive_hierarchy_version": ADAPTIVE_HIERARCHY_VERSION,
                        "executable_bridge_version": ADAPTIVE_EXECUTABLE_BRIDGE_VERSION,
                        "proposal_freeze_sha256": frozen_round.freeze_sha256,
                        "proposer_compilation_id": proposer_compilation_id,
                        "operation_index": operation_index,
                        "canonical_name": request.canonical_name,
                        "fold_index": fold_index,
                        "review_input_ids": list(review_input_ids),
                        "fresh_review_ids": [row[0] for row in fresh],
                        "fresh_evidence_ids": [row[3] for row in fresh],
                        "prior_accumulator_id": accumulator_id,
                        "prior_accumulator_response_sha256": (
                            _sha(accumulator_wire) if accumulator_wire is not None else None
                        ),
                        "complete_supporting_evidence_ids_sha256": support_sha256,
                        "value_shape_hypothesis": request.value_shape_hypothesis,
                        "vocabulary_grounding_policy_sha256": _sha(
                            extraction_vocabulary_grounding_policy()
                        ),
                        "deterministic_role_routing": routing.audit(),
                        "role_routing_sha256": _sha(routing.audit()),
                        "raw_support_sampling": False,
                        "semantic_truncation_applied": False,
                    },
                )
                fold_response = run_job(
                    fold_job,
                    lambda raw, request=request, review_input_ids=review_input_ids: (
                        _validate_extraction_evidence_fold_response(
                            raw,
                            request=request,
                            review_input_ids=review_input_ids,
                        )
                    ),
                )
                accumulator_wire = _clone(fold_response["fold_wire"])
                terminal_definition = _clone(fold_response["definition"])
                accumulator_job_id = fold_job.job_id
                accumulator_id = (
                    "adaptive_extraction_accumulator_"
                    f"{_sha({'canonical_name': request.canonical_name, 'fold_index': fold_index, 'response': accumulator_wire})}"
                )
                fold_records.append(
                    {
                        "fold_index": fold_index,
                        "job_id": fold_job.job_id,
                        "review_input_ids": list(review_input_ids),
                        "fresh_evidence_ids": [row[3] for row in fresh],
                        "normalized_response": _clone(fold_response),
                        "normalized_response_sha256": _sha(fold_response),
                    }
                )
                all_job_ids.append(fold_job.job_id)
                all_response_sha256s.append(_sha(fold_response))
                consumed += len(fresh)
                fold_index += 1
            if terminal_definition is None or accumulator_job_id is None:
                raise AssertionError("adaptive extraction support schedule produced no fold")
            if tuple(terminal_definition["supporting_evidence_ids"]) != support:
                raise ValueError("terminal adaptive definition lost compiler-owned support")
            feature_record = {
                "operation_index": operation_index,
                "canonical_name": request.canonical_name,
                "complete_supporting_evidence_ids": list(support),
                "complete_supporting_evidence_ids_sha256": support_sha256,
                "page_records": page_records,
                "fold_records": fold_records,
                "terminal_fold_job_id": accumulator_job_id,
                "terminal_definition": _clone(terminal_definition),
                "terminal_definition_sha256": _sha(terminal_definition),
                "every_support_item_reviewed_exactly_once": True,
                "all_page_reviews_transitively_folded": True,
                "raw_support_sampling": False,
                "semantic_truncation_applied": False,
            }
            feature_records.append(feature_record)
            artifacts[request.canonical_name] = {
                "operation_index": operation_index,
                "request": request,
                "definition": terminal_definition,
                "terminal_job_id": accumulator_job_id,
                "job_ids": tuple(
                    [row["job_id"] for row in page_records]
                    + [row["job_id"] for row in fold_records]
                ),
                "response_sha256s": tuple(
                    [row["normalized_response_sha256"] for row in page_records]
                    + [row["normalized_response_sha256"] for row in fold_records]
                ),
                "feature_record_sha256": _sha(feature_record),
            }
        compiler_body = {
            "schema_version": _ADAPTIVE_PHASED_EXTRACTION_COMPILER_VERSION,
            "proposal_freeze_sha256": frozen_round.freeze_sha256,
            "proposer_compilation_id": proposer_compilation_id,
            "feature_records": feature_records,
            "all_job_ids": all_job_ids,
            "all_response_sha256s": all_response_sha256s,
            "every_support_item_reviewed_exactly_once": True,
            "all_page_reviews_transitively_folded": True,
            "complete_support_single_prompt_present": False,
            "semantic_truncation_applied": False,
        }
        compiler_id = f"adaptive_extraction_compilation_{_sha(compiler_body)}"
        compiler_record = {
            "record_type": "adaptive_phased_extraction_compilation",
            "extraction_compilation_id": compiler_id,
            **compiler_body,
        }
        return artifacts, _clone(compiler_record)

    def validate_extraction_definition_job_response(
        self,
        *,
        job: DiscoveryJsonJob,
        request: ExtractionDefinitionRequest,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        if job.job_kind != EXTRACTION_DEFINITION_JOB:
            raise ValueError("definition validator received another adaptive job kind")
        if job.input_bindings.get("canonical_name") != request.canonical_name:
            raise ValueError("definition job and request canonical names differ")
        if job.input_bindings.get("supporting_evidence_ids") != list(
            request.supporting_evidence_ids
        ):
            raise ValueError("definition job and request evidence IDs differ")
        validated = validate_extraction_definition_response(response, request=request)
        _scan_model_safe(validated, path="adaptive_extraction_definition_response")
        return validated

    def revalidate_normalized_extraction_definition_job_response(
        self,
        *,
        job: DiscoveryJsonJob,
        request: ExtractionDefinitionRequest,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Revalidate an internal projection without weakening the remote wire contract."""

        if job.job_kind != EXTRACTION_DEFINITION_JOB:
            raise ValueError("definition validator received another adaptive job kind")
        if job.input_bindings.get("canonical_name") != request.canonical_name:
            raise ValueError("definition job and request canonical names differ")
        if job.input_bindings.get("supporting_evidence_ids") != list(
            request.supporting_evidence_ids
        ):
            raise ValueError("definition job and request evidence IDs differ")
        validated = revalidate_normalized_extraction_definition_response(
            response,
            request=request,
        )
        _scan_model_safe(validated, path="adaptive_extraction_definition_response")
        return validated

    @staticmethod
    def _definition_description(definition: Mapping[str, Any]) -> str:
        representation = definition["representation"]
        parts = [str(definition["measurement"]).strip()]
        if representation["kind"] == "continuous":
            if representation["unit"] == AS_DOCUMENTED_UNIT:
                parts.append(
                    "Extract a continuous value using the source-documented scale; "
                    "as_documented is an extraction mechanic, not a clinical unit assertion."
                )
            else:
                parts.append(f"Extract a continuous value in {representation['unit']}.")
        elif tuple(representation["categories"]) == MECHANICAL_MENTION_CATEGORIES:
            parts.append(
                "Use exactly not_mentioned and mentioned as a document-observation encoding; "
                "these are extraction mechanics, not a clinical status ontology."
            )
        else:
            parts.append("Use exactly the declared categorical values.")
        if definition["aliases"]:
            parts.append(f"Feature-name aliases: {', '.join(definition['aliases'])}.")
        if definition["distinguish_from"]:
            parts.append(
                "Distinguish this feature from: " f"{', '.join(definition['distinguish_from'])}."
            )
        parts.append(str(definition["missing_or_ambiguous"]).strip())
        return " ".join(parts)

    def _contract_from_definition(
        self,
        *,
        operation: Mapping[str, Any],
        request: ExtractionDefinitionRequest,
        definition: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        representation = definition["representation"]
        if representation["kind"] == "unresolved":
            raise ValueError(
                f"adaptive feature {request.canonical_name!r} has unresolved extraction"
            )
        routing = route_concept_roles(
            evidence=request.evidence,
            supporting_evidence_ids=request.supporting_evidence_ids,
        )
        roles = [
            role
            for role, enabled in (
                ("confounder", bool(routing.adjustment_roles)),
                ("effect_modifier", routing.effect_modifier),
            )
            if enabled
        ]
        if not roles:
            raise ValueError(
                f"adaptive feature {request.canonical_name!r} has no deterministic modeled role"
            )
        spec: dict[str, Any] = {
            "name": request.canonical_name,
            "type": representation["kind"],
            "roles": roles,
            "description": self._definition_description(definition),
        }
        if representation["kind"] == "categorical":
            spec["categories"] = list(representation["categories"])
        proposed = operation["proposed_feature"]
        contract = CandidateContract(
            spec,
            source_families=tuple(proposed["source_families"]),
        ).extraction_spec
        grounding_rows: list[dict[str, Any]] = []
        unrelated: list[str] = []
        for evidence_id, item in zip(request.supporting_evidence_ids, request.evidence):
            grounding = ground_evidence_to_extraction_contract(
                item.as_prompt_item(),
                contract,
            )
            grounding_rows.append({"evidence_id": evidence_id, **grounding.as_dict()})
            if not grounding.supported:
                unrelated.append(evidence_id)
        if unrelated:
            raise ValueError(
                "adaptive executable contract cites requested evidence unrelated to its "
                f"canonical name: {unrelated}"
            )
        return contract, {
            "deterministic_role_routing": routing.audit(),
            "role_routing_sha256": _sha(routing.audit()),
            "evidence_contract_grounding": grounding_rows,
        }

    def freeze_executable_revision(
        self,
        *,
        current_specs: Sequence[Mapping[str, Any]],
        frozen_round: FrozenAdaptiveReconsiderationRound,
        lookback: ResolvedAdaptiveLookback,
        definition_responses: Mapping[str, Mapping[str, Any]],
        max_contracts: int,
        phased_definition_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> FrozenAdaptiveExecutableRevision:
        """Compile adds/splits and other operations without legacy target filtering."""

        before = [CandidateContract(spec).extraction_spec for spec in current_specs]
        if not before:
            raise ValueError("adaptive executable bridge requires a current registry")
        if isinstance(max_contracts, bool) or not isinstance(max_contracts, int):
            raise TypeError("max_contracts must be an integer")
        definition_by_name: dict[str, tuple[str, ExtractionDefinitionRequest, dict[str, Any]]] = {}
        if phased_definition_artifacts is None:
            definitions = self.build_extraction_definition_jobs(
                frozen_round=frozen_round,
                lookback=lookback,
            )
            expected_job_ids = {job.job_id for job, _request in definitions}
            if set(definition_responses) != expected_job_ids:
                raise ValueError("definition response job IDs differ from executable jobs")
            definition_job_ids = tuple(job.job_id for job, _request in definitions)
            definition_response_sha256s = tuple(
                _sha(definition_responses[job.job_id]) for job, _request in definitions
            )
            for job, request in definitions:
                raw_definition = definition_responses[job.job_id]
                if isinstance(raw_definition, Mapping) and isinstance(
                    raw_definition.get("supporting_evidence_ids"), list
                ):
                    validated = self.revalidate_normalized_extraction_definition_job_response(
                        job=job,
                        request=request,
                        response=raw_definition,
                    )
                else:
                    validated = self.validate_extraction_definition_job_response(
                        job=job,
                        request=request,
                        response=raw_definition,
                    )
                if request.canonical_name in definition_by_name:
                    raise ValueError("adaptive definition jobs repeat a result feature name")
                definition_by_name[request.canonical_name] = (
                    job.job_id,
                    request,
                    validated,
                )
            phased_definition_compilation = False
        else:
            if definition_responses:
                raise ValueError(
                    "phased executable freeze cannot also receive legacy definition responses"
                )
            artifacts = dict(phased_definition_artifacts)
            definition_job_id_values: list[str] = []
            definition_response_sha_values: list[str] = []
            for canonical_name, raw_artifact in artifacts.items():
                if not isinstance(raw_artifact, Mapping):
                    raise TypeError("phased definition artifact must be one mapping")
                request = raw_artifact.get("request")
                definition = raw_artifact.get("definition")
                terminal_job_id = raw_artifact.get("terminal_job_id")
                job_ids = raw_artifact.get("job_ids")
                response_sha256s = raw_artifact.get("response_sha256s")
                if not isinstance(request, ExtractionDefinitionRequest):
                    raise TypeError("phased definition artifact lost its exact request")
                if canonical_name != request.canonical_name:
                    raise ValueError("phased definition artifact changed its canonical name")
                _require_identifier(terminal_job_id, label="terminal definition job ID")
                if (
                    not isinstance(job_ids, tuple)
                    or not isinstance(response_sha256s, tuple)
                    or len(job_ids) != len(response_sha256s)
                    or not job_ids
                ):
                    raise ValueError("phased definition artifact job ledger is invalid")
                for job_id in job_ids:
                    _require_identifier(job_id, label="phased definition job ID")
                for digest in response_sha256s:
                    _require_sha(digest, label="phased definition response SHA")
                if terminal_job_id != job_ids[-1]:
                    raise ValueError("phased definition terminal job is not its final fold")
                validated = validate_extraction_definition_response(
                    definition,
                    request=request,
                )
                if tuple(validated["supporting_evidence_ids"]) != tuple(
                    request.supporting_evidence_ids
                ):
                    raise ValueError("phased definition lost compiler-owned support")
                definition_by_name[canonical_name] = (
                    terminal_job_id,
                    request,
                    validated,
                )
                definition_job_id_values.extend(job_ids)
                definition_response_sha_values.extend(response_sha256s)
            if len(definition_job_id_values) != len(set(definition_job_id_values)):
                raise ValueError("phased definition physical job IDs repeat")
            definition_job_ids = tuple(definition_job_id_values)
            definition_response_sha256s = tuple(definition_response_sha_values)
            phased_definition_compilation = True

        expected_definition_names = {
            str(operation["proposed_feature"]["feature_name"])
            for operation in frozen_round.proposal["operations"]
            if operation["operation"] != "drop"
        }
        if set(definition_by_name) != expected_definition_names:
            raise ValueError("definition artifacts differ from the frozen non-drop operations")

        after = [_clone(spec) for spec in before]
        operation_audit: list[dict[str, Any]] = []
        for operation_index, operation in enumerate(frozen_round.proposal["operations"]):
            kind = str(operation["operation"])
            targets = tuple(operation["targets"])
            if kind == "drop":
                after = [spec for spec in after if str(spec["name"]) not in set(targets)]
                operation_audit.append(
                    {
                        "bridge_version": ADAPTIVE_EXECUTABLE_BRIDGE_VERSION,
                        "operation_index": operation_index,
                        "adaptive_operation": kind,
                        "target_names": list(targets),
                        "contract": None,
                        "supporting_diagnostic_ids": list(operation["diagnostic_ids"]),
                        "supporting_evidence_ids": [],
                        "reason": operation["reason"],
                        "definition_job_id": None,
                        "definition_response_sha256": None,
                        "evidence_contract_grounding": [],
                    }
                )
                continue

            proposed_name = str(operation["proposed_feature"]["feature_name"])
            definition_job_id, request, definition = definition_by_name[proposed_name]
            contract, grounding_audit = self._contract_from_definition(
                operation=operation,
                request=request,
                definition=definition,
            )
            names_before = [str(spec["name"]) for spec in after]
            if kind == "add":
                if proposed_name in names_before:
                    raise ValueError("adaptive add collides with the executable registry")
                after.append(contract)
            elif kind == "split":
                target_index = names_before.index(targets[0])
                if proposed_name in names_before:
                    raise ValueError("adaptive split collides with the executable registry")
                after.insert(target_index + 1, contract)
            else:
                target_indices = [names_before.index(name) for name in targets]
                first = min(target_indices)
                target_set = set(targets)
                after = [spec for spec in after if str(spec["name"]) not in target_set]
                remaining_names = {str(spec["name"]) for spec in after}
                if proposed_name in remaining_names:
                    raise ValueError("adaptive replacement collides with an untargeted feature")
                after.insert(first, contract)
            operation_audit.append(
                {
                    "bridge_version": ADAPTIVE_EXECUTABLE_BRIDGE_VERSION,
                    "operation_index": operation_index,
                    "adaptive_operation": kind,
                    "target_names": list(targets),
                    "proposed_feature": _clone(operation["proposed_feature"]),
                    "contract": contract,
                    "supporting_diagnostic_ids": list(operation["diagnostic_ids"]),
                    "supporting_evidence_ids": list(operation["supporting_evidence_ids"]),
                    "reason": operation["reason"],
                    "definition_job_id": definition_job_id,
                    "definition_response_sha256": _sha(definition),
                    **grounding_audit,
                }
            )

        if not after:
            raise ValueError("adaptive executable revision cannot remove every contract")
        if len(after) > max_contracts:
            raise ValueError("adaptive executable revision exceeds max_contracts")
        names = [str(spec["name"]) for spec in after]
        if len(names) != len(set(names)):
            raise ValueError("adaptive executable revision produced duplicate contract names")
        before_by_name = {str(spec["name"]): spec for spec in before}
        after_by_name = {str(spec["name"]): spec for spec in after}
        extraction_changed = tuple(
            name
            for name, spec in after_by_name.items()
            if name not in before_by_name
            or extraction_semantics_sha256(before_by_name[name])
            != extraction_semantics_sha256(spec)
        )
        role_only = tuple(
            name
            for name, spec in after_by_name.items()
            if name in before_by_name
            and extraction_semantics_sha256(before_by_name[name])
            == extraction_semantics_sha256(spec)
            and tuple(before_by_name[name].get("roles") or ()) != tuple(spec.get("roles") or ())
        )
        applied = {
            "specs": after,
            "reextract_specs": [after_by_name[name] for name in extraction_changed],
            "removed_names": [name for name in before_by_name if name not in after_by_name],
            "added_names": [name for name in after_by_name if name not in before_by_name],
            "extraction_changed_names": list(extraction_changed),
            "role_only_changed_names": list(role_only),
            "operation_audit": operation_audit,
        }
        audit = {
            "schema_version": ADAPTIVE_EXECUTABLE_BRIDGE_VERSION,
            "proposal_freeze_sha256": frozen_round.freeze_sha256,
            "paged_lookback_sha256": lookback.lookback_sha256,
            "definition_job_count": len(definition_job_ids),
            "definition_jobs_use_requested_atoms_only": True,
            "phased_definition_compilation": phased_definition_compilation,
            "one_raw_support_item_per_definition_page": phased_definition_compilation,
            "all_definition_pages_folded_to_terminal_full_support": (phased_definition_compilation),
            "definition_jobs_thinking_enabled": False,
            "proposer_summary_used_to_invent_categories": False,
            "proposer_summary_used_to_invent_roles": False,
            "roles_routed_from_exact_cited_evidence_axes": True,
            "legacy_current_target_validator_used_for_add_or_split": False,
            "unavailable_historical_support_ids_cited": False,
            "executable_revision_frozen_before_next_gate": True,
        }
        identity = {
            "schema_version": ADAPTIVE_EXECUTABLE_BRIDGE_VERSION,
            "proposal_freeze_sha256": frozen_round.freeze_sha256,
            "definition_job_ids": list(definition_job_ids),
            "definition_response_sha256s": list(definition_response_sha256s),
            "applied_specs_sha256": _sha(after),
            "applied": applied,
            "audit": audit,
        }
        return FrozenAdaptiveExecutableRevision(
            proposal_freeze_sha256=frozen_round.freeze_sha256,
            definition_job_ids=definition_job_ids,
            definition_response_sha256s=definition_response_sha256s,
            applied_specs_sha256=_sha(after),
            executable_freeze_sha256=_sha(identity),
            _applied_json=canonical_json(applied),
            _audit_json=canonical_json(audit),
        )

    def execute_authenticated(
        self,
        *,
        runner: JsonDiscoveryJobRunner,
        job_cache: AuthenticatedHierarchicalDiscoveryJobCache,
        approved_adaptive_identity: Mapping[str, Any],
        approved_runner_identity: Mapping[str, Any],
        approved_cache_identity: Mapping[str, Any],
        current_specs: Sequence[Mapping[str, Any]],
        max_contracts: int,
    ) -> ExecutedAdaptiveReconsiderationRound:
        """Run every adaptive stage through the same approved transport/cache identities."""

        if not isinstance(approved_adaptive_identity, Mapping):
            raise TypeError("approved_adaptive_identity must be one JSON object")
        expected_adaptive_identity = _clone(approved_adaptive_identity)
        current_adaptive_identity = adaptive_hierarchical_stage1_reconsideration_identity(
            self.config
        )
        if canonical_json(expected_adaptive_identity) != canonical_json(current_adaptive_identity):
            raise ValueError(
                "adaptive implementation or policy differs from the offline-approved identity"
            )

        if not isinstance(runner, JsonDiscoveryJobRunner):
            raise TypeError("runner must implement JsonDiscoveryJobRunner")
        metadata = getattr(runner, "execution_metadata", None)
        if isinstance(metadata, (str, bytes, Mapping)) or not isinstance(metadata, Sequence):
            raise TypeError("runner must expose execution_metadata")
        if not isinstance(job_cache, AuthenticatedHierarchicalDiscoveryJobCache):
            raise TypeError("job_cache has the wrong type")
        expected_runner = _validated_identity(
            approved_runner_identity,
            label="approved_runner_identity",
        )
        expected_cache = _validated_identity(
            approved_cache_identity,
            label="approved_cache_identity",
        )

        def assert_identities() -> tuple[dict[str, Any], dict[str, Any]]:
            observed_runner = _validated_identity(runner.identity(), label="runner_identity")
            observed_cache = _validated_identity(job_cache.identity(), label="cache_identity")
            if canonical_json(observed_runner) != canonical_json(expected_runner):
                raise ValueError("adaptive transport differs from the initially approved runner")
            if canonical_json(observed_cache) != canonical_json(expected_cache):
                raise ValueError("adaptive cache differs from the initially approved cache")
            return observed_runner, observed_cache

        runner_identity, cache_identity = assert_identities()
        current_file_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if current_file_sha256 != self._implementation_file_sha256:
            raise ValueError("adaptive implementation changed before authenticated execution")
        if (
            local_json_schema_validator_identity(refresh=True)
            != self._implementation_bundle["local_json_schema_validator"]
        ):
            raise ValueError("local JSON-Schema validator changed before adaptive execution")
        if adaptive_hierarchical_implementation_bundle() != self._implementation_bundle:
            raise ValueError("adaptive dependency bundle changed before authenticated execution")
        inner_precommit = self.authenticated_cache_namespace_sha256
        job_cache.begin_execution(
            hierarchy_inner_precommit_sha256=inner_precommit,
            runner_identity=runner_identity,
        )
        before_remote = tuple(_clone(row) for row in runner.execution_metadata)
        job_records: list[dict[str, Any]] = []
        compiler_records: list[dict[str, Any]] = []

        def run_job(
            job: DiscoveryJsonJob,
            validator: Callable[[Any], Mapping[str, Any]],
        ) -> dict[str, Any]:
            assert_identities()
            if (
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                != self._implementation_file_sha256
            ):
                raise ValueError("adaptive semantic validator changed during execution")
            if adaptive_hierarchical_implementation_bundle() != self._implementation_bundle:
                raise ValueError("adaptive dependency bundle changed during execution")

            def validate_wire(raw_wire: Any) -> Mapping[str, Any]:
                _validate_local_discovery_wire_schema(job=job, response=raw_wire)
                try:
                    normalized = validator(_clone(raw_wire))
                except (TypeError, ValueError) as exc:
                    raise DiscoverySemanticNormalizationFailure(
                        "schema-valid adaptive response could not be safely normalized"
                    ) from exc
                if not isinstance(normalized, Mapping):
                    raise DiscoverySemanticNormalizationFailure(
                        "adaptive semantic normalizer did not return one JSON object"
                    )
                return normalized

            replay = job_cache.replay_validated(
                job=job,
                hierarchy_inner_precommit_sha256=inner_precommit,
                runner_identity=runner_identity,
                validator_code_sha256=self.implementation_bundle_sha256,
                validator=validate_wire,
            )
            if replay is not None:
                response = _clone(replay.validated_response)
                wire_response = _clone(replay.wire_response)
                response_attempt_trace = _validated_response_attempt_trace(
                    logical_job=job,
                    validated_response_sha256=_sha(response),
                    trace=replay.response_attempt_trace,
                )
                outcome = "authenticated_cache_hit"
                cache_record_sha256 = replay.execution_metadata["record_sha256"]
                remote_record_sha256 = None
                remote_record_sha256s: list[str] = []
            else:
                attempts: list[dict[str, Any]] = []
                remote_record_sha256s = []
                try:
                    initial_wire, initial_metadata = (
                        _run_adaptive_remote_call_with_projection_authentication(
                            runner=runner,
                            job=job,
                            runner_identity_sha256=runner_identity["identity_sha256"],
                        )
                    )
                except Exception as exc:
                    category = getattr(exc, "discovery_response_failure_category", None)
                    prior_content = getattr(exc, "failed_response_content", None)
                    if category not in {
                        STRICT_JSON_PARSE_FAILURE,
                        RAW_TRANSPORT_BUDGET_FAILURE,
                    } or not isinstance(prior_content, str):
                        raise
                    remote_record_sha256s.append(_sha(_clone(runner.execution_metadata[-1])))
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
                    remote_record_sha256s.append(_sha(initial_metadata))
                    try:
                        response = _clone(validate_wire(initial_wire))
                    except DiscoveryWireSchemaValidationFailure:
                        prior_content = canonical_json(initial_wire)
                        attempts.append(
                            _response_attempt_entry(
                                job=job,
                                validation_outcome=LOCAL_JSON_SCHEMA_VALIDATION_FAILURE,
                                raw_response_projection_sha256=_sha(initial_wire),
                            )
                        )
                        failure_category = LOCAL_JSON_SCHEMA_VALIDATION_FAILURE
                    else:
                        wire_response = initial_wire
                        attempts.append(
                            _response_attempt_entry(
                                job=job,
                                validation_outcome=VALIDATED_RESPONSE,
                                raw_response_projection_sha256=_sha(wire_response),
                                normalized_validated_response_sha256=_sha(response),
                            )
                        )

                if attempts[-1]["validation_outcome"] != VALIDATED_RESPONSE:
                    repair_job = _build_response_repair_job(
                        original_job=job,
                        prior_response_content=prior_content,
                        failure_category=failure_category,
                    )
                    if (
                        len(repair_job.rendered_messages_bytes)
                        > self.config.max_rendered_prompt_bytes
                    ):
                        raise ValueError(
                            "adaptive response-repair prompt exceeds its fixed byte guard"
                        )
                    try:
                        repaired_wire, repaired_metadata = (
                            _run_adaptive_remote_call_with_projection_authentication(
                                runner=runner,
                                job=repair_job,
                                runner_identity_sha256=runner_identity["identity_sha256"],
                            )
                        )
                    except Exception as exc:
                        category = getattr(exc, "discovery_response_failure_category", None)
                        if category in {
                            STRICT_JSON_PARSE_FAILURE,
                            RAW_TRANSPORT_BUDGET_FAILURE,
                        }:
                            raise DiscoveryResponseRepairExhausted(
                                "adaptive discovery exhausted its single authenticated "
                                "response repair after transport validation failed"
                            ) from exc
                        raise
                    remote_record_sha256s.append(_sha(repaired_metadata))
                    try:
                        response = _clone(validate_wire(repaired_wire))
                    except DiscoveryWireSchemaValidationFailure as exc:
                        raise DiscoveryResponseRepairExhausted(
                            "adaptive discovery exhausted its single authenticated response "
                            "repair after local JSON-Schema validation failed"
                        ) from exc
                    wire_response = repaired_wire
                    attempts.append(
                        _response_attempt_entry(
                            job=repair_job,
                            validation_outcome=VALIDATED_RESPONSE,
                            raw_response_projection_sha256=_sha(wire_response),
                            normalized_validated_response_sha256=_sha(response),
                        )
                    )
                response_attempt_trace = _response_attempt_trace(
                    logical_job=job,
                    attempts=attempts,
                )
                response_attempt_trace = _validated_response_attempt_trace(
                    logical_job=job,
                    validated_response_sha256=_sha(response),
                    trace=response_attempt_trace,
                )
                job_cache.store_validated(
                    job=job,
                    hierarchy_inner_precommit_sha256=inner_precommit,
                    runner_identity=runner_identity,
                    validator_code_sha256=self.implementation_bundle_sha256,
                    wire_response=wire_response,
                    validated_response=response,
                    response_attempt_trace=response_attempt_trace,
                )
                outcome = "remote_validated_and_cached"
                cache_record_sha256 = None
                remote_record_sha256 = remote_record_sha256s[-1]
            assert_identities()
            job_records.append(
                {
                    "job_id": job.job_id,
                    "job_kind": job.job_kind,
                    "job_sha256": _sha(job.as_dict()),
                    "wire_response_sha256": _sha(wire_response),
                    "normalized_validated_response_sha256": _sha(response),
                    "validated_response_sha256": _sha(response),
                    "outcome": outcome,
                    "remote_record_sha256": remote_record_sha256,
                    "remote_record_sha256s": remote_record_sha256s,
                    "cache_record_sha256": cache_record_sha256,
                    "response_attempt_trace_sha256": _sha(response_attempt_trace),
                }
            )
            return response

        interpretation_responses: dict[str, dict[str, Any]] = {}
        for job in self.interpret_jobs:
            interpretation_responses[job.job_id] = run_job(
                job,
                lambda raw, job=job: self.validate_interpretation_job_response(
                    job=job,
                    response=raw,
                ),
            )
        interpretation_responses = self.validate_interpretation_responses(interpretation_responses)

        family_consolidations, consolidation_compiler_records = (
            self._execute_phased_family_consolidations(
                interpretation_responses=interpretation_responses,
                run_job=run_job,
            )
        )
        compiler_records.extend(consolidation_compiler_records)
        chunk_coverages, coverage_compiler_records = self._execute_phased_chunk_coverage(
            interpretation_responses=interpretation_responses,
            family_consolidations=family_consolidations,
            run_job=run_job,
        )
        compiler_records.extend(coverage_compiler_records)
        dossiers = self._compile_phased_dossiers(
            interpretation_responses=interpretation_responses,
            family_consolidations=family_consolidations,
            chunk_coverages=chunk_coverages,
        )

        (
            planner_response,
            lookback,
            planner_compilation_id,
            planner_compiler_record,
        ) = self._execute_phased_adaptive_planner(
            dossiers=dossiers,
            run_job=run_job,
        )
        compiler_records.append(planner_compiler_record)
        (
            proposer_response,
            proposer_compilation_id,
            proposer_compiler_record,
        ) = self._execute_phased_adaptive_proposer(
            dossiers=dossiers,
            planner_response=planner_response,
            lookback=lookback,
            planner_compilation_id=planner_compilation_id,
            planner_dependency_ids=(
                *planner_compiler_record["page_job_ids"],
                *planner_compiler_record["fold_job_ids"],
            ),
            run_job=run_job,
        )
        compiler_records.append(proposer_compiler_record)
        frozen_round = self._freeze_phased_round(
            dossiers=dossiers,
            planner_response=planner_response,
            planner_compilation_id=planner_compilation_id,
            planner_compiler_record=planner_compiler_record,
            lookback=lookback,
            proposer_response=proposer_response,
            proposer_compilation_id=proposer_compilation_id,
            proposer_compiler_record=proposer_compiler_record,
        )

        phased_definition_artifacts, extraction_compiler_record = (
            self._execute_phased_extraction_definitions(
                frozen_round=frozen_round,
                lookback=lookback,
                proposer_compilation_id=proposer_compilation_id,
                proposer_dependency_ids=proposer_compiler_record["all_job_ids"],
                run_job=run_job,
            )
        )
        compiler_records.append(extraction_compiler_record)
        executable_revision = self.freeze_executable_revision(
            current_specs=current_specs,
            frozen_round=frozen_round,
            lookback=lookback,
            definition_responses={},
            max_contracts=max_contracts,
            phased_definition_artifacts=phased_definition_artifacts,
        )

        if (
            local_json_schema_validator_identity(refresh=True)
            != self._implementation_bundle["local_json_schema_validator"]
        ):
            raise ValueError("local JSON-Schema validator changed during adaptive execution")
        if adaptive_hierarchical_implementation_bundle() != self._implementation_bundle:
            raise ValueError("adaptive dependency bundle changed during authenticated execution")

        after_remote = tuple(_clone(row) for row in runner.execution_metadata)
        if after_remote[: len(before_remote)] != before_remote:
            raise ValueError("runner metadata prefix changed during adaptive execution")
        remote_delta = after_remote[len(before_remote) :]
        counts = {
            kind: sum(record["job_kind"] == kind for record in job_records)
            for kind in dict.fromkeys(record["job_kind"] for record in job_records)
        }
        execution_audit = {
            "schema_version": ADAPTIVE_AUTHENTICATED_EXECUTION_VERSION,
            "cache_namespace_sha256": inner_precommit,
            "offline_contract_sha256": self.offline_contract_sha256,
            "cache_namespace_excludes_retry_varying_registry_and_diagnostics": True,
            "approved_adaptive_identity_sha256": _sha(expected_adaptive_identity),
            "implementation_bundle_sha256": self.implementation_bundle_sha256,
            "family_explanations_sha256": _sha(self.family_explanations),
            "catalog_sha256": self.catalog.catalog_sha256,
            "chunk_plan_sha256": self.chunk_plan.plan_sha256,
            "exact_spent_authentication_sha256": (
                self.exact_spent_authentication.authentication_sha256
            ),
            "runner_identity_sha256": runner_identity["identity_sha256"],
            "cache_identity_sha256": cache_identity["identity_sha256"],
            "same_initially_approved_transport_identity": True,
            "same_initially_approved_cache_identity": True,
            "job_records": job_records,
            "compiler_records": compiler_records,
            "compiler_records_sha256": _sha(compiler_records),
            "job_count_by_kind": counts,
            "remote_execution_record_count": len(remote_delta),
            "remote_execution_records_sha256": _sha(list(remote_delta)),
            "cache_execution_records": list(job_cache.execution_metadata),
            "cache_execution_records_sha256": _sha(list(job_cache.execution_metadata)),
            "all_jobs_through_authenticated_cache_boundary": True,
            "all_candidate_pairs_and_coverage_pages_compiled_without_truncation": True,
            "planner_and_proposer_lossless_paging_complete": True,
            "every_planner_page_and_revision_proposal_has_explicit_disposition": True,
            "semantic_target_evidence_or_operation_truncation_applied": False,
            "arbitrary_count_production_gate_open": True,
            "interpret_then_consolidate_then_coverage_then_planner_then_proposer": True,
            "exactly_ten_compact_dossiers": len(dossiers) == 10,
            "definition_jobs_after_proposal_freeze": True,
            "extraction_support_lossless_paging_and_folds_complete": True,
            "complete_support_single_extraction_prompt_present": False,
            "executable_revision_frozen_before_next_gate": True,
            "raw_responses_persisted_in_execution_audit": False,
            "raw_reasoning_persisted_in_execution_audit": False,
        }
        identity = {
            "schema_version": ADAPTIVE_AUTHENTICATED_EXECUTION_VERSION,
            "freeze_sha256": frozen_round.freeze_sha256,
            "executable_freeze_sha256": executable_revision.executable_freeze_sha256,
            "dossier_sha256s": [item.dossier_sha256 for item in dossiers],
            "lookback_sha256": lookback.lookback_sha256,
            "runner_identity_sha256": runner_identity["identity_sha256"],
            "cache_identity_sha256": cache_identity["identity_sha256"],
            "audit": execution_audit,
        }
        return ExecutedAdaptiveReconsiderationRound(
            frozen_round=frozen_round,
            executable_revision=executable_revision,
            dossiers=dossiers,
            lookback=lookback,
            runner_identity_sha256=runner_identity["identity_sha256"],
            cache_identity_sha256=cache_identity["identity_sha256"],
            execution_sha256=_sha(identity),
            _audit_json=canonical_json(execution_audit),
        )


__all__ = [
    "ADAPTIVE_AUTHENTICATED_EXECUTION_VERSION",
    "ADAPTIVE_CHUNK_COVERAGE_VERSION",
    "ADAPTIVE_DOSSIER_VERSION",
    "ADAPTIVE_EXECUTABLE_BRIDGE_VERSION",
    "ADAPTIVE_FAMILY_CONSOLIDATION_VERSION",
    "ADAPTIVE_HIERARCHY_VERSION",
    "ADAPTIVE_IMPLEMENTATION_BUNDLE_VERSION",
    "ADAPTIVE_PLANNER_INTERFACE_VERSION",
    "ADAPTIVE_PROPOSER_INTERFACE_VERSION",
    "ADAPTIVE_PROMPT_CONTRACT_VERSION",
    "ADAPTIVE_ROUND_FREEZE_VERSION",
    "EXACT_SPENT_CATALOG_AUTHENTICATION_VERSION",
    "NEW_MISSING_CONSTRUCT",
    "AdaptiveArchitectureDossier",
    "AdaptiveChunkCoverage",
    "AdaptiveCoverageRequiresRevision",
    "AdaptiveCurrentFeature",
    "AdaptiveDiagnostic",
    "AdaptiveFamilyConsolidation",
    "AdaptiveHierarchicalStage1Reconsideration",
    "AdaptiveReconsiderationConfig",
    "ExactSpentCatalogAuthentication",
    "FrozenAdaptiveReconsiderationRound",
    "FrozenAdaptiveExecutableRevision",
    "ExecutedAdaptiveReconsiderationRound",
    "ResolvedAdaptiveLookback",
    "adaptive_hierarchical_implementation_bundle",
    "adaptive_hierarchical_stage1_prompt_contract",
    "adaptive_hierarchical_stage1_reconsideration_identity",
]
