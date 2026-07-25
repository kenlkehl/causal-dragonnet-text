"""Dynamic strict response contracts for hierarchical discovery jobs.

The model-facing hierarchy contains opaque identifiers whose legal values vary
with every authenticated request.  This module derives both a strict JSON
Schema and an ownership-domain contract exclusively from designated request
fields.  It never discovers identifiers by scanning arbitrary prompt text or
evidence content.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import re

from dataclasses import asdict, dataclass, fields as dataclass_fields
from typing import Any, Mapping, Sequence

HIERARCHICAL_DISCOVERY_RESPONSE_CONTRACT_VERSION = (
    "hierarchical_discovery_dynamic_response_contract_v9"
)

HIERARCHICAL_DISCOVERY_WIRE_RESPONSE_BUDGET_VERSION = (
    "hierarchical_discovery_wire_response_budget_v3"
)
HIERARCHY_WIRE_BUDGET_SCHEMA_VERSION = "hierarchy_wire_budget_v1"

HIERARCHICAL_DISCOVERY_EXACT_COVERAGE_REPRESENTATION = (
    "closed_object_keyed_by_authenticated_identifier_v1"
)

HIERARCHICAL_DISCOVERY_MAX_OPAQUE_IDENTIFIER_LENGTH = 128
HIERARCHICAL_DISCOVERY_MAX_GENERATED_NAME_LENGTH = 64
HIERARCHICAL_DISCOVERY_MAX_DESCRIPTION_LENGTH = 128
HIERARCHICAL_DISCOVERY_MAX_REASON_LENGTH = 128
HIERARCHICAL_DISCOVERY_MAX_AMBIGUITY_LENGTH = 128
HIERARCHICAL_DISCOVERY_MAX_TEXT_LENGTH = 128
HIERARCHICAL_DISCOVERY_MAX_GENERATED_LIST_ITEMS = 8
HIERARCHICAL_DISCOVERY_MAX_FEATURE_NAMES_PER_MEMBER = 4
HIERARCHICAL_DISCOVERY_MAX_FINDINGS_PER_ATOMIC_REVIEW = 4
HIERARCHICAL_DISCOVERY_MAX_PAIR_RELATION_PEERS = 7
HIERARCHICAL_DISCOVERY_MAX_DEFINITION_FOLD_MEMBERS = 8
HIERARCHICAL_DISCOVERY_MAX_GROUP_LOOKBACK_IDS = 8
HIERARCHICAL_DISCOVERY_MAX_ADAPTIVE_REVIEW_TARGETS = 4
HIERARCHICAL_DISCOVERY_MAX_ATOMS_PER_INTERPRET_JOB = 2
HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB = 3
HIERARCHICAL_DISCOVERY_MAX_INTERPRET_NAME_LENGTH = 64
HIERARCHICAL_DISCOVERY_MAX_INTERPRET_DESCRIPTION_LENGTH = 96
HIERARCHICAL_DISCOVERY_MAX_INTERPRET_AMBIGUITY_LENGTH = 96
HIERARCHICAL_DISCOVERY_MAX_INTERPRET_REASON_LENGTH = 64
HIERARCHICAL_DISCOVERY_MAX_INTERPRET_CANONICAL_JSON_BYTES = 20_000
HIERARCHICAL_DISCOVERY_MAX_INTERPRET_TRANSPORT_BYTES = 20_000
HIERARCHICAL_DISCOVERY_INTERPRET_TOKEN_BUDGET = 20_000
HIERARCHICAL_DISCOVERY_MAX_TRANSPORT_BYTES = 20_000
HIERARCHICAL_DISCOVERY_GENERATION_TOKEN_BUDGET = 20_000
HIERARCHICAL_DISCOVERY_CONSERVATIVE_UTF8_BYTES_PER_TOKEN = 1


@dataclass(frozen=True)
class HierarchyWireBudget:
    """Exact scientific response/schema and lossless paging bounds.

    There are deliberately no constructor defaults.  Every field changes the
    language accepted from the model, the number of items compiled into a
    lossless page, or the authenticated output-size proof.  Production callers
    must therefore supply and identity-bind the complete object.

    ``legacy`` exists only to keep older non-production callers readable while
    they migrate.  The production hierarchy never calls it.
    """

    max_opaque_identifier_chars: int
    max_generated_name_chars: int
    max_description_chars: int
    max_reason_chars: int
    max_ambiguity_chars: int
    max_free_text_chars: int
    max_generated_list_items: int
    max_feature_names_per_member: int
    max_findings_per_atomic_review: int
    max_pair_relation_peers_per_page: int
    max_definition_fold_inputs: int
    max_group_lookback_ids: int
    max_adaptive_review_targets: int
    max_interpret_atoms_per_job: int
    max_interpret_members_per_job: int
    max_interpret_name_chars: int
    max_interpret_description_chars: int
    max_interpret_ambiguity_chars: int
    max_interpret_reason_chars: int
    max_interpret_canonical_json_bytes: int
    max_interpret_transport_bytes: int
    interpret_generation_token_budget: int
    max_response_transport_bytes: int
    generation_token_budget: int

    def __post_init__(self) -> None:
        for field in dataclass_fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"HierarchyWireBudget.{field.name} must be an integer")
            if value < 1:
                raise ValueError(f"HierarchyWireBudget.{field.name} must be positive")
        if self.max_interpret_name_chars > self.max_generated_name_chars:
            raise ValueError(
                "max_interpret_name_chars cannot exceed max_generated_name_chars"
            )
        if (
            self.max_interpret_canonical_json_bytes
            > self.max_interpret_transport_bytes
        ):
            raise ValueError(
                "max_interpret_canonical_json_bytes cannot exceed "
                "max_interpret_transport_bytes"
            )
        if self.max_interpret_transport_bytes > self.max_response_transport_bytes:
            raise ValueError(
                "max_interpret_transport_bytes cannot exceed "
                "max_response_transport_bytes"
            )
        if (
            self.interpret_generation_token_budget
            > self.generation_token_budget
        ):
            raise ValueError(
                "interpret_generation_token_budget cannot exceed "
                "generation_token_budget"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HierarchyWireBudget":
        row = _mapping(value, label="hierarchy_wire_budget")
        expected = {"budget_version", *(field.name for field in dataclass_fields(cls))}
        actual = set(row)
        if actual != expected:
            raise ValueError(
                "hierarchy_wire_budget keys differ; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        if row["budget_version"] != HIERARCHY_WIRE_BUDGET_SCHEMA_VERSION:
            raise ValueError("hierarchy_wire_budget budget_version is unsupported")
        return cls(
            **{
                field.name: row[field.name]
                for field in dataclass_fields(cls)
            }
        )

    @classmethod
    def legacy(cls) -> "HierarchyWireBudget":
        """Return the frozen pre-v9 bounds for compatibility tests only."""

        return cls(
            max_opaque_identifier_chars=HIERARCHICAL_DISCOVERY_MAX_OPAQUE_IDENTIFIER_LENGTH,
            max_generated_name_chars=HIERARCHICAL_DISCOVERY_MAX_GENERATED_NAME_LENGTH,
            max_description_chars=HIERARCHICAL_DISCOVERY_MAX_DESCRIPTION_LENGTH,
            max_reason_chars=HIERARCHICAL_DISCOVERY_MAX_REASON_LENGTH,
            max_ambiguity_chars=HIERARCHICAL_DISCOVERY_MAX_AMBIGUITY_LENGTH,
            max_free_text_chars=HIERARCHICAL_DISCOVERY_MAX_TEXT_LENGTH,
            max_generated_list_items=HIERARCHICAL_DISCOVERY_MAX_GENERATED_LIST_ITEMS,
            max_feature_names_per_member=(
                HIERARCHICAL_DISCOVERY_MAX_FEATURE_NAMES_PER_MEMBER
            ),
            max_findings_per_atomic_review=(
                HIERARCHICAL_DISCOVERY_MAX_FINDINGS_PER_ATOMIC_REVIEW
            ),
            max_pair_relation_peers_per_page=(
                HIERARCHICAL_DISCOVERY_MAX_PAIR_RELATION_PEERS
            ),
            max_definition_fold_inputs=(
                HIERARCHICAL_DISCOVERY_MAX_DEFINITION_FOLD_MEMBERS
            ),
            max_group_lookback_ids=HIERARCHICAL_DISCOVERY_MAX_GROUP_LOOKBACK_IDS,
            max_adaptive_review_targets=(
                HIERARCHICAL_DISCOVERY_MAX_ADAPTIVE_REVIEW_TARGETS
            ),
            max_interpret_atoms_per_job=(
                HIERARCHICAL_DISCOVERY_MAX_ATOMS_PER_INTERPRET_JOB
            ),
            max_interpret_members_per_job=(
                HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB
            ),
            max_interpret_name_chars=HIERARCHICAL_DISCOVERY_MAX_INTERPRET_NAME_LENGTH,
            max_interpret_description_chars=(
                HIERARCHICAL_DISCOVERY_MAX_INTERPRET_DESCRIPTION_LENGTH
            ),
            max_interpret_ambiguity_chars=(
                HIERARCHICAL_DISCOVERY_MAX_INTERPRET_AMBIGUITY_LENGTH
            ),
            max_interpret_reason_chars=(
                HIERARCHICAL_DISCOVERY_MAX_INTERPRET_REASON_LENGTH
            ),
            max_interpret_canonical_json_bytes=(
                HIERARCHICAL_DISCOVERY_MAX_INTERPRET_CANONICAL_JSON_BYTES
            ),
            max_interpret_transport_bytes=(
                HIERARCHICAL_DISCOVERY_MAX_INTERPRET_TRANSPORT_BYTES
            ),
            interpret_generation_token_budget=(
                HIERARCHICAL_DISCOVERY_INTERPRET_TOKEN_BUDGET
            ),
            max_response_transport_bytes=HIERARCHICAL_DISCOVERY_MAX_TRANSPORT_BYTES,
            generation_token_budget=HIERARCHICAL_DISCOVERY_GENERATION_TOKEN_BUDGET,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget_version": HIERARCHY_WIRE_BUDGET_SCHEMA_VERSION,
            **asdict(self),
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


LEGACY_HIERARCHY_WIRE_BUDGET = HierarchyWireBudget.legacy()
_ACTIVE_HIERARCHY_WIRE_BUDGET: contextvars.ContextVar[
    HierarchyWireBudget | None
] = contextvars.ContextVar("active_hierarchy_wire_budget", default=None)


def _wire_budget() -> HierarchyWireBudget:
    budget = _ACTIVE_HIERARCHY_WIRE_BUDGET.get()
    if budget is None:
        raise RuntimeError("hierarchy response compilation lacks an explicit wire budget")
    return budget

_ABSOLUTE_END_PATTERN = r"(?![\s\S])"
_IDENTIFIER_BODY_PATTERN = r"[a-z][a-z0-9_.:-]*"
_FEATURE_NAME_BODY_PATTERN = r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*"
_IDENTIFIER_PATTERN = rf"^{_IDENTIFIER_BODY_PATTERN}{_ABSOLUTE_END_PATTERN}"
_FEATURE_NAME_PATTERN = rf"^{_FEATURE_NAME_BODY_PATTERN}{_ABSOLUTE_END_PATTERN}"
_OPTIONAL_FEATURE_NAME_PATTERN = rf"^(?:{_FEATURE_NAME_BODY_PATTERN})?{_ABSOLUTE_END_PATTERN}"
_SAFE_MODEL_TEXT_PATTERN = rf"^[^\u0000-\u001f\u007f-\u009f\ud800-\udfff]*{_ABSOLUTE_END_PATTERN}"
_SAFE_NONEMPTY_MODEL_TEXT_PATTERN = (
    r"^[^\u0000-\u001f\u007f-\u009f\ud800-\udfff]*"
    r"[^\s\u0000-\u001f\u007f-\u009f\ud800-\udfff]"
    rf"[^\u0000-\u001f\u007f-\u009f\ud800-\udfff]*{_ABSOLUTE_END_PATTERN}"
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one JSON object")
    return value


def _rows(value: Any, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    return tuple(_mapping(row, label=f"{label}[{index}]") for index, row in enumerate(value))


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be one non-empty string")
    return value


def _identifier(value: Any, *, label: str) -> str:
    result = _string(value, label=label)
    if len(result) > _wire_budget().max_opaque_identifier_chars:
        raise ValueError(f"{label} exceeds the authenticated opaque-identifier length bound")
    if re.fullmatch(_IDENTIFIER_BODY_PATTERN, result) is None:
        raise ValueError(f"{label} must be one lowercase opaque identifier")
    return result


def _feature_name(value: Any, *, label: str) -> str:
    result = _string(value, label=label)
    if len(result) > _wire_budget().max_generated_name_chars:
        raise ValueError(f"{label} exceeds the authenticated feature-name length bound")
    if re.fullmatch(_FEATURE_NAME_BODY_PATTERN, result) is None:
        raise ValueError(f"{label} must be one lower_snake_case feature name")
    return result


def _strings(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    return tuple(_string(row, label=f"{label}[{index}]") for index, row in enumerate(value))


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _primary_ids(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    result = tuple(
        _identifier(value, label=f"{label}[{index}]") for index, value in enumerate(values)
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{label} cannot contain duplicate designated identifiers")
    return result


def _object(
    properties: Mapping[str, Any], *, required: Sequence[str] | None = None
) -> dict[str, Any]:
    names = tuple(properties) if required is None else tuple(required)
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(names),
        "additionalProperties": False,
    }


def _string_value(
    *,
    pattern: str | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> dict[str, Any]:
    if maximum is None:
        maximum = _wire_budget().max_free_text_chars
    result: dict[str, Any] = {"type": "string"}
    if pattern is not None:
        result["pattern"] = pattern
    elif minimum is not None and minimum > 0:
        # ``minLength`` alone admits whitespace-only strings while the local
        # scientific validators intentionally do not.  The shared language
        # also excludes control characters and UTF-16 surrogate code points:
        # they either expand six-fold in JSON or cannot be encoded as UTF-8.
        result["pattern"] = _SAFE_NONEMPTY_MODEL_TEXT_PATTERN
    else:
        result["pattern"] = _SAFE_MODEL_TEXT_PATTERN
    if minimum is not None:
        result["minLength"] = minimum
    result["maxLength"] = maximum
    return result


def _name_value(*, allow_empty: bool = False) -> dict[str, Any]:
    pattern = _OPTIONAL_FEATURE_NAME_PATTERN if allow_empty else _FEATURE_NAME_PATTERN
    return _string_value(
        pattern=pattern,
        maximum=_wire_budget().max_generated_name_chars,
    )


def _description_value(*, minimum: int = 0) -> dict[str, Any]:
    return _string_value(
        minimum=minimum,
        maximum=_wire_budget().max_description_chars,
    )


def _reason_value() -> dict[str, Any]:
    return _string_value(
        minimum=1,
        maximum=_wire_budget().max_reason_chars,
    )


def _ambiguity_value(*, minimum: int = 0) -> dict[str, Any]:
    return _string_value(
        minimum=minimum,
        maximum=_wire_budget().max_ambiguity_chars,
    )


def _choice(values: Sequence[str], *, label: str) -> dict[str, Any]:
    allowed = _unique(tuple(values))
    if not allowed:
        raise ValueError(f"{label} cannot define an empty scalar identifier domain")
    return {"type": "string", "enum": list(allowed)}


def _array(
    items: Mapping[str, Any],
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "array", "items": dict(items), "minItems": minimum}
    if maximum is not None:
        result["maxItems"] = maximum
    return result


def _domain_array(
    values: Sequence[str],
    *,
    minimum: int = 0,
    maximum: int | None = None,
    label: str,
) -> dict[str, Any]:
    allowed = _unique(tuple(values))
    if not allowed:
        if minimum:
            raise ValueError(f"{label} requires values but its identifier domain is empty")
        return _array(_string_value(pattern=_IDENTIFIER_PATTERN), minimum=0, maximum=0)
    return _array(_choice(allowed, label=label), minimum=minimum, maximum=maximum)


def _name_array(*, minimum: int = 0, maximum: int | None = None) -> dict[str, Any]:
    return _array(
        _name_value(),
        minimum=minimum,
        maximum=maximum,
    )


def _closed_schema(properties: Mapping[str, Any]) -> dict[str, Any]:
    return _object(properties)


def _keyed_object(
    values: Sequence[str],
    *,
    value_schemas: Mapping[str, Mapping[str, Any]] | None = None,
    value_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an exact-coverage object keyed by authenticated identifiers.

    Arrays whose items select from an enum cannot enforce exact-once coverage
    without ``uniqueItems``, which the production grammar does not support.
    A closed JSON object can: every identifier is one required property and
    the strict transport parser independently rejects duplicate object keys.
    """

    keys = tuple(values)
    if len(keys) != len(set(keys)):
        raise ValueError("exact-coverage object keys cannot contain duplicates")
    if (value_schemas is None) == (value_schema is None):
        raise ValueError("provide exactly one keyed-object value schema source")
    if value_schemas is not None:
        if set(value_schemas) != set(keys):
            raise ValueError("keyed-object value schemas must exactly match its keys")
        properties = {key: dict(value_schemas[key]) for key in keys}
    else:
        assert value_schema is not None
        properties = {key: dict(value_schema) for key in keys}
    return _object(properties, required=keys)


def _exact_acknowledgements(values: Sequence[str]) -> dict[str, Any]:
    return _keyed_object(
        values,
        value_schema={"type": "boolean", "const": True},
    )


def _contract(
    *,
    job: str,
    domains: Mapping[str, Sequence[str]],
    ownership: Mapping[str, Any],
    response_path_domains: Mapping[str, str],
    generated_name_paths: Sequence[str] = (),
) -> dict[str, Any]:
    wire_budget = _wire_budget()
    return {
        "contract_version": HIERARCHICAL_DISCOVERY_RESPONSE_CONTRACT_VERSION,
        "hierarchy_wire_budget": wire_budget.as_dict(),
        "job": job,
        "derivation_policy": "designated_request_fields_only_no_arbitrary_text_scan_v1",
        "identifier_domains": {
            name: list(_unique(tuple(values))) for name, values in domains.items()
        },
        "ownership": dict(ownership),
        "response_path_domains": dict(response_path_domains),
        "generated_lower_snake_case_paths": list(generated_name_paths),
        "domain_names_are_not_response_identifiers": True,
        "exact_coverage_representation": (HIERARCHICAL_DISCOVERY_EXACT_COVERAGE_REPRESENTATION),
    }


def _canonical_json_byte_count(value: Any) -> int:
    return len(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _maximum_schema_json_bytes(schema: Mapping[str, Any], *, path: str = "$") -> int:
    """Return the exact maximum canonical UTF-8 size admitted by a closed schema.

    This deliberately supports only the finite JSON-Schema language emitted by
    this module.  A newly introduced unbounded or unfamiliar construct fails
    closed instead of silently weakening the output-budget proof.
    """

    node = _mapping(schema, label=f"response schema at {path}")
    if "const" in node:
        return _canonical_json_byte_count(node["const"])
    if "enum" in node:
        values = node["enum"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"response schema enum at {path} must be non-empty")
        return max(_canonical_json_byte_count(value) for value in values)
    if "anyOf" in node:
        variants = node["anyOf"]
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"response schema anyOf at {path} must be non-empty")
        return max(
            _maximum_schema_json_bytes(variant, path=f"{path}.anyOf[{index}]")
            for index, variant in enumerate(variants)
        )

    kind = node.get("type")
    if kind == "object":
        if node.get("additionalProperties") is not False:
            raise ValueError(f"response object at {path} is not closed")
        properties = _mapping(node.get("properties"), label=f"properties at {path}")
        property_sizes = [
            _canonical_json_byte_count(name)
            + 1
            + _maximum_schema_json_bytes(child, path=f"{path}.{name}")
            for name, child in properties.items()
        ]
        return 2 + sum(property_sizes) + max(0, len(property_sizes) - 1)
    if kind == "array":
        maximum = node.get("maxItems")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
            raise ValueError(f"response array at {path} lacks a finite maxItems")
        item_size = _maximum_schema_json_bytes(
            _mapping(node.get("items"), label=f"items at {path}"),
            path=f"{path}[]",
        )
        return 2 + maximum * item_size + max(0, maximum - 1)
    if kind == "string":
        maximum = node.get("maxLength")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
            raise ValueError(f"response string at {path} lacks a finite maxLength")
        pattern = node.get("pattern")
        if pattern in {_SAFE_MODEL_TEXT_PATTERN, _SAFE_NONEMPTY_MODEL_TEXT_PATTERN}:
            # U+10FFFF is admitted and occupies four UTF-8 bytes.  JSON quote
            # and backslash escaping occupies only two, so four is tight.
            return 2 + 4 * maximum
        if pattern in {
            _IDENTIFIER_PATTERN,
            _FEATURE_NAME_PATTERN,
            _OPTIONAL_FEATURE_NAME_PATTERN,
        }:
            return 2 + maximum
        raise ValueError(f"response string at {path} uses an unaudited pattern")
    if kind == "boolean":
        return 5  # ``false`` is the longer boolean token.
    raise ValueError(f"response schema at {path} uses an unaudited finite type")


def _wire_response_budget(schema: Mapping[str, Any]) -> dict[str, Any]:
    configured = _wire_budget()
    maximum_canonical_json_bytes = _maximum_schema_json_bytes(schema)
    if maximum_canonical_json_bytes > configured.max_response_transport_bytes:
        raise ValueError(
            "hierarchical discovery response schema exceeds the authenticated "
            "configured wire budget; compile bounded pages before creating this job "
            f"(exact canonical maximum={maximum_canonical_json_bytes})"
        )
    maximum_estimated_tokens = (
        maximum_canonical_json_bytes + HIERARCHICAL_DISCOVERY_CONSERVATIVE_UTF8_BYTES_PER_TOKEN - 1
    ) // HIERARCHICAL_DISCOVERY_CONSERVATIVE_UTF8_BYTES_PER_TOKEN
    if maximum_estimated_tokens > configured.generation_token_budget:
        raise ValueError(
            "hierarchical discovery response schema exceeds the authenticated "
            "generation-token budget; increase the configured budget or compile "
            "bounded lossless pages"
        )
    return {
        "budget_contract_version": HIERARCHICAL_DISCOVERY_WIRE_RESPONSE_BUDGET_VERSION,
        "maximum_canonical_json_bytes": maximum_canonical_json_bytes,
        "canonical_json_byte_proof": ("closed_json_schema_exact_structural_utf8_upper_bound_v2"),
        "maximum_transport_bytes": configured.max_response_transport_bytes,
        "transport_byte_policy": "raw_utf8_response_before_json_parsing_v1",
        "conservative_utf8_bytes_per_estimated_token": (
            HIERARCHICAL_DISCOVERY_CONSERVATIVE_UTF8_BYTES_PER_TOKEN
        ),
        "maximum_estimated_tokens": maximum_estimated_tokens,
        "generation_token_budget": configured.generation_token_budget,
    }


def _interpret_wire_budget(
    *,
    evidence_ids: Sequence[str],
    members_by_evidence: Mapping[str, Sequence[str]],
) -> dict[str, int]:
    """Calculate a true UTF-8 upper bound for the admitted interpret wire.

    Every unconstrained model-text code point is admitted only from the shared
    control/surrogate-free string language.  Four UTF-8 bytes is therefore a
    tight per-code-point upper bound; JSON quote/backslash escaping costs only
    two bytes and cannot exceed it.  Generated feature names are ASCII by
    grammar.  Request-owned object keys are included at their exact byte size.

    The token proof intentionally assumes only that one output token cannot
    encode less than one byte.  This avoids trusting an unauthenticated model
    tokenizer or an empirical bytes/token ratio.
    """

    configured = _wire_budget()
    maximum_utf8_code_point = "\U0010ffff"
    finding = {
        "feature_name": "f" * configured.max_interpret_name_chars,
        "description": (
            maximum_utf8_code_point * configured.max_interpret_description_chars
        ),
        "value_shape_hypothesis": "categorical",
        "unresolved_ambiguity": (
            maximum_utf8_code_point * configured.max_interpret_ambiguity_chars
        ),
    }
    repeated_findings = [
        finding for _ in range(configured.max_findings_per_atomic_review)
    ]
    dispositions = {
        evidence_id: {
            "evidence_findings": repeated_findings,
            "member_dispositions": {
                member_id: {"findings": repeated_findings}
                for member_id in members_by_evidence[evidence_id]
            },
            "reason": (
                maximum_utf8_code_point * configured.max_interpret_reason_chars
            ),
        }
        for evidence_id in evidence_ids
    }
    byte_count = len(
        json.dumps(
            {"evidence_dispositions": dispositions},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    estimated_tokens = (
        byte_count + HIERARCHICAL_DISCOVERY_CONSERVATIVE_UTF8_BYTES_PER_TOKEN - 1
    ) // HIERARCHICAL_DISCOVERY_CONSERVATIVE_UTF8_BYTES_PER_TOKEN
    return {
        "atomic_review_count": len(evidence_ids)
        + sum(len(values) for values in members_by_evidence.values()),
        "maximum_findings_per_atomic_review": (
            configured.max_findings_per_atomic_review
        ),
        "maximum_findings": configured.max_findings_per_atomic_review
        * (len(evidence_ids) + sum(len(values) for values in members_by_evidence.values())),
        "maximum_canonical_json_bytes": byte_count,
        "canonical_json_byte_proof": (
            "exact_structure_plus_four_utf8_bytes_per_safe_free_text_code_point_v1"
        ),
        "maximum_transport_bytes": configured.max_interpret_transport_bytes,
        "transport_byte_policy": "raw_utf8_response_before_json_parsing_v1",
        "conservative_utf8_bytes_per_estimated_token": (
            HIERARCHICAL_DISCOVERY_CONSERVATIVE_UTF8_BYTES_PER_TOKEN
        ),
        "maximum_estimated_tokens": estimated_tokens,
        "generation_token_budget": configured.interpret_generation_token_budget,
    }


def _interpret(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    configured = _wire_budget()
    evidence = _rows(request.get("evidence"), label="evidence")
    evidence_ids = _primary_ids(
        tuple(
            _string(row.get("evidence_id"), label=f"evidence[{index}].evidence_id")
            for index, row in enumerate(evidence)
        ),
        label="interpret evidence",
    )
    if not evidence_ids:
        raise ValueError("interpret evidence cannot be empty")
    members_by_evidence = {
        evidence_id: list(
            _primary_ids(
                _strings(row.get("member_ids"), label=f"evidence[{index}].member_ids"),
                label=f"evidence[{index}] member identifiers",
            )
        )
        for index, (evidence_id, row) in enumerate(zip(evidence_ids, evidence))
    }
    member_ids = _unique(
        tuple(member for values in members_by_evidence.values() for member in values)
    )
    if len(evidence_ids) > configured.max_interpret_atoms_per_job:
        raise ValueError(
            "interpret request exceeds its response-budget atom bound; split the "
            "lossless architecture chunk"
        )
    total_member_reviews = sum(len(values) for values in members_by_evidence.values())
    if total_member_reviews > configured.max_interpret_members_per_job:
        raise ValueError(
            "interpret request exceeds its response-budget member bound; split the "
            "lossless architecture chunk"
        )
    finding = _object(
        {
            "feature_name": _string_value(
                pattern=_FEATURE_NAME_PATTERN,
                maximum=configured.max_interpret_name_chars,
            ),
            "description": _string_value(
                minimum=1,
                maximum=configured.max_interpret_description_chars,
            ),
            "value_shape_hypothesis": {
                "type": "string",
                "enum": ["continuous", "categorical", "ambiguous"],
            },
            "unresolved_ambiguity": _string_value(
                maximum=configured.max_interpret_ambiguity_chars,
            ),
        }
    )
    # Evidence-level findings cover whole-atom and zero-member concepts that
    # cannot honestly be assigned to a member. Four proposals retain the prior
    # per-member recall allowance; lossless chunking and compact descriptor
    # limits keep the wire response below the production output budget.
    findings = _array(
        finding,
        maximum=configured.max_findings_per_atomic_review,
    )
    disposition_by_evidence: dict[str, dict[str, Any]] = {}
    for evidence_id in evidence_ids:
        owned_member_ids = tuple(members_by_evidence[evidence_id])
        if owned_member_ids:
            member_dispositions: dict[str, Any] = _keyed_object(
                owned_member_ids,
                value_schema=_object(
                    {
                        "findings": findings,
                    }
                ),
            )
        else:
            member_dispositions = _keyed_object((), value_schema=_object({}))
        disposition_by_evidence[evidence_id] = _object(
            {
                "evidence_findings": findings,
                "member_dispositions": member_dispositions,
                "reason": _string_value(
                    minimum=1,
                    maximum=configured.max_interpret_reason_chars,
                ),
            }
        )
    schema = _closed_schema(
        {
            "evidence_dispositions": _keyed_object(
                evidence_ids,
                value_schemas=disposition_by_evidence,
            ),
        }
    )
    response_budget = _interpret_wire_budget(
        evidence_ids=evidence_ids,
        members_by_evidence=members_by_evidence,
    )
    if (
        response_budget["maximum_canonical_json_bytes"]
        > configured.max_interpret_canonical_json_bytes
        or response_budget["maximum_transport_bytes"]
        > configured.max_interpret_transport_bytes
        or response_budget["maximum_estimated_tokens"]
        > configured.interpret_generation_token_budget
    ):
        raise ValueError(
            "interpret request exceeds its authenticated worst-case response budget; "
            "split the lossless architecture chunk"
        )
    ownership = _contract(
        job=str(request["job"]),
        domains={"evidence_ids": evidence_ids, "member_ids": member_ids},
        ownership={
            "member_ids_by_evidence_id": members_by_evidence,
            "response_domain_bounds": {
                "atomic_review_count": response_budget["atomic_review_count"],
                "maximum_findings_per_atomic_review": response_budget[
                    "maximum_findings_per_atomic_review"
                ],
                "maximum_findings": response_budget["maximum_findings"],
            },
        },
        response_path_domains={
            "evidence_dispositions.<evidence_id>": "one required property per evidence_id",
            "evidence_dispositions.<evidence_id>.member_dispositions.<member_id>": "one required property per member_id owned by evidence_id",
            "evidence_dispositions.<evidence_id>.evidence_findings[]": "atomic evidence-level findings owned by that evidence_id; support is compiler-derived",
            "evidence_dispositions.<evidence_id>.member_dispositions.<member_id>.findings[]": "atomic member-level findings owned by that member_id and its evidence_id; support is compiler-derived",
        },
        generated_name_paths=(
            "evidence_dispositions.<evidence_id>.evidence_findings[].feature_name",
            "evidence_dispositions.<evidence_id>.member_dispositions.<member_id>.findings[].feature_name",
        ),
    )
    return schema, ownership


def _consolidation(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = _rows(request.get("candidates"), label="candidates")
    candidate_ids = _primary_ids(
        tuple(
            _string(row.get("candidate_id"), label=f"candidates[{index}].candidate_id")
            for index, row in enumerate(candidates)
        ),
        label="consolidation candidates",
    )
    evidence_by_candidate = {
        candidate_id: list(
            _primary_ids(
                _strings(
                    row.get("supporting_evidence_ids"),
                    label=f"candidates[{index}].supporting_evidence_ids",
                ),
                label=f"candidates[{index}] evidence identifiers",
            )
        )
        for index, (candidate_id, row) in enumerate(zip(candidate_ids, candidates))
    }
    evidence_ids = _unique(
        tuple(value for values in evidence_by_candidate.values() for value in values)
    )
    source_family = _string(request.get("source_family"), label="source_family")
    slots = tuple(f"consolidation_slot_{index:03d}" for index in range(1, len(candidate_ids) + 1))
    if not candidate_ids:
        schema = _closed_schema(
            {
                "candidate_assignments": _keyed_object((), value_schema=_object({})),
                "slot_definitions": _keyed_object((), value_schema=_object({})),
            }
        )
    else:
        assignment = _object(
            {
                "cluster_slot": _choice(slots, label="consolidation cluster slots"),
                "reason": _reason_value(),
            }
        )
        definition = _object(
            {
                "canonical_name": _name_value(),
                "description": _description_value(minimum=1),
                "unresolved_ambiguity": _ambiguity_value(),
            }
        )
        schema = _closed_schema(
            {
                "candidate_assignments": _keyed_object(
                    candidate_ids,
                    value_schema=assignment,
                ),
                "slot_definitions": _keyed_object(
                    slots,
                    value_schema=definition,
                ),
            }
        )
    ownership = _contract(
        job=str(request["job"]),
        domains={
            "candidate_ids": candidate_ids,
            "evidence_ids": evidence_ids,
            "source_families": (source_family,),
            "cluster_slots": slots,
        },
        ownership={
            "evidence_ids_by_candidate_id": evidence_by_candidate,
            "cluster_slots_are_compiler_owned": True,
            "active_slots_are_derived_from_candidate_assignments": True,
        },
        response_path_domains={
            "candidate_assignments.<candidate_id>": "one required property per candidate_id; cluster_slot must be one fixed compiler-owned slot",
            "slot_definitions.<cluster_slot>": "one required definition per fixed slot; unused definitions are retained only in the normalization audit",
        },
        generated_name_paths=("slot_definitions.<cluster_slot>.canonical_name",),
    )
    return schema, ownership


def _candidate_relation_page(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    configured = _wire_budget()
    anchor_candidate_id = _identifier(
        request.get("anchor_candidate_id"), label="anchor_candidate_id"
    )
    peer_candidate_ids = _primary_ids(
        _strings(request.get("peer_candidate_ids"), label="peer_candidate_ids"),
        label="peer candidate identifiers",
    )
    if not peer_candidate_ids:
        raise ValueError("candidate relation page must contain at least one later peer")
    if len(peer_candidate_ids) > configured.max_pair_relation_peers_per_page:
        raise ValueError("candidate relation page exceeds its authenticated peer bound")
    if anchor_candidate_id in peer_candidate_ids:
        raise ValueError("candidate relation page cannot compare an anchor with itself")
    comparison = _object(
        {
            "relation": {
                "type": "string",
                "enum": ["same_construct", "distinct", "uncertain"],
            },
            "reason": _reason_value(),
        }
    )
    schema = _closed_schema(
        {
            "comparisons": _keyed_object(
                peer_candidate_ids,
                value_schema=comparison,
            )
        }
    )
    ownership = _contract(
        job=str(request["job"]),
        domains={
            "anchor_candidate_ids": (anchor_candidate_id,),
            "peer_candidate_ids": peer_candidate_ids,
        },
        ownership={
            "anchor_candidate_id": anchor_candidate_id,
            "every_peer_is_later_in_canonical_candidate_order": True,
            "every_unordered_pair_is_compiler_scheduled_exactly_once": True,
            "response_domain_bounds": {
                "maximum_peer_comparisons": configured.max_pair_relation_peers_per_page,
                "actual_peer_comparisons": len(peer_candidate_ids),
            },
        },
        response_path_domains={
            "comparisons.<peer_candidate_id>": (
                "one required ternary relation for every exact peer_candidate_id"
            )
        },
    )
    return schema, ownership


def _candidate_definition_fold(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    configured = _wire_budget()
    group_id = _identifier(request.get("group_id"), label="group_id")
    member_candidate_ids = _primary_ids(
        _strings(request.get("member_candidate_ids"), label="member_candidate_ids"),
        label="definition-fold member candidate identifiers",
    )
    if not member_candidate_ids:
        raise ValueError("candidate definition fold cannot be empty")
    if len(member_candidate_ids) > configured.max_definition_fold_inputs:
        raise ValueError("candidate definition fold exceeds its authenticated member bound")
    schema = _closed_schema(
        {
            "canonical_name": _name_value(),
            "description": _description_value(minimum=1),
            "unresolved_ambiguity": _ambiguity_value(),
            "reason": _reason_value(),
        }
    )
    ownership = _contract(
        job=str(request["job"]),
        domains={
            "group_ids": (group_id,),
            "member_candidate_ids": member_candidate_ids,
        },
        ownership={
            "group_id": group_id,
            "member_candidate_ids": list(member_candidate_ids),
            "membership_and_support_are_compiler_owned": True,
            "response_domain_bounds": {
                "maximum_fold_members": configured.max_definition_fold_inputs,
                "actual_fold_members": len(member_candidate_ids),
            },
        },
        response_path_domains={
            "canonical_name": "one generated definition for the exact compiler-owned group"
        },
        generated_name_paths=("canonical_name",),
    )
    return schema, ownership


def _coverage(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = _rows(request.get("evidence"), label="evidence")
    evidence_ids = _primary_ids(
        tuple(
            _string(row.get("evidence_id"), label=f"evidence[{index}].evidence_id")
            for index, row in enumerate(evidence)
        ),
        label="coverage evidence",
    )
    consolidation = _mapping(
        request.get("consolidation", request.get("family_consolidation")),
        label="coverage consolidation",
    )
    concepts = _rows(consolidation.get("canonical_concepts"), label="canonical_concepts")
    canonical_names = _primary_ids(
        tuple(
            _string(
                row.get("canonical_name"),
                label=f"canonical_concepts[{index}].canonical_name",
            )
            for index, row in enumerate(concepts)
        ),
        label="coverage canonical names",
    )
    empty_array = _array(_name_value(), maximum=0)
    affected_required = (
        _domain_array(
            canonical_names,
            minimum=1,
            maximum=len(canonical_names),
            label="coverage canonical names",
        )
        if canonical_names
        else empty_array
    )
    support_required = _domain_array(
        evidence_ids,
        minimum=1,
        maximum=len(evidence_ids),
        label="coverage evidence identifiers",
    )
    common = {"reason": _reason_value()}
    finding = {
        "anyOf": [
            _object(
                {
                    "action": {"type": "string", "const": "add_concept"},
                    "affected_canonical_names": empty_array,
                    "proposed_name": _name_value(),
                    "description": _description_value(minimum=1),
                    "supporting_evidence_ids": support_required,
                    **common,
                }
            ),
            _object(
                {
                    "action": {"type": "string", "const": "split_concept"},
                    "affected_canonical_names": affected_required,
                    "proposed_name": _name_value(),
                    "description": _description_value(minimum=1),
                    "supporting_evidence_ids": support_required,
                    **common,
                }
            ),
            _object(
                {
                    "action": {"type": "string", "const": "restore_support"},
                    "affected_canonical_names": affected_required,
                    "proposed_name": {"type": "string", "const": ""},
                    "description": {"type": "string", "const": ""},
                    "supporting_evidence_ids": support_required,
                    **common,
                }
            ),
            _object(
                {
                    "action": {"type": "string", "const": "no_change"},
                    "affected_canonical_names": empty_array,
                    "proposed_name": {"type": "string", "const": ""},
                    "description": {"type": "string", "const": ""},
                    "supporting_evidence_ids": _array(
                        _string_value(pattern=_IDENTIFIER_PATTERN), maximum=0
                    ),
                    **common,
                }
            ),
        ]
    }
    finding_limit = max(1, len(evidence_ids) + len(canonical_names))
    schema = _closed_schema(
        {
            "findings": _array(finding, maximum=finding_limit),
            "reviewed_evidence_ids": _exact_acknowledgements(evidence_ids),
        }
    )
    ownership = _contract(
        job=str(request["job"]),
        domains={"evidence_ids": evidence_ids, "canonical_names": canonical_names},
        ownership={"all_reviewed_evidence_ids": list(evidence_ids)},
        response_path_domains={
            "findings[].affected_canonical_names[]": "canonical_names",
            "findings[].supporting_evidence_ids[]": "evidence_ids",
            "reviewed_evidence_ids.<evidence_id>": "one required true acknowledgement per evidence_id",
        },
        generated_name_paths=("findings[].proposed_name when action adds or splits",),
    )
    return schema, ownership


def _atomic_coverage(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    configured = _wire_budget()
    atomic_review_id = _identifier(request.get("atomic_review_id"), label="atomic_review_id")
    evidence_id = _identifier(request.get("evidence_id"), label="evidence_id")
    canonical_names = tuple(
        _feature_name(value, label=f"canonical_names[{index}]")
        for index, value in enumerate(
            _strings(request.get("canonical_names"), label="canonical_names")
        )
    )
    if len(canonical_names) != len(set(canonical_names)):
        raise ValueError("atomic coverage canonical_names cannot contain duplicates")
    if len(canonical_names) > configured.max_findings_per_atomic_review:
        raise ValueError("atomic coverage page exceeds its canonical-name bound")
    empty_names = _array(_name_value(), maximum=0)
    affected = (
        _domain_array(
            canonical_names,
            minimum=1,
            maximum=len(canonical_names),
            label="atomic coverage canonical names",
        )
        if canonical_names
        else empty_names
    )
    variants = [
        _object(
            {
                "action": {"type": "string", "const": "add_concept"},
                "affected_canonical_names": empty_names,
                "proposed_name": _name_value(),
                "description": _description_value(minimum=1),
                "reason": _reason_value(),
            }
        ),
        _object(
            {
                "action": {"type": "string", "const": "no_change"},
                "affected_canonical_names": empty_names,
                "proposed_name": {"type": "string", "const": ""},
                "description": {"type": "string", "const": ""},
                "reason": _reason_value(),
            }
        ),
    ]
    if canonical_names:
        variants.extend(
            [
                _object(
                    {
                        "action": {"type": "string", "const": "split_concept"},
                        "affected_canonical_names": affected,
                        "proposed_name": _name_value(),
                        "description": _description_value(minimum=1),
                        "reason": _reason_value(),
                    }
                ),
                _object(
                    {
                        "action": {"type": "string", "const": "restore_support"},
                        "affected_canonical_names": affected,
                        "proposed_name": {"type": "string", "const": ""},
                        "description": {"type": "string", "const": ""},
                        "reason": _reason_value(),
                    }
                ),
            ]
        )
    schema = _closed_schema(
        {
            "findings": _array(
                {"anyOf": variants},
                maximum=configured.max_findings_per_atomic_review,
            ),
            "reviewed_atomic_review": {"type": "boolean", "const": True},
        }
    )
    ownership = _contract(
        job=str(request["job"]),
        domains={
            "atomic_review_ids": (atomic_review_id,),
            "evidence_ids": (evidence_id,),
            "canonical_names": canonical_names,
        },
        ownership={
            "atomic_review_id": atomic_review_id,
            "supporting_evidence_id": evidence_id,
            "support_is_compiler_derived": True,
            "response_domain_bounds": {
                "maximum_canonical_names": configured.max_findings_per_atomic_review,
                "maximum_findings": configured.max_findings_per_atomic_review,
            },
        },
        response_path_domains={
            "findings[].affected_canonical_names[]": "canonical_names",
            "reviewed_atomic_review": "exact true acknowledgement for atomic_review_id",
        },
        generated_name_paths=("findings[].proposed_name when action adds or splits",),
    )
    return schema, ownership


def _review_input_ids(request: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    values = _primary_ids(
        _strings(request.get("review_input_ids"), label="review_input_ids"),
        label=label,
    )
    if not values:
        raise ValueError(f"{label} cannot be empty")
    if len(values) > _wire_budget().max_definition_fold_inputs:
        raise ValueError(f"{label} exceeds its authenticated fold-input bound")
    return values


def _integration_evidence_page(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    group_id = _identifier(request.get("group_id"), label="group_id")
    evidence_id = _identifier(request.get("evidence_id"), label="evidence_id")
    common = {
        "measurement_summary": _description_value(minimum=1),
        "unresolved_ambiguity": _ambiguity_value(),
        "reason": _reason_value(),
        "reviewed_evidence": {"type": "boolean", "const": True},
    }
    variants = [
        _object(
            {
                "relationship": {"type": "string", "const": "distinct_measurement"},
                "proposed_distinct_name": _name_value(),
                **common,
            }
        )
    ]
    variants.extend(
        _object(
            {
                "relationship": {"type": "string", "const": relationship},
                "proposed_distinct_name": {"type": "string", "const": ""},
                **common,
            }
        )
        for relationship in ("supports_group", "contradicts_group", "ambiguous")
    )
    schema = {"anyOf": variants}
    ownership = _contract(
        job=str(request["job"]),
        domains={"group_ids": (group_id,), "evidence_ids": (evidence_id,)},
        ownership={
            "group_id": group_id,
            "evidence_id": evidence_id,
            "one_raw_evidence_item_per_review_page": True,
            "complete_support_is_compiler_scheduled_without_sampling": True,
        },
        response_path_domains={
            "reviewed_evidence": "exact true acknowledgement for the one evidence_id"
        },
        generated_name_paths=("proposed_distinct_name when relationship is distinct_measurement",),
    )
    return schema, ownership


def _integration_evidence_fold(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    group_id = _identifier(request.get("group_id"), label="group_id")
    review_input_ids = _review_input_ids(
        request,
        label="integration evidence-fold review inputs",
    )
    disposition = _object(
        {
            "action": {
                "type": "string",
                "enum": [
                    "integrated",
                    "contradiction_preserved",
                    "distinct_measurement_preserved",
                    "ambiguity_preserved",
                ],
            },
            "reason": _reason_value(),
        }
    )
    common = {
        "input_dispositions": _keyed_object(
            review_input_ids,
            value_schema=disposition,
        ),
        "complete_support_reviewed": {"type": "boolean", "const": True},
        "reason": _reason_value(),
    }
    schema = {
        "anyOf": [
            _object(
                {
                    "decision": {"type": "string", "const": "accept"},
                    "canonical_name": _name_value(),
                    "description": _description_value(minimum=1),
                    "unresolved_ambiguity": _ambiguity_value(),
                    **common,
                }
            ),
            _object(
                {
                    "decision": {"type": "string", "const": "reject"},
                    "canonical_name": {"type": "string", "const": ""},
                    "description": {"type": "string", "const": ""},
                    "unresolved_ambiguity": {"type": "string", "const": ""},
                    **common,
                }
            ),
        ]
    }
    ownership = _contract(
        job=str(request["job"]),
        domains={"group_ids": (group_id,), "review_input_ids": review_input_ids},
        ownership={
            "group_id": group_id,
            "review_input_ids": list(review_input_ids),
            "every_fresh_page_or_prior_accumulator_is_acknowledged_exactly_once": True,
            "fold_schedule_and_transitive_raw_support_are_compiler_owned": True,
        },
        response_path_domains={
            "input_dispositions.<review_input_id>": (
                "one required disposition for every exact review_input_id"
            )
        },
        generated_name_paths=("canonical_name when decision is accept",),
    )
    return schema, ownership


def _rejection_evidence_page(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_id = _identifier(request.get("candidate_id"), label="candidate_id")
    evidence_id = _identifier(request.get("evidence_id"), label="evidence_id")
    common = {
        "measurement_summary": _description_value(minimum=1),
        "reason": _reason_value(),
        "reviewed_evidence": {"type": "boolean", "const": True},
    }
    variants = [
        _object(
            {
                "signal": {"type": "string", "const": signal},
                "proposed_name": _name_value(),
                **common,
            }
        )
        for signal in ("supports_restore", "supports_split")
    ]
    variants.extend(
        _object(
            {
                "signal": {"type": "string", "const": signal},
                "proposed_name": {"type": "string", "const": ""},
                **common,
            }
        )
        for signal in ("supports_uphold", "ambiguous")
    )
    schema = {"anyOf": variants}
    ownership = _contract(
        job=str(request["job"]),
        domains={"candidate_ids": (candidate_id,), "evidence_ids": (evidence_id,)},
        ownership={
            "candidate_id": candidate_id,
            "evidence_id": evidence_id,
            "one_raw_evidence_item_per_review_page": True,
            "complete_candidate_support_is_compiler_scheduled_without_sampling": True,
        },
        response_path_domains={
            "reviewed_evidence": "exact true acknowledgement for the one evidence_id"
        },
        generated_name_paths=("proposed_name when signal supports restoration or splitting",),
    )
    return schema, ownership


def _rejection_evidence_fold(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_id = _identifier(request.get("candidate_id"), label="candidate_id")
    review_input_ids = _review_input_ids(
        request,
        label="rejection evidence-fold review inputs",
    )
    disposition = _object(
        {
            "action": {
                "type": "string",
                "enum": ["integrated", "overruled", "ambiguity_preserved"],
            },
            "reason": _reason_value(),
        }
    )
    common = {
        "measurement_summary": _description_value(minimum=1),
        "input_dispositions": _keyed_object(
            review_input_ids,
            value_schema=disposition,
        ),
        "complete_support_reviewed": {"type": "boolean", "const": True},
        "reason": _reason_value(),
    }
    schema = {
        "anyOf": [
            _object(
                {
                    "decision": {"type": "string", "const": "uphold"},
                    "proposed_name": {"type": "string", "const": ""},
                    **common,
                }
            ),
            *(
                _object(
                    {
                        "decision": {"type": "string", "const": decision},
                        "proposed_name": _name_value(),
                        **common,
                    }
                )
                for decision in ("restore", "split")
            ),
        ]
    }
    ownership = _contract(
        job=str(request["job"]),
        domains={"candidate_ids": (candidate_id,), "review_input_ids": review_input_ids},
        ownership={
            "candidate_id": candidate_id,
            "review_input_ids": list(review_input_ids),
            "every_fresh_page_or_prior_accumulator_is_acknowledged_exactly_once": True,
            "complete_candidate_support_is_restored_by_the_compiler": True,
        },
        response_path_domains={
            "input_dispositions.<review_input_id>": (
                "one required disposition for every exact review_input_id"
            )
        },
        generated_name_paths=("proposed_name when decision restores or splits",),
    )
    return schema, ownership


def _extraction_evidence_page(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical_name = _feature_name(request.get("canonical_name"), label="canonical_name")
    evidence_id = _identifier(request.get("evidence_id"), label="evidence_id")
    literal_values = _array(
        _string_value(minimum=1),
        maximum=_wire_budget().max_generated_list_items,
    )
    schema = _closed_schema(
        {
            "measurement_observation": _description_value(minimum=1),
            "shape_observation": {
                "type": "string",
                "enum": ["continuous", "categorical", "ambiguous", "unresolved"],
            },
            "literal_aliases": literal_values,
            "literal_units": literal_values,
            "literal_categories": literal_values,
            "literal_distinctions": literal_values,
            "missing_or_ambiguous": _ambiguity_value(minimum=1),
            "reviewed_evidence": {"type": "boolean", "const": True},
        }
    )
    ownership = _contract(
        job=str(request["job"]),
        domains={"canonical_names": (canonical_name,), "evidence_ids": (evidence_id,)},
        ownership={
            "canonical_name": canonical_name,
            "evidence_id": evidence_id,
            "one_raw_evidence_item_per_review_page": True,
            "literal_vocabulary_must_come_from_that_exact_evidence_item": True,
        },
        response_path_domains={
            "reviewed_evidence": "exact true acknowledgement for the one evidence_id"
        },
    )
    return schema, ownership


def _extraction_representation_schema(value_shape: str) -> dict[str, Any]:
    variants = [
        _object(
            {
                "kind": {"type": "string", "const": "unresolved"},
                "unit": {"type": "string", "const": ""},
                "categories": _array(_string_value(), maximum=0),
            }
        )
    ]
    if value_shape in {"continuous", "ambiguous"}:
        variants.append(
            _object(
                {
                    "kind": {"type": "string", "const": "continuous"},
                    "unit": _string_value(minimum=1),
                    "categories": _array(_string_value(), maximum=0),
                }
            )
        )
    if value_shape in {"categorical", "ambiguous"}:
        variants.append(
            _object(
                {
                    "kind": {"type": "string", "const": "categorical"},
                    "unit": {"type": "string", "const": ""},
                    "categories": _array(
                        _string_value(minimum=1),
                        minimum=2,
                        maximum=_wire_budget().max_generated_list_items,
                    ),
                }
            )
        )
    return {"anyOf": variants}


def _extraction_evidence_fold(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical_name = _feature_name(request.get("canonical_name"), label="canonical_name")
    value_shape = _string(
        request.get("value_shape_hypothesis"),
        label="value_shape_hypothesis",
    )
    if value_shape not in {"continuous", "categorical", "ambiguous"}:
        raise ValueError("value_shape_hypothesis is invalid")
    review_input_ids = _review_input_ids(
        request,
        label="extraction evidence-fold review inputs",
    )
    disposition = _object(
        {
            "action": {
                "type": "string",
                "enum": ["integrated", "not_selected", "conflict_preserved"],
            },
            "reason": _reason_value(),
        }
    )
    schema = _closed_schema(
        {
            "feature_name": {"type": "string", "const": canonical_name},
            "measurement": _string_value(minimum=1),
            "representation": _extraction_representation_schema(value_shape),
            "aliases": _array(
                _string_value(),
                maximum=_wire_budget().max_generated_list_items,
            ),
            "distinguish_from": _array(
                _string_value(),
                maximum=_wire_budget().max_generated_list_items,
            ),
            "missing_or_ambiguous": _ambiguity_value(minimum=1),
            "input_dispositions": _keyed_object(
                review_input_ids,
                value_schema=disposition,
            ),
            "supporting_evidence_reviewed": {"type": "boolean", "const": True},
        }
    )
    ownership = _contract(
        job=str(request["job"]),
        domains={
            "canonical_names": (canonical_name,),
            "review_input_ids": review_input_ids,
        },
        ownership={
            "canonical_name": canonical_name,
            "review_input_ids": list(review_input_ids),
            "every_fresh_page_or_prior_accumulator_is_acknowledged_exactly_once": True,
            "complete_feature_support_is_compiler_owned_and_transitively_reviewed": True,
        },
        response_path_domains={
            "feature_name": "the exact canonical_name",
            "input_dispositions.<review_input_id>": (
                "one required disposition for every exact review_input_id"
            ),
            "supporting_evidence_reviewed": (
                "exact true acknowledgement of the complete transitive page schedule"
            ),
        },
    )
    return schema, ownership


def _base_planner(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    dossiers = _rows(request.get("architecture_dossiers"), label="architecture_dossiers")
    candidates = tuple(
        candidate
        for dossier_index, dossier in enumerate(dossiers)
        for candidate in _rows(
            dossier.get("architecture_candidates"),
            label=f"architecture_dossiers[{dossier_index}].architecture_candidates",
        )
    )
    candidate_ids = _primary_ids(
        tuple(
            _string(
                row.get("candidate_id"),
                label=f"architecture candidate {index}.candidate_id",
            )
            for index, row in enumerate(candidates)
        ),
        label="planner candidates",
    )
    evidence_by_candidate = {
        candidate_id: list(
            _primary_ids(
                _strings(
                    row.get("supporting_evidence_ids"),
                    label=f"architecture candidate {index}.supporting_evidence_ids",
                ),
                label=f"architecture candidate {index} evidence identifiers",
            )
        )
        for index, (candidate_id, row) in enumerate(zip(candidate_ids, candidates))
    }
    evidence_ids = _unique(
        tuple(value for values in evidence_by_candidate.values() for value in values)
    )
    maximum_lookback_ids = request.get("maximum_raw_evidence_lookback_ids")
    if (
        isinstance(maximum_lookback_ids, bool)
        or not isinstance(maximum_lookback_ids, int)
        or maximum_lookback_ids < 0
    ):
        raise ValueError("maximum_raw_evidence_lookback_ids must be a non-negative integer")
    group_slots = tuple(
        f"planner_group_slot_{index:03d}" for index in range(1, len(candidate_ids) + 1)
    )
    lookback_slots = tuple(
        f"planner_lookback_slot_{index:03d}"
        for index in range(1, min(maximum_lookback_ids, len(evidence_ids)) + 1)
    )
    assignment = (
        _object({"group_slot": _choice(group_slots, label="planner group slots")})
        if candidate_ids
        else _object({})
    )
    group_definition = _object({"provisional_name": _name_value(), "reason": _reason_value()})
    lookback_definition = _object(
        {
            "selection": {
                "type": "string",
                "enum": ["unused", *evidence_ids],
            },
            "question": _string_value(minimum=1),
            "reason": _reason_value(),
        }
    )
    schema = _closed_schema(
        {
            "candidate_assignments": _keyed_object(
                candidate_ids,
                value_schema=assignment,
            ),
            "group_slot_definitions": _keyed_object(
                group_slots,
                value_schema=group_definition,
            ),
            "lookback_slot_definitions": _keyed_object(
                lookback_slots,
                value_schema=lookback_definition,
            ),
        }
    )
    ownership = _contract(
        job=str(request["job"]),
        domains={
            "candidate_ids": candidate_ids,
            "evidence_ids": evidence_ids,
            "planner_group_slots": group_slots,
            "planner_lookback_slots": lookback_slots,
        },
        ownership={
            "evidence_ids_by_candidate_id": evidence_by_candidate,
            "maximum_raw_evidence_lookback_ids": maximum_lookback_ids,
            "group_slots_are_compiler_owned": True,
            "lookback_slots_are_compiler_owned": True,
            "duplicate_lookback_selections_are_compiler_deduplicated": True,
        },
        response_path_domains={
            "candidate_assignments.<candidate_id>": "one required compiler-owned group_slot assignment per candidate_id",
            "group_slot_definitions.<planner_group_slot>": "one required definition per fixed group slot; unused definitions are audit-only",
            "lookback_slot_definitions.<planner_lookback_slot>.selection": "unused or one dossier-owned evidence_id; repeated selections are deterministically deduplicated",
        },
        generated_name_paths=("group_slot_definitions.<planner_group_slot>.provisional_name",),
    )
    return schema, ownership


def _base_integration(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    architecture_context = _mapping(
        request.get("architecture_context"), label="architecture_context"
    )
    dossiers = _rows(
        architecture_context.get("architecture_dossiers"), label="architecture_dossiers"
    )
    candidates = tuple(
        candidate
        for dossier_index, dossier in enumerate(dossiers)
        for candidate in _rows(
            dossier.get("architecture_candidates"),
            label=f"architecture_dossiers[{dossier_index}].architecture_candidates",
        )
    )
    candidate_ids = _primary_ids(
        tuple(
            _string(
                row.get("candidate_id"),
                label=f"architecture candidate {index}.candidate_id",
            )
            for index, row in enumerate(candidates)
        ),
        label="integration candidates",
    )
    evidence_by_candidate = {
        candidate_id: list(
            _primary_ids(
                _strings(
                    row.get("supporting_evidence_ids"),
                    label=f"architecture candidate {index}.supporting_evidence_ids",
                ),
                label=f"architecture candidate {index} evidence identifiers",
            )
        )
        for index, (candidate_id, row) in enumerate(zip(candidate_ids, candidates))
    }
    families_by_candidate = {
        candidate_id: list(
            _primary_ids(
                _strings(
                    row.get("source_families"),
                    label=f"architecture candidate {index}.source_families",
                ),
                label=f"architecture candidate {index} source families",
            )
        )
        for index, (candidate_id, row) in enumerate(zip(candidate_ids, candidates))
    }
    evidence_ids = _unique(
        tuple(value for values in evidence_by_candidate.values() for value in values)
    )
    source_families = _unique(
        tuple(value for values in families_by_candidate.values() for value in values)
    )
    maximum_integrated_features = request.get("maximum_integrated_features")
    if (
        isinstance(maximum_integrated_features, bool)
        or not isinstance(maximum_integrated_features, int)
        or maximum_integrated_features < 1
    ):
        raise ValueError("maximum_integrated_features must be a positive integer")
    slots = tuple(
        f"integration_slot_{index:03d}"
        for index in range(
            1,
            min(maximum_integrated_features, len(candidate_ids)) + 1,
        )
    )
    if not candidate_ids:
        schema = _closed_schema(
            {
                "candidate_routes": _keyed_object((), value_schema=_object({})),
                "slot_definitions": _keyed_object((), value_schema=_object({})),
            }
        )
        ownership = _contract(
            job=str(request["job"]),
            domains={
                "candidate_ids": (),
                "evidence_ids": (),
                "source_families": (),
                "integration_slots": (),
            },
            ownership={
                "evidence_ids_by_candidate_id": {},
                "source_families_by_candidate_id": {},
                "maximum_integrated_features": maximum_integrated_features,
            },
            response_path_domains={
                "candidate_routes": "exact empty object because candidate_ids is empty",
                "slot_definitions": "exact empty object because candidate_ids is empty",
            },
        )
        return schema, ownership
    route = _object(
        {
            "route": {
                "type": "string",
                "enum": ["reject", *slots],
            },
            "reason": _reason_value(),
        }
    )
    definition = _object(
        {
            "canonical_name": _name_value(),
            "description": _description_value(minimum=1),
            "unresolved_ambiguity": _ambiguity_value(),
        }
    )
    schema = _closed_schema(
        {
            "candidate_routes": _keyed_object(
                candidate_ids,
                value_schema=route,
            ),
            "slot_definitions": _keyed_object(
                slots,
                value_schema=definition,
            ),
        }
    )
    ownership = _contract(
        job=str(request["job"]),
        domains={
            "candidate_ids": candidate_ids,
            "evidence_ids": evidence_ids,
            "source_families": source_families,
            "integration_slots": slots,
        },
        ownership={
            "evidence_ids_by_candidate_id": evidence_by_candidate,
            "source_families_by_candidate_id": families_by_candidate,
            "maximum_integrated_features": maximum_integrated_features,
            "integration_slots_are_compiler_owned": True,
            "accepted_feature_relations_are_derived_from_candidate_routes": True,
            "extraction_constraints_are_deferred_to_grounded_definition_jobs": True,
        },
        response_path_domains={
            "candidate_routes.<candidate_id>": "one required route per candidate_id; route is reject or one fixed compiler-owned integration slot",
            "slot_definitions.<integration_slot>": "one required definition per fixed slot; unused definitions are retained only in the normalization audit",
        },
        generated_name_paths=("slot_definitions.<integration_slot>.canonical_name",),
    )
    return schema, ownership


def _bounded_group_integration(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    group_id = _identifier(request.get("group_id"), label="group_id")
    common = {"reason": _reason_value()}
    schema = {
        "anyOf": [
            _object(
                {
                    "decision": {"type": "string", "const": "accept"},
                    "canonical_name": _name_value(),
                    "description": _description_value(minimum=1),
                    "unresolved_ambiguity": _ambiguity_value(),
                    **common,
                }
            ),
            _object(
                {
                    "decision": {"type": "string", "const": "reject"},
                    "canonical_name": {"type": "string", "const": ""},
                    "description": {"type": "string", "const": ""},
                    "unresolved_ambiguity": {"type": "string", "const": ""},
                    **common,
                }
            ),
        ]
    }
    ownership = _contract(
        job=str(request["job"]),
        domains={"group_ids": (group_id,)},
        ownership={
            "group_id": group_id,
            "membership_support_provenance_and_shape_are_compiler_owned": True,
            "one_group_per_integration_job": True,
        },
        response_path_domains={
            "decision": "one final decision for the exact compiler-owned provisional group"
        },
        generated_name_paths=("canonical_name when decision is accept",),
    )
    return schema, ownership


def _rejection(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = _rows(request.get("rejected_candidates"), label="rejected_candidates")
    candidate_ids = _primary_ids(
        tuple(
            _string(
                row.get("candidate_id"),
                label=f"rejected_candidates[{index}].candidate_id",
            )
            for index, row in enumerate(candidates)
        ),
        label="rejected candidates",
    )
    if not candidate_ids:
        raise ValueError("rejected candidates cannot be empty")
    evidence_by_candidate = {
        candidate_id: list(
            _primary_ids(
                _strings(
                    row.get("supporting_evidence_ids"),
                    label=f"rejected_candidates[{index}].supporting_evidence_ids",
                ),
                label=f"rejected_candidates[{index}] evidence identifiers",
            )
        )
        for index, (candidate_id, row) in enumerate(zip(candidate_ids, candidates))
    }
    evidence_ids = _unique(
        tuple(value for values in evidence_by_candidate.values() for value in values)
    )
    reconsideration_by_candidate = {}
    for candidate_id in candidate_ids:
        variants = [
            _object(
                {
                    "decision": {"type": "string", "const": "uphold"},
                    "proposed_name": {"type": "string", "const": ""},
                    "supporting_evidence_ids": _array(
                        _string_value(pattern=_IDENTIFIER_PATTERN), maximum=0
                    ),
                    "reason": _reason_value(),
                }
            )
        ]
        if evidence_by_candidate[candidate_id]:
            variants.extend(
                _object(
                    {
                        "decision": {"type": "string", "const": decision},
                        "proposed_name": _name_value(),
                        "supporting_evidence_ids": _domain_array(
                            evidence_by_candidate[candidate_id],
                            minimum=1,
                            maximum=len(evidence_by_candidate[candidate_id]),
                            label=f"rejection evidence owned by {candidate_id}",
                        ),
                        "reason": _reason_value(),
                    }
                )
                for decision in ("restore", "split")
            )
        reconsideration_by_candidate[candidate_id] = {"anyOf": variants}
    schema = _closed_schema(
        {
            "reconsiderations": _keyed_object(
                candidate_ids,
                value_schemas=reconsideration_by_candidate,
            )
        }
    )
    ownership = _contract(
        job=str(request["job"]),
        domains={"candidate_ids": candidate_ids, "evidence_ids": evidence_ids},
        ownership={"evidence_ids_by_candidate_id": evidence_by_candidate},
        response_path_domains={
            "reconsiderations.<candidate_id>": "one required property per candidate_id",
            "reconsiderations.<candidate_id>.supporting_evidence_ids[]": "evidence_ids owned by candidate_id",
        },
        generated_name_paths=(
            "reconsiderations.<candidate_id>.proposed_name when restored or split",
        ),
    )
    return schema, ownership


def _extraction(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical_name = _string(request.get("canonical_name"), label="canonical_name")
    value_shape = _string(request.get("value_shape_hypothesis"), label="value_shape_hypothesis")
    if value_shape not in {"continuous", "categorical", "ambiguous"}:
        raise ValueError("value_shape_hypothesis is invalid")
    evidence_ids = _primary_ids(
        _strings(request.get("supporting_evidence_ids"), label="supporting_evidence_ids"),
        label="extraction evidence",
    )
    representation_variants = [
        _object(
            {
                "kind": {"type": "string", "const": "unresolved"},
                "unit": {"type": "string", "const": ""},
                "categories": _array(_string_value(), maximum=0),
            }
        )
    ]
    if value_shape in {"continuous", "ambiguous"}:
        representation_variants.append(
            _object(
                {
                    "kind": {"type": "string", "const": "continuous"},
                    "unit": _string_value(minimum=1),
                    "categories": _array(_string_value(), maximum=0),
                }
            )
        )
    if value_shape in {"categorical", "ambiguous"}:
        representation_variants.append(
            _object(
                {
                    "kind": {"type": "string", "const": "categorical"},
                    "unit": {"type": "string", "const": ""},
                    "categories": _array(
                        _string_value(minimum=1),
                        minimum=2,
                        maximum=_wire_budget().max_generated_list_items,
                    ),
                }
            )
        )
    representation = {"anyOf": representation_variants}
    schema = _closed_schema(
        {
            "feature_name": {"type": "string", "const": canonical_name},
            "measurement": _string_value(minimum=1),
            "representation": representation,
            "aliases": _array(
                _string_value(),
                maximum=_wire_budget().max_generated_list_items,
            ),
            "distinguish_from": _array(
                _string_value(),
                maximum=_wire_budget().max_generated_list_items,
            ),
            "missing_or_ambiguous": _ambiguity_value(minimum=1),
            "supporting_evidence_reviewed": {"type": "boolean", "const": True},
        }
    )
    ownership = _contract(
        job=str(request["job"]),
        domains={"evidence_ids": evidence_ids, "canonical_names": (canonical_name,)},
        ownership={"model_visible_supporting_evidence_ids": list(evidence_ids)},
        response_path_domains={
            "feature_name": "the exact canonical_name",
            "supporting_evidence_reviewed": (
                "exact true acknowledgement of the bounded visible lookback; the complete "
                "feature-support relation is compiler-derived"
            ),
        },
    )
    return schema, ownership


def _adaptive_planner(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    dossiers = _rows(request.get("architecture_dossiers"), label="architecture_dossiers")
    families = _primary_ids(
        tuple(
            _string(
                row.get("source_family"),
                label=f"architecture_dossiers[{index}].source_family",
            )
            for index, row in enumerate(dossiers)
        ),
        label="adaptive planner source families",
    )
    evidence_by_family = {
        family: list(
            _primary_ids(
                _strings(
                    _mapping(
                        row.get("coverage"),
                        label=f"architecture_dossiers[{index}].coverage",
                    ).get("lookback_evidence_ids"),
                    label=(f"architecture_dossiers[{index}].coverage.lookback_evidence_ids"),
                ),
                label=f"architecture_dossiers[{index}] lookback evidence identifiers",
            )
        )
        for index, (family, row) in enumerate(zip(families, dossiers))
    }
    evidence_ids = _unique(
        tuple(value for values in evidence_by_family.values() for value in values)
    )
    registry = _rows(request.get("current_registry"), label="current_registry")
    registry_names = _primary_ids(
        tuple(
            _string(
                row.get("feature_name"),
                label=f"current_registry[{index}].feature_name",
            )
            for index, row in enumerate(registry)
        ),
        label="adaptive planner registry names",
    )
    lookback_bounds = _mapping(request.get("lookback_bounds"), label="lookback_bounds")
    maximum_ids_per_target = lookback_bounds.get("max_ids_per_target")
    maximum_total_ids = lookback_bounds.get("max_total_ids")
    maximum_total_bytes = lookback_bounds.get("max_total_bytes")
    for label, value in (
        ("lookback_bounds.max_ids_per_target", maximum_ids_per_target),
        ("lookback_bounds.max_total_ids", maximum_total_ids),
        ("lookback_bounds.max_total_bytes", maximum_total_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    assert isinstance(maximum_ids_per_target, int)
    assert isinstance(maximum_total_ids, int)
    if maximum_ids_per_target > maximum_total_ids:
        raise ValueError("adaptive planner per-target bound exceeds its total bound")
    requested_evidence_maximum = min(
        maximum_ids_per_target,
        maximum_total_ids,
        len(evidence_ids),
    )
    target_values = (*registry_names, "new_missing_construct")
    target_properties = {
        "target": _choice(target_values, label="adaptive planner targets"),
        "problem": _string_value(minimum=1),
        "relevant_architectures": _domain_array(
            families,
            minimum=1,
            maximum=len(families),
            label="adaptive planner source families",
        ),
        "reason": _reason_value(),
    }
    target = _object(
        {
            **target_properties,
            "requested_evidence_ids": _domain_array(
                evidence_ids,
                maximum=requested_evidence_maximum,
                label="adaptive planner evidence identifiers",
            ),
        }
    )
    no_lookback_target = _object(
        {
            **target_properties,
            "requested_evidence_ids": _array(
                _string_value(pattern=_IDENTIFIER_PATTERN),
                maximum=0,
            ),
        }
    )
    # The downstream proposer executes an explicitly configured number of
    # operations.  The budget is identity-bound and never silently narrows the
    # request-specific maximum.
    maximum_targets = min(
        _wire_budget().max_adaptive_review_targets,
        max(1, len(registry_names) + len(families)),
    )
    schema = _object(
        {
            "review_targets": _array(target, maximum=maximum_targets),
            "no_lookback_needed": {"type": "boolean"},
        }
    )
    schema["anyOf"] = [
        _object(
            {
                "review_targets": _array(
                    no_lookback_target,
                    maximum=maximum_targets,
                ),
                "no_lookback_needed": {"type": "boolean", "const": True},
            }
        ),
        _object(
            {
                "review_targets": _array(
                    target,
                    minimum=1,
                    maximum=maximum_targets,
                ),
                "no_lookback_needed": {"type": "boolean", "const": False},
            }
        ),
    ]
    ownership = _contract(
        job=str(request["job"]),
        domains={
            "registry_or_reserved_targets": target_values,
            "source_families": families,
            "evidence_ids": evidence_ids,
        },
        ownership={
            "evidence_ids_by_source_family": evidence_by_family,
            "lookback_bounds": {
                "max_ids_per_target": maximum_ids_per_target,
                "max_total_ids": maximum_total_ids,
                "max_total_bytes": maximum_total_bytes,
            },
            "duplicate_or_conflicting_selections_are_compiler_normalized": True,
        },
        response_path_domains={
            "review_targets[].target": "registry_or_reserved_targets",
            "review_targets[].relevant_architectures[]": "source_families",
            "review_targets[].requested_evidence_ids[]": "evidence_ids owned by relevant_architectures",
        },
    )
    return schema, ownership


def _adaptive_proposer(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    dossiers = _rows(request.get("architecture_dossiers"), label="architecture_dossiers")
    families = _primary_ids(
        tuple(
            _string(
                row.get("source_family"),
                label=f"architecture_dossiers[{index}].source_family",
            )
            for index, row in enumerate(dossiers)
        ),
        label="adaptive proposer source families",
    )
    registry = _rows(request.get("current_registry"), label="current_registry")
    registry_names = _primary_ids(
        tuple(
            _string(
                row.get("feature_name"),
                label=f"current_registry[{index}].feature_name",
            )
            for index, row in enumerate(registry)
        ),
        label="adaptive proposer registry names",
    )
    diagnostics = _rows(request.get("diagnostics"), label="diagnostics")
    diagnostic_ids = _primary_ids(
        tuple(
            _string(
                row.get("diagnostic_id"),
                label=f"diagnostics[{index}].diagnostic_id",
            )
            for index, row in enumerate(diagnostics)
        ),
        label="adaptive proposer diagnostics",
    )
    requested = _rows(request.get("requested_evidence"), label="requested_evidence")
    evidence_ids = _primary_ids(
        tuple(
            _string(
                row.get("evidence_id"),
                label=f"requested_evidence[{index}].evidence_id",
            )
            for index, row in enumerate(requested)
        ),
        label="adaptive proposer requested evidence",
    )
    evidence_family = {
        evidence_id: _string(
            row.get("source_family"), label=f"requested_evidence[{index}].source_family"
        )
        for index, (evidence_id, row) in enumerate(zip(evidence_ids, requested))
    }
    unknown_evidence_families = set(evidence_family.values()) - set(families)
    if unknown_evidence_families:
        raise ValueError(
            "requested_evidence cites source families absent from architecture_dossiers: "
            f"{sorted(unknown_evidence_families)}"
        )
    review_plan = _mapping(request.get("review_plan"), label="review_plan")
    review_targets = _rows(review_plan.get("review_targets"), label="review_plan.review_targets")
    planned_targets = tuple(
        _string(row.get("target"), label=f"review_plan.review_targets[{index}].target")
        for index, row in enumerate(review_targets)
    )
    maximum_operations = request.get("maximum_operations")
    if (
        isinstance(maximum_operations, bool)
        or not isinstance(maximum_operations, int)
        or maximum_operations < 1
    ):
        raise ValueError("maximum_operations must be a positive integer")
    if maximum_operations > _wire_budget().max_generated_list_items:
        raise ValueError(
            "maximum_operations exceeds the configured generated-list wire "
            "budget; increase the authenticated budget or compile more rounds"
        )
    proposed_feature = _object(
        {
            "feature_name": _name_value(),
            "description": _description_value(minimum=1),
            "value_shape_hypothesis": {
                "type": "string",
                "enum": ["continuous", "categorical", "ambiguous", "unresolved"],
            },
            "definition_summary": _description_value(minimum=1),
            "source_families": _domain_array(
                families,
                minimum=1,
                maximum=len(families),
                label="adaptive proposer source families",
            ),
        }
    )
    planned_existing_targets = tuple(
        name for name in registry_names if name in set(planned_targets)
    )
    operation_variants: list[dict[str, Any]] = []

    def add_operation_variant(
        *,
        operation: str,
        targets: Mapping[str, Any],
        proposed: Mapping[str, Any],
        support: Mapping[str, Any],
    ) -> None:
        operation_variants.append(
            _object(
                {
                    "operation": {"type": "string", "const": operation},
                    "targets": dict(targets),
                    "proposed_feature": dict(proposed),
                    "supporting_evidence_ids": dict(support),
                    "diagnostic_ids": _domain_array(
                        diagnostic_ids,
                        minimum=1,
                        maximum=len(diagnostic_ids),
                        label="adaptive proposer diagnostic identifiers",
                    ),
                    "reason": _reason_value(),
                }
            )
        )

    if "new_missing_construct" in planned_targets and evidence_ids:
        add_operation_variant(
            operation="add",
            targets=_name_array(minimum=1, maximum=1),
            proposed=proposed_feature,
            support=_domain_array(
                evidence_ids,
                minimum=1,
                maximum=len(evidence_ids),
                label="adaptive proposer evidence identifiers",
            ),
        )
    if planned_existing_targets:
        single_existing_target = _domain_array(
            planned_existing_targets,
            minimum=1,
            maximum=1,
            label="adaptive proposer planned registry targets",
        )
        add_operation_variant(
            operation="drop",
            targets=single_existing_target,
            proposed=_object({}, required=()),
            support={"type": "array", "const": []},
        )
        if evidence_ids:
            evidence_support = _domain_array(
                evidence_ids,
                minimum=1,
                maximum=len(evidence_ids),
                label="adaptive proposer evidence identifiers",
            )
            for operation_name in ("split", "rename", "revise_definition"):
                add_operation_variant(
                    operation=operation_name,
                    targets=single_existing_target,
                    proposed=proposed_feature,
                    support=evidence_support,
                )
            if len(planned_existing_targets) >= 2:
                add_operation_variant(
                    operation="merge",
                    targets=_domain_array(
                        planned_existing_targets,
                        minimum=2,
                        maximum=len(planned_existing_targets),
                        label="adaptive proposer planned registry targets",
                    ),
                    proposed=proposed_feature,
                    support=evidence_support,
                )
    operations_schema: dict[str, Any]
    if operation_variants:
        operations_schema = _array(
            {"anyOf": operation_variants},
            minimum=1,
            maximum=maximum_operations,
        )
        schema = _object(
            {
                "operations": _array(
                    {"anyOf": operation_variants},
                    maximum=maximum_operations,
                ),
                "converged": {"type": "boolean"},
            }
        )
        schema["anyOf"] = [
            _object(
                {
                    "operations": _array(
                        {"anyOf": operation_variants},
                        maximum=0,
                    ),
                    "converged": {"type": "boolean", "const": True},
                }
            ),
            _object(
                {
                    "operations": operations_schema,
                    "converged": {"type": "boolean", "const": False},
                }
            ),
        ]
    else:
        operations_schema = _array(_object({}, required=()), maximum=0)
        schema = _closed_schema(
            {
                "operations": operations_schema,
                "converged": {"type": "boolean", "const": True},
            }
        )
    ownership = _contract(
        job=str(request["job"]),
        domains={
            "registry_feature_names": registry_names,
            "planned_targets": planned_targets,
            "source_families": families,
            "evidence_ids": evidence_ids,
            "diagnostic_ids": diagnostic_ids,
        },
        ownership={
            "source_family_by_evidence_id": evidence_family,
            "maximum_operations": maximum_operations,
            "duplicate_or_conflicting_operations_are_compiler_normalized": True,
        },
        response_path_domains={
            "operations[].targets[]": "planned current registry names, or the generated added feature name for add",
            "operations[].proposed_feature.source_families[]": "source_families matching cited evidence or preserved targets",
            "operations[].supporting_evidence_ids[]": "evidence_ids from the exact bounded lookback",
            "operations[].diagnostic_ids[]": "diagnostic_ids",
        },
        generated_name_paths=("operations[].proposed_feature.feature_name",),
    )
    return schema, ownership


def _build_hierarchical_discovery_response_contract_with_active_budget(
    *,
    job_kind: str,
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = _mapping(request, label="discovery request")
    job = _string(row.get("job"), label="discovery request job")
    dispatch = {
        "interpret_evidence_chunk": _interpret,
        "consolidate_candidate_ledger": _consolidation,
        "consolidate_adaptive_architecture_candidates": _consolidation,
        "compare_consolidation_candidate_relations": _candidate_relation_page,
        "compare_adaptive_candidate_relations": _candidate_relation_page,
        "compare_cross_architecture_candidate_relations": _candidate_relation_page,
        "fold_consolidation_group_definition": _candidate_definition_fold,
        "fold_adaptive_group_definition": _candidate_definition_fold,
        "fold_cross_architecture_group_definition": _candidate_definition_fold,
        "audit_architecture_chunk_coverage": _coverage,
        "audit_adaptive_architecture_coverage": _coverage,
        "audit_architecture_atomic_coverage": _atomic_coverage,
        "audit_adaptive_atomic_coverage": _atomic_coverage,
        "plan_cross_architecture_integration": _base_planner,
        "integrate_cross_architecture_candidates": _base_integration,
        "integrate_cross_architecture_group": _bounded_group_integration,
        "review_integration_group_evidence": _integration_evidence_page,
        "fold_integration_group_evidence_reviews": _integration_evidence_fold,
        "audit_every_rejected_candidate": _rejection,
        "review_rejection_candidate_evidence": _rejection_evidence_page,
        "fold_rejection_candidate_evidence_reviews": _rejection_evidence_fold,
        "define_one_extraction_feature": _extraction,
        "review_extraction_feature_evidence": _extraction_evidence_page,
        "fold_extraction_evidence_definitions": _extraction_evidence_fold,
        "plan_adaptive_stage1_reconsideration": _adaptive_planner,
        "propose_adaptive_registry_revision": _adaptive_proposer,
    }
    builder = dispatch.get(job)
    if builder is None:
        raise ValueError(f"unsupported hierarchical discovery request job: {job!r}")
    expected_kind_by_job = {
        "interpret_evidence_chunk": "interpret_architecture_chunk",
        "consolidate_candidate_ledger": "consolidate_architecture_candidates",
        "consolidate_adaptive_architecture_candidates": "consolidate_architecture_candidates",
        "compare_consolidation_candidate_relations": "consolidate_architecture_candidates",
        "compare_adaptive_candidate_relations": "consolidate_architecture_candidates",
        "compare_cross_architecture_candidate_relations": ("plan_cross_architecture_integration"),
        "fold_consolidation_group_definition": "consolidate_architecture_candidates",
        "fold_adaptive_group_definition": "consolidate_architecture_candidates",
        "fold_cross_architecture_group_definition": "plan_cross_architecture_integration",
        "audit_architecture_chunk_coverage": "audit_architecture_coverage",
        "audit_adaptive_architecture_coverage": "audit_architecture_coverage",
        "audit_architecture_atomic_coverage": "audit_architecture_coverage",
        "audit_adaptive_atomic_coverage": "audit_architecture_coverage",
        "plan_cross_architecture_integration": "plan_cross_architecture_integration",
        "integrate_cross_architecture_candidates": "integrate_cross_architecture_candidates",
        "integrate_cross_architecture_group": "integrate_cross_architecture_candidates",
        "review_integration_group_evidence": "integrate_cross_architecture_candidates",
        "fold_integration_group_evidence_reviews": ("integrate_cross_architecture_candidates"),
        "audit_every_rejected_candidate": "audit_rejected_candidates",
        "review_rejection_candidate_evidence": "audit_rejected_candidates",
        "fold_rejection_candidate_evidence_reviews": "audit_rejected_candidates",
        "define_one_extraction_feature": "define_one_extraction_feature",
        "review_extraction_feature_evidence": "define_one_extraction_feature",
        "fold_extraction_evidence_definitions": "define_one_extraction_feature",
        "plan_adaptive_stage1_reconsideration": "plan_cross_architecture_integration",
        "propose_adaptive_registry_revision": "integrate_cross_architecture_candidates",
    }
    if job_kind != expected_kind_by_job[job]:
        raise ValueError(
            f"request job {job!r} is incompatible with discovery job kind {job_kind!r}"
        )
    schema, ownership = builder(row)
    ownership_root = _mapping(ownership.get("ownership"), label="identifier ownership relations")
    if "wire_response_budget" in ownership_root:
        raise ValueError("response builder attempted to override the common wire budget")
    ownership_root["wire_response_budget"] = _wire_response_budget(schema)
    return dict(schema), dict(ownership)


def _resolved_wire_budget(
    *,
    request: Mapping[str, Any],
    wire_budget: HierarchyWireBudget | None,
    allow_legacy_default: bool,
) -> HierarchyWireBudget:
    if wire_budget is not None and not isinstance(wire_budget, HierarchyWireBudget):
        raise TypeError("wire_budget must be a HierarchyWireBudget")
    embedded_raw = request.get("hierarchy_wire_budget")
    embedded = (
        None
        if embedded_raw is None
        else HierarchyWireBudget.from_mapping(
            _mapping(embedded_raw, label="hierarchy_wire_budget")
        )
    )
    if wire_budget is not None and embedded is not None and embedded != wire_budget:
        raise ValueError(
            "explicit wire_budget differs from the authenticated request budget"
        )
    if wire_budget is not None:
        return wire_budget
    if embedded is not None:
        return embedded
    if allow_legacy_default:
        return LEGACY_HIERARCHY_WIRE_BUDGET
    raise ValueError("hierarchy response compilation requires an explicit wire_budget")


def build_hierarchical_discovery_response_contract(
    *,
    job_kind: str,
    request: Mapping[str, Any],
    wire_budget: HierarchyWireBudget | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive an exact response contract under one authenticated wire budget.

    New callers pass ``wire_budget`` or include its exact versioned mapping in
    ``request``.  The legacy fallback remains only for source-compatible unit
    callers; :func:`attach_hierarchical_discovery_response_contract` always
    writes the resolved budget into the authenticated model request.
    """

    row = _mapping(request, label="discovery request")
    resolved = _resolved_wire_budget(
        request=row,
        wire_budget=wire_budget,
        allow_legacy_default=True,
    )
    token = _ACTIVE_HIERARCHY_WIRE_BUDGET.set(resolved)
    try:
        return _build_hierarchical_discovery_response_contract_with_active_budget(
            job_kind=job_kind,
            request=row,
        )
    finally:
        _ACTIVE_HIERARCHY_WIRE_BUDGET.reset(token)


def attach_hierarchical_discovery_response_contract(
    *,
    job_kind: str,
    request: Mapping[str, Any],
    wire_budget: HierarchyWireBudget | None = None,
) -> dict[str, Any]:
    """Attach the exact model-facing schema and ownership contract once."""

    row = dict(_mapping(request, label="discovery request"))
    reserved = {"output_schema", "identifier_ownership"}.intersection(row)
    if reserved:
        raise ValueError(f"discovery request already contains reserved contract fields: {reserved}")
    resolved = _resolved_wire_budget(
        request=row,
        wire_budget=wire_budget,
        allow_legacy_default=True,
    )
    row["hierarchy_wire_budget"] = resolved.as_dict()
    schema, ownership = build_hierarchical_discovery_response_contract(
        job_kind=job_kind,
        request=row,
        wire_budget=resolved,
    )
    row["identifier_ownership"] = ownership
    row["output_schema"] = schema
    return row


__all__ = [
    "HIERARCHY_WIRE_BUDGET_SCHEMA_VERSION",
    "HierarchyWireBudget",
    "LEGACY_HIERARCHY_WIRE_BUDGET",
    "HIERARCHICAL_DISCOVERY_EXACT_COVERAGE_REPRESENTATION",
    "HIERARCHICAL_DISCOVERY_MAX_AMBIGUITY_LENGTH",
    "HIERARCHICAL_DISCOVERY_MAX_ADAPTIVE_REVIEW_TARGETS",
    "HIERARCHICAL_DISCOVERY_MAX_ATOMS_PER_INTERPRET_JOB",
    "HIERARCHICAL_DISCOVERY_MAX_DESCRIPTION_LENGTH",
    "HIERARCHICAL_DISCOVERY_MAX_FEATURE_NAMES_PER_MEMBER",
    "HIERARCHICAL_DISCOVERY_MAX_GENERATED_LIST_ITEMS",
    "HIERARCHICAL_DISCOVERY_MAX_GENERATED_NAME_LENGTH",
    "HIERARCHICAL_DISCOVERY_MAX_FINDINGS_PER_ATOMIC_REVIEW",
    "HIERARCHICAL_DISCOVERY_MAX_PAIR_RELATION_PEERS",
    "HIERARCHICAL_DISCOVERY_MAX_DEFINITION_FOLD_MEMBERS",
    "HIERARCHICAL_DISCOVERY_MAX_GROUP_LOOKBACK_IDS",
    "HIERARCHICAL_DISCOVERY_MAX_INTERPRET_AMBIGUITY_LENGTH",
    "HIERARCHICAL_DISCOVERY_MAX_INTERPRET_DESCRIPTION_LENGTH",
    "HIERARCHICAL_DISCOVERY_MAX_INTERPRET_NAME_LENGTH",
    "HIERARCHICAL_DISCOVERY_MAX_INTERPRET_REASON_LENGTH",
    "HIERARCHICAL_DISCOVERY_MAX_INTERPRET_CANONICAL_JSON_BYTES",
    "HIERARCHICAL_DISCOVERY_MAX_INTERPRET_TRANSPORT_BYTES",
    "HIERARCHICAL_DISCOVERY_MAX_OPAQUE_IDENTIFIER_LENGTH",
    "HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB",
    "HIERARCHICAL_DISCOVERY_MAX_TRANSPORT_BYTES",
    "HIERARCHICAL_DISCOVERY_MAX_REASON_LENGTH",
    "HIERARCHICAL_DISCOVERY_MAX_TEXT_LENGTH",
    "HIERARCHICAL_DISCOVERY_INTERPRET_TOKEN_BUDGET",
    "HIERARCHICAL_DISCOVERY_GENERATION_TOKEN_BUDGET",
    "HIERARCHICAL_DISCOVERY_CONSERVATIVE_UTF8_BYTES_PER_TOKEN",
    "HIERARCHICAL_DISCOVERY_RESPONSE_CONTRACT_VERSION",
    "HIERARCHICAL_DISCOVERY_WIRE_RESPONSE_BUDGET_VERSION",
    "attach_hierarchical_discovery_response_contract",
    "build_hierarchical_discovery_response_contract",
]
