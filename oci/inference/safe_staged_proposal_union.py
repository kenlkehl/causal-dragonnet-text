"""Conservative, deterministic union of independently validated proposals.

This module is intentionally isolated from the live all-evidence runner.  It
does not import an agent, a dataset, or any fitting/extraction implementation.
Its only input is a closed sequence of already validated candidate mappings.

The compatibility rule is deliberately narrow: proposals sharing a name may
combine support and causal roles only when their complete extraction specs,
after removing ``roles``, have identical canonical JSON.  In particular,
type, ordered categories, description (including temporal wording), and
value aliases must match exactly.  Incompatible variants compete using only
precommitted generic support counts, never proposal vocabulary:

1. validated occurrence count;
2. distinct source-family count;
3. distinct evidence-ID count;
4. opaque candidate ID, then original ordinal.

Only the winning compatible group contributes roles, evidence, families, or
occurrences.  Every input candidate ID receives exactly one final disposition.
Canonical hashes and a runtime implementation identity make the pure result
auditable without retaining patient rows, treatment/outcome labels, or oracle
fields.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

SAFE_STAGED_PROPOSAL_UNION_POLICY_VERSION = "safe_staged_proposal_union_policy_v1"
SAFE_STAGED_PROPOSAL_UNION_INPUT_SCHEMA_VERSION = "safe_staged_proposal_union_input_v1"
SAFE_STAGED_PROPOSAL_UNION_OUTPUT_SCHEMA_VERSION = "safe_staged_proposal_union_output_v1"
SAFE_STAGED_PROPOSAL_UNION_HASH_DOMAIN_VERSION = "safe_staged_proposal_union_hash_domain_v1"

# Kept local so this safety policy has no import-time dependency on a live-run
# module.  These are method-family identifiers, not dataset vocabulary.
SAFE_STAGED_PROPOSAL_SOURCE_FAMILIES = (
    "bow_nuisance",
    "bow_r_loss",
    "matched_pair_uplift",
    "htr_neural",
    "embedding_whole_cohort",
    "embedding_clustered",
    "tfidf_topics",
    "tfidf_orphan_ngrams",
    "neural_query_moments",
    "sparse_query_moments",
)

_SOURCE_FAMILY_SET = frozenset(SAFE_STAGED_PROPOSAL_SOURCE_FAMILIES)
_ROLE_ORDER = ("confounder", "effect_modifier")
_ROLE_SET = frozenset(_ROLE_ORDER)
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "extraction_spec",
        "supporting_evidence_ids",
        "supporting_source_families",
        "validated_occurrence_count",
    }
)
_SPEC_REQUIRED_FIELDS = frozenset({"name", "type", "roles", "description"})
_SPEC_OPTIONAL_FIELDS = frozenset({"categories", "value_aliases"})
_NON_ROLE_FIELD_ORDER = (
    "type",
    "categories",
    "description",
    "value_aliases",
)
_DISPOSITIONS = frozenset(
    {
        "representative",
        "exact_duplicate",
        "compatible_role_merge",
        "omitted_conflict",
    }
)
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_SNAKE_CASE_NAME = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("proposal-union values must be finite JSON values") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _detached_json_object(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    try:
        detached = json.loads(_canonical_json(value))
    except ValueError as exc:
        raise ValueError(f"{path} must contain only finite JSON values") from exc
    if not isinstance(detached, dict):  # pragma: no cover - Mapping guard
        raise TypeError(f"{path} must be an object")
    return detached


def _strict_string(value: Any, *, path: str, max_chars: int = 2_000) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{path} must be non-empty and have no edge whitespace")
    if len(value) > max_chars:
        raise ValueError(f"{path} is too long")
    return value


def _opaque_id(value: Any, *, path: str) -> str:
    text = _strict_string(value, path=path, max_chars=256)
    if _OPAQUE_ID.fullmatch(text) is None:
        raise ValueError(f"{path} must be an opaque identifier")
    return text


def _sequence(value: Any, *, path: str, nonempty: bool = True) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence")
    output = tuple(value)
    if nonempty and not output:
        raise ValueError(f"{path} cannot be empty")
    return output


def _unique_strings(
    value: Any,
    *,
    path: str,
    opaque: bool = False,
    nonempty: bool = True,
) -> tuple[str, ...]:
    raw = _sequence(value, path=path, nonempty=nonempty)
    output = tuple(
        (
            _opaque_id(item, path=f"{path}[{index}]")
            if opaque
            else _strict_string(item, path=f"{path}[{index}]", max_chars=256)
        )
        for index, item in enumerate(raw)
    )
    if len(output) != len(set(output)):
        raise ValueError(f"{path} must contain unique strings")
    return output


def _normalized_category_token(value: Any) -> str:
    return re.sub(r"[\s_-]+", " ", str(value)).strip().casefold()


def _validate_extraction_spec(value: Any, *, path: str) -> dict[str, Any]:
    raw = _detached_json_object(value, path=path)
    fields = set(raw)
    missing = _SPEC_REQUIRED_FIELDS - fields
    unexpected = fields - _SPEC_REQUIRED_FIELDS - _SPEC_OPTIONAL_FIELDS
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"{path} has unsupported fields: {sorted(unexpected)}")

    name = _strict_string(raw["name"], path=f"{path}.name", max_chars=128)
    if _SNAKE_CASE_NAME.fullmatch(name) is None:
        raise ValueError(f"{path}.name must be lower snake_case")

    kind = _strict_string(raw["type"], path=f"{path}.type", max_chars=32)
    if kind not in {"categorical", "continuous"}:
        raise ValueError(f"{path}.type must be categorical or continuous")

    raw_roles = _unique_strings(raw["roles"], path=f"{path}.roles")
    unknown_roles = set(raw_roles) - _ROLE_SET
    if unknown_roles:
        raise ValueError(f"{path}.roles contains unknown roles: {sorted(unknown_roles)}")
    roles = [role for role in _ROLE_ORDER if role in set(raw_roles)]

    description = _strict_string(raw["description"], path=f"{path}.description", max_chars=2_000)
    normalized: dict[str, Any] = {
        "name": name,
        "type": kind,
    }

    if kind == "continuous":
        categories = raw.get("categories")
        value_aliases = raw.get("value_aliases")
        if categories not in (None, []):
            raise ValueError(f"{path}.categories must be absent for a continuous spec")
        if value_aliases not in (None, {}):
            raise ValueError(f"{path}.value_aliases must be absent for a continuous spec")
        # The authoritative contract accepts explicit null/empty optional
        # fields and preserves them.  Presence therefore remains part of this
        # policy's conservative exact compatibility signature.
        if "categories" in raw:
            normalized["categories"] = categories
        if "value_aliases" in raw:
            normalized["value_aliases"] = value_aliases
    else:
        if "categories" not in raw:
            raise ValueError(f"{path}.categories is required for a categorical spec")
        raw_categories = raw["categories"]
        if not isinstance(raw_categories, list) or not 2 <= len(raw_categories) <= 8:
            raise ValueError(
                f"{path}.categories must contain at least two and at most eight values"
            )
        category_text = [str(category).strip() for category in raw_categories]
        if any(not category for category in category_text):
            raise ValueError(f"{path}.categories cannot contain empty values")
        normalized_categories = [_normalized_category_token(category) for category in category_text]
        if len(normalized_categories) != len(set(normalized_categories)):
            raise ValueError(f"{path}.categories must be distinct after case/spacing normalization")
        # Preserve the exact category values/order accepted at the upstream
        # validation boundary; normalization is used only to detect aliases
        # that would create an ambiguous extraction mapping.
        normalized["categories"] = list(raw_categories)

        if "value_aliases" in raw:
            alias_path = f"{path}.value_aliases"
            raw_aliases = raw["value_aliases"]
            if raw_aliases in (None, {}):
                normalized["value_aliases"] = raw_aliases
            else:
                aliases = _detached_json_object(raw_aliases, path=alias_path)
                unknown_keys = set(map(str, aliases)) - set(category_text)
                if unknown_keys:
                    raise ValueError(
                        f"{alias_path} keys must be a subset of declared categories; "
                        f"unknown={sorted(unknown_keys)}"
                    )
                normalized_owner = {
                    normalized_category: category
                    for normalized_category, category in zip(
                        normalized_categories, category_text, strict=True
                    )
                }
                for raw_category, alias_values in aliases.items():
                    category = str(raw_category)
                    if not isinstance(alias_values, list) or not alias_values:
                        raise ValueError(
                            f"{alias_path}[{category!r}] must be a non-empty string list"
                        )
                    if not all(isinstance(alias, str) and alias.strip() for alias in alias_values):
                        raise ValueError(
                            f"{alias_path}[{category!r}] cannot contain " "empty/non-string aliases"
                        )
                    for alias in alias_values:
                        normalized_alias = _normalized_category_token(alias)
                        prior_owner = normalized_owner.get(normalized_alias)
                        if prior_owner is not None:
                            raise ValueError(
                                f"{alias_path} contains a normalized collision between "
                                f"{prior_owner!r} and {category!r}"
                            )
                        normalized_owner[normalized_alias] = category
                # A partial map is valid.  Do not synthesize missing category
                # keys or normalize/reorder alias lists: the exact accepted
                # map is part of the non-role compatibility signature.
                normalized["value_aliases"] = aliases

    normalized["roles"] = roles
    normalized["description"] = description
    return normalized


@dataclass(frozen=True)
class SafeProposalUnionIdentity:
    policy_version: str
    input_schema_version: str
    output_schema_version: str
    hash_domain_version: str
    implementation_module: str
    implementation_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "policy_version": self.policy_version,
            "input_schema_version": self.input_schema_version,
            "output_schema_version": self.output_schema_version,
            "hash_domain_version": self.hash_domain_version,
            "implementation_module": self.implementation_module,
            "implementation_sha256": self.implementation_sha256,
        }


def safe_staged_proposal_union_identity() -> SafeProposalUnionIdentity:
    """Return the exact policy and current source-byte identity."""

    implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return SafeProposalUnionIdentity(
        policy_version=SAFE_STAGED_PROPOSAL_UNION_POLICY_VERSION,
        input_schema_version=SAFE_STAGED_PROPOSAL_UNION_INPUT_SCHEMA_VERSION,
        output_schema_version=SAFE_STAGED_PROPOSAL_UNION_OUTPUT_SCHEMA_VERSION,
        hash_domain_version=SAFE_STAGED_PROPOSAL_UNION_HASH_DOMAIN_VERSION,
        implementation_module=__name__,
        implementation_sha256=implementation_sha256,
    )


def assert_safe_staged_proposal_union_identity(expected_sha256: str) -> None:
    """Reject execution unless the current source bytes have the expected hash."""

    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 must be one lowercase SHA-256 digest")
    actual = safe_staged_proposal_union_identity().implementation_sha256
    if actual != expected_sha256:
        raise RuntimeError(
            "safe staged proposal-union implementation identity mismatch: "
            f"expected {expected_sha256}, observed {actual}"
        )


@dataclass(frozen=True)
class SafeProposalStrength:
    validated_occurrence_count: int
    independent_source_family_breadth: int
    evidence_breadth: int

    def as_dict(self) -> dict[str, int]:
        return {
            "validated_occurrence_count": self.validated_occurrence_count,
            "independent_source_family_breadth": (self.independent_source_family_breadth),
            "evidence_breadth": self.evidence_breadth,
        }


@dataclass(frozen=True)
class SafeProposalUnionCandidate:
    candidate_id: str
    supporting_evidence_ids: tuple[str, ...]
    supporting_source_families: tuple[str, ...]
    validated_occurrence_count: int
    _extraction_spec_json: str = field(repr=False)

    @property
    def extraction_spec(self) -> Mapping[str, Any]:
        """Return an immutable detached view of the extraction contract."""

        return _freeze_json(json.loads(self._extraction_spec_json))

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "extraction_spec": json.loads(self._extraction_spec_json),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "supporting_source_families": list(self.supporting_source_families),
            "validated_occurrence_count": self.validated_occurrence_count,
        }


@dataclass(frozen=True)
class SafeProposalDisposition:
    candidate_id: str
    disposition: str
    retained_candidate_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "disposition": self.disposition,
            "retained_candidate_id": self.retained_candidate_id,
        }


@dataclass(frozen=True)
class SafeProposalConflict:
    conflict_id: str
    retained_candidate_id: str
    omitted_candidate_ids: tuple[str, ...]
    differing_non_role_fields: tuple[str, ...]
    retained_strength: SafeProposalStrength
    omitted_strength: SafeProposalStrength

    def as_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "retained_candidate_id": self.retained_candidate_id,
            "omitted_candidate_ids": list(self.omitted_candidate_ids),
            "differing_non_role_fields": list(self.differing_non_role_fields),
            "retained_strength": self.retained_strength.as_dict(),
            "omitted_strength": self.omitted_strength.as_dict(),
        }


@dataclass(frozen=True)
class SafeStagedProposalUnionResult:
    identity: SafeProposalUnionIdentity
    input_sha256: str
    output_sha256: str
    candidates: tuple[SafeProposalUnionCandidate, ...]
    input_candidate_ids: tuple[str, ...]
    representative_candidate_ids: tuple[str, ...]
    exact_duplicate_candidate_ids: tuple[str, ...]
    compatible_role_merge_candidate_ids: tuple[str, ...]
    omitted_conflict_candidate_ids: tuple[str, ...]
    dispositions: tuple[SafeProposalDisposition, ...]
    conflicts: tuple[SafeProposalConflict, ...]
    _canonical_output_json: str = field(repr=False)

    @property
    def canonical_output_json(self) -> str:
        return self._canonical_output_json

    def as_dict(self) -> dict[str, Any]:
        """Return a detached JSON envelope; mutating it cannot alter the result."""

        return json.loads(self._canonical_output_json)

    def verify(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]] | None = None,
        expected_implementation_sha256: str | None = None,
    ) -> None:
        verify_safe_staged_proposal_union(
            self,
            candidates=candidates,
            expected_implementation_sha256=expected_implementation_sha256,
        )


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    extraction_spec_json: str
    supporting_evidence_ids: tuple[str, ...]
    supporting_source_families: tuple[str, ...]
    validated_occurrence_count: int
    ordinal: int

    @property
    def extraction_spec(self) -> dict[str, Any]:
        return json.loads(self.extraction_spec_json)

    @property
    def non_role_spec_json(self) -> str:
        spec = self.extraction_spec
        del spec["roles"]
        return _canonical_json(spec)

    @property
    def strength(self) -> SafeProposalStrength:
        return SafeProposalStrength(
            validated_occurrence_count=self.validated_occurrence_count,
            independent_source_family_breadth=len(self.supporting_source_families),
            evidence_breadth=len(self.supporting_evidence_ids),
        )

    def as_input_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "extraction_spec": self.extraction_spec,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "supporting_source_families": list(self.supporting_source_families),
            "validated_occurrence_count": self.validated_occurrence_count,
        }


@dataclass(frozen=True)
class _CompatibleGroup:
    members: tuple[_Candidate, ...]
    representative: _Candidate
    extraction_spec_json: str
    supporting_evidence_ids: tuple[str, ...]
    supporting_source_families: tuple[str, ...]
    validated_occurrence_count: int
    first_ordinal: int

    @property
    def strength(self) -> SafeProposalStrength:
        return SafeProposalStrength(
            validated_occurrence_count=self.validated_occurrence_count,
            independent_source_family_breadth=len(self.supporting_source_families),
            evidence_breadth=len(self.supporting_evidence_ids),
        )


def _validate_candidates(value: Any) -> tuple[_Candidate, ...]:
    rows = _sequence(value, path="candidates")
    output: list[_Candidate] = []
    seen_ids: set[str] = set()
    for index, raw_value in enumerate(rows):
        path = f"candidates[{index}]"
        raw = _detached_json_object(raw_value, path=path)
        if set(raw) != _CANDIDATE_FIELDS:
            missing = _CANDIDATE_FIELDS - set(raw)
            unexpected = set(raw) - _CANDIDATE_FIELDS
            raise ValueError(
                f"{path} does not match the closed schema; "
                f"missing={sorted(missing)}, unsupported={sorted(unexpected)}"
            )
        candidate_id = _opaque_id(raw["candidate_id"], path=f"{path}.candidate_id")
        if candidate_id in seen_ids:
            raise ValueError(f"{path}.candidate_id duplicates {candidate_id!r}")
        seen_ids.add(candidate_id)

        spec = _validate_extraction_spec(raw["extraction_spec"], path=f"{path}.extraction_spec")
        evidence_ids = tuple(
            sorted(
                _unique_strings(
                    raw["supporting_evidence_ids"],
                    path=f"{path}.supporting_evidence_ids",
                    opaque=True,
                )
            )
        )
        raw_families = _unique_strings(
            raw["supporting_source_families"],
            path=f"{path}.supporting_source_families",
        )
        unknown_families = set(raw_families) - _SOURCE_FAMILY_SET
        if unknown_families:
            raise ValueError(
                f"{path}.supporting_source_families contains unknown families: "
                f"{sorted(unknown_families)}"
            )
        family_set = set(raw_families)
        families = tuple(
            family for family in SAFE_STAGED_PROPOSAL_SOURCE_FAMILIES if family in family_set
        )
        occurrence = raw["validated_occurrence_count"]
        if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 1:
            raise ValueError(f"{path}.validated_occurrence_count must be a positive integer")
        output.append(
            _Candidate(
                candidate_id=candidate_id,
                extraction_spec_json=_canonical_json(spec),
                supporting_evidence_ids=evidence_ids,
                supporting_source_families=families,
                validated_occurrence_count=occurrence,
                ordinal=index,
            )
        )
    return tuple(output)


def _candidate_rank(candidate: _Candidate) -> tuple[int, int, int, str, int]:
    strength = candidate.strength
    return (
        -strength.validated_occurrence_count,
        -strength.independent_source_family_breadth,
        -strength.evidence_breadth,
        candidate.candidate_id,
        candidate.ordinal,
    )


def _compatible_group(members: Sequence[_Candidate]) -> _CompatibleGroup:
    if not members:  # pragma: no cover - internal group construction
        raise RuntimeError("cannot construct an empty compatible group")
    ordered_members = tuple(sorted(members, key=lambda item: item.candidate_id))
    representative = min(ordered_members, key=_candidate_rank)
    non_role_signatures = {member.non_role_spec_json for member in ordered_members}
    if len(non_role_signatures) != 1:  # pragma: no cover - group map guards this
        raise RuntimeError("compatible group contains incompatible extraction specs")

    roles = {role for member in ordered_members for role in member.extraction_spec.get("roles", [])}
    merged_spec = representative.extraction_spec
    merged_spec["roles"] = [role for role in _ROLE_ORDER if role in roles]
    evidence_ids = tuple(
        sorted(
            {
                evidence_id
                for member in ordered_members
                for evidence_id in member.supporting_evidence_ids
            }
        )
    )
    family_set = {
        family for member in ordered_members for family in member.supporting_source_families
    }
    families = tuple(
        family for family in SAFE_STAGED_PROPOSAL_SOURCE_FAMILIES if family in family_set
    )
    return _CompatibleGroup(
        members=ordered_members,
        representative=representative,
        extraction_spec_json=_canonical_json(merged_spec),
        supporting_evidence_ids=evidence_ids,
        supporting_source_families=families,
        validated_occurrence_count=sum(
            member.validated_occurrence_count for member in ordered_members
        ),
        first_ordinal=min(member.ordinal for member in ordered_members),
    )


def _group_rank(group: _CompatibleGroup) -> tuple[int, int, int, str, int]:
    strength = group.strength
    return (
        -strength.validated_occurrence_count,
        -strength.independent_source_family_breadth,
        -strength.evidence_breadth,
        group.representative.candidate_id,
        group.first_ordinal,
    )


def _differing_non_role_fields(
    retained: _CompatibleGroup, omitted: _CompatibleGroup
) -> tuple[str, ...]:
    retained_spec = json.loads(retained.extraction_spec_json)
    omitted_spec = json.loads(omitted.extraction_spec_json)
    return tuple(
        field_name
        for field_name in _NON_ROLE_FIELD_ORDER
        if (field_name in retained_spec) != (field_name in omitted_spec)
        or retained_spec.get(field_name) != omitted_spec.get(field_name)
    )


def _conflict(*, retained: _CompatibleGroup, omitted: _CompatibleGroup) -> SafeProposalConflict:
    omitted_ids = tuple(member.candidate_id for member in omitted.members)
    differing_fields = _differing_non_role_fields(retained, omitted)
    if not differing_fields:  # pragma: no cover - signatures would share a group
        raise RuntimeError("incompatible proposal variants have no differing fields")
    identity_payload = {
        "policy_version": SAFE_STAGED_PROPOSAL_UNION_POLICY_VERSION,
        "retained_candidate_id": retained.representative.candidate_id,
        "omitted_candidate_ids": list(omitted_ids),
        "differing_non_role_fields": list(differing_fields),
    }
    return SafeProposalConflict(
        conflict_id=f"conflict_{_sha256_json(identity_payload)}",
        retained_candidate_id=retained.representative.candidate_id,
        omitted_candidate_ids=omitted_ids,
        differing_non_role_fields=differing_fields,
        retained_strength=retained.strength,
        omitted_strength=omitted.strength,
    )


def _canonical_input_payload(candidates: Sequence[_Candidate]) -> dict[str, Any]:
    return {
        "schema_version": SAFE_STAGED_PROPOSAL_UNION_INPUT_SCHEMA_VERSION,
        "candidates": [candidate.as_input_dict() for candidate in candidates],
    }


def _accounting_payload(result: SafeStagedProposalUnionResult) -> dict[str, Any]:
    return {
        "input_candidate_count": len(result.input_candidate_ids),
        "output_candidate_count": len(result.candidates),
        "input_candidate_ids": list(result.input_candidate_ids),
        "representative_candidate_ids": list(result.representative_candidate_ids),
        "exact_duplicate_candidate_ids": list(result.exact_duplicate_candidate_ids),
        "compatible_role_merge_candidate_ids": list(result.compatible_role_merge_candidate_ids),
        "omitted_conflict_candidate_ids": list(result.omitted_conflict_candidate_ids),
        "dispositions": [row.as_dict() for row in result.dispositions],
    }


def _output_payload(result: SafeStagedProposalUnionResult) -> dict[str, Any]:
    return {
        "schema_version": SAFE_STAGED_PROPOSAL_UNION_OUTPUT_SCHEMA_VERSION,
        "identity": result.identity.as_dict(),
        "input_sha256": result.input_sha256,
        "candidates": [candidate.as_dict() for candidate in result.candidates],
        "accounting": _accounting_payload(result),
        "conflicts": [conflict.as_dict() for conflict in result.conflicts],
    }


def _output_envelope(result: SafeStagedProposalUnionResult) -> dict[str, Any]:
    payload = _output_payload(result)
    payload["output_sha256"] = result.output_sha256
    return payload


def _build_safe_staged_proposal_union(
    candidates: Sequence[Mapping[str, Any]],
    *,
    identity: SafeProposalUnionIdentity,
) -> SafeStagedProposalUnionResult:
    validated = _validate_candidates(candidates)
    input_payload = _canonical_input_payload(validated)
    input_sha256 = _sha256_json(input_payload)

    by_name: dict[str, list[_Candidate]] = {}
    for candidate in validated:
        name = str(candidate.extraction_spec["name"])
        by_name.setdefault(name, []).append(candidate)

    retained_groups: list[_CompatibleGroup] = []
    conflicts: list[SafeProposalConflict] = []
    disposition_by_id: dict[str, SafeProposalDisposition] = {}
    exact_duplicate_ids: list[str] = []
    compatible_role_ids: list[str] = []
    omitted_conflict_ids: list[str] = []

    for name_members in by_name.values():
        by_non_role_spec: dict[str, list[_Candidate]] = {}
        for member in name_members:
            by_non_role_spec.setdefault(member.non_role_spec_json, []).append(member)
        groups = tuple(_compatible_group(members) for members in by_non_role_spec.values())
        retained = min(groups, key=_group_rank)
        retained_groups.append(retained)
        retained_id = retained.representative.candidate_id

        for member in retained.members:
            if member.candidate_id == retained_id:
                disposition = "representative"
            elif member.extraction_spec_json == retained.representative.extraction_spec_json:
                disposition = "exact_duplicate"
                exact_duplicate_ids.append(member.candidate_id)
            else:
                disposition = "compatible_role_merge"
                compatible_role_ids.append(member.candidate_id)
            disposition_by_id[member.candidate_id] = SafeProposalDisposition(
                candidate_id=member.candidate_id,
                disposition=disposition,
                retained_candidate_id=retained_id,
            )

        for omitted in groups:
            if omitted is retained:
                continue
            conflicts.append(_conflict(retained=retained, omitted=omitted))
            for member in omitted.members:
                omitted_conflict_ids.append(member.candidate_id)
                disposition_by_id[member.candidate_id] = SafeProposalDisposition(
                    candidate_id=member.candidate_id,
                    disposition="omitted_conflict",
                    retained_candidate_id=retained_id,
                )

    retained_groups.sort(key=lambda group: group.representative.candidate_id)
    union_candidates = tuple(
        SafeProposalUnionCandidate(
            candidate_id=group.representative.candidate_id,
            supporting_evidence_ids=group.supporting_evidence_ids,
            supporting_source_families=group.supporting_source_families,
            validated_occurrence_count=group.validated_occurrence_count,
            _extraction_spec_json=group.extraction_spec_json,
        )
        for group in retained_groups
    )
    conflicts.sort(
        key=lambda conflict: (
            conflict.retained_candidate_id,
            conflict.omitted_candidate_ids,
        )
    )
    input_ids = tuple(candidate.candidate_id for candidate in validated)
    dispositions = tuple(disposition_by_id[candidate_id] for candidate_id in input_ids)

    provisional = SafeStagedProposalUnionResult(
        identity=identity,
        input_sha256=input_sha256,
        output_sha256="0" * 64,
        candidates=union_candidates,
        input_candidate_ids=input_ids,
        representative_candidate_ids=tuple(
            candidate.candidate_id for candidate in union_candidates
        ),
        exact_duplicate_candidate_ids=tuple(sorted(exact_duplicate_ids)),
        compatible_role_merge_candidate_ids=tuple(sorted(compatible_role_ids)),
        omitted_conflict_candidate_ids=tuple(sorted(omitted_conflict_ids)),
        dispositions=dispositions,
        conflicts=tuple(conflicts),
        _canonical_output_json="",
    )
    output_sha256 = _sha256_json(_output_payload(provisional))
    result = SafeStagedProposalUnionResult(
        identity=provisional.identity,
        input_sha256=provisional.input_sha256,
        output_sha256=output_sha256,
        candidates=provisional.candidates,
        input_candidate_ids=provisional.input_candidate_ids,
        representative_candidate_ids=provisional.representative_candidate_ids,
        exact_duplicate_candidate_ids=provisional.exact_duplicate_candidate_ids,
        compatible_role_merge_candidate_ids=(provisional.compatible_role_merge_candidate_ids),
        omitted_conflict_candidate_ids=provisional.omitted_conflict_candidate_ids,
        dispositions=provisional.dispositions,
        conflicts=provisional.conflicts,
        _canonical_output_json="",
    )
    canonical_output_json = _canonical_json(_output_envelope(result))
    return SafeStagedProposalUnionResult(
        identity=result.identity,
        input_sha256=result.input_sha256,
        output_sha256=result.output_sha256,
        candidates=result.candidates,
        input_candidate_ids=result.input_candidate_ids,
        representative_candidate_ids=result.representative_candidate_ids,
        exact_duplicate_candidate_ids=result.exact_duplicate_candidate_ids,
        compatible_role_merge_candidate_ids=result.compatible_role_merge_candidate_ids,
        omitted_conflict_candidate_ids=result.omitted_conflict_candidate_ids,
        dispositions=result.dispositions,
        conflicts=result.conflicts,
        _canonical_output_json=canonical_output_json,
    )


def safe_staged_proposal_union(
    candidates: Sequence[Mapping[str, Any]],
) -> SafeStagedProposalUnionResult:
    """Validate and safely consolidate one staged proposal inventory."""

    return _build_safe_staged_proposal_union(
        candidates,
        identity=safe_staged_proposal_union_identity(),
    )


def _validate_result_accounting(result: SafeStagedProposalUnionResult) -> None:
    input_ids = result.input_candidate_ids
    if len(input_ids) != len(set(input_ids)):
        raise ValueError("result input candidate IDs are not unique")
    partitions = (
        result.representative_candidate_ids,
        result.exact_duplicate_candidate_ids,
        result.compatible_role_merge_candidate_ids,
        result.omitted_conflict_candidate_ids,
    )
    flattened = tuple(candidate_id for part in partitions for candidate_id in part)
    if len(flattened) != len(set(flattened)) or set(flattened) != set(input_ids):
        raise ValueError("result candidate-ID dispositions are not a full disjoint partition")
    if tuple(candidate.candidate_id for candidate in result.candidates) != (
        result.representative_candidate_ids
    ):
        raise ValueError("result representatives disagree with output candidates")
    if len(result.dispositions) != len(input_ids):
        raise ValueError("result disposition rows do not cover every input candidate")
    dispositions_by_id = {row.candidate_id: row for row in result.dispositions}
    if len(dispositions_by_id) != len(input_ids) or set(dispositions_by_id) != set(input_ids):
        raise ValueError("result disposition rows have duplicate or unknown IDs")

    expected_by_disposition = {
        "representative": set(result.representative_candidate_ids),
        "exact_duplicate": set(result.exact_duplicate_candidate_ids),
        "compatible_role_merge": set(result.compatible_role_merge_candidate_ids),
        "omitted_conflict": set(result.omitted_conflict_candidate_ids),
    }
    representatives = set(result.representative_candidate_ids)
    for row in result.dispositions:
        if row.disposition not in _DISPOSITIONS:
            raise ValueError("result contains an unknown disposition")
        if row.candidate_id not in expected_by_disposition[row.disposition]:
            raise ValueError("result disposition row disagrees with its partition")
        if row.retained_candidate_id not in representatives:
            raise ValueError("result disposition names a non-representative target")
        if row.disposition == "representative" and (row.candidate_id != row.retained_candidate_id):
            raise ValueError("representative disposition does not retain itself")

    conflict_omissions: list[str] = []
    for conflict in result.conflicts:
        if conflict.retained_candidate_id not in representatives:
            raise ValueError("conflict names an unknown retained candidate")
        if not conflict.omitted_candidate_ids or not conflict.differing_non_role_fields:
            raise ValueError("conflict metadata is incomplete")
        identity_payload = {
            "policy_version": SAFE_STAGED_PROPOSAL_UNION_POLICY_VERSION,
            "retained_candidate_id": conflict.retained_candidate_id,
            "omitted_candidate_ids": list(conflict.omitted_candidate_ids),
            "differing_non_role_fields": list(conflict.differing_non_role_fields),
        }
        if conflict.conflict_id != f"conflict_{_sha256_json(identity_payload)}":
            raise ValueError("conflict identity hash is invalid")
        for candidate_id in conflict.omitted_candidate_ids:
            row = dispositions_by_id.get(candidate_id)
            if row is None or row.disposition != "omitted_conflict":
                raise ValueError("conflict metadata names a non-omitted candidate")
            if row.retained_candidate_id != conflict.retained_candidate_id:
                raise ValueError("conflict metadata disagrees with disposition target")
            conflict_omissions.append(candidate_id)
    if len(conflict_omissions) != len(set(conflict_omissions)) or set(conflict_omissions) != set(
        result.omitted_conflict_candidate_ids
    ):
        raise ValueError("conflicts do not exactly account for omitted candidate IDs")


def verify_safe_staged_proposal_union(
    result: SafeStagedProposalUnionResult,
    *,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    expected_implementation_sha256: str | None = None,
) -> None:
    """Verify result integrity and, optionally, recompute it from exact inputs."""

    if not isinstance(result, SafeStagedProposalUnionResult):
        raise TypeError("result must be a SafeStagedProposalUnionResult")
    current_identity = safe_staged_proposal_union_identity()
    if expected_implementation_sha256 is not None:
        assert_safe_staged_proposal_union_identity(expected_implementation_sha256)
    if result.identity != current_identity:
        raise RuntimeError("proposal-union result has a stale or foreign implementation identity")
    if (
        _SHA256.fullmatch(result.input_sha256) is None
        or _SHA256.fullmatch(result.output_sha256) is None
    ):
        raise ValueError("result contains a malformed canonical hash")

    _validate_result_accounting(result)
    payload = _output_payload(result)
    if _sha256_json(payload) != result.output_sha256:
        raise ValueError("proposal-union output hash mismatch")
    expected_json = _canonical_json(_output_envelope(result))
    if result.canonical_output_json != expected_json:
        raise ValueError("proposal-union canonical output envelope was modified")

    # Revalidate the emitted candidate schema independently of the frozen
    # dataclass constructors, which are public and can be used directly.
    _validate_candidates([candidate.as_dict() for candidate in result.candidates])

    if candidates is not None:
        expected = _build_safe_staged_proposal_union(
            candidates,
            identity=current_identity,
        )
        if (
            result.input_sha256 != expected.input_sha256
            or result.output_sha256 != expected.output_sha256
            or result.canonical_output_json != expected.canonical_output_json
        ):
            raise ValueError("proposal-union result does not match the supplied exact inputs")


__all__ = [
    "SAFE_STAGED_PROPOSAL_SOURCE_FAMILIES",
    "SAFE_STAGED_PROPOSAL_UNION_HASH_DOMAIN_VERSION",
    "SAFE_STAGED_PROPOSAL_UNION_INPUT_SCHEMA_VERSION",
    "SAFE_STAGED_PROPOSAL_UNION_OUTPUT_SCHEMA_VERSION",
    "SAFE_STAGED_PROPOSAL_UNION_POLICY_VERSION",
    "SafeProposalConflict",
    "SafeProposalDisposition",
    "SafeProposalStrength",
    "SafeProposalUnionCandidate",
    "SafeProposalUnionIdentity",
    "SafeStagedProposalUnionResult",
    "assert_safe_staged_proposal_union_identity",
    "safe_staged_proposal_union",
    "safe_staged_proposal_union_identity",
    "verify_safe_staged_proposal_union",
]
