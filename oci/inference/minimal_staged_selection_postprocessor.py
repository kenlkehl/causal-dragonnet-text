"""Recall-safe deterministic postprocessing for staged concept selection.

The remote reasoning selector is authoritative about which validated contracts
it wants to retain.  A deterministic safeguard may still be useful for two
bounded purposes:

* cover an evidence family or causal role represented in the validated
  candidate pool but accidentally absent from the remote selection; and
* retain candidates with precommitted strong recurrence/independent-family
  support so the later honest extraction/gate loop can assess them.

A maximum-candidate setting is a cap, not a target.  This module therefore
never fills unused capacity with weak candidates.  It also performs no lexical
or semantic alias merging: changing a contract's ontology, role, categories,
aliases, or temporal meaning belongs to the reasoning-enabled post-extraction
reviewer, where observable diagnostics and a complexity penalty are available.

The API consumes only already-validated, row-free contract metadata.  It
records both candidate-pool coverage and original-request family coverage so a
pool that omitted an upstream family cannot call itself fully covered.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .all_evidence_fusion import ALL_SOURCE_FAMILIES, CandidateContract

MINIMAL_STAGED_SELECTION_POSTPROCESSOR_VERSION = "recall_safe_staged_selection_postprocessor_v2"
MINIMAL_STAGED_SELECTION_INPUT_SCHEMA = "recall_safe_staged_selection_input_v1"
MINIMAL_STAGED_SELECTION_OUTPUT_SCHEMA = "recall_safe_staged_selection_output_v1"

_ROLE_ORDER = ("confounder", "effect_modifier")
_CANDIDATE_ID = re.compile(r"^candidate_[0-9]{4}$")
_EXTRACTION_SPEC_FIELDS = frozenset(
    {"name", "type", "categories", "roles", "description", "value_aliases"}
)
_PROPOSAL_GROUNDING_FIELDS = frozenset(
    {"supporting_evidence_ids", "supporting_source_families", "rationale"}
)
_MAX_CANDIDATES = 64


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
        raise ValueError("staged-selection metadata must be finite canonical JSON") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _detached(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _unique_strings(values: Any, *, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of strings")
    output = tuple(str(value).strip() for value in tuple(values))
    if (not allow_empty and not output) or any(not value for value in output):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(output) != len(set(output)):
        raise ValueError(f"{name} must contain unique strings")
    return output


def _ordered_families(values: Sequence[str], *, allow_empty: bool = False) -> tuple[str, ...]:
    family_set = set(values)
    unknown = family_set - set(ALL_SOURCE_FAMILIES)
    if unknown:
        raise ValueError(f"unknown source families: {sorted(unknown)}")
    output = tuple(family for family in ALL_SOURCE_FAMILIES if family in family_set)
    if not allow_empty and not output:
        raise ValueError("source-family collection cannot be empty")
    return output


def _ordered_roles(values: Sequence[str]) -> tuple[str, ...]:
    role_set = set(values)
    unknown = role_set - set(_ROLE_ORDER)
    if unknown:
        raise ValueError(f"unknown causal roles: {sorted(unknown)}")
    output = tuple(role for role in _ROLE_ORDER if role in role_set)
    if not output:
        raise ValueError("causal-role collection cannot be empty")
    return output


def _proposal_extraction_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    unexpected = set(value) - _EXTRACTION_SPEC_FIELDS - _PROPOSAL_GROUNDING_FIELDS
    if unexpected:
        raise ValueError(f"remote proposal contains unsupported fields: {sorted(unexpected)}")
    missing = _PROPOSAL_GROUNDING_FIELDS - set(value)
    if missing:
        raise ValueError("remote proposal lacks grounding fields: " + ", ".join(sorted(missing)))
    return {
        key: copy.deepcopy(child) for key, child in value.items() if key in _EXTRACTION_SPEC_FIELDS
    }


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    spec: dict[str, Any]
    evidence_ids: tuple[str, ...]
    source_families: tuple[str, ...]
    validated_occurrence_count: int

    @property
    def roles(self) -> tuple[str, ...]:
        return _ordered_roles(tuple(str(value) for value in self.spec.get("roles", ())))

    def as_input_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "extraction_spec": copy.deepcopy(self.spec),
            "supporting_evidence_ids": list(self.evidence_ids),
            "supporting_source_families": list(self.source_families),
            "validated_occurrence_count": self.validated_occurrence_count,
        }


def _candidate(raw: Mapping[str, Any], *, index: int) -> _Candidate:
    if not isinstance(raw, Mapping):
        raise TypeError(f"candidate_pool[{index}] must be an object")
    required = {
        "candidate_id",
        "extraction_spec",
        "supporting_evidence_ids",
        "supporting_source_families",
        "validated_occurrence_count",
    }
    if set(raw) != required:
        raise ValueError(f"candidate_pool[{index}] does not match the closed schema")
    candidate_id = str(raw["candidate_id"]).strip()
    if not _CANDIDATE_ID.fullmatch(candidate_id):
        raise ValueError(f"candidate_pool[{index}].candidate_id is not canonical")
    spec = _detached(raw["extraction_spec"])
    if not isinstance(spec, dict):
        raise TypeError(f"candidate_pool[{index}].extraction_spec must be an object")
    evidence_ids = _unique_strings(
        raw["supporting_evidence_ids"],
        name=f"candidate_pool[{index}].supporting_evidence_ids",
    )
    families = _ordered_families(
        _unique_strings(
            raw["supporting_source_families"],
            name=f"candidate_pool[{index}].supporting_source_families",
        )
    )
    occurrence = raw["validated_occurrence_count"]
    if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 1:
        raise ValueError(f"candidate_pool[{index}].validated_occurrence_count must be positive")
    CandidateContract(spec, source_families=families)
    return _Candidate(
        candidate_id=candidate_id,
        spec=spec,
        evidence_ids=evidence_ids,
        source_families=families,
        validated_occurrence_count=int(occurrence),
    )


def _quality_key(candidate: _Candidate) -> tuple[int, int, int, str]:
    """Ascending deterministic strength key; opaque ID is the final tie-break."""

    return (
        -candidate.validated_occurrence_count,
        -len(candidate.source_families),
        -len(candidate.evidence_ids),
        candidate.candidate_id,
    )


def _is_high_confidence(
    candidate: _Candidate,
    *,
    minimum_recurrence: int,
    minimum_independent_families: int,
    minimum_distinct_evidence: int,
) -> bool:
    return bool(
        candidate.validated_occurrence_count >= minimum_recurrence
        or (
            len(candidate.source_families) >= minimum_independent_families
            and len(candidate.evidence_ids) >= minimum_distinct_evidence
        )
    )


def _coverage_mask(
    candidate: _Candidate,
    *,
    family_bit: Mapping[str, int],
    role_bit: Mapping[str, int],
) -> int:
    mask = 0
    for family in candidate.source_families:
        bit = family_bit.get(family)
        if bit is not None:
            mask |= 1 << bit
    for role in candidate.roles:
        bit = role_bit.get(role)
        if bit is not None:
            mask |= 1 << bit
    return mask


def _minimum_coverage_set(
    candidates: Sequence[_Candidate],
    *,
    missing_families: tuple[str, ...],
    missing_roles: tuple[str, ...],
    maximum_additions: int,
) -> tuple[_Candidate, ...]:
    """Return an exact minimum full cover, or the best cap-limited partial cover."""

    if (not missing_families and not missing_roles) or maximum_additions < 1:
        return ()
    family_bit = {family: index for index, family in enumerate(missing_families)}
    role_bit = {role: len(family_bit) + index for index, role in enumerate(missing_roles)}
    full_mask = (1 << (len(family_bit) + len(role_bit))) - 1
    ordered = tuple(sorted(candidates, key=_quality_key))
    states: dict[int, tuple[_Candidate, ...]] = {0: ()}
    for candidate in ordered:
        candidate_mask = _coverage_mask(
            candidate,
            family_bit=family_bit,
            role_bit=role_bit,
        )
        if candidate_mask == 0:
            continue
        updates = dict(states)
        for mask, combination in states.items():
            if len(combination) >= maximum_additions:
                continue
            next_mask = mask | candidate_mask
            proposed = (*combination, candidate)
            prior = updates.get(next_mask)
            proposed_key = (
                len(proposed),
                tuple(_quality_key(value) for value in proposed),
            )
            prior_key = (
                (
                    len(prior),
                    tuple(_quality_key(value) for value in prior),
                )
                if prior is not None
                else None
            )
            if prior_key is None or proposed_key < prior_key:
                updates[next_mask] = proposed
        states = updates
    complete = states.get(full_mask)
    if complete is not None:
        return complete
    partial = [(mask, combination) for mask, combination in states.items() if mask and combination]
    if not partial:
        return ()
    return min(
        partial,
        key=lambda item: (
            -item[0].bit_count(),
            len(item[1]),
            tuple(_quality_key(value) for value in item[1]),
        ),
    )[1]


def _broaden_validated_support(
    raw_proposal: Mapping[str, Any],
    candidate: _Candidate,
) -> dict[str, Any]:
    proposal_spec = _proposal_extraction_spec(raw_proposal)
    cited_evidence = _unique_strings(
        raw_proposal["supporting_evidence_ids"],
        name="remote proposal supporting_evidence_ids",
    )
    cited_families = _ordered_families(
        _unique_strings(
            raw_proposal["supporting_source_families"],
            name="remote proposal supporting_source_families",
        )
    )
    if not set(cited_evidence) <= set(candidate.evidence_ids):
        raise ValueError("remote proposal cites evidence outside its candidate support")
    if not set(cited_families) <= set(candidate.source_families):
        raise ValueError("remote proposal cites a family outside its candidate support")
    rationale = raw_proposal["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("remote proposal rationale must be a non-empty string")
    contract = CandidateContract(proposal_spec, source_families=cited_families)
    if contract.extraction_spec != candidate.spec:
        raise ValueError("remote proposal changed its selected extraction contract")
    proposal = copy.deepcopy(candidate.spec)
    proposal.update(
        {
            "supporting_evidence_ids": list(candidate.evidence_ids),
            "supporting_source_families": list(candidate.source_families),
            "rationale": rationale,
        }
    )
    return proposal


def _backfill_proposal(candidate: _Candidate, *, rationale: str) -> dict[str, Any]:
    proposal = copy.deepcopy(candidate.spec)
    proposal.update(
        {
            "supporting_evidence_ids": list(candidate.evidence_ids),
            "supporting_source_families": list(candidate.source_families),
            "rationale": rationale,
        }
    )
    return proposal


@dataclass(frozen=True)
class MinimalStagedSelectionResult:
    _response_json: str = field(repr=False)
    remote_selected_candidate_ids: tuple[str, ...]
    mandatory_coverage_candidate_ids: tuple[str, ...]
    high_confidence_reserve_candidate_ids: tuple[str, ...]
    omitted_candidate_ids: tuple[str, ...]
    candidate_pool_target_source_families: tuple[str, ...]
    candidate_pool_source_family_counts: tuple[tuple[str, int], ...]
    original_request_source_families: tuple[str, ...]
    original_request_families_without_candidate: tuple[str, ...]
    target_roles: tuple[str, ...]
    covered_source_families: tuple[str, ...]
    covered_roles: tuple[str, ...]
    candidate_pool_coverage_complete: bool
    original_request_candidate_coverage_complete: bool
    high_confidence_reserve_complete: bool
    cap_limited: bool
    input_sha256: str
    output_sha256: str
    postprocessor_code_sha256: str
    input_schema: str = MINIMAL_STAGED_SELECTION_INPUT_SCHEMA
    output_schema: str = MINIMAL_STAGED_SELECTION_OUTPUT_SCHEMA
    version: str = MINIMAL_STAGED_SELECTION_POSTPROCESSOR_VERSION

    @property
    def response(self) -> dict[str, Any]:
        """Return a detached response; callers cannot mutate the sealed result."""

        value = json.loads(self._response_json)
        if not isinstance(value, dict):  # pragma: no cover - constructor seals an object
            raise RuntimeError("sealed staged-selection response is not an object")
        return value

    def audit(self) -> dict[str, Any]:
        """Return a content-free closed audit suitable for a fresh schema."""

        return {
            "schema_version": self.output_schema,
            "postprocessor_version": self.version,
            "postprocessor_code_sha256": self.postprocessor_code_sha256,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "remote_selected_candidate_ids": list(self.remote_selected_candidate_ids),
            "mandatory_coverage_candidate_ids": list(self.mandatory_coverage_candidate_ids),
            "high_confidence_reserve_candidate_ids": list(
                self.high_confidence_reserve_candidate_ids
            ),
            "omitted_candidate_ids": list(self.omitted_candidate_ids),
            "candidate_pool_target_source_families": list(
                self.candidate_pool_target_source_families
            ),
            "candidate_pool_source_family_counts": {
                family: count for family, count in self.candidate_pool_source_family_counts
            },
            "original_request_source_families": list(self.original_request_source_families),
            "original_request_families_without_candidate": list(
                self.original_request_families_without_candidate
            ),
            "target_roles": list(self.target_roles),
            "covered_source_families": list(self.covered_source_families),
            "covered_roles": list(self.covered_roles),
            "candidate_pool_coverage_complete": self.candidate_pool_coverage_complete,
            "original_request_candidate_coverage_complete": (
                self.original_request_candidate_coverage_complete
            ),
            "high_confidence_reserve_complete": self.high_confidence_reserve_complete,
            "cap_limited": self.cap_limited,
            "final_count": len(self.response["proposals"]),
        }


def postprocess_minimal_staged_selection(
    *,
    remote_response: Mapping[str, Any],
    remote_selected_candidate_ids: Sequence[str],
    candidate_pool: Sequence[Mapping[str, Any]],
    original_request_source_families: Sequence[str],
    max_candidates: int,
    minimum_recurrence: int = 2,
    minimum_independent_families: int = 2,
    minimum_distinct_evidence: int = 2,
) -> MinimalStagedSelectionResult:
    """Preserve remote judgment, then add bounded coverage/recall safeguards."""

    code_sha256 = _module_sha256()
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise TypeError("max_candidates must be an integer")
    if not 1 <= max_candidates <= _MAX_CANDIDATES:
        raise ValueError("max_candidates must be in [1, 64]")
    for name, value in (
        ("minimum_recurrence", minimum_recurrence),
        ("minimum_independent_families", minimum_independent_families),
        ("minimum_distinct_evidence", minimum_distinct_evidence),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(remote_response, Mapping) or set(remote_response) != {"proposals"}:
        raise ValueError("remote_response must match the closed proposal schema")
    raw_proposals = remote_response["proposals"]
    if not isinstance(raw_proposals, list):
        raise TypeError("remote_response.proposals must be a list")
    selected_ids = _unique_strings(
        remote_selected_candidate_ids,
        name="remote_selected_candidate_ids",
    )
    if any(not _CANDIDATE_ID.fullmatch(value) for value in selected_ids):
        raise ValueError("remote selected candidate IDs are not canonical")
    if len(selected_ids) != len(raw_proposals):
        raise ValueError("remote proposals must align with selected candidate IDs")
    if len(selected_ids) > max_candidates:
        raise ValueError("remote selection exceeds the configured candidate cap")

    raw_pool = tuple(candidate_pool)
    if not raw_pool or len(raw_pool) > _MAX_CANDIDATES:
        raise ValueError("candidate_pool size must be in [1, 64]")
    candidates = tuple(_candidate(raw, index=index) for index, raw in enumerate(raw_pool))
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate_pool candidate IDs must be unique")
    unknown_selected = set(selected_ids) - set(by_id)
    if unknown_selected:
        raise ValueError(f"remote selection contains unknown IDs: {sorted(unknown_selected)}")
    request_families = _ordered_families(
        _unique_strings(
            original_request_source_families,
            name="original_request_source_families",
        )
    )

    target_families = _ordered_families(
        tuple(family for candidate in candidates for family in candidate.source_families)
    )
    family_counts = tuple(
        (
            family,
            sum(family in candidate.source_families for candidate in candidates),
        )
        for family in target_families
    )
    target_roles = _ordered_roles(
        tuple(role for candidate in candidates for role in candidate.roles)
    )
    families_without_candidate = tuple(
        family for family in request_families if family not in set(target_families)
    )

    normalized_input = {
        "schema_version": MINIMAL_STAGED_SELECTION_INPUT_SCHEMA,
        "postprocessor_version": MINIMAL_STAGED_SELECTION_POSTPROCESSOR_VERSION,
        "remote_response": _detached(remote_response),
        "remote_selected_candidate_ids": list(selected_ids),
        "candidate_pool": [candidate.as_input_dict() for candidate in candidates],
        "original_request_source_families": list(request_families),
        "max_candidates": max_candidates,
        "minimum_recurrence": minimum_recurrence,
        "minimum_independent_families": minimum_independent_families,
        "minimum_distinct_evidence": minimum_distinct_evidence,
        "semantic_alias_mutation_performed": False,
    }
    input_sha256 = _sha256_json(normalized_input)

    proposals = [
        _broaden_validated_support(raw, by_id[candidate_id])
        for candidate_id, raw in zip(selected_ids, raw_proposals, strict=True)
    ]
    selected_set = set(selected_ids)
    covered_families = {
        family for candidate_id in selected_ids for family in by_id[candidate_id].source_families
    }
    covered_roles = {role for candidate_id in selected_ids for role in by_id[candidate_id].roles}
    missing_families = tuple(family for family in target_families if family not in covered_families)
    missing_roles = tuple(role for role in target_roles if role not in covered_roles)
    eligible = tuple(
        candidate for candidate in candidates if candidate.candidate_id not in selected_set
    )
    mandatory = _minimum_coverage_set(
        eligible,
        missing_families=missing_families,
        missing_roles=missing_roles,
        maximum_additions=max_candidates - len(proposals),
    )
    mandatory_ids = tuple(candidate.candidate_id for candidate in mandatory)
    for candidate in mandatory:
        proposals.append(
            _backfill_proposal(
                candidate,
                rationale=(
                    "Deterministic fold-honest minimum candidate-pool coverage "
                    "backfill from validated staged evidence."
                ),
            )
        )
        covered_families.update(candidate.source_families)
        covered_roles.update(candidate.roles)
    selected_set.update(mandatory_ids)

    reserve_eligible = tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if candidate.candidate_id not in selected_set
                and _is_high_confidence(
                    candidate,
                    minimum_recurrence=minimum_recurrence,
                    minimum_independent_families=minimum_independent_families,
                    minimum_distinct_evidence=minimum_distinct_evidence,
                )
            ),
            key=_quality_key,
        )
    )
    available_slots = max_candidates - len(proposals)
    reserve = reserve_eligible[:available_slots]
    reserve_ids = tuple(candidate.candidate_id for candidate in reserve)
    for candidate in reserve:
        proposals.append(
            _backfill_proposal(
                candidate,
                rationale=(
                    "Deterministic fold-honest high-confidence reserve from "
                    "recurrent or independently supported staged evidence."
                ),
            )
        )
        covered_families.update(candidate.source_families)
        covered_roles.update(candidate.roles)
    selected_set.update(reserve_ids)

    omitted_ids = tuple(
        candidate.candidate_id
        for candidate in candidates
        if candidate.candidate_id not in selected_set
    )
    if selected_set | set(omitted_ids) != set(by_id) or selected_set & set(omitted_ids):
        raise RuntimeError("staged-selection candidate ID partition is inconsistent")
    covered_family_order = _ordered_families(tuple(covered_families))
    covered_role_order = _ordered_roles(tuple(covered_roles))
    candidate_pool_complete = set(target_families) <= set(covered_family_order) and set(
        target_roles
    ) <= set(covered_role_order)
    original_request_complete = not families_without_candidate and set(request_families) <= set(
        covered_family_order
    )
    reserve_complete = len(reserve) == len(reserve_eligible)
    cap_limited = bool(not candidate_pool_complete or not reserve_complete)
    response = {"proposals": _detached(proposals)}
    output_sha256 = _sha256_json(response)
    if _module_sha256() != code_sha256:
        raise RuntimeError("staged-selection postprocessor code changed during evaluation")
    return MinimalStagedSelectionResult(
        _response_json=_canonical_json(response),
        remote_selected_candidate_ids=selected_ids,
        mandatory_coverage_candidate_ids=mandatory_ids,
        high_confidence_reserve_candidate_ids=reserve_ids,
        omitted_candidate_ids=omitted_ids,
        candidate_pool_target_source_families=target_families,
        candidate_pool_source_family_counts=family_counts,
        original_request_source_families=request_families,
        original_request_families_without_candidate=families_without_candidate,
        target_roles=target_roles,
        covered_source_families=covered_family_order,
        covered_roles=covered_role_order,
        candidate_pool_coverage_complete=candidate_pool_complete,
        original_request_candidate_coverage_complete=original_request_complete,
        high_confidence_reserve_complete=reserve_complete,
        cap_limited=cap_limited,
        input_sha256=input_sha256,
        output_sha256=output_sha256,
        postprocessor_code_sha256=code_sha256,
    )


__all__ = [
    "MINIMAL_STAGED_SELECTION_INPUT_SCHEMA",
    "MINIMAL_STAGED_SELECTION_OUTPUT_SCHEMA",
    "MINIMAL_STAGED_SELECTION_POSTPROCESSOR_VERSION",
    "MinimalStagedSelectionResult",
    "postprocess_minimal_staged_selection",
]
