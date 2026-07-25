"""Strict, role-independent interfaces for lossless Stage-1 discovery.

The historical all-evidence prompt path remains untouched in
``all_evidence_fusion``.  This module defines the smaller scientific jobs used
by the replacement path: interpret complementary evidence, consolidate without
loss, audit coverage/rejections, route roles deterministically, and freeze an
extraction definition.

Only concept-bearing evidence belongs in these interfaces.  Authenticated
row-aligned numerical signals are integrated by the final estimator through a
separate channel and can never ground a patient-feature name here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .hierarchical_discovery_response_contract import (
    HierarchyWireBudget,
    LEGACY_HIERARCHY_WIRE_BUDGET,
    attach_hierarchical_discovery_response_contract,
)

DISCOVERY_INTERFACE_SCHEMA_VERSION = "all_evidence_discovery_interfaces_v10"
DISCOVERY_WIRE_NORMALIZATION_VERSION = "atomic_occurrence_compiler_normalization_v3"
INTERPRET_JOB_VERSION = "interpret_complementary_evidence_chunk_v5"
CONSOLIDATE_JOB_VERSION = "lossless_candidate_consolidation_v4"
COVERAGE_CRITIC_JOB_VERSION = "complete_evidence_coverage_critic_v4"
REJECTION_CRITIC_JOB_VERSION = "complete_rejection_critic_v4"
CROSS_ARCHITECTURE_PLANNER_JOB_VERSION = "cross_architecture_lookback_planner_v4"
CROSS_ARCHITECTURE_INTEGRATION_JOB_VERSION = "cross_architecture_integration_v4"
ARCHITECTURE_DOSSIER_VERSION = "complete_architecture_dossier_v2"
ROLE_ROUTING_VERSION = "observable_axis_role_routing_v1"
EXTRACTION_DEFINITION_JOB_VERSION = "grounded_extraction_definition_v5"
EXTRACTION_VOCABULARY_GROUNDING_VERSION = "support_evidence_vocabulary_grounding_v2"

# These reserved values describe extraction mechanics, not clinical ontology.
# They are the only executable fallback when the exact supporting evidence
# establishes a value shape but does not spell out a clinical unit or category
# vocabulary.  All other vocabulary must occur literally in the supporting raw
# evidence for the one feature being extracted.
AS_DOCUMENTED_UNIT = "as_documented"
MECHANICAL_MENTION_CATEGORIES = ("not_mentioned", "mentioned")

BOW_NUISANCE = "bow_nuisance"
BOW_R_LOSS = "bow_r_loss"
HTR_NEURAL = "htr_neural"
MATCHED_PAIR_UPLIFT = "matched_pair_uplift"
EMBEDDING_WHOLE_COHORT = "embedding_whole_cohort"
EMBEDDING_CLUSTERED = "embedding_clustered"
TFIDF_SEMANTIC_RETRIEVAL = "tfidf_semantic_retrieval_contrasts"
TFIDF_TOPICS = "tfidf_topics"
TFIDF_ORPHAN_NGRAMS = "tfidf_orphan_ngrams"
NEURAL_QUERY_MOMENTS = "neural_query_moments"

# This tuple is the benchmark contract.  In particular, it deliberately does
# not include the inactive sparse-query fallback.
ACTIVE_STAGE1_CONCEPT_FAMILIES = (
    BOW_NUISANCE,
    BOW_R_LOSS,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    EMBEDDING_WHOLE_COHORT,
    EMBEDDING_CLUSTERED,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
    TFIDF_ORPHAN_NGRAMS,
    NEURAL_QUERY_MOMENTS,
)
ACTIVE_STAGE1_CONCEPT_FAMILY_SET = frozenset(ACTIVE_STAGE1_CONCEPT_FAMILIES)

DIRECT_UPSTREAM_NUMERICAL_CHANNEL = "direct_upstream_numerical"
DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST = "direct_upstream_numerical_manifest"
DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT = "first_gate_materialization_intent"
DIRECT_NUMERICAL_CONTRACT_KINDS = frozenset(
    {
        DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST,
        DIRECT_NUMERICAL_CONTRACT_KIND_FIRST_GATE_INTENT,
    }
)

TREATMENT_AXIS = "treatment"
OUTCOME_AXIS = "outcome"
HETEROGENEITY_AXIS = "heterogeneity"
PAIR_UPLIFT_AXIS = "pair_uplift"
EXTRACTION_SUPPORT_AXIS = "extraction_support"
OBSERVABLE_AXES = (
    TREATMENT_AXIS,
    OUTCOME_AXIS,
    HETEROGENEITY_AXIS,
    PAIR_UPLIFT_AXIS,
    EXTRACTION_SUPPORT_AXIS,
)
_OBSERVABLE_AXIS_SET = frozenset(OBSERVABLE_AXES)

_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_OPAQUE_ID = re.compile(r"^[a-z][a-z0-9_.:-]*$")
_FORBIDDEN_CONTENT_KEY = re.compile(
    r"(?:^|_)(?:"
    r"oracle|ground_truth|true_ite|true_cate|true_effect|"
    r"row_id|row_ids|patient_id|patient_ids|mrn|medical_record_number|"
    r"heldout_row_ids|validation_row_ids|test_row_ids|"
    r"raw_vector|raw_vectors|embedding_vector|embedding_vectors|activations|"
    r"backend_path|artifact_path|cache_path|full_note|full_notes|raw_note|raw_notes"
    r")(?:_|$)",
    flags=re.IGNORECASE,
)
_VALUE_SHAPES = frozenset({"continuous", "categorical", "ambiguous"})
_UNSAFE_MODEL_TEXT = re.compile(r"[\x00-\x1f\x7f-\x9f\ud800-\udfff]")


def canonical_json(value: Any) -> str:
    """Return the single canonical JSON representation used by these jobs."""

    try:
        result = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        result.encode("utf-8")
        return result
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "discovery values must be finite, JSON serializable, and valid UTF-8"
        ) from exc


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _detached(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one JSON object")
    return value


def _require_exact_keys(value: Mapping[str, Any], *, expected: Iterable[str], label: str) -> None:
    expected_set = frozenset(expected)
    actual = frozenset(value)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        extra = sorted(actual - expected_set)
        raise ValueError(f"{label} keys differ; missing={missing}, extra={extra}")


def _require_string(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if _UNSAFE_MODEL_TEXT.search(value) is not None:
        raise ValueError(f"{label} contains a forbidden control or surrogate code point")
    if not allow_empty and not value.strip():
        raise ValueError(f"{label} cannot be empty")
    return value


def _require_id(value: Any, *, label: str) -> str:
    result = _require_string(value, label=label)
    if _OPAQUE_ID.fullmatch(result) is None:
        raise ValueError(f"{label} must be an opaque lowercase identifier")
    return result


def _require_name(value: Any, *, label: str) -> str:
    result = _require_string(value, label=label)
    if _SNAKE_CASE.fullmatch(result) is None:
        raise ValueError(f"{label} must be lower snake_case")
    return result


def _require_string_list(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
    validate_item: Any = None,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    if not allow_empty and not value:
        raise ValueError(f"{label} cannot be empty")
    parsed: list[str] = []
    for index, item in enumerate(value):
        if validate_item is None:
            parsed.append(_require_string(item, label=f"{label}[{index}]"))
        else:
            parsed.append(validate_item(item, label=f"{label}[{index}]"))
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} cannot contain duplicates")
    return tuple(parsed)


def _require_exact_id_keyed_mapping(
    value: Any,
    *,
    identifiers: Sequence[str],
    label: str,
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Validate and request-order an exact-coverage keyed object.

    The strict JSON transport parser rejects duplicate keys before this helper
    runs.  Exact keys then provide grammar-supported exact-once coverage without
    relying on unsupported JSON-Schema ``uniqueItems``.
    """

    ordered_ids = tuple(identifiers)
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError(f"{label} expected identifiers cannot contain duplicates")
    row = _require_mapping(value, label=label)
    _require_exact_keys(row, expected=ordered_ids, label=label)
    return tuple(
        (
            identifier,
            _require_mapping(row[identifier], label=f"{label}.{identifier}"),
        )
        for identifier in ordered_ids
    )


def _require_exact_id_acknowledgements(
    value: Any,
    *,
    identifiers: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    ordered = _require_exact_id_keyed_mapping_values(
        value,
        identifiers=identifiers,
        label=label,
    )
    for identifier, acknowledgement in ordered:
        if acknowledgement is not True:
            raise ValueError(f"{label}.{identifier} must be true")
    return tuple(identifier for identifier, _ in ordered)


def _require_exact_id_keyed_mapping_values(
    value: Any,
    *,
    identifiers: Sequence[str],
    label: str,
) -> tuple[tuple[str, Any], ...]:
    ordered_ids = tuple(identifiers)
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError(f"{label} expected identifiers cannot contain duplicates")
    row = _require_mapping(value, label=label)
    _require_exact_keys(row, expected=ordered_ids, label=label)
    return tuple((identifier, row[identifier]) for identifier in ordered_ids)


def _scan_forbidden_content(value: Any, *, path: str = "content") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if _FORBIDDEN_CONTENT_KEY.search(key):
                raise ValueError(f"{path}.{key} is forbidden in discovery evidence")
            _scan_forbidden_content(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden_content(child, path=f"{path}[{index}]")


def extraction_vocabulary_grounding_policy() -> dict[str, Any]:
    """Return the closed, precommittable extraction-vocabulary policy."""

    return {
        "schema_version": EXTRACTION_VOCABULARY_GROUNDING_VERSION,
        "clinical_vocabulary": (
            "literal_casefolded_whitespace_normalized_leaf_value_from_exact_"
            "supporting_raw_evidence"
        ),
        "cross_field_phrase_assembly": False,
        "mechanical_encodings_are_clinical_ontology": False,
        "continuous_scale_fallback": AS_DOCUMENTED_UNIT,
        "continuous_scale_fallback_meaning": (
            "preserve the value exactly as documented; this is an extraction mechanic, "
            "not a clinical unit assertion"
        ),
        "categorical_observation_fallback": list(MECHANICAL_MENTION_CATEGORIES),
        "categorical_observation_fallback_meaning": (
            "whether the supported concept language is mentioned; this is an observation "
            "encoding, not a clinical status ontology"
        ),
        "mechanical_and_clinical_categories_may_mix": False,
    }


def _evidence_leaf_texts(value: Any) -> tuple[str, ...]:
    """Collect separate scalar leaves without joining unrelated evidence fields."""

    if isinstance(value, Mapping):
        return tuple(text for child in value.values() for text in _evidence_leaf_texts(child))
    if isinstance(value, (list, tuple)):
        return tuple(text for child in value for text in _evidence_leaf_texts(child))
    if isinstance(value, str):
        compact = " ".join(value.casefold().split())
        return (compact,) if compact else ()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (str(value).casefold(),)
    return ()


def _literal_supported_evidence_value(
    value: str,
    *,
    evidence: Sequence["DiscoveryEvidenceItem"],
    supporting_evidence_ids: Sequence[str],
) -> bool:
    """Require one literal phrase within one supporting raw-evidence leaf."""

    needle = " ".join(value.casefold().split())
    if not needle:
        return False
    support = set(supporting_evidence_ids)
    for item in evidence:
        if item.evidence_id not in support:
            continue
        for haystack in _evidence_leaf_texts(item.content):
            start = 0
            while True:
                index = haystack.find(needle, start)
                if index < 0:
                    break
                end = index + len(needle)
                left_ok = not needle[0].isalnum() or index == 0 or not haystack[index - 1].isalnum()
                right_ok = (
                    not needle[-1].isalnum() or end == len(haystack) or not haystack[end].isalnum()
                )
                if left_ok and right_ok:
                    return True
                start = index + 1
    return False


@dataclass(frozen=True)
class DiscoveryEvidenceItem:
    """One concept-bearing atom supplied to a discovery job."""

    evidence_id: str
    source_family: str
    observable_axes: tuple[str, ...]
    content: Mapping[str, Any]
    member_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.evidence_id, label="evidence_id")
        if self.source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError(f"inactive or unknown source_family: {self.source_family!r}")
        if not isinstance(self.observable_axes, tuple) or not self.observable_axes:
            raise ValueError("observable_axes must be a non-empty tuple")
        if len(set(self.observable_axes)) != len(self.observable_axes):
            raise ValueError("observable_axes cannot contain duplicates")
        unknown = set(self.observable_axes) - _OBSERVABLE_AXIS_SET
        if unknown:
            raise ValueError(f"unknown observable axes: {sorted(unknown)}")
        content = _require_mapping(self.content, label="content")
        if not content:
            raise ValueError("content cannot be empty")
        _scan_forbidden_content(content)
        if not isinstance(self.member_ids, tuple):
            raise TypeError("member_ids must be a tuple")
        if len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError("member_ids cannot contain duplicates")
        for index, member_id in enumerate(self.member_ids):
            _require_id(member_id, label=f"member_ids[{index}]")
        object.__setattr__(self, "content", _detached(content))

    def as_prompt_item(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_family": self.source_family,
            "observable_axes": list(self.observable_axes),
            "member_ids": list(self.member_ids),
            "content": _detached(self.content),
        }


@dataclass(frozen=True)
class DiscoveryCandidate:
    candidate_id: str
    feature_name: str
    description: str
    supporting_evidence_ids: tuple[str, ...]
    source_families: tuple[str, ...]
    value_shape_hypothesis: str
    unresolved_ambiguity: str = ""

    def __post_init__(self) -> None:
        _require_id(self.candidate_id, label="candidate_id")
        _require_name(self.feature_name, label="feature_name")
        _require_string(self.description, label="description")
        if not self.supporting_evidence_ids:
            raise ValueError("supporting_evidence_ids cannot be empty")
        for index, evidence_id in enumerate(self.supporting_evidence_ids):
            _require_id(evidence_id, label=f"supporting_evidence_ids[{index}]")
        if len(set(self.supporting_evidence_ids)) != len(self.supporting_evidence_ids):
            raise ValueError("supporting_evidence_ids cannot contain duplicates")
        if not self.source_families:
            raise ValueError("source_families cannot be empty")
        unknown = set(self.source_families) - ACTIVE_STAGE1_CONCEPT_FAMILY_SET
        if unknown:
            raise ValueError(f"unknown source_families: {sorted(unknown)}")
        if len(set(self.source_families)) != len(self.source_families):
            raise ValueError("source_families cannot contain duplicates")
        if self.value_shape_hypothesis not in _VALUE_SHAPES:
            raise ValueError("value_shape_hypothesis is invalid")
        _require_string(
            self.unresolved_ambiguity,
            label="unresolved_ambiguity",
            allow_empty=True,
        )

    def as_prompt_item(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "feature_name": self.feature_name,
            "description": self.description,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "source_families": list(self.source_families),
            "value_shape_hypothesis": self.value_shape_hypothesis,
            "unresolved_ambiguity": self.unresolved_ambiguity,
        }


def bounded_candidate_relation_pages(
    candidates: Sequence[DiscoveryCandidate],
    *,
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, Any], ...]:
    """Enumerate every unordered candidate pair once in bounded anchor pages."""

    items = tuple(candidates)
    _ensure_unique_ids(items, attribute="candidate_id", label="candidate")
    pages: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(items[:-1]):
        later = items[anchor_index + 1 :]
        for page_offset in range(
            0,
            len(later),
            wire_budget.max_pair_relation_peers_per_page,
        ):
            peers = later[
                page_offset : page_offset + wire_budget.max_pair_relation_peers_per_page
            ]
            identity = {
                "anchor_candidate_id": anchor.candidate_id,
                "peer_candidate_ids": [peer.candidate_id for peer in peers],
            }
            pages.append(
                {
                    "relation_page_id": f"relation_page_{content_sha256(identity)}",
                    "anchor_candidate_id": anchor.candidate_id,
                    "peer_candidate_ids": [peer.candidate_id for peer in peers],
                    "pair_count": len(peers),
                }
            )
    expected_pairs = len(items) * (len(items) - 1) // 2
    if sum(int(page["pair_count"]) for page in pages) != expected_pairs:
        raise AssertionError("bounded candidate relation schedule lost an unordered pair")
    return tuple(_detached(page) for page in pages)


def validate_candidate_relation_page_response(
    response: Any,
    *,
    anchor_candidate_id: str,
    peer_candidate_ids: Sequence[str],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> dict[str, Any]:
    """Validate one exact ternary-relation page and normalize it to pair rows."""

    anchor = _require_id(anchor_candidate_id, label="anchor_candidate_id")
    peers = tuple(
        _require_id(value, label=f"peer_candidate_ids[{index}]")
        for index, value in enumerate(peer_candidate_ids)
    )
    if not peers or len(peers) > wire_budget.max_pair_relation_peers_per_page:
        raise ValueError("peer_candidate_ids violates the bounded relation-page contract")
    if len(peers) != len(set(peers)) or anchor in peers:
        raise ValueError("candidate relation page contains a duplicate/self pair")
    root = _require_mapping(response, label="candidate relation page response")
    _require_exact_keys(
        root,
        expected={"comparisons"},
        label="candidate relation page response",
    )
    comparisons = _require_exact_id_keyed_mapping(
        root["comparisons"],
        identifiers=peers,
        label="comparisons",
    )
    rows: list[dict[str, str]] = []
    for peer_id, raw in comparisons:
        label = f"comparisons.{peer_id}"
        _require_exact_keys(raw, expected={"relation", "reason"}, label=label)
        relation = _require_string(raw["relation"], label=f"{label}.relation")
        if relation not in {"same_construct", "distinct", "uncertain"}:
            raise ValueError(f"{label}.relation is invalid")
        rows.append(
            {
                "anchor_candidate_id": anchor,
                "peer_candidate_id": peer_id,
                "relation": relation,
                "reason": _require_string(raw["reason"], label=f"{label}.reason"),
            }
        )
    return _detached(
        {
            "anchor_candidate_id": anchor,
            "peer_candidate_ids": list(peers),
            "pair_relations": rows,
        }
    )


def compile_complete_link_candidate_groups(
    *,
    candidate_ids: Sequence[str],
    relation_pages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile exhaustive pair judgments without transitive false-positive collapse."""

    ordered_ids = tuple(
        _require_id(value, label=f"candidate_ids[{index}]")
        for index, value in enumerate(candidate_ids)
    )
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("candidate_ids cannot contain duplicates")
    order = {candidate_id: index for index, candidate_id in enumerate(ordered_ids)}
    expected_pairs = {
        (ordered_ids[left], ordered_ids[right])
        for left in range(len(ordered_ids))
        for right in range(left + 1, len(ordered_ids))
    }
    pair_rows: dict[tuple[str, str], dict[str, str]] = {}
    for page_index, raw_page in enumerate(relation_pages):
        page = _require_mapping(raw_page, label=f"relation_pages[{page_index}]")
        rows = page.get("pair_relations")
        if not isinstance(rows, list):
            raise TypeError(f"relation_pages[{page_index}].pair_relations must be a list")
        for row_index, raw_row in enumerate(rows):
            row = _require_mapping(
                raw_row,
                label=f"relation_pages[{page_index}].pair_relations[{row_index}]",
            )
            left = _require_id(
                row.get("anchor_candidate_id"),
                label=f"relation_pages[{page_index}].pair_relations[{row_index}].anchor",
            )
            right = _require_id(
                row.get("peer_candidate_id"),
                label=f"relation_pages[{page_index}].pair_relations[{row_index}].peer",
            )
            if left not in order or right not in order or order[left] >= order[right]:
                raise ValueError("relation row is not one canonical earlier-to-later pair")
            pair = (left, right)
            if pair in pair_rows:
                raise ValueError("one unordered candidate pair was judged more than once")
            relation = _require_string(
                row.get("relation"), label=f"relation_pages[{page_index}].relation"
            )
            if relation not in {"same_construct", "distinct", "uncertain"}:
                raise ValueError("relation row carries an invalid ternary relation")
            pair_rows[pair] = {
                "relation": relation,
                "reason": _require_string(
                    row.get("reason"), label=f"relation_pages[{page_index}].reason"
                ),
            }
    missing = expected_pairs - set(pair_rows)
    extra = set(pair_rows) - expected_pairs
    if missing or extra:
        raise ValueError("relation pages do not cover every unordered candidate pair exactly once")

    groups: list[list[str]] = []
    multi_fit_events: list[dict[str, Any]] = []
    for candidate_id in ordered_ids:
        fitting = [
            group_index
            for group_index, members in enumerate(groups)
            if all(
                pair_rows[(member, candidate_id)]["relation"] == "same_construct"
                for member in members
            )
        ]
        if fitting:
            chosen = fitting[0]
            if len(fitting) > 1:
                multi_fit_events.append(
                    {
                        "candidate_id": candidate_id,
                        "fitting_group_ordinals": [index + 1 for index in fitting],
                        "chosen_group_ordinal": chosen + 1,
                    }
                )
            groups[chosen].append(candidate_id)
        else:
            groups.append([candidate_id])

    group_rows: list[dict[str, Any]] = []
    group_by_candidate: dict[str, str] = {}
    for ordinal, members in enumerate(groups, start=1):
        group_id = f"candidate_group_{content_sha256({'members': members})}"
        group_rows.append(
            {
                "group_id": group_id,
                "group_ordinal": ordinal,
                "member_candidate_ids": list(members),
            }
        )
        group_by_candidate.update({candidate_id: group_id for candidate_id in members})
    cross_group_same_edges = [
        {
            "left_candidate_id": left,
            "right_candidate_id": right,
            "reason": row["reason"],
        }
        for (left, right), row in pair_rows.items()
        if row["relation"] == "same_construct"
        and group_by_candidate[left] != group_by_candidate[right]
    ]
    return _detached(
        {
            "groups": group_rows,
            "pair_relation_audit": {
                "audit_version": "exhaustive_complete_link_pair_compiler_v1",
                "candidate_order": list(ordered_ids),
                "expected_unordered_pair_count": len(expected_pairs),
                "observed_unordered_pair_count": len(pair_rows),
                "relation_counts": {
                    relation: sum(row["relation"] == relation for row in pair_rows.values())
                    for relation in ("same_construct", "distinct", "uncertain")
                },
                "multi_fit_events": multi_fit_events,
                "cross_group_same_construct_edges": cross_group_same_edges,
                "distinct_and_uncertain_edges_never_merge_groups": True,
            },
        }
    )


def candidate_definition_fold_batches(
    *,
    group_id: str,
    member_candidate_ids: Sequence[str],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, Any], ...]:
    """Schedule terminating accumulator folds for one proven candidate group."""

    canonical_group_id = _require_id(group_id, label="group_id")
    members = tuple(
        _require_id(value, label=f"member_candidate_ids[{index}]")
        for index, value in enumerate(member_candidate_ids)
    )
    if not members or len(members) != len(set(members)):
        raise ValueError("definition fold requires unique non-empty group membership")
    batches: list[dict[str, Any]] = []
    consumed = 0
    fold_index = 0
    while consumed < len(members):
        capacity = (
            wire_budget.max_definition_fold_inputs
            if fold_index == 0
            else wire_budget.max_definition_fold_inputs - 1
        )
        if capacity < 1:
            raise ValueError(
                "max_definition_fold_inputs must be at least two when an "
                "accumulator fold is required"
            )
        fresh = members[consumed : consumed + capacity]
        batches.append(
            {
                "group_id": canonical_group_id,
                "fold_index": fold_index,
                "uses_prior_accumulator": fold_index > 0,
                "member_candidate_ids": list(fresh),
            }
        )
        consumed += len(fresh)
        fold_index += 1
    return tuple(_detached(batch) for batch in batches)


def _ensure_unique_ids(items: Sequence[Any], *, attribute: str, label: str) -> None:
    values = [getattr(item, attribute) for item in items]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} IDs must be unique")


def interpret_evidence_chunk_context(
    *, family_explanation: str, evidence: Sequence[DiscoveryEvidenceItem]
) -> dict[str, Any]:
    explanation = _require_string(family_explanation, label="family_explanation")
    items = tuple(evidence)
    if not items:
        raise ValueError("interpretation evidence cannot be empty")
    _ensure_unique_ids(items, attribute="evidence_id", label="evidence")
    families = {item.source_family for item in items}
    if len(families) != 1:
        raise ValueError("one interpretation request must contain exactly one architecture")
    return {
        "job": "interpret_evidence_chunk",
        "schema_version": INTERPRET_JOB_VERSION,
        "family_explanation": explanation,
        "evidence": [item.as_prompt_item() for item in items],
    }


INTERPRET_SYSTEM_PROMPT = """You interpret concept-bearing Stage 1 evidence from one training fold. Inspect every supplied clue and build a broad inventory of concrete patient characteristics. Infer only what the visible words, phrases, topics, or semantic witnesses support. Do not estimate effects, assign causal roles, invent benchmark variables, aliases, units, categories, patient facts, or directions. Numerical summaries alone cannot name a feature. Return JSON only in the requested shape. The evidence_dispositions object is keyed by every exact evidence_id, and each member_dispositions object is keyed by every exact member_id owned by that evidence. Put a self-contained finding directly under the exact evidence or member that supports it; use evidence_findings for a whole evidence atom or a zero-member atom. Empty findings explicitly mean that exact atom was reviewed without a specific patient concept. Do not repeat concept names in reference lists or report support, status, or parent unions; those relations are derived locally."""


def render_interpret_evidence_chunk_messages(
    *,
    family_explanation: str,
    evidence: Sequence[DiscoveryEvidenceItem],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    context = interpret_evidence_chunk_context(
        family_explanation=family_explanation,
        evidence=evidence,
    )
    # The version remains part of the authenticated job envelope.  It is
    # deliberately not scientific context for the model.
    context.pop("schema_version")
    request = attach_hierarchical_discovery_response_contract(
        job_kind="interpret_architecture_chunk",
        request=context,
        wire_budget=wire_budget,
    )
    return (
        {"role": "system", "content": INTERPRET_SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(request)},
    )


_INTERPRET_WIRE_NORMALIZATION_AUDIT_VERSION = "atomic_interpret_wire_normalization_audit_v1"


def _validated_interpret_finding(
    value: Any,
    *,
    label: str,
    include_compiler_fields: bool = False,
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> dict[str, str]:
    finding = _require_mapping(value, label=label)
    if not include_compiler_fields:
        _require_exact_keys(
            finding,
            expected={
                "feature_name",
                "description",
                "value_shape_hypothesis",
                "unresolved_ambiguity",
            },
            label=label,
        )
    name = _require_name(finding.get("feature_name"), label=f"{label}.feature_name")
    description = _require_string(finding.get("description"), label=f"{label}.description")
    shape = finding.get("value_shape_hypothesis")
    if shape not in _VALUE_SHAPES:
        raise ValueError(f"{label}.value_shape_hypothesis is invalid")
    ambiguity = _require_string(
        finding.get("unresolved_ambiguity"),
        label=f"{label}.unresolved_ambiguity",
        allow_empty=True,
    )
    bounds = (
        (name, wire_budget.max_interpret_name_chars, "feature_name"),
        (
            description,
            wire_budget.max_interpret_description_chars,
            "description",
        ),
        (
            ambiguity,
            wire_budget.max_interpret_ambiguity_chars,
            "unresolved_ambiguity",
        ),
    )
    for text, maximum, field_name in bounds:
        if len(text) > maximum:
            raise ValueError(f"{label}.{field_name} exceeds its wire bound")
    return {
        "feature_name": name,
        "description": description,
        "value_shape_hypothesis": str(shape),
        "unresolved_ambiguity": ambiguity,
    }


def _interpret_occurrence_id(
    *,
    evidence: DiscoveryEvidenceItem,
    member_id: str,
    finding_ordinal: int,
    finding: Mapping[str, Any],
) -> str:
    identity = {
        "wire_normalization_version": DISCOVERY_WIRE_NORMALIZATION_VERSION,
        "source_family": evidence.source_family,
        "evidence_id": evidence.evidence_id,
        "member_id": member_id,
        "finding_ordinal": finding_ordinal,
        "finding": _detached(finding),
    }
    return f"interpret_occurrence_{content_sha256(identity)}"


def _interpret_findings(
    value: Any,
    *,
    label: str,
    evidence: DiscoveryEvidenceItem,
    member_id: str,
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    if len(value) > wire_budget.max_findings_per_atomic_review:
        raise ValueError(f"{label} exceeds its atomic proposal bound")
    concepts: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(value):
        finding = _validated_interpret_finding(
            raw,
            label=f"{label}[{ordinal}]",
            wire_budget=wire_budget,
        )
        concepts.append(
            {
                "concept_occurrence_id": _interpret_occurrence_id(
                    evidence=evidence,
                    member_id=member_id,
                    finding_ordinal=ordinal,
                    finding=finding,
                ),
                **finding,
                "supporting_evidence_ids": [evidence.evidence_id],
                "origin": {
                    "evidence_id": evidence.evidence_id,
                    "member_id": member_id,
                    "finding_ordinal": ordinal,
                },
            }
        )
    return concepts


def _interpret_wire_normalization_audit(
    concepts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for concept in concepts:
        by_name.setdefault(str(concept["feature_name"]), []).append(concept)
    duplicate_groups: list[dict[str, Any]] = []
    for name, occurrences in by_name.items():
        if len(occurrences) < 2:
            continue
        descriptions = tuple(dict.fromkeys(str(row["description"]) for row in occurrences))
        shapes = tuple(dict.fromkeys(str(row["value_shape_hypothesis"]) for row in occurrences))
        ambiguities = tuple(dict.fromkeys(str(row["unresolved_ambiguity"]) for row in occurrences))
        duplicate_groups.append(
            {
                "feature_name": name,
                "concept_occurrence_ids": [
                    str(row["concept_occurrence_id"]) for row in occurrences
                ],
                "supporting_evidence_ids": list(
                    dict.fromkeys(
                        str(evidence_id)
                        for row in occurrences
                        for evidence_id in row["supporting_evidence_ids"]
                    )
                ),
                "description_variants": list(descriptions),
                "value_shape_hypothesis_variants": list(shapes),
                "unresolved_ambiguity_variants": list(ambiguities),
                "definition_conflict": (
                    len(descriptions) > 1 or len(shapes) > 1 or len(ambiguities) > 1
                ),
            }
        )
    return {
        "audit_version": _INTERPRET_WIRE_NORMALIZATION_AUDIT_VERSION,
        "occurrence_policy": ("retain_every_atomic_occurrence_no_interpret_name_merge_v1"),
        "derived_relation_fields": [
            "concept_occurrence_id",
            "supporting_evidence_ids",
            "evidence_dispositions.status",
            "evidence_dispositions.feature_names",
            "member_dispositions.feature_names",
        ],
        "occurrence_count": len(concepts),
        "duplicate_feature_name_groups": duplicate_groups,
    }


def interpretation_model_view(response: Mapping[str, Any]) -> dict[str, Any]:
    """Project normalized interpretation science into a machine-metadata-free prompt view."""

    root = _require_mapping(response, label="normalized interpretation model view source")
    concepts = root.get("concepts")
    dispositions = root.get("evidence_dispositions")
    if not isinstance(concepts, list) or not isinstance(dispositions, list):
        raise TypeError("normalized interpretation model view source is incomplete")
    projected_concepts: list[dict[str, Any]] = []
    for index, raw in enumerate(concepts):
        concept = _require_mapping(raw, label=f"normalized concepts[{index}]")
        projected_concepts.append(
            {
                "feature_name": concept["feature_name"],
                "description": concept["description"],
                "value_shape_hypothesis": concept["value_shape_hypothesis"],
                "supporting_evidence_ids": _detached(concept["supporting_evidence_ids"]),
                "unresolved_ambiguity": concept["unresolved_ambiguity"],
            }
        )
    return _detached(
        {
            "concepts": projected_concepts,
            "evidence_dispositions": dispositions,
        }
    )


def validate_interpret_evidence_chunk_response(
    response: Any,
    *,
    evidence: Sequence[DiscoveryEvidenceItem],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> dict[str, Any]:
    """Compile atomic inline findings into the legacy ordered projection.

    The model is responsible only for an exact review of each authenticated
    evidence/member atom and for the text of any finding attached to that atom.
    Occurrence identity, evidence support, member/parent unions, and status are
    compiler-owned relations.  Equal generated names remain distinct atomic
    occurrences here and may be merged only by the authenticated family
    consolidation job.
    """

    items = tuple(evidence)
    if not items:
        raise ValueError("interpretation evidence cannot be empty")
    _ensure_unique_ids(items, attribute="evidence_id", label="evidence")
    root = _require_mapping(response, label="interpret response")
    _require_exact_keys(
        root,
        expected={"evidence_dispositions"},
        label="interpret response",
    )
    ordered_dispositions = _require_exact_id_keyed_mapping(
        root["evidence_dispositions"],
        identifiers=tuple(item.evidence_id for item in items),
        label="evidence_dispositions",
    )
    items_by_id = {item.evidence_id: item for item in items}
    concepts: list[dict[str, Any]] = []
    normalized_dispositions: list[dict[str, Any]] = []
    for evidence_id, disposition in ordered_dispositions:
        label = f"evidence_dispositions.{evidence_id}"
        _require_exact_keys(
            disposition,
            expected={
                "evidence_findings",
                "member_dispositions",
                "reason",
            },
            label=label,
        )
        evidence_findings = _interpret_findings(
            disposition["evidence_findings"],
            label=f"{label}.evidence_findings",
            evidence=items_by_id[evidence_id],
            member_id="",
            wire_budget=wire_budget,
        )
        concepts.extend(evidence_findings)
        evidence_feature_names = [row["feature_name"] for row in evidence_findings]
        expected_member_ids = items_by_id[evidence_id].member_ids
        ordered_members = _require_exact_id_keyed_mapping(
            disposition["member_dispositions"],
            identifiers=expected_member_ids,
            label=f"{label}.member_dispositions",
        )
        normalized_members: list[dict[str, Any]] = []
        for member_id, member in ordered_members:
            member_label = f"{label}.member_dispositions.{member_id}"
            _require_exact_keys(
                member,
                expected={"findings"},
                label=member_label,
            )
            member_findings = _interpret_findings(
                member["findings"],
                label=f"{member_label}.findings",
                evidence=items_by_id[evidence_id],
                member_id=member_id,
                wire_budget=wire_budget,
            )
            concepts.extend(member_findings)
            parsed_member_names = tuple(
                dict.fromkeys(row["feature_name"] for row in member_findings)
            )
            evidence_feature_names.extend(parsed_member_names)
            normalized_members.append(
                {"member_id": member_id, "feature_names": list(parsed_member_names)}
            )
        names = tuple(dict.fromkeys(evidence_feature_names))
        status = "supports_concept" if names else "reviewed_no_specific_concept"
        reason = _require_string(disposition["reason"], label=f"{label}.reason")
        if len(reason) > wire_budget.max_interpret_reason_chars:
            raise ValueError(f"{label}.reason exceeds its wire bound")
        normalized_dispositions.append(
            {
                "evidence_id": evidence_id,
                "status": status,
                "feature_names": list(names),
                "member_dispositions": normalized_members,
                "reason": reason,
            }
        )
    return _detached(
        {
            "concepts": concepts,
            "evidence_dispositions": normalized_dispositions,
            "wire_normalization_audit": _interpret_wire_normalization_audit(concepts),
        }
    )


def revalidate_normalized_interpret_evidence_chunk_response(
    response: Any,
    *,
    evidence: Sequence[DiscoveryEvidenceItem],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> dict[str, Any]:
    """Revalidate only the deterministic internal ordered projection."""
    items = tuple(evidence)
    if not items:
        raise ValueError("interpretation evidence cannot be empty")
    _ensure_unique_ids(items, attribute="evidence_id", label="evidence")
    root = _require_mapping(response, label="normalized interpret response")
    _require_exact_keys(
        root,
        expected={"concepts", "evidence_dispositions", "wire_normalization_audit"},
        label="normalized interpret response",
    )
    raw_concepts = root["concepts"]
    if not isinstance(raw_concepts, list):
        raise TypeError("normalized concepts must be a JSON list")
    items_by_id = {item.evidence_id: item for item in items}
    concepts: list[dict[str, Any]] = []
    seen_occurrence_ids: set[str] = set()
    seen_origins: set[tuple[str, str, int]] = set()
    for index, raw in enumerate(raw_concepts):
        concept = _require_mapping(raw, label=f"normalized concepts[{index}]")
        _require_exact_keys(
            concept,
            expected={
                "concept_occurrence_id",
                "feature_name",
                "description",
                "value_shape_hypothesis",
                "supporting_evidence_ids",
                "unresolved_ambiguity",
                "origin",
            },
            label=f"normalized concepts[{index}]",
        )
        origin = _require_mapping(concept["origin"], label=f"normalized concepts[{index}].origin")
        _require_exact_keys(
            origin,
            expected={"evidence_id", "member_id", "finding_ordinal"},
            label=f"normalized concepts[{index}].origin",
        )
        evidence_id = _require_id(
            origin["evidence_id"], label=f"normalized concepts[{index}].origin.evidence_id"
        )
        if evidence_id not in items_by_id:
            raise ValueError("normalized concept origin cites unsupplied evidence")
        member_id = _require_string(
            origin["member_id"],
            label=f"normalized concepts[{index}].origin.member_id",
            allow_empty=True,
        )
        if member_id:
            _require_id(member_id, label=f"normalized concepts[{index}].origin.member_id")
            if member_id not in items_by_id[evidence_id].member_ids:
                raise ValueError("normalized concept origin cites a non-owned member")
        ordinal = origin["finding_ordinal"]
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
            or ordinal >= wire_budget.max_findings_per_atomic_review
        ):
            raise ValueError("normalized concept finding_ordinal is outside its atomic bound")
        origin_key = (evidence_id, member_id, ordinal)
        if origin_key in seen_origins:
            raise ValueError("normalized concepts repeat one atomic finding origin")
        seen_origins.add(origin_key)
        finding = _validated_interpret_finding(
            concept,
            label=f"normalized concepts[{index}]",
            include_compiler_fields=True,
            wire_budget=wire_budget,
        )
        occurrence_id = _require_id(
            concept["concept_occurrence_id"],
            label=f"normalized concepts[{index}].concept_occurrence_id",
        )
        if occurrence_id in seen_occurrence_ids:
            raise ValueError("normalized concepts repeat a concept_occurrence_id")
        seen_occurrence_ids.add(occurrence_id)
        expected_occurrence_id = _interpret_occurrence_id(
            evidence=items_by_id[evidence_id],
            member_id=member_id,
            finding_ordinal=ordinal,
            finding=finding,
        )
        if occurrence_id != expected_occurrence_id:
            raise ValueError("normalized concept_occurrence_id is not compiler-derived")
        support = _require_string_list(
            concept["supporting_evidence_ids"],
            label=f"normalized concepts[{index}].supporting_evidence_ids",
            validate_item=_require_id,
        )
        if support != (evidence_id,):
            raise ValueError("normalized concept support is not its owning evidence")
        concepts.append(_detached(concept))

    evidence_rank = {item.evidence_id: index for index, item in enumerate(items)}
    member_rank = {
        (item.evidence_id, member_id): index + 1
        for item in items
        for index, member_id in enumerate(item.member_ids)
    }
    expected_order = sorted(
        concepts,
        key=lambda row: (
            evidence_rank[row["origin"]["evidence_id"]],
            member_rank.get(
                (row["origin"]["evidence_id"], row["origin"]["member_id"]),
                0,
            ),
            row["origin"]["finding_ordinal"],
        ),
    )
    if canonical_json(expected_order) != canonical_json(concepts):
        raise ValueError("normalized concepts do not follow deterministic atomic traversal")

    dispositions = root["evidence_dispositions"]
    if not isinstance(dispositions, list):
        raise TypeError("normalized evidence_dispositions must be a JSON list")
    if len(dispositions) != len(items):
        raise ValueError("normalized evidence dispositions lack exact evidence coverage")
    for index, (raw, item) in enumerate(zip(dispositions, items)):
        label = f"normalized evidence_dispositions[{index}]"
        row = _require_mapping(raw, label=label)
        _require_exact_keys(
            row,
            expected={"evidence_id", "status", "feature_names", "member_dispositions", "reason"},
            label=label,
        )
        if row["evidence_id"] != item.evidence_id:
            raise ValueError("normalized evidence dispositions changed request order")
        owned_concepts = [
            concept for concept in concepts if concept["origin"]["evidence_id"] == item.evidence_id
        ]
        expected_names = tuple(dict.fromkeys(concept["feature_name"] for concept in owned_concepts))
        names = _require_string_list(
            row["feature_names"],
            label=f"{label}.feature_names",
            allow_empty=True,
            validate_item=_require_name,
        )
        if names != expected_names:
            raise ValueError("normalized evidence feature_names are not compiler-derived")
        expected_status = "supports_concept" if expected_names else "reviewed_no_specific_concept"
        if row["status"] != expected_status:
            raise ValueError("normalized evidence status is not compiler-derived")
        reason = _require_string(row["reason"], label=f"{label}.reason")
        if len(reason) > wire_budget.max_interpret_reason_chars:
            raise ValueError(f"{label}.reason exceeds its wire bound")
        members = row["member_dispositions"]
        if not isinstance(members, list):
            raise TypeError("normalized member_dispositions must be a JSON list")
        if len(members) != len(item.member_ids):
            raise ValueError("normalized member dispositions lack exact member coverage")
        for member_index, (raw_member, expected_member_id) in enumerate(
            zip(members, item.member_ids)
        ):
            member = _require_mapping(
                raw_member,
                label=f"{label}.member_dispositions[{member_index}]",
            )
            _require_exact_keys(
                member,
                expected={"member_id", "feature_names"},
                label=f"{label}.member_dispositions[{member_index}]",
            )
            if member["member_id"] != expected_member_id:
                raise ValueError("normalized member dispositions changed request order")
            expected_member_names = tuple(
                dict.fromkeys(
                    concept["feature_name"]
                    for concept in owned_concepts
                    if concept["origin"]["member_id"] == expected_member_id
                )
            )
            member_names = _require_string_list(
                member["feature_names"],
                label=f"{label}.member_dispositions[{member_index}].feature_names",
                allow_empty=True,
                validate_item=_require_name,
            )
            if member_names != expected_member_names:
                raise ValueError("normalized member feature_names are not compiler-derived")
    expected_audit = _interpret_wire_normalization_audit(concepts)
    if canonical_json(root["wire_normalization_audit"]) != canonical_json(expected_audit):
        raise ValueError("normalized interpret audit is not compiler-derived")
    return _detached(root)


def consolidate_candidate_context(
    *, source_family: str, candidates: Sequence[DiscoveryCandidate]
) -> dict[str, Any]:
    if source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError(f"inactive or unknown source_family: {source_family!r}")
    items = tuple(candidates)
    _ensure_unique_ids(items, attribute="candidate_id", label="candidate")
    if any(item.source_families != (source_family,) for item in items):
        raise ValueError("within-architecture candidates must cite exactly that architecture")
    return {
        "job": "consolidate_candidate_ledger",
        "schema_version": CONSOLIDATE_JOB_VERSION,
        "source_family": source_family,
        "candidates": [item.as_prompt_item() for item in items],
    }


def _derive_unique_slot_name(
    *,
    proposed: str,
    slot: str,
    used: set[str],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> str:
    if proposed not in used:
        return proposed
    ordinal = 1
    while True:
        suffix = f"_{slot}" if ordinal == 1 else f"_{slot}_{ordinal}"
        available = wire_budget.max_generated_name_chars - len(suffix)
        if available < 1:
            raise ValueError(
                "generated-name wire budget cannot encode a compiler-owned "
                "disambiguation suffix"
            )
        prefix = proposed[:available].rstrip("_") or "feature"
        derived = f"{prefix}{suffix}"
        if derived not in used:
            return derived
        ordinal += 1


def validate_consolidation_response(
    response: Any,
    *,
    source_family: str,
    candidates: Sequence[DiscoveryCandidate],
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> dict[str, Any]:
    if source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise ValueError(f"inactive or unknown source_family: {source_family!r}")
    items = tuple(candidates)
    _ensure_unique_ids(items, attribute="candidate_id", label="candidate")
    if any(item.source_families != (source_family,) for item in items):
        raise ValueError("within-architecture candidates must cite exactly that architecture")
    by_id = {item.candidate_id: item for item in items}
    slots = tuple(f"consolidation_slot_{index:03d}" for index in range(1, len(items) + 1))
    root = _require_mapping(response, label="consolidation response")
    _require_exact_keys(
        root,
        expected={"candidate_assignments", "slot_definitions"},
        label="consolidation response",
    )
    ordered_assignments = _require_exact_id_keyed_mapping(
        root["candidate_assignments"],
        identifiers=tuple(item.candidate_id for item in items),
        label="candidate_assignments",
    )
    slot_by_candidate: dict[str, str] = {}
    reason_by_candidate: dict[str, str] = {}
    assignment_audit: list[dict[str, str]] = []
    for candidate_id, assignment in ordered_assignments:
        label = f"candidate_assignments.{candidate_id}"
        _require_exact_keys(assignment, expected={"cluster_slot", "reason"}, label=label)
        slot = _require_id(assignment["cluster_slot"], label=f"{label}.cluster_slot")
        if slot not in slots:
            raise ValueError(f"{label}.cluster_slot is not compiler-owned")
        reason = _require_string(assignment["reason"], label=f"{label}.reason")
        slot_by_candidate[candidate_id] = slot
        reason_by_candidate[candidate_id] = reason
        assignment_audit.append({"candidate_id": candidate_id, "cluster_slot": slot})

    ordered_definitions = _require_exact_id_keyed_mapping(
        root["slot_definitions"], identifiers=slots, label="slot_definitions"
    )
    definitions: dict[str, dict[str, str]] = {}
    definition_audit: list[dict[str, str]] = []
    for slot, raw_definition in ordered_definitions:
        label = f"slot_definitions.{slot}"
        _require_exact_keys(
            raw_definition,
            expected={"canonical_name", "description", "unresolved_ambiguity"},
            label=label,
        )
        definition = {
            "canonical_name": _require_name(
                raw_definition["canonical_name"], label=f"{label}.canonical_name"
            ),
            "description": _require_string(
                raw_definition["description"], label=f"{label}.description"
            ),
            "unresolved_ambiguity": _require_string(
                raw_definition["unresolved_ambiguity"],
                label=f"{label}.unresolved_ambiguity",
                allow_empty=True,
            ),
        }
        definitions[slot] = definition
        definition_audit.append({"cluster_slot": slot, **definition})

    assigned_slots = set(slot_by_candidate.values())
    active_slots = tuple(slot for slot in slots if slot in assigned_slots)
    used_names: set[str] = set()
    output_name_by_slot: dict[str, str] = {}
    disambiguations: list[dict[str, str]] = []
    for slot in active_slots:
        proposed = definitions[slot]["canonical_name"]
        derived = _derive_unique_slot_name(
            proposed=proposed,
            slot=slot,
            used=used_names,
            wire_budget=wire_budget,
        )
        if derived != proposed:
            disambiguations.append(
                {
                    "cluster_slot": slot,
                    "proposed_canonical_name": proposed,
                    "derived_canonical_name": derived,
                    "reason": "distinct active slots cannot share canonical identity",
                }
            )
        used_names.add(derived)
        output_name_by_slot[slot] = derived

    normalized_concepts: list[dict[str, Any]] = []
    for slot in active_slots:
        members = tuple(
            item.candidate_id for item in items if slot_by_candidate[item.candidate_id] == slot
        )
        supporting_evidence = tuple(
            dict.fromkeys(
                evidence_id
                for member in members
                for evidence_id in by_id[member].supporting_evidence_ids
            )
        )
        source_families = tuple(
            dict.fromkeys(family for member in members for family in by_id[member].source_families)
        )
        member_shapes = {by_id[member].value_shape_hypothesis for member in members}
        shape = next(iter(member_shapes)) if len(member_shapes) == 1 else "ambiguous"
        normalized_concepts.append(
            {
                "canonical_name": output_name_by_slot[slot],
                "description": definitions[slot]["description"],
                "member_candidate_ids": list(members),
                "supporting_evidence_ids": list(supporting_evidence),
                "source_families": list(source_families),
                "value_shape_hypothesis": shape,
                "unresolved_ambiguity": definitions[slot]["unresolved_ambiguity"],
            }
        )
    normalized_dispositions = [
        {
            "candidate_id": item.candidate_id,
            "canonical_name": output_name_by_slot[slot_by_candidate[item.candidate_id]],
            "reason": reason_by_candidate[item.candidate_id],
        }
        for item in items
    ]
    audit = {
        "audit_version": "fixed_slot_consolidation_normalization_audit_v1",
        "slot_policy": "exact_candidate_assignment_then_compiler_derived_groups_v1",
        "derived_relation_fields": [
            "active_slots",
            "canonical_concepts.member_candidate_ids",
            "canonical_concepts.supporting_evidence_ids",
            "canonical_concepts.source_families",
            "canonical_concepts.value_shape_hypothesis",
            "candidate_dispositions.canonical_name",
        ],
        "candidate_slot_assignments": assignment_audit,
        "slot_definitions": definition_audit,
        "active_slots": list(active_slots),
        "unused_slots": [slot for slot in slots if slot not in assigned_slots],
        "canonical_name_disambiguations": disambiguations,
    }
    return _detached(
        {
            "canonical_concepts": normalized_concepts,
            "candidate_dispositions": normalized_dispositions,
            "wire_normalization_audit": audit,
        }
    )


def revalidate_normalized_consolidation_response(
    response: Any,
    *,
    source_family: str,
    candidates: Sequence[DiscoveryCandidate],
    wire_budget: HierarchyWireBudget,
) -> dict[str, Any]:
    root = _require_mapping(response, label="normalized consolidation response")
    _require_exact_keys(
        root,
        expected={
            "canonical_concepts",
            "candidate_dispositions",
            "wire_normalization_audit",
        },
        label="normalized consolidation response",
    )
    dispositions = root["candidate_dispositions"]
    if not isinstance(dispositions, list):
        raise TypeError("normalized candidate_dispositions must be a JSON list")
    reason_by_candidate: dict[str, str] = {}
    for index, raw in enumerate(dispositions):
        row = _require_mapping(raw, label=f"normalized candidate_dispositions[{index}]")
        candidate_id = _require_id(
            row.get("candidate_id"),
            label=f"normalized candidate_dispositions[{index}].candidate_id",
        )
        if candidate_id in reason_by_candidate:
            raise ValueError("normalized candidate dispositions contain duplicate IDs")
        reason_by_candidate[candidate_id] = _require_string(
            row.get("reason"),
            label=f"normalized candidate_dispositions[{index}].reason",
        )
    audit = _require_mapping(root["wire_normalization_audit"], label="wire_normalization_audit")
    assignments = audit.get("candidate_slot_assignments")
    definitions = audit.get("slot_definitions")
    if not isinstance(assignments, list) or not isinstance(definitions, list):
        raise TypeError("normalized consolidation audit is incomplete")
    wire_assignments: dict[str, Any] = {}
    for index, raw in enumerate(assignments):
        row = _require_mapping(raw, label=f"candidate_slot_assignments[{index}]")
        candidate_id = _require_id(
            row.get("candidate_id"),
            label=f"candidate_slot_assignments[{index}].candidate_id",
        )
        wire_assignments[candidate_id] = {
            "cluster_slot": row.get("cluster_slot"),
            "reason": reason_by_candidate.get(candidate_id),
        }
    wire_definitions: dict[str, Any] = {}
    for index, raw in enumerate(definitions):
        row = _require_mapping(raw, label=f"slot_definitions[{index}]")
        slot = _require_id(row.get("cluster_slot"), label=f"slot_definitions[{index}].cluster_slot")
        wire_definitions[slot] = {
            "canonical_name": row.get("canonical_name"),
            "description": row.get("description"),
            "unresolved_ambiguity": row.get("unresolved_ambiguity"),
        }
    validated = validate_consolidation_response(
        {
            "candidate_assignments": wire_assignments,
            "slot_definitions": wire_definitions,
        },
        source_family=source_family,
        candidates=candidates,
        wire_budget=wire_budget,
    )
    if canonical_json(validated) != canonical_json(root):
        raise ValueError("normalized consolidation is not the deterministic projection")
    return validated


@dataclass(frozen=True)
class ArchitectureDossier:
    """Compact cross-architecture view backed by a complete private audit.

    ``catalog_evidence_ids`` and ``coverage_disposition_ids`` are retained in
    the authenticated object but omitted from the model-facing dossier.  Their
    equality proves complete architecture-local review without dumping every
    raw atom or disposition into the integration prompt.
    """

    source_family: str
    catalog_sha256: str
    catalog_evidence_ids: tuple[str, ...]
    coverage_disposition_ids: tuple[str, ...]
    coverage_audit_sha256: str
    architecture_candidates: tuple[DiscoveryCandidate, ...]
    # Backward-compatible construction/read path for already-realized manifests.
    # It remains empty for a pre-fit intent, so an intent digest can never be
    # represented or reported as a manifest digest.
    direct_numerical_manifest_sha256: str = ""
    direct_numerical_signal_count: int = 0
    direct_numerical_zero_reason: str = ""
    direct_numerical_contract_kind: str = ""
    direct_numerical_contract_sha256: str = ""

    def __post_init__(self) -> None:
        if self.source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError(f"inactive or unknown source_family: {self.source_family!r}")
        contract_kind = self.direct_numerical_contract_kind
        contract_sha256 = self.direct_numerical_contract_sha256
        legacy_manifest_sha256 = self.direct_numerical_manifest_sha256
        if not contract_kind and not contract_sha256 and legacy_manifest_sha256:
            contract_kind = DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST
            contract_sha256 = legacy_manifest_sha256
            object.__setattr__(self, "direct_numerical_contract_kind", contract_kind)
            object.__setattr__(self, "direct_numerical_contract_sha256", contract_sha256)
        if contract_kind not in DIRECT_NUMERICAL_CONTRACT_KINDS:
            raise ValueError("direct_numerical_contract_kind is unsupported")
        if bool(contract_sha256) is False:
            raise ValueError("direct_numerical_contract_sha256 cannot be empty")
        if contract_kind == DIRECT_NUMERICAL_CONTRACT_KIND_REALIZED_MANIFEST:
            if legacy_manifest_sha256 and legacy_manifest_sha256 != contract_sha256:
                raise ValueError("manifest and generic numerical contract digests differ")
            if not legacy_manifest_sha256:
                object.__setattr__(
                    self,
                    "direct_numerical_manifest_sha256",
                    contract_sha256,
                )
        elif legacy_manifest_sha256:
            raise ValueError("a pre-fit intent cannot expose a manifest SHA-256")
        for label, value in (
            ("catalog_sha256", self.catalog_sha256),
            ("coverage_audit_sha256", self.coverage_audit_sha256),
            ("direct_numerical_contract_sha256", contract_sha256),
        ):
            _require_string(value, label=label)
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        for label, values in (
            ("catalog_evidence_ids", self.catalog_evidence_ids),
            ("coverage_disposition_ids", self.coverage_disposition_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{label} cannot contain duplicates")
            for index, value in enumerate(values):
                _require_id(value, label=f"{label}[{index}]")
        if set(self.catalog_evidence_ids) != set(self.coverage_disposition_ids):
            raise ValueError("architecture dossier requires a disposition for every catalog atom")
        _ensure_unique_ids(
            self.architecture_candidates,
            attribute="candidate_id",
            label="architecture candidate",
        )
        catalog_ids = set(self.catalog_evidence_ids)
        for candidate in self.architecture_candidates:
            if candidate.source_families != (self.source_family,):
                raise ValueError(
                    "architecture candidate must cite exactly its dossier architecture"
                )
            if not set(candidate.supporting_evidence_ids) <= catalog_ids:
                raise ValueError("architecture candidate cites evidence outside its catalog")
        if isinstance(self.direct_numerical_signal_count, bool) or not isinstance(
            self.direct_numerical_signal_count, int
        ):
            raise TypeError("direct_numerical_signal_count must be a non-negative integer")
        if self.direct_numerical_signal_count < 0:
            raise ValueError("direct_numerical_signal_count must be non-negative")
        _require_string(
            self.direct_numerical_zero_reason,
            label="direct_numerical_zero_reason",
            allow_empty=True,
        )
        if self.direct_numerical_signal_count == 0 and not self.direct_numerical_zero_reason:
            raise ValueError("zero direct numerical signals require an explicit zero reason")
        if self.direct_numerical_signal_count > 0 and self.direct_numerical_zero_reason:
            raise ValueError("a nonzero direct numerical signal count cannot have a zero reason")

    def as_cross_architecture_prompt_item(self) -> dict[str, Any]:
        """Return only concept-bearing dossier context suitable for a model call.

        The approved direct numerical contract remains authenticated in
        ``as_authenticated_dict`` and flows separately to review and final
        estimation.  Even aggregate signal counts are intentionally absent here
        so the discovery model cannot use the numerical channel while naming or
        merging patient characteristics.
        """

        return {
            "source_family": self.source_family,
            "coverage": {
                "catalog_evidence_count": len(self.catalog_evidence_ids),
                "coverage_disposition_count": len(self.coverage_disposition_ids),
                "complete": True,
            },
            "architecture_candidates": [
                item.as_prompt_item() for item in self.architecture_candidates
            ],
        }

    def as_authenticated_dict(self) -> dict[str, Any]:
        """Return the complete internal identity omitted from model messages."""

        return {
            "schema_version": ARCHITECTURE_DOSSIER_VERSION,
            "source_family": self.source_family,
            "catalog_sha256": self.catalog_sha256,
            "catalog_evidence_ids": list(self.catalog_evidence_ids),
            "coverage_disposition_ids": list(self.coverage_disposition_ids),
            "coverage_audit_sha256": self.coverage_audit_sha256,
            "architecture_candidates": [
                item.as_prompt_item() for item in self.architecture_candidates
            ],
            "direct_numerical_channel": {
                "channel": DIRECT_UPSTREAM_NUMERICAL_CHANNEL,
                "direct_numerical_contract_kind": (self.direct_numerical_contract_kind),
                "direct_numerical_contract_sha256": (self.direct_numerical_contract_sha256),
                "signal_count": self.direct_numerical_signal_count,
                "zero_reason": self.direct_numerical_zero_reason,
                "concept_grounding_allowed": False,
            },
        }


def cross_architecture_planner_context(
    dossiers: Sequence[ArchitectureDossier],
) -> dict[str, Any]:
    items = tuple(dossiers)
    if not items:
        raise ValueError("architecture dossiers cannot be empty")
    families = [item.source_family for item in items]
    if len(set(families)) != len(families):
        raise ValueError("architecture dossiers must have unique source families")
    missing = ACTIVE_STAGE1_CONCEPT_FAMILY_SET - set(families)
    extra = set(families) - ACTIVE_STAGE1_CONCEPT_FAMILY_SET
    if missing or extra:
        raise ValueError(
            "cross-architecture planning requires every active architecture; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    candidate_ids = [
        candidate.candidate_id for dossier in items for candidate in dossier.architecture_candidates
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be globally unique across architecture dossiers")
    by_family = {item.source_family: item for item in items}
    return {
        "job": "plan_cross_architecture_integration",
        "schema_version": CROSS_ARCHITECTURE_PLANNER_JOB_VERSION,
        "architecture_dossiers": [
            by_family[family].as_cross_architecture_prompt_item()
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        ],
    }


_CROSS_ARCHITECTURE_PLANNER_SYSTEM_PROMPT = """Compare compact, independently completed Stage 1 architecture dossiers. Group only likely spelling, abbreviation, or formatting aliases; preserve distinct patient measurements and every candidate. Assign each exact candidate_id to one compiler-owned group slot and define every fixed group slot; unused definitions are ignored. Use the fixed lookback slots to select the minimum exact raw evidence IDs needed to decide, or select unused. Duplicate evidence selections are deterministically deduplicated. Do not invent evidence, assign causal roles, define extraction, estimate effects, or make final rejections. Return JSON only in the requested shape."""


def render_cross_architecture_planner_messages(
    dossiers: Sequence[ArchitectureDossier],
    *,
    maximum_raw_evidence_lookback_ids: int,
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> tuple[dict[str, str], ...]:
    context = cross_architecture_planner_context(dossiers)
    # The interface version is bound to the immutable job, not shown to the
    # scientific reasoner.
    context.pop("schema_version")
    context["maximum_raw_evidence_lookback_ids"] = maximum_raw_evidence_lookback_ids
    request = attach_hierarchical_discovery_response_contract(
        job_kind="plan_cross_architecture_integration",
        request=context,
        wire_budget=wire_budget,
    )
    return (
        {"role": "system", "content": _CROSS_ARCHITECTURE_PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(request)},
    )


def validate_cross_architecture_planner_response(
    response: Any,
    *,
    dossiers: Sequence[ArchitectureDossier],
    maximum_raw_evidence_lookback_ids: int,
    wire_budget: HierarchyWireBudget = LEGACY_HIERARCHY_WIRE_BUDGET,
) -> dict[str, Any]:
    cross_architecture_planner_context(dossiers)
    if (
        isinstance(maximum_raw_evidence_lookback_ids, bool)
        or not isinstance(maximum_raw_evidence_lookback_ids, int)
        or maximum_raw_evidence_lookback_ids < 0
    ):
        raise ValueError("maximum_raw_evidence_lookback_ids must be non-negative")
    by_candidate = {
        candidate.candidate_id: candidate
        for dossier in dossiers
        for candidate in dossier.architecture_candidates
    }
    candidate_id_order = tuple(by_candidate)
    cited_evidence_ids = tuple(
        dict.fromkeys(
            evidence_id
            for candidate in by_candidate.values()
            for evidence_id in candidate.supporting_evidence_ids
        )
    )
    group_slots = tuple(
        f"planner_group_slot_{index:03d}" for index in range(1, len(candidate_id_order) + 1)
    )
    lookback_slots = tuple(
        f"planner_lookback_slot_{index:03d}"
        for index in range(
            1,
            min(maximum_raw_evidence_lookback_ids, len(cited_evidence_ids)) + 1,
        )
    )

    root = _require_mapping(response, label="cross-architecture planner response")
    _require_exact_keys(
        root,
        expected={
            "candidate_assignments",
            "group_slot_definitions",
            "lookback_slot_definitions",
        },
        label="cross-architecture planner response",
    )
    assignments = _require_exact_id_keyed_mapping(
        root["candidate_assignments"],
        identifiers=candidate_id_order,
        label="candidate_assignments",
    )
    group_by_candidate: dict[str, str] = {}
    assignment_audit: list[dict[str, str]] = []
    for candidate_id, assignment in assignments:
        label = f"candidate_assignments.{candidate_id}"
        _require_exact_keys(assignment, expected={"group_slot"}, label=label)
        slot = _require_id(assignment["group_slot"], label=f"{label}.group_slot")
        if slot not in group_slots:
            raise ValueError(f"{label}.group_slot is not compiler-owned")
        group_by_candidate[candidate_id] = slot
        assignment_audit.append({"candidate_id": candidate_id, "group_slot": slot})

    definitions: dict[str, dict[str, str]] = {}
    definition_audit: list[dict[str, str]] = []
    for slot, raw_definition in _require_exact_id_keyed_mapping(
        root["group_slot_definitions"],
        identifiers=group_slots,
        label="group_slot_definitions",
    ):
        label = f"group_slot_definitions.{slot}"
        _require_exact_keys(
            raw_definition,
            expected={"provisional_name", "reason"},
            label=label,
        )
        definition = {
            "provisional_name": _require_name(
                raw_definition["provisional_name"],
                label=f"{label}.provisional_name",
            ),
            "reason": _require_string(raw_definition["reason"], label=f"{label}.reason"),
        }
        definitions[slot] = definition
        definition_audit.append({"group_slot": slot, **definition})

    active_slots_set = set(group_by_candidate.values())
    active_slots = tuple(slot for slot in group_slots if slot in active_slots_set)
    names: set[str] = set()
    name_by_slot: dict[str, str] = {}
    disambiguations: list[dict[str, str]] = []
    for slot in active_slots:
        proposed = definitions[slot]["provisional_name"]
        derived = _derive_unique_slot_name(
            proposed=proposed,
            slot=slot,
            used=names,
            wire_budget=wire_budget,
        )
        names.add(derived)
        name_by_slot[slot] = derived
        if derived != proposed:
            disambiguations.append(
                {
                    "group_slot": slot,
                    "proposed_provisional_name": proposed,
                    "derived_provisional_name": derived,
                    "reason": "distinct active slots cannot share provisional identity",
                }
            )
    groups = [
        {
            "member_candidate_ids": [
                candidate_id
                for candidate_id in candidate_id_order
                if group_by_candidate[candidate_id] == slot
            ],
            "provisional_name": name_by_slot[slot],
            "reason": definitions[slot]["reason"],
        }
        for slot in active_slots
    ]

    requested_ids: set[str] = set()
    requests: list[dict[str, Any]] = []
    lookback_audit: list[dict[str, str]] = []
    duplicate_selections: list[dict[str, str]] = []
    cited_set = set(cited_evidence_ids)
    for slot, raw_definition in _require_exact_id_keyed_mapping(
        root["lookback_slot_definitions"],
        identifiers=lookback_slots,
        label="lookback_slot_definitions",
    ):
        label = f"lookback_slot_definitions.{slot}"
        _require_exact_keys(
            raw_definition,
            expected={"selection", "question", "reason"},
            label=label,
        )
        selection = _require_string(raw_definition["selection"], label=f"{label}.selection")
        if selection != "unused" and selection not in cited_set:
            raise ValueError(f"{label}.selection is not dossier-owned")
        question = _require_string(raw_definition["question"], label=f"{label}.question")
        reason = _require_string(raw_definition["reason"], label=f"{label}.reason")
        lookback_audit.append(
            {
                "lookback_slot": slot,
                "selection": selection,
                "question": question,
                "reason": reason,
            }
        )
        if selection == "unused":
            continue
        if selection in requested_ids:
            duplicate_selections.append(
                {
                    "lookback_slot": slot,
                    "evidence_id": selection,
                    "reason": "duplicate selection ignored after its first slot",
                }
            )
            continue
        requested_ids.add(selection)
        requests.append({"evidence_ids": [selection], "question": question, "reason": reason})
    return _detached(
        {
            "provisional_groups": groups,
            "raw_evidence_requests": requests,
            "wire_normalization_audit": {
                "audit_version": "fixed_slot_planner_normalization_audit_v1",
                "slot_policy": "candidate_group_and_bounded_lookback_slots_v1",
                "candidate_group_assignments": assignment_audit,
                "group_slot_definitions": definition_audit,
                "active_group_slots": list(active_slots),
                "unused_group_slots": [
                    slot for slot in group_slots if slot not in active_slots_set
                ],
                "provisional_name_disambiguations": disambiguations,
                "lookback_slot_definitions": lookback_audit,
                "duplicate_lookback_selections": duplicate_selections,
                "maximum_raw_evidence_lookback_ids": (maximum_raw_evidence_lookback_ids),
            },
        }
    )


def _validate_normalized_cross_architecture_planner_response(
    response: Any,
    *,
    dossiers: Sequence[ArchitectureDossier],
    maximum_raw_evidence_lookback_ids: int,
) -> dict[str, Any]:
    root = _require_mapping(response, label="normalized cross-architecture planner response")
    _require_exact_keys(
        root,
        expected={"provisional_groups", "raw_evidence_requests", "wire_normalization_audit"},
        label="normalized cross-architecture planner response",
    )
    audit = _require_mapping(root["wire_normalization_audit"], label="wire_normalization_audit")
    assignment_rows = audit.get("candidate_group_assignments")
    definition_rows = audit.get("group_slot_definitions")
    lookback_rows = audit.get("lookback_slot_definitions")
    if not all(
        isinstance(rows, list) for rows in (assignment_rows, definition_rows, lookback_rows)
    ):
        raise TypeError("normalized planner audit is incomplete")
    wire = {
        "candidate_assignments": {
            row["candidate_id"]: {"group_slot": row["group_slot"]} for row in assignment_rows
        },
        "group_slot_definitions": {
            row["group_slot"]: {
                "provisional_name": row["provisional_name"],
                "reason": row["reason"],
            }
            for row in definition_rows
        },
        "lookback_slot_definitions": {
            row["lookback_slot"]: {
                "selection": row["selection"],
                "question": row["question"],
                "reason": row["reason"],
            }
            for row in lookback_rows
        },
    }
    reconstructed = validate_cross_architecture_planner_response(
        wire,
        dossiers=dossiers,
        maximum_raw_evidence_lookback_ids=maximum_raw_evidence_lookback_ids,
    )
    if canonical_json(reconstructed) != canonical_json(root):
        raise ValueError("normalized planner response is not compiler-derived")
    return reconstructed


def resolve_raw_evidence_lookback(
    *,
    planner_response: Mapping[str, Any],
    dossiers: Sequence[ArchitectureDossier],
    catalog: Mapping[str, DiscoveryEvidenceItem],
    maximum_raw_evidence_lookback_ids: int,
) -> tuple[dict[str, Any], ...]:
    groups = _require_mapping(
        planner_response,
        label="planner_response",
    ).get("provisional_groups")
    if isinstance(groups, list):
        validated = _validate_normalized_cross_architecture_planner_response(
            planner_response,
            dossiers=dossiers,
            maximum_raw_evidence_lookback_ids=maximum_raw_evidence_lookback_ids,
        )
    else:
        validated = validate_cross_architecture_planner_response(
            planner_response,
            dossiers=dossiers,
            maximum_raw_evidence_lookback_ids=maximum_raw_evidence_lookback_ids,
        )
    requested = [
        evidence_id
        for request in validated["raw_evidence_requests"]
        for evidence_id in request["evidence_ids"]
    ]
    if set(catalog) != {
        evidence_id for dossier in dossiers for evidence_id in dossier.catalog_evidence_ids
    }:
        raise ValueError("lookback catalog does not exactly match the authenticated dossiers")
    for evidence_id, item in catalog.items():
        if evidence_id != item.evidence_id:
            raise ValueError("lookback catalog keys must equal evidence item IDs")
    return tuple(catalog[evidence_id].as_prompt_item() for evidence_id in requested)


def validate_coverage_critic_response(
    response: Any,
    *,
    evidence_ids: Sequence[str],
    canonical_names: Sequence[str],
) -> dict[str, Any]:
    supplied_ids = set(evidence_ids)
    if not supplied_ids or len(supplied_ids) != len(evidence_ids):
        raise ValueError("evidence_ids must be non-empty and unique")
    for index, value in enumerate(evidence_ids):
        _require_id(value, label=f"evidence_ids[{index}]")
    supplied_names = set(canonical_names)
    if len(supplied_names) != len(canonical_names):
        raise ValueError("canonical_names must be unique")
    for index, value in enumerate(canonical_names):
        _require_name(value, label=f"canonical_names[{index}]")

    root = _require_mapping(response, label="coverage critic response")
    _require_exact_keys(
        root,
        expected={"findings", "reviewed_evidence_ids"},
        label="coverage critic response",
    )
    reviewed = _require_exact_id_acknowledgements(
        root["reviewed_evidence_ids"],
        identifiers=evidence_ids,
        label="reviewed_evidence_ids",
    )
    findings = root["findings"]
    if not isinstance(findings, list):
        raise TypeError("findings must be a JSON list")

    def wire_identifier_values(value: Any, *, label: str, validate_item: Any) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise TypeError(f"{label} must be a JSON list")
        return tuple(
            validate_item(item, label=f"{label}[{index}]") for index, item in enumerate(value)
        )

    for index, raw in enumerate(findings):
        finding = _require_mapping(raw, label=f"findings[{index}]")
        _require_exact_keys(
            finding,
            expected={
                "action",
                "affected_canonical_names",
                "proposed_name",
                "description",
                "supporting_evidence_ids",
                "reason",
            },
            label=f"findings[{index}]",
        )
        action = finding["action"]
        if action not in {"add_concept", "split_concept", "restore_support", "no_change"}:
            raise ValueError(f"findings[{index}].action is invalid")
        affected = set(
            wire_identifier_values(
                finding["affected_canonical_names"],
                label=f"findings[{index}].affected_canonical_names",
                validate_item=_require_name,
            )
        )
        if not affected <= supplied_names:
            raise ValueError(f"findings[{index}] cites unknown canonical names")
        support = set(
            wire_identifier_values(
                finding["supporting_evidence_ids"],
                label=f"findings[{index}].supporting_evidence_ids",
                validate_item=_require_id,
            )
        )
        if action != "no_change" and not support:
            raise ValueError(f"findings[{index}] requires supporting evidence")
        if not support <= supplied_ids:
            raise ValueError(f"findings[{index}] cites unsupplied evidence")
        proposed = _require_string(
            finding["proposed_name"],
            label=f"findings[{index}].proposed_name",
            allow_empty=True,
        )
        description = _require_string(
            finding["description"],
            label=f"findings[{index}].description",
            allow_empty=True,
        )
        if action in {"add_concept", "split_concept"}:
            _require_name(proposed, label=f"findings[{index}].proposed_name")
            if not description.strip():
                raise ValueError(f"findings[{index}] requires a description")
        elif proposed:
            _require_name(proposed, label=f"findings[{index}].proposed_name")
        _require_string(finding["reason"], label=f"findings[{index}].reason")
    return _detached(
        {
            "findings": findings,
            "reviewed_evidence_ids": list(reviewed),
        }
    )


def revalidate_normalized_coverage_critic_response(
    response: Any,
    *,
    evidence_ids: Sequence[str],
    canonical_names: Sequence[str],
) -> dict[str, Any]:
    root = _require_mapping(response, label="normalized coverage critic response")
    reviewed = root.get("reviewed_evidence_ids")
    if not isinstance(reviewed, list):
        raise TypeError("normalized reviewed_evidence_ids must be a JSON list")
    if len(reviewed) != len(set(reviewed)):
        raise ValueError("normalized reviewed_evidence_ids contain duplicates")
    validated = validate_coverage_critic_response(
        {
            "findings": root.get("findings"),
            "reviewed_evidence_ids": {evidence_id: True for evidence_id in reviewed},
        },
        evidence_ids=evidence_ids,
        canonical_names=canonical_names,
    )
    if canonical_json(validated) != canonical_json(root):
        raise ValueError("normalized coverage response is not the deterministic projection")
    return validated


def validate_rejection_critic_response(
    response: Any,
    *,
    rejected_candidate_evidence: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if not rejected_candidate_evidence:
        raise ValueError("rejected_candidate_evidence cannot be empty")
    supplied: dict[str, set[str]] = {}
    for candidate_id, evidence_ids in rejected_candidate_evidence.items():
        _require_id(candidate_id, label="rejected candidate ID")
        parsed = {_require_id(value, label=f"{candidate_id}.evidence_id") for value in evidence_ids}
        if len(parsed) != len(evidence_ids):
            raise ValueError(f"rejected candidate {candidate_id!r} has duplicate evidence")
        supplied[candidate_id] = parsed

    root = _require_mapping(response, label="rejection critic response")
    _require_exact_keys(root, expected={"reconsiderations"}, label="rejection critic response")
    candidate_order = tuple(rejected_candidate_evidence)
    rows = _require_exact_id_keyed_mapping(
        root["reconsiderations"],
        identifiers=candidate_order,
        label="reconsiderations",
    )
    normalized_rows: list[dict[str, Any]] = []
    for candidate_id, row in rows:
        label = f"reconsiderations.{candidate_id}"
        _require_exact_keys(
            row,
            expected={
                "decision",
                "proposed_name",
                "supporting_evidence_ids",
                "reason",
            },
            label=label,
        )
        decision = row["decision"]
        if decision not in {"uphold", "restore", "split"}:
            raise ValueError(f"{label}.decision is invalid")
        proposed = _require_string(
            row["proposed_name"],
            label=f"{label}.proposed_name",
            allow_empty=True,
        )
        raw_support = row["supporting_evidence_ids"]
        if not isinstance(raw_support, list):
            raise TypeError(f"{label}.supporting_evidence_ids must be a JSON list")
        support = tuple(
            _require_id(value, label=f"{label}.supporting_evidence_ids[{index}]")
            for index, value in enumerate(raw_support)
        )
        if decision != "uphold" and not support:
            raise ValueError(f"{label}.supporting_evidence_ids cannot be empty")
        if not set(support) <= supplied[candidate_id]:
            raise ValueError(f"{label} cites evidence from another candidate")
        if decision in {"restore", "split"}:
            _require_name(proposed, label=f"{label}.proposed_name")
        elif proposed:
            raise ValueError("uphold reconsideration must use an empty proposed_name")
        _require_string(row["reason"], label=f"{label}.reason")
        normalized_rows.append(
            {
                "candidate_id": candidate_id,
                "decision": decision,
                "proposed_name": proposed,
                "supporting_evidence_ids": list(support),
                "reason": row["reason"],
            }
        )
    return _detached({"reconsiderations": normalized_rows})


@dataclass(frozen=True)
class RoleRoutingResult:
    observable_axes: tuple[str, ...]
    adjustment_roles: tuple[str, ...]
    effect_modifier: bool
    treatment_prediction_support: bool
    extraction_definition_support: bool
    applied_rules: tuple[str, ...]

    def audit(self) -> dict[str, Any]:
        return {
            "schema_version": ROLE_ROUTING_VERSION,
            "observable_axes": list(self.observable_axes),
            "adjustment_roles": list(self.adjustment_roles),
            "effect_modifier": self.effect_modifier,
            "treatment_prediction_support": self.treatment_prediction_support,
            "extraction_definition_support": self.extraction_definition_support,
            "applied_rules": list(self.applied_rules),
        }


def route_concept_roles(
    *, evidence: Sequence[DiscoveryEvidenceItem], supporting_evidence_ids: Sequence[str]
) -> RoleRoutingResult:
    items = tuple(evidence)
    _ensure_unique_ids(items, attribute="evidence_id", label="evidence")
    by_id = {item.evidence_id: item for item in items}
    if not supporting_evidence_ids:
        raise ValueError("supporting_evidence_ids cannot be empty")
    support = tuple(supporting_evidence_ids)
    if len(set(support)) != len(support):
        raise ValueError("supporting_evidence_ids cannot contain duplicates")
    unknown = set(support) - set(by_id)
    if unknown:
        raise ValueError(f"role routing cites unknown evidence: {sorted(unknown)}")
    axes = tuple(
        axis
        for axis in OBSERVABLE_AXES
        if any(axis in by_id[item].observable_axes for item in support)
    )
    axis_set = set(axes)
    adjustment: list[str] = []
    rules: list[str] = []
    if {TREATMENT_AXIS, OUTCOME_AXIS} <= axis_set:
        adjustment.append("confounder_adjustment")
        rules.append("treatment_plus_outcome_to_confounder_adjustment")
    elif OUTCOME_AXIS in axis_set:
        adjustment.append("prognostic_adjustment")
        rules.append("outcome_only_to_prognostic_adjustment")
    if HETEROGENEITY_AXIS in axis_set or PAIR_UPLIFT_AXIS in axis_set:
        rules.append("heterogeneity_or_pair_uplift_to_effect_modifier")
    if TREATMENT_AXIS in axis_set:
        rules.append("treatment_axis_to_treatment_prediction_support")
    if EXTRACTION_SUPPORT_AXIS in axis_set:
        rules.append("extraction_axis_to_definition_support_only")
    return RoleRoutingResult(
        observable_axes=axes,
        adjustment_roles=tuple(adjustment),
        effect_modifier=bool({HETEROGENEITY_AXIS, PAIR_UPLIFT_AXIS} & axis_set),
        treatment_prediction_support=TREATMENT_AXIS in axis_set,
        extraction_definition_support=EXTRACTION_SUPPORT_AXIS in axis_set,
        applied_rules=tuple(rules),
    )


@dataclass(frozen=True)
class ExtractionDefinitionRequest:
    canonical_name: str
    evidence: tuple[DiscoveryEvidenceItem, ...]
    supporting_evidence_ids: tuple[str, ...]
    value_shape_hypothesis: str = "ambiguous"
    allowed_aliases: tuple[str, ...] = ()
    allowed_units: tuple[str, ...] = ()
    allowed_categories: tuple[str, ...] = ()
    allowed_distinguish_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_name(self.canonical_name, label="canonical_name")
        _ensure_unique_ids(self.evidence, attribute="evidence_id", label="evidence")
        if not self.supporting_evidence_ids:
            raise ValueError("supporting_evidence_ids cannot be empty")
        available = {item.evidence_id for item in self.evidence}
        if not set(self.supporting_evidence_ids) <= available:
            raise ValueError("supporting_evidence_ids cite unavailable extraction evidence")
        if len(set(self.supporting_evidence_ids)) != len(self.supporting_evidence_ids):
            raise ValueError("supporting_evidence_ids cannot contain duplicates")
        if self.value_shape_hypothesis not in _VALUE_SHAPES:
            raise ValueError("value_shape_hypothesis is invalid")
        for label, values in (
            ("allowed_aliases", self.allowed_aliases),
            ("allowed_units", self.allowed_units),
            ("allowed_categories", self.allowed_categories),
            ("allowed_distinguish_from", self.allowed_distinguish_from),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{label} cannot contain duplicates")
            for index, value in enumerate(values):
                _require_string(value, label=f"{label}[{index}]")
                if not _literal_supported_evidence_value(
                    value,
                    evidence=self.evidence,
                    supporting_evidence_ids=self.supporting_evidence_ids,
                ):
                    raise ValueError(
                        f"{label}[{index}] is not literally grounded in supporting evidence"
                    )


def _validate_extraction_vocabulary_value(
    value: str,
    *,
    request: ExtractionDefinitionRequest,
    label: str,
) -> None:
    if not _literal_supported_evidence_value(
        value,
        evidence=request.evidence,
        supporting_evidence_ids=request.supporting_evidence_ids,
    ):
        raise ValueError(f"{label} is not literally grounded in supporting evidence")


def validate_extraction_definition_response(
    response: Any, *, request: ExtractionDefinitionRequest
) -> dict[str, Any]:
    root = _require_mapping(response, label="extraction definition response")
    _require_exact_keys(
        root,
        expected={
            "feature_name",
            "measurement",
            "representation",
            "aliases",
            "distinguish_from",
            "missing_or_ambiguous",
            "supporting_evidence_reviewed",
        },
        label="extraction definition response",
    )
    if root["feature_name"] != request.canonical_name:
        raise ValueError("extraction definition must preserve the canonical feature name")
    _require_string(root["measurement"], label="measurement")
    representation = _require_mapping(root["representation"], label="representation")
    _require_exact_keys(
        representation,
        expected={"kind", "unit", "categories"},
        label="representation",
    )
    kind = representation["kind"]
    if kind not in {"continuous", "categorical", "unresolved"}:
        raise ValueError("representation.kind is invalid")
    unit = _require_string(representation["unit"], label="representation.unit", allow_empty=True)

    def wire_strings(value: Any, *, label: str, allow_empty_items: bool) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise TypeError(f"{label} must be a JSON list")
        return tuple(
            _require_string(
                item,
                label=f"{label}[{index}]",
                allow_empty=allow_empty_items,
            )
            for index, item in enumerate(value)
        )

    categories = wire_strings(
        representation["categories"],
        label="representation.categories",
        allow_empty_items=False,
    )
    original_model_fields = {
        "representation": {
            "kind": kind,
            "unit": unit,
            "categories": list(categories),
        },
        "aliases": list(wire_strings(root["aliases"], label="aliases", allow_empty_items=True)),
        "distinguish_from": list(
            wire_strings(
                root["distinguish_from"],
                label="distinguish_from",
                allow_empty_items=True,
            )
        ),
    }
    normalization_events: list[dict[str, Any]] = []
    normalized_kind = str(kind)
    normalized_unit = unit
    category_identities = tuple(
        re.sub(r"[\s_-]+", " ", category).strip().casefold()
        for category in categories
    )
    if len(category_identities) != len(set(category_identities)):
        raise ValueError(
            "representation.categories must be distinct after case/spacing normalization"
        )
    normalized_categories = categories
    if kind == "continuous":
        if request.value_shape_hypothesis == "categorical":
            raise ValueError("continuous extraction conflicts with the categorical value shape")
        if not unit:
            raise ValueError("continuous extraction requires an evidence-supported unit statement")
        if categories:
            raise ValueError("continuous extraction cannot define categories")
        if unit != AS_DOCUMENTED_UNIT:
            if not _literal_supported_evidence_value(
                unit,
                evidence=request.evidence,
                supporting_evidence_ids=request.supporting_evidence_ids,
            ):
                normalized_kind = "unresolved"
                normalized_unit = ""
                normalized_categories = ()
                normalization_events.append(
                    {
                        "field": "representation.unit",
                        "value": unit,
                        "action": "representation_set_unresolved",
                        "reason": "unit is not literally grounded in supporting evidence",
                    }
                )
    elif kind == "categorical":
        if request.value_shape_hypothesis == "continuous":
            raise ValueError("categorical extraction conflicts with the continuous value shape")
        if len(categories) < 2:
            raise ValueError("categorical extraction requires at least two concrete categories")
        if unit:
            raise ValueError("categorical extraction cannot define a unit")
        if set(normalized_categories) & set(MECHANICAL_MENTION_CATEGORIES):
            if normalized_categories != MECHANICAL_MENTION_CATEGORIES:
                normalized_kind = "unresolved"
                normalized_categories = ()
                normalization_events.append(
                    {
                        "field": "representation.categories",
                        "value": list(categories),
                        "action": "representation_set_unresolved",
                        "reason": "mechanical categories do not equal the exact reserved pair",
                    }
                )
        else:
            grounded_categories: list[str] = []
            for category in normalized_categories:
                if _literal_supported_evidence_value(
                    category,
                    evidence=request.evidence,
                    supporting_evidence_ids=request.supporting_evidence_ids,
                ):
                    grounded_categories.append(category)
                else:
                    normalization_events.append(
                        {
                            "field": "representation.categories",
                            "value": category,
                            "action": "filtered",
                            "reason": "category is not literally grounded in supporting evidence",
                        }
                    )
            normalized_categories = tuple(grounded_categories)
            if len(normalized_categories) < 2:
                normalized_kind = "unresolved"
                normalized_categories = ()
                normalization_events.append(
                    {
                        "field": "representation.categories",
                        "value": list(categories),
                        "action": "representation_set_unresolved",
                        "reason": "fewer than two distinct grounded categories remain",
                    }
                )
    elif unit or categories:
        raise ValueError("unresolved extraction cannot define a unit or categories")

    def grounded_optional_values(values: Sequence[str], *, field: str) -> tuple[str, ...]:
        retained: list[str] = []
        for value in values:
            if value in retained:
                normalization_events.append(
                    {
                        "field": field,
                        "value": value,
                        "action": "filtered",
                        "reason": "duplicate optional vocabulary",
                    }
                )
            elif _literal_supported_evidence_value(
                value,
                evidence=request.evidence,
                supporting_evidence_ids=request.supporting_evidence_ids,
            ):
                retained.append(value)
            else:
                normalization_events.append(
                    {
                        "field": field,
                        "value": value,
                        "action": "filtered",
                        "reason": "optional vocabulary is not literally grounded",
                    }
                )
        return tuple(retained)

    aliases = grounded_optional_values(original_model_fields["aliases"], field="aliases")
    distinctions = grounded_optional_values(
        original_model_fields["distinguish_from"], field="distinguish_from"
    )
    _require_string(root["missing_or_ambiguous"], label="missing_or_ambiguous")
    if root["supporting_evidence_reviewed"] is not True:
        raise ValueError("extraction definition did not review its exact compiler-owned support")
    support = tuple(request.supporting_evidence_ids)
    normalized = dict(root)
    normalized.pop("supporting_evidence_reviewed")
    normalized["representation"] = {
        "kind": normalized_kind,
        "unit": normalized_unit if normalized_kind == "continuous" else "",
        "categories": (list(normalized_categories) if normalized_kind == "categorical" else []),
    }
    normalized["aliases"] = list(aliases)
    normalized["distinguish_from"] = list(distinctions)
    normalized["supporting_evidence_ids"] = list(support)
    normalized["vocabulary_normalization_audit"] = {
        "audit_version": "grounded_extraction_vocabulary_normalization_audit_v1",
        "original_model_fields": original_model_fields,
        "normalization_events": normalization_events,
    }
    return _detached(normalized)


def revalidate_normalized_extraction_definition_response(
    response: Any, *, request: ExtractionDefinitionRequest
) -> dict[str, Any]:
    """Revalidate only the deterministic internal extraction projection."""

    root = _require_mapping(response, label="normalized extraction definition response")
    support = root.get("supporting_evidence_ids")
    if not isinstance(support, list):
        raise TypeError("normalized supporting_evidence_ids must be a JSON list")
    if len(support) != len(set(support)):
        raise ValueError("normalized supporting_evidence_ids contain duplicates")
    audit = _require_mapping(
        root.get("vocabulary_normalization_audit"),
        label="vocabulary_normalization_audit",
    )
    original = _require_mapping(
        audit.get("original_model_fields"),
        label="vocabulary_normalization_audit.original_model_fields",
    )
    wire = _detached(root)
    wire.pop("vocabulary_normalization_audit", None)
    wire.pop("supporting_evidence_ids", None)
    wire["representation"] = original.get("representation")
    wire["aliases"] = original.get("aliases")
    wire["distinguish_from"] = original.get("distinguish_from")
    wire["supporting_evidence_reviewed"] = True
    validated = validate_extraction_definition_response(wire, request=request)
    if canonical_json(validated) != canonical_json(root):
        raise ValueError("normalized extraction response is not the deterministic projection")
    return validated
