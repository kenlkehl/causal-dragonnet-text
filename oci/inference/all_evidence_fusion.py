"""Fold-local fusion of heterogeneous text-discovery evidence.

This module is intentionally orchestration-only.  It does not fit a language
model and it does not know any dataset-specific clinical variables.  Its job is
to place a narrow, auditable boundary between fold-training evidence and a
remote feature-selection/proposal agent:

* every evidence input carries the same explicit train/heldout provenance;
* only documented fields from each discovery family cross the boundary;
* row identifiers are checked and then removed from the prompt;
* candidates receive opaque IDs and their extraction contracts are frozen;
* an agent can select contracts by ID, but cannot rewrite them.

When no candidate pool is supplied, the same evidence can be used for a small,
grounded proposal response.  Response validation remains deterministic and
does not call an agent.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Hashable, Mapping, Sequence

FUSION_PROMPT_VERSION = "all_evidence_candidate_fusion_v10"
EVIDENCE_CONTRACT_GROUNDING_VERSION = "all_evidence_contract_name_grounding_v2"
LEGACY_COMPACTION_STRATEGY_VERSION = "legacy_complete_canonical_order_v2"
EXACT_INNER_RECURRENCE_VERSION = "exact_inner_normalized_term_recurrence_v2"
SOURCE_TEXT_TEMPORAL_POLICY = "source_text_temporally_valid_by_design_v1"
SOURCE_TEXT_TEMPORAL_BOUNDARY_ENFORCED = False

LEGACY_ALL_SOURCE = "legacy_all_source"
TFIDF_TOPIC_SOURCE = "tfidf_topics"
NEURAL_QUERY_SOURCE = "neural_query_moments"
SPARSE_QUERY_SOURCE = "sparse_query_moments"
VALID_EVIDENCE_INPUT_KINDS = frozenset(
    {
        LEGACY_ALL_SOURCE,
        TFIDF_TOPIC_SOURCE,
        NEURAL_QUERY_SOURCE,
        SPARSE_QUERY_SOURCE,
    }
)

BOW_NUISANCE = "bow_nuisance"
BOW_R_LOSS = "bow_r_loss"
MATCHED_PAIR_UPLIFT = "matched_pair_uplift"
HTR_NEURAL = "htr_neural"
EMBEDDING_WHOLE_COHORT = "embedding_whole_cohort"
EMBEDDING_CLUSTERED = "embedding_clustered"
TFIDF_SEMANTIC_RETRIEVAL = "tfidf_semantic_retrieval_contrasts"
TFIDF_TOPICS = "tfidf_topics"
TFIDF_ORPHAN_NGRAMS = "tfidf_orphan_ngrams"
NEURAL_QUERY_MOMENTS = "neural_query_moments"
SPARSE_QUERY_MOMENTS = "sparse_query_moments"
# Backward-compatible import name.  It intentionally denotes only learned
# neural cohort-query evidence now; sparse lexical moments have their own
# family and can no longer masquerade as the neural method.
QUERY_MOMENTS = NEURAL_QUERY_MOMENTS

ALL_SOURCE_FAMILIES = (
    BOW_NUISANCE,
    BOW_R_LOSS,
    MATCHED_PAIR_UPLIFT,
    HTR_NEURAL,
    EMBEDDING_WHOLE_COHORT,
    EMBEDDING_CLUSTERED,
    TFIDF_TOPICS,
    TFIDF_ORPHAN_NGRAMS,
    NEURAL_QUERY_MOMENTS,
    SPARSE_QUERY_MOMENTS,
)
_SOURCE_FAMILY_SET = frozenset(ALL_SOURCE_FAMILIES)
_CANDIDATE_SOURCE_FAMILY_SET = _SOURCE_FAMILY_SET | {TFIDF_SEMANTIC_RETRIEVAL}

_VALID_SCOPES = frozenset({"outer_train", "inner_train"})
_VALID_ROLES = frozenset({"confounder", "effect_modifier"})
_VALID_TYPES = frozenset({"categorical", "continuous"})
_FORBIDDEN_KEY = re.compile(
    r"(?:^|_)(?:oracle|true|ground_truth)(?:_|$)|(?:oracle|ground_truth)",
    flags=re.IGNORECASE,
)
_FORBIDDEN_STRING = re.compile(
    r"\boracle\b|\bground[\s_-]*truth\b|\btrue\b|\btrue[_-][a-z0-9_]+\b",
    flags=re.IGNORECASE,
)
_SNAKE_CASE_NAME = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_FORBIDDEN_IDENTIFIER_NAME = re.compile(
    r"(?:^|_)(?:patient_id|medical_record_number|mrn|account_number|"
    r"accession_number|patient_name|email_address|phone_number)(?:_|$)",
    flags=re.IGNORECASE,
)
_FORBIDDEN_IDENTIFIER_DESCRIPTION = re.compile(
    r"\b(?:patient identifier|medical record number|account number|"
    r"accession number|email address|phone number)\b",
    flags=re.IGNORECASE,
)
_CONCEPT_WORD = re.compile(r"[a-z0-9]+")
_STRUCTURAL_NAME_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "baseline",
        "before",
        "by",
        "current",
        "documented",
        "effect",
        "feature",
        "for",
        "from",
        "in",
        "initial",
        "into",
        "is",
        "not",
        "of",
        "on",
        "or",
        "per",
        "pretreatment",
        "pre",
        "patient",
        "clinical",
        "recorded",
        "reported",
        "the",
        "to",
        "via",
        "with",
        "without",
        "measure",
        "measurement",
        "value",
        "level",
        "result",
        "status",
        "state",
        "present",
        "absent",
        "positive",
        "negative",
        "unknown",
        "other",
        "missing",
        "available",
        "confounder",
        "modifier",
        "variable",
        "group",
        "numeric",
        "continuous",
        "categorical",
        "pattern",
        "phrase",
        "term",
        "normal",
        "abnormal",
        "impaired",
        "unimpaired",
        "mild",
        "moderate",
        "severe",
        "low",
        "high",
        "never",
        "former",
        "wild",
        "type",
        "category",
        "class",
        "code",
        "score",
        "indicator",
        "flag",
        "number",
        "amount",
        "metric",
    }
)
_EVIDENCE_CONCEPT_TEXT_FIELDS = frozenset(
    {
        "term",
        "terms",
        "feature",
        "features",
        "phrase",
        "phrases",
        "ngram",
        "ngrams",
        "concept",
        "concepts",
        "description",
        "summary",
        "summaries",
        "top_terms",
        "top_ngrams",
        "top_contrastive_ngrams",
        "contrastive_ngrams",
        "text",
        "texts",
        "chunks",
        "retrieved_training_excerpts",
        "evidence_snippet",
        "attended_token_summary",
        "top_token_spans",
    }
)
_CATEGORY_META_LANGUAGE = re.compile(
    r"\bcategor(?:y|ies|ical\s+variables?)\b.*\b(?:required|canonical|allowed|valid|"
    r"provide|list|values?)\b|"
    r"\b(?:required|canonical|allowed|valid|provide|list|values?)\b.*"
    r"\bcategorical\s+variables?\b",
    flags=re.IGNORECASE,
)
_CATEGORY_PLACEHOLDER_TOKEN = re.compile(
    r"^(?:category|value)\s*(?:[a-z]|\d+)$",
    flags=re.IGNORECASE,
)
_ROW_ID_KEY = re.compile(
    r"(?:^|_)(?:row_id|row_ids|candidate_row_id|control_row_id)$",
    flags=re.IGNORECASE,
)

_MAX_CANDIDATE_POOL = 256

# Put compact, causally targeted summaries before large sparse banks while
# preserving canonical content as the deterministic tie-breaker. This prevents
# pair/HTR and residualized signals from being buried in long prompts; it does
# not remove or weaken later evidence.
_PROMPT_FAMILY_PRIORITY = {
    MATCHED_PAIR_UPLIFT: 0,
    HTR_NEURAL: 1,
    BOW_R_LOSS: 2,
    BOW_NUISANCE: 3,
    EMBEDDING_CLUSTERED: 4,
    EMBEDDING_WHOLE_COHORT: 5,
    TFIDF_ORPHAN_NGRAMS: 6,
    TFIDF_TOPICS: 7,
    NEURAL_QUERY_MOMENTS: 8,
    SPARSE_QUERY_MOMENTS: 9,
}


def _normalize_row_id(value: Any, *, field_name: str) -> Hashable:
    if value is None:
        raise ValueError(f"{field_name} cannot contain missing row IDs")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"{field_name} cannot contain missing row IDs")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} row IDs must be hashable") from exc
    return value


def _unique_row_tuple(values: Sequence[Hashable], *, field_name: str) -> tuple[Hashable, ...]:
    normalized = tuple(_normalize_row_id(value, field_name=field_name) for value in values)
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique row IDs")
    return normalized


def _jsonable_row_id(value: Hashable) -> dict[str, str]:
    """Represent possibly mixed-type IDs without relying on cross-type sorting."""

    return {"type": type(value).__name__, "value": repr(value)}


@dataclass(frozen=True)
class FoldEvidenceProvenance:
    """Exact split lineage for one fold-local evidence artifact."""

    outer_fold: int
    train_row_ids: tuple[Hashable, ...]
    heldout_row_ids: tuple[Hashable, ...]
    scope: str = "outer_train"
    inner_fold: int | None = None
    artifact_id: str = "unspecified"

    def __post_init__(self) -> None:
        if int(self.outer_fold) < 1:
            raise ValueError("outer_fold must be a positive one-based fold number")
        train = _unique_row_tuple(self.train_row_ids, field_name="train_row_ids")
        heldout = _unique_row_tuple(
            self.heldout_row_ids,
            field_name="heldout_row_ids",
        )
        overlap = set(train).intersection(heldout)
        if overlap:
            example = next(iter(overlap))
            raise ValueError(f"train and heldout provenance overlap at row {example!r}")
        scope = str(self.scope).strip().lower()
        if scope not in _VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(_VALID_SCOPES)}")
        inner_fold = self.inner_fold
        if scope == "inner_train" and (inner_fold is None or int(inner_fold) < 1):
            raise ValueError("inner_train evidence requires a positive inner_fold")
        if scope == "outer_train" and inner_fold is not None:
            raise ValueError("outer_train evidence cannot declare an inner_fold")
        artifact_id = str(self.artifact_id).strip()
        if not artifact_id:
            raise ValueError("artifact_id must be non-empty")
        object.__setattr__(self, "outer_fold", int(self.outer_fold))
        object.__setattr__(self, "train_row_ids", train)
        object.__setattr__(self, "heldout_row_ids", heldout)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(
            self,
            "inner_fold",
            None if inner_fold is None else int(inner_fold),
        )
        object.__setattr__(self, "artifact_id", artifact_id)

    @property
    def split_fingerprint(self) -> str:
        payload = {
            "outer_fold": self.outer_fold,
            "scope": self.scope,
            "inner_fold": self.inner_fold,
            "train_row_ids": sorted(
                (_jsonable_row_id(value) for value in self.train_row_ids),
                key=lambda item: (item["type"], item["value"]),
            ),
            "heldout_row_ids": sorted(
                (_jsonable_row_id(value) for value in self.heldout_row_ids),
                key=lambda item: (item["type"], item["value"]),
            ),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FoldEvidenceInput:
    """A raw source payload accompanied by mandatory fold provenance."""

    source_kind: str
    payload: Mapping[str, Any] = field(repr=False)
    provenance: FoldEvidenceProvenance

    def __post_init__(self) -> None:
        source_kind = str(self.source_kind).strip().lower()
        if source_kind not in VALID_EVIDENCE_INPUT_KINDS:
            raise ValueError(f"source_kind must be one of {sorted(VALID_EVIDENCE_INPUT_KINDS)}")
        if not isinstance(self.payload, Mapping):
            raise TypeError("evidence payload must be a mapping")
        if not isinstance(self.provenance, FoldEvidenceProvenance):
            raise TypeError("provenance must be FoldEvidenceProvenance")
        object.__setattr__(self, "source_kind", source_kind)


@dataclass(frozen=True, init=False)
class CandidateContract:
    """An extraction contract frozen before an agent sees its opaque ID."""

    _spec_json: str = field(repr=False)
    source_families: tuple[str, ...]

    def __init__(
        self,
        extraction_spec: Mapping[str, Any] | Any,
        *,
        source_families: Sequence[str] = (),
    ) -> None:
        if is_dataclass(extraction_spec) and not isinstance(extraction_spec, type):
            extraction_spec = asdict(extraction_spec)
        if not isinstance(extraction_spec, Mapping):
            raise TypeError("extraction_spec must be a mapping or dataclass instance")
        spec = dict(extraction_spec)
        _reject_forbidden_content(spec, path="candidate.extraction_spec")
        _validate_extraction_spec(spec, source="candidate.extraction_spec")
        families = tuple(dict.fromkeys(str(value).strip() for value in source_families))
        unknown = set(families) - _CANDIDATE_SOURCE_FAMILY_SET
        if unknown:
            raise ValueError(f"unknown candidate source families: {sorted(unknown)}")
        object.__setattr__(self, "_spec_json", _canonical_json(spec))
        object.__setattr__(self, "source_families", families)

    @property
    def extraction_spec(self) -> dict[str, Any]:
        """Return a fresh copy so callers cannot mutate the frozen contract."""

        return json.loads(self._spec_json)


@dataclass(frozen=True)
class EvidenceContractGrounding:
    """Versioned, value-free audit of one evidence-to-contract binding."""

    supported: bool
    contract_name: str
    match_rule: str
    required_name_anchors: tuple[str, ...]
    matched_evidence_anchors: tuple[str, ...]
    matched_evidence_paths: tuple[str, ...]
    schema_version: str = EVIDENCE_CONTRACT_GROUNDING_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "supported": self.supported,
            "contract_name": self.contract_name,
            "match_rule": self.match_rule,
            "required_name_anchors": list(self.required_name_anchors),
            "matched_evidence_anchors": list(self.matched_evidence_anchors),
            "matched_evidence_paths": list(self.matched_evidence_paths),
        }


def _normalized_concept_token(raw: str) -> str | None:
    token = str(raw).strip().casefold()
    if not token:
        return None
    # Numeric values and one-letter prose fragments cannot identify a concept.
    # All other tokens are syntax-eligible, including arbitrary two- and
    # three-character acronyms and alphanumeric forms such as L1.  Such a short
    # token provides support only through the same exact-token check as a long
    # token; there is deliberately no domain vocabulary or acronym expansion.
    if token.isdecimal():
        return None
    if token in _STRUCTURAL_NAME_TOKENS:
        return None
    if len(token) == 1 and token.isalpha():
        return None
    return token


def _normalized_concept_tokens(text: Any) -> tuple[str, ...]:
    tokens: list[str] = []
    for raw in _CONCEPT_WORD.findall(str(text).casefold()):
        normalized = _normalized_concept_token(raw)
        if normalized is not None and normalized not in tokens:
            tokens.append(normalized)
    return tuple(tokens)


def _evidence_concept_entries(
    content: Any,
    *,
    parent_key: str = "",
    path: str = "content",
) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    if isinstance(content, Mapping):
        for raw_key, child in content.items():
            key = str(raw_key).strip().casefold()
            entries.extend(
                _evidence_concept_entries(
                    child,
                    parent_key=key,
                    path=f"{path}.{key}",
                )
            )
    elif isinstance(content, (list, tuple)):
        for index, child in enumerate(content):
            entries.extend(
                _evidence_concept_entries(
                    child,
                    parent_key=parent_key,
                    path=f"{path}[{index}]",
                )
            )
    elif isinstance(content, str) and parent_key in _EVIDENCE_CONCEPT_TEXT_FIELDS:
        compact = " ".join(content.split())
        if compact and len(compact) <= 420 and not re.fullmatch(r"[0-9a-f]{32,}", compact):
            entries.append((path, compact))
    return tuple(entries)


def ground_evidence_to_extraction_contract(
    evidence: Mapping[str, Any],
    contract: Mapping[str, Any] | CandidateContract,
) -> EvidenceContractGrounding:
    """Ground evidence using normalized exact lexical identity in the name.

    Descriptions, categories, and aliases are extraction instructions controlled
    by the proposing agent.  They can never authenticate a citation.  Every
    meaningful name anchor must occur verbatim after case/punctuation
    normalization within one evidence concept entry.  Tokens from unrelated
    entries cannot be assembled into synthetic support.  No synonym map,
    domain vocabulary, or acronym expansion participates in this decision.
    """

    canonical = (
        contract.extraction_spec
        if isinstance(contract, CandidateContract)
        else CandidateContract(contract).extraction_spec
    )
    contract_name = str(canonical["name"])
    required_name_anchors = _normalized_concept_tokens(contract_name.replace("_", " "))
    if not required_name_anchors:
        return EvidenceContractGrounding(
            supported=False,
            contract_name=contract_name,
            match_rule="invalid_generic_or_numeric_contract_name",
            required_name_anchors=(),
            matched_evidence_anchors=(),
            matched_evidence_paths=(),
        )

    content = evidence.get("content")
    if not isinstance(content, Mapping):
        return EvidenceContractGrounding(
            supported=False,
            contract_name=contract_name,
            match_rule="missing_evidence_concept_content",
            required_name_anchors=required_name_anchors,
            matched_evidence_anchors=(),
            matched_evidence_paths=(),
        )

    entries = _evidence_concept_entries(content)
    tokens_by_path = {path: frozenset(_normalized_concept_tokens(text)) for path, text in entries}
    evidence_tokens = frozenset().union(*tokens_by_path.values()) if tokens_by_path else frozenset()
    canonical_anchors = frozenset(required_name_anchors)
    matched_paths = tuple(
        sorted(
            path
            for path, path_tokens in tokens_by_path.items()
            if canonical_anchors and canonical_anchors.issubset(path_tokens)
        )
    )
    if matched_paths:
        return EvidenceContractGrounding(
            supported=True,
            contract_name=contract_name,
            match_rule="all_required_name_anchors",
            required_name_anchors=required_name_anchors,
            matched_evidence_anchors=tuple(sorted(canonical_anchors)),
            matched_evidence_paths=matched_paths,
        )

    partial = canonical_anchors.intersection(evidence_tokens)
    partial_paths = tuple(
        sorted(
            path
            for path, path_tokens in tokens_by_path.items()
            if path_tokens.intersection(partial)
        )
    )
    return EvidenceContractGrounding(
        supported=False,
        contract_name=contract_name,
        match_rule="missing_required_name_anchors",
        required_name_anchors=required_name_anchors,
        matched_evidence_anchors=tuple(sorted(partial)),
        matched_evidence_paths=partial_paths,
    )


def evidence_supports_extraction_contract(
    evidence: Mapping[str, Any],
    contract: Mapping[str, Any] | CandidateContract,
) -> bool:
    """Compatibility boolean for the versioned grounding result."""

    return ground_evidence_to_extraction_contract(evidence, contract).supported


@dataclass(frozen=True)
class FusionEvidenceBlock:
    evidence_id: str
    source_families: tuple[str, ...]
    role_hint: str
    _content_json: str = field(repr=False)

    @property
    def content(self) -> dict[str, Any]:
        return json.loads(self._content_json)

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_families": list(self.source_families),
            "role_hint": self.role_hint,
            "content": self.content,
        }


@dataclass(frozen=True)
class FusionPromptCandidate:
    candidate_id: str
    contract: CandidateContract = field(repr=False)

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "extraction_spec": self.contract.extraction_spec,
            "source_families": list(self.contract.source_families),
        }


@dataclass(frozen=True)
class AllEvidenceFusionRequest:
    """Sanitized deterministic request ready for a remote JSON agent."""

    outer_fold: int
    split_fingerprint: str
    mode: str
    max_candidates: int
    evidence_blocks: tuple[FusionEvidenceBlock, ...]
    candidates: tuple[FusionPromptCandidate, ...]
    _coverage_json: str = field(repr=False)

    @property
    def source_family_coverage(self) -> dict[str, Any]:
        return json.loads(self._coverage_json)

    def context(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "prompt_version": FUSION_PROMPT_VERSION,
            "source_text_temporal_policy": source_text_temporal_policy_audit(),
            "outer_fold": self.outer_fold,
            "split_fingerprint": self.split_fingerprint,
            "mode": self.mode,
            "max_candidates": self.max_candidates,
            "source_family_coverage": self.source_family_coverage,
            "evidence": [block.as_prompt_dict() for block in self.evidence_blocks],
        }
        if self.mode == "select":
            base["candidates"] = [item.as_prompt_dict() for item in self.candidates]
            base["response_contract"] = {
                "selected_candidate_ids": ["candidate_0001"],
                "selection_notes": [
                    {
                        "candidate_id": "candidate_0001",
                        "supporting_evidence_ids": ["evidence_0001"],
                        "supporting_source_families": [BOW_NUISANCE],
                        "reason": "brief evidence-grounded reason",
                    }
                ],
            }
        else:
            base["response_contract"] = {
                "proposals": [
                    {
                        "name": "snake_case_variable",
                        "type": "categorical|continuous",
                        "categories": ["absent", "present"],
                        "value_aliases": {
                            "absent": ["negative"],
                            "present": ["positive"],
                        },
                        "roles": ["confounder|effect_modifier"],
                        "description": "precise extraction contract for the supported construct",
                        "supporting_evidence_ids": ["evidence_0001"],
                        "supporting_source_families": [BOW_NUISANCE],
                        "rationale": "brief evidence-grounded reason",
                    }
                ]
            }
        return base

    def render_prompt(self) -> str:
        return render_all_evidence_fusion_prompt(self)


@dataclass(frozen=True)
class FusionResult:
    mode: str
    selected_candidate_ids: tuple[str, ...] = ()
    selected_contracts: tuple[CandidateContract, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    proposed_specs: tuple[dict[str, Any], ...] = field(
        default_factory=tuple,
        repr=False,
    )
    response_audit: Mapping[str, Any] = field(default_factory=dict)

    @property
    def selected_specs(self) -> list[dict[str, Any]]:
        return [contract.extraction_spec for contract in self.selected_contracts]


def prepare_all_evidence_fusion(
    evidence_inputs: Sequence[FoldEvidenceInput],
    *,
    candidates: Sequence[CandidateContract | Mapping[str, Any] | Any] = (),
    max_candidates: int = 16,
) -> AllEvidenceFusionRequest:
    """Validate, compact, and freeze one fold-local fusion request.

    ``max_candidates`` caps selected contracts or newly proposed contracts; it
    does not truncate a supplied candidate pool.
    """

    if not evidence_inputs:
        raise ValueError("at least one fold evidence input is required")
    if not 1 <= int(max_candidates) <= 64:
        raise ValueError("max_candidates must be in [1, 64]")
    inputs = tuple(evidence_inputs)
    if not all(isinstance(item, FoldEvidenceInput) for item in inputs):
        raise TypeError("all evidence inputs must be FoldEvidenceInput instances")
    kinds = [item.source_kind for item in inputs]
    if len(kinds) != len(set(kinds)):
        raise ValueError("each source_kind can appear at most once in a fusion request")

    reference = inputs[0].provenance
    for item in inputs:
        if item.provenance.split_fingerprint != reference.split_fingerprint:
            raise ValueError(
                "all evidence sources must have identical fold train/heldout provenance"
            )
        _reject_forbidden_content(item.payload, path=f"evidence.{item.source_kind}")
        _validate_payload_provenance(item.payload, item.provenance)

    raw_blocks: list[tuple[tuple[str, ...], str, dict[str, Any]]] = []
    legacy_compaction_audit: Mapping[str, Any] | None = None
    for item in sorted(inputs, key=lambda value: value.source_kind):
        if item.source_kind == LEGACY_ALL_SOURCE:
            legacy_blocks, legacy_compaction_audit = _compact_legacy_all_source(item.payload)
            raw_blocks.extend(legacy_blocks)
        elif item.source_kind == TFIDF_TOPIC_SOURCE:
            raw_blocks.extend(_compact_tfidf_evidence(item.payload))
        elif item.source_kind == NEURAL_QUERY_SOURCE:
            raw_blocks.extend(
                _compact_query_moment_evidence(
                    item.payload,
                    source_family=NEURAL_QUERY_MOMENTS,
                    content_kind="neural_query_moment",
                )
            )
        elif item.source_kind == SPARSE_QUERY_SOURCE:
            raw_blocks.extend(
                _compact_query_moment_evidence(
                    item.payload,
                    source_family=SPARSE_QUERY_MOMENTS,
                    content_kind="sparse_query_moment",
                )
            )
    if not raw_blocks:
        raise ValueError("no allowlisted evidence was found in the supplied payloads")

    # Sorting by canonical content makes the prompt independent of source-map
    # insertion order.  Exact duplicates are removed before opaque IDs are made.
    unique_blocks: dict[str, tuple[tuple[str, ...], str, dict[str, Any]]] = {}
    for families, role, content in raw_blocks:
        normalized_families = tuple(
            family for family in ALL_SOURCE_FAMILIES if family in set(families)
        )
        if not normalized_families:
            continue
        _reject_forbidden_content(content, path="compacted_evidence")
        key = _canonical_json(
            {
                "source_families": normalized_families,
                "role_hint": role,
                "content": content,
            }
        )
        unique_blocks[key] = (normalized_families, role, content)

    def prompt_order_key(
        item: tuple[str, tuple[tuple[str, ...], str, dict[str, Any]]],
    ) -> tuple[int, int, str]:
        canonical, (families, _role, _content) = item
        priority = min(_PROMPT_FAMILY_PRIORITY[family] for family in families)
        return priority, -len(families), canonical

    ordered_blocks = [value for _key, value in sorted(unique_blocks.items(), key=prompt_order_key)]
    evidence_blocks = tuple(
        FusionEvidenceBlock(
            evidence_id=f"evidence_{index:04d}",
            source_families=families,
            role_hint=role,
            _content_json=_canonical_json(content),
        )
        for index, (families, role, content) in enumerate(ordered_blocks, start=1)
    )

    contracts = tuple(
        candidate if isinstance(candidate, CandidateContract) else CandidateContract(candidate)
        for candidate in candidates
    )
    if len(contracts) > _MAX_CANDIDATE_POOL:
        raise ValueError(f"candidate pool exceeds the audited limit of {_MAX_CANDIDATE_POOL}")
    present_families = {family for block in evidence_blocks for family in block.source_families}
    for contract in contracts:
        unavailable = set(contract.source_families) - present_families
        if unavailable:
            raise ValueError(
                "candidate cites source families absent from this fold: " f"{sorted(unavailable)}"
            )
    prompt_candidates = tuple(
        FusionPromptCandidate(
            candidate_id=f"candidate_{index:04d}",
            contract=contract,
        )
        for index, contract in enumerate(contracts, start=1)
    )
    coverage = _source_family_coverage(
        evidence_blocks,
        prompt_candidates,
        legacy_compaction_audit=legacy_compaction_audit,
    )
    return AllEvidenceFusionRequest(
        outer_fold=reference.outer_fold,
        split_fingerprint=reference.split_fingerprint,
        mode="select" if prompt_candidates else "propose",
        max_candidates=int(max_candidates),
        evidence_blocks=evidence_blocks,
        candidates=prompt_candidates,
        _coverage_json=_canonical_json(coverage),
    )


def render_all_evidence_fusion_prompt(request: AllEvidenceFusionRequest) -> str:
    """Render a deterministic prompt containing no dataset-specific anchors."""

    if not isinstance(request, AllEvidenceFusionRequest):
        raise TypeError("request must be AllEvidenceFusionRequest")
    action = (
        "Select only from the supplied immutable extraction contracts. Return IDs; "
        "do not rewrite a contract."
        if request.mode == "select"
        else (
            "Propose a comprehensive but nonredundant set of executable extraction "
            "contracts grounded in evidence IDs. Return as many distinct variables as "
            "the evidence supports, up to the cap; do not default to a short list."
        )
    )
    prompt_context = request.context()
    # Preserve provenance in the immutable request without turning it into a
    # model-facing policy instruction.
    prompt_context.pop("source_text_temporal_policy", None)
    payload = json.dumps(
        prompt_context,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    )
    return (
        "You are fusing noisy, fold-training-only evidence about "
        "patient variables for causal adjustment and treatment-effect heterogeneity.\n\n"
        "Prefer variables supported by independent source families. Treat neural, "
        "matched-pair, embedding, and sparse-text signals as fallible discovery "
        "evidence. Exclude administrative artifacts, identifiers, and "
        "redundant aliases. Use only "
        "the supplied evidence; do not use held-out rows or external dataset "
        "knowledge. Prefer one specific, directly extractable construct per contract; "
        "do not collapse distinct measurements or categorical states into a broad "
        "composite variable. Seek complementary variables across evidence families "
        "and across demographic, functional, laboratory, diagnostic, molecular, and "
        "clinical-history domains when those domains are actually supported. "
        "Judge recurrence across the entire context, not evidence order: first form "
        "an inventory of specific constructs that recur across blocks or independent "
        "families, then cover both confounding and effect-modification roles. Do not "
        "promote an isolated personal name, family-history anecdote, rare subtype, "
        "medication option, administrative phrase, or single numeric example unless "
        "other supplied evidence independently supports that same patient variable. "
        "Candidate IDs and evidence IDs are opaque.\n\n"
        "Every cited evidence block must itself contain all normalized exact lexical "
        "identity anchors in the selected or proposed contract name within one concept "
        "entry. Short acronyms count only when that exact token occurs; do not expand "
        "or infer acronyms or synonyms. Descriptions, categories, aliases, numeric "
        "codes, and structural name words cannot establish that match. Never use an "
        "unrelated real block merely because its opaque ID and source family are "
        "valid.\n\n"
        "For a categorical proposal, provide every supported mutually exclusive, "
        "canonical value specific to that variable, with at least two values. Do not "
        "collapse or omit supported values to meet an implicit category-count cap. "
        "Optional value_aliases must map exact canonical categories to nonoverlapping "
        "normalized surface forms. Omit both categories and value_aliases for a "
        "continuous proposal. The response-contract values are shape examples, not "
        "reusable content: never return schema instructions or placeholder labels as "
        "category values.\n\n"
        f"{action} Return exactly one JSON object and at most "
        f"{request.max_candidates} candidates.\n\n"
        f"Fusion context:\n{payload}"
    )


def render_all_evidence_fusion_context_prompt(context: Mapping[str, Any]) -> str:
    """Render a serialized request context after reconstructing its invariants.

    Agent clients conventionally pass dictionaries rather than request objects.
    Reconstructing the request here keeps the prompt branch stateless while
    ensuring opaque IDs, candidate contracts, and evidence families still obey
    the same checks as a freshly prepared request.
    """

    return render_all_evidence_fusion_prompt(_fusion_request_from_context(context))


def validate_all_evidence_fusion_response(
    request: AllEvidenceFusionRequest,
    response: Mapping[str, Any],
) -> FusionResult:
    """Validate an agent response and recover frozen specs or grounded proposals."""

    if not isinstance(request, AllEvidenceFusionRequest):
        raise TypeError("request must be AllEvidenceFusionRequest")
    if not isinstance(response, Mapping):
        raise TypeError("fusion response must be one JSON object")
    _reject_forbidden_content(response, path="response")
    if request.mode == "select":
        return _validate_selection_response(request, response)
    return _validate_proposal_response(request, response)


def _normalize_agent_response_citation_families(
    response: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Conservatively normalize one freshly parsed remote-agent response.

    This is deliberately an agent-boundary helper, not part of standalone
    response validation. Cached or injected responses therefore retain the
    strict contract enforced by :func:`validate_all_evidence_fusion_response`.
    The caller must invoke this only on a freshly parsed remote-agent response.

    Citation families are redundant with the cited opaque evidence IDs, so
    they are derived from the request.  A malformed row cannot invalidate
    otherwise valid higher-ranked rows: proposal rows with invalid specs or
    citations and selection entries without a valid note are removed.  The
    original order is retained, duplicate names/IDs keep their first valid
    occurrence, and the request cap is applied by stable truncation.  No spec
    fields (in particular categorical values) are repaired or invented.

    If a non-empty response contains no salvageable row, the response is left
    unchanged so the ordinary schema-repair path can ask the remote agent for
    a corrected response instead of silently accepting an empty inventory.
    """

    if not isinstance(response, Mapping):
        raise TypeError("fusion response must be one JSON object")
    # This check intentionally precedes every row-level decision.  Forbidden
    # content in a row that would otherwise be dropped must still fail closed.
    _reject_forbidden_content(response, path="response")
    request = _fusion_request_from_context(context)
    normalized = copy.deepcopy(dict(response))
    evidence_by_id = {
        block.evidence_id: tuple(block.source_families) for block in request.evidence_blocks
    }
    evidence_blocks_by_id = {
        block.evidence_id: block.as_prompt_dict() for block in request.evidence_blocks
    }
    item_key = "selection_notes" if request.mode == "select" else "proposals"
    citation_rows: list[dict[str, Any]] = []
    grounding_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    if request.mode == "select":
        retained_count, input_count, salvage_applied = _normalize_fresh_selection_response(
            normalized,
            request=request,
            evidence_by_id=evidence_by_id,
            evidence_blocks_by_id=evidence_blocks_by_id,
            citation_rows=citation_rows,
            grounding_rows=grounding_rows,
            dropped=dropped,
        )
    else:
        retained_count, input_count, salvage_applied = _normalize_fresh_proposal_response(
            normalized,
            request=request,
            evidence_by_id=evidence_by_id,
            evidence_blocks_by_id=evidence_blocks_by_id,
            citation_rows=citation_rows,
            grounding_rows=grounding_rows,
            dropped=dropped,
        )

    citation_audit = {
        "schema_version": "all_evidence_citation_family_normalization_v2",
        "authoritative_source": "request_evidence_blocks",
        "canonical_family_order": list(ALL_SOURCE_FAMILIES),
        "item_container": item_key,
        "eligible_item_count": len(citation_rows),
        "changed_item_count": sum(bool(row["changed"]) for row in citation_rows),
        "skipped_invalid_citation_item_count": sum(
            row["reason_code"] == "invalid_citations" for row in dropped
        ),
        "skipped_invalid_citation_paths": [
            row["path"] for row in dropped if row["reason_code"] == "invalid_citations"
        ],
        "items": citation_rows,
    }
    grounding_audit = {
        "schema_version": "all_evidence_citation_grounding_normalization_v1",
        "grounding_schema_version": EVIDENCE_CONTRACT_GROUNDING_VERSION,
        "authoritative_source": (
            "request_evidence_blocks_and_frozen_candidate_contract_names"
            if request.mode == "select"
            else "request_evidence_blocks_and_fresh_proposal_contract_names"
        ),
        "evaluated_item_count": len(grounding_rows),
        "evaluated_citation_count": sum(
            len(row["original_supporting_evidence_ids"]) for row in grounding_rows
        ),
        "retained_citation_count": sum(
            len(row["retained_supporting_evidence_ids"]) for row in grounding_rows
        ),
        "dropped_unrelated_citation_count": sum(
            len(row["dropped_unrelated_evidence_ids"]) for row in grounding_rows
        ),
        "changed_item_count": sum(bool(row["changed"]) for row in grounding_rows),
        "zero_grounding_item_count": sum(not bool(row["item_retained"]) for row in grounding_rows),
        "removed_zero_grounding_item_count": sum(
            row.get("normalization_disposition") == "removed_zero_grounding"
            for row in grounding_rows
        ),
        "left_for_remote_repair_item_count": sum(
            row.get("normalization_disposition") == "left_for_remote_repair"
            for row in grounding_rows
        ),
        "items": grounding_rows,
    }
    audit = {
        "schema_version": "all_evidence_fresh_response_normalization_v3",
        "mode": request.mode,
        "item_container": item_key,
        "input_item_count": input_count,
        "retained_item_count": retained_count,
        "max_candidates": request.max_candidates,
        "salvage_applied": salvage_applied,
        "rejected_item_count": len(dropped),
        "rejected_items": dropped,
        "rejections_applied_to_response": bool(dropped) and bool(salvage_applied),
        "citation_family_normalization": citation_audit,
        "citation_grounding_normalization": grounding_audit,
    }
    return normalized, audit


def _validated_fresh_evidence_ids(
    item: Mapping[str, Any],
    *,
    evidence_by_id: Mapping[str, tuple[str, ...]],
    path: str,
) -> list[str]:
    evidence_ids = item.get("supporting_evidence_ids")
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or not all(isinstance(value, str) for value in evidence_ids)
    ):
        raise ValueError(f"{path}.supporting_evidence_ids must be a non-empty string list")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError(f"{path}.supporting_evidence_ids contains duplicates")
    unknown = set(evidence_ids) - set(evidence_by_id)
    if unknown:
        raise ValueError(f"{path} cites unknown evidence IDs: {sorted(unknown)}")
    return list(evidence_ids)


class _NoGroundedEvidenceCitations(ValueError):
    """A fresh remote item cites known blocks but none ground its contract."""

    def __init__(self, message: str, audit: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.audit = copy.deepcopy(dict(audit))


def _prune_fresh_ungrounded_evidence_ids(
    item: dict[str, Any],
    *,
    contract: Mapping[str, Any] | CandidateContract,
    evidence_by_id: Mapping[str, tuple[str, ...]],
    evidence_blocks_by_id: Mapping[str, Mapping[str, Any]],
    path: str,
) -> dict[str, Any]:
    """Retain only citations grounded by the existing strict name-anchor rule."""

    original_ids = _validated_fresh_evidence_ids(
        item,
        evidence_by_id=evidence_by_id,
        path=path,
    )
    retained_ids: list[str] = []
    dropped_ids: list[str] = []
    decisions: list[dict[str, Any]] = []
    for evidence_id in original_ids:
        grounding = ground_evidence_to_extraction_contract(
            evidence_blocks_by_id[evidence_id],
            contract,
        )
        decisions.append(
            {
                "evidence_id": evidence_id,
                "supported": bool(grounding.supported),
                "match_rule": grounding.match_rule,
                "grounding_schema_version": grounding.schema_version,
            }
        )
        if grounding.supported:
            retained_ids.append(evidence_id)
        else:
            dropped_ids.append(evidence_id)
    audit = {
        "path": path,
        "original_supporting_evidence_ids": original_ids,
        "retained_supporting_evidence_ids": retained_ids,
        "dropped_unrelated_evidence_ids": dropped_ids,
        "decisions": decisions,
        "changed": retained_ids != original_ids,
        "item_retained": bool(retained_ids),
    }
    if not retained_ids:
        raise _NoGroundedEvidenceCitations(
            f"{path} has no citations grounded by normalized contract-name anchors; "
            f"unrelated evidence IDs: {dropped_ids}",
            audit,
        )
    item["supporting_evidence_ids"] = retained_ids
    return audit


def _derive_fresh_citation_families(
    item: dict[str, Any],
    *,
    evidence_by_id: Mapping[str, tuple[str, ...]],
    path: str,
) -> dict[str, Any]:
    """Validate opaque citations and derive their redundant family labels."""

    evidence_ids = _validated_fresh_evidence_ids(
        item,
        evidence_by_id=evidence_by_id,
        path=path,
    )

    cited_family_set = {
        family for evidence_id in evidence_ids for family in evidence_by_id[evidence_id]
    }
    derived_families = [family for family in ALL_SOURCE_FAMILIES if family in cited_family_set]
    original_present = "supporting_source_families" in item
    original_families = copy.deepcopy(item.get("supporting_source_families"))
    changed = original_families != derived_families
    item["supporting_source_families"] = derived_families
    return {
        "path": path,
        "supporting_evidence_ids": list(evidence_ids),
        "original_field_present": original_present,
        "original_supporting_source_families": original_families,
        "derived_supporting_source_families": derived_families,
        "changed": changed,
    }


def _normalization_drop(
    *,
    path: str,
    reason_code: str,
    reason: str,
) -> dict[str, str]:
    return {
        "path": path,
        "reason_code": reason_code,
        "reason": _normalize_evidence_text(reason),
    }


def _normalize_fresh_proposal_response(
    response: dict[str, Any],
    *,
    request: AllEvidenceFusionRequest,
    evidence_by_id: Mapping[str, tuple[str, ...]],
    evidence_blocks_by_id: Mapping[str, Mapping[str, Any]],
    citation_rows: list[dict[str, Any]],
    grounding_rows: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
) -> tuple[int, int | None, bool]:
    """Retain valid proposal rows without modifying extraction semantics."""

    raw_items = response.get("proposals")
    if not isinstance(raw_items, list):
        return 0, None, False

    # Schema prose copied into category values is not an independently bad
    # proposal that may safely be discarded. It is evidence that the response
    # contract itself was followed incorrectly, and silently dropping only the
    # affected variables can materially change the proposed inventory. Leave
    # the response untouched so strict validation below triggers schema repair
    # before any contract can enter final selection or row extraction.
    if any(
        isinstance(item, Mapping)
        and isinstance(item.get("categories"), list)
        and any(_is_instructional_or_placeholder_category(value) for value in item["categories"])
        for item in raw_items
    ):
        return 0, len(raw_items), False

    allowed = {
        "name",
        "type",
        "categories",
        "value_aliases",
        "roles",
        "description",
        "supporting_evidence_ids",
        "supporting_source_families",
        "rationale",
    }
    valid: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    seen_names: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        path = f"proposals[{index}]"
        if not isinstance(raw_item, Mapping):
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="not_object",
                    reason=f"{path} must be an object",
                )
            )
            continue
        item = copy.deepcopy(dict(raw_item))
        unexpected = set(item) - allowed
        if unexpected:
            reason = f"{path} has unsupported fields: {sorted(map(str, unexpected))}"
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="unsupported_fields",
                    reason=reason,
                )
            )
            continue
        spec = {
            key: item.get(key)
            for key in (
                "name",
                "type",
                "categories",
                "value_aliases",
                "roles",
                "description",
            )
            if item.get(key) is not None
        }
        try:
            _validate_extraction_spec(spec, source=path)
        except (TypeError, ValueError) as exc:
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="malformed_spec",
                    reason=str(exc),
                )
            )
            continue
        name = str(spec["name"])
        if name in seen_names:
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="duplicate_proposal_name",
                    reason=f"{path}.name duplicates an earlier valid proposal",
                )
            )
            continue
        try:
            grounding_row = _prune_fresh_ungrounded_evidence_ids(
                item,
                contract=spec,
                evidence_by_id=evidence_by_id,
                evidence_blocks_by_id=evidence_blocks_by_id,
                path=path,
            )
            citation_row = _derive_fresh_citation_families(
                item,
                evidence_by_id=evidence_by_id,
                path=path,
            )
        except _NoGroundedEvidenceCitations as exc:
            exc.audit["normalization_disposition"] = "pending_zero_grounding"
            grounding_rows.append(exc.audit)
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="no_lexically_grounded_citations",
                    reason=str(exc),
                )
            )
            continue
        except (TypeError, ValueError) as exc:
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="invalid_citations",
                    reason=str(exc),
                )
            )
            continue
        seen_names.add(name)
        if grounding_row["changed"]:
            citation_row["changed"] = True
        grounding_row["normalization_disposition"] = "eligible"
        grounding_rows.append(grounding_row)
        valid.append((index, item, citation_row, grounding_row))

    # If every attempted proposal was malformed, preserve the original object
    # so strict validation triggers the configured remote schema-repair pass.
    if raw_items and not valid:
        for grounding_row in grounding_rows:
            grounding_row["normalization_disposition"] = "left_for_remote_repair"
        return 0, len(raw_items), False

    for grounding_row in grounding_rows:
        if grounding_row["normalization_disposition"] == "pending_zero_grounding":
            grounding_row["normalization_disposition"] = "removed_zero_grounding"
    retained = valid[: request.max_candidates]
    for _index, _item, _citation, grounding_row in retained:
        grounding_row["normalization_disposition"] = "retained"
    for index, _item, _citation, _grounding in valid[request.max_candidates :]:
        _grounding["normalization_disposition"] = "max_candidates_truncation"
        path = f"proposals[{index}]"
        dropped.append(
            _normalization_drop(
                path=path,
                reason_code="max_candidates_truncation",
                reason=f"{path} was beyond the stable max_candidates cutoff",
            )
        )
    response["proposals"] = [item for _index, item, _citation, _grounding in retained]
    citation_rows.extend(citation for _index, _item, citation, _grounding in retained)
    return len(retained), len(raw_items), response["proposals"] != raw_items


def _normalize_fresh_selection_response(
    response: dict[str, Any],
    *,
    request: AllEvidenceFusionRequest,
    evidence_by_id: Mapping[str, tuple[str, ...]],
    evidence_blocks_by_id: Mapping[str, Mapping[str, Any]],
    citation_rows: list[dict[str, Any]],
    grounding_rows: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
) -> tuple[int, int | None, bool]:
    """Retain ranked known IDs that have one valid grounded selection note."""

    raw_selected = response.get("selected_candidate_ids")
    raw_notes = response.get("selection_notes")
    if not isinstance(raw_selected, list) or not isinstance(raw_notes, list):
        return 0, None, False

    candidate_ids = {candidate.candidate_id for candidate in request.candidates}
    contracts_by_id = {
        candidate.candidate_id: candidate.contract for candidate in request.candidates
    }
    allowed_note_fields = {
        "candidate_id",
        "supporting_evidence_ids",
        "supporting_source_families",
        "reason",
    }
    valid_notes: dict[
        str,
        tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]],
    ] = {}
    for index, raw_note in enumerate(raw_notes):
        path = f"selection_notes[{index}]"
        if not isinstance(raw_note, Mapping):
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="not_object",
                    reason=f"{path} must be an object",
                )
            )
            continue
        note = copy.deepcopy(dict(raw_note))
        unexpected = set(note) - allowed_note_fields
        if unexpected:
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="unsupported_fields",
                    reason=f"{path} has unsupported fields: {sorted(map(str, unexpected))}",
                )
            )
            continue
        candidate_id = note.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in candidate_ids:
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="unknown_note_candidate_id",
                    reason=f"{path}.candidate_id must be a known candidate ID",
                )
            )
            continue
        if candidate_id in valid_notes:
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="duplicate_selection_note",
                    reason=f"{path} duplicates an earlier valid note",
                )
            )
            continue
        try:
            grounding_row = _prune_fresh_ungrounded_evidence_ids(
                note,
                contract=contracts_by_id[candidate_id],
                evidence_by_id=evidence_by_id,
                evidence_blocks_by_id=evidence_blocks_by_id,
                path=path,
            )
            citation_row = _derive_fresh_citation_families(
                note,
                evidence_by_id=evidence_by_id,
                path=path,
            )
            # Exercise the same strict row contract used at the standalone
            # boundary after filling only the redundant family field.
            _validate_selection_notes(request, [note], {candidate_id})
        except _NoGroundedEvidenceCitations as exc:
            exc.audit["normalization_disposition"] = "pending_zero_grounding"
            grounding_rows.append(exc.audit)
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="no_lexically_grounded_citations",
                    reason=str(exc),
                )
            )
            continue
        except (TypeError, ValueError) as exc:
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="invalid_citations",
                    reason=str(exc),
                )
            )
            continue
        if grounding_row["changed"]:
            citation_row["changed"] = True
        grounding_row["normalization_disposition"] = "eligible_selection_note"
        grounding_rows.append(grounding_row)
        valid_notes[candidate_id] = (index, note, citation_row, grounding_row)

    eligible: list[tuple[int, str, int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    seen_selected: set[str] = set()
    for index, candidate_id in enumerate(raw_selected):
        path = f"selected_candidate_ids[{index}]"
        if not isinstance(candidate_id, str):
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="invalid_selected_id_type",
                    reason=f"{path} must be a string",
                )
            )
            continue
        if candidate_id not in candidate_ids:
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="unknown_selected_candidate_id",
                    reason=f"{path} is not a known candidate ID",
                )
            )
            continue
        if candidate_id in seen_selected:
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="duplicate_selected_candidate_id",
                    reason=f"{path} duplicates an earlier selected ID",
                )
            )
            continue
        seen_selected.add(candidate_id)
        note_entry = valid_notes.get(candidate_id)
        if note_entry is None:
            dropped.append(
                _normalization_drop(
                    path=path,
                    reason_code="missing_valid_selection_note",
                    reason=f"{path} has no valid grounded selection note",
                )
            )
            continue
        note_index, note, citation_row, grounding_row = note_entry
        eligible.append((index, candidate_id, note_index, note, citation_row, grounding_row))

    # As in proposal mode, retain the strict repair behavior if a non-empty
    # attempted selection has no salvageable selected candidate.
    if raw_selected and not eligible:
        for grounding_row in grounding_rows:
            grounding_row["normalization_disposition"] = "left_for_remote_repair"
        return 0, len(raw_selected), False

    for grounding_row in grounding_rows:
        if grounding_row["normalization_disposition"] == "pending_zero_grounding":
            grounding_row["normalization_disposition"] = "removed_zero_grounding"
    retained = eligible[: request.max_candidates]
    for _i, _candidate_id, _note_index, _note, _citation, grounding_row in retained:
        grounding_row["normalization_disposition"] = "retained"
    retained_ids = {candidate_id for _i, candidate_id, _ni, _n, _c, _g in retained}
    for (
        index,
        _candidate_id,
        note_index,
        _note,
        _citation,
        _grounding,
    ) in eligible[request.max_candidates :]:
        _grounding["normalization_disposition"] = "max_candidates_truncation"
        selected_path = f"selected_candidate_ids[{index}]"
        note_path = f"selection_notes[{note_index}]"
        dropped.extend(
            [
                _normalization_drop(
                    path=selected_path,
                    reason_code="max_candidates_truncation",
                    reason=f"{selected_path} was beyond the stable max_candidates cutoff",
                ),
                _normalization_drop(
                    path=note_path,
                    reason_code="unretained_selection_note",
                    reason=f"{note_path} belongs to a candidate beyond the stable cutoff",
                ),
            ]
        )

    raw_selected_set = {value for value in raw_selected if isinstance(value, str)}
    retained_note_indices = {note_index for _i, _c, note_index, _n, _a, _g in retained}
    for candidate_id, (
        note_index,
        _note,
        _citation,
        _grounding,
    ) in valid_notes.items():
        if note_index in retained_note_indices:
            continue
        # Notes paired with cap-truncated candidates were already audited.
        if candidate_id in raw_selected_set and candidate_id not in retained_ids:
            continue
        _grounding["normalization_disposition"] = "note_for_unselected_candidate"
        path = f"selection_notes[{note_index}]"
        dropped.append(
            _normalization_drop(
                path=path,
                reason_code="note_for_unselected_candidate",
                reason=f"{path} does not belong to a retained selected candidate",
            )
        )

    response["selected_candidate_ids"] = [
        candidate_id for _index, candidate_id, _note_index, _note, _citation, _grounding in retained
    ]
    response["selection_notes"] = [note for _i, _c, _ni, note, _a, _g in retained]
    citation_rows.extend(citation for _i, _c, _ni, _note, citation, _grounding in retained)
    changed = (
        response["selected_candidate_ids"] != raw_selected
        or response["selection_notes"] != raw_notes
    )
    return len(retained), len(raw_selected), changed


def all_evidence_fusion_response_issues(
    response: Any,
    context: Mapping[str, Any],
) -> list[str]:
    """Return complete response issues using only the serialized agent context."""

    try:
        request = _fusion_request_from_context(context)
        validate_all_evidence_fusion_response(request, response)
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def build_all_evidence_fusion_repair_prompt(
    issues: Sequence[str],
    context: Mapping[str, Any],
) -> str:
    """Build a schema-specific repair message without repeating evidence text."""

    request = _fusion_request_from_context(context)
    issue_lines = "\n".join(f"- {str(issue)}" for issue in issues)
    evidence_ids = [block.evidence_id for block in request.evidence_blocks]
    evidence_family_allowlist = {
        block.evidence_id: list(block.source_families) for block in request.evidence_blocks
    }
    evidence_family_allowlist_json = json.dumps(
        evidence_family_allowlist,
        sort_keys=True,
        indent=2,
    )
    if request.mode == "select":
        candidate_ids = [item.candidate_id for item in request.candidates]
        contract = """{
  "selected_candidate_ids": ["candidate_0001"],
  "selection_notes": [{
    "candidate_id": "candidate_0001",
    "supporting_evidence_ids": ["evidence_0001"],
    "supporting_source_families": ["one family present in that evidence block"],
    "reason": "brief evidence-grounded reason"
  }]
}"""
        mode_rules = (
            "Use only these candidate IDs: "
            + json.dumps(candidate_ids)
            + ". Return one selection note for every selected ID. Do not return "
            "or alter extraction specs."
        )
    else:
        contract = """{
  "proposals": [{
    "name": "snake_case_variable",
    "type": "categorical|continuous",
    "categories": ["absent", "present"],
    "value_aliases": {"absent": ["negative"], "present": ["positive"]},
    "roles": ["confounder|effect_modifier"],
    "description": "precise extraction contract for the supported construct",
    "supporting_evidence_ids": ["evidence_0001"],
    "supporting_source_families": ["one family present in that evidence block"],
    "rationale": "brief evidence-grounded reason"
  }]
}"""
        mode_rules = (
            "Propose only variables grounded in the supplied evidence IDs. For a "
            "categorical proposal, replace the example categories with every supported "
            "mutually exclusive canonical value specific to that variable, with at "
            "least two values; never omit values to meet an implicit category-count "
            "cap. Omit categories and value_aliases for continuous proposals. For a "
            "categorical proposal, value_aliases is optional; when supplied, its keys "
            "must exactly match declared categories and every normalized alias must "
            "belong to only one category. Never use schema instructions or placeholder "
            "labels as category values."
        )
    return f"""The previous all-evidence fusion response failed these checks:
{issue_lines}

Return corrected JSON only with this exact top-level shape:
{contract}

At most {request.max_candidates} candidates may be returned. {mode_rules}
Preserve the original request's comprehensive, nonredundant coverage; do not
silently shorten a well-grounded proposal inventory during schema repair.
Use only these evidence IDs: {json.dumps(evidence_ids)}.

Authoritative evidence_id -> allowed source_families mapping:
{evidence_family_allowlist_json}

Citation correction rules (the mapping above is exact and exhaustive):
1. For each proposal or selection note, treat supporting_evidence_ids as the
   complete set of evidence cited by that item.
2. supporting_source_families must be a non-empty subset of the UNION of the
   allowed source-family lists mapped to exactly those cited evidence IDs. The
   evidence-ID and source-family arrays are not positional pairs.
3. If a source family is outside that union, either cite an existing evidence
   ID whose mapped allowlist contains the family and whose evidence genuinely
   supports the reason, or replace/remove the unsupported family so that only
   genuinely supported mapped families remain. Never invent an ID or family.
4. Every cited block must have one sanitized concept entry containing all
   normalized exact lexical identity anchors in that item's contract name; a
   short acronym counts only when that exact token occurs. A valid ID/family
   combination alone is not grounding. Description text, categories, aliases,
   numeric codes, structural name words, inferred synonyms, and acronym
   expansions cannot establish the match.
5. Keep the candidate/proposal inventory unchanged when a valid citation
   correction is possible. If no valid mapped citation genuinely supports an
   item, omit that item instead of fabricating grounding.

Do not add prose, markdown, comments, or code fences.
"""


def _fusion_request_from_context(
    context: Mapping[str, Any],
) -> AllEvidenceFusionRequest:
    if not isinstance(context, Mapping):
        raise TypeError("fusion context must be an object")
    _reject_forbidden_content(context, path="fusion_context")
    if context.get("prompt_version") != FUSION_PROMPT_VERSION:
        raise ValueError("fusion context has an unsupported prompt_version")
    if context.get("source_text_temporal_policy") != source_text_temporal_policy_audit():
        raise ValueError("fusion context has an unsupported source-text temporal policy")
    try:
        outer_fold = int(context.get("outer_fold"))
        maximum = int(context.get("max_candidates"))
    except (TypeError, ValueError) as exc:
        raise ValueError("fusion context fold and cap must be integers") from exc
    if outer_fold < 1:
        raise ValueError("fusion context outer_fold must be positive")
    if not 1 <= maximum <= 64:
        raise ValueError("fusion context max_candidates must be in [1, 64]")
    fingerprint = str(context.get("split_fingerprint") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("fusion context split_fingerprint must be a SHA-256 digest")

    raw_evidence = context.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise ValueError("fusion context evidence must be a non-empty list")
    evidence_blocks: list[FusionEvidenceBlock] = []
    for index, raw_block in enumerate(raw_evidence, start=1):
        if not isinstance(raw_block, Mapping):
            raise ValueError(f"fusion context evidence[{index - 1}] must be an object")
        expected_id = f"evidence_{index:04d}"
        if raw_block.get("evidence_id") != expected_id:
            raise ValueError("fusion context evidence IDs must be opaque and sequential")
        families = raw_block.get("source_families")
        if not isinstance(families, list) or not families:
            raise ValueError(f"{expected_id} must contain source_families")
        normalized_families = tuple(str(value) for value in families)
        if len(normalized_families) != len(set(normalized_families)):
            raise ValueError(f"{expected_id} source_families contain duplicates")
        unknown = set(normalized_families) - _SOURCE_FAMILY_SET
        if unknown:
            raise ValueError(f"{expected_id} has unknown source families: {sorted(unknown)}")
        role = str(raw_block.get("role_hint") or "")
        if role not in _VALID_ROLES:
            raise ValueError(f"{expected_id} has an invalid role_hint")
        content = raw_block.get("content")
        if not isinstance(content, Mapping):
            raise ValueError(f"{expected_id}.content must be an object")
        evidence_blocks.append(
            FusionEvidenceBlock(
                evidence_id=expected_id,
                source_families=normalized_families,
                role_hint=role,
                _content_json=_canonical_json(dict(content)),
            )
        )

    raw_candidates = context.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("fusion context candidates must be a list")
    if len(raw_candidates) > _MAX_CANDIDATE_POOL:
        raise ValueError(f"fusion context candidate pool exceeds {_MAX_CANDIDATE_POOL}")
    prompt_candidates: list[FusionPromptCandidate] = []
    present_families = {family for block in evidence_blocks for family in block.source_families}
    for index, raw_candidate in enumerate(raw_candidates, start=1):
        if not isinstance(raw_candidate, Mapping):
            raise ValueError(f"fusion context candidates[{index - 1}] must be an object")
        expected_id = f"candidate_{index:04d}"
        if raw_candidate.get("candidate_id") != expected_id:
            raise ValueError("fusion context candidate IDs must be opaque and sequential")
        families = raw_candidate.get("source_families", [])
        if not isinstance(families, list):
            raise ValueError(f"{expected_id}.source_families must be a list")
        contract = CandidateContract(
            raw_candidate.get("extraction_spec"),
            source_families=[str(value) for value in families],
        )
        unavailable = set(contract.source_families) - present_families
        if unavailable:
            raise ValueError(
                f"{expected_id} cites unavailable source families: {sorted(unavailable)}"
            )
        prompt_candidates.append(FusionPromptCandidate(candidate_id=expected_id, contract=contract))

    mode = str(context.get("mode") or "")
    if mode == "select" and not prompt_candidates:
        raise ValueError("select-mode fusion context requires candidates")
    if mode == "propose" and prompt_candidates:
        raise ValueError("propose-mode fusion context cannot contain candidates")
    if mode not in {"select", "propose"}:
        raise ValueError("fusion context mode must be select or propose")
    supplied_coverage = context.get("source_family_coverage")
    coverage = _source_family_coverage(evidence_blocks, prompt_candidates)
    if supplied_coverage is not None:
        if not isinstance(supplied_coverage, Mapping):
            raise ValueError("fusion context source_family_coverage must be an object")
        for key, expected_value in coverage.items():
            if supplied_coverage.get(key) != expected_value:
                raise ValueError(
                    "fusion context source_family_coverage does not match reconstructed evidence"
                )
        legacy_compaction = supplied_coverage.get("legacy_compaction")
        if legacy_compaction is not None:
            coverage = _source_family_coverage(
                evidence_blocks,
                prompt_candidates,
                legacy_compaction_audit=_validate_legacy_compaction_audit(legacy_compaction),
            )
        unexpected_coverage = set(supplied_coverage) - set(coverage)
        if unexpected_coverage:
            raise ValueError(
                "fusion context source_family_coverage has unsupported fields: "
                f"{sorted(map(str, unexpected_coverage))}"
            )
    return AllEvidenceFusionRequest(
        outer_fold=outer_fold,
        split_fingerprint=fingerprint,
        mode=mode,
        max_candidates=maximum,
        evidence_blocks=tuple(evidence_blocks),
        candidates=tuple(prompt_candidates),
        _coverage_json=_canonical_json(coverage),
    )


def _validate_selection_response(
    request: AllEvidenceFusionRequest,
    response: Mapping[str, Any],
) -> FusionResult:
    allowed_top = {"selected_candidate_ids", "selection_notes"}
    unexpected = set(response) - allowed_top
    if unexpected:
        raise ValueError(
            "selection response contains unsupported fields: " f"{sorted(map(str, unexpected))}"
        )
    selected = response.get("selected_candidate_ids")
    if not isinstance(selected, list) or not all(isinstance(value, str) for value in selected):
        raise ValueError("selected_candidate_ids must be a list of strings")
    if len(selected) > request.max_candidates:
        raise ValueError("selection exceeds max_candidates")
    if len(selected) != len(set(selected)):
        raise ValueError("selected_candidate_ids cannot contain duplicates")
    by_id = {item.candidate_id: item for item in request.candidates}
    unknown = set(selected) - set(by_id)
    if unknown:
        raise ValueError(f"selection contains unknown candidate IDs: {sorted(unknown)}")

    notes = response.get("selection_notes", [])
    note_audit = _validate_selection_notes(request, notes, set(selected))
    contracts = tuple(by_id[candidate_id].contract for candidate_id in selected)
    family_counts = {family: 0 for family in ALL_SOURCE_FAMILIES}
    for contract in contracts:
        for family in contract.source_families:
            family_counts[family] += 1
    citation_family_counts = {family: 0 for family in ALL_SOURCE_FAMILIES}
    for note in note_audit:
        for family in note["supporting_source_families"]:
            citation_family_counts[family] += 1
    return FusionResult(
        mode="select",
        selected_candidate_ids=tuple(selected),
        selected_contracts=contracts,
        response_audit={
            "selected_count": len(selected),
            "max_candidates": request.max_candidates,
            "evidence_contract_grounding_version": EVIDENCE_CONTRACT_GROUNDING_VERSION,
            "selected_candidate_count_by_source_family": family_counts,
            "selection_citation_count_by_source_family": citation_family_counts,
            "selection_notes": note_audit,
        },
    )


def _validate_selection_notes(
    request: AllEvidenceFusionRequest,
    notes: Any,
    selected: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(notes, list):
        raise ValueError("selection_notes must be a list")
    evidence_by_id = {
        block.evidence_id: set(block.source_families) for block in request.evidence_blocks
    }
    evidence_blocks_by_id = {
        block.evidence_id: block.as_prompt_dict() for block in request.evidence_blocks
    }
    candidates_by_id = {item.candidate_id: item.contract for item in request.candidates}
    audited: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, note in enumerate(notes):
        if not isinstance(note, Mapping):
            raise ValueError(f"selection_notes[{index}] must be an object")
        allowed = {
            "candidate_id",
            "supporting_evidence_ids",
            "supporting_source_families",
            "reason",
        }
        unexpected = set(note) - allowed
        if unexpected:
            raise ValueError(
                f"selection_notes[{index}] has unsupported fields: {sorted(unexpected)}"
            )
        candidate_id = str(note.get("candidate_id") or "")
        if candidate_id not in selected:
            raise ValueError("selection note cites a candidate that was not selected")
        if candidate_id in seen:
            raise ValueError("selection_notes can contain at most one note per candidate")
        seen.add(candidate_id)
        evidence_ids, families = _validate_evidence_citations(
            note,
            evidence_by_id,
            path=f"selection_notes[{index}]",
        )
        grounding_rows = []
        unrelated = []
        for evidence_id in evidence_ids:
            grounding = ground_evidence_to_extraction_contract(
                evidence_blocks_by_id[evidence_id],
                candidates_by_id[candidate_id],
            )
            grounding_rows.append({"evidence_id": evidence_id, **grounding.as_dict()})
            if not grounding.supported:
                unrelated.append(evidence_id)
        if unrelated:
            raise ValueError(
                f"selection_notes[{index}] cites evidence unrelated to the selected "
                f"contract: {unrelated}"
            )
        audited.append(
            {
                "candidate_id": candidate_id,
                "supporting_evidence_ids": evidence_ids,
                "supporting_source_families": families,
                "evidence_contract_grounding": grounding_rows,
                "reason": _normalize_evidence_text(note.get("reason")),
            }
        )
    missing_notes = selected - seen
    if missing_notes:
        raise ValueError(
            "selection_notes must ground every selected candidate; missing notes for "
            f"{sorted(missing_notes)}"
        )
    return audited


def _validate_proposal_response(
    request: AllEvidenceFusionRequest,
    response: Mapping[str, Any],
) -> FusionResult:
    if set(response) != {"proposals"}:
        raise ValueError("proposal response must contain only the 'proposals' field")
    proposals = response.get("proposals")
    if not isinstance(proposals, list):
        raise ValueError("proposals must be a list")
    if len(proposals) > request.max_candidates:
        raise ValueError("proposal response exceeds max_candidates")
    evidence_by_id = {
        block.evidence_id: set(block.source_families) for block in request.evidence_blocks
    }
    evidence_blocks_by_id = {
        block.evidence_id: block.as_prompt_dict() for block in request.evidence_blocks
    }
    specs: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    names: set[str] = set()
    allowed = {
        "name",
        "type",
        "categories",
        "value_aliases",
        "roles",
        "description",
        "supporting_evidence_ids",
        "supporting_source_families",
        "rationale",
    }
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, Mapping):
            raise ValueError(f"proposals[{index}] must be an object")
        unexpected = set(proposal) - allowed
        if unexpected:
            raise ValueError(f"proposals[{index}] has unsupported fields: {sorted(unexpected)}")
        spec = {
            key: proposal.get(key)
            for key in (
                "name",
                "type",
                "categories",
                "value_aliases",
                "roles",
                "description",
            )
            if proposal.get(key) is not None
        }
        _validate_extraction_spec(spec, source=f"proposals[{index}]")
        name = str(spec["name"])
        if name in names:
            raise ValueError("proposal names must be unique")
        names.add(name)
        evidence_ids, families = _validate_evidence_citations(
            proposal,
            evidence_by_id,
            path=f"proposals[{index}]",
        )
        grounding_rows = []
        unrelated = []
        for evidence_id in evidence_ids:
            grounding = ground_evidence_to_extraction_contract(
                evidence_blocks_by_id[evidence_id],
                spec,
            )
            grounding_rows.append({"evidence_id": evidence_id, **grounding.as_dict()})
            if not grounding.supported:
                unrelated.append(evidence_id)
        if unrelated:
            raise ValueError(
                f"proposals[{index}] cites evidence unrelated to the proposed contract: "
                f"{unrelated}"
            )
        specs.append(json.loads(_canonical_json(spec)))
        audit_rows.append(
            {
                "name": name,
                "supporting_evidence_ids": evidence_ids,
                "supporting_source_families": families,
                "evidence_contract_grounding": grounding_rows,
                "rationale": _normalize_evidence_text(proposal.get("rationale")),
            }
        )
    family_counts = {family: 0 for family in ALL_SOURCE_FAMILIES}
    for row in audit_rows:
        for family in row["supporting_source_families"]:
            family_counts[family] += 1
    return FusionResult(
        mode="propose",
        proposed_specs=tuple(specs),
        response_audit={
            "proposed_count": len(specs),
            "max_candidates": request.max_candidates,
            "evidence_contract_grounding_version": EVIDENCE_CONTRACT_GROUNDING_VERSION,
            "proposal_count_by_source_family": family_counts,
            "proposal_grounding": audit_rows,
        },
    )


def _validate_evidence_citations(
    payload: Mapping[str, Any],
    evidence_by_id: Mapping[str, set[str]],
    *,
    path: str,
) -> tuple[list[str], list[str]]:
    evidence_ids = payload.get("supporting_evidence_ids")
    families = payload.get("supporting_source_families")
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or not all(isinstance(value, str) for value in evidence_ids)
    ):
        raise ValueError(f"{path}.supporting_evidence_ids must be a non-empty string list")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError(f"{path}.supporting_evidence_ids contains duplicates")
    unknown_evidence = set(evidence_ids) - set(evidence_by_id)
    if unknown_evidence:
        raise ValueError(f"{path} cites unknown evidence IDs: {sorted(unknown_evidence)}")
    if (
        not isinstance(families, list)
        or not families
        or not all(isinstance(value, str) for value in families)
    ):
        raise ValueError(f"{path}.supporting_source_families must be a non-empty string list")
    if len(families) != len(set(families)):
        raise ValueError(f"{path}.supporting_source_families contains duplicates")
    unknown_families = set(families) - _SOURCE_FAMILY_SET
    if unknown_families:
        raise ValueError(f"{path} cites unknown source families: {sorted(unknown_families)}")
    cited_available = set().union(*(evidence_by_id[value] for value in evidence_ids))
    unsupported = set(families) - cited_available
    if unsupported:
        raise ValueError(
            f"{path} cites source families not present in cited evidence: " f"{sorted(unsupported)}"
        )
    return list(evidence_ids), list(families)


def _validate_extraction_spec(spec: Mapping[str, Any], *, source: str) -> None:
    allowed = {"name", "type", "categories", "roles", "description", "value_aliases"}
    unexpected = set(spec) - allowed
    if unexpected:
        raise ValueError(f"{source} contains unsupported spec fields: {sorted(unexpected)}")
    name = str(spec.get("name") or "")
    if not _SNAKE_CASE_NAME.fullmatch(name):
        raise ValueError(f"{source}.name must be a non-empty snake_case name")
    description = str(spec.get("description") or "").strip()
    if _FORBIDDEN_IDENTIFIER_NAME.search(name) or _FORBIDDEN_IDENTIFIER_DESCRIPTION.search(
        description
    ):
        raise ValueError(f"{source} describes an identifier rather than a patient variable")
    feature_type = str(spec.get("type") or "")
    if feature_type not in _VALID_TYPES:
        raise ValueError(f"{source}.type must be categorical or continuous")
    roles = spec.get("roles")
    if (
        not isinstance(roles, list)
        or not roles
        or not all(isinstance(value, str) and value in _VALID_ROLES for value in roles)
    ):
        raise ValueError(f"{source}.roles must contain valid causal roles")
    if len(roles) != len(set(roles)):
        raise ValueError(f"{source}.roles cannot contain duplicates")
    categories = spec.get("categories")
    value_aliases = spec.get("value_aliases")
    if feature_type == "categorical":
        if not isinstance(categories, list) or len(categories) < 2:
            raise ValueError(f"{source}.categories requires at least two values")
        if not all(str(value).strip() for value in categories):
            raise ValueError(f"{source}.categories cannot contain empty values")
        category_text = [str(value).strip() for value in categories]
        normalized_categories = [
            re.sub(r"[\s_-]+", " ", value).strip().casefold() for value in category_text
        ]
        if len(normalized_categories) != len(set(normalized_categories)):
            raise ValueError(
                f"{source}.categories must be distinct after case/spacing normalization"
            )
        placeholder_values = [
            str(value).strip()
            for value in categories
            if _is_instructional_or_placeholder_category(value)
        ]
        if placeholder_values:
            raise ValueError(
                f"{source}.categories contains instructional or placeholder "
                f"values: {placeholder_values}"
            )
        if value_aliases not in (None, {}):
            if not isinstance(value_aliases, Mapping):
                raise ValueError(f"{source}.value_aliases must be a category-to-alias-list map")
            unknown_keys = set(map(str, value_aliases)) - set(category_text)
            if unknown_keys:
                raise ValueError(
                    f"{source}.value_aliases keys must exactly match declared categories: "
                    f"{sorted(unknown_keys)}"
                )
            normalized_owner = {
                normalized: category
                for normalized, category in zip(normalized_categories, category_text)
            }
            for raw_category, raw_aliases in value_aliases.items():
                category = str(raw_category)
                if not isinstance(raw_aliases, list) or not raw_aliases:
                    raise ValueError(
                        f"{source}.value_aliases[{category!r}] must be a non-empty string list"
                    )
                if not all(isinstance(alias, str) and alias.strip() for alias in raw_aliases):
                    raise ValueError(
                        f"{source}.value_aliases[{category!r}] cannot contain empty/non-string aliases"
                    )
                for alias in raw_aliases:
                    normalized = re.sub(r"[\s_-]+", " ", alias).strip().casefold()
                    prior_owner = normalized_owner.get(normalized)
                    if prior_owner is not None:
                        raise ValueError(
                            f"{source}.value_aliases contains a normalized collision between "
                            f"{prior_owner!r} and {category!r}"
                        )
                    normalized_owner[normalized] = category
    elif categories not in (None, []):
        raise ValueError(f"{source}.categories must be absent for continuous features")
    elif value_aliases not in (None, {}):
        raise ValueError(f"{source}.value_aliases must be absent for continuous features")
    if not description:
        raise ValueError(f"{source}.description must be non-empty")


def _is_instructional_or_placeholder_category(value: Any) -> bool:
    """Reject schema prose while retaining ordinary domain category labels."""

    text = str(value).strip()
    normalized = re.sub(r"[\s_-]+", " ", text).strip().casefold()
    if not normalized:
        return False
    if _CATEGORY_META_LANGUAGE.search(normalized):
        return True
    return bool(_CATEGORY_PLACEHOLDER_TOKEN.fullmatch(normalized))


def _validate_payload_provenance(
    payload: Mapping[str, Any],
    provenance: FoldEvidenceProvenance,
) -> None:
    train = set(provenance.train_row_ids)
    heldout = set(provenance.heldout_row_ids)
    payload_scope = str(payload.get("scope") or "").strip().lower()
    if payload_scope:
        normalized_scope = {
            "outer_train": "outer_train",
            "full_outer_train": "outer_train",
            "inner_train": "inner_train",
            "candidate_selection_inner_fit": "inner_train",
            "candidate_consistency_inner_train": "inner_train",
        }.get(payload_scope)
        if normalized_scope is not None and normalized_scope != provenance.scope:
            raise ValueError("payload scope does not match fold provenance scope")
    payload_inner_fold = payload.get("inner_fold")
    if payload_inner_fold is not None:
        if provenance.inner_fold is None or int(payload_inner_fold) != provenance.inner_fold:
            raise ValueError("payload inner_fold does not match fold provenance")

    def visit(value: Any, path: str, key_hint: str = "") -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                child_path = f"{path}.{key}"
                lowered = key.lower()
                if lowered == "outer_fold":
                    try:
                        fold_value = int(child)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{child_path} must be an integer") from exc
                    if fold_value != provenance.outer_fold:
                        raise ValueError(f"{child_path} does not match provenance outer_fold")
                elif _ROW_ID_KEY.search(lowered):
                    row_values = child if isinstance(child, (list, tuple, set)) else [child]
                    for row_value in row_values:
                        normalized = _normalize_row_id(
                            row_value,
                            field_name=child_path,
                        )
                        if "heldout" in lowered:
                            if normalized not in heldout:
                                raise ValueError(
                                    f"{child_path} contains a row outside heldout provenance"
                                )
                        elif normalized not in train or normalized in heldout:
                            raise ValueError(
                                f"heldout or unknown row {normalized!r} entered {child_path}"
                            )
                visit(child, child_path, lowered)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]", key_hint)

    visit(payload, "payload")


def _reject_forbidden_content(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if _FORBIDDEN_KEY.search(key):
                raise ValueError(f"forbidden oracle/true field at {path}.{key}")
            _reject_forbidden_content(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple, set)):
        for index, child in enumerate(value):
            _reject_forbidden_content(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and _FORBIDDEN_STRING.search(value):
        raise ValueError(f"forbidden oracle/true string at {path}")


def _legacy_axis_token(value: Any, *, default: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return token or default


def _bow_model_axis(group: Mapping[str, Any]) -> str:
    explicit = group.get("bow_model") or group.get("model_family")
    if explicit:
        return _legacy_axis_token(explicit, default="unspecified_bow_model")
    descriptor = " ".join(
        str(group.get(key) or "").lower()
        for key in ("source", "view_name", "evidence_type", "meaning")
    )
    if "matched_pair_uplift" in descriptor or "pair uplift" in descriptor:
        return "matched_pair_uplift"
    if "ensemble_r" in descriptor or "ensemble r" in descriptor:
        return "ensemble_r"
    for model in ("random_forest", "extratrees", "xgboost", "linear"):
        if model in descriptor or model.replace("_", " ") in descriptor:
            return model
    return "unspecified_bow_model"


def _objective_and_sign(value: Any, *, default: str) -> tuple[str, str]:
    token = _legacy_axis_token(value, default=default)
    if token.endswith("_positive"):
        return token[: -len("_positive")], "positive"
    if token.endswith("_negative"):
        return token[: -len("_negative")], "negative"
    if "positive" in token and "negative" in token:
        return token, "both"
    return token, "unsigned"


def _legacy_group_dimensions(
    *,
    kind: str,
    families: Sequence[str],
    raw: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    if kind == "bow":
        objective, sign = _objective_and_sign(
            raw.get("evidence_type") or raw.get("meaning"),
            default="unspecified_bow_objective",
        )
        return {
            "source_family": tuple(sorted(set(families))),
            "bow_view": (
                _legacy_axis_token(
                    raw.get("view_name") or raw.get("source"),
                    default="unspecified_bow_view",
                ),
            ),
            "bow_model": (_bow_model_axis(raw),),
            "objective": (objective,),
            "sign": (sign,),
        }
    if kind == "embedding":
        objective, sign = _objective_and_sign(
            raw.get("contrast_family") or raw.get("direction_source") or raw.get("name"),
            default="unspecified_embedding_objective",
        )
        # A compact embedding contrast normally carries both aligned tails.
        if any(
            raw.get(key) for key in ("positive_aligned_chunks", "positive_external_chunks")
        ) and any(raw.get(key) for key in ("negative_aligned_chunks", "negative_external_chunks")):
            sign = "both"
        return {
            "source_family": tuple(sorted(set(families))),
            "bow_view": ("not_applicable",),
            "bow_model": ("not_applicable",),
            "objective": (objective,),
            "sign": (sign,),
        }
    stage = _legacy_axis_token(raw.get("stage"), default="unspecified_htr_stage")
    row_models = {
        _legacy_axis_token(row.get("model_family"), default="unspecified_htr_model")
        for row in raw.get("rows") or []
        if isinstance(row, Mapping)
    }
    row_objectives = {
        _legacy_axis_token(row.get("effect_objective"), default=stage)
        for row in raw.get("rows") or []
        if isinstance(row, Mapping)
    }
    return {
        "source_family": tuple(sorted(set(families))),
        "bow_view": ("not_applicable",),
        "bow_model": tuple(sorted(row_models)) or ("unspecified_htr_model",),
        "objective": tuple(sorted(row_objectives)) or (stage,),
        "sign": ("unsigned",),
    }


def _canonicalize_legacy_groups(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return every legacy group in content-defined order.

    Older implementations selected a fixed number of groups.  Even with
    multi-axis interleaving, that silently omitted authenticated evidence when
    a dataset produced more groups than the built-in allowance.  The legacy
    adapter now preserves every group.  Prompt/page capacity is an orchestration
    concern and must be handled by an explicit complete-paging protocol, never
    by slicing this scientific evidence bank.
    """

    return sorted(records, key=lambda row: _canonical_json(row["canonical"]))


def _legacy_compaction_accounting(
    discovered: Sequence[Mapping[str, Any]],
    retained: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def family_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        counts = {family: 0 for family in ALL_SOURCE_FAMILIES}
        for row in rows:
            for family in row["families"]:
                counts[family] += 1
        return counts

    def kind_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            kind: sum(row["kind"] == kind for row in rows) for kind in ("bow", "embedding", "htr")
        }

    def unique_axis_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        values: dict[str, set[str]] = {
            dimension: set()
            for dimension in ("source_family", "bow_view", "bow_model", "objective", "sign")
        }
        for row in rows:
            for dimension, row_values in row["dimensions"].items():
                values[dimension].update(map(str, row_values))
        return {dimension: len(axis_values) for dimension, axis_values in values.items()}

    return {
        "schema_version": LEGACY_COMPACTION_STRATEGY_VERSION,
        "selection_strategy": "complete_canonical_order_no_omission",
        "maximum_groups_per_role_and_kind": None,
        "discovered_group_count": len(discovered),
        "retained_group_count": len(retained),
        "dropped_group_count": len(discovered) - len(retained),
        "discovered_group_count_by_source_family": family_counts(discovered),
        "retained_group_count_by_source_family": family_counts(retained),
        "discovered_group_count_by_kind": kind_counts(discovered),
        "retained_group_count_by_kind": kind_counts(retained),
        "discovered_unique_value_count_by_axis": unique_axis_counts(discovered),
        "retained_unique_value_count_by_axis": unique_axis_counts(retained),
    }


def _compact_exact_inner_recurrence(
    value: Any,
) -> list[tuple[tuple[str, ...], str, dict[str, Any]]]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        raise ValueError("exact-inner recurrence evidence must be an object")
    expected_top_fields = {
        "schema_version",
        "normalization",
        "inner_fold_count",
        "latent_topic_ids_compared_across_folds",
        "minimum_inner_fold_support",
        "groups",
    }
    if set(value) != expected_top_fields:
        raise ValueError("exact-inner recurrence evidence must use the closed schema")
    if value.get("schema_version") != EXACT_INNER_RECURRENCE_VERSION:
        raise ValueError("exact-inner recurrence evidence has an unsupported schema")
    if value.get("normalization") != "unicode_nfkc_casefold_nonword_to_space_exact_match":
        raise ValueError("exact-inner recurrence evidence has an unsupported normalization")
    if value.get("latent_topic_ids_compared_across_folds") is not False:
        raise ValueError("exact-inner recurrence must not align latent topic IDs")
    if value.get("minimum_inner_fold_support") != 2:
        raise ValueError("exact-inner recurrence minimum support must equal two folds")
    try:
        inner_fold_count = int(value.get("inner_fold_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError("exact-inner recurrence inner_fold_count must be an integer") from exc
    if inner_fold_count < 2:
        raise ValueError("exact-inner recurrence requires at least two inner folds")
    groups = value.get("groups")
    if not isinstance(groups, list):
        raise ValueError("exact-inner recurrence groups must be a list")
    output: list[tuple[tuple[str, ...], str, dict[str, Any]]] = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"exact-inner recurrence groups[{group_index}] must be an object")
        if set(group) != {
            "source_family",
            "role",
            "discovered_recurrent_term_count",
            "retained_term_count",
            "terms",
        }:
            raise ValueError("exact-inner recurrence group must use the closed schema")
        family = str(group.get("source_family") or "")
        role = str(group.get("role") or "")
        if family not in _SOURCE_FAMILY_SET or role not in _VALID_ROLES:
            raise ValueError("exact-inner recurrence group has an invalid family or role")
        raw_terms = group.get("terms")
        if not isinstance(raw_terms, list):
            raise ValueError("exact-inner recurrence group terms must be a list")
        try:
            discovered_count = int(group.get("discovered_recurrent_term_count"))
            retained_count = int(group.get("retained_term_count"))
        except (TypeError, ValueError) as exc:
            raise ValueError("exact-inner recurrence group counts must be integers") from exc
        if (
            discovered_count < retained_count
            or retained_count != len(raw_terms)
        ):
            raise ValueError("exact-inner recurrence group accounting is inconsistent")
        compact_terms: list[dict[str, Any]] = []
        seen_terms: set[str] = set()
        for term_index, raw_term in enumerate(raw_terms):
            if not isinstance(raw_term, Mapping):
                raise ValueError(f"exact-inner recurrence terms[{term_index}] must be an object")
            term = _normalize_evidence_text(raw_term.get("term"))
            if not term or term in seen_terms:
                continue
            seen_terms.add(term)
            try:
                support_count = int(raw_term.get("inner_fold_support_count"))
                occurrence_count = int(raw_term.get("occurrence_count"))
            except (TypeError, ValueError) as exc:
                raise ValueError("exact-inner recurrence support counts must be integers") from exc
            if not 2 <= support_count <= inner_fold_count or occurrence_count < support_count:
                raise ValueError("exact-inner recurrence contains inconsistent support counts")
            compact_terms.append(
                {
                    "term": term,
                    "inner_fold_support_count": support_count,
                    "inner_fold_support_fraction": round(support_count / inner_fold_count, 8),
                    "occurrence_count": occurrence_count,
                }
            )
        if compact_terms:
            output.append(
                (
                    (family,),
                    role,
                    {
                        "kind": "exact_inner_normalized_term_recurrence",
                        "normalization_version": EXACT_INNER_RECURRENCE_VERSION,
                        "inner_fold_count": inner_fold_count,
                        "discovered_recurrent_term_count": discovered_count,
                        "retained_term_count": len(compact_terms),
                        "terms": compact_terms,
                    },
                )
            )
    return output


def _compact_legacy_all_source(
    payload: Mapping[str, Any],
) -> tuple[
    list[tuple[tuple[str, ...], str, dict[str, Any]]],
    dict[str, Any],
]:
    context = payload.get("context") if isinstance(payload.get("context"), Mapping) else payload
    digest = context.get("evidence_digest") if isinstance(context, Mapping) else None
    if not isinstance(digest, Mapping):
        raise ValueError("legacy_all_source payload is missing context.evidence_digest")
    discovered_records: list[dict[str, Any]] = []
    retained_records: list[Mapping[str, Any]] = []
    for role_key, role in (
        ("confounders", "confounder"),
        ("effect_modifiers", "effect_modifier"),
    ):
        section = digest.get(role_key)
        if not isinstance(section, Mapping):
            continue
        bow_records: list[dict[str, Any]] = []
        for group in list(section.get("bow_blurbs") or []):
            if not isinstance(group, Mapping):
                continue
            family = _bow_group_family(group, role)
            rows = _compact_bow_rows(group.get("rows"))
            if rows:
                content = {
                    "kind": "sparse_text_terms",
                    "signal": _normalize_evidence_text(
                        group.get("meaning") or group.get("evidence_type")
                    ),
                    "terms": rows,
                }
                bow_records.append(
                    {
                        "kind": "bow",
                        "families": (family,),
                        "role": role,
                        "content": content,
                        "dimensions": _legacy_group_dimensions(
                            kind="bow",
                            families=(family,),
                            raw=group,
                        ),
                        "canonical": {
                            "families": (family,),
                            "role": role,
                            "content": content,
                        },
                    }
                )
        discovered_records.extend(bow_records)
        selected_bow = _canonicalize_legacy_groups(bow_records)
        retained_records.extend(selected_bow)

        embedding_records: list[dict[str, Any]] = []
        for contrast in list(section.get("embedding_chunks") or []):
            if not isinstance(contrast, Mapping):
                continue
            family = _embedding_family(contrast)
            compact = _compact_embedding_contrast(contrast)
            if compact.get("chunks") or compact.get("concept_scores"):
                embedding_records.append(
                    {
                        "kind": "embedding",
                        "families": (family,),
                        "role": role,
                        "content": compact,
                        "dimensions": _legacy_group_dimensions(
                            kind="embedding",
                            families=(family,),
                            raw=contrast,
                        ),
                        "canonical": {
                            "families": (family,),
                            "role": role,
                            "content": compact,
                        },
                    }
                )
        discovered_records.extend(embedding_records)
        selected_embedding = _canonicalize_legacy_groups(embedding_records)
        retained_records.extend(selected_embedding)

        htr_records: list[dict[str, Any]] = []
        for group in list(section.get("htr_blurbs") or []):
            if not isinstance(group, Mapping):
                continue
            summaries = _compact_htr_rows(group.get("rows"))
            if not summaries:
                continue
            stage = str(group.get("stage") or "").lower()
            families = (
                (HTR_NEURAL, MATCHED_PAIR_UPLIFT)
                if "pair" in stage or "uplift" in stage
                else (HTR_NEURAL,)
            )
            content = {
                "kind": "neural_attention_summaries",
                "stage": _normalize_evidence_text(stage),
                "summaries": summaries,
            }
            htr_records.append(
                {
                    "kind": "htr",
                    "families": families,
                    "role": role,
                    "content": content,
                    "dimensions": _legacy_group_dimensions(
                        kind="htr",
                        families=families,
                        raw=group,
                    ),
                    "canonical": {
                        "families": families,
                        "role": role,
                        "content": content,
                    },
                }
            )
        discovered_records.extend(htr_records)
        selected_htr = _canonicalize_legacy_groups(htr_records)
        retained_records.extend(selected_htr)

    output = [
        (tuple(row["families"]), str(row["role"]), dict(row["content"])) for row in retained_records
    ]
    recurrence = context.get("exact_inner_recurrence") if isinstance(context, Mapping) else None
    output.extend(_compact_exact_inner_recurrence(recurrence))
    return output, _legacy_compaction_accounting(discovered_records, retained_records)


def _bow_group_family(group: Mapping[str, Any], role: str) -> str:
    if role == "confounder":
        return BOW_NUISANCE
    descriptor = " ".join(
        str(group.get(key) or "").lower()
        for key in ("source", "evidence_type", "meaning", "view_name")
    )
    if "pair" in descriptor or "uplift" in descriptor:
        return MATCHED_PAIR_UPLIFT
    return BOW_R_LOSS


def _compact_bow_rows(value: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if not isinstance(value, (list, tuple)):
        return output
    for row in value:
        if not isinstance(row, Mapping):
            continue
        feature = _normalize_evidence_text(
            row.get("feature") or row.get("term") or row.get("phrase"),
        )
        if not feature:
            continue
        item: dict[str, Any] = {"term": feature}
        for key in (
            "coefficient",
            "importance",
            "score",
            "signed_score",
            "frequency",
            "source_count",
            "rank",
        ):
            scalar = _finite_scalar(row.get(key))
            if scalar is not None:
                item[key] = scalar
        output.append(item)
    return output


def _embedding_family(contrast: Mapping[str, Any]) -> str:
    descriptor = " ".join(
        str(contrast.get(key) or "").lower()
        for key in (
            "name",
            "contrast_family",
            "direction_source",
            "role_hint",
            "cluster_component_index",
        )
    )
    return EMBEDDING_CLUSTERED if "cluster" in descriptor else EMBEDDING_WHOLE_COHORT


def _compact_embedding_contrast(contrast: Mapping[str, Any]) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    for side in (
        "positive_aligned_chunks",
        "negative_aligned_chunks",
        "positive_external_chunks",
        "negative_external_chunks",
    ):
        rows = contrast.get(side)
        if not isinstance(rows, (list, tuple)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            text = _normalize_evidence_text(row.get("text") or row.get("chunk_text"))
            if text:
                chunks.append({"side": side, "text": text})
    concept_scores: list[dict[str, Any]] = []
    raw_scores = contrast.get("concept_probe_scores")
    if isinstance(raw_scores, (list, tuple)):
        for score in raw_scores:
            if not isinstance(score, Mapping):
                continue
            label = _normalize_evidence_text(
                score.get("concept") or score.get("phrase") or score.get("label"),
            )
            if not label:
                continue
            item: dict[str, Any] = {"concept": label}
            numeric = _finite_scalar(
                score.get("score") or score.get("similarity") or score.get("cosine")
            )
            if numeric is not None:
                item["score"] = numeric
            concept_scores.append(item)
    return {
        "kind": "embedding_contrast",
        "contrast_label": _normalize_evidence_text(
            contrast.get("name") or contrast.get("contrast_family"),
        ),
        "chunks": chunks,
        "concept_scores": concept_scores,
    }


def _compact_htr_rows(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    output: list[str] = []
    for row in value:
        if not isinstance(row, Mapping):
            continue
        text_parts: list[str] = []
        summary = _normalize_evidence_text(
            row.get("attended_token_summary") or row.get("evidence_snippet") or row.get("feature"),
        )
        if summary:
            text_parts.append(summary)
        spans = row.get("top_token_spans")
        if isinstance(spans, (list, tuple)):
            for span in spans:
                if isinstance(span, Mapping):
                    token = _normalize_evidence_text(
                        span.get("text") or span.get("token") or span.get("span"),
                    )
                    if token:
                        text_parts.append(token)
        joined = _normalize_evidence_text("; ".join(dict.fromkeys(text_parts)))
        if joined:
            output.append(joined)
    return output


def _compact_tfidf_evidence(
    payload: Mapping[str, Any],
) -> list[tuple[tuple[str, ...], str, dict[str, Any]]]:
    discovery = (
        payload.get("discovery") if isinstance(payload.get("discovery"), Mapping) else payload
    )
    banks = discovery.get("topic_banks") if isinstance(discovery, Mapping) else None
    if not isinstance(banks, Mapping):
        raise ValueError("tfidf_topics payload is missing discovery.topic_banks")
    output: list[tuple[tuple[str, ...], str, dict[str, Any]]] = []
    for bank in ("treatment", "outcome", "effect"):
        bank_payload = banks.get(bank)
        if not isinstance(bank_payload, Mapping):
            continue
        role = "effect_modifier" if bank == "effect" else "confounder"
        topics = bank_payload.get("topics")
        if not isinstance(topics, (list, tuple)):
            continue
        for topic in topics:
            if not isinstance(topic, Mapping):
                continue
            terms = _compact_topic_terms(topic.get("terms"))
            if not terms:
                continue
            output.append(
                (
                    (TFIDF_TOPICS,),
                    role,
                    {
                        "kind": "tfidf_topic",
                        "bank": bank,
                        "topic_id": _normalize_evidence_text(topic.get("topic_id")),
                        "terms": terms,
                    },
                )
            )

    orphan = _find_orphan_branch(discovery)
    if isinstance(orphan, Mapping):
        clusters = orphan.get("selected_clusters") or orphan.get("clusters") or []
        selected_ids = {str(value) for value in orphan.get("selected_cluster_ids") or []}
        if isinstance(clusters, (list, tuple)):
            for cluster in clusters:
                if not isinstance(cluster, Mapping):
                    continue
                cluster_id = str(cluster.get("cluster_id") or cluster.get("topic_id") or "")
                if selected_ids and cluster_id not in selected_ids:
                    continue
                terms = _compact_topic_terms(
                    cluster.get("terms")
                    or cluster.get("member_terms")
                    or cluster.get("supporting_terms")
                )
                if not terms:
                    continue
                output.append(
                    (
                        (TFIDF_ORPHAN_NGRAMS,),
                        "effect_modifier",
                        {
                            "kind": "tfidf_orphan_ngram_cluster",
                            "cluster_id": _normalize_evidence_text(cluster_id),
                            "terms": terms,
                        },
                    )
                )
    output.extend(_compact_exact_inner_recurrence(discovery.get("exact_inner_recurrence")))
    return output


def _find_orphan_branch(discovery: Mapping[str, Any]) -> Any:
    direct = discovery.get("effect_orphan_ngram_branch")
    if isinstance(direct, Mapping):
        return direct
    for key in ("topic_score_tests", "topic_score_selection", "score_tests"):
        nested = discovery.get(key)
        if isinstance(nested, Mapping) and isinstance(
            nested.get("effect_orphan_ngram_branch"), Mapping
        ):
            return nested["effect_orphan_ngram_branch"]
    return None


def _compact_topic_terms(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    output: list[dict[str, Any]] = []
    for raw_term in value:
        row = raw_term if isinstance(raw_term, Mapping) else {"term": raw_term}
        term = _normalize_evidence_text(
            row.get("term") or row.get("feature") or row.get("ngram"),
        )
        if not term:
            continue
        item: dict[str, Any] = {"term": term}
        for key in (
            "tfidf_contrast",
            "loading",
            "signed_score",
            "fit_signed_score",
            "standardized_score",
            "rank",
            "fit_rank",
        ):
            scalar = _finite_scalar(row.get(key))
            if scalar is not None:
                item[key] = scalar
        output.append(item)
    return output


def _compact_query_moment_evidence(
    payload: Mapping[str, Any],
    *,
    source_family: str,
    content_kind: str,
) -> list[tuple[tuple[str, ...], str, dict[str, Any]]]:
    raw = payload.get("query_evidence")
    if raw is None:
        raw = payload.get("queries") or payload.get("evidence")
    if not isinstance(raw, (list, tuple)):
        raise ValueError("neural_query_moments payload is missing query_evidence")
    output: list[tuple[tuple[str, ...], str, dict[str, Any]]] = []
    for query in raw:
        if not isinstance(query, Mapping):
            continue
        bank = str(query.get("bank") or "").lower()
        role = "effect_modifier" if bank == "effect" else "confounder"
        chunks: list[str] = []
        for row in list(query.get("top_chunks") or []):
            if isinstance(row, Mapping):
                text = _normalize_evidence_text(row.get("text") or row.get("chunk_text"))
                if text:
                    chunks.append(text)
        ngrams = _compact_topic_terms(query.get("top_contrastive_ngrams"))
        if not chunks and not ngrams:
            continue
        content: dict[str, Any] = {
            "kind": content_kind,
            "bank": bank,
            "query_id": _normalize_evidence_text(query.get("query_id")),
            "retrieved_training_excerpts": chunks,
            "contrastive_ngrams": ngrams,
        }
        for key in ("fit_standardized_score", "member_count"):
            scalar = _finite_scalar(query.get(key))
            if scalar is not None:
                content[key] = scalar
        output.append(((source_family,), role, content))
    return output


def _source_family_coverage(
    blocks: Sequence[FusionEvidenceBlock],
    candidates: Sequence[FusionPromptCandidate],
    *,
    legacy_compaction_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_counts = {family: 0 for family in ALL_SOURCE_FAMILIES}
    candidate_counts = {family: 0 for family in ALL_SOURCE_FAMILIES}
    for block in blocks:
        for family in block.source_families:
            evidence_counts[family] += 1
    for candidate in candidates:
        for family in candidate.contract.source_families:
            candidate_counts[family] += 1
    present = [family for family in ALL_SOURCE_FAMILIES if evidence_counts[family]]
    coverage = {
        "present_source_families": present,
        "missing_source_families": [
            family for family in ALL_SOURCE_FAMILIES if not evidence_counts[family]
        ],
        "evidence_block_count_by_source_family": evidence_counts,
        "candidate_count_by_source_family": candidate_counts,
        "evidence_block_count": len(blocks),
        "candidate_pool_count": len(candidates),
    }
    if legacy_compaction_audit is not None:
        coverage["legacy_compaction"] = _validate_legacy_compaction_audit(legacy_compaction_audit)
    return coverage


def _validate_legacy_compaction_audit(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("legacy compaction audit must be an object")
    required = {
        "schema_version",
        "selection_strategy",
        "maximum_groups_per_role_and_kind",
        "discovered_group_count",
        "retained_group_count",
        "dropped_group_count",
        "discovered_group_count_by_source_family",
        "retained_group_count_by_source_family",
        "discovered_group_count_by_kind",
        "retained_group_count_by_kind",
        "discovered_unique_value_count_by_axis",
        "retained_unique_value_count_by_axis",
    }
    if set(value) != required:
        raise ValueError("legacy compaction audit must use the closed accounting schema")
    if value.get("schema_version") != LEGACY_COMPACTION_STRATEGY_VERSION:
        raise ValueError("legacy compaction audit has an unsupported schema")
    if value.get("selection_strategy") != "complete_canonical_order_no_omission":
        raise ValueError("legacy compaction audit has an unsupported selection strategy")

    def nonnegative_int(raw: Any, *, field_name: str) -> int:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"legacy compaction audit {field_name} must be nonnegative")
        return raw

    if value.get("maximum_groups_per_role_and_kind") is not None:
        raise ValueError("legacy compaction audit must not declare an evidence cap")
    discovered = nonnegative_int(
        value.get("discovered_group_count"), field_name="discovered_group_count"
    )
    retained = nonnegative_int(value.get("retained_group_count"), field_name="retained_group_count")
    dropped = nonnegative_int(value.get("dropped_group_count"), field_name="dropped_group_count")
    if discovered != retained + dropped:
        raise ValueError("legacy compaction audit total accounting is inconsistent")
    if dropped != 0:
        raise ValueError("legacy compaction audit reports forbidden evidence omission")

    def count_map(raw: Any, *, keys: set[str], field_name: str) -> dict[str, int]:
        if not isinstance(raw, Mapping) or set(map(str, raw)) != keys:
            raise ValueError(f"legacy compaction audit {field_name} has invalid keys")
        return {
            key: nonnegative_int(raw[key], field_name=f"{field_name}.{key}") for key in sorted(keys)
        }

    discovered_families = count_map(
        value.get("discovered_group_count_by_source_family"),
        keys=set(ALL_SOURCE_FAMILIES),
        field_name="discovered_group_count_by_source_family",
    )
    retained_families = count_map(
        value.get("retained_group_count_by_source_family"),
        keys=set(ALL_SOURCE_FAMILIES),
        field_name="retained_group_count_by_source_family",
    )
    if any(retained_families[key] > discovered_families[key] for key in ALL_SOURCE_FAMILIES):
        raise ValueError("legacy compaction audit family accounting is inconsistent")
    kind_keys = {"bow", "embedding", "htr"}
    discovered_kinds = count_map(
        value.get("discovered_group_count_by_kind"),
        keys=kind_keys,
        field_name="discovered_group_count_by_kind",
    )
    retained_kinds = count_map(
        value.get("retained_group_count_by_kind"),
        keys=kind_keys,
        field_name="retained_group_count_by_kind",
    )
    if sum(discovered_kinds.values()) != discovered or sum(retained_kinds.values()) != retained:
        raise ValueError("legacy compaction audit kind accounting is inconsistent")
    if any(retained_kinds[key] > discovered_kinds[key] for key in kind_keys):
        raise ValueError("legacy compaction audit retained kind count exceeds discovered count")
    axis_keys = {"source_family", "bow_view", "bow_model", "objective", "sign"}
    discovered_axes = count_map(
        value.get("discovered_unique_value_count_by_axis"),
        keys=axis_keys,
        field_name="discovered_unique_value_count_by_axis",
    )
    retained_axes = count_map(
        value.get("retained_unique_value_count_by_axis"),
        keys=axis_keys,
        field_name="retained_unique_value_count_by_axis",
    )
    if any(retained_axes[key] > discovered_axes[key] for key in axis_keys):
        raise ValueError("legacy compaction audit retained axis coverage is inconsistent")
    return json.loads(_canonical_json(dict(value)))


def _normalize_evidence_text(value: Any) -> str:
    """Normalize evidence text without shortening or dropping its suffix."""

    return re.sub(r"\s+", " ", str(value or "")).strip()


def _finite_scalar(value: Any) -> int | float | bool | None:
    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 8) if math.isfinite(value) else None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(numeric, 8) if math.isfinite(numeric) else None


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
        raise ValueError("fusion payload must be finite and JSON serializable") from exc


def source_text_temporal_policy_audit() -> dict[str, Any]:
    """Return the closed, JSON-safe source-text timing policy for every audit."""

    return {
        "policy": SOURCE_TEXT_TEMPORAL_POLICY,
        "source_text_temporally_valid_by_design": True,
        "temporal_boundary_enforced": SOURCE_TEXT_TEMPORAL_BOUNDARY_ENFORCED,
        "post_treatment_semantic_filtering_enabled": False,
        "temporal_eligibility_affects_selection_or_acceptance": False,
        "semantic_timepoint_fields_allowed_as_extraction_meaning": True,
    }


__all__ = [
    "ALL_SOURCE_FAMILIES",
    "BOW_NUISANCE",
    "BOW_R_LOSS",
    "CandidateContract",
    "EMBEDDING_CLUSTERED",
    "EMBEDDING_WHOLE_COHORT",
    "EVIDENCE_CONTRACT_GROUNDING_VERSION",
    "EXACT_INNER_RECURRENCE_VERSION",
    "EvidenceContractGrounding",
    "FUSION_PROMPT_VERSION",
    "FoldEvidenceInput",
    "FoldEvidenceProvenance",
    "FusionResult",
    "HTR_NEURAL",
    "LEGACY_ALL_SOURCE",
    "LEGACY_COMPACTION_STRATEGY_VERSION",
    "MATCHED_PAIR_UPLIFT",
    "NEURAL_QUERY_MOMENTS",
    "NEURAL_QUERY_SOURCE",
    "QUERY_MOMENTS",
    "SPARSE_QUERY_MOMENTS",
    "SPARSE_QUERY_SOURCE",
    "SOURCE_TEXT_TEMPORAL_BOUNDARY_ENFORCED",
    "SOURCE_TEXT_TEMPORAL_POLICY",
    "TFIDF_ORPHAN_NGRAMS",
    "TFIDF_TOPICS",
    "TFIDF_TOPIC_SOURCE",
    "AllEvidenceFusionRequest",
    "all_evidence_fusion_response_issues",
    "build_all_evidence_fusion_repair_prompt",
    "evidence_supports_extraction_contract",
    "ground_evidence_to_extraction_contract",
    "prepare_all_evidence_fusion",
    "render_all_evidence_fusion_context_prompt",
    "render_all_evidence_fusion_prompt",
    "source_text_temporal_policy_audit",
    "validate_all_evidence_fusion_response",
]
